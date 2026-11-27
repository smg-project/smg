use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

use axum::{body::Body, http::Request, routing::get, Router};
use smg_worker::metrics::writers::is_interned;
use tokio::sync::mpsc;
use tower::{ServiceBuilder, ServiceExt};

use super::*;

#[test]
fn matched_path_label_defaults_to_other_when_absent() {
    // No routing has run, so there is no MatchedPath extension.
    let extensions = http::Extensions::new();
    assert_eq!(matched_path_label(&extensions), "other");
}

/// Drive `request_uri` through a router that has one dynamic route and a
/// fallback, returning the label `matched_path_label` observes at a
/// `Router::layer`-applied middleware. The layer is the outermost wrap (it
/// also covers the fallback) so both the matched and unmatched branches are
/// exercised at the same layer the production metrics middleware uses.
async fn label_at_layer(request_uri: &str) -> String {
    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let sink = captured.clone();

    let app = Router::new()
        .route("/v1/responses/{response_id}", get(|| async { "ok" }))
        .fallback(|| async { "fallback" })
        .layer(
            ServiceBuilder::new().map_request(move |req: Request<Body>| {
                *sink.lock().unwrap() = Some(matched_path_label(req.extensions()).to_owned());
                req
            }),
        );

    let response = app
        .oneshot(
            Request::builder()
                .uri(request_uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(response.status().is_success());

    let label = captured.lock().unwrap().clone();
    label.expect("label-capturing layer ran")
}

#[tokio::test]
async fn matched_route_uses_template_label() {
    // A matched dynamic route is labeled by its template, never the raw id.
    assert_eq!(
        label_at_layer("/v1/responses/resp_abc123").await,
        "/v1/responses/{response_id}"
    );
}

#[tokio::test]
async fn unmatched_path_collapses_to_other() {
    // An unmatched path must collapse to "other", not echo the raw URI.
    assert_eq!(label_at_layer("/totally/unregistered/aaaa").await, "other");
}

/// A body the JSON extractor refused is labelled with the code its envelope
/// shows (the extractor builds the envelope outside the gateway's error
/// builder); the builder's own code keeps precedence; a success has none.
#[tokio::test]
async fn validation_rejections_carry_their_code_in_the_metric_label() {
    use axum::{http::header::CONTENT_TYPE, response::IntoResponse, routing::post};
    use openai_protocol::{chat::ChatCompletionRequest, validated::ValidatedJson};

    use crate::routers::error::create_error;

    let app = Router::new().route(
        "/v1/chat/completions",
        post(|ValidatedJson(_): ValidatedJson<ChatCompletionRequest>| async { "ok" }),
    );
    for (body, code) in [
        (
            r#"{"model":"m","messages":[{"role":"user","con"#,
            "invalid_json",
        ),
        (
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#,
            "integer_below_min_value",
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), http::StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(error_code_label(&response), code, "{body}");
    }
    let built = create_error(http::StatusCode::NOT_FOUND, "model_not_found", "no");
    assert_eq!(error_code_label(&built), "model_not_found");
    let ok: Response = "ok".into_response();
    assert_eq!(error_code_label(&ok), "");
}

/// A body the extractor refused leaves with its code in
/// `X-SMG-Error-Code`; the builder's refusal keeps the header it set and a
/// success gets none.
#[test]
fn the_head_of_a_refused_body_gets_the_code_header() {
    use axum::response::IntoResponse;

    use crate::routers::error::create_error;

    let header_after = |mut response: Response| {
        ResponseOutcome::pending(PathLabel::Static("/test")).head(&mut response);
        response
            .headers()
            .get(HEADER_X_SMG_ERROR_CODE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
    };
    let mut refused = http::StatusCode::BAD_REQUEST.into_response();
    refused
        .extensions_mut()
        .insert(RejectionCode("invalid_type".to_string()));
    assert_eq!(header_after(refused).as_deref(), Some("invalid_type"));
    let built = create_error(http::StatusCode::NOT_FOUND, "model_not_found", "no");
    assert_eq!(header_after(built).as_deref(), Some("model_not_found"));
    assert_eq!(header_after("ok".into_response()), None);
}

/// A code an upstream forges in `X-SMG-Error-Code` never becomes an
/// `error_code` label; the code the gateway's own error builder sets
/// does. Asked of each value by name, so what neighbouring tests intern
/// meanwhile cannot move the verdict.
#[tokio::test]
async fn an_upstream_forged_error_code_never_becomes_a_label() {
    use axum::{
        extract::Path,
        http::HeaderValue,
        response::{IntoResponse, Response},
    };

    use crate::{
        observability::inflight_tracker::InFlightRequestTracker,
        routers::error::{create_error, HEADER_X_SMG_ERROR_CODE},
    };

    // A handler that returns a fresh X-SMG-Error-Code per response, as an
    // upstream would, and one that refuses through the gateway's builder.
    const GATEWAY_CODE: &str = "forged_code_test_gateway_refusal";
    let app = Router::new()
        .route(
            "/echo/{id}",
            get(|Path(id): Path<String>| async move {
                let mut response: Response = "ok".into_response();
                response.headers_mut().insert(
                    HEADER_X_SMG_ERROR_CODE,
                    HeaderValue::from_str(&format!("evil-{id}")).unwrap(),
                );
                response
            }),
        )
        .route(
            "/refused",
            get(|| async { create_error(http::StatusCode::NOT_FOUND, GATEWAY_CODE, "no") }),
        )
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));

    let send = |uri: String, expected: http::StatusCode| {
        let app = app.clone();
        async move {
            let response = app
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
        }
    };

    const ITERS: usize = 1000;
    for i in 0..ITERS {
        send(format!("/echo/{i}"), http::StatusCode::OK).await;
    }
    send("/refused".to_owned(), http::StatusCode::NOT_FOUND).await;

    for i in 0..ITERS {
        assert!(
            !is_interned(&format!("evil-{i}")),
            "the upstream's forged code evil-{i} became a label"
        );
    }
    assert!(
        is_interned(GATEWAY_CODE),
        "the gateway's own code is the error_code label"
    );
}

/// A request id in the path of a matched route never becomes a `path`
/// label; the route's template does. Asked of each value by name, so
/// what neighbouring tests intern meanwhile cannot move the verdict.
#[tokio::test]
async fn a_request_id_on_a_matched_route_never_becomes_a_label() {
    // Drive the real `HttpMetricsLayer`. Every request matches the dynamic
    // route `/v1/responses/{response_id}`, so each distinct id must record
    // the bounded template label and nothing of its own.
    const TEMPLATE: &str = "/v1/responses/{response_id}";
    let app = Router::new()
        .route(TEMPLATE, get(|| async { "ok" }))
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));

    let send = |uri: String| {
        let app = app.clone();
        async move {
            let response = app
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert!(response.status().is_success());
        }
    };

    const ITERS: usize = 1000;
    for i in 0..ITERS {
        send(format!("/v1/responses/resp_{i}")).await;
    }

    for i in 0..ITERS {
        assert!(
            !is_interned(&format!("/v1/responses/resp_{i}")),
            "the request path /v1/responses/resp_{i} became a label"
        );
    }
    assert!(
        is_interned(TEMPLATE),
        "the route's template is the path label"
    );
}

/// Build a router whose `/stream` handler hands out the given body once,
/// wrapped by the real metrics layer over `tracker`.
fn stream_app(tracker: Arc<InFlightRequestTracker>, body: Body) -> Router {
    let slot = Arc::new(Mutex::new(Some(body)));
    Router::new()
        .route(
            "/stream",
            get(move || {
                let body = slot.lock().unwrap().take().expect("handler called once");
                async move { Response::new(body) }
            }),
        )
        .layer(HttpMetricsLayer::new(tracker))
}

#[tokio::test]
async fn streaming_body_holds_inflight_until_completion() {
    use http_body_util::BodyExt;
    use tokio_stream::wrappers::ReceiverStream;

    let tracker = InFlightRequestTracker::new();
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
    let app = stream_app(tracker.clone(), Body::from_stream(ReceiverStream::new(rx)));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/stream")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // The handler has returned but the body is still open: the request
    // must still count as in flight (this is exactly what a detached SSE
    // producer looks like to the middleware).
    assert_eq!(tracker.len(), 1);

    tx.send(Ok(Bytes::from_static(b"chunk"))).await.unwrap();
    drop(tx); // end of stream

    let collected = response.into_body().collect().await.unwrap();
    assert_eq!(collected.to_bytes(), Bytes::from_static(b"chunk"));
    assert_eq!(tracker.len(), 0);
    assert!(
        tracker
            .wait_for_drain(std::time::Duration::from_secs(1))
            .await
    );
}

#[tokio::test]
async fn dropped_streaming_body_releases_inflight() {
    use tokio_stream::wrappers::ReceiverStream;

    let tracker = InFlightRequestTracker::new();
    // Keep the sender alive: the stream never ends on its own.
    let (_tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(1);
    let app = stream_app(tracker.clone(), Body::from_stream(ReceiverStream::new(rx)));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/stream")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(tracker.len(), 1);

    // Client disconnect: the response (and its body) is dropped while the
    // stream is still open — the guard must release immediately.
    drop(response);
    assert_eq!(tracker.len(), 0);
}

#[tokio::test]
async fn already_complete_body_releases_at_wrap() {
    use crate::observability::inflight_tracker::InFlightRequestTracker;

    let tracker = InFlightRequestTracker::new();
    let app = Router::new()
        .route("/empty", get(|| async { Response::new(Body::empty()) }))
        .layer(HttpMetricsLayer::new(tracker.clone()));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/empty")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // An empty body is end-of-stream at wrap time; a consumer may never
    // poll it, so the guard must already be gone.
    assert_eq!(tracker.len(), 0);
    drop(response);
}

#[tokio::test]
async fn final_frame_releases_without_trailing_poll() {
    use http_body_util::BodyExt;

    use crate::observability::inflight_tracker::InFlightRequestTracker;

    let tracker = InFlightRequestTracker::new();
    let app = Router::new()
        .route("/ok", get(|| async { "ok" }))
        .layer(HttpMetricsLayer::new(tracker.clone()));

    let response = app
        .oneshot(Request::builder().uri("/ok").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let mut body = response.into_body();

    // A sized body is not end-of-stream before its data frame...
    assert_eq!(tracker.len(), 1);

    // ...but after the final successful frame the http_body contract lets
    // the consumer stop polling: release must not wait for the None.
    let frame = body.frame().await.expect("one frame").expect("frame ok");
    assert!(frame.is_data());
    assert_eq!(tracker.len(), 0);
    drop(body);
}

#[tokio::test]
async fn unary_response_releases_inflight_after_body() {
    use http_body_util::BodyExt;

    let tracker = InFlightRequestTracker::new();
    let app = Router::new()
        .route("/ok", get(|| async { "ok" }))
        .layer(HttpMetricsLayer::new(tracker.clone()));

    let response = app
        .oneshot(Request::builder().uri("/ok").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let _ = response.into_body().collect().await.unwrap();
    assert_eq!(tracker.len(), 0);
    assert!(tracker.is_empty());
}
