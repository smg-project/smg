use std::{
    convert::Infallible,
    future::Future,
    pin::pin,
    sync::{Arc, Mutex},
    task::Poll,
};

use axum::{
    body::{Body, Bytes},
    http::Request,
    response::Response,
    routing::get,
    Router,
};
use http::header::CONTENT_LENGTH;
use http_body_util::BodyExt;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower::ServiceExt;

use super::*;
use crate::{
    middleware::edge_body::BodyFailure, observability::inflight_tracker::InFlightRequestTracker,
    routers::error::create_error,
};

/// Run `scenario` under a fresh recorder, on a current-thread runtime so
/// everything it records lands there: the rendered metrics afterwards.
fn render<F, Fut>(scenario: F) -> String
where
    F: FnOnce(PrometheusHandle) -> Fut,
    Fut: Future<Output = ()>,
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    ::metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(scenario(handle.clone()));
    });
    handle.render()
}

/// [`render`] on a paused clock: the scenario advances time itself.
fn render_paused<F, Fut>(scenario: F) -> String
where
    F: FnOnce(PrometheusHandle) -> Fut,
    Fut: Future<Output = ()>,
{
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    ::metrics::with_local_recorder(&recorder, || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .start_paused(true)
            .build()
            .expect("runtime");
        rt.block_on(scenario(handle.clone()));
    });
    handle.render()
}

/// The `smg_http_request_duration_seconds` sum of `path`.
fn duration_sum(rendered: &str, path: &str) -> f64 {
    rendered
        .lines()
        .find(|line| {
            line.starts_with("smg_http_request_duration_seconds_sum{")
                && line.contains(&format!("path=\"{path}\""))
        })
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("a duration sum for {path} in {rendered}"))
}

/// The `smg_http_connections_active` sample: the requests in flight at
/// the edge, as the gauge reports them.
fn connections_active(rendered: &str) -> f64 {
    rendered
        .lines()
        .find(|line| line.starts_with("smg_http_connections_active "))
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("the in-flight gauge in {rendered}"))
}

/// The value of the label `name` on a rendered sample line.
fn label(line: &str, name: &str) -> String {
    let key = format!("{name}=\"");
    let start = line
        .find(&key)
        .unwrap_or_else(|| panic!("{name} on {line}"))
        + key.len();
    let end = start + line[start..].find('"').expect("a closed label");
    line[start..end].to_owned()
}

/// The `smg_http_responses_total` samples of `path`, as
/// `(status_code, error_code, count)`.
fn responses(rendered: &str, path: &str) -> Vec<(String, String, u64)> {
    rendered
        .lines()
        .filter(|line| {
            line.starts_with("smg_http_responses_total{")
                && line.contains(&format!("path=\"{path}\""))
        })
        .map(|line| {
            let count = line
                .rsplit(' ')
                .next()
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| panic!("a count on {line}"));
            (label(line, "status_code"), label(line, "error_code"), count)
        })
        .collect()
}

fn sample(status_code: &str, error_code: &str) -> Vec<(String, String, u64)> {
    vec![(status_code.to_owned(), error_code.to_owned(), 1)]
}

fn request(uri: &str) -> Request<Body> {
    Request::builder()
        .uri(uri)
        .body(Body::empty())
        .expect("request")
}

/// A router whose `/stream` handler hands out `body` once, under the
/// metrics layer.
fn stream_app(body: Body) -> Router {
    let slot = Arc::new(Mutex::new(Some(body)));
    Router::new()
        .route(
            "/stream",
            get(move || {
                let body = slot
                    .lock()
                    .expect("slot")
                    .take()
                    .expect("handler called once");
                async move { Response::new(body) }
            }),
        )
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
}

fn unary_app() -> Router {
    Router::new()
        .route("/ok", get(|| async { "ok" }))
        .route(
            "/refused",
            get(|| async {
                create_error(
                    http::StatusCode::SERVICE_UNAVAILABLE,
                    "no_available_workers",
                    "no worker".to_owned(),
                )
            }),
        )
        .route(
            "/slow",
            get(|| async {
                std::future::pending::<()>().await;
                "never"
            }),
        )
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
}

/// A router whose `/relay` handler answers `status` with `body` under a
/// `Content-Length` of `length`, as the HTTP relay answers an upstream
/// that declared one.
fn relay_app(status: u16, length: u64, body: Body) -> Router {
    let slot = Arc::new(Mutex::new(Some(body)));
    Router::new()
        .route(
            "/relay",
            get(move || {
                let body = slot
                    .lock()
                    .expect("slot")
                    .take()
                    .expect("handler called once");
                async move {
                    Response::builder()
                        .status(status)
                        .header(CONTENT_LENGTH, length)
                        .body(body)
                        .expect("response")
                }
            }),
        )
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
}

/// A router whose `/refused` handler answers the gateway's own 503 error
/// (its gateway error code on the head) with `body` in place of the
/// error's JSON.
fn refused_stream_app(body: Body) -> Router {
    let slot = Arc::new(Mutex::new(Some(body)));
    Router::new()
        .route(
            "/refused",
            get(move || {
                let body = slot
                    .lock()
                    .expect("slot")
                    .take()
                    .expect("handler called once");
                async move {
                    let (parts, _) = create_error(
                        http::StatusCode::SERVICE_UNAVAILABLE,
                        "no_available_workers",
                        "no worker".to_owned(),
                    )
                    .into_parts();
                    Response::from_parts(parts, body)
                }
            }),
        )
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
}

/// The error a relayed body ends with when its upstream stream breaks.
fn broken() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::ConnectionReset, "upstream reset")
}

#[test]
fn a_stream_whose_body_fails_after_its_head_counts_the_head_status_with_the_failure_named() {
    // The HTTP relay's body ends with an error when the upstream stream
    // breaks after the head: the client received a truncated answer.
    let rendered = render(|handle| async move {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let app = stream_app(Body::from_stream(ReceiverStream::new(rx)));
        let response = app.oneshot(request("/stream")).await.expect("response");
        assert_eq!(response.status(), http::StatusCode::OK);
        let mut body = response.into_body();
        tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
            .await
            .expect("send");
        let frame = body.frame().await.expect("one frame").expect("frame ok");
        assert!(frame.is_data());
        assert!(responses(&handle.render(), "/stream").is_empty());
        tx.send(Err(broken())).await.expect("send");
        assert!(
            body.frame().await.expect("a frame").is_err(),
            "the body ends with the error"
        );
        // Counted at the failure, once: the head's status, the failure
        // named; the drop that follows adds nothing.
        assert_eq!(
            responses(&handle.render(), "/stream"),
            sample("200", "stream_error")
        );
        drop(body);
        drop(tx);
    });
    assert_eq!(
        responses(&rendered, "/stream"),
        sample("200", "stream_error"),
        "{rendered}"
    );
}

/// A router whose `/stream` handler hands out `body` once with the
/// producer's word `failure` on it, under the metrics layer.
fn told_stream_app(body: Body, failure: BodyFailure) -> Router {
    let slot = Arc::new(Mutex::new(Some(body)));
    Router::new()
        .route(
            "/stream",
            get(move || {
                let body = slot
                    .lock()
                    .expect("slot")
                    .take()
                    .expect("handler called once");
                let failure = failure.clone();
                async move {
                    let mut response = Response::new(body);
                    failure.carry(&mut response);
                    response
                }
            }),
        )
        .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
}

/// A streamed body whose producer told the client of a failure in the
/// body and then ended it as planned (the error event, then `[DONE]`)
/// counts as a body that failed, what the client read; one whose
/// producer told of nothing counts as it went out.
#[test]
fn a_body_whose_producer_told_of_a_failure_counts_as_a_failed_body() {
    for (told, error_code) in [(true, "stream_error"), (false, "")] {
        let rendered = render(|handle| async move {
            let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
            let failure = BodyFailure::default();
            let app = told_stream_app(Body::from_stream(ReceiverStream::new(rx)), failure.clone());
            let response = app.oneshot(request("/stream")).await.expect("response");
            let mut body = response.into_body();
            tx.send(Ok(Bytes::from_static(b"data: {}\n\n")))
                .await
                .expect("send");
            assert!(body.frame().await.expect("a frame").is_ok());
            if told {
                failure.tell();
            }
            tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n")))
                .await
                .expect("send");
            drop(tx);
            assert!(body.frame().await.expect("a frame").is_ok());
            assert!(body.frame().await.is_none(), "the body ends as planned");
            // Counted at the end, once: the head's status, the failure
            // named when the producer told of one.
            assert_eq!(
                responses(&handle.render(), "/stream"),
                sample("200", error_code),
                "told: {told}"
            );
            drop(body);
        });
        assert_eq!(
            responses(&rendered, "/stream"),
            sample("200", error_code),
            "told: {told}: {rendered}"
        );
    }
}

#[test]
fn a_body_that_fails_under_an_error_head_keeps_a_gateway_error_code_the_head_carried() {
    // A relayed error head carries no gateway code: the failure names it.
    let rendered = render(|_| async {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let app = relay_app(503, 12, Body::from_stream(ReceiverStream::new(rx)));
        let response = app.oneshot(request("/relay")).await.expect("response");
        let mut body = response.into_body();
        tx.send(Err(broken())).await.expect("send");
        assert!(body.frame().await.expect("a frame").is_err());
        drop(body);
    });
    assert_eq!(
        responses(&rendered, "/relay"),
        sample("503", "stream_error"),
        "{rendered}"
    );
    // The gateway's own refusal already names its outcome on the head.
    let rendered = render(|_| async {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        let app = refused_stream_app(Body::from_stream(ReceiverStream::new(rx)));
        let response = app.oneshot(request("/refused")).await.expect("response");
        let mut body = response.into_body();
        tx.send(Err(broken())).await.expect("send");
        assert!(body.frame().await.expect("a frame").is_err());
        drop(body);
    });
    assert_eq!(
        responses(&rendered, "/refused"),
        sample("503", "no_available_workers"),
        "{rendered}"
    );
}

#[test]
fn a_body_under_a_content_length_counts_as_the_response_once_its_declared_bytes_are_out() {
    // A relayed upstream answer that declared its length: on HTTP/1.1
    // the server writes the declared bytes and stops asking for frames,
    // dropping the body before its trailing None. Sent in full, it counts
    // as its head's status, once the last declared byte is out.
    let rendered = render(|handle| async move {
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let app = relay_app(503, 12, Body::from_stream(ReceiverStream::new(rx)));
        let response = app.oneshot(request("/relay")).await.expect("response");
        assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
        let mut body = response.into_body();
        tx.send(Ok(Bytes::from_static(b"{\"error\":")))
            .await
            .expect("send");
        let frame = body.frame().await.expect("a frame").expect("frame ok");
        assert_eq!(frame.data_ref().map(Bytes::len), Some(9));
        // Nine of the twelve bytes out: the response is still being sent.
        assert!(responses(&handle.render(), "/relay").is_empty());
        tx.send(Ok(Bytes::from_static(b"{}}"))).await.expect("send");
        let frame = body.frame().await.expect("a frame").expect("frame ok");
        assert_eq!(frame.data_ref().map(Bytes::len), Some(3));
        // The declared bytes are out: the server drops the body here,
        // without asking for the stream's end.
        assert_eq!(responses(&handle.render(), "/relay"), sample("503", ""));
        drop(body);
        drop(tx);
    });
    assert_eq!(
        responses(&rendered, "/relay"),
        sample("503", ""),
        "{rendered}"
    );
}

#[test]
fn a_body_under_a_content_length_dropped_before_its_declared_bytes_counts_as_a_client_closed_request(
) {
    // The other side of the declared-length rule: nine of twelve bytes
    // out when the client leaves is a body cut short.
    let rendered = render(|_| async {
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let app = relay_app(503, 12, Body::from_stream(ReceiverStream::new(rx)));
        let response = app.oneshot(request("/relay")).await.expect("response");
        let mut body = response.into_body();
        tx.send(Ok(Bytes::from_static(b"{\"error\":")))
            .await
            .expect("send");
        let frame = body.frame().await.expect("a frame").expect("frame ok");
        assert_eq!(frame.data_ref().map(Bytes::len), Some(9));
        drop(body);
        drop(tx);
    });
    assert_eq!(
        responses(&rendered, "/relay"),
        sample("499", "client_closed_request"),
        "{rendered}"
    );
}

#[test]
fn a_stream_the_client_dropped_counts_as_a_client_closed_request() {
    let rendered = render(|handle| async move {
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let app = stream_app(Body::from_stream(ReceiverStream::new(rx)));
        let response = app.oneshot(request("/stream")).await.expect("response");
        assert_eq!(response.status(), http::StatusCode::OK);
        tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
            .await
            .expect("send");
        let mut body = response.into_body();
        let frame = body.frame().await.expect("one frame").expect("frame ok");
        assert!(frame.is_data());
        // Nothing is counted while the stream is open.
        assert!(responses(&handle.render(), "/stream").is_empty());
        // The client goes away mid-stream: the server drops the body it
        // was writing.
        drop(body);
        drop(tx);
    });
    assert_eq!(
        responses(&rendered, "/stream"),
        sample("499", "client_closed_request"),
        "{rendered}"
    );
}

#[test]
fn a_stream_the_client_dropped_before_its_first_frame_counts_as_a_client_closed_request() {
    let rendered = render(|_| async {
        // The sender stays open: the stream never produces a frame.
        let (_tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(1);
        let app = stream_app(Body::from_stream(ReceiverStream::new(rx)));
        let response = app.oneshot(request("/stream")).await.expect("response");
        let mut body = response.into_body();
        {
            let mut first = pin!(body.frame());
            // The server asked for the first frame (none yet) when the
            // client closed the connection.
            assert!(matches!(futures::poll!(first.as_mut()), Poll::Pending));
        }
        drop(body);
    });
    assert_eq!(
        responses(&rendered, "/stream"),
        sample("499", "client_closed_request"),
        "{rendered}"
    );
}

#[test]
fn a_stream_counts_its_status_once_when_it_ends() {
    let rendered = render(|handle| async move {
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let app = stream_app(Body::from_stream(ReceiverStream::new(rx)));
        let response = app.oneshot(request("/stream")).await.expect("response");
        tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
            .await
            .expect("send");
        drop(tx); // end of stream
        let collected = response.into_body().collect().await.expect("body");
        assert_eq!(collected.to_bytes(), Bytes::from_static(b"data: x\n\n"));
        assert_eq!(responses(&handle.render(), "/stream"), sample("200", ""));
    });
    assert_eq!(
        responses(&rendered, "/stream"),
        sample("200", ""),
        "{rendered}"
    );
}

#[test]
fn a_request_the_client_dropped_before_the_response_counts_as_a_client_closed_request() {
    let rendered = render(|_| async {
        let mut pending = Box::pin(unary_app().oneshot(request("/slow")));
        // The handler is at work (polled once, not done) when the client
        // closes the connection: the server drops the request's future.
        assert!(matches!(futures::poll!(pending.as_mut()), Poll::Pending));
        drop(pending);
    });
    assert_eq!(
        responses(&rendered, "/slow"),
        sample("499", "client_closed_request"),
        "{rendered}"
    );
}

/// The gauge is set from the tracker's count at the request's arrival
/// and at every release of its guard. The future dropped before the
/// route answered is one of the releases: left out, an idle server
/// reported one phantom request in flight until its next request.
#[test]
fn a_request_the_client_dropped_before_the_response_leaves_the_in_flight_gauge_at_the_tracker_s_count(
) {
    let tracker = InFlightRequestTracker::new();
    let rendered = render(|handle| {
        let tracker = Arc::clone(&tracker);
        async move {
            let app = Router::new()
                .route(
                    "/slow",
                    get(|| async {
                        std::future::pending::<()>().await;
                        "never"
                    }),
                )
                .layer(HttpMetricsLayer::new(Arc::clone(&tracker)));
            let mut pending = Box::pin(app.oneshot(request("/slow")));
            assert!(matches!(futures::poll!(pending.as_mut()), Poll::Pending));
            assert_eq!(tracker.len(), 1);
            assert_eq!(connections_active(&handle.render()), 1.0);
            // The client closes the connection: the server drops the
            // request's future, and the guard with it.
            drop(pending);
            assert_eq!(tracker.len(), 0, "the guard went with the future");
        }
    });
    assert_eq!(
        connections_active(&rendered),
        0.0,
        "the gauge reads the tracker's count once the future is gone: {rendered}"
    );
}

/// The layers inside read the stamp from their `call`s, before their
/// futures run (the access log does): this layer calls the inner service
/// under the stamp's scope, not only the future it returns.
#[tokio::test]
async fn the_inner_service_s_call_runs_inside_the_entry_stamp_s_scope() {
    let seen = Arc::new(Mutex::new(None));
    let inner = tower::service_fn({
        let seen = Arc::clone(&seen);
        move |_request: Request<Body>| {
            *seen.lock().expect("the slot") = Some(RequestStarted::current().is_some());
            async { Ok::<_, Infallible>(Response::new(Body::empty())) }
        }
    });
    let mut layered = HttpMetricsLayer::new(InFlightRequestTracker::new()).layer(inner);
    let response = layered
        .ready()
        .await
        .expect("ready")
        .call(request("/scoped"))
        .await
        .expect("response");
    let _ = response.into_body().collect().await.expect("body");
    assert_eq!(
        *seen.lock().expect("the slot"),
        Some(true),
        "the inner service's call ran inside the entry stamp's scope"
    );
}

/// The layer reads the clock once at the request's entry and shares the
/// stamp: the handler finds it in the request's scope, and the
/// request's duration runs from it to the head, on the runtime's clock.
#[test]
fn the_entry_stamp_scopes_the_request_and_the_duration_runs_from_it() {
    let rendered = render_paused(|_| async {
        let app = Router::new()
            .route(
                "/stamped",
                get(|| async move {
                    let started =
                        RequestStarted::current().expect("the entry stamp scopes the request");
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    assert_eq!(
                        clock_now().saturating_duration_since(started),
                        std::time::Duration::from_secs(3),
                        "the stamp is on the runtime's clock"
                    );
                    "ok"
                }),
            )
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));
        let response = app.oneshot(request("/stamped")).await.expect("response");
        let _ = response.into_body().collect().await.expect("body");
    });
    let seconds = duration_sum(&rendered, "/stamped");
    assert!(
        (seconds - 3.0).abs() < 0.01,
        "duration {seconds} s, expected the 3 s from the entry stamp to the head: {rendered}"
    );
}

#[test]
fn a_unary_response_counts_its_status_once_its_body_is_out() {
    let rendered = render(|_| async {
        let response = unary_app().oneshot(request("/ok")).await.expect("response");
        let _ = response.into_body().collect().await.expect("body");
    });
    assert_eq!(responses(&rendered, "/ok"), sample("200", ""), "{rendered}");
}

#[test]
fn a_refusal_keeps_its_status_and_gateway_error_code() {
    let rendered = render(|_| async {
        let response = unary_app()
            .oneshot(request("/refused"))
            .await
            .expect("response");
        let _ = response.into_body().collect().await.expect("body");
    });
    assert_eq!(
        responses(&rendered, "/refused"),
        sample("503", "no_available_workers"),
        "{rendered}"
    );
}

#[test]
fn a_body_the_server_never_polled_counts_as_the_response_it_was() {
    // The answer to a HEAD request: the server drops the body without
    // asking for a frame. Not a client that left.
    let rendered = render(|_| async {
        let response = unary_app().oneshot(request("/ok")).await.expect("response");
        drop(response);
    });
    assert_eq!(responses(&rendered, "/ok"), sample("200", ""), "{rendered}");
}
