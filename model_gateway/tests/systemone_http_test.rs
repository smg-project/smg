//! Native SystemOne contract through the production server, Gateway, and HTTP router.
#![expect(clippy::unwrap_used, reason = "test failures should panic")]

mod common;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::{
    body::{to_bytes, Body},
    extract::{Request, State},
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE},
        HeaderMap, StatusCode,
    },
    response::Response,
    routing::post,
    Router,
};
use common::test_app::{create_test_app_context, create_test_app_with_context};
use serde_json::{json, Value};
use smg::{
    routers::{
        factory::router_ids, gateway::Gateway, http::router::Router as HttpRouter, RouterTrait,
    },
    worker::{BasicWorkerBuilder, ModelCard, RuntimeType},
};
use tokio::{net::TcpListener, task::JoinHandle};
use tower::ServiceExt;

const MODEL: &str = "systemone-model";
const ALIAS: &str = "jev-latest";

#[derive(Clone)]
struct UpstreamState {
    requests: Arc<Mutex<Vec<(String, HeaderMap, Value)>>>,
    status: StatusCode,
    body: String,
}

struct Upstream {
    url: String,
    state: UpstreamState,
    task: JoinHandle<()>,
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn record_request(State(state): State<UpstreamState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body = to_bytes(body, 1024 * 1024).await.unwrap();
    state.requests.lock().unwrap().push((
        parts.uri.path().to_string(),
        parts.headers,
        serde_json::from_slice(&body).unwrap(),
    ));
    Response::builder()
        .status(state.status)
        .header(CONTENT_TYPE, "application/json")
        .header(CONTENT_LENGTH, state.body.len())
        .header("x-systemone-request-id", "native-backend-id")
        .header("retry-after", "2")
        .body(Body::from(state.body))
        .unwrap()
}

#[expect(
    clippy::disallowed_methods,
    reason = "test server is aborted by the owning Upstream on drop"
)]
async fn upstream(status: StatusCode, body: String) -> Upstream {
    let state = UpstreamState {
        requests: Arc::new(Mutex::new(Vec::new())),
        status,
        body,
    };
    let app = Router::new()
        .route("/v1/systemone", post(record_request))
        .with_state(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Upstream { url, state, task }
}

async fn app_for(upstream: &Upstream, runtime: RuntimeType) -> Router {
    app_for_dispatch(upstream, runtime, true).await
}

async fn app_for_dispatch(upstream: &Upstream, runtime: RuntimeType, use_gateway: bool) -> Router {
    let ctx = create_test_app_context().await;
    ctx.worker_registry
        .register(Arc::new(
            BasicWorkerBuilder::new(&upstream.url)
                .runtime_type(runtime)
                .model(ModelCard::new(MODEL).with_alias(ALIAS))
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        ))
        .unwrap();
    let router = Arc::new(HttpRouter::new(&ctx).await.unwrap());
    if !use_gateway {
        return create_test_app_with_context(router, ctx);
    }
    let gateway = Arc::new(Gateway::new(ctx.worker_registry.clone()));
    gateway.register_router(router_ids::HTTP_REGULAR, router);
    create_test_app_with_context(gateway, ctx)
}

fn request(body: impl Into<Body>) -> Request {
    Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header(CONTENT_TYPE, "application/json")
        .header("x-request-id", "client-systemone-id")
        .body(body.into())
        .unwrap()
}

fn native_request() -> Value {
    // Parse a literal to retain deliberately nonalphabetical question/option order.
    serde_json::from_str(
        r#"{
            "model": "systemone-model",
            "state": "The package arrived intact. The review is excellent.",
            "questions": {
                "z-delivered": {"type": "noul", "instructions": "Did the package arrive?",
                    "criteria": {"true": "Arrived", "false": null}},
                "a-condition": {"type": "choice", "instructions": null,
                    "criteria": {"intact": {"description": "Undamaged", "flag": false}, "broken": null}},
                "m-quality": {"type": "score", "criteria": [
                    {"label": "poor", "details": [false, null, 0]}, "excellent"]}
            }
        }"#,
    )
    .unwrap()
}

fn native_response() -> String {
    // SGLang v0.5.21 uses native answer fields and maps, never an OpenAI answer list.
    // Whitespace and future fields must survive without deserialize/reserialize.
    r#"{ "model": "systemone-model", "answers": {
        "m-quality": {"type": "score", "score": 0.75, "confidence": 0.5,
            "legend": {"0": {"label": "poor", "details": [false, null, 0]}, "1": "excellent"},
            "probabilities": {"0": 0.25, "1": 0.75}, "x_label_mass": 0.8},
        "z-delivered": {"type": "noul", "noul": 0.9, "x_label_mass": 0.7,
            "native_answer": {"false": false, "true": true, "null": null, "zero": 0}},
        "a-condition": {"type": "choice", "choice": "intact", "confidence": 0.6,
            "probabilities": {"intact": 0.8, "broken": 0.2}, "x_label_mass": 0.9}
        }, "usage": {"input_tokens": 27, "output_tokens": 0},
        "native_response": {"empty": {}, "list": [], "text": ""} }"#
        .to_string()
}

#[tokio::test]
async fn native_roundtrip_preserves_payloads_headers_and_insertion_order_for_http_runtimes() {
    for runtime in [RuntimeType::Generic, RuntimeType::Sglang] {
        let expected = native_response();
        let upstream = upstream(StatusCode::OK, expected.clone()).await;
        let app = app_for(&upstream, runtime).await;
        let mut body = native_request();
        body["skip_special_tokens"] = json!(false);
        body["separate_reasoning"] = json!(true);
        body["chat_template_kwargs"] = Value::Null;
        body["max_tokens"] = json!(0);
        body["stream"] = json!(false);
        body["temperature"] = Value::Null;
        body["native_request"] = json!({"empty": {}, "list": [], "text": ""});
        body["questions"]["z-delivered"]["native_question"] =
            json!({"false": false, "true": true, "null": null, "zero": 0});
        let response = app.oneshot(request(body.to_string())).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
        assert_eq!(
            response.headers()["x-systemone-request-id"],
            "native-backend-id"
        );
        assert_eq!(
            response.headers()[CONTENT_LENGTH],
            expected.len().to_string()
        );
        assert_eq!(
            to_bytes(response.into_body(), 1024 * 1024).await.unwrap(),
            expected
        );
        let requests = upstream.state.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].0, "/v1/systemone");
        assert_eq!(requests[0].1["x-request-id"], "client-systemone-id");
        assert_eq!(requests[0].2, body);
        assert_eq!(
            requests[0].2["questions"]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            ["z-delivered", "a-condition", "m-quality"]
        );
        assert_eq!(
            requests[0].2["questions"]["a-condition"]["criteria"]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            ["intact", "broken"]
        );
    }
}

#[tokio::test]
async fn explicit_model_alias_resolves_without_changing_native_questions() {
    for runtime in [RuntimeType::Generic, RuntimeType::Sglang] {
        let upstream = upstream(StatusCode::OK, native_response()).await;
        let app = app_for(&upstream, runtime).await;
        let mut body = native_request();
        body["model"] = json!(ALIAS);
        let response = app.oneshot(request(body.to_string())).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        body["model"] = json!(MODEL);
        let requests = upstream.state.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].2, body);
    }
}

#[tokio::test]
async fn structured_states_reach_the_backend_unchanged() {
    for state in [
        json!({"message": "Arrived", "metadata": {"ok": true, "missing": null}}),
        json!(["Arrived", {"condition": "intact"}, false, null, 0]),
    ] {
        let upstream = upstream(StatusCode::OK, native_response()).await;
        let app = app_for(&upstream, RuntimeType::Sglang).await;
        let mut body = native_request();
        body["state"] = state;
        let response = app.oneshot(request(body.to_string())).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let requests = upstream.state.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].2, body);
    }
}

#[tokio::test]
async fn missing_model_is_rejected_before_dispatch() {
    let upstream = upstream(StatusCode::OK, native_response()).await;
    let app = app_for(&upstream, RuntimeType::Sglang).await;
    let mut body = native_request();
    body.as_object_mut().unwrap().remove("model");
    let response = app.oneshot(request(body.to_string())).await.unwrap();
    assert!(response.status().is_client_error());
    assert!(upstream.state.requests.lock().unwrap().is_empty());
}

async fn assert_explicit_unknown_model_is_rejected(use_gateway: bool) {
    let upstream = upstream(StatusCode::OK, native_response()).await;
    let app = app_for_dispatch(&upstream, RuntimeType::Sglang, use_gateway).await;
    let mut body = native_request();
    // "unknown" is an internal routing sentinel, not an alias for this model.
    body["model"] = json!("unknown");
    let response = app.oneshot(request(body.to_string())).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert!(upstream.state.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn explicit_unknown_model_is_rejected_by_single_router_gateway() {
    assert_explicit_unknown_model_is_rejected(true).await;
}

#[tokio::test]
async fn explicit_unknown_model_is_rejected_by_direct_http_router() {
    assert_explicit_unknown_model_is_rejected(false).await;
}

#[tokio::test]
async fn invalid_json_or_native_schema_is_rejected_before_dispatch() {
    let upstream = upstream(StatusCode::OK, native_response()).await;
    let app = app_for(&upstream, RuntimeType::Sglang).await;
    let mut bodies = vec!["{\"model\":".to_string()];
    for (pointer, value) in [
        ("/model", json!(17)),
        ("/state", json!(false)),
        ("/state", Value::Null),
        ("/questions", json!([])),
        ("/questions/z-delivered/type", json!("predicate")),
        (
            "/questions/a-condition/criteria",
            json!(["intact", "broken"]),
        ),
        ("/questions/m-quality/criteria", json!([null])),
    ] {
        let mut body = native_request();
        *body.pointer_mut(pointer).unwrap() = value;
        bodies.push(body.to_string());
    }
    for body in bodies {
        let response = app.clone().oneshot(request(body.clone())).await.unwrap();
        assert!(response.status().is_client_error(), "accepted {body}");
    }
    assert!(upstream.state.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn backend_validation_and_server_errors_preserve_status_body_and_headers() {
    for runtime in [RuntimeType::Generic, RuntimeType::Sglang] {
        for status in [
            StatusCode::UNPROCESSABLE_ENTITY,
            StatusCode::INTERNAL_SERVER_ERROR,
        ] {
            let expected = r#"{ "detail": [{"loc": ["body", "questions", "z-delivered", "noul", "native_extension"], "msg": "Extra inputs are not permitted", "type": "extra_forbidden", "input": false}], "native_error": true }"#;
            let upstream = upstream(status, expected.to_string()).await;
            let app = app_for(&upstream, runtime).await;
            let mut body = native_request();
            body["questions"]["z-delivered"]["native_extension"] = json!(false);
            let response = app.oneshot(request(body.to_string())).await.unwrap();
            assert_eq!(response.status(), status);
            assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
            assert_eq!(
                response.headers()["x-systemone-request-id"],
                "native-backend-id"
            );
            assert_eq!(response.headers()["retry-after"], "2");
            assert_eq!(
                to_bytes(response.into_body(), 1024 * 1024).await.unwrap(),
                expected
            );
            let requests = upstream.state.requests.lock().unwrap();
            assert!(!requests.is_empty());
            assert!(requests
                .iter()
                .all(|(path, _, forwarded)| path == "/v1/systemone" && forwarded == &body));
        }
    }
}

#[derive(Debug)]
struct UnsupportedRouter(&'static str);

#[async_trait]
impl RouterTrait for UnsupportedRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn router_type(&self) -> &'static str {
        self.0
    }
}

#[tokio::test]
async fn unsupported_router_transports_return_not_implemented() {
    for family in ["pd", "grpc", "grpc_pd"] {
        let ctx = create_test_app_context().await;
        let app = create_test_app_with_context(Arc::new(UnsupportedRouter(family)), ctx);
        let response = app
            .oneshot(request(native_request().to_string()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED, "{family}");
    }
}
