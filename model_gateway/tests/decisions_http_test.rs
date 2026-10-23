//! Decisions contract at the production server/Gateway/HTTP-router boundary.
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

const MODEL: &str = "decision-model";
const ALIAS: &str = "decision-alias";

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
        .header("x-request-id", "upstream-decision-id")
        .body(Body::from(state.body))
        .unwrap()
}
#[expect(
    clippy::disallowed_methods,
    reason = "test server is aborted by the owning Upstream on drop"
)]
async fn upstream(status: StatusCode, body: Value) -> Upstream {
    let state = UpstreamState {
        requests: Arc::new(Mutex::new(Vec::new())),
        status,
        body: body.to_string(),
    };
    let app = Router::new()
        .route("/v1/decisions", post(record_request))
        .route("/v1/systemone", post(record_request))
        .with_state(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Upstream { url, state, task }
}
async fn app_for(upstream: &Upstream, runtime: RuntimeType) -> Router {
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
    let gateway = Arc::new(Gateway::new(ctx.worker_registry.clone()));
    gateway.register_router(router_ids::HTTP_REGULAR, router);
    create_test_app_with_context(gateway, ctx)
}
fn request(body: Value) -> Request {
    Request::builder()
        .method("POST")
        .uri("/v1/decisions")
        .header(CONTENT_TYPE, "application/json")
        .header("x-request-id", "client-decision-id")
        .body(Body::from(body.to_string()))
        .unwrap()
}
async fn json_body(response: Response) -> Value {
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
fn predicate_request() -> Value {
    json!({"model": MODEL, "input": "The sky is blue.", "questions": [{"type": "predicate", "instructions": "Is the sky blue?"}]})
}
fn native_response() -> Value {
    json!({"model": MODEL, "answers": [{"type": "predicate", "name": null, "probability": 0.9, "native_answer": true}],
        "usage": {"input_tokens": 7, "output_tokens": 0, "total_tokens": 7,
            "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 0}}, "native_response": {"keep": 1}})
}
fn sglang_response() -> Value {
    // Intentionally not in request order: correlation must use qN, not map order/name.
    json!({"model": "served-model", "answers": {
        "q2": {"type": "score", "score": 0.75, "confidence": 0.5,
            "legend": {"0": {"label": "low"}, "1": {"label": "high", "description": "excellent"}},
            "probabilities": {"0": 0.25, "1": 0.75}, "x_label_mass": 0.8},
        "q0": {"type": "noul", "noul": 0.8, "x_label_mass": 0.4},
        "q1": {"type": "choice", "choice": "o1", "confidence": 0.6,
            "probabilities": {"o0": 0.2, "o1": 0.8}, "x_label_mass": 0.7}
    }, "usage": {"input_tokens": 22, "output_tokens": 0}})
}
fn all_questions_request() -> Value {
    json!({"model": ALIAS, "input": [{"role": "user", "content": "first evidence"},
    {"role": "user", "content": [{"type": "input_text", "text": "second evidence"}]}],
    "questions": [
        {"type": "predicate", "instructions": "Is this positive?"},
        {"type": "choice", "name": "same", "instructions": "Choose", "choices": [
            {"value": true}, {"value": "true", "description": "string"}]},
        {"type": "score", "name": "same", "instructions": "Score", "levels": [
            {"label": "low"}, {"label": "high", "description": "excellent"}]}
    ]})
}

#[tokio::test]
async fn native_roundtrip_preserves_extensions_and_resolves_model_alias() {
    let expected = native_response();
    let upstream = upstream(StatusCode::OK, expected.clone()).await;
    let app = app_for(&upstream, RuntimeType::Generic).await;
    let mut body = predicate_request();
    body["model"] = json!(ALIAS);
    body["input"] = json!([{"role": "user", "type": "message", "native_message": 1,
    "content": [
        {"type": "input_text", "text": "Inspect", "native_part": true},
        {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "original"},
        {"type": "input_text", "text": "carefully"}
    ]}]);
    body["safety_identifier"] = json!("opaque-user");
    body["native_request"] = json!({"keep": [1, true]});
    body["skip_special_tokens"] = json!(false);
    body["separate_reasoning"] = json!(true);
    body["chat_template_kwargs"] = Value::Null;
    body["questions"][0]["native_question"] = json!("preserved");
    let response = app.oneshot(request(body.clone())).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(json_body(response).await, expected);
    let requests = upstream.state.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].0, "/v1/decisions");
    body["model"] = json!(MODEL);
    assert_eq!(requests[0].2, body);
    assert_eq!(requests[0].1["x-request-id"], "client-decision-id");
}

#[tokio::test]
async fn sglang_translates_questions_and_restores_order_names_and_typed_values() {
    let upstream = upstream(StatusCode::OK, sglang_response()).await;
    let app = app_for(&upstream, RuntimeType::Sglang).await;
    let body = all_questions_request();
    let response = app.oneshot(request(body.clone())).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[CONTENT_TYPE], "application/json");
    // Middleware may replace x-request-id with the incoming ID; it must survive conversion.
    assert!(response.headers().contains_key("x-request-id"));
    let length = response
        .headers()
        .get(CONTENT_LENGTH)
        .map(|v| v.to_str().unwrap().parse::<usize>().unwrap());
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    if let Some(length) = length {
        assert_eq!(length, bytes.len(), "stale upstream content length");
    }
    let result: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        result["answers"],
        json!([
            {"type": "predicate", "name": null, "probability": 0.8},
            {"type": "choice", "name": "same", "choice": "true", "confidence": 0.6,
                "probabilities": [{"value": true, "probability": 0.2}, {"value": "true", "probability": 0.8}]},
            {"type": "score", "name": "same", "score": 0.75, "confidence": 0.5,
                "probabilities": [{"value": 0, "label": "low", "probability": 0.25}, {"value": 1, "label": "high", "probability": 0.75}]}
        ])
    );
    assert_eq!(
        result["usage"],
        json!({"input_tokens": 22, "output_tokens": 0, "total_tokens": 22,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}})
    );
    let requests = upstream.state.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].0, "/v1/systemone");
    assert_eq!(
        requests[0].2,
        json!({"model": MODEL, "state": body["input"], "questions": {
            "q0": {"type": "noul", "instructions": "Is this positive?"},
            "q1": {"type": "choice", "instructions": "Choose", "criteria": {"o0": {"value": true}, "o1": {"value": "true", "description": "string"}}},
            "q2": {"type": "score", "instructions": "Score", "criteria": [{"label": "low"}, {"label": "high", "description": "excellent"}]}
        }})
    );
}

#[tokio::test]
async fn missing_model_is_rejected_before_dispatch() {
    let upstream = upstream(StatusCode::OK, native_response()).await;
    let app = app_for(&upstream, RuntimeType::Generic).await;
    let mut body = predicate_request();
    body.as_object_mut().unwrap().remove("model");
    let response = app.oneshot(request(body)).await.unwrap();
    assert!(response.status().is_client_error());
    assert!(upstream.state.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn sglang_image_is_rejected_without_dispatch() {
    let upstream = upstream(StatusCode::OK, json!({})).await;
    let app = app_for(&upstream, RuntimeType::Sglang).await;
    let mut body = predicate_request();
    body["input"] = json!([{"role": "user", "content": [{"type": "input_text", "text": "Inspect this"},
        {"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}]);
    let response = app.oneshot(request(body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(json_body(response).await["error"]["message"].is_string());
    assert!(upstream.state.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn sglang_backend_errors_pass_through_without_success_conversion() {
    let expected =
        json!({"error": {"message": "backend validation failed", "type": "invalid_request_error"}});
    let upstream = upstream(StatusCode::UNPROCESSABLE_ENTITY, expected.clone()).await;
    let app = app_for(&upstream, RuntimeType::Sglang).await;
    let response = app.oneshot(request(predicate_request())).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(json_body(response).await, expected);
}

#[tokio::test]
async fn sglang_malformed_success_is_a_gateway_error() {
    let upstream = upstream(StatusCode::OK, json!({"not_a_decision": true})).await;
    let app = app_for(&upstream, RuntimeType::Sglang).await;
    let response = app.oneshot(request(predicate_request())).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(json_body(response).await["error"]["message"].is_string());
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
        let response = app.oneshot(request(predicate_request())).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED, "{family}");
    }
}

#[tokio::test]
async fn sglang_unsupported_extensions_are_rejected_instead_of_dropped() {
    let upstream = upstream(StatusCode::OK, sglang_response()).await;
    let app = app_for(&upstream, RuntimeType::Sglang).await;
    for pointer in [
        "",
        "/input/0",
        "/questions/0",
        "/questions/1/choices/0",
        "/questions/2/levels/0",
    ] {
        let mut body = all_questions_request();
        body.pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(
                "unsupported_extension".into(),
                json!({"changes_semantics": true}),
            );
        let response = app.clone().oneshot(request(body)).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "extension at {pointer}"
        );
        assert!(json_body(response).await["error"]["message"].is_string());
    }
    assert!(upstream.state.requests.lock().unwrap().is_empty());
}
