//! A stored streamed response whose client leaves mid-stream is retrievable
//! afterwards under the id `response.created` announced, with its status, and
//! a chain on that id is served.

use std::{sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Method, Request, StatusCode},
    Router,
};
use futures_util::StreamExt;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    routers::RouterFactory,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_test_harness::held_worker::HeldWorker;
use tokio::net::TcpListener;
use tower::ServiceExt;

use crate::common;

const MODEL: &str = "responses-departure-model";

/// The production route wrapper over one gRPC worker that answers its stream
/// at once and its first token two seconds later (the shape of a prompt
/// still being read when the client leaves), with the memory stores.
#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process worker is aborted after the request"
)]
async fn app() -> (Router, tokio::task::AbortHandle) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let engine = tokio::spawn(
        HeldWorker {
            first_token_after: Duration::from_secs(2),
            output_tokens: 3,
        }
        .serve(listener),
    );
    let worker = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    let tokenizers = common::mock_tokenizer_registry(MODEL).await;
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let app = common::test_app::create_test_app_with_context(Arc::from(router), context);
    (app, engine.abort_handle())
}

/// One request to the app: the status and the body's JSON.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn call(app: &Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(body.map_or_else(Body::empty, |body| Body::from(body.to_string())))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn a_stored_stream_whose_client_leaves_is_retrievable_with_its_status() {
    let (app, engine) = app().await;
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": MODEL, "input": "Hello", "stream": true, "store": true, "max_output_tokens": 8})
                .to_string(),
        ))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // The client reads the first frames (response.created names the id) and
    // leaves before the first token.
    let mut body = response.into_body().into_data_stream();
    let mut id = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while id.is_none() {
        let frame = tokio::time::timeout_at(deadline, body.next())
            .await
            .expect("the first frames arrive before the first token")
            .expect("the stream is open")
            .unwrap();
        let text = String::from_utf8_lossy(&frame);
        for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
            let event: Value = serde_json::from_str(data).unwrap();
            if event["type"] == "response.created" {
                id = event["response"]["id"].as_str().map(str::to_string);
            }
        }
    }
    drop(body);
    let id = id.unwrap();

    // The record is written when the departure is met; the GET answers from it.
    let mut stored = None;
    for _ in 0..50 {
        let (status, body) = call(&app, Method::GET, &format!("/v1/responses/{id}"), None).await;
        if status == StatusCode::OK {
            stored = Some(body);
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let stored = stored.expect("the stored stream's record is retrievable");
    assert_eq!(stored["id"], id.as_str());
    assert_eq!(stored["status"], "cancelled", "{stored}");
    assert!(stored["error"].is_null(), "{stored}");
    assert_eq!(stored["store"], true, "{stored}");

    // A chain on the id is served, not refused as a missing previous response.
    let (status, next) = call(
        &app,
        Method::POST,
        "/v1/responses",
        Some(json!({"model": MODEL, "input": "And then?", "previous_response_id": id, "max_output_tokens": 4})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{next}");
    assert_eq!(next["previous_response_id"], id.as_str(), "{next}");
    engine.abort();
}
