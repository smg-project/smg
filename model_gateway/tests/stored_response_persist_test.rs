//! A response the client expects stored (`store` unset or true, not
//! `background`) is answered as stored or not at all. With the history store
//! down, a buffered request answers 503 `storage_unavailable` instead of a
//! completed response whose id is unknown afterwards, and a stream ends with
//! `response.failed` carrying the same code after its deltas; a `store: false`
//! request stores nothing and is unaffected. Through the real app over an
//! in-process scripted engine, with a history backend whose server is not
//! there (a Redis URL nothing listens on: the client connects per call).

#[path = "common/mod.rs"]
mod common;

#[path = "common/scripted_worker.rs"]
mod scripted_worker;

use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Method, Request, StatusCode},
    Router,
};
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    routers::RouterFactory,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_data_connector::{create_storage, HistoryBackend, RedisConfig, StorageFactoryConfig};
use tokio::net::TcpListener;
use tower::ServiceExt;

const MODEL: &str = "stored-response-model";

/// The app over one scripted engine, its history on a store whose server is
/// not there. The engine's task is aborted when the handle drops.
#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process engine is aborted with its handle"
)]
async fn app() -> (Router, tokio::task::AbortHandle) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let engine = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: 3,
            finish_reasons: vec!["stop"],
            generation: Default::default(),
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
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    tokenizers
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    // Port 1 of the loopback: nothing listens, every call is refused.
    let redis: RedisConfig = serde_json::from_value(json!({"url": "redis://127.0.0.1:1"})).unwrap();
    let storages = create_storage(StorageFactoryConfig {
        backend: &HistoryBackend::Redis,
        oracle: None,
        postgres: None,
        redis: Some(&redis),
        hook: None,
    })
    .await
    .unwrap();
    let context = common::create_test_context_with_storages(config, tokenizers, storages).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let app = common::test_app::create_test_app_with_context(Arc::from(router), context);
    (app, engine.abort_handle())
}

/// One `POST /v1/responses`: the status, the whole body and the body's JSON
/// (`Null` for a stream).
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn post(app: &Router, body: Value) -> (StatusCode, String, Value) {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, text, json)
}

/// The `data:` payloads of an SSE body, in order.
fn events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

#[tokio::test]
async fn a_buffered_response_whose_record_could_not_be_written_is_not_answered_as_stored() {
    let (app, engine) = app().await;

    let (status, _, body) = post(&app, json!({"model": MODEL, "input": "Hello"})).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "storage_unavailable", "{body}");
    assert_eq!(body["error"]["type"], "server_error", "{body}");

    // Nothing of an unstored request touches the store.
    let (status, _, body) = post(
        &app,
        json!({"model": MODEL, "input": "Hello", "store": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "completed", "{body}");

    engine.abort();
}

#[tokio::test]
async fn a_stream_whose_record_could_not_be_written_ends_with_response_failed() {
    let (app, engine) = app().await;

    let (status, text, _) = post(
        &app,
        json!({"model": MODEL, "input": "Hello", "stream": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let seen = events(&text);
    let terminal = seen.last().expect("a terminal event");
    assert_eq!(terminal["type"], "response.failed", "{text}");
    assert_eq!(terminal["response"]["status"], "failed", "{text}");
    assert_eq!(
        terminal["response"]["error"]["code"], "storage_unavailable",
        "{text}"
    );
    assert!(
        seen.iter()
            .any(|event| event["type"] == "response.in_progress"),
        "the stream ran before its terminal: {text}"
    );
    assert_eq!(
        seen.iter()
            .filter(|event| {
                let kind = event["type"].as_str().unwrap_or("");
                kind == "response.completed" || kind == "response.failed"
            })
            .count(),
        1,
        "exactly one terminal event: {text}"
    );

    let (status, text, _) = post(
        &app,
        json!({"model": MODEL, "input": "Hello", "stream": true, "store": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let terminal = events(&text).pop().expect("a terminal event");
    assert_eq!(terminal["type"], "response.completed", "{text}");

    engine.abort();
}
