//! Streamed request-body pass-through (`--stream-request-bodies-over`).
//!
//! Above the threshold, with a policy that needs no request text, the raw
//! body must flow to the worker as a chunked stream, byte-for-byte. Below the
//! threshold, without a Content-Length, or under a text-needing policy, the
//! buffered typed path (which re-serializes and sends a Content-Length) must
//! be used instead.

use std::sync::Arc;

use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE, TRANSFER_ENCODING},
        HeaderMap, StatusCode,
    },
    Json,
};
use futures_util::stream;
use serde_json::{json, Value};
use smg::{
    config::{PolicyConfig, RouterConfig},
    worker::{BasicWorkerBuilder, ModelCard, Worker, WorkerType},
};
use tokio::sync::Mutex;
use tower::ServiceExt;

use crate::common::AppTestContext;

type Captured = Arc<Mutex<Vec<(HeaderMap, Bytes)>>>;

/// Loopback engine stub: captures every forwarded generate/chat request and
/// answers with canned JSON.
#[expect(
    clippy::disallowed_methods,
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
async fn spawn_capture_worker() -> (String, Captured) {
    let captured: Captured = Arc::new(Mutex::new(Vec::new()));

    async fn capture(State(sink): State<Captured>, headers: HeaderMap, body: Bytes) -> Json<Value> {
        sink.lock().await.push((headers, body));
        Json(json!({"text": "ok"}))
    }

    let app = axum::Router::new()
        .route("/generate", axum::routing::post(capture))
        .route("/v1/chat/completions", axum::routing::post(capture))
        .with_state(Arc::clone(&captured));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (format!("http://{addr}"), captured)
}

fn least_load_policy() -> PolicyConfig {
    PolicyConfig::LeastLoad {
        load_check_interval_secs: 10,
        kv_pressure_weight: 0.15,
        mean_prefill_tokens: 1024,
        default_throughput: 2000.0,
        max_waiting_requests: 0,
    }
}

fn cache_aware_policy() -> PolicyConfig {
    PolicyConfig::CacheAware {
        cache_threshold: 0.5,
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.1,
        eviction_interval_secs: 0,
        max_tree_size: 4096,
        block_size: 16,
        balance_token_usage_threshold: 1.0,
        overload_token_usage_threshold: 1.0,
        overlap_decay: 0.0,
        selection_temperature: 0.0,
    }
}

/// App wired through the real `build_app` stack with the capture stub as the
/// only worker.
async fn streaming_app(
    policy: PolicyConfig,
    threshold: u64,
    max_payload_size: usize,
) -> (axum::Router, Captured, AppTestContext) {
    let (worker_url, captured) = spawn_capture_worker().await;

    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .host("127.0.0.1")
        .port(3002)
        .max_payload_size(max_payload_size)
        .stream_request_bodies_over(threshold)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked();
    config.policy = policy;
    config.health_check.disable_health_check = true;

    let ctx = AppTestContext::new_with_config(config, vec![]).await;

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(&worker_url)
            .worker_type(WorkerType::Regular)
            .models(vec![ModelCard::new("mock-model")])
            .health_config(openai_protocol::worker::HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    ctx.app_context.worker_registry.register(worker);

    let app = ctx.create_app();
    (app, captured, ctx)
}

#[expect(
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
fn generate_payload(text_len: usize) -> Vec<u8> {
    serde_json::to_vec(&json!({"text": "x".repeat(text_len), "stream": false})).unwrap()
}

#[expect(
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
fn chunked_json_request(uri: &str, payload: Vec<u8>, content_length: Option<u64>) -> Request<Body> {
    let chunks: Vec<Result<Bytes, std::io::Error>> = payload
        .chunks(1024)
        .map(|c| Ok(Bytes::copy_from_slice(c)))
        .collect();
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json");
    if let Some(len) = content_length {
        builder = builder.header(CONTENT_LENGTH, len);
    }
    builder
        .body(Body::from_stream(stream::iter(chunks)))
        .unwrap()
}

#[tokio::test]
async fn large_generate_body_streams_to_worker_chunked() {
    let (app, captured, ctx) = streaming_app(least_load_policy(), 1024, 64 * 1024 * 1024).await;
    let payload = generate_payload(16 * 1024);

    let req = chunked_json_request("/generate", payload.clone(), Some(payload.len() as u64));
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["text"], "ok");

    let captured = captured.lock().await;
    let (headers, body) = captured.first().unwrap();
    assert_eq!(
        body.as_ref(),
        payload.as_slice(),
        "streamed body must reach the worker byte-for-byte"
    );
    assert!(
        headers.get(CONTENT_LENGTH).is_none(),
        "streamed forward must not carry a Content-Length"
    );
    assert_eq!(
        headers.get(TRANSFER_ENCODING).and_then(|v| v.to_str().ok()),
        Some("chunked")
    );
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn large_chat_body_streams_to_worker_chunked() {
    let (app, captured, ctx) = streaming_app(least_load_policy(), 1024, 64 * 1024 * 1024).await;
    let payload = serde_json::to_vec(&json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "y".repeat(16 * 1024)}]
    }))
    .unwrap();

    let req = chunked_json_request(
        "/v1/chat/completions",
        payload.clone(),
        Some(payload.len() as u64),
    );
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, body) = captured.first().unwrap();
    assert_eq!(body.as_ref(), payload.as_slice());
    assert!(headers.get(CONTENT_LENGTH).is_none());
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn below_threshold_body_stays_buffered() {
    let (app, captured, ctx) =
        streaming_app(least_load_policy(), 64 * 1024, 64 * 1024 * 1024).await;
    let payload = generate_payload(64);

    let req = chunked_json_request("/generate", payload.clone(), Some(payload.len() as u64));
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, _) = captured.first().unwrap();
    assert!(
        headers.get(CONTENT_LENGTH).is_some(),
        "the buffered path sends a fixed-length body"
    );
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn missing_content_length_stays_buffered() {
    let (app, captured, ctx) = streaming_app(least_load_policy(), 1024, 64 * 1024 * 1024).await;
    let payload = generate_payload(16 * 1024);

    let req = chunked_json_request("/generate", payload, None);
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, _) = captured.first().unwrap();
    assert!(headers.get(CONTENT_LENGTH).is_some());
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn text_needing_policy_stays_buffered_above_threshold() {
    let (app, captured, ctx) = streaming_app(cache_aware_policy(), 1024, 64 * 1024 * 1024).await;
    let payload = generate_payload(16 * 1024);

    let req = chunked_json_request("/generate", payload.clone(), Some(payload.len() as u64));
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, _) = captured.first().unwrap();
    assert!(
        headers.get(CONTENT_LENGTH).is_some(),
        "a text-needing policy must keep the buffered path"
    );
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn oversized_streamed_body_is_rejected_with_413() {
    let (app, captured, ctx) = streaming_app(least_load_policy(), 1024, 4096).await;
    // The declared length passes the ingress check; the actual stream is far
    // larger, so the cap must trip while forwarding.
    let payload = generate_payload(64 * 1024);

    let req = chunked_json_request("/generate", payload, Some(2048));
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        resp.headers()
            .get("X-SMG-Error-Code")
            .and_then(|v| v.to_str().ok()),
        Some("request_body_too_large")
    );
    assert!(captured.lock().await.is_empty());
    ctx.shutdown().await;
}
