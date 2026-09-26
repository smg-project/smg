//! An input longer than the model's context window is a client error, and
//! the gateway says so before any engine sees it.
//!
//! Before this check, an over-long prompt on the disaggregated gRPC path was
//! forwarded to the prefill engine, whose rejection came back as
//! `500 prefill_worker_failed_to_start` -- a server error for a client
//! mistake, blamed on a healthy worker (#2380). The mock engines here accept
//! anything, so a 400 can only have come from the gateway, and a 200 for the
//! same request means the gateway forwarded it.
//!
//! Workers advertise the window on their model card; the mock tokenizer
//! yields one token per known word, so `"Hello ".repeat(n)` is exactly `n`
//! input tokens.

#[path = "common/mod.rs"]
mod common;

use std::{sync::Arc, time::Duration};

use axum::http::StatusCode;
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, model_card::ModelCard,
    worker::HealthCheckConfig,
};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::net::TcpListener;

const MODEL: &str = "context-length-test-model";
/// Context window every worker in these tests advertises.
const CONTEXT_LENGTH: u32 = 16;
const CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";
const PREFILL_FAILED: &str = "prefill_worker_failed_to_start";

/// Spawn an in-process mock gRPC worker that accepts any prompt length.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_grpc_worker() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock gRPC worker");
    let port = listener
        .local_addr()
        .expect("mock gRPC worker address")
        .port();
    let cfg = Arc::new(mock_worker::config::Config {
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 2,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
        server_args: std::collections::BTreeMap::new(),
        weight_version: None,
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

fn worker(
    port: u16,
    worker_type: WorkerType,
    context_length: Option<u32>,
) -> Arc<dyn smg::worker::Worker> {
    let mut card = ModelCard::new(MODEL);
    if let Some(len) = context_length {
        card = card.with_context_length(len);
    }
    Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(card)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    )
}

#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_router(
    mode: RoutingMode,
    workers: Vec<Arc<dyn smg::worker::Worker>>,
) -> Box<dyn RouterTrait> {
    let mut config = RouterConfig::builder()
        .mode(mode)
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = Arc::new(MockTokenizer::new()) as Arc<dyn Tokenizer>;
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let app_context =
        common::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await;
    for worker in workers {
        app_context.worker_registry.register(worker).unwrap();
    }
    RouterFactory::create_router(&app_context)
        .await
        .expect("gRPC router should build")
}

async fn pd_router(prefill_ctx: Option<u32>, decode_ctx: Option<u32>) -> Box<dyn RouterTrait> {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    build_router(
        RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        },
        vec![
            worker(prefill, WorkerType::Prefill, prefill_ctx),
            worker(decode, WorkerType::Decode, decode_ctx),
        ],
    )
    .await
}

async fn regular_router(context_length: Option<u32>) -> Box<dyn RouterTrait> {
    let port = start_mock_grpc_worker().await;
    build_router(
        RoutingMode::Regular {
            worker_urls: vec![],
        },
        vec![worker(port, WorkerType::Regular, context_length)],
    )
    .await
}

/// `tokens` words the mock tokenizer knows: exactly `tokens` input tokens.
fn prompt_of(tokens: usize) -> String {
    "Hello ".repeat(tokens).trim_end().to_string()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn completion(tokens: usize, stream: bool) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": prompt_of(tokens),
        "max_tokens": 2,
        "stream": stream,
    }))
    .unwrap()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn batched_completion(token_counts: &[usize]) -> CompletionRequest {
    let prompts: Vec<String> = token_counts.iter().map(|&n| prompt_of(n)).collect();
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": prompts,
        "max_tokens": 2,
    }))
    .unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("context-length-test-tenant"))
}

#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn read_body(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("response body");
    String::from_utf8(bytes.to_vec()).expect("utf-8 body")
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn assert_context_length_rejection(response: axum::response::Response) {
    let status = response.status();
    let content_type = response
        .headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body = read_body(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        content_type.starts_with("application/json"),
        "rejection must be a JSON error, got {content_type:?}: {body}"
    );
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["error"]["code"], CONTEXT_LENGTH_EXCEEDED, "{body}");
    assert!(
        !body.contains(PREFILL_FAILED),
        "a client mistake must not be reported as a worker failure: {body}"
    );
    let message = json["error"]["message"].as_str().unwrap();
    assert!(
        message.contains(&format!("{CONTEXT_LENGTH} tokens")),
        "message should name the window: {message}"
    );
}

/// The headline case from #2380: over-long prompt, disaggregated gRPC.
#[tokio::test]
async fn pd_over_long_prompt_is_a_400_not_a_prefill_failure() {
    let router = pd_router(Some(CONTEXT_LENGTH), Some(CONTEXT_LENGTH)).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize + 1, false),
            MODEL,
        )
        .await;

    assert_context_length_rejection(response).await;
}

/// A streaming request is rejected the same way, as a JSON error rather
/// than an event stream (the client has to be able to read the error).
#[tokio::test]
async fn pd_over_long_streaming_prompt_is_a_json_400() {
    let router = pd_router(Some(CONTEXT_LENGTH), Some(CONTEXT_LENGTH)).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize * 4, true),
            MODEL,
        )
        .await;

    assert_context_length_rejection(response).await;
}

/// The window is a capacity: a prompt that fits (even exactly) goes through
/// to the engines untouched.
#[tokio::test]
async fn pd_prompt_within_the_window_is_forwarded() {
    let router = pd_router(Some(CONTEXT_LENGTH), Some(CONTEXT_LENGTH)).await;

    for tokens in [1usize, CONTEXT_LENGTH as usize / 2, CONTEXT_LENGTH as usize] {
        let response = router
            .route_completion(None, &tenant(), completion(tokens, false), MODEL)
            .await;
        let status = response.status();
        let body = read_body(response).await;
        assert_eq!(status, StatusCode::OK, "{tokens} tokens: {body}");
    }
}

/// A disaggregated prompt must fit both engines; the decode leg's tighter
/// window governs even when the prefill leg would have taken it.
#[tokio::test]
async fn pd_is_bounded_by_the_tighter_leg() {
    let router = pd_router(Some(CONTEXT_LENGTH * 8), Some(CONTEXT_LENGTH)).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize + 1, false),
            MODEL,
        )
        .await;

    assert_context_length_rejection(response).await;
}

/// Workers that never advertised a window keep today's behavior: the gateway
/// has no opinion and the engine remains the arbiter.
#[tokio::test]
async fn pd_without_an_advertised_window_forwards_as_before() {
    let router = pd_router(None, None).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize * 8, false),
            MODEL,
        )
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// The window bounds each prompt of a batch, not the batch total: three
/// prompts that together exceed it are fine, one that alone exceeds it is not.
#[tokio::test]
async fn pd_batched_completion_is_bounded_per_prompt() {
    let router = pd_router(Some(CONTEXT_LENGTH), Some(CONTEXT_LENGTH)).await;
    let half = CONTEXT_LENGTH as usize / 2;

    let response = router
        .route_completion(
            None,
            &tenant(),
            batched_completion(&[half, half, half]),
            MODEL,
        )
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let response = router
        .route_completion(
            None,
            &tenant(),
            batched_completion(&[1, CONTEXT_LENGTH as usize + 1]),
            MODEL,
        )
        .await;
    assert_context_length_rejection(response).await;
}

/// The same check guards the single-worker gRPC path.
#[tokio::test]
async fn regular_over_long_prompt_is_a_400() {
    let router = regular_router(Some(CONTEXT_LENGTH)).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize + 1, false),
            MODEL,
        )
        .await;
    assert_context_length_rejection(response).await;

    let response = router
        .route_completion(
            None,
            &tenant(),
            completion(CONTEXT_LENGTH as usize, false),
            MODEL,
        )
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

/// Chat requests are bounded on their rendered, tokenized prompt.
#[tokio::test]
async fn regular_over_long_chat_prompt_is_a_400() {
    let router = regular_router(Some(CONTEXT_LENGTH)).await;
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_of(CONTEXT_LENGTH as usize * 4)}],
        "max_tokens": 2,
    }))
    .unwrap();

    let response = router.route_chat(None, &tenant(), request, MODEL).await;

    assert_context_length_rejection(response).await;
}
