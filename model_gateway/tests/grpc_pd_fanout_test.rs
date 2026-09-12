//! End-to-end check of the gRPC PD n>1 fan-out against a mock TokenSpeed
//! prefill/decode pair.
//!
//! The mock worker in canned mode answers every generate request with one
//! index-0 completion whatever `n` says, which is what a rendezvous-room
//! engine effectively does with a sample count it cannot serve from a single
//! room. So `n` choices can only come back if the gateway fans the request
//! out into `n` single-sample pairs and restamps each pair's index, which is
//! what these tests pin, non-streaming and streaming.

#[path = "common/mod.rs"]
mod common;

use std::{collections::BTreeSet, sync::Arc, time::Duration};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    completion::CompletionRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::net::TcpListener;

const MODEL: &str = "pd-fanout-test-model";
/// Tokens the canned mock emits per request; every sample reports exactly this.
const OUTPUT_TOKENS: u32 = 3;

/// Spawn a canned mock gRPC worker in-process: one index-0 completion of
/// `OUTPUT_TOKENS` tokens per request, `n` ignored.
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
        output_tokens: OUTPUT_TOKENS,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

/// A PD gRPC router over one mock prefill and one mock decode worker, both
/// registered as TokenSpeed so the dispatch takes the rendezvous-room path.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_pd_router(prefill_port: u16, decode_port: u16) -> Box<dyn RouterTrait> {
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        })
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
    for (port, worker_type) in [
        (prefill_port, WorkerType::Prefill),
        (decode_port, WorkerType::Decode),
    ] {
        let worker = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build();
        app_context
            .worker_registry
            .register(Arc::new(worker))
            .unwrap();
    }
    RouterFactory::create_router(&app_context)
        .await
        .expect("PD gRPC router should build")
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn completion_request(n: u32, stream: bool) -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": "Hello world",
        "n": n,
        "max_tokens": 8,
        "stream": stream,
    }))
    .unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-fanout-test-tenant"))
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

/// Three samples through one rendezvous-room PD pair: the fan-out runs three
/// single-sample pairs and the merged stream carries choices 0, 1 and 2.
#[tokio::test]
async fn n_samples_come_back_as_n_choices_through_a_room_based_pd_pair() {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let router = build_pd_router(prefill, decode).await;

    let response = router
        .route_completion(None, &tenant(), completion_request(3, false), MODEL)
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, http::StatusCode::OK, "{body}");

    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    let choices = json["choices"].as_array().unwrap();
    let indices: BTreeSet<u64> = choices
        .iter()
        .map(|choice| choice["index"].as_u64().unwrap())
        .collect();
    assert_eq!(indices, BTreeSet::from([0, 1, 2]), "{body}");
    assert_eq!(choices.len(), 3, "{body}");
    // Every sample reports the full prompt once and its own completion:
    // prompt tokens are counted once, completion tokens add up.
    let usage = &json["usage"];
    assert_eq!(usage["prompt_tokens"].as_u64().unwrap(), 1, "{body}");
    assert_eq!(
        usage["completion_tokens"].as_u64().unwrap(),
        u64::from(OUTPUT_TOKENS) * 3,
        "{body}"
    );
}

/// The same request streamed: every choice index shows up in the SSE events
/// and the stream still terminates with `[DONE]`.
#[tokio::test]
async fn streamed_samples_carry_every_choice_index() {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let router = build_pd_router(prefill, decode).await;

    let response = router
        .route_completion(None, &tenant(), completion_request(2, true), MODEL)
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, http::StatusCode::OK, "{body}");

    let mut indices = BTreeSet::new();
    let mut done = false;
    for line in body.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload.trim() == "[DONE]" {
            done = true;
            continue;
        }
        let event: serde_json::Value = serde_json::from_str(payload).unwrap();
        for choice in event["choices"].as_array().into_iter().flatten() {
            indices.insert(choice["index"].as_u64().unwrap());
        }
    }
    assert_eq!(indices, BTreeSet::from([0, 1]), "{body}");
    assert!(done, "stream did not terminate with [DONE]: {body}");
}

/// A single sample takes the plain single-pair path unchanged.
#[tokio::test]
async fn a_single_sample_is_one_choice() {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let router = build_pd_router(prefill, decode).await;

    let response = router
        .route_completion(None, &tenant(), completion_request(1, false), MODEL)
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, http::StatusCode::OK, "{body}");
    let json: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(json["choices"].as_array().unwrap().len(), 1, "{body}");
    assert_eq!(
        json["usage"]["completion_tokens"].as_u64().unwrap(),
        u64::from(OUTPUT_TOKENS),
        "{body}"
    );
}
