//! End-to-end check of the PD admission gate against a mock prefill/decode
//! pair, on the running window the decode worker reports.
//!
//! The gate claims a room in the decode engine's running window before a
//! dispatch goes out, and abstains for a worker that reports none. The decode
//! worker here carries the `max_num_seqs` label a vLLM servicer's
//! `GetServerInfo` flattens into (as TokenSpeed's and SGLang's already did),
//! set to a window of one, and the admission wait is zero: a request arriving
//! while another still holds the room is shed with the overload 503, where
//! the same pair without the label admits both.

#[path = "common/mod.rs"]
mod common;

use std::{collections::HashMap, sync::Arc, time::Duration};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    completion::CompletionRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{common::pd_admission, RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::net::TcpListener;

const MODEL: &str = "pd-admission-test-model";
/// How long the canned mock holds each request before answering, so that a
/// second request lands while the first still holds its decode room.
const HOLD: Duration = Duration::from_secs(2);
/// Head start the first request gets before the second is sent.
const HEAD_START: Duration = Duration::from_millis(200);

#[expect(
    clippy::disallowed_methods,
    clippy::expect_used,
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
        admin_port: None,
        context_length: 32768,
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
        gen_delay: HOLD,
        output_tokens: 3,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
        ..mock_worker::config::Config::default()
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

/// A PD gRPC router over one mock prefill and one mock decode worker, the
/// decode carrying `decode_labels` as discovery would have set them, with a
/// zero admission wait so a full window sheds at once.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_pd_router(
    prefill_port: u16,
    decode_port: u16,
    decode_labels: HashMap<String, String>,
) -> Box<dyn RouterTrait> {
    // The test context skips the start-up path that latches the config's
    // wait, so latch it here: a full window answers at once.
    pd_admission::set_pd_admission_wait_secs(0);
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
        .pd_admission_wait_secs(0)
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
    for (port, worker_type, labels) in [
        (prefill_port, WorkerType::Prefill, HashMap::new()),
        (decode_port, WorkerType::Decode, decode_labels),
    ] {
        let worker = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .labels(labels)
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
fn completion_request() -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": "Hello world",
        "max_tokens": 8,
    }))
    .unwrap()
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("pd-admission-test-tenant"))
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

/// Two requests that overlap on the pair: the status of each, in the order
/// they were sent.
async fn two_overlapping_requests(router: &dyn RouterTrait) -> [(http::StatusCode, String); 2] {
    let tenant = tenant();
    let first = router.route_completion(None, &tenant, completion_request(), MODEL);
    let second = async {
        tokio::time::sleep(HEAD_START).await;
        router
            .route_completion(None, &tenant, completion_request(), MODEL)
            .await
    };
    let (first, second) = tokio::join!(first, second);
    let mut out = Vec::with_capacity(2);
    for response in [first, second] {
        let status = response.status();
        out.push((status, read_body(response).await));
    }
    [out.swap_remove(0), out.swap_remove(0)]
}

/// A decode worker reporting a window of one admits one request at a time:
/// the second, arriving while the first holds the room, is shed with the
/// overload 503 rather than posted to the engines, and the room is free
/// again once the first completes.
#[tokio::test]
async fn a_reported_window_of_one_sheds_the_overlapping_request() {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let labels = HashMap::from([("max_num_seqs".to_string(), "1".to_string())]);
    let router = build_pd_router(prefill, decode, labels).await;

    let outcomes = two_overlapping_requests(router.as_ref()).await;
    let shed: Vec<_> = outcomes
        .iter()
        .filter(|(status, _)| *status == http::StatusCode::SERVICE_UNAVAILABLE)
        .collect();
    assert_eq!(shed.len(), 1, "exactly one request is shed: {outcomes:?}");
    assert!(
        shed[0].1.contains("worker_overload_protection_shed"),
        "the shed is the overload guard's answer: {}",
        shed[0].1
    );
    assert!(
        outcomes
            .iter()
            .any(|(status, _)| *status == http::StatusCode::OK),
        "the request that holds the room completes: {outcomes:?}"
    );

    let response = router
        .route_completion(None, &tenant(), completion_request(), MODEL)
        .await;
    let status = response.status();
    let body = read_body(response).await;
    assert_eq!(status, http::StatusCode::OK, "the room is released: {body}");
}

/// A decode worker that reports no window is not gated: both overlapping
/// requests go to the engines, as dispatch always did.
#[tokio::test]
async fn an_unreported_window_admits_overlapping_requests() {
    let (prefill, decode) = (
        start_mock_grpc_worker().await,
        start_mock_grpc_worker().await,
    );
    let router = build_pd_router(prefill, decode, HashMap::new()).await;

    let outcomes = two_overlapping_requests(router.as_ref()).await;
    for (status, body) in &outcomes {
        assert_eq!(*status, http::StatusCode::OK, "{body}");
    }
}
