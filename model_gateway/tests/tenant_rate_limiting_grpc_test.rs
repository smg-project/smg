//! Integration tests for tenant rate limiting wired into the gRPC router
//! (`routers/grpc/common/stages/rate_limit.rs`), verified end-to-end against
//! a real (mock) gRPC backend rather than by driving `RateLimitReserveStage`
//! in isolation.
//!
//! Calls `RouterTrait` methods directly, mirroring
//! `routers/grpc/router.rs`'s own `pd_tests` module, rather than going
//! through the full HTTP app: `tests/common/test_app.rs`'s lightweight test
//! app doesn't wire the tenant-resolution middleware that would otherwise
//! populate `tenant_request_meta`, so a real per-request tenant identity is
//! constructed here directly, same as the real middleware would produce.
//!
//! Determinism relies on `mock-worker`'s canned (non-`realistic`) gRPC mode:
//! it always reports `prompt_tokens: 1` in its final response regardless of
//! the real input length, while `completion_tokens` is exactly the
//! configured `output_tokens`. Reserve-time debits use the gateway's own
//! tokenization (via `MockTokenizer`'s tiny fixed vocab); settle-time trueup
//! uses the backend's reported usage. Choosing a token budget between those
//! two numbers proves settle actually ran with real backend-reported usage,
//! not just the initial reserve estimate.

#[path = "common/mod.rs"]
mod common;

use std::{sync::Arc, time::Duration};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    completion::CompletionRequest, generate::GenerateRequest, model_card::ModelCard,
    worker::HealthCheckConfig,
};
use portpicker::pick_unused_port;
use smg::{
    config::RouterConfig,
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::net::TcpStream;

/// "Hello world" through `MockTokenizer`'s fixed vocab ("Hello"=1, "world"=2)
/// tokenizes to exactly 2 ids -- the gateway's reserve-time estimate.
const PROMPT_TEXT: &str = "Hello world";
const ESTIMATED_INPUT_TOKENS: u32 = 2;
const MODEL: &str = "tenant-rl-test-model";

#[expect(
    clippy::panic,
    reason = "test helper - panicking on failure is intentional"
)]
async fn wait_for_port(port: u16) {
    let addr = format!("127.0.0.1:{port}");
    for _ in 0..100 {
        if TcpStream::connect(&addr).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("mock gRPC worker on {addr} never came up");
}

/// Spawn a real (mock) gRPC worker in-process, canned mode: always reports
/// `prompt_tokens: 1` and `completion_tokens: output_tokens`, regardless of
/// what was actually sent.
#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned \
              mock server task is fire-and-forget for the test process's lifetime"
)]
async fn start_mock_grpc_worker(output_tokens: u32) -> u16 {
    let port = pick_unused_port().expect("no free port for mock gRPC worker");
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
        output_tokens,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
    });
    tokio::spawn(mock_worker::grpc::serve(cfg, "127.0.0.1".to_string(), port));
    wait_for_port(port).await;
    port
}

/// Build a real gRPC router with tenant rate limiting configured against a
/// tiny in-memory budget, and a real mock gRPC worker registered. Returns
/// the router and the config's backing tempdir (must outlive the call --
/// the YAML is read synchronously during router construction, but keeping
/// it bound for the caller's whole test avoids any doubt about timing).
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_router(
    tokens_per_minute: Option<u32>,
    grpc_port: u16,
) -> (Box<dyn RouterTrait>, Option<tempfile::TempDir>) {
    let mut builder = RouterConfig::builder()
        .grpc_connection()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60);

    let dir = if let Some(tpm) = tokens_per_minute {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate_limit.yaml");
        std::fs::write(
            &path,
            format!("default_policy:\n  tokens_per_minute: {tpm}\n  requests_per_minute: 1000\n"),
        )
        .unwrap();
        builder = builder
            .tenant_rate_limit_enabled(true)
            .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()));
        Some(dir)
    } else {
        None
    };

    let mut config = builder.build_unchecked();
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

    let worker = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{grpc_port}"))
        .worker_type(WorkerType::Regular)
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

    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("gRPC router should build");
    (router, dir)
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn generate_request() -> GenerateRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "text": PROMPT_TEXT,
        "stream": false,
    }))
    .unwrap()
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn completion_request() -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": PROMPT_TEXT,
        "stream": false,
    }))
    .unwrap()
}

fn fresh_tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("integration-test-tenant"))
}

/// A denial with a finite wait (the request *could* eventually fit, just
/// not right now) must carry `Retry-After`. A fresh bucket always starts
/// full, so a finite-wait denial requires a prior request to have already
/// consumed some of the budget -- unlike the "can never fit, no matter how
/// long you wait" case (see `impossible_request_denies_without_retry_after`
/// below), which can be hit on the very first request.
#[tokio::test]
async fn denial_returns_429_with_retry_after() {
    let grpc_port = start_mock_grpc_worker(4).await;
    let (router, _dir) = build_router(Some(6), grpc_port).await;
    let request = generate_request();

    let first = router
        .route_generate(None, &fresh_tenant(), &request, MODEL)
        .await;
    assert_eq!(first.status(), http::StatusCode::OK);

    // Same budget arithmetic as the settle test below: after settling the
    // first request's real usage, only 1 token remains -- not enough for a
    // second 2-token reserve, but well within the 6-token capacity, so this
    // is a genuine "wait and retry" denial, not an impossible one.
    let second = router
        .route_generate(None, &fresh_tenant(), &request, MODEL)
        .await;
    assert_eq!(second.status(), http::StatusCode::TOO_MANY_REQUESTS);
    assert!(
        second.headers().get(http::header::RETRY_AFTER).is_some(),
        "a finite-wait denial must carry Retry-After"
    );
}

/// A request whose estimated cost exceeds the tenant's *total* budget can
/// never be admitted, no matter how long the client waits -- the backend's
/// `u64::MAX` "never" sentinel must not leak into the HTTP response as a
/// literal ~584-billion-year `Retry-After` value.
#[tokio::test]
async fn impossible_request_denies_without_retry_after() {
    let grpc_port = start_mock_grpc_worker(4).await;
    // 1 token/minute can never admit this request's 2-token reserve, ever.
    let (router, _dir) = build_router(Some(1), grpc_port).await;

    let request = generate_request();
    let response = router
        .route_generate(None, &fresh_tenant(), &request, MODEL)
        .await;

    assert_eq!(response.status(), http::StatusCode::TOO_MANY_REQUESTS);
    assert!(
        response.headers().get(http::header::RETRY_AFTER).is_none(),
        "an impossible request's 'never' sentinel must not leak into Retry-After"
    );
}

/// Proves settle_success runs with the backend's real reported usage, not
/// just the reserve-time estimate. Canned mock backend always reports
/// prompt_tokens=1 and completion_tokens=OUTPUT_TOKENS regardless of what
/// was actually sent, so real settled usage (1 + OUTPUT_TOKENS) is provably
/// larger than the reserve-time estimate alone (ESTIMATED_INPUT_TOKENS).
///
/// Budget=6: request 1 reserves 2 (estimate), settles to a true cost of
/// 1+4=5 (trueup delta +3), leaving 1 token remaining. Request 2 needs to
/// reserve another 2 estimated tokens -- only 1 remains, so it's denied. If
/// settle had never run (a regression back to "just reserve, no trueup"),
/// only 2 tokens would ever have been spent, leaving 4 remaining, and
/// request 2 (needing 2) would have been admitted instead.
#[tokio::test]
async fn non_streaming_settle_uses_real_usage_not_just_the_estimate() {
    const OUTPUT_TOKENS: u32 = 4;
    let grpc_port = start_mock_grpc_worker(OUTPUT_TOKENS).await;
    let (router, _dir) = build_router(Some(6), grpc_port).await;

    let request = generate_request();

    let first = router
        .route_generate(None, &fresh_tenant(), &request, MODEL)
        .await;
    assert_eq!(
        first.status(),
        http::StatusCode::OK,
        "first request should be admitted and dispatched successfully"
    );

    let second = router
        .route_generate(None, &fresh_tenant(), &request, MODEL)
        .await;
    assert_eq!(
        second.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "second request should be denied -- only true if settle used real \
         usage (1 + {OUTPUT_TOKENS} completion) rather than leaving only the \
         {ESTIMATED_INPUT_TOKENS}-token reserve estimate debited"
    );
}

/// With tenant rate limiting disabled entirely, requests must behave exactly
/// as they did before this feature existed -- no interference.
#[tokio::test]
async fn feature_disabled_is_a_no_op() {
    let grpc_port = start_mock_grpc_worker(4).await;
    let (router, _dir) = build_router(None, grpc_port).await;

    let request = generate_request();
    let response = router
        .route_generate(None, &fresh_tenant(), &request, MODEL)
        .await;

    assert_eq!(response.status(), http::StatusCode::OK);
}

/// `RateLimitReserveStage` is inserted identically into the Chat, Messages,
/// Completion, and Generate pipelines (`pipeline.rs::build()`), and the
/// tests above already prove the mechanism end-to-end (real reserve, real
/// settle, real denial with/without `Retry-After`) through Generate. This
/// mirrors that same impossible-budget proof through Completion, to confirm
/// the stage is actually live on a second wired pipeline too, not just
/// constructed and never reached.
///
/// Chat and Messages are deliberately not covered the same way here: both
/// need `apply_chat_template` to turn their `messages` array into a prompt
/// before preparation ever reaches the reserve stage, and `MockTokenizer`
/// (`crates/tokenizer/src/mock.rs`) doesn't implement it -- confirmed by
/// running this exact test against `route_chat`, which fails earlier with
/// `process_messages_failed: Chat template not supported by this
/// tokenizer`, a 400 from the Chat preparation stage, not a 429 from rate
/// limiting. That's a pre-existing gap in the shared mock tokenizer,
/// unrelated to this PR; extending it (or swapping in a template-capable
/// fake) is out of scope here. Harmony and PD dispatch are also not covered:
/// Harmony needs a harmony-flavored model registered so `HarmonyDetector`
/// routes into it, and PD needs a distinct prefill+decode worker pair --
/// both larger harness investments than this suite carries for its other
/// cases, and the underlying reserve/settle mechanics they'd exercise are
/// identical to what's already proven above and in `pipeline.rs`'s own
/// `rate_limit_reserve_tests` (which drives `RateLimitReserveStage` directly
/// rather than through a real backend).
#[tokio::test]
async fn completion_denial_reaches_the_client() {
    let grpc_port = start_mock_grpc_worker(4).await;
    let (router, _dir) = build_router(Some(1), grpc_port).await;

    let request = completion_request();
    let response = router
        .route_completion(None, &fresh_tenant(), &request, MODEL)
        .await;

    assert_eq!(
        response.status(),
        http::StatusCode::TOO_MANY_REQUESTS,
        "RateLimitReserveStage must be reachable through route_completion, not just route_generate"
    );
}
