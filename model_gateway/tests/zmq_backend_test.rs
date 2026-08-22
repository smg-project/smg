//! Gateway-level integration test for the direct ZMQ backend
//! (`routers/grpc/zmq_client.rs`), driven against a real in-process mock
//! EngineCore (`mock-worker`'s `zmq` mode) over `ipc://` data-plane sockets.
//!
//! Everything below the router is real: the worker's background handshake
//! driver binds the frontend sockets, completes the vLLM handshake with the
//! mock engine(s), and streams `EngineCoreOutputs` back through the shared
//! gRPC request pipeline (`ConnectionMode::Zmq` rides `uses_grpc_pipeline`). Before this
//! suite, that whole path had CPU coverage only inside `engine-zmq-client`;
//! the gateway half was exercised only on GPU e2e lanes.
//!
//! Determinism relies on `mock-worker`'s canned (non-`realistic`) ZMQ mode: it
//! emits exactly `output_tokens` synthetic token ids then a `stop`-terminated
//! output, regardless of the prompt. The ids are outside `MockTokenizer`'s tiny
//! vocab, so decoded text is empty by construction -- the assertions are on
//! token accounting and stream shape, not on text.

#[path = "common/mod.rs"]
mod common;

use std::{sync::Arc, time::Duration};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    generate::GenerateRequest, model_card::ModelCard, worker::HealthCheckConfig,
};
use portpicker::pick_unused_port;
use smg::{
    config::RouterConfig,
    middleware::{RouteRequestMeta, TenantKey},
    routers::{RouterFactory, RouterTrait},
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType},
};

const MODEL: &str = "zmq-backend-test-model";
/// Canned tokens the mock engine emits per request.
const OUTPUT_TOKENS: u32 = 4;

/// A ZMQ frontend under test: the gateway's worker URL plus the TCP handshake
/// address the mock engines dial. The tempdir backs the `ipc://` data plane and
/// must stay alive for the test's whole body.
struct ZmqFixture {
    worker_url: String,
    handshake: String,
    _dir: tempfile::TempDir,
}

#[expect(
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn zmq_fixture() -> ZmqFixture {
    let dir = tempfile::tempdir().expect("tempdir for ipc sockets");
    // The gateway derives `<path>-in.sock` / `-out.sock` from this base path.
    let worker_url = format!("ipc://{}", dir.path().join("engine").display());
    // Pin the handshake port rather than letting the gateway derive it from the
    // socket path: the derived port is a hash of a tempdir path, so it could
    // collide with anything else on the runner.
    let port = pick_unused_port().expect("no free port for the zmq handshake");
    ZmqFixture {
        worker_url,
        handshake: format!("tcp://127.0.0.1:{port}"),
        _dir: dir,
    }
}

/// Spawn `count` mock EngineCore ranks dialing `handshake`, canned mode.
/// Each rank runs until the test process exits.
#[expect(
    clippy::disallowed_methods,
    reason = "test helper - the mock engine tasks are fire-and-forget for the test process's lifetime"
)]
fn start_mock_zmq_engines(handshake: &str, count: u16) {
    let cfg = Arc::new(mock_worker::config::Config {
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: 0,
        grpc_count: 0,
        zmq_handshake: Some(handshake.to_string()),
        zmq_count: count,
        zmq_start_index: 0,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: OUTPUT_TOKENS,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
    });
    for rank in 0..u32::from(count) {
        tokio::spawn(mock_worker::zmq::serve(
            cfg.clone(),
            handshake.to_string(),
            rank,
        ));
    }
}

/// Build a real gRPC-pipeline router with a single ZMQ worker registered
/// against `fixture`. `engine_count > 1` registers a grouped worker: one
/// worker, one socket set, N engines dialing in.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_router(fixture: &ZmqFixture, engine_count: usize) -> Box<dyn RouterTrait> {
    let mut config = RouterConfig::builder()
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
        .queue_timeout_secs(60)
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

    let mut builder = BasicWorkerBuilder::new(fixture.worker_url.clone())
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Zmq)
        .runtime_type(RuntimeType::Vllm)
        .zmq_handshake_address(fixture.handshake.clone())
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        });
    if engine_count > 1 {
        builder = builder.zmq_engine_group(engine_count);
    }
    let worker = Arc::new(builder.build());
    app_context
        .worker_registry
        .register(worker.clone())
        .unwrap();

    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("router should build over a ZMQ worker");

    // Acquisition fails fast while the background driver completes the
    // handshake; wait for it to land so requests hit a connected worker.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !matches!(worker.get_backend_client().await, Ok(Some(_))) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "ZMQ handshake never completed against the mock engine(s)"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    router
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
fn generate_request(stream: bool) -> GenerateRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "text": "Hello world",
        "stream": stream,
        "sampling_params": { "max_new_tokens": OUTPUT_TOKENS },
    }))
    .unwrap()
}

fn test_meta() -> RouteRequestMeta {
    RouteRequestMeta::new(TenantKey::from("zmq-backend-test"))
}

/// The full gateway path over ZMQ: router -> handshaked backend -> mock
/// EngineCore -> streamed outputs -> assembled non-streaming response. A regression
/// anywhere in the ipc:// bind, handshake, or output translation fails here on
/// CPU, without an H100 e2e lane.
#[tokio::test]
async fn non_streaming_generate_over_zmq() {
    let fixture = zmq_fixture();
    start_mock_zmq_engines(&fixture.handshake, 1);
    let router = build_router(&fixture, 1).await;

    let response = router
        .route_generate(None, &test_meta(), generate_request(false), MODEL)
        .await;
    assert_eq!(response.status(), http::StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    let usage = json
        .pointer("/0/meta_info/completion_tokens")
        .or_else(|| json.pointer("/meta_info/completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| panic!("no completion token count in {json}"));
    assert_eq!(
        usage,
        u64::from(OUTPUT_TOKENS),
        "canned mock engine emits exactly {OUTPUT_TOKENS} tokens"
    );
}

/// The streaming half of the same path: every engine output must reach the
/// client as its own SSE event, terminated by `[DONE]`. Non-streaming assembly
/// buffers the same outputs, so a stream that stalls mid-generation would pass
/// the test above and fail here.
#[tokio::test]
async fn streaming_generate_over_zmq() {
    let fixture = zmq_fixture();
    start_mock_zmq_engines(&fixture.handshake, 1);
    let router = build_router(&fixture, 1).await;

    let response = router
        .route_generate(None, &test_meta(), generate_request(true), MODEL)
        .await;
    assert_eq!(response.status(), http::StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("response body");
    let text = String::from_utf8(body.to_vec()).expect("utf8 sse body");
    assert!(
        text.contains("data: [DONE]"),
        "stream must terminate with [DONE]: {text}"
    );
    assert_eq!(
        text.matches("\"finish_reason\":null").count(),
        OUTPUT_TOKENS as usize,
        "one in-flight chunk per canned engine token: {text}"
    );
    assert_eq!(
        text.matches("\"finish_reason\":\"stop\"").count(),
        1,
        "exactly one terminal chunk: {text}"
    );
}

/// Two concurrent requests over one ZMQ transport must not cross-talk: the
/// connector demultiplexes outputs by request id off a single shared output
/// socket, so a mix-up here would show as a request never terminating.
#[tokio::test]
async fn concurrent_requests_share_one_zmq_transport() {
    let fixture = zmq_fixture();
    start_mock_zmq_engines(&fixture.handshake, 1);
    let router = build_router(&fixture, 1).await;

    let (first_meta, second_meta) = (test_meta(), test_meta());
    let (first_request, second_request) = (generate_request(false), generate_request(false));
    let (first, second) = tokio::join!(
        router.route_generate(None, &first_meta, first_request, MODEL),
        router.route_generate(None, &second_meta, second_request, MODEL),
    );
    assert_eq!(first.status(), http::StatusCode::OK);
    assert_eq!(second.status(), http::StatusCode::OK);
}

/// A grouped ZMQ worker: two engines dial one socket set and the connector
/// balances across them. This is the `dp_size`-on-one-worker topology
/// `zmq_engine_group` builds, exercised end-to-end through the router.
#[tokio::test]
async fn grouped_zmq_worker_serves_requests_from_either_rank() {
    let fixture = zmq_fixture();
    start_mock_zmq_engines(&fixture.handshake, 2);
    let router = build_router(&fixture, 2).await;

    for _ in 0..4 {
        let response = router
            .route_generate(None, &test_meta(), generate_request(false), MODEL)
            .await;
        assert_eq!(response.status(), http::StatusCode::OK);
    }
}
