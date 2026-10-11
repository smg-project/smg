//! A gRPC worker's engine dies and comes back on the same address. The
//! gateway's health probes see it return within their interval, and the
//! worker has to be routable again right then, not when the circuit breaker
//! that opened during the outage times out on its own (60 s by default).

#[path = "common/mod.rs"]
mod common;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    completion::CompletionRequest,
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus},
};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        manager::{WorkerManager, WorkerManagerConfig},
        BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType,
    },
};
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};

const MODEL: &str = "restart-test-model";

/// The probes' cadence: a demotion or a promotion is one probe away.
const PROBE_INTERVAL: Duration = Duration::from_secs(1);
/// Far longer than the test: the breaker's own timeout must play no part.
const BREAKER_TIMEOUT: Duration = Duration::from_secs(600);
/// How long after the restart a request must have succeeded: a few probe
/// intervals, a small fraction of the breaker's timeout.
const RECOVERY_BOUND: Duration = Duration::from_secs(10);

/// An in-process mock engine on a fixed port that can be stopped: the stop
/// closes its listener and its connections, which is what a killed engine
/// process looks like to the gateway, and the port is free for a restart.
struct MockEngine {
    stop: oneshot::Sender<()>,
    served: JoinHandle<()>,
}

impl MockEngine {
    #[expect(
        clippy::disallowed_methods,
        reason = "test helper: the mock engine's serve task ends with its stop signal, \
                  which the test sends before it returns"
    )]
    async fn start(port: u16) -> Self {
        let listener = bind(port).await;
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
            gen_delay: Duration::ZERO,
            output_tokens: 3,
            realistic: false,
            engine: mock_worker::engine::EngineParams::default(),
            ..mock_worker::config::Config::default()
        });
        let (stop, stopped) = oneshot::channel();
        let served = tokio::spawn(mock_worker::grpc::serve_with_listener_until(
            cfg,
            listener,
            async move {
                let _ = stopped.await;
            },
        ));
        Self { stop, served }
    }

    /// Stop serving and wait until the listener and the connections are gone.
    async fn kill(self) {
        let _ = self.stop.send(());
        let _ = self.served.await;
    }
}

/// Bind the engine's fixed port; a restart may find the address still being
/// released by the kernel, so the bind is retried for a moment.
async fn bind(port: u16) -> TcpListener {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match TcpListener::bind(("127.0.0.1", port)).await {
            Ok(listener) => return listener,
            Err(err) => {
                assert!(
                    Instant::now() < deadline,
                    "bind 127.0.0.1:{port} for the mock engine: {err}"
                );
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a port the OS cannot hand out fails the fixture"
)]
async fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind an ephemeral port")
        .local_addr()
        .expect("ephemeral port address")
        .port()
}

/// Wait for `done`, within `bound`.
async fn wait_until(what: &str, bound: Duration, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < bound, "{what}: not within {bound:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a malformed fixture request fails the fixture"
)]
fn completion_request() -> CompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": "Hello world",
        "max_tokens": 4,
        "stream": false,
    }))
    .expect("completion request")
}

async fn request_status(router: &dyn RouterTrait) -> u16 {
    let tenant = TenantRequestMeta::new(TenantKey::new("restart-test-tenant"));
    router
        .route_completion(None, &tenant, completion_request(), MODEL)
        .await
        .status()
        .as_u16()
}

/// A gRPC regular router over one worker on `port`, with the health probes
/// running: one failed probe demotes the worker, one passing probe promotes
/// it, and its circuit breaker opens on the first failed request.
#[expect(
    clippy::expect_used,
    reason = "test helper: a router that does not build fails the fixture"
)]
async fn build_router(port: u16) -> (Box<dyn RouterTrait>, Arc<dyn Worker>, WorkerManager) {
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .random_policy()
        .disable_retries()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = false;

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
        .expect("mock tokenizer");
    let app_context =
        common::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await;

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                timeout_secs: 1,
                check_interval_secs: PROBE_INTERVAL.as_secs(),
                success_threshold: 1,
                failure_threshold: 1,
                disable_health_check: false,
                drain_settle_secs: 0,
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: 1,
                success_threshold: 1,
                timeout_duration: BREAKER_TIMEOUT,
                window_duration: BREAKER_TIMEOUT,
            })
            .build(),
    );
    app_context
        .worker_registry
        .register(worker.clone())
        .expect("register the worker");
    let manager = WorkerManager::start(
        app_context.worker_registry.clone(),
        WorkerManagerConfig {
            default_check_interval_secs: PROBE_INTERVAL.as_secs(),
            remove_unhealthy: false,
        },
        None,
    );
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("gRPC router");
    (router, worker, manager)
}

/// The engine is killed under traffic (the breaker opens, the probes demote
/// the worker) and restarted on the same address. Requests succeed again
/// within a few probe intervals of the restart: the promotion puts the open
/// breaker on trial instead of leaving it to its timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restarted_worker_is_routable_when_its_probes_pass_again() {
    let port = free_port().await;
    let engine = MockEngine::start(port).await;
    let (router, worker, mut manager) = build_router(port).await;

    // Pending until the first probe passes; then it serves.
    wait_until("the worker's first promotion", RECOVERY_BOUND, || {
        worker.status() == WorkerStatus::Ready
    })
    .await;
    assert_eq!(request_status(router.as_ref()).await, 200);

    // The engine dies: the next request fails and opens the breaker, the
    // next probe demotes the worker, and requests are refused.
    engine.kill().await;
    for _ in 0..20 {
        if worker.circuit_breaker_state() == CircuitState::Open {
            break;
        }
        let _ = request_status(router.as_ref()).await;
    }
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
    wait_until("the worker's demotion", RECOVERY_BOUND, || {
        worker.status() != WorkerStatus::Ready
    })
    .await;
    assert_eq!(request_status(router.as_ref()).await, 503);

    // The engine comes back on the same address.
    let restarted = Instant::now();
    let engine = MockEngine::start(port).await;
    let mut first_success = None;
    while restarted.elapsed() < RECOVERY_BOUND {
        if request_status(router.as_ref()).await == 200 {
            first_success = Some(restarted.elapsed());
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let first_success = first_success.unwrap_or_else(|| {
        panic!(
            "no request succeeded within {RECOVERY_BOUND:?} of the restart (worker {:?}, breaker {:?}): \
             the worker waits for the breaker's timeout",
            worker.status(),
            worker.circuit_breaker_state()
        )
    });
    assert_eq!(worker.status(), WorkerStatus::Ready);
    assert_ne!(
        worker.circuit_breaker_state(),
        CircuitState::Open,
        "the promotion put the breaker on trial and the successes closed it"
    );
    assert!(
        first_success < RECOVERY_BOUND,
        "first success {first_success:?} after the restart"
    );

    engine.kill().await;
    manager.shutdown().await;
}
