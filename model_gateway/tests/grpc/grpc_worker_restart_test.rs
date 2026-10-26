//! A gRPC worker's engine dies on its address. Comes back: the gateway's
//! health probes see it return within their interval, and the worker has to
//! be routable again right then, not when the circuit breaker that opened
//! during the outage times out on its own (60 s by default). Dies under a
//! request's retries, with its address resetting every connection: each
//! attempt is lost to the socket under the call, and the client is answered
//! as for every other way the worker's transport fails, on the last attempt
//! as on the ones before it.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use axum::{body::to_bytes, response::Response};
use openai_protocol::{
    completion::CompletionRequest,
    model_card::ModelCard,
    worker::{HealthCheckConfig, WorkerStatus},
};
use smg::{
    config::{RetryConfig, RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        manager::{WorkerManager, WorkerManagerConfig},
        BasicWorkerBuilder, ConnectionMode, RuntimeType, Worker, WorkerType,
    },
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    task::JoinHandle,
};

use crate::common;

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

async fn request(router: &dyn RouterTrait) -> Response {
    let tenant = TenantRequestMeta::new(TenantKey::new("restart-test-tenant"));
    router
        .route_completion(None, &tenant, completion_request(), MODEL)
        .await
}

async fn request_status(router: &dyn RouterTrait) -> u16 {
    request(router).await.status().as_u16()
}

/// A gRPC regular router over one worker on `port`, with the health probes
/// running: one failed probe demotes the worker, one passing probe promotes
/// it. Its circuit breaker opens after `breaker_failures` failed requests;
/// `retries` is the retry policy, none for a router that answers the first
/// attempt.
async fn build_router(
    port: u16,
    breaker_failures: u32,
    retries: Option<RetryConfig>,
) -> (Box<dyn RouterTrait>, Arc<dyn Worker>, WorkerManager) {
    build_router_with(port, breaker_failures, retries, 1, PROBE_INTERVAL).await
}

/// [`build_router`] with the worker promoted on `success_threshold` passing
/// probes `probe_interval` apart, under round robin (a policy that opens no
/// stream to the worker and polls no demoted one: nothing contacts the
/// worker while it is out, its probes alone see its engine return).
#[expect(
    clippy::expect_used,
    reason = "test helper: a router that does not build fails the fixture"
)]
async fn build_router_with(
    port: u16,
    breaker_failures: u32,
    retries: Option<RetryConfig>,
    success_threshold: u32,
    probe_interval: Duration,
) -> (Box<dyn RouterTrait>, Arc<dyn Worker>, WorkerManager) {
    let builder = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .round_robin_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024);
    let builder = match retries {
        Some(retries) => builder.retry_config(retries),
        None => builder.disable_retries(),
    };
    let mut config = builder.build_unchecked();
    config.health_check.disable_health_check = false;

    let app_context = common::create_test_context_with_mock_tokenizer(config, MODEL).await;

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                timeout_secs: 1,
                check_interval_secs: probe_interval.as_secs(),
                success_threshold,
                failure_threshold: 1,
                disable_health_check: false,
                drain_settle_secs: 0,
            })
            .circuit_breaker_config(CircuitBreakerConfig {
                failure_threshold: breaker_failures,
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
            default_check_interval_secs: probe_interval.as_secs(),
            remove_unhealthy: smg::config::UnhealthyRemovalScope::Disabled,
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
    let (router, worker, mut manager) = build_router(port, 1, None).await;

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

/// Under a success threshold of two reads, a worker the gateway kept through
/// its engine's restart is routable again one probe interval after the
/// restart plus a probe timeout, not two intervals: the probe that passes
/// short of the threshold brings the next one to a timeout from now. Nothing
/// contacts the worker while it is out (round robin opens no stream and a
/// demoted worker is not load-polled), so its probes alone see the return;
/// the engine comes back right after the probe that demoted the worker, so
/// the first passing probe is a whole interval away.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_kept_worker_returns_within_one_interval_under_a_success_threshold_of_two() {
    const INTERVAL: Duration = Duration::from_secs(3);
    let port = free_port().await;
    let engine = MockEngine::start(port).await;
    let (router, worker, mut manager) = build_router_with(port, 1, None, 2, INTERVAL).await;

    // Two passing probes an interval apart: the registration's promotion.
    wait_until("the worker's first promotion", RECOVERY_BOUND, || {
        worker.status() == WorkerStatus::Ready
    })
    .await;
    assert_eq!(request_status(router.as_ref()).await, 200);

    // The engine dies; the next probe demotes the worker, and the engine is
    // back right after it.
    engine.kill().await;
    wait_until("the worker's demotion", RECOVERY_BOUND, || {
        worker.status() != WorkerStatus::Ready
    })
    .await;
    let restarted = Instant::now();
    let engine = MockEngine::start(port).await;

    let mut first_success = None;
    while restarted.elapsed() < RECOVERY_BOUND {
        if request_status(router.as_ref()).await == 200 {
            first_success = Some(restarted.elapsed());
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let first_success = first_success.unwrap_or_else(|| {
        panic!(
            "no request succeeded within {RECOVERY_BOUND:?} of the restart (worker {:?})",
            worker.status()
        )
    });
    assert!(
        first_success >= INTERVAL - Duration::from_millis(500),
        "the first passing probe is the scheduled one, an interval after the demotion: {first_success:?}"
    );
    assert!(
        first_success < INTERVAL + Duration::from_secs(2),
        "the second read follows the first a probe timeout (one second) later, not an interval \
         later: {first_success:?}"
    );

    engine.kill().await;
    manager.shutdown().await;
}

/// A peer on the engine's address that takes every call as far as the
/// socket goes, then resets the connection: it answers the connection
/// preface with its own (an empty SETTINGS frame), reads the request up to
/// the frame that ends the stream and closes under a zero linger (RST). What
/// an engine process dying under a request looks like to the gateway's
/// channel: the call is answered by the layers below gRPC alone.
struct ResettingPeer {
    served: JoinHandle<()>,
    connections: Arc<AtomicUsize>,
}

impl ResettingPeer {
    #[expect(
        clippy::disallowed_methods,
        reason = "test helper: the peer's accept loop is aborted by the test before it returns"
    )]
    async fn start(port: u16) -> Self {
        let listener = bind(port).await;
        let connections = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&connections);
        let served = tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                counted.fetch_add(1, Ordering::SeqCst);
                if stream
                    .write_all(&[0, 0, 0, 0x4, 0, 0, 0, 0, 0])
                    .await
                    .is_err()
                {
                    continue;
                }
                read_the_request(&mut stream).await;
                let _ = stream.set_zero_linger();
                drop(stream);
            }
        });
        Self {
            served,
            connections,
        }
    }

    /// How many connections the peer took so far.
    fn connections(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }

    fn stop(self) {
        self.served.abort();
    }
}

/// Read `stream` until the client's request is in or the client is gone:
/// the 24-byte connection preface, then frames of a 9-byte header (a 24-bit
/// length, the type, the flags, the stream id) up to a complete HEADERS or
/// DATA frame that ends the stream.
async fn read_the_request(stream: &mut TcpStream) {
    const PREFACE: usize = 24;
    const DATA: u8 = 0x0;
    const HEADERS: u8 = 0x1;
    const END_STREAM: u8 = 0x1;
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(read) => bytes.extend_from_slice(&chunk[..read]),
        }
        let mut at = PREFACE;
        while bytes.len() >= at + 9 {
            let length = usize::from(bytes[at]) << 16
                | usize::from(bytes[at + 1]) << 8
                | usize::from(bytes[at + 2]);
            let (kind, flags) = (bytes[at + 3], bytes[at + 4]);
            if bytes.len() < at + 9 + length {
                break;
            }
            if matches!(kind, DATA | HEADERS) && flags & END_STREAM != 0 {
                return;
            }
            at += 9 + length;
        }
    }
}

/// The engine dies under a request's retries and its address resets every
/// connection once the request is in: each attempt is lost to the socket
/// under the call, which tonic reports with no gRPC code of its own
/// (`Unknown`, the words "transport error"). The client is answered as for
/// the other ways the worker's transport fails under a call (a cancelled
/// call, an HTTP/2 fault): a 502 on the last attempt as on the ones before
/// it, never a 500 that reads as the worker's own answer for the request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_transport_reset_on_the_last_attempt_is_answered_like_the_attempts_before_it() {
    const ATTEMPTS: u32 = 3;
    let port = free_port().await;
    let engine = MockEngine::start(port).await;
    // The breaker outlasts the attempts: every one of them reaches the worker.
    let (router, worker, mut manager) = build_router(
        port,
        ATTEMPTS + 1,
        Some(RetryConfig {
            max_retries: ATTEMPTS,
            initial_backoff_ms: 10,
            max_backoff_ms: 20,
            backoff_multiplier: 1.0,
            jitter_factor: 0.0,
        }),
    )
    .await;
    wait_until("the worker's first promotion", RECOVERY_BOUND, || {
        worker.status() == WorkerStatus::Ready
    })
    .await;
    assert_eq!(request_status(router.as_ref()).await, 200);
    // No probe runs from here on: the worker stays routable, and the
    // request's own attempts alone meet the reset.
    manager.shutdown().await;

    engine.kill().await;
    let peer = ResettingPeer::start(port).await;
    let response = request(router.as_ref()).await;
    let status = response.status().as_u16();
    let body = to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("the error body");
    let body = String::from_utf8_lossy(&body);
    let connections = peer.connections();
    peer.stop();

    assert!(
        body.contains("transport error"),
        "the attempts were lost to the socket under the call: {status} {body}"
    );
    assert_eq!(
        connections, ATTEMPTS as usize,
        "one connection per attempt, each reset: {status} {body}"
    );
    assert_eq!(status, 502, "{body}");
    assert_eq!(
        worker.circuit_breaker_state(),
        CircuitState::Closed,
        "{ATTEMPTS} transport failures under a threshold of {}",
        ATTEMPTS + 1
    );
}
