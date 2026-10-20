//! The wedge veto against an engine building a kernel for a prompt shape it
//! has not run. A realistic mock engine behind a gRPC router pushes its load
//! records, the engine's silence on its in-flight requests among them, over
//! the KV-event stream; the wedge budget is the default three seconds (the
//! built context applies its configuration's liveness budgets and this
//! test's configuration leaves them). The first prompt longer than any the worker
//! has answered holds
//! the engine for several budgets with nothing produced (the mock's
//! cold-shape hook; a minute on a real engine): the gateway must not veto
//! the worker, and the request completes on it. An engine frozen on a shape
//! the worker has shown (the mock's pause hook) is still vetoed within the
//! budget, and re-admitted when it produces again.

#[path = "common/mod.rs"]
mod common;

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use common::grpc_harness::{grpc_router_config, grpc_test_context, grpc_worker};
use openai_protocol::{completion::CompletionRequest, model_card::ModelCard, worker::WorkerStatus};
use smg::{
    app_context::AppContext,
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{KvEventMonitor, RuntimeType, Worker, WorkerType},
};
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle};

const MODEL: &str = "cold-shape-test-model";
/// The wedge budget, the default: a worker whose engine is silent on its
/// requests this long is vetoed.
const WEDGE: Duration = Duration::from_secs(3);
/// The engine's hold before the pass that first runs a prompt longer than
/// any it has run: several budgets, as a kernel build is on a real engine.
const BUILD: Duration = Duration::from_secs(8);
/// A pushed record carries the engine's silence within about a second of
/// it crossing a whole second: a veto due at the budget shows within this.
const VERDICT_BOUND: Duration = Duration::from_secs(5);
/// A shape shown, the same prompt again: served within this.
const SERVED_BOUND: Duration = Duration::from_secs(2);
const SHORT_PROMPT: usize = 8;
const LONG_PROMPT: usize = 64;

/// An in-process realistic mock engine with its admin API, both stopped
/// when the test ends.
struct MockEngine {
    port: u16,
    admin_port: u16,
    name: String,
    client: reqwest::Client,
    stop: Option<(oneshot::Sender<()>, oneshot::Sender<()>)>,
    served: Option<(JoinHandle<()>, JoinHandle<()>)>,
}

impl MockEngine {
    #[expect(
        clippy::expect_used,
        clippy::disallowed_methods,
        reason = "test helper: the serve tasks end with their stop signals, which the fixture \
                  sends when it is dropped; a fixture that does not come up fails the test"
    )]
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the mock engine");
        let port = listener.local_addr().expect("mock engine address").port();
        let admin_port = free_port().await;
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
            output_tokens: 4,
            realistic: true,
            step_reports: true,
            engine: mock_worker::engine::EngineParams {
                prefix_cache: true,
                ..mock_worker::engine::EngineParams::default()
            },
            loads_like: mock_worker::engine::LoadsLike::Vllm,
            ..mock_worker::config::Config::default()
        });
        let (stop, stopped) = oneshot::channel();
        let served = tokio::spawn(mock_worker::grpc::serve_with_listener_until(
            Arc::clone(&cfg),
            listener,
            async move {
                let _ = stopped.await;
            },
        ));
        let (stop_admin, admin_stopped) = oneshot::channel();
        let admin = tokio::spawn(mock_worker::admin::serve_until(
            cfg,
            "127.0.0.1".to_string(),
            admin_port,
            async move {
                let _ = admin_stopped.await;
            },
        ));
        let client = smg::clients::http_client_builder()
            .no_proxy()
            .build()
            .expect("test client");
        let engine = Self {
            port,
            admin_port,
            name: format!("grpc:{port}"),
            client,
            stop: Some((stop, stop_admin)),
            served: Some((served, admin)),
        };
        let health = format!("http://127.0.0.1:{admin_port}/admin/health");
        let deadline = Instant::now() + Duration::from_secs(5);
        while engine.client.get(&health).send().await.is_err() {
            assert!(Instant::now() < deadline, "the admin API did not come up");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        engine
    }

    /// Switch a fault hook of the engine through its admin API.
    #[expect(
        clippy::expect_used,
        reason = "test helper: an admin API that does not answer fails the test"
    )]
    async fn fault(&self, hook: &str) {
        let url = format!(
            "http://127.0.0.1:{}/admin/fault/{}/{hook}",
            self.admin_port, self.name
        );
        let status = self
            .client
            .post(url)
            .send()
            .await
            .expect("admin API reachable")
            .status();
        assert_eq!(status.as_u16(), 200, "{hook}");
    }

    async fn stop(mut self) {
        if let Some((stop, stop_admin)) = self.stop.take() {
            let _ = stop.send(());
            let _ = stop_admin.send(());
        }
        if let Some((served, admin)) = self.served.take() {
            let _ = served.await;
            let _ = admin.await;
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
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// A completion of `prompt_tokens` tokens of the mock tokenizer's vocabulary.
#[expect(
    clippy::expect_used,
    reason = "test helper: a malformed fixture request fails the fixture"
)]
fn completion_request(prompt_tokens: usize) -> CompletionRequest {
    const WORDS: [&str; 4] = ["Hello", "world", "test", "token"];
    let prompt = (0..prompt_tokens)
        .map(|i| WORDS[i % WORDS.len()])
        .collect::<Vec<_>>()
        .join(" ");
    serde_json::from_value(serde_json::json!({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 4,
        "stream": false,
    }))
    .expect("completion request")
}

async fn request_status(router: &dyn RouterTrait, prompt_tokens: usize) -> u16 {
    let tenant = TenantRequestMeta::new(TenantKey::new("cold-shape-test-tenant"));
    router
        .route_completion(None, &tenant, completion_request(prompt_tokens), MODEL)
        .await
        .status()
        .as_u16()
}

/// What a request saw of the worker's veto: its status, how long it took,
/// and the first veto while it ran with when the veto came.
struct Watched {
    status: u16,
    took: Duration,
    first_veto: Option<(String, Duration)>,
}

/// Run a completion of `prompt_tokens` tokens while watching the worker's
/// veto.
async fn request_watched(
    router: &dyn RouterTrait,
    worker: &Arc<dyn Worker>,
    prompt_tokens: usize,
) -> Watched {
    let started = Instant::now();
    let request = request_status(router, prompt_tokens);
    tokio::pin!(request);
    let mut first_veto = None;
    let status = loop {
        tokio::select! {
            status = &mut request => break status,
            () = tokio::time::sleep(Duration::from_millis(20)) => {
                if first_veto.is_none() {
                    if let Some(reason) = worker.stall_reason() {
                        first_veto = Some((format!("{reason:?}"), started.elapsed()));
                    }
                }
            }
        }
    };
    Watched {
        status,
        took: started.elapsed(),
        first_veto,
    }
}

/// A gRPC regular router over one worker, with the worker's pushed load
/// records reaching the gateway through its KV-event stream and the default
/// wedge budget. The health probes are off: the worker is set ready.
struct Bed {
    router: Box<dyn RouterTrait>,
    worker: Arc<dyn Worker>,
    /// Held for the test: the KV-event monitor keeps the subscription up,
    /// the context the worker monitor the records are applied through.
    _kv_monitor: Arc<KvEventMonitor>,
    _app_context: Arc<AppContext>,
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a router that does not build fails the fixture"
)]
async fn build_bed(port: u16) -> Bed {
    // The gateway's log: what the KV-event subscription and the liveness
    // rules did, for a failing run.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();
    let config = grpc_router_config()
        .regular_mode(vec![])
        .random_policy()
        .disable_retries()
        .build_unchecked();
    let app_context = grpc_test_context(config, MODEL).await;
    let worker_monitor = app_context
        .worker_monitor
        .clone()
        .expect("the test context builds a worker monitor");
    let kv_monitor = Arc::new(KvEventMonitor::new());
    kv_monitor.set_load_sink(&worker_monitor);

    let worker: Arc<dyn Worker> = Arc::new(
        grpc_worker(
            port,
            WorkerType::Regular,
            RuntimeType::TokenSpeed,
            ModelCard::new(MODEL),
        )
        .build(),
    );
    worker.set_status(WorkerStatus::Ready);
    app_context
        .worker_registry
        .register(Arc::clone(&worker))
        .expect("register the worker");
    kv_monitor.on_worker_added(&worker).await;
    let router = RouterFactory::create_router(&app_context)
        .await
        .expect("gRPC router");
    Bed {
        router,
        worker,
        _kv_monitor: kv_monitor,
        _app_context: app_context,
    }
}

/// The worker has answered a short prompt. The first longer prompt holds
/// the engine for several budgets with nothing produced while its records
/// keep coming: no veto, and the request completes on the worker. The shape
/// built, the next prompt of it is served at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_kernel_built_for_a_new_shape_is_not_a_wedge() {
    let engine = MockEngine::start().await;
    let bed = build_bed(engine.port).await;
    let (router, worker) = (bed.router.as_ref(), &bed.worker);

    assert_eq!(request_status(router, SHORT_PROMPT).await, 200);
    wait_until(
        "the worker's stream pushes load records",
        Duration::from_secs(10),
        || worker.pushes_load_records(),
    )
    .await;

    engine
        .fault(&format!("cold-shape?ms={}", BUILD.as_millis()))
        .await;
    let watched = request_watched(router, worker, LONG_PROMPT).await;
    tracing::info!(
        "cold shape: status {} after {:?}, first veto {:?}",
        watched.status,
        watched.took,
        watched.first_veto
    );
    assert_eq!(watched.status, 200);
    assert!(
        watched.took >= BUILD,
        "the engine built for the shape first: {:?}",
        watched.took
    );
    assert_eq!(
        watched.first_veto, None,
        "the worker was vetoed while its engine built for the new shape (the request \
         completed on it after {:?})",
        watched.took
    );
    assert!(worker.stall_reason().is_none());

    let started = Instant::now();
    assert_eq!(request_status(router, LONG_PROMPT).await, 200);
    assert!(
        started.elapsed() < SERVED_BOUND,
        "the shape built, the next prompt of it is served at once: {:?}",
        started.elapsed()
    );
    engine.stop().await;
}

/// The worker has answered the long prompt. Frozen on another one of the
/// same shape (the mock's pause: no passes, records still flowing with the
/// engine's silence), it is vetoed within the budget and a record; thawed,
/// its first token re-admits it and the request completes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::disallowed_methods,
    reason = "the thaw runs beside the request, once the veto is seen; the task ends with it"
)]
async fn a_frozen_engine_on_a_shape_shown_is_still_vetoed_within_the_budget() {
    let engine = Arc::new(MockEngine::start().await);
    let bed = build_bed(engine.port).await;
    let (router, worker) = (bed.router.as_ref(), &bed.worker);

    assert_eq!(request_status(router, LONG_PROMPT).await, 200);
    wait_until(
        "the worker's stream pushes load records",
        Duration::from_secs(10),
        || worker.pushes_load_records(),
    )
    .await;

    engine.fault("pause").await;
    let thaw = {
        let engine = Arc::clone(&engine);
        let worker = Arc::clone(worker);
        tokio::spawn(async move {
            let started = Instant::now();
            while worker.stall_reason().is_none() {
                if started.elapsed() > VERDICT_BOUND + Duration::from_secs(2) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            engine.fault("resume").await;
        })
    };
    let watched = request_watched(router, worker, LONG_PROMPT).await;
    let _ = thaw.await;
    tracing::info!(
        "frozen engine: status {} after {:?}, first veto {:?}",
        watched.status,
        watched.took,
        watched.first_veto
    );
    assert_eq!(watched.status, 200, "served once thawed");
    let (reason, at) = watched
        .first_veto
        .unwrap_or_else(|| panic!("a frozen engine was not vetoed within {VERDICT_BOUND:?}"));
    assert_eq!(reason, "Wedged");
    assert!(
        at >= WEDGE && at <= VERDICT_BOUND,
        "vetoed {at:?} after the dispatch: the budget is {WEDGE:?}"
    );
    wait_until("the re-admission", SERVED_BOUND, || {
        worker.stall_reason().is_none()
    })
    .await;
    match Arc::try_unwrap(engine) {
        Ok(engine) => engine.stop().await,
        Err(_) => panic!("the thaw task still holds the engine"),
    }
}
