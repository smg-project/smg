//! What every servicer shares whatever its wire: the half of its state the
//! handlers and the lifecycle owner read (the engine link, the admitted
//! requests, the KV-event relay, the metrics, the serving switch) and the
//! running server (the gRPC listener on its runtime thread, the `/metrics`
//! listener beside it, the engine connect it launches once the listener is
//! bound, the drain-and-stop path). A wire's module adds its model facts and
//! its proto on top; the drain semantics, the connect's launch, the listener
//! budget and the stop path are written here once.

use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime},
};

use engine_zmq_adapter::ZmqEngineClient;
use engine_zmq_client::protocol::handshake::EngineCoreReadyResponse;
use futures::FutureExt;
use tokio::net::TcpListener;
use tokio_stream::{wrappers::TcpListenerStream, StreamExt};
use tonic::{service::Routes, transport, Status};
use tonic_health::pb::health_server::HealthServer;
use tracing::{info, warn};

use crate::{
    engine_link::{ConnectInputs, EngineLink},
    health::HealthReporter,
    kv_events::{KvEventRelay, LoadFromState, RankLoads},
    metrics::{MetricsListener, ServicerMetrics, ServicerState, Snapshot},
    requests::RequestRegistry,
    ServerThread, ServicerConfig, ServicerError, Shutdown,
};

/// A wire's state as the shared code sees it: where its shared half is.
pub(crate) trait WireState: Send + Sync + 'static {
    fn shared(&self) -> &Shared;
}

/// The half of a servicer's state every wire shares.
pub(crate) struct Shared {
    /// The KV-event relay for the engine's ZMQ publisher; `None` when events
    /// are off (`SubscribeKvEvents` is then UNIMPLEMENTED).
    pub(crate) kv_relay: Option<Arc<KvEventRelay>>,
    /// The local tokenizer directory `GetTokenizer` bundles; `None` when none
    /// resolved.
    pub(crate) tokenizer_dir: Option<String>,
    pub(crate) engine: EngineLink,
    /// The request, token and latency metrics; the token totals feed a wire's
    /// periodic stats line whether or not a listener renders them.
    pub(crate) metrics: Arc<ServicerMetrics>,
    pub(crate) registry: Arc<RequestRegistry>,
    /// Cleared by the lifecycle owner to drain (`set_serving`, admission
    /// closed with it) or to hold readiness alone (`set_ready`, admission
    /// open): health answers NOT_SERVING either way.
    serving: AtomicBool,
    pub(crate) started: Instant,
    pub(crate) started_at: SystemTime,
}

impl Shared {
    /// A servicer about to start: serving, the engine link unset, no request
    /// admitted yet.
    pub(crate) fn new(
        kv_relay: Option<Arc<KvEventRelay>>,
        metrics: Arc<ServicerMetrics>,
        tokenizer_dir: Option<String>,
    ) -> Self {
        Self {
            kv_relay,
            tokenizer_dir,
            engine: EngineLink::default(),
            metrics,
            registry: Arc::new(RequestRegistry::default()),
            serving: AtomicBool::new(true),
            started: Instant::now(),
            started_at: SystemTime::now(),
        }
    }

    pub(crate) fn engine(&self) -> Result<&ZmqEngineClient, Status> {
        if let Some(client) = self.engine.client.get() {
            return Ok(client);
        }
        Err(match self.engine.error() {
            Some(error) => Status::unavailable(format!("engine connection failed: {error}")),
            None => Status::unavailable("engine is still connecting"),
        })
    }

    /// The engine's handshake answer, once the link is up.
    pub(crate) fn ready(&self) -> Option<&EngineCoreReadyResponse> {
        self.engine
            .client
            .get()
            .and_then(ZmqEngineClient::ready_response)
    }

    /// The KV capacity in tokens, from the handshake; 0 before the engine is
    /// up.
    pub(crate) fn kv_capacity_tokens(&self) -> i32 {
        self.ready()
            .and_then(|ready| ready.kv_cache_size_tokens)
            .map(|tokens| i32::try_from(tokens).unwrap_or(i32::MAX))
            .unwrap_or(0)
    }

    /// The scheduler's running window: the launcher's figure when it reported
    /// one, else the handshake's; 0 before the engine is up on a launcher that
    /// reported none.
    pub(crate) fn running_window(&self, launcher: i32) -> i32 {
        if launcher > 0 {
            return launcher;
        }
        self.ready()
            .map(|ready| i32::try_from(ready.max_num_seqs).unwrap_or(i32::MAX))
            .unwrap_or(0)
    }

    /// The engine version to report: the launcher's, else the handshake's
    /// with the wire's `prefix` stripped, else `unknown`.
    pub(crate) fn engine_version(&self, launcher: &str, prefix: &str) -> String {
        Some(launcher.to_string())
            .filter(|version| !version.is_empty())
            .or_else(|| {
                self.ready().map(|ready| {
                    ready
                        .vllm_version
                        .strip_prefix(prefix)
                        .unwrap_or(&ready.vllm_version)
                        .to_string()
                })
            })
            .filter(|version| !version.is_empty())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// The data-parallel rank count a zero-filled load report covers: the
    /// handshake's, else the launcher's, else one.
    pub(crate) fn rank_count(&self, launcher: i32) -> usize {
        self.ready()
            .and_then(|ready| usize::try_from(ready.data_parallel_size).ok())
            .filter(|&ranks| ranks > 0)
            .or_else(|| usize::try_from(launcher).ok())
            .unwrap_or(1)
            .max(1)
    }

    /// Whether the engine handshake completed.
    pub(crate) fn engine_ready(&self) -> bool {
        self.engine.client.get().is_some()
    }

    /// A sign of life from the engine process while the handshake runs: the
    /// lifecycle owner that launched it reports one each time it polls the
    /// process and finds it alive. Each report resets the handshake's
    /// silence bound; the ceiling still holds. A no-op once the engine is
    /// connected.
    pub(crate) fn note_engine_alive(&self) {
        if !self.engine_ready() {
            self.engine.liveness.touch();
        }
    }

    /// Whether the KV-event relay's subscriptions have reached the engine's
    /// publishers, or there is none to wait for (events off, a relay told to
    /// start lazily, one that gave up). The lifecycle owner holds its planned
    /// warm-up until then, so the warm-up's batches start the relay's
    /// history and a gateway that subscribes later gets the engine's whole
    /// life.
    pub(crate) fn kv_relay_subscribed(&self) -> bool {
        self.kv_relay
            .as_ref()
            .is_none_or(|relay| relay.subscribed())
    }

    /// SERVING only once the engine link is up and alive and the lifecycle
    /// owner has not started draining.
    pub(crate) fn is_serving(&self) -> bool {
        self.serving.load(Ordering::Acquire)
            && self
                .engine
                .client
                .get()
                .is_some_and(ZmqEngineClient::is_alive)
    }

    /// Announce SERVING (`true`) or drain (`false`): health flips at once,
    /// new admission closes while in-flight streams keep running until the
    /// stop.
    pub(crate) fn set_serving(&self, serving: bool) {
        self.registry.set_accepting(serving);
        self.serving.store(serving, Ordering::Release);
    }

    /// Hold (`false`) or release (`true`) readiness alone: health flips as
    /// with `set_serving`, admission stays open to the lifecycle owner's own
    /// requests (its planned warm-up before SERVING, `smg-warmup-<n>`) and to
    /// nothing else, so the warm-up's steps share the engine with no client
    /// request. `set_serving(false)` is the drain switch and closes admission
    /// as well.
    pub(crate) fn set_ready(&self, ready: bool) {
        self.registry.set_warming(!ready);
        self.serving.store(ready, Ordering::Release);
    }

    /// What `HealthCheck` says of the servicer: `healthy` is the wire's own
    /// word for a serving engine (the Python servicers differ on it), the
    /// rest reads the same on every wire.
    pub(crate) fn health_message(&self, healthy: &'static str) -> &'static str {
        if !self.serving.load(Ordering::Acquire) {
            // Admission still open: the lifecycle owner holds readiness for
            // its own requests (the planned warm-up); closed: it drains.
            return if self.registry.is_accepting() {
                "Warming up"
            } else {
                "Draining"
            };
        }
        match (self.engine.client.get(), self.engine.error()) {
            (Some(client), _) if client.is_alive() => healthy,
            (Some(_), _) => "Engine is not alive",
            (None, Some(_)) => "Engine connection failed",
            (None, None) => "Engine is starting",
        }
    }

    /// The lifecycle state a scrape reports: the drain is read off the
    /// registry (admission closed); a cleared serving flag with admission
    /// open is readiness held back, not a drain.
    fn lifecycle_state(&self) -> ServicerState {
        if !self.registry.is_accepting() {
            return ServicerState::Draining;
        }
        match (self.engine.client.get(), self.engine.error()) {
            (Some(client), _) if client.is_alive() => {
                if self.serving.load(Ordering::Acquire) {
                    ServicerState::Serving
                } else {
                    ServicerState::WarmingUp
                }
            }
            (Some(_), _) | (None, Some(_)) => ServicerState::EngineDown,
            (None, None) => ServicerState::Starting,
        }
    }

    /// What a metrics scrape reads off the servicer at that moment.
    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            active: self.registry.active(),
            state: Some(self.lifecycle_state()),
        }
    }
}

/// A running servicer's server: the gRPC listener on its runtime thread and
/// the `/metrics` listener beside it, with the lifecycle every wire's server
/// type delegates to.
pub(crate) struct Server {
    thread: ServerThread,
    /// The `/metrics` listener, when the configuration asked for one.
    metrics_listener: Option<MetricsListener>,
}

impl Server {
    /// Bind `config.bind_address` and serve `routes` (the wire's service) with
    /// the health service beside it, on a dedicated runtime thread named
    /// `thread_name`; returns once the listener is bound. The `/metrics`
    /// listener binds first, when `config.metrics_bind_address` asks for one,
    /// so a scrape during the engine's start answers; the KV-event relay reads
    /// its load records off `state` and follows the publisher from the start.
    /// Once the listener is bound, `connect` gets the inputs of the engine
    /// connect (the endpoints and the engine count off `config`, its two
    /// start-up bounds, its silence bounds and the server's last-error slot)
    /// and the future it returns runs on the server runtime, fire-and-forget:
    /// the wire's engine connect, which gates health; dropping the runtime
    /// with the server thread cancels it. The listening line names `engine`
    /// with the bound address as the engines log it (the listener's
    /// `local_addr`, empty when the read fails).
    pub(crate) fn start<S, C, Fut>(
        thread_name: &str,
        config: ServicerConfig,
        service_name: &str,
        engine: &'static str,
        state: &Arc<S>,
        routes: Routes,
        connect: C,
    ) -> Result<Self, ServicerError>
    where
        S: WireState + RankLoads,
        C: FnOnce(ConnectInputs) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let shared = state.shared();
        if let Some(relay) = &shared.kv_relay {
            relay.set_load_source(Arc::new(LoadFromState(Arc::downgrade(state))));
        }
        let metrics_listener = config
            .metrics_bind_address
            .as_deref()
            .map(|address| {
                let probe = Arc::downgrade(state);
                MetricsListener::start(
                    address,
                    Arc::clone(&shared.metrics),
                    Arc::new(move || {
                        probe
                            .upgrade()
                            .map(|state| state.shared().snapshot())
                            .unwrap_or_default()
                    }),
                )
            })
            .transpose()?;
        let health_state = Arc::clone(state);
        let health = HealthReporter::new(
            &["", service_name],
            Arc::new(move || health_state.shared().is_serving()),
        );
        let kv_relay = shared.kv_relay.clone();
        let silence = config.silence_bounds();
        let ServicerConfig {
            bind_address,
            ipc_base_url,
            handshake_address,
            engine_count,
            engine_startup_timeout,
            engine_startup_ceiling,
            ..
        } = config;
        let thread = ServerThread::start(
            thread_name,
            &bind_address,
            move |listener: TcpListener, shutdown: Shutdown, last_error| async move {
                // The KV-event relay follows the publisher from the start,
                // before any gateway asks, so its history and live-block
                // record cover the engine's whole life.
                if let Some(relay) = &kv_relay {
                    relay.start_at_boot();
                }
                // The address as the engines logged it before the start
                // moved here: a failed read leaves it empty rather than ending
                // the server thread (`ServerThread::start` read it once already).
                let address = listener
                    .local_addr()
                    .map(|address| address.to_string())
                    .unwrap_or_default();
                #[expect(
                    clippy::disallowed_methods,
                    reason = "engine connect is fire-and-forget; the runtime drop cancels it"
                )]
                let _connect = tokio::spawn(connect(ConnectInputs {
                    ipc_base_url,
                    handshake_address,
                    engine_count,
                    startup_timeout: engine_startup_timeout,
                    startup_ceiling: engine_startup_ceiling,
                    silence,
                    last_error,
                }));
                info!(%address, "{engine} gRPC servicer listening");
                // Graceful first: stop accepting, let open streams finish. A
                // connected client's idle keepalive connection would hold a
                // purely graceful shutdown forever, so the drain grace bounds
                // it and the server future is dropped (connections closed)
                // after.
                let graceful = transport::Server::builder()
                    // Router-preprocessed media arrives as multi-megabyte
                    // inline tensors; hyper's server defaults (1 MiB stream
                    // and connection windows, 16 KiB frames) would stall each
                    // such message on window updates. Size the windows and
                    // frames for them (grpc-core's BDP probing gets the
                    // Python servicer there on its own).
                    .initial_stream_window_size(Some(16 * 1024 * 1024))
                    .initial_connection_window_size(Some(64 * 1024 * 1024))
                    .max_frame_size(Some(1024 * 1024))
                    .add_routes(routes)
                    .add_service(HealthServer::new(health))
                    .serve_with_incoming_shutdown(
                        // tonic sets TCP_NODELAY only on listeners it owns; on
                        // this one Nagle would coalesce per-token frames into
                        // bursts whenever a connection is lightly loaded.
                        TcpListenerStream::new(listener).map(|conn| {
                            if let Ok(stream) = &conn {
                                if let Err(error) = stream.set_nodelay(true) {
                                    warn!(%error, "could not set TCP_NODELAY on an accepted connection");
                                }
                            }
                            conn
                        }),
                        shutdown.clone().map(|_| ()),
                    );
                let forced = async move {
                    let grace = shutdown.await;
                    tokio::time::sleep(grace).await;
                };
                tokio::select! {
                    result = graceful => result.map_err(|error| error.to_string()),
                    () = forced => {
                        warn!("gRPC servicer closed its remaining connections after the drain grace");
                        Ok(())
                    }
                }
            },
        )?;
        Ok(Self {
            thread,
            metrics_listener,
        })
    }

    /// The bound `ip:port`.
    pub(crate) fn address(&self) -> String {
        self.thread.address().to_string()
    }

    /// The metrics listener's bound `ip:port`; `None` when none was asked for.
    pub(crate) fn metrics_address(&self) -> Option<String> {
        self.metrics_listener
            .as_ref()
            .map(|listener| listener.address().to_string())
    }

    pub(crate) fn running(&self) -> bool {
        self.thread.running()
    }

    pub(crate) fn last_error(&self) -> Result<Option<String>, ServicerError> {
        self.thread.last_error()
    }

    /// Drain `shared` and stop: admission closes, every registered
    /// cancellation fires so the streams end (which aborts their engine
    /// requests) before the listener closes, then the server thread gets up
    /// to `timeout` to wind down. The metrics listener goes last, inside what
    /// is left of the budget, so a scrape during the drain still reads the
    /// servicer draining.
    pub(crate) fn stop(&self, shared: &Shared, timeout: Duration) -> Result<(), ServicerError> {
        shared.set_serving(false);
        shared.registry.cancel_all()?;
        let started = Instant::now();
        let stopped = self.thread.stop(timeout);
        if let Some(listener) = &self.metrics_listener {
            listener.stop(timeout.saturating_sub(started.elapsed()))?;
        }
        stopped
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;
    use crate::{metrics::Identity, record_error};

    fn shared() -> Shared {
        let metrics = ServicerMetrics::new(
            &Identity {
                model: "m".to_string(),
                wire: "test",
                ranks: 1,
                max_model_len: 8,
                matches_string_stops: false,
            },
            false,
        )
        .expect("metrics without a listener");
        Shared::new(None, metrics, None)
    }

    /// Before the engine is up every reader answers from the launcher's
    /// figures alone: no capacity, the launcher's window or none, the
    /// launcher's version or `unknown`, the launcher's ranks or one.
    #[test]
    fn the_handshake_readers_fall_back_to_the_launcher_before_the_engine_is_up() {
        let shared = shared();
        assert!(shared.ready().is_none());
        assert_eq!(shared.kv_capacity_tokens(), 0);
        assert_eq!(shared.running_window(64), 64);
        assert_eq!(shared.running_window(0), 0);
        assert_eq!(shared.engine_version("0.5.1", "wire-"), "0.5.1");
        assert_eq!(shared.engine_version("", "wire-"), "unknown");
        assert_eq!(shared.rank_count(4), 4);
        assert_eq!(shared.rank_count(0), 1);
        assert_eq!(shared.rank_count(-2), 1);
    }

    /// The drain switch closes admission with health; the readiness switch
    /// leaves admission open, so the owner's own warm-up requests still pass.
    #[test]
    fn the_two_switches_differ_in_admission() {
        let shared = shared();
        assert!(shared.registry.is_accepting());
        assert!(!shared.is_serving(), "no engine link yet");
        assert_eq!(shared.health_message("healthy"), "Engine is starting");
        assert_eq!(shared.snapshot().state, Some(ServicerState::Starting));

        shared.set_ready(false);
        assert!(
            shared.registry.is_accepting(),
            "readiness held, admission open"
        );
        assert_eq!(shared.health_message("healthy"), "Warming up");

        shared.set_serving(false);
        assert!(!shared.registry.is_accepting(), "a drain closes admission");
        assert_eq!(shared.health_message("healthy"), "Draining");
        assert_eq!(shared.snapshot().state, Some(ServicerState::Draining));

        shared.set_serving(true);
        assert!(shared.registry.is_accepting());
        assert_eq!(shared.health_message("healthy"), "Engine is starting");
    }

    /// Without a link the engine is "still connecting"; a recorded failure
    /// names itself.
    #[test]
    fn the_engine_error_reads_the_link() {
        let shared = shared();
        assert!(!shared.engine_ready());
        let status = shared.engine().err().expect("no client yet");
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert_eq!(status.message(), "engine is still connecting");

        shared.engine.fail("refused".to_string());
        let status = shared.engine().err().expect("still no client");
        assert_eq!(status.message(), "engine connection failed: refused");
        assert_eq!(shared.health_message("healthy"), "Engine connection failed");
        assert_eq!(shared.snapshot().state, Some(ServicerState::EngineDown));
    }

    /// A wire of the shared state alone: no engine, no load figures.
    struct TestState(Shared);

    impl WireState for TestState {
        fn shared(&self) -> &Shared {
            &self.0
        }
    }

    impl RankLoads for TestState {
        fn rank_load(
            &self,
            _dp_rank: Option<i32>,
        ) -> Option<smg_grpc_client::common_proto::EngineLoad> {
            None
        }

        fn engine_link(&self) -> &EngineLink {
            &self.0.engine
        }
    }

    /// Once the listener is bound, the start hands the wire's connect the
    /// inputs the configuration said and runs it on the server runtime; what
    /// the connect records in its error slot is the server's last error.
    #[test]
    fn the_start_hands_the_connect_the_configurations_inputs_once_bound() {
        let state = Arc::new(TestState(shared()));
        let mut config =
            ServicerConfig::new("127.0.0.1:0", "ipc:///tmp/engine", "tcp://127.0.0.1:1");
        config.engine_count = 3;
        config.engine_startup_timeout = Duration::from_secs(7);
        config.engine_startup_ceiling = None;
        config.engine_silence_timeout = Duration::from_secs(11);
        config.engine_prefill_tokens_per_sec = 500;
        let (seen_tx, seen_rx) = mpsc::channel();
        let server = Server::start(
            "smg-test-servicer",
            config,
            "test.Service",
            "Test",
            &state,
            Routes::default(),
            move |inputs| async move {
                record_error(&inputs.last_error, "the test's connect failed".to_string());
                let _ = seen_tx.send((
                    inputs.ipc_base_url,
                    inputs.handshake_address,
                    inputs.engine_count,
                    inputs.startup_timeout,
                    inputs.startup_ceiling,
                    inputs.silence.window,
                    inputs.silence.prefill_tokens_per_sec,
                    std::thread::current().name().map(str::to_string),
                ));
            },
        )
        .expect("the server starts");
        let (ipc, handshake, engines, timeout, ceiling, window, rate, thread) = seen_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the connect ran");
        assert_eq!(ipc, "ipc:///tmp/engine");
        assert_eq!(handshake, "tcp://127.0.0.1:1");
        assert_eq!(engines, 3);
        assert_eq!(timeout, Duration::from_secs(7));
        assert_eq!(ceiling, None);
        assert_eq!(window, Duration::from_secs(11));
        assert_eq!(rate, 500);
        assert_eq!(
            thread.as_deref(),
            Some("smg-servicer-rt"),
            "the server runtime"
        );
        assert!(server.running());
        assert_eq!(
            server
                .last_error()
                .expect("the slot is readable")
                .as_deref(),
            Some("the test's connect failed")
        );
        server
            .stop(&state.0, Duration::from_secs(5))
            .expect("the server stops");
    }
}
