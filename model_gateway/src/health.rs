//! Liveness, readiness, and health endpoints: O(1) event-maintained readiness
//! state and an optional isolated probe listener. Not k8s-specific, though
//! Kubernetes is the motivating consumer.
//!
//! # Why this exists (#1694)
//!
//! `/readiness` used to scan the whole fleet on every probe (`get_all()`
//! Arc-clones every worker plus a tokenizer lookup per gRPC worker —
//! O(workers) per probe), and all probes were ordinary tasks on the request
//! runtime behind the full middleware stack. Under reactor starvation at fleet
//! scale, probe latency spiked past the caller's timeout (e.g. a kubelet
//! liveness deadline) and pods were killed while perfectly able to serve.
//!
//! Two countermeasures, both additive:
//!
//! 1. **O(1) readiness reads.** A background maintainer task derives the
//!    readiness decision from `WorkerRegistry` state and publishes it as a
//!    [`ReadinessSnapshot`] behind an [`ArcSwap`]. Probe handlers load the
//!    snapshot (one atomic pointer load) and never touch the registry. The
//!    maintainer recomputes on every `WorkerEvent` (bursts coalesced) and on
//!    a short checkpoint interval. The checkpoint covers the status
//!    mutations that bypass the registry broadcast — `set_status()` called
//!    directly by `ActivateWorkersStep`, the mesh inbound health refresh in
//!    `WorkerRegistry::on_remote_worker_state`, FFI bindings — and tokenizer
//!    registrations, which emit no worker event at all.
//!
//! 2. **Dedicated probe listener.** When `--health-check-port` (config
//!    `health_check_port`) is set, `/liveness`, `/readiness`, and `/health`
//!    are *also* served on that port by a minimal router with no middleware,
//!    running on a current-thread tokio runtime driven by its own OS thread
//!    (the `build_in_runtime` pattern), so probes cannot be starved by the
//!    request runtime. The routes always remain on the main listener too;
//!    point the orchestrator's probes at the dedicated port. Unset means no
//!    extra listener. The listener is plain HTTP and serves until process
//!    exit, so probes stay answerable through the entire drain window.
//!
//! # Drain semantics
//!
//! Once graceful shutdown begins (`InFlightRequestTracker::begin_drain`),
//! `/readiness` reports `503` with reason `"draining"` on **both** listeners
//! while `/liveness` and `/health` keep returning `200`, so the load balancer
//! or orchestrator stops routing new connections during the
//! endpoint-propagation window without restarting the process.

use std::{net::SocketAddr, sync::Arc, time::Duration};

use arc_swap::ArcSwap;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use llm_tokenizer::TokenizerRegistry;
use serde_json::json;
use tokio::{
    sync::broadcast::{
        error::{RecvError, TryRecvError},
        Receiver,
    },
    task::JoinHandle,
};
use tracing::{debug, error, info, warn};

use crate::{
    config::{RouterConfig, RoutingMode},
    middleware::ProbeResponse,
    observability::inflight_tracker::InFlightRequestTracker,
    worker::{event::WorkerEvent, WorkerRegistry, WorkerType},
};

/// How often the maintainer re-derives readiness from the registry even
/// without worker events. This bounds the staleness of mutations that
/// bypass the `WorkerRegistry` broadcast (direct `set_status()` callers,
/// tokenizer registrations) and is deliberately shorter than the metrics
/// collectors' 3s checkpoint: readiness gates traffic, so a flip should
/// land within one probe interval.
const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(1);

/// Result of one readiness evaluation. Field semantics are identical to the
/// values the `/readiness` handler used to compute inline per probe.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReadinessSnapshot {
    /// Per-mode worker presence: any healthy worker in a single-pool mode,
    /// or at least one healthy worker for every required disaggregated role.
    pub workers_ready: bool,
    /// Tokenizer-autoload gate: every healthy gRPC/ZMQ worker's tokenizer is
    /// registered (`true` when autoload is disabled — the gateway does not
    /// manage tokenizers at all then).
    pub tokenizers_ready: bool,
    /// Number of healthy workers (reported in the ready response body).
    pub healthy_workers: usize,
    /// Total number of registered workers (reported in the ready response
    /// body).
    pub total_workers: usize,
}

/// Shared O(1) probe state: the cached readiness snapshot plus the drain
/// flag. Handlers on the main listener and on the dedicated probe listener
/// read the same instance.
pub struct ProbeState {
    readiness: ArcSwap<ReadinessSnapshot>,
    inflight_tracker: Arc<InFlightRequestTracker>,
}

impl std::fmt::Debug for ProbeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProbeState")
            .field("readiness", &self.readiness.load())
            .field("draining", &self.inflight_tracker.is_draining())
            .finish_non_exhaustive()
    }
}

impl ProbeState {
    /// Create probe state with an all-not-ready snapshot (matches the
    /// previous behaviour for an empty registry). The first recompute by
    /// the maintainer replaces it.
    pub fn new(inflight_tracker: Arc<InFlightRequestTracker>) -> Arc<Self> {
        Arc::new(Self {
            readiness: ArcSwap::from_pointee(ReadinessSnapshot::default()),
            inflight_tracker,
        })
    }

    /// Current readiness snapshot (single atomic pointer load).
    pub fn readiness(&self) -> Arc<ReadinessSnapshot> {
        self.readiness.load_full()
    }

    /// Re-derive the readiness snapshot from live state.
    ///
    /// This is the exact computation the `/readiness` handler used to run
    /// inline per probe (see `server::readiness` before #1694), relocated
    /// onto the maintainer task so probes become O(1) reads.
    pub fn recompute(
        &self,
        worker_registry: &WorkerRegistry,
        tokenizer_registry: &TokenizerRegistry,
        router_config: &RouterConfig,
    ) {
        self.recompute_with(worker_registry, router_config, |model_id| {
            tokenizer_registry.get(model_id).is_some()
        });
    }

    /// Core of [`Self::recompute`] with the tokenizer lookup injected, so
    /// tests can exercise the gating logic without loading real tokenizers.
    fn recompute_with(
        &self,
        worker_registry: &WorkerRegistry,
        router_config: &RouterConfig,
        tokenizer_registered: impl Fn(&str) -> bool,
    ) {
        let workers = worker_registry.get_all();
        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();

        let workers_ready = if router_config.enable_igw
            && !matches!(
                &router_config.mode,
                RoutingMode::PrefillDecode { .. } | RoutingMode::EncodePrefillDecode { .. }
            ) {
            !healthy_workers.is_empty()
        } else {
            match &router_config.mode {
                RoutingMode::PrefillDecode { .. } => {
                    let has_prefill = healthy_workers
                        .iter()
                        .any(|w| matches!(w.worker_type(), WorkerType::Prefill));
                    let has_decode = healthy_workers
                        .iter()
                        .any(|w| matches!(w.worker_type(), WorkerType::Decode));
                    has_prefill && has_decode
                }
                RoutingMode::EncodePrefillDecode { .. } => {
                    let has_encode = healthy_workers
                        .iter()
                        .any(|w| matches!(w.worker_type(), WorkerType::Encode));
                    let has_prefill = healthy_workers
                        .iter()
                        .any(|w| matches!(w.worker_type(), WorkerType::Prefill));
                    let has_decode = healthy_workers
                        .iter()
                        .any(|w| matches!(w.worker_type(), WorkerType::Decode));
                    has_encode && has_prefill && has_decode
                }
                RoutingMode::Regular { .. } => !healthy_workers.is_empty(),
                RoutingMode::OpenAI { .. } => !healthy_workers.is_empty(),
                RoutingMode::Anthropic { .. } => !healthy_workers.is_empty(),
                RoutingMode::Gemini { .. } => !healthy_workers.is_empty(),
            }
        };

        // A worker reports healthy (engine SERVING) as soon as its process is
        // up, but the gateway autoloads each gRPC-pipeline worker's tokenizer
        // asynchronously afterward (`SubmitTokenizerJobStep`,
        // fire-and-forget). Until that lands, generation requests fail with
        // `tokenizer_not_found`, so `/readiness` must not report ready yet.
        // Hold readiness until every healthy gRPC and ZMQ worker's tokenizer
        // is registered — both ride the gRPC pipeline and speak tokens on the
        // wire. HTTP/proxy workers never autoload a local tokenizer and are
        // exempt; when autoload is disabled the gateway does not manage
        // tokenizers at all.
        let tokenizers_ready = router_config.disable_tokenizer_autoload
            || healthy_workers
                .iter()
                .filter(|w| w.connection_mode().uses_grpc_pipeline())
                .all(|w| tokenizer_registered(w.model_id()));

        self.readiness.store(Arc::new(ReadinessSnapshot {
            workers_ready,
            tokenizers_ready,
            healthy_workers: healthy_workers.len(),
            total_workers: workers.len(),
        }));
    }

    /// Build the `/readiness` response from cached state only: one drain
    /// flag load plus one snapshot pointer load — no registry access.
    ///
    /// Response bodies are byte-identical to the previous inline handler,
    /// with one addition: while draining, readiness reports not-ready with
    /// reason `"draining"` so load balancers stop routing new connections
    /// during shutdown (liveness intentionally stays OK — see module docs).
    pub fn readiness_response(&self) -> Response {
        if self.inflight_tracker.is_draining() {
            return mark_probe(
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "status": "not ready",
                        "reason": "draining"
                    })),
                )
                    .into_response(),
            );
        }

        let snapshot = self.readiness.load();
        if snapshot.workers_ready && snapshot.tokenizers_ready {
            mark_probe(
                (
                    StatusCode::OK,
                    Json(json!({
                        "status": "ready",
                        "healthy_workers": snapshot.healthy_workers,
                        "total_workers": snapshot.total_workers
                    })),
                )
                    .into_response(),
            )
        } else {
            let reason = if snapshot.workers_ready {
                "tokenizer not yet registered"
            } else {
                "insufficient healthy workers"
            };
            mark_probe(
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "status": "not ready",
                        "reason": reason
                    })),
                )
                    .into_response(),
            )
        }
    }
}

/// Body served by `/liveness` and `/health` on both listeners. Liveness
/// deliberately stays OK while draining: the process is healthy, it is
/// just not accepting new work.
pub fn liveness_response() -> Response {
    mark_probe((StatusCode::OK, "OK").into_response())
}

/// Attach the [`ProbeResponse`] logging marker: probe statuses (503 "not
/// ready" included) are expected states polled on a tight interval, and the
/// HTTP logging layer demotes marked responses to DEBUG.
fn mark_probe(mut response: Response) -> Response {
    response.extensions_mut().insert(ProbeResponse);
    response
}

/// Spawn the readiness maintainer: recomputes the snapshot on every
/// `WorkerRegistry` event (bursts coalesced into one recompute) and on a
/// checkpoint interval that catches broadcast-bypassing mutations and
/// tokenizer registrations. The loop runs on the caller's runtime — if that
/// runtime is starved the snapshot goes stale but probes still answer in
/// O(1), which is the failure mode we want (stale-but-served, never timed
/// out).
///
/// The initial snapshot is computed synchronously, after subscribing (so no
/// event can fall between snapshot and subscription): workers registered
/// before this call are reflected the moment it returns.
///
/// The task holds the worker and tokenizer registries by [`Weak`](std::sync::Weak) and
/// `upgrade()`s them each iteration, stopping the moment the owning server
/// drops them. Holding strong `Arc`s would form a reference cycle — the
/// registry owns the event `Sender`, so a task that also owned a registry
/// `Arc` would keep that `Sender` (and thus the whole registry graph and
/// `ProbeState`) alive for the entire process, never observing
/// `RecvError::Closed` and never exiting. That leak is invisible for a
/// single long-lived process but matters for tests and embedded callers that
/// create and drop servers. This task takes the Weak route since readiness
/// gates traffic and the fix is self-contained. `probe_state` stays a strong
/// `Arc` because the dedicated listener thread shares it and outlives the
/// maintainer.
pub fn spawn_readiness_maintainer(
    probe_state: Arc<ProbeState>,
    worker_registry: Arc<WorkerRegistry>,
    tokenizer_registry: Arc<TokenizerRegistry>,
    router_config: RouterConfig,
) -> JoinHandle<()> {
    // Subscribe before downgrading: the broadcast `Receiver` is independent
    // of the registry `Arc`, so it keeps delivering events (and reports
    // `Closed` once every `Sender` is gone) even though the task no longer
    // holds the registry strongly.
    let mut rx = worker_registry.subscribe_events();
    // Initial synchronous recompute while the strong Arcs are still in scope,
    // so workers registered before this call are reflected on return.
    probe_state.recompute(&worker_registry, &tokenizer_registry, &router_config);

    let worker_registry = Arc::downgrade(&worker_registry);
    let tokenizer_registry = Arc::downgrade(&tokenizer_registry);

    #[expect(
        clippy::disallowed_methods,
        reason = "readiness maintainer runs for the lifetime of the server (and exits when the registry is dropped)"
    )]
    tokio::spawn(async move {
        let mut checkpoint = tokio::time::interval(CHECKPOINT_INTERVAL);
        // Recompute is idempotent, so a starved maintainer only needs one
        // catch-up tick, not the default `Burst` of backlogged ticks.
        checkpoint.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        checkpoint.tick().await; // skip first immediate tick

        loop {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Ok(_) => drain_pending(&mut rx),
                        Err(RecvError::Lagged(n)) => {
                            warn!("readiness maintainer lagged by {n} worker events, recomputing");
                        }
                        Err(RecvError::Closed) => {
                            debug!("worker broadcast closed, readiness maintainer stopping");
                            break;
                        }
                    }
                }
                _ = checkpoint.tick() => {
                    // Catch changes that bypass the broadcast channel and
                    // tokenizer registrations (which emit no worker event).
                }
            }

            // Stop once the owning server drops the registries: upgrading
            // fails, breaking the would-be reference cycle so the task and
            // the registries deallocate on teardown.
            let (Some(workers), Some(tokenizers)) =
                (worker_registry.upgrade(), tokenizer_registry.upgrade())
            else {
                debug!("worker registry dropped, readiness maintainer stopping");
                break;
            };
            probe_state.recompute(&workers, &tokenizers, &router_config);
        }
    })
}

/// Drain every event already queued behind the one just received so a burst
/// collapses into a single recompute. The recompute reads live registry
/// state, so dropped/lagged event payloads are irrelevant.
fn drain_pending(rx: &mut Receiver<WorkerEvent>) {
    loop {
        match rx.try_recv() {
            Ok(_) => {}
            Err(TryRecvError::Empty | TryRecvError::Closed) => break,
            Err(TryRecvError::Lagged(n)) => {
                warn!("readiness maintainer lagged by {n} worker events while draining");
            }
        }
    }
}

// ── Dedicated probe listener ────────────────────────────────────────────

async fn probe_liveness() -> Response {
    liveness_response()
}

async fn probe_readiness(State(state): State<Arc<ProbeState>>) -> Response {
    state.readiness_response()
}

/// Minimal router for the dedicated probe listener: the three trivial probe
/// routes only (`health_generate` stays on the main listener — it proxies
/// to workers and is not an orchestrator probe), no middleware, no fallback
/// surprises beyond axum's default 404.
pub fn probe_router(probe_state: Arc<ProbeState>) -> Router {
    Router::new()
        .route("/liveness", get(probe_liveness))
        .route("/readiness", get(probe_readiness))
        .route("/health", get(probe_liveness))
        .with_state(probe_state)
}

/// Start the dedicated probe listener on `host:port`.
///
/// Binds synchronously so startup fails fast on port conflicts, then serves
/// the [`probe_router`] from a current-thread tokio runtime that drives the
/// event loop directly on its own OS thread (the `build_in_runtime`
/// pattern): probes keep answering even when every request-runtime worker is
/// busy, and the O(1) handlers never need extra worker threads. The thread
/// and runtime live until process exit — probes must stay answerable through
/// the whole drain window. Returns the bound address (useful when `port` is
/// `0`).
pub fn start_probe_listener(
    host: &str,
    port: u16,
    probe_state: Arc<ProbeState>,
) -> Result<SocketAddr, String> {
    // Parse host+port exactly like the main listener (server.rs): a
    // `host:port` SocketAddr parse handles bracketed IPv6 literals
    // (`[::1]`, `[::]`) that a bare `IpAddr::parse` of the host rejects, and
    // a bad host is a hard error rather than a silent rebind to 0.0.0.0
    // (which would hide a config typo and widen exposure). `port` may be 0
    // here on purpose — the ephemeral-port tests bind through this path; the
    // config-sourced value is rejected for 0 at the validation layer.
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|err| format!("invalid probe listener host '{host}': {err}"))?;

    let listener = std::net::TcpListener::bind(addr)
        .map_err(|err| format!("failed to bind probe listener on {addr}: {err}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|err| format!("failed to set probe listener non-blocking: {err}"))?;
    let local_addr = listener
        .local_addr()
        .map_err(|err| format!("failed to read probe listener address: {err}"))?;

    std::thread::Builder::new()
        .name("smg-probe".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(err) => {
                    error!("Failed to build probe listener runtime: {err}");
                    return;
                }
            };
            runtime.block_on(async move {
                let listener = match tokio::net::TcpListener::from_std(listener) {
                    Ok(listener) => listener,
                    Err(err) => {
                        error!("Failed to adopt probe listener socket: {err}");
                        return;
                    }
                };
                info!(
                    "Probe listener serving /liveness, /readiness, /health on {local_addr} \
                     (dedicated current-thread runtime)"
                );
                if let Err(err) = axum::serve(listener, probe_router(probe_state)).await {
                    error!("Probe listener error: {err}");
                }
            });
        })
        .map_err(|err| format!("failed to spawn probe listener thread: {err}"))?;

    Ok(local_addr)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};
    use tower::ServiceExt;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, Worker};

    fn worker(
        url: &str,
        model: &str,
        worker_type: WorkerType,
        connection_mode: ConnectionMode,
        status: WorkerStatus,
    ) -> Arc<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .connection_mode(connection_mode)
            .model(ModelCard::new(model))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .status(status)
            .build();
        Arc::new(worker)
    }

    fn http_worker(url: &str, status: WorkerStatus) -> Arc<dyn Worker> {
        worker(
            url,
            "llama-3",
            WorkerType::Regular,
            ConnectionMode::Http,
            status,
        )
    }

    fn config(mode: RoutingMode) -> RouterConfig {
        RouterConfig {
            mode,
            ..RouterConfig::default()
        }
    }

    fn regular_config() -> RouterConfig {
        config(RoutingMode::Regular {
            worker_urls: vec![],
        })
    }

    fn probe_state() -> Arc<ProbeState> {
        ProbeState::new(InFlightRequestTracker::new())
    }

    /// Recompute against `registry` with every tokenizer reported present.
    fn recompute(state: &ProbeState, registry: &WorkerRegistry, router_config: &RouterConfig) {
        state.recompute_with(registry, router_config, |_| true);
    }

    #[test]
    fn readiness_follows_worker_add_ready_not_ready_remove() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = regular_config();

        // Empty registry: not ready.
        recompute(&state, &registry, &router_config);
        let snapshot = state.readiness();
        assert!(!snapshot.workers_ready);
        assert_eq!(snapshot.total_workers, 0);
        assert_eq!(snapshot.healthy_workers, 0);

        // Registered but Pending: present, not healthy, not ready.
        let id = registry
            .register(http_worker("http://w1:8080", WorkerStatus::Pending))
            .unwrap();
        recompute(&state, &registry, &router_config);
        let snapshot = state.readiness();
        assert!(!snapshot.workers_ready);
        assert_eq!(snapshot.total_workers, 1);
        assert_eq!(snapshot.healthy_workers, 0);

        // Promoted to Ready: ready.
        registry.transition_status(&id, WorkerStatus::Ready);
        recompute(&state, &registry, &router_config);
        let snapshot = state.readiness();
        assert!(snapshot.workers_ready);
        assert!(snapshot.tokenizers_ready);
        assert_eq!(snapshot.healthy_workers, 1);

        // Demoted to NotReady: not ready again.
        registry.transition_status(&id, WorkerStatus::NotReady);
        recompute(&state, &registry, &router_config);
        assert!(!state.readiness().workers_ready);

        // Back to Ready then removed: not ready, empty counts.
        registry.transition_status(&id, WorkerStatus::Ready);
        registry.remove(&id);
        recompute(&state, &registry, &router_config);
        let snapshot = state.readiness();
        assert!(!snapshot.workers_ready);
        assert_eq!(snapshot.total_workers, 0);
        assert_eq!(snapshot.healthy_workers, 0);
    }

    #[test]
    fn pd_mode_requires_healthy_prefill_and_decode_with_igw() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = RouterConfig {
            enable_igw: true,
            ..config(RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
            })
        };

        let prefill_id = registry
            .register(worker(
                "http://p1:8080",
                "llama-3",
                WorkerType::Prefill,
                ConnectionMode::Http,
                WorkerStatus::Ready,
            ))
            .unwrap();
        recompute(&state, &registry, &router_config);
        assert!(
            !state.readiness().workers_ready,
            "prefill alone must not be ready"
        );

        registry
            .register(worker(
                "http://d1:8080",
                "llama-3",
                WorkerType::Decode,
                ConnectionMode::Http,
                WorkerStatus::Ready,
            ))
            .unwrap();
        recompute(&state, &registry, &router_config);
        assert!(state.readiness().workers_ready);

        registry.transition_status(&prefill_id, WorkerStatus::NotReady);
        recompute(&state, &registry, &router_config);
        assert!(!state.readiness().workers_ready);
    }

    #[test]
    fn igw_epd_does_not_accept_an_untyped_healthy_worker() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = RouterConfig {
            enable_igw: true,
            ..config(RoutingMode::EncodePrefillDecode {
                encode_urls: vec![],
                prefill_urls: vec![],
                decode_urls: vec![],
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            })
        };

        registry
            .register(http_worker("http://w1:8080", WorkerStatus::Ready))
            .unwrap();
        recompute(&state, &registry, &router_config);
        assert!(!state.readiness().workers_ready);
    }

    #[test]
    fn grpc_workers_gate_readiness_on_tokenizer_registration() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = regular_config();

        registry
            .register(worker(
                "grpc://g1:9000",
                "llama-3",
                WorkerType::Regular,
                ConnectionMode::Grpc,
                WorkerStatus::Ready,
            ))
            .unwrap();

        // Tokenizer missing: workers ready, tokenizers not.
        state.recompute_with(&registry, &router_config, |_| false);
        let snapshot = state.readiness();
        assert!(snapshot.workers_ready);
        assert!(!snapshot.tokenizers_ready);
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        // Tokenizer registered: ready.
        state.recompute_with(&registry, &router_config, |model_id| model_id == "llama-3");
        assert!(state.readiness().tokenizers_ready);
        assert_eq!(state.readiness_response().status(), StatusCode::OK);

        // Autoload disabled: tokenizer state is irrelevant.
        let no_autoload = RouterConfig {
            disable_tokenizer_autoload: true,
            ..regular_config()
        };
        state.recompute_with(&registry, &no_autoload, |_| false);
        assert!(state.readiness().tokenizers_ready);
    }

    /// ZMQ workers ride the same gRPC pipeline and the same fire-and-forget
    /// tokenizer autoload, and their wire is token-only, so they must gate
    /// readiness exactly like gRPC workers.
    #[test]
    fn zmq_workers_gate_readiness_on_tokenizer_registration() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = regular_config();

        registry
            .register(worker(
                "ipc:///tmp/smg-z1",
                "llama-3",
                WorkerType::Regular,
                ConnectionMode::Zmq,
                WorkerStatus::Ready,
            ))
            .unwrap();

        state.recompute_with(&registry, &router_config, |_| false);
        let snapshot = state.readiness();
        assert!(snapshot.workers_ready);
        assert!(!snapshot.tokenizers_ready);
        assert_eq!(
            state.readiness_response().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );

        state.recompute_with(&registry, &router_config, |model_id| model_id == "llama-3");
        assert!(state.readiness().tokenizers_ready);
        assert_eq!(state.readiness_response().status(), StatusCode::OK);
    }

    #[test]
    fn http_workers_are_exempt_from_tokenizer_gate() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = regular_config();

        registry
            .register(http_worker("http://w1:8080", WorkerStatus::Ready))
            .unwrap();

        // No tokenizer registered anywhere, but the only healthy worker is
        // HTTP, so the gate passes.
        state.recompute_with(&registry, &router_config, |_| false);
        let snapshot = state.readiness();
        assert!(snapshot.workers_ready);
        assert!(snapshot.tokenizers_ready);
    }

    #[test]
    fn recompute_resolves_tokenizers_through_registry() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let tokenizer_registry = TokenizerRegistry::new();
        let router_config = regular_config();

        registry
            .register(worker(
                "grpc://g1:9000",
                "llama-3",
                WorkerType::Regular,
                ConnectionMode::Grpc,
                WorkerStatus::Ready,
            ))
            .unwrap();

        // Empty tokenizer registry: the gRPC worker's tokenizer is missing.
        state.recompute(&registry, &tokenizer_registry, &router_config);
        let snapshot = state.readiness();
        assert!(snapshot.workers_ready);
        assert!(!snapshot.tokenizers_ready);
    }

    #[test]
    fn drain_flips_readiness_but_not_liveness() {
        let inflight_tracker = InFlightRequestTracker::new();
        let state = ProbeState::new(inflight_tracker.clone());
        let registry = WorkerRegistry::new();
        let router_config = regular_config();

        registry
            .register(http_worker("http://w1:8080", WorkerStatus::Ready))
            .unwrap();
        recompute(&state, &registry, &router_config);
        assert_eq!(state.readiness_response().status(), StatusCode::OK);

        inflight_tracker.begin_drain();
        assert_eq!(
            state.readiness_response().status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "readiness must flip as soon as drain begins"
        );
        assert_eq!(
            liveness_response().status(),
            StatusCode::OK,
            "liveness must stay OK while draining"
        );
    }

    /// Every probe response — ready and not-ready alike — must carry the
    /// [`ProbeResponse`] marker, or the HTTP logging layer floods ERROR with
    /// expected not-ready 503s on every poll (two lines per 2s in the e2e
    /// gateway logs before this marker existed).
    #[tokio::test]
    async fn probe_responses_carry_the_logging_marker() {
        let inflight_tracker = InFlightRequestTracker::new();
        let state = ProbeState::new(inflight_tracker.clone());

        // Not ready (initial snapshot), ready is irrelevant to the marker:
        // check the 503 path, the liveness 200 path, and the draining path.
        let not_ready = state.readiness_response();
        assert_eq!(not_ready.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(not_ready.extensions().get::<ProbeResponse>().is_some());

        assert!(liveness_response()
            .extensions()
            .get::<ProbeResponse>()
            .is_some());

        inflight_tracker.begin_drain();
        let draining = state.readiness_response();
        assert_eq!(draining.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(draining.extensions().get::<ProbeResponse>().is_some());
    }

    async fn get_probe(router: &Router, path: &str) -> (StatusCode, String) {
        use axum::body::{to_bytes, Body};

        let response = router
            .clone()
            .oneshot(
                http::Request::builder()
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        (status, String::from_utf8(bytes.to_vec()).unwrap())
    }

    #[tokio::test]
    async fn probe_router_serves_readiness_from_shared_state() {
        let inflight_tracker = InFlightRequestTracker::new();
        let state = ProbeState::new(inflight_tracker.clone());
        let registry = WorkerRegistry::new();
        let router_config = regular_config();
        let router = probe_router(state.clone());

        // Initial snapshot: not ready.
        let (status, body) = get_probe(&router, "/readiness").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains("insufficient healthy workers"), "got: {body}");

        // Liveness and health are always OK.
        let (status, body) = get_probe(&router, "/liveness").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "OK");
        let (status, _) = get_probe(&router, "/health").await;
        assert_eq!(status, StatusCode::OK);

        // A healthy worker lands and the maintainer recomputes: ready, with
        // counts in the body.
        registry
            .register(http_worker("http://w1:8080", WorkerStatus::Ready))
            .unwrap();
        recompute(&state, &registry, &router_config);
        let (status, body) = get_probe(&router, "/readiness").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("\"healthy_workers\":1"), "got: {body}");
        assert!(body.contains("\"total_workers\":1"), "got: {body}");

        // Drain flips readiness on this listener too; liveness stays OK.
        inflight_tracker.begin_drain();
        let (status, body) = get_probe(&router, "/readiness").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains("draining"), "got: {body}");
        let (status, _) = get_probe(&router, "/liveness").await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn probe_listener_serves_on_dedicated_runtime() {
        let inflight_tracker = InFlightRequestTracker::new();
        let state = ProbeState::new(inflight_tracker.clone());
        let registry = WorkerRegistry::new();
        let router_config = regular_config();

        // Port 0: bind an ephemeral port so parallel test runs never clash.
        let addr = start_probe_listener("127.0.0.1", 0, state.clone()).unwrap();
        let base = format!("http://{addr}");

        let resp = reqwest::get(format!("{base}/liveness")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.text().await.unwrap(), "OK");

        let resp = reqwest::get(format!("{base}/readiness")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        registry
            .register(http_worker("http://w1:8080", WorkerStatus::Ready))
            .unwrap();
        recompute(&state, &registry, &router_config);
        let resp = reqwest::get(format!("{base}/readiness")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain flips readiness on the dedicated listener; liveness stays OK.
        inflight_tracker.begin_drain();
        let resp = reqwest::get(format!("{base}/readiness")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(resp.text().await.unwrap().contains("draining"));
        let resp = reqwest::get(format!("{base}/health")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn maintainer_recomputes_on_registry_events() {
        let state = probe_state();
        let registry = Arc::new(WorkerRegistry::new());
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());
        let router_config = regular_config();

        // Keep strong Arcs to both registries for the duration of the test:
        // the maintainer now holds them by Weak (so it can exit on teardown),
        // mirroring production where `AppContext` owns them for the server's
        // life.
        let handle = spawn_readiness_maintainer(
            state.clone(),
            registry.clone(),
            tokenizer_registry.clone(),
            router_config,
        );

        // HTTP worker so the (empty) tokenizer registry is irrelevant.
        registry
            .register(http_worker("http://w1:8080", WorkerStatus::Ready))
            .unwrap();

        let became_ready = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if state.readiness().workers_ready {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(
            became_ready.is_ok(),
            "maintainer did not pick up the Registered event"
        );

        handle.abort();
    }

    /// Dropping the registries must let the maintainer exit: it holds them by
    /// `Weak`, so once the owning Arcs are gone `upgrade()` fails and the loop
    /// breaks. Without that, the task would keep the registries (and
    /// `ProbeState`) alive for the whole process — a leak for create/drop
    /// server lifecycles (tests, embedded callers).
    #[tokio::test]
    async fn maintainer_exits_when_registry_dropped() {
        let state = probe_state();
        let registry = Arc::new(WorkerRegistry::new());
        let tokenizer_registry = Arc::new(TokenizerRegistry::new());

        let handle = spawn_readiness_maintainer(
            state,
            registry.clone(),
            tokenizer_registry.clone(),
            regular_config(),
        );

        // Drop every strong Arc the owner held; only the maintainer's Weaks
        // (which cannot keep them alive) remain.
        drop(registry);
        drop(tokenizer_registry);

        // The task observes the dropped registry on its next checkpoint tick
        // (and the dropped broadcast Sender on recv) and returns; the
        // JoinHandle then completes on its own without an abort.
        let exited = tokio::time::timeout(Duration::from_secs(5), handle).await;
        assert!(
            matches!(exited, Ok(Ok(()))),
            "maintainer must exit once the registries are dropped, got {exited:?}"
        );
    }
}
