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
//! # The per-model section
//!
//! Both readiness bodies carry `models`: one entry per served model with the
//! gateway's own predicates applied to that model's workers (`ready`, a
//! `reason` when it is not, its healthy and registered worker counts), so a
//! rollout or a traffic switch can ask about one model from the same probe
//! while the status keeps answering for the gateway as a whole. A model whose
//! last worker left is no longer listed.
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
use openai_protocol::worker::{MmProcessingMode, WorkerStatus};
use serde::Serialize;
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
    config::{ReadinessRoleConfig, RouterConfig, RoutingMode},
    middleware::ProbeResponse,
    observability::inflight_tracker::InFlightRequestTracker,
    routers::grpc::multimodal::{mm_settings, models_without_media_processing},
    worker::{event::WorkerEvent, Worker, WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID},
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
    /// Per-mode worker capacity: any healthy worker in IGW/single-pool modes,
    /// or every pipeline role meeting its configured threshold in PD/EPD.
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
    /// Models whose requests that forward media references worker-side media
    /// placement (`--mm-processing worker`) cannot serve: a worker of the model
    /// reports a vision encoder and no worker of it processes media references
    /// (`multimodal::models_without_media_processing`, over every registered
    /// worker: the capability is configuration, not health). Every such
    /// request is refused `no_media_ref_capable_worker`; the ready body names
    /// the models (and the change is logged once) rather than holding
    /// readiness, since the router still serves the model's text requests
    /// and every other model, and a probe that failed would take all of them
    /// out of rotation. Empty under the other placements.
    pub models_without_media_processing: Vec<String>,
    /// Per-role capacity details for PD/EPD modes.
    pub roles: Option<ReadinessRolesSnapshot>,
    /// One entry per served model, sorted by id: the gateway's predicates
    /// above applied to the model's own workers (reported in both bodies).
    pub models: Vec<ModelReadiness>,
}

/// One served model's readiness, the `models` section of the `/readiness`
/// body. The same predicates the gateway-level decision reads, over the
/// workers of this model alone: a Ready worker (one per role the PD/EPD mode
/// needs; the fleet-level thresholds stay the gateway's), then the tokenizer
/// of every healthy worker on the gRPC pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelReadiness {
    pub model: String,
    pub ready: bool,
    /// Why the model is not ready (`no_healthy_worker`, `draining` when every
    /// worker of it drains, `no_healthy_prefill_worker` and its encode and
    /// decode siblings, `tokenizer_not_ready`); absent on a ready model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    pub healthy_workers: usize,
    pub total_workers: usize,
}

impl ModelReadiness {
    /// The entry for `model` from its workers, under the gateway's mode and
    /// tokenizer rule.
    fn derive(
        model: String,
        workers: &[Arc<dyn Worker>],
        router_config: &RouterConfig,
        tokenizer_registered: &impl Fn(&str) -> bool,
    ) -> Self {
        let healthy_of = |role: Option<WorkerType>| {
            workers
                .iter()
                .filter(|worker| {
                    worker.is_healthy() && role.is_none_or(|role| *worker.worker_type() == role)
                })
                .count()
        };
        let healthy_workers = healthy_of(None);
        // The roles the mode needs a healthy worker of, with the reason a
        // model that lacks one reports.
        const PREFILL: (WorkerType, &str) = (WorkerType::Prefill, "no_healthy_prefill_worker");
        const DECODE: (WorkerType, &str) = (WorkerType::Decode, "no_healthy_decode_worker");
        const ENCODE: (WorkerType, &str) = (WorkerType::Encode, "no_healthy_encode_worker");
        let roles_needed: &[(WorkerType, &'static str)] = match &router_config.mode {
            RoutingMode::PrefillDecode { .. } => &[PREFILL, DECODE],
            RoutingMode::EncodePrefillDecode { .. } => &[ENCODE, PREFILL, DECODE],
            RoutingMode::Regular { .. }
            | RoutingMode::OpenAI { .. }
            | RoutingMode::Anthropic { .. }
            | RoutingMode::Gemini { .. } => &[],
        };
        let reason = if healthy_workers == 0 {
            if workers
                .iter()
                .all(|worker| worker.status() == WorkerStatus::Draining)
            {
                Some("draining")
            } else {
                Some("no_healthy_worker")
            }
        } else if let Some((_, lacking)) = roles_needed
            .iter()
            .find(|(role, _)| healthy_of(Some(*role)) == 0)
        {
            Some(*lacking)
        } else if !router_config.disable_tokenizer_autoload
            && !workers
                .iter()
                .filter(|worker| {
                    worker.is_healthy() && worker.connection_mode().uses_grpc_pipeline()
                })
                .all(|worker| tokenizer_registered(worker.model_id()))
        {
            Some("tokenizer_not_ready")
        } else {
            None
        };
        Self {
            model,
            ready: reason.is_none(),
            reason,
            healthy_workers,
            total_workers: workers.len(),
        }
    }
}

/// Cached readiness evaluation for one pipeline role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReadinessRoleSnapshot {
    #[serde(rename = "healthy")]
    pub healthy_workers: usize,
    #[serde(rename = "registered")]
    pub registered_workers: usize,
    #[serde(rename = "expected", skip_serializing_if = "Option::is_none")]
    pub expected_workers: Option<usize>,
    #[serde(rename = "minimum")]
    pub min_healthy_workers: usize,
    #[serde(rename = "minimum_percent", skip_serializing_if = "Option::is_none")]
    pub min_healthy_percent: Option<u8>,
    #[serde(rename = "required")]
    pub required_workers: usize,
    pub ready: bool,
}

/// Role snapshots included in PD/EPD readiness responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ReadinessRolesSnapshot {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encode: Option<ReadinessRoleSnapshot>,
    pub prefill: ReadinessRoleSnapshot,
    pub decode: ReadinessRoleSnapshot,
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
        self.recompute_with(
            worker_registry,
            router_config,
            |model_id| tokenizer_registry.get(model_id).is_some(),
            mm_settings().processing.value,
        );
    }

    /// Core of [`Self::recompute`] with the tokenizer lookup and the media
    /// placement injected, so tests can exercise the gating logic without
    /// loading real tokenizers or seeding the process-wide settings.
    fn recompute_with(
        &self,
        worker_registry: &WorkerRegistry,
        router_config: &RouterConfig,
        tokenizer_registered: impl Fn(&str) -> bool,
        mm_processing: MmProcessingMode,
    ) {
        let workers = worker_registry.get_all();
        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();

        let role_snapshot = |worker_type: WorkerType, threshold: ReadinessRoleConfig| {
            let registered_workers = workers
                .iter()
                .filter(|worker| *worker.worker_type() == worker_type)
                .count();
            let healthy_count = healthy_workers
                .iter()
                .filter(|worker| *worker.worker_type() == worker_type)
                .count();
            let required_workers = threshold.required_workers();
            ReadinessRoleSnapshot {
                healthy_workers: healthy_count,
                registered_workers,
                expected_workers: threshold.expected_workers,
                min_healthy_workers: threshold.min_healthy_workers,
                min_healthy_percent: threshold.min_healthy_percent,
                required_workers,
                ready: healthy_count >= required_workers,
            }
        };

        // PD/EPD capacity gates take precedence over the IGW shortcut. Service
        // discovery automatically enables IGW, so checking IGW first would
        // silently reduce configured role thresholds back to "any healthy
        // worker" in the deployment mode that needs them most.
        let (workers_ready, roles) = match &router_config.mode {
            RoutingMode::PrefillDecode { .. } => {
                let thresholds = router_config.pd_readiness.unwrap_or_default();
                let prefill = role_snapshot(WorkerType::Prefill, thresholds.prefill);
                let decode = role_snapshot(WorkerType::Decode, thresholds.decode);
                (
                    prefill.ready && decode.ready,
                    Some(ReadinessRolesSnapshot {
                        encode: None,
                        prefill,
                        decode,
                    }),
                )
            }
            RoutingMode::EncodePrefillDecode { .. } => {
                let thresholds = router_config.pd_readiness.unwrap_or_default();
                let encode = role_snapshot(WorkerType::Encode, Default::default());
                let prefill = role_snapshot(WorkerType::Prefill, thresholds.prefill);
                let decode = role_snapshot(WorkerType::Decode, thresholds.decode);
                (
                    encode.ready && prefill.ready && decode.ready,
                    Some(ReadinessRolesSnapshot {
                        encode: Some(encode),
                        prefill,
                        decode,
                    }),
                )
            }
            RoutingMode::Regular { .. }
            | RoutingMode::OpenAI { .. }
            | RoutingMode::Anthropic { .. }
            | RoutingMode::Gemini { .. } => (!healthy_workers.is_empty(), None),
        };

        // A worker reports healthy (engine SERVING) as soon as its process is
        // up, but the gateway autoloads each gRPC-pipeline worker's tokenizer
        // asynchronously afterward (`SubmitTokenizerJobStep`,
        // fire-and-forget). Until that lands, generation requests are answered
        // 503 `tokenizer_not_ready` (503 `tokenizer_load_failed` once a load
        // was skipped or failed for good, or the tokenizer was removed), so
        // `/readiness` must not report ready yet.
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

        // Under worker-side media placement, a model whose workers report a
        // vision encoder and none of which processes media references refuses
        // every request that forwards media references
        // (`no_media_ref_capable_worker`): the ready body names it, and the
        // change is said once, here, not per probe.
        // Readiness itself is not held on it: the router still serves the
        // model's text requests and every other model, and a failed probe
        // would take the whole pod out of rotation for all of them.
        let media_unserved = models_without_media_processing(mm_processing, &workers);

        // The per-model section, from the registry's model index (a worker
        // that serves several models counts for each); the gateway's own
        // wildcard id is not a model.
        let mut model_ids = worker_registry.get_models();
        model_ids.sort_unstable();
        let models = model_ids
            .into_iter()
            .filter(|model| model != UNKNOWN_MODEL_ID)
            .map(|model| {
                let of_model = worker_registry.get_by_model(&model);
                ModelReadiness::derive(model, &of_model, router_config, &tokenizer_registered)
            })
            .collect();
        {
            let previous = self.readiness.load();
            if media_unserved != previous.models_without_media_processing {
                if media_unserved.is_empty() {
                    info!(
                        models = ?previous.models_without_media_processing,
                        "every multimodal model has a worker that processes media references \
                         again; /readiness no longer names any"
                    );
                } else {
                    warn!(
                        models = ?media_unserved,
                        "no worker processes media references for a model whose workers report \
                         a vision encoder: under --mm-processing worker its requests that \
                         forward media references are refused (no_media_ref_capable_worker) and \
                         /readiness names the model under models_without_media_processing until \
                         such a worker registers"
                    );
                }
            }
        }

        self.readiness.store(Arc::new(ReadinessSnapshot {
            workers_ready,
            tokenizers_ready,
            healthy_workers: healthy_workers.len(),
            total_workers: workers.len(),
            roles,
            models_without_media_processing: media_unserved,
            models,
        }));
    }

    /// Build the `/readiness` response from cached state only: one drain
    /// flag load plus one snapshot pointer load — no registry access.
    ///
    /// While draining, readiness reports not-ready with reason `"draining"`
    /// so load balancers stop routing new connections during shutdown
    /// (liveness intentionally stays OK). Under worker-side media placement,
    /// the ready body names models no worker processes media for under
    /// `models_without_media_processing`; status stays ready because the pod
    /// serves everything else. PD/EPD responses additionally expose the
    /// cached per-role capacity decision. Both bodies carry the per-model
    /// section (`models`).
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
            let mut body = json!({
                "status": "ready",
                "healthy_workers": snapshot.healthy_workers,
                "total_workers": snapshot.total_workers
            });
            if let Some(roles) = snapshot.roles {
                body["roles"] = json!(roles);
            }
            if !snapshot.models_without_media_processing.is_empty() {
                body["models_without_media_processing"] =
                    json!(snapshot.models_without_media_processing);
            }
            body["models"] = json!(snapshot.models);
            mark_probe((StatusCode::OK, Json(body)).into_response())
        } else {
            let reason = if snapshot.workers_ready {
                "tokenizer not yet registered"
            } else {
                "insufficient healthy workers"
            };
            let mut body = json!({"status": "not ready", "reason": reason});
            if let Some(roles) = snapshot.roles {
                body["roles"] = json!(roles);
            }
            body["models"] = json!(snapshot.models);
            mark_probe((StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response())
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
    // Parse host+port exactly like the main listener (server.rs), through the
    // shared bind-host rule: IPv6 with or without brackets (`::`, `[::]`,
    // `::1`), and a bad host is a hard error rather than a silent rebind to
    // 0.0.0.0 (which would hide a config typo and widen exposure). `port` may
    // be 0 here on purpose — the ephemeral-port tests bind through this path;
    // the config-sourced value is rejected for 0 at the validation layer.
    let addr = crate::config::bind_socket_addr(host, port)
        .map_err(|err| format!("probe listener: {err}"))?;

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
    use crate::{
        config::PdReadinessConfig,
        worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, Worker},
    };

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
        state.recompute_with(registry, router_config, |_| true, MmProcessingMode::Auto);
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

    /// Default readiness still requires both roles and follows health transitions.
    #[test]
    fn pd_mode_requires_healthy_prefill_and_decode() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = config(RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        });

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

        // Losing the prefill side flips readiness back off.
        registry.transition_status(&prefill_id, WorkerStatus::NotReady);
        recompute(&state, &registry, &router_config);
        assert!(!state.readiness().workers_ready);
    }

    /// Worker removal cannot shrink the configured percentage denominator.
    #[tokio::test]
    async fn pd_readiness_uses_expected_count_as_stable_percentage_denominator() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = RouterConfig {
            pd_readiness: Some(PdReadinessConfig {
                prefill: ReadinessRoleConfig {
                    expected_workers: Some(8),
                    min_healthy_workers: 4,
                    min_healthy_percent: Some(80),
                },
                decode: ReadinessRoleConfig {
                    expected_workers: Some(1),
                    min_healthy_workers: 1,
                    min_healthy_percent: Some(100),
                },
            }),
            ..config(RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
            })
        };

        registry
            .register(worker(
                "http://d1:8080",
                "llama-3",
                WorkerType::Decode,
                ConnectionMode::Http,
                WorkerStatus::Ready,
            ))
            .unwrap();
        for index in 1..=6 {
            registry
                .register(worker(
                    &format!("http://p{index}:8080"),
                    "llama-3",
                    WorkerType::Prefill,
                    ConnectionMode::Http,
                    WorkerStatus::Ready,
                ))
                .unwrap();
        }

        recompute(&state, &registry, &router_config);
        let snapshot = state.readiness();
        let roles = snapshot
            .roles
            .expect("PD readiness must expose role counts");
        assert_eq!(roles.prefill.required_workers, 7);
        assert_eq!(roles.prefill.healthy_workers, 6);
        assert!(!snapshot.workers_ready);
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["reason"], "insufficient healthy workers");
        assert_eq!(body["roles"]["prefill"]["required"], 7);

        registry
            .register(worker(
                "http://p7:8080",
                "llama-3",
                WorkerType::Prefill,
                ConnectionMode::Http,
                WorkerStatus::Ready,
            ))
            .unwrap();
        recompute(&state, &registry, &router_config);
        assert!(state.readiness().workers_ready);

        // The configured expectation remains the denominator after registry
        // removal; it must not silently become 6/6 = 100%.
        let id = registry.get_id_by_url("http://p7:8080").unwrap();
        registry.remove(&id);
        recompute(&state, &registry, &router_config);
        let snapshot = state.readiness();
        let roles = snapshot.roles.unwrap();
        assert_eq!(roles.prefill.expected_workers, Some(8));
        assert_eq!(roles.prefill.required_workers, 7);
        assert!(!snapshot.workers_ready);
    }

    /// Service discovery and unrelated workers cannot bypass role capacity gates.
    #[test]
    fn pd_role_thresholds_take_precedence_over_igw_shortcut() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = RouterConfig {
            enable_igw: true,
            pd_readiness: Some(PdReadinessConfig {
                prefill: ReadinessRoleConfig {
                    expected_workers: Some(2),
                    min_healthy_workers: 1,
                    min_healthy_percent: Some(100),
                },
                decode: ReadinessRoleConfig::default(),
            }),
            ..config(RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
            })
        };

        recompute(&state, &registry, &router_config);
        assert!(!state.readiness().workers_ready);

        registry
            .register(http_worker("http://w1:8080", WorkerStatus::Ready))
            .unwrap();
        recompute(&state, &registry, &router_config);
        assert!(
            !state.readiness().workers_ready,
            "a regular IGW worker must not satisfy PD role thresholds"
        );

        for (url, worker_type) in [
            ("http://p1:8080", WorkerType::Prefill),
            ("http://d1:8080", WorkerType::Decode),
        ] {
            registry
                .register(worker(
                    url,
                    "llama-3",
                    worker_type,
                    ConnectionMode::Http,
                    WorkerStatus::Ready,
                ))
                .unwrap();
        }
        recompute(&state, &registry, &router_config);
        assert!(
            !state.readiness().workers_ready,
            "one prefill must remain below the configured threshold of two"
        );

        registry
            .register(worker(
                "http://p2:8080",
                "llama-3",
                WorkerType::Prefill,
                ConnectionMode::Http,
                WorkerStatus::Ready,
            ))
            .unwrap();
        recompute(&state, &registry, &router_config);
        assert!(state.readiness().workers_ready);
    }

    /// EPD still needs an encode worker as well as both PD roles.
    #[test]
    fn epd_role_thresholds_take_precedence_over_igw_shortcut() {
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = RouterConfig {
            enable_igw: true,
            pd_readiness: Some(PdReadinessConfig {
                decode: ReadinessRoleConfig {
                    min_healthy_workers: 2,
                    ..Default::default()
                },
                ..Default::default()
            }),
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

        for (url, worker_type) in [
            ("http://e1:8080", WorkerType::Encode),
            ("http://p1:8080", WorkerType::Prefill),
            ("http://d1:8080", WorkerType::Decode),
        ] {
            registry
                .register(worker(
                    url,
                    "llama-3",
                    worker_type,
                    ConnectionMode::Http,
                    WorkerStatus::Ready,
                ))
                .unwrap();
        }
        recompute(&state, &registry, &router_config);
        assert!(!state.readiness().workers_ready);
        registry
            .register(worker(
                "http://d2:8080",
                "llama-3",
                WorkerType::Decode,
                ConnectionMode::Http,
                WorkerStatus::Ready,
            ))
            .unwrap();
        recompute(&state, &registry, &router_config);
        assert!(state.readiness().workers_ready);
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
        state.recompute_with(&registry, &router_config, |_| false, MmProcessingMode::Auto);
        let snapshot = state.readiness();
        assert!(snapshot.workers_ready);
        assert!(!snapshot.tokenizers_ready);
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        // Tokenizer registered: ready.
        state.recompute_with(
            &registry,
            &router_config,
            |model_id| model_id == "llama-3",
            MmProcessingMode::Auto,
        );
        assert!(state.readiness().tokenizers_ready);
        assert_eq!(state.readiness_response().status(), StatusCode::OK);

        // Autoload disabled: tokenizer state is irrelevant.
        let no_autoload = RouterConfig {
            disable_tokenizer_autoload: true,
            ..regular_config()
        };
        state.recompute_with(&registry, &no_autoload, |_| false, MmProcessingMode::Auto);
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

        state.recompute_with(&registry, &router_config, |_| false, MmProcessingMode::Auto);
        let snapshot = state.readiness();
        assert!(snapshot.workers_ready);
        assert!(!snapshot.tokenizers_ready);
        assert_eq!(
            state.readiness_response().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );

        state.recompute_with(
            &registry,
            &router_config,
            |model_id| model_id == "llama-3",
            MmProcessingMode::Auto,
        );
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
        state.recompute_with(&registry, &router_config, |_| false, MmProcessingMode::Auto);
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

    /// Under worker-side media placement, a model whose workers report a
    /// vision encoder and none of which processes media references is named
    /// in the ready body, until a capable worker registers; the other
    /// placements, and pools that say nothing about vision, name nothing.
    /// Readiness itself stays ready: the pod serves the model's text requests
    /// and every other model, and a failed probe would take all of them out
    /// of rotation. The pool's own holds still answer not ready, without the
    /// field.
    #[tokio::test]
    async fn worker_media_placement_names_models_without_media_processing_while_ready() {
        use crate::{routers::grpc::multimodal::SUPPORTS_VISION_LABEL, worker::RuntimeType};

        fn vision_worker(url: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
            let mut builder = BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Vllm)
                .model(ModelCard::new("qwen-vl"))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .status(WorkerStatus::Ready)
                .label(SUPPORTS_VISION_LABEL, "true");
            for (key, value) in labels {
                builder = builder.label(*key, *value);
            }
            Arc::new(builder.build())
        }
        async fn body_of(response: Response) -> serde_json::Value {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice(&bytes).unwrap()
        }

        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = regular_config();
        registry
            .register(vision_worker("grpc://v1:9000", &[]))
            .unwrap();

        state.recompute_with(
            &registry,
            &router_config,
            |_| true,
            MmProcessingMode::Worker,
        );
        let snapshot = state.readiness();
        assert!(snapshot.workers_ready && snapshot.tokenizers_ready);
        assert_eq!(
            snapshot.models_without_media_processing,
            vec!["qwen-vl".to_string()]
        );
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.extensions().get::<ProbeResponse>().is_some());
        let body = body_of(response).await;
        assert_eq!(body["status"], "ready");
        assert_eq!(body["healthy_workers"], 1);
        assert_eq!(body["models_without_media_processing"], json!(["qwen-vl"]));

        // The other placements serve such a model's media on the router.
        for placement in [MmProcessingMode::Auto, MmProcessingMode::Router] {
            state.recompute_with(&registry, &router_config, |_| true, placement);
            assert!(state.readiness().models_without_media_processing.is_empty());
            let body = body_of(state.readiness_response()).await;
            assert_eq!(body["status"], "ready");
            assert!(body.get("models_without_media_processing").is_none());
        }

        // A worker that processes media references takes the model off the list.
        registry
            .register(vision_worker(
                "grpc://v2:9000",
                &[("mm_processor", "inprocess")],
            ))
            .unwrap();
        state.recompute_with(
            &registry,
            &router_config,
            |_| true,
            MmProcessingMode::Worker,
        );
        assert!(state.readiness().models_without_media_processing.is_empty());
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_of(response).await;
        assert_eq!(body["healthy_workers"], 2);
        assert!(body.get("models_without_media_processing").is_none());

        // The pool's own hold still answers not ready, without the field.
        let empty = WorkerRegistry::new();
        state.recompute_with(&empty, &router_config, |_| true, MmProcessingMode::Worker);
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = body_of(response).await;
        assert_eq!(body["reason"], "insufficient healthy workers");
        assert!(body.get("models_without_media_processing").is_none());
    }

    /// The per-model section: the gateway's predicates applied to each
    /// served model's workers, with a reason on a model that is not ready,
    /// while the status keeps answering for the gateway as a whole; a model
    /// whose last worker left is no longer listed.
    #[tokio::test]
    async fn readiness_names_each_served_model_with_a_reason() {
        async fn body_of(response: Response) -> serde_json::Value {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice(&bytes).unwrap()
        }
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = regular_config();
        let m1 = registry
            .register(worker(
                "http://a:8080",
                "m1",
                WorkerType::Regular,
                ConnectionMode::Http,
                WorkerStatus::Ready,
            ))
            .unwrap();
        let m2 = registry
            .register(worker(
                "grpc://b:9000",
                "m2",
                WorkerType::Regular,
                ConnectionMode::Grpc,
                WorkerStatus::NotReady,
            ))
            .unwrap();
        let only_m1_tokenizer = |model: &str| model == "m1";

        // The gateway is ready on m1; m2 has no healthy worker and says so.
        state.recompute_with(
            &registry,
            &router_config,
            only_m1_tokenizer,
            MmProcessingMode::Auto,
        );
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_of(response).await;
        assert_eq!(body["status"], "ready");
        assert_eq!(
            body["models"],
            json!([
                {"model": "m1", "ready": true, "healthy_workers": 1, "total_workers": 1},
                {"model": "m2", "ready": false, "reason": "no_healthy_worker",
                 "healthy_workers": 0, "total_workers": 1},
            ])
        );

        // m2's worker passes its checks before its tokenizer lands: the model
        // waits on the tokenizer, and so does the gateway, which holds on
        // every gRPC worker's.
        registry.transition_status(&m2, WorkerStatus::Ready);
        state.recompute_with(
            &registry,
            &router_config,
            only_m1_tokenizer,
            MmProcessingMode::Auto,
        );
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = body_of(response).await;
        assert_eq!(body["reason"], "tokenizer not yet registered");
        assert_eq!(body["models"][0]["ready"], true);
        assert_eq!(
            body["models"][1],
            json!({"model": "m2", "ready": false, "reason": "tokenizer_not_ready",
                   "healthy_workers": 1, "total_workers": 1})
        );

        // The tokenizer lands: every model ready, no reason anywhere.
        state.recompute_with(&registry, &router_config, |_| true, MmProcessingMode::Auto);
        let body = body_of(state.readiness_response()).await;
        assert_eq!(body["status"], "ready");
        let models = body["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);
        assert!(models
            .iter()
            .all(|model| model["ready"] == true && model.get("reason").is_none()));

        // m2 drains: not ready with that reason; the gateway stays ready on m1.
        registry.transition_status(&m2, WorkerStatus::Draining);
        state.recompute_with(&registry, &router_config, |_| true, MmProcessingMode::Auto);
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_of(response).await;
        assert_eq!(body["models"][1]["reason"], "draining");
        assert_eq!(body["models"][1]["healthy_workers"], 0);

        // m2's last worker leaves: the model is no longer served, so no
        // longer listed.
        registry.remove(&m2);
        state.recompute_with(&registry, &router_config, |_| true, MmProcessingMode::Auto);
        let body = body_of(state.readiness_response()).await;
        assert_eq!(
            body["models"],
            json!([{"model": "m1", "ready": true, "healthy_workers": 1, "total_workers": 1}])
        );

        // No worker at all: the not-ready body carries an empty section.
        registry.remove(&m1);
        state.recompute_with(&registry, &router_config, |_| true, MmProcessingMode::Auto);
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = body_of(response).await;
        assert_eq!(body["reason"], "insufficient healthy workers");
        assert_eq!(body["models"], json!([]));
    }

    /// Under PD a model is ready when each role the mode needs has a healthy
    /// worker of it; the role it lacks is the reason, while the gateway's
    /// own answer reads the fleet-level thresholds.
    #[tokio::test]
    async fn pd_readiness_names_the_role_a_model_lacks() {
        async fn body_of(response: Response) -> serde_json::Value {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice(&bytes).unwrap()
        }
        let state = probe_state();
        let registry = WorkerRegistry::new();
        let router_config = config(RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        });
        for (url, model, role) in [
            ("http://p1:8080", "m1", WorkerType::Prefill),
            ("http://d1:8080", "m1", WorkerType::Decode),
            ("http://p2:8080", "m2", WorkerType::Prefill),
        ] {
            registry
                .register(worker(
                    url,
                    model,
                    role,
                    ConnectionMode::Http,
                    WorkerStatus::Ready,
                ))
                .unwrap();
        }
        recompute(&state, &registry, &router_config);
        let response = state.readiness_response();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_of(response).await;
        assert_eq!(body["roles"]["decode"]["healthy"], 1);
        assert_eq!(
            body["models"],
            json!([
                {"model": "m1", "ready": true, "healthy_workers": 2, "total_workers": 2},
                {"model": "m2", "ready": false, "reason": "no_healthy_decode_worker",
                 "healthy_workers": 1, "total_workers": 1},
            ])
        );
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
    async fn probe_listener_accepts_ipv6_with_or_without_brackets() {
        // Nothing to assert on a host without an IPv6 loopback.
        if std::net::TcpListener::bind("[::1]:0").is_err() {
            return;
        }
        let state = ProbeState::new(InFlightRequestTracker::new());
        for host in ["::1", "[::1]"] {
            let addr = start_probe_listener(host, 0, state.clone()).unwrap();
            assert!(addr.ip().is_ipv6(), "host {host} bound {addr}");
        }
        let err = start_probe_listener("probe-host", 0, state).unwrap_err();
        assert!(err.contains("invalid bind host 'probe-host'"), "got: {err}");
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

        let client = crate::clients::http_client();
        let resp = client.get(format!("{base}/liveness")).send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.text().await.unwrap(), "OK");

        let resp = client
            .get(format!("{base}/readiness"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        registry
            .register(http_worker("http://w1:8080", WorkerStatus::Ready))
            .unwrap();
        recompute(&state, &registry, &router_config);
        let resp = client
            .get(format!("{base}/readiness"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain flips readiness on the dedicated listener; liveness stays OK.
        inflight_tracker.begin_drain();
        let resp = client
            .get(format!("{base}/readiness"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(resp.text().await.unwrap().contains("draining"));
        let resp = client.get(format!("{base}/health")).send().await.unwrap();
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
