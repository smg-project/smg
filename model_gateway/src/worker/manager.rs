//! The worker fleet's lifecycle: the health probe schedule, the status
//! state machine the probes drive, the removal of failed workers, and the
//! admin fan-outs (cache flush, profiling, engine metrics, fleet loads).
//!
//! The schedule is one deadline per probed worker. Every slot a worker is
//! given comes from one decision, `ProbeSlot::of`: what the schedule does
//! with a worker of a given status under the manager's configuration (a
//! removed worker's slot and the lag recovery's rebuild are the only writes
//! beside it). A worker's deadline is re-armed when its probe launches
//! (never when it completes), and `in_flight` keeps a slow probe from being
//! joined by a second one.

use std::{
    collections::{HashMap, HashSet},
    future::Future,
    ops::ControlFlow,
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use axum::response::{IntoResponse, Response};
use chrono::Utc;
use futures::stream::{self, FuturesUnordered, StreamExt};
use http::StatusCode;
use openai_protocol::worker::{
    EngineAggregateMetricsSnapshot, FlushCacheResult, HealthCheckConfig, ProfileOptions,
    ProfileResult, SchedulerLoadSnapshot, WorkerLoadResponse, WorkerStatus,
};
use tokio::{
    sync::{broadcast, mpsc, Notify},
    task::JoinHandle,
};
use tracing::{debug, error, info, warn};

use crate::{
    config::UnhealthyRemovalScope,
    observability::metrics::{metrics_labels, Metrics},
    service_discovery::DISCOVERY_PROVIDER_LABEL,
    worker::{
        event::{WorkerConnected, WorkerEvent},
        liveness,
        load_state::LoadSnapshot,
        metrics_aggregator::{self, MetricPack},
        registry::WorkerId,
        worker::WorkerTypeExt,
        ConnectionMode, RuntimeType, Worker, WorkerOrigin, WorkerRegistry, WorkerResult,
        WorkerType,
    },
    workflow::{steps::local::discover_grpc_kv_engine_id, Job, JobQueue},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;
const MAX_CONCURRENT_HEALTH_PROBES: usize = 128;

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
///
/// Requests are addressed to `worker.base_url()`, so under `--dp-aware`
/// every rank of the same base worker targets the rank-free endpoint —
/// `worker.url()` carries an `@{dp_rank}` suffix that is SMG-internal and
/// not a valid server address (#1993). Callers that want one request per
/// base worker must dedupe by base URL first (see `get_engine_metrics`).
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = worker.http_client().clone();
            let url = worker.base_url().to_string();
            let full_url = format!("{url}/{endpoint}");
            let api_key = worker.api_key().cloned();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

/// Lifecycle coordinator for the worker fleet.
///
/// Owns the background health check loop, applies the state machine to
/// probe outcomes, and triggers removal of `Failed` workers when
/// `--remove-unhealthy-workers` is set. Subscribes to `WorkerRegistry`
/// events to keep its internal schedule in sync with registrations,
/// removals, and replacements.
///
/// The static helpers (`get_worker_urls`, `flush_cache_all`,
/// `get_engine_metrics`, `fleet_loads`) are operational commands that
/// don't depend on lifecycle state and remain associated functions.
pub struct WorkerManager {
    handle: Option<JoinHandle<()>>,
    shutdown_notify: Arc<Notify>,
}

impl std::fmt::Debug for WorkerManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerManager").finish()
    }
}

/// Configuration for the WorkerManager health check loop.
#[derive(Debug, Clone)]
pub struct WorkerManagerConfig {
    /// Default check interval used when a worker has no override.
    pub default_check_interval_secs: u64,
    /// Which workers that reach Failed get a `Job::RemoveWorker`.
    pub remove_unhealthy: UnhealthyRemovalScope,
}

/// How a profiling call is spelled in its log line and its summary.
struct ProfileVerb {
    /// The log line as the call goes out (`Starting profiling on 3 workers`).
    doing: &'static str,
    /// The summary when every worker answered (`Successfully started profiling on all 3 workers`).
    done: &'static str,
    /// The summary when some did not (`Profile start: 2 succeeded, 1 failed`).
    call: &'static str,
}

const PROFILE_START: ProfileVerb = ProfileVerb {
    doing: "Starting",
    done: "started",
    call: "start",
};

const PROFILE_STOP: ProfileVerb = ProfileVerb {
    doing: "Stopping",
    done: "stopped",
    call: "stop",
};

impl WorkerManager {
    /// Start the manager only if the background loop has work to do.
    ///
    /// The loop serves two roles: health polling and consuming the connect
    /// signal that promotes workers waiting on a backend handshake (the
    /// manager is that signal's sole consumer). When health checks are
    /// globally disabled AND the ZMQ transport is off AND no worker is already
    /// awaiting the connect signal, the loop would idle with nothing to
    /// service, so it is skipped and `None` is returned.
    ///
    /// `zmq_transport_enabled` is decided from configuration up front, not from
    /// the registry, because config workers register in the background after
    /// this call — the registry snapshot here cannot be trusted to already show
    /// them. When ZMQ is configured the loop starts regardless of the
    /// health-check flag so it is ready to consume the connect signal the
    /// instant a ZMQ worker's handshake lands. Health polling continues to
    /// honor each worker's own disable flag, so starting the loop with global
    /// health checks off does not resurrect polling for other transports.
    pub fn maybe_start(
        registry: Arc<WorkerRegistry>,
        config: WorkerManagerConfig,
        job_queue: Option<Arc<JobQueue>>,
        health_checks_enabled: bool,
        zmq_transport_enabled: bool,
    ) -> Option<Self> {
        if !health_checks_enabled
            && !zmq_transport_enabled
            && !registry.has_workers_awaiting_connect_signal()
        {
            info!(
                "Global health checks disabled, ZMQ transport off, and no workers await the \
                 connect signal; skipping WorkerManager"
            );
            return None;
        }
        let default_check_interval_secs = config.default_check_interval_secs;
        let manager = Self::start(registry, config, job_queue);
        debug!("Started WorkerManager loop with {default_check_interval_secs}s default interval");
        Some(manager)
    }

    /// Create and start the WorkerManager background loop.
    ///
    /// Spawns a single task that:
    ///   - Subscribes to `WorkerRegistry` events to maintain a per-worker
    ///     deadline schedule.
    ///   - Probes due workers via `Worker::check_health_async()`.
    ///   - Applies the state machine to probe outcomes and calls
    ///     `WorkerRegistry::transition_status()` to publish StatusChanged.
    ///   - Submits `Job::RemoveWorker` for Failed workers when removal is
    ///     enabled.
    pub fn start(
        registry: Arc<WorkerRegistry>,
        config: WorkerManagerConfig,
        job_queue: Option<Arc<JobQueue>>,
    ) -> Self {
        let shutdown_notify = Arc::new(Notify::new());
        let shutdown_clone = shutdown_notify.clone();

        // Subscribe BEFORE snapshotting the registry. Any registration that
        // lands after this line either (a) is already in the snapshot below
        // because it happened synchronously on this thread, or (b) arrives
        // as a Registered event in the broadcast buffer and is idempotently
        // applied by the event loop. The "a or b" dichotomy is what makes
        // startup deterministic regardless of task scheduling.
        let events_rx = registry.subscribe_events();

        // Run the bootstrap reconcile synchronously on the caller's thread
        // so the initial schedule is captured deterministically — not
        // whenever the spawned task happens to be scheduled. A worker
        // registered between WorkerManager::start() returning and the task
        // running (e.g. the mesh replay loop in server.rs, which runs
        // synchronously right after start()) would otherwise race the
        // task's own reconcile call.
        let mut next_check: HashMap<WorkerId, tokio::time::Instant> = HashMap::new();
        reconcile_from_registry(&registry, &mut next_check, &config);

        // Drain the connect-readiness signal (workers wake us the instant
        // their backend handshake lands). Taken once — a second manager on
        // the same registry would simply run poll-only.
        let connect_rx = registry.take_connect_signal_receiver();

        #[expect(
            clippy::disallowed_methods,
            reason = "WorkerManager loop runs for the lifetime of the registry; handle is stored and abort() runs on drop"
        )]
        let handle = tokio::spawn(async move {
            run_health_loop(
                registry,
                events_rx,
                connect_rx,
                next_check,
                config,
                job_queue,
                shutdown_clone,
            )
            .await;
        });

        Self {
            handle: Some(handle),
            shutdown_notify,
        }
    }

    /// Gracefully shut down the WorkerManager loop, awaiting the task.
    /// Prefer this over dropping when an async context is available — it
    /// lets the in-flight probe iteration finish instead of aborting.
    pub async fn shutdown(&mut self) {
        self.shutdown_notify.notify_one();
        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

impl Drop for WorkerManager {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

struct ProbeCompletion {
    worker_id: WorkerId,
    worker: Arc<dyn Worker>,
    expected_revision: u64,
    launched_status: WorkerStatus,
    probe_result: WorkerResult<()>,
}

struct RemovalCandidate {
    worker_id: WorkerId,
    url: String,
    expected_revision: u64,
}

enum ProbeApplyResult {
    Applied(Option<(WorkerStatus, WorkerStatus)>),
    Stale,
}

type ProbeFutures = FuturesUnordered<Pin<Box<dyn Future<Output = ProbeCompletion> + Send>>>;

/// What the schedule does with a worker. Decided in one place,
/// [`ProbeSlot::of`], for every writer of the schedule (the bootstrap and
/// lag-recovery reconcile, the registration and replacement events, the
/// connect signal, the due probes) and for the readers that ask whether a
/// Failed worker is being removed (the connect signal, the loop after a
/// probe).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeSlot {
    /// Probed at this interval: the worker's own when it sets one, else the
    /// manager's default.
    Probe(Duration),
    /// Never probed: health checks are disabled for the worker, or its drain
    /// is the removal workflow's (`DrainWorkersStep`) and ends in a removal;
    /// a probe would only churn metrics, and `compute_next_status` is a
    /// no-op for Draining anyway.
    Unprobed,
    /// Failed under `--remove-unhealthy-workers`, and the removal will take
    /// the worker: the schedule never holds it (a reconcile stays
    /// side-effect-free, and a slot would queue the removal at its due
    /// time). Without the flag, and for a worker the removal skips (a mesh
    /// import is owner-managed: [`should_remove_failed`]), a Failed worker
    /// keeps its slot: the probe is the only way an engine that comes back
    /// on the same address rejoins.
    Removal,
}

impl ProbeSlot {
    /// The slot of a worker at `status`; `removable` is the registry's word
    /// on whether a removal would take this worker
    /// ([`should_remove_failed`]). A Failed worker under removal is `Removal`
    /// whatever its health settings, so the connect signal can ask this to
    /// recognise a last poll's contact (the order only shows there: a worker
    /// with its checks disabled never holds a deadline).
    fn of(
        status: WorkerStatus,
        disable_health_check: bool,
        check_interval_secs: u64,
        config: &WorkerManagerConfig,
        removable: bool,
    ) -> Self {
        if status == WorkerStatus::Failed && removable {
            return Self::Removal;
        }
        if disable_health_check || status == WorkerStatus::Draining {
            return Self::Unprobed;
        }
        Self::Probe(Duration::from_secs(if check_interval_secs > 0 {
            check_interval_secs
        } else {
            config.default_check_interval_secs
        }))
    }

    /// The slot of a registered worker at `status`: the status the caller
    /// read at a probe's launch, or the one a transition just installed.
    fn of_worker(
        registry: &Arc<WorkerRegistry>,
        worker_id: &WorkerId,
        worker: &dyn Worker,
        status: WorkerStatus,
        config: &WorkerManagerConfig,
    ) -> Self {
        let health = &worker.metadata().health_config;
        Self::of(
            status,
            health.disable_health_check,
            health.check_interval_secs,
            config,
            should_remove_failed(registry, worker_id, config.remove_unhealthy),
        )
    }
}

/// Writes `worker_id`'s slot. A probed worker is due at once (`immediate`: a
/// registration or replacement is probed without waiting an interval) or
/// after its interval; any other slot leaves the schedule.
fn schedule_at(
    next_check: &mut HashMap<WorkerId, tokio::time::Instant>,
    worker_id: WorkerId,
    slot: ProbeSlot,
    now: tokio::time::Instant,
    immediate: bool,
) {
    match slot {
        ProbeSlot::Probe(interval) => {
            let delay = if immediate { Duration::ZERO } else { interval };
            next_check.insert(worker_id, now + delay);
        }
        ProbeSlot::Unprobed | ProbeSlot::Removal => {
            next_check.remove(&worker_id);
        }
    }
}

/// Background task body: deadline-driven probe loop + event subscription.
///
/// The loop keeps deadline scheduling, in-flight probe tracking, and event
/// handling in one place. Probes run concurrently via `FuturesUnordered` so a
/// slow worker does not block unrelated health checks or registry events.
///
/// `next_check` is seeded by `WorkerManager::start()` via a synchronous
/// `reconcile_from_registry` call, so this function never runs a bootstrap
/// reconcile of its own — by the time the task is scheduled, the caller's
/// thread has already captured a consistent registry snapshot. The only
/// in-loop reconcile is the lag-recovery rebuild triggered by
/// `RecvError::Lagged`.
async fn run_health_loop(
    registry: Arc<WorkerRegistry>,
    mut events_rx: broadcast::Receiver<WorkerEvent>,
    mut connect_rx: Option<mpsc::UnboundedReceiver<WorkerConnected>>,
    mut next_check: HashMap<WorkerId, tokio::time::Instant>,
    config: WorkerManagerConfig,
    job_queue: Option<Arc<JobQueue>>,
    shutdown: Arc<Notify>,
) {
    // Removals go to the queue only under a removal scope; label refreshes
    // (a worker coming into service) always do.
    let removal_queue = if config.remove_unhealthy == UnhealthyRemovalScope::Disabled {
        None
    } else {
        job_queue.clone()
    };
    let mut probes: ProbeFutures = FuturesUnordered::new();
    let mut in_flight: HashSet<WorkerId> = HashSet::new();

    loop {
        let now = tokio::time::Instant::now();
        let removals = queue_due_probes(
            &registry,
            &config,
            &mut next_check,
            &mut in_flight,
            &mut probes,
            now,
        );
        for removal in removals {
            if let Some(jq) = removal_queue.as_ref() {
                submit_removal_job(
                    &registry,
                    &removal.worker_id,
                    &removal.url,
                    removal.expected_revision,
                    jq,
                    config.remove_unhealthy,
                )
                .await;
            }
        }

        let sleep_until = next_check
            .values()
            .min()
            .copied()
            .unwrap_or_else(|| now + Duration::from_secs(config.default_check_interval_secs));

        tokio::select! {
            Some(completion) = probes.next(), if !probes.is_empty() => {
                let worker_id = completion.worker_id.clone();
                let worker = Arc::clone(&completion.worker);
                in_flight.remove(&worker_id);
                if let ProbeApplyResult::Applied(Some((_, new))) = apply_probe_completion(
                    &registry,
                    completion,
                    removal_queue.as_ref().map(|queue| (queue, config.remove_unhealthy)),
                    job_queue.as_ref(),
                )
                .await
                {
                    // The one slot a probe's outcome changes: a worker that
                    // just failed under removal leaves the schedule (its
                    // removal is in flight). Every other transition keeps the
                    // deadline armed at the probe's launch.
                    if ProbeSlot::of_worker(&registry, &worker_id, worker.as_ref(), new, &config)
                        == ProbeSlot::Removal
                    {
                        next_check.remove(&worker_id);
                    }
                }
            }
            () = tokio::time::sleep_until(sleep_until) => {}
            signal = recv_connect_signal(&mut connect_rx) => {
                match signal {
                    Some(connected) => {
                        apply_connect_signal_and_refresh(
                            &registry,
                            connected,
                            &mut next_check,
                            &config,
                            job_queue.as_ref(),
                        )
                        .await;
                    }
                    // Every sender lives on the registry (Arc), so recv() only
                    // returns None if the registry itself is gone. Drop the
                    // branch so the loop stops polling a dead channel.
                    None => connect_rx = None,
                }
            }
            event = events_rx.recv() => {
                if apply_registry_event(event, &registry, &mut next_check, &config).is_break() {
                    return;
                }
            }
            () = shutdown.notified() => {
                debug!("WorkerManager received shutdown signal");
                return;
            }
        }
    }
}

/// Keeps the schedule in step with the registry: a registered or replaced
/// worker is probed at once, a removed one leaves the schedule, and a lagged
/// subscription rebuilds it from a fresh snapshot. `Break` when the registry
/// is gone and the loop has nothing left to serve.
fn apply_registry_event(
    event: Result<WorkerEvent, broadcast::error::RecvError>,
    registry: &Arc<WorkerRegistry>,
    next_check: &mut HashMap<WorkerId, tokio::time::Instant>,
    config: &WorkerManagerConfig,
) -> ControlFlow<()> {
    match event {
        Ok(WorkerEvent::Registered { worker_id, worker })
        | Ok(WorkerEvent::Replaced {
            worker_id,
            new: worker,
            ..
        }) => {
            let slot = ProbeSlot::of_worker(
                registry,
                &worker_id,
                worker.as_ref(),
                worker.status(),
                config,
            );
            schedule_at(
                next_check,
                worker_id,
                slot,
                tokio::time::Instant::now(),
                true,
            );
        }
        Ok(WorkerEvent::Removed { worker_id, .. }) => {
            next_check.remove(&worker_id);
        }
        Ok(WorkerEvent::StatusChanged { .. }) => {
            // Self-published; nothing to do.
        }
        Err(broadcast::error::RecvError::Lagged(n)) => {
            warn!("WorkerManager lagged {n} events; rebuilding schedule from registry");
            next_check.clear();
            reconcile_from_registry(registry, next_check, config);
        }
        Err(broadcast::error::RecvError::Closed) => {
            debug!("WorkerEvent channel closed; WorkerManager exiting");
            return ControlFlow::Break(());
        }
    }
    ControlFlow::Continue(())
}

/// Side-effect-free startup reconcile: rebuild the schedule from the
/// current registry snapshot without publishing new events or removals.
fn reconcile_from_registry(
    registry: &Arc<WorkerRegistry>,
    next_check: &mut HashMap<WorkerId, tokio::time::Instant>,
    config: &WorkerManagerConfig,
) {
    let now = tokio::time::Instant::now();
    for descriptor in registry.reconcile_snapshot() {
        let slot = ProbeSlot::of(
            descriptor.status,
            descriptor.disable_health_check,
            descriptor.check_interval_secs,
            config,
            should_remove_failed(registry, &descriptor.worker_id, config.remove_unhealthy),
        );
        schedule_at(next_check, descriptor.worker_id, slot, now, false);
    }
}

fn queue_due_probes(
    registry: &Arc<WorkerRegistry>,
    config: &WorkerManagerConfig,
    next_check: &mut HashMap<WorkerId, tokio::time::Instant>,
    in_flight: &mut HashSet<WorkerId>,
    probes: &mut ProbeFutures,
    now: tokio::time::Instant,
) -> Vec<RemovalCandidate> {
    let capacity = MAX_CONCURRENT_HEALTH_PROBES.saturating_sub(in_flight.len());
    if capacity == 0 {
        return Vec::new();
    }

    let due_ids: Vec<WorkerId> = next_check
        .iter()
        .filter(|(worker_id, deadline)| now >= **deadline && !in_flight.contains(*worker_id))
        .map(|(worker_id, _)| worker_id.clone())
        .take(capacity)
        .collect();

    let mut removals = Vec::new();
    for worker_id in due_ids {
        let Some(worker) = registry.get(&worker_id) else {
            next_check.remove(&worker_id);
            continue;
        };

        let launched_status = worker.status();
        let slot = ProbeSlot::of_worker(
            registry,
            &worker_id,
            worker.as_ref(),
            launched_status,
            config,
        );
        let interval = match slot {
            ProbeSlot::Probe(interval) => interval,
            ProbeSlot::Unprobed => {
                next_check.remove(&worker_id);
                continue;
            }
            ProbeSlot::Removal => {
                next_check.remove(&worker_id);
                removals.push(RemovalCandidate {
                    worker_id: worker_id.clone(),
                    url: worker.base_url().to_string(),
                    expected_revision: worker.revision(),
                });
                continue;
            }
        };

        let expected_revision = worker.revision();
        next_check.insert(worker_id.clone(), now + interval);
        in_flight.insert(worker_id.clone());
        probes.push(Box::pin(async move {
            let probe_result = worker.check_health_async().await;
            if probe_result.is_ok() && engine_id_may_be_stale(launched_status, worker.as_ref()) {
                let timeout =
                    Duration::from_secs(worker.metadata().health_config.timeout_secs.max(1));
                refresh_kv_engine_id_after_recovery(&worker, timeout).await;
            }
            ProbeCompletion {
                worker_id,
                worker,
                expected_revision,
                launched_status,
                probe_result,
            }
        }));
    }

    removals
}

/// `removal` is where a worker that just failed goes, and which ones go:
/// `None` when nothing is removed.
async fn apply_probe_completion(
    registry: &Arc<WorkerRegistry>,
    completion: ProbeCompletion,
    removal: Option<(&Arc<JobQueue>, UnhealthyRemovalScope)>,
    refresh_queue: Option<&Arc<JobQueue>>,
) -> ProbeApplyResult {
    let ProbeCompletion {
        worker_id,
        worker,
        expected_revision,
        launched_status,
        probe_result,
    } = completion;
    let health_config = &worker.metadata().health_config;

    let probe_ok = match probe_result {
        Ok(()) => true,
        Err(err) => {
            warn!(
                worker_url = %worker.url(),
                error = %err,
                "Health probe failed"
            );
            false
        }
    };
    Metrics::record_worker_health_check(
        worker.worker_type().as_metric_label(),
        if probe_ok {
            metrics_labels::CB_SUCCESS
        } else {
            metrics_labels::CB_FAILURE
        },
    );
    if probe_ok {
        liveness::on_probe_passed(&worker);
    }

    let Some(((), transition)) =
        registry.apply_if_revision(&worker_id, expected_revision, |current_worker| {
            if launched_status == WorkerStatus::Pending {
                current_worker.total_pending_probes_increment();
            }
            (
                (),
                compute_next_status(current_worker, probe_ok, health_config),
            )
        })
    else {
        debug!(
            worker_url = %worker.url(),
            expected_revision,
            "Discarding stale probe outcome after worker replacement"
        );
        return ProbeApplyResult::Stale;
    };

    if let Some((old, new)) = transition {
        // At INFO: a worker that leaves or rejoins routing is the line an
        // outage is read from (a refusal names the pool's healthy count, not
        // why it moved); the probe failures behind it are WARN lines already.
        info!(
            worker_url = %worker.url(),
            ?old,
            ?new,
            "Worker status transition"
        );
        if new == WorkerStatus::Failed {
            if let Some((jq, scope)) = removal {
                submit_removal_job(
                    registry,
                    &worker_id,
                    worker.base_url(),
                    expected_revision,
                    jq,
                    scope,
                )
                .await;
            }
        }
        if new == WorkerStatus::Ready && labels_follow_the_engine(registry, &worker_id, &worker) {
            if let Some(jq) = refresh_queue {
                submit_label_refresh(&worker, jq).await;
            }
        }
    }

    ProbeApplyResult::Applied(transition)
}

/// Await the next connect-readiness signal, or park forever when there is no
/// receiver (health checks disabled, or a second manager already took it). The
/// `pending()` arm keeps this `select!` branch dormant instead of busy-looping.
async fn recv_connect_signal(
    rx: &mut Option<mpsc::UnboundedReceiver<WorkerConnected>>,
) -> Option<WorkerConnected> {
    match rx {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

/// A connect signal is a transition into service like a probe's: the
/// promoted worker's labels follow its engine under the same rule
/// (`labels_follow_the_engine`, a job queue to run the refresh on).
async fn apply_connect_signal_and_refresh(
    registry: &Arc<WorkerRegistry>,
    connected: WorkerConnected,
    next_check: &mut HashMap<WorkerId, tokio::time::Instant>,
    config: &WorkerManagerConfig,
    job_queue: Option<&Arc<JobQueue>>,
) {
    let Some((worker_id, worker)) = apply_connect_signal(registry, connected, next_check, config)
    else {
        return;
    };
    if let Some(jq) = job_queue {
        if labels_follow_the_engine(registry, &worker_id, &worker) {
            submit_label_refresh(&worker, jq).await;
        }
    }
}

/// Act on a connect signal: a backend handshake that just completed, or a
/// contact from a worker the health checker demoted while its transport was
/// gone (see `worker::liveness::on_contact`). Resolves the URL to a live
/// worker id; a signal that lost a race with a same-URL replacement, or
/// names a worker already removed, is discarded.
///
/// A contact is not a health read. A servicer answers its load poll and
/// relays KV events through a warm-up whose health it holds at not-serving,
/// and an engine whose `/health` answers 503 while it drains still answers
/// `GetLoads`; a worker promoted on that contact takes requests its engine is
/// not ready for. So where the worker has a health probe, the probe decides:
/// the signal moves it to now (one round trip instead of up to an interval
/// of waiting) and the state machine promotes on its success threshold, as
/// for any demoted worker. The signal flips the status itself only where it
/// is the health read: a ZMQ handshake (there is no health RPC on that wire;
/// the handshake is what its probe reads), or a worker whose checks are off
/// (nothing to read). Returns the worker the signal moved into service, for
/// what follows a transition (the label refresh); `None` when nothing moved.
fn apply_connect_signal(
    registry: &Arc<WorkerRegistry>,
    connected: WorkerConnected,
    next_check: &mut HashMap<WorkerId, tokio::time::Instant>,
    config: &WorkerManagerConfig,
) -> Option<(WorkerId, Arc<dyn Worker>)> {
    let WorkerConnected { url, revision } = connected;
    let Some(worker_id) = registry.get_id_by_url(&url) else {
        debug!(worker_url = %url, "Connect signal for an unknown worker; ignoring");
        return None;
    };
    let Some(worker) = registry.get(&worker_id) else {
        debug!(worker_url = %url, "Connect signal for a removed worker; ignoring");
        return None;
    };
    let slot = ProbeSlot::of_worker(
        registry,
        &worker_id,
        worker.as_ref(),
        worker.status(),
        config,
    );
    // A Failed worker under removal is on its way out: the contact that
    // signalled it was a last poll, not a return.
    if slot == ProbeSlot::Removal {
        debug!(worker_url = %url, "Connect signal for a worker being removed; ignoring");
        return None;
    }
    if matches!(slot, ProbeSlot::Probe(_)) && *worker.connection_mode() != ConnectionMode::Zmq {
        if worker.revision() != revision {
            debug!(worker_url = %url, "Connect signal for a replaced worker; ignoring");
            return None;
        }
        debug!(worker_url = %url, "Contact from a demoted worker: its health probe runs now");
        schedule_at(
            next_check,
            worker_id,
            slot,
            tokio::time::Instant::now(),
            true,
        );
        return None;
    }
    match registry.transition_status_if_revision(&worker_id, revision, WorkerStatus::Ready) {
        Some((old, new)) => {
            debug!(worker_url = %url, ?old, ?new, "Promoted worker on connect signal");
            // A promoted worker serves traffic, so it is on the schedule from
            // here whatever its slot was (a Failed worker's may have gone):
            // otherwise a later engine death would leave it Ready with a dead
            // client forever.
            let worker = registry.get(&worker_id)?;
            let slot = ProbeSlot::of_worker(registry, &worker_id, worker.as_ref(), new, config);
            schedule_at(
                next_check,
                worker_id.clone(),
                slot,
                tokio::time::Instant::now(),
                false,
            );
            Some((worker_id, worker))
        }
        None => {
            // No transition: already Ready (idempotent), or a stale revision
            // after a same-URL replacement. Polling covers either case.
            debug!(worker_url = %url, "Connect signal applied no transition");
            None
        }
    }
}

/// Whether a probe launched at `launched_status` that succeeds should re-read
/// the worker's KV engine id: after a failure (the engine may be a new
/// process), while a worker that registered before its engine reported an id
/// is still pending, or while a Ready worker's last recovery re-read has not
/// been confirmed. A Draining worker, or a pending one that already has an
/// id, is the same process registration read.
fn engine_id_may_be_stale(launched_status: WorkerStatus, worker: &dyn Worker) -> bool {
    match launched_status {
        WorkerStatus::Failed | WorkerStatus::NotReady => true,
        WorkerStatus::Pending => worker.kv_engine_id().is_none(),
        WorkerStatus::Ready => !worker.kv_engine_id_confirmed(),
        WorkerStatus::Draining => false,
    }
}

/// A PD worker that answers its probe again may be a new engine process on
/// the same address, with a new KV transfer engine id (#2491); a worker
/// registered while its engine was still coming up may have none at all.
/// Re-read the id before the worker is promoted, so the next handoff is
/// minted for the engine that is actually there. Only gRPC engines report
/// the id. The read is bounded by the probe timeout: one that fails or
/// expires keeps the previous id, says so, and is retried on the next probe;
/// one that completes without an id keeps the previous id and is not
/// retried. A stale id is a better outcome than a probe slot held forever or
/// a handoff with no id at all, and an id the engine does report wins over
/// the spec's: it is the one a handoff must target.
async fn refresh_kv_engine_id_after_recovery(worker: &Arc<dyn Worker>, timeout: Duration) {
    let spec = &worker.metadata().spec;
    if !matches!(spec.worker_type, WorkerType::Prefill | WorkerType::Decode)
        || *worker.connection_mode() != ConnectionMode::Grpc
        || spec.kv_connector.is_none()
    {
        return;
    }
    let read = tokio::time::timeout(
        timeout,
        discover_grpc_kv_engine_id(worker.url(), spec.runtime_type.as_str()),
    )
    .await;
    // A read that did not complete leaves the id unconfirmed, so the next
    // probe retries instead of the worker serving with a possibly stale id
    // until its next outage. A completed read confirms it, whether or not
    // the engine reports an id: there is nothing to retry.
    worker.set_kv_engine_id_confirmed(false);
    match read {
        Ok(Ok(Some(discovered))) => {
            let previous = worker.kv_engine_id();
            if worker.refresh_kv_engine_id(Some(discovered.clone())) {
                info!(
                    worker_url = %worker.url(),
                    ?previous,
                    discovered,
                    "Recovered PD worker reports a new KV engine id"
                );
            }
            worker.set_kv_engine_id_confirmed(true);
        }
        // The engine completed the read and reports no id (a servicer that
        // predates it, or a connector without one). A known id is kept, never
        // cleared, and nothing is retried.
        Ok(Ok(None)) => {
            match worker.kv_engine_id() {
                Some(previous) => warn!(
                    worker_url = %worker.url(),
                    previous,
                    "Recovered PD worker reports no KV engine id; keeping the previous one"
                ),
                None => debug!(
                    worker_url = %worker.url(),
                    "PD worker reports no KV engine id; handoffs carry none"
                ),
            }
            worker.set_kv_engine_id_confirmed(true);
        }
        Ok(Err(error)) => warn!(
            worker_url = %worker.url(),
            %error,
            "Could not re-read the KV engine id of a recovered PD worker; keeping the previous one and retrying on the next probe"
        ),
        Err(_) => warn!(
            worker_url = %worker.url(),
            timeout_secs = timeout.as_secs(),
            "Re-reading the KV engine id of a recovered PD worker timed out; keeping the previous one and retrying on the next probe"
        ),
    }
}

/// Whether a worker's labels are re-read from its engine when it comes into
/// service: a worker this node registered itself (a mesh import carries its
/// owner's labels), over a transport with metadata endpoints (a ZMQ engine
/// reports its geometry at the handshake), that is a self-hosted engine (an
/// external provider is never read for engine metadata, at registration or
/// later). Judged by kind, not by what registration's read returned: a read
/// that failed leaves empty labels behind, and the first transition into
/// service is what fills them in; an engine that offers no metadata answers
/// a refresh with no labels, which the refresh step skips.
fn labels_follow_the_engine(
    registry: &WorkerRegistry,
    worker_id: &WorkerId,
    worker: &Arc<dyn Worker>,
) -> bool {
    registry.origin_of(worker_id) == Some(WorkerOrigin::Local)
        && *worker.connection_mode() != ConnectionMode::Zmq
        && worker.metadata().spec.runtime_type != RuntimeType::External
}

/// A worker that came into service may be a new engine process on the same
/// address, or an engine that finished loading after the worker was
/// registered on provisional information. Its labels, and everything derived
/// from them (the decode admission window, multimodal placement, PD pairing),
/// are re-read through the update workflow, which rebuilds the worker only if
/// they changed. While another job for the worker is in flight (its
/// registration, which just read the labels; an update) nothing is submitted:
/// the refresh would take that job's status under the URL. A skipped or
/// failed submission is logged; the next transition retries.
async fn submit_label_refresh(worker: &Arc<dyn Worker>, job_queue: &Arc<JobQueue>) {
    let url = worker.url().to_string();
    match job_queue
        .submit_unless_in_flight(Job::RefreshWorkerLabels { url: url.clone() })
        .await
    {
        Ok(true) => debug!(
            worker_url = %url,
            "Submitted a label refresh for a worker that came into service"
        ),
        Ok(false) => debug!(
            worker_url = %url,
            "A job for the worker is in flight; its label refresh waits for the next transition into service"
        ),
        Err(error) => warn!(
            worker_url = %url,
            %error,
            "Could not submit a label refresh for a worker that came into service"
        ),
    }
}

/// Apply the state machine to a probe outcome. Returns the next status if
/// a transition is needed, `None` if the worker stays in its current state.
///
/// State machine rules:
///   - Pending → Ready on `success_threshold` consecutive successes
///   - Pending → Failed on `max_pending_probes` (10 × failure_threshold) total
///   - NotReady → Ready on `success_threshold` consecutive successes
///   - NotReady → Failed on `liveness_failure_threshold` (3 × failure_threshold)
///   - Ready → NotReady on `failure_threshold` consecutive failures
///   - Failed → Ready on `success_threshold` consecutive successes: the engine
///     came back on the same address (a restart), and without
///     `--remove-unhealthy-workers` this is the only way it rejoins
///   - Failed stays Failed on failure; Draining never transitions here
fn compute_next_status(
    worker: &Arc<dyn Worker>,
    probe_ok: bool,
    health_config: &HealthCheckConfig,
) -> Option<WorkerStatus> {
    let current_status = worker.status();
    let success_threshold = health_config.success_threshold as usize;
    let failure_threshold = health_config.failure_threshold as usize;
    // Liveness threshold: tolerate longer outages before declaring Failed,
    // analogous to K8s having separate readiness and liveness probes.
    let liveness_threshold = failure_threshold * 3;
    // Pending cap: prevent misconfigured URLs from sitting in Pending forever.
    let max_pending_probes = failure_threshold * 10;

    if probe_ok {
        worker.consecutive_failures_reset();
        let successes = worker.consecutive_successes_increment();

        if matches!(
            current_status,
            WorkerStatus::Pending | WorkerStatus::NotReady | WorkerStatus::Failed
        ) && successes >= success_threshold
        {
            worker.consecutive_successes_reset();
            worker.total_pending_probes_reset();
            return Some(WorkerStatus::Ready);
        }

        // Even on a successful probe, enforce the Pending cap. A worker
        // that flaps F,S,F,S,... never reaches success_threshold and would
        // otherwise grow `total_pending_probes` without bound.
        if current_status == WorkerStatus::Pending
            && worker.total_pending_probes() >= max_pending_probes
        {
            worker.consecutive_successes_reset();
            worker.consecutive_failures_reset();
            return Some(WorkerStatus::Failed);
        }

        None
    } else {
        worker.consecutive_successes_reset();
        let failures = worker.consecutive_failures_increment();

        match current_status {
            WorkerStatus::Ready => {
                if failures >= failure_threshold {
                    worker.consecutive_failures_reset();
                    return Some(WorkerStatus::NotReady);
                }
            }
            WorkerStatus::NotReady => {
                if failures >= liveness_threshold {
                    worker.consecutive_failures_reset();
                    return Some(WorkerStatus::Failed);
                }
            }
            WorkerStatus::Pending => {
                if worker.total_pending_probes() >= max_pending_probes {
                    worker.consecutive_failures_reset();
                    return Some(WorkerStatus::Failed);
                }
            }
            WorkerStatus::Failed | WorkerStatus::Draining => {
                // Nowhere further down to go. Failed is removed by
                // `--remove-unhealthy-workers` and otherwise stays probed so
                // a restart is noticed; Draining is removed by the discovery
                // drain timer once in-flight requests settle.
            }
        }

        None
    }
}

/// Whether removing `worker_id`, once Failed, is within `scope`.
///
/// Failed mesh-imported workers stay registered (demoted, unroutable) under
/// every scope: they are owner-managed, and a local removal only
/// desynchronizes this node — the live CRDT key re-imports the worker on the
/// next reconcile, probes fail again, and the remove/re-import loop churns
/// forever. Under `DiscoveryOwned` a local worker goes only when a discovery
/// provider registered it (it carries the provider label): the provider
/// registers it again once its engine returns, while nothing would re-add a
/// static or API-added one.
fn should_remove_failed(
    registry: &Arc<WorkerRegistry>,
    worker_id: &WorkerId,
    scope: UnhealthyRemovalScope,
) -> bool {
    if registry.origin_of(worker_id) == Some(WorkerOrigin::Mesh) {
        return false;
    }
    match scope {
        UnhealthyRemovalScope::Disabled => false,
        UnhealthyRemovalScope::AllLocal => true,
        UnhealthyRemovalScope::DiscoveryOwned => registry.get(worker_id).is_some_and(|worker| {
            worker
                .metadata()
                .spec
                .labels
                .contains_key(DISCOVERY_PROVIDER_LABEL)
        }),
    }
}

async fn submit_removal_job(
    registry: &Arc<WorkerRegistry>,
    worker_id: &WorkerId,
    worker_url: &str,
    expected_revision: u64,
    job_queue: &Arc<JobQueue>,
    scope: UnhealthyRemovalScope,
) {
    if !should_remove_failed(registry, worker_id, scope) {
        debug!(
            worker_id = %worker_id.as_str(),
            ?scope,
            "skipping removal of a failed worker outside the removal scope \
             (mesh-imported, or not discovered under the discovery default)"
        );
        return;
    }
    let url = worker_url.to_string();
    warn!(
        worker_id = %worker_id.as_str(),
        worker_url = %url,
        expected_revision,
        "Removing failed worker from registry"
    );
    // Guard the failed worker by its own id. The url is a DP base, so the
    // removal step matches the whole rank group; a bare revision would take
    // out every sibling that happens to sit at the same revision.
    let expected_revisions = HashMap::from([(worker_id.as_str().to_string(), expected_revision)]);
    if let Err(e) = job_queue
        .submit(Job::RemoveWorker {
            url: url.clone(),
            expected_revisions: Some(expected_revisions),
        })
        .await
    {
        error!(
            worker_url = %url,
            error = %e,
            "Failed to submit worker removal job"
        );
    }
}

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    /// Fan an admin operation out to workers in parallel, collecting
    /// successful worker URLs and per-worker failure messages.
    ///
    /// The connection-mode dispatch (HTTP endpoint vs gRPC RPC) lives in
    /// the [`Worker`] admin methods, so the fan-out is uniform.
    async fn admin_fan_out<F, Fut>(
        workers: Vec<Arc<dyn Worker>>,
        op: F,
    ) -> (Vec<String>, Vec<(String, String)>)
    where
        F: Fn(Arc<dyn Worker>) -> Fut,
        Fut: Future<Output = WorkerResult<()>>,
    {
        let futures: Vec<_> = workers
            .into_iter()
            .map(|worker| {
                let url = worker.url().to_string();
                let fut = op(worker);
                async move { (url, fut.await) }
            })
            .collect();

        let results: Vec<(String, WorkerResult<()>)> = stream::iter(futures)
            .buffer_unordered(MAX_CONCURRENT)
            .collect()
            .await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();
        for (url, result) in results {
            match result {
                Ok(()) => successful.push(url),
                Err(e) => failed.push((url, e.to_string())),
            }
        }
        (successful, failed)
    }

    /// Workers targeted by an admin op: all registered workers, or only
    /// the worker(s) whose URL matches the filter.
    fn admin_target_workers(
        worker_registry: &WorkerRegistry,
        url_filter: Option<&str>,
    ) -> Vec<Arc<dyn Worker>> {
        let workers = worker_registry.get_all();
        match url_filter {
            Some(url) => workers.into_iter().filter(|w| w.url() == url).collect(),
            None => workers,
        }
    }

    pub async fn flush_cache_all(worker_registry: &WorkerRegistry) -> FlushCacheResult {
        let all_workers = worker_registry.get_all();
        let total_workers = all_workers.len();
        let http_workers = all_workers
            .iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .count();
        // ZMQ engines have no cache-flush RPC; fan out only to workers that can
        // succeed instead of reporting every ZMQ worker as failed.
        let (workers, zmq_workers): (Vec<_>, Vec<_>) = all_workers
            .into_iter()
            .partition(|w| !matches!(w.connection_mode(), ConnectionMode::Zmq));
        let zmq_skipped = zmq_workers.len();
        let grpc_workers = total_workers - http_workers - zmq_skipped;

        if workers.is_empty() {
            let message = if zmq_skipped > 0 {
                format!(
                    "No cache-flush-capable workers available \
                     ({zmq_skipped} ZMQ workers skipped: no cache-flush RPC)"
                )
            } else {
                "No workers available for cache flush".to_string()
            };
            info!("{}", message);
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers,
                grpc_workers,
                zmq_workers: zmq_skipped,
                message,
            };
        }

        info!(
            "Flushing cache on {} workers ({} HTTP, {} gRPC, {} ZMQ skipped)",
            workers.len(),
            http_workers,
            grpc_workers,
            zmq_skipped
        );

        let (successful, failed) =
            Self::admin_fan_out(workers, |w| async move { w.flush_cache().await }).await;

        let mut message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };
        if zmq_skipped > 0 {
            message.push_str(&format!(
                " ({zmq_skipped} ZMQ workers skipped: no cache-flush RPC)"
            ));
        }

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers,
            grpc_workers,
            zmq_workers: zmq_skipped,
            message,
        }
    }

    /// Start a profiling run on all workers, or on the single worker
    /// matching `worker_url`.
    pub async fn start_profile_all(
        worker_registry: &WorkerRegistry,
        options: &ProfileOptions,
        worker_url: Option<&str>,
    ) -> ProfileResult {
        let options = options.clone();
        Self::admin_profile(worker_registry, worker_url, &PROFILE_START, move |w| {
            let options = options.clone();
            async move { w.start_profile(&options).await }
        })
        .await
    }

    /// Stop the in-flight profiling run on all workers, or on the single
    /// worker matching `worker_url`.
    pub async fn stop_profile_all(
        worker_registry: &WorkerRegistry,
        worker_url: Option<&str>,
    ) -> ProfileResult {
        Self::admin_profile(worker_registry, worker_url, &PROFILE_STOP, |w| async move {
            w.stop_profile().await
        })
        .await
    }

    /// One profiling call on all workers, or on the single worker matching
    /// `worker_url`: an empty fleet is answered with why, the call fans out
    /// ([`Self::admin_fan_out`]), and the summary says all succeeded or how
    /// many did not; `verb` spells the call in the log and the summary.
    async fn admin_profile<F, Fut>(
        worker_registry: &WorkerRegistry,
        worker_url: Option<&str>,
        verb: &ProfileVerb,
        op: F,
    ) -> ProfileResult
    where
        F: Fn(Arc<dyn Worker>) -> Fut,
        Fut: Future<Output = WorkerResult<()>>,
    {
        let workers = Self::admin_target_workers(worker_registry, worker_url);
        let total_workers = workers.len();

        if workers.is_empty() {
            return ProfileResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                message: match worker_url {
                    Some(url) => format!("No worker matching url '{url}'"),
                    None => "No workers available for profiling".to_string(),
                },
            };
        }

        info!("{} profiling on {} workers", verb.doing, total_workers);

        let (successful, failed) = Self::admin_fan_out(workers, op).await;

        let message = if failed.is_empty() {
            format!(
                "Successfully {} profiling on all {} workers",
                verb.done,
                successful.len()
            )
        } else {
            format!(
                "Profile {}: {} succeeded, {} failed",
                verb.call,
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        ProfileResult {
            successful,
            failed,
            total_workers,
            message,
        }
    }

    /// Build the fleet-wide load body from the monitor's published
    /// snapshot: the same numbers the routing policies are acting on.
    ///
    /// No request reaches a worker. That is what makes the route safe to
    /// stack — a gateway registered as a worker under another gateway
    /// answers from its cache instead of fanning one upstream poll out
    /// across its whole fleet.
    ///
    /// Workers the monitor has no load for are left out rather than reported
    /// as idle, so a gateway that has not polled yet answers with an empty
    /// `loads` array.
    pub(crate) fn fleet_loads(
        worker_registry: &WorkerRegistry,
        snapshot: &LoadSnapshot,
        model_id: Option<&str>,
    ) -> WorkerLoadResponse {
        let mut workers = worker_registry.get_all();
        workers.sort_by(|a, b| a.url().cmp(b.url()));

        let mut loads = Vec::new();
        for worker in workers {
            if model_id.is_some_and(|model| !worker.supports_model(model)) {
                continue;
            }
            let Some(report) = snapshot.get(worker.url()) else {
                continue;
            };
            let worker_type = worker.worker_type().to_string();
            loads.extend(report.loads.iter().map(|rank| SchedulerLoadSnapshot {
                worker: Some(worker.url().to_string()),
                worker_type: Some(worker_type.clone()),
                ..rank.clone()
            }));
        }

        WorkerLoadResponse {
            timestamp: Utc::now().to_rfc3339(),
            version: format!("smg-{}", env!("CARGO_PKG_VERSION")),
            dp_rank_count: loads.len() as i32,
            aggregate: EngineAggregateMetricsSnapshot::from_ranks(&loads),
            loads,
            sampled_at: None,
        }
    }

    pub async fn get_engine_metrics(worker_registry: &WorkerRegistry) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        // Scrape each base worker once: under `--dp-aware` all ranks share
        // the base URL, so per-rank fan-out would repeat identical scrapes
        // (#1993).
        let mut seen = HashSet::new();
        let workers: Vec<Arc<dyn Worker>> = workers
            .into_iter()
            .filter(|w| seen.insert(w.base_url().to_string()))
            .collect();

        let responses = fan_out(&workers, "metrics", reqwest::Method::GET).await;

        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {e}")),
        }
    }
}

#[cfg(test)]
mod tests;
