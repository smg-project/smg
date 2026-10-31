//! Worker load monitoring service.
//!
//! `WorkerMonitor` consolidates the previous `LoadMonitor` (per-group
//! polling loops) and `WorkerLoadManager` (DP rank load cache) into a
//! single coordinator that subscribes to `WorkerRegistry` events.
//!
//! ## Lifecycle
//!
//! - `new()` creates the cache and watch channel without spawning any
//!   tasks. The factory layer reads `worker_load_manager` immediately so
//!   policies that need the DP cache can be wired before workers exist.
//! - `start_event_loop()` subscribes to the registry, runs a synchronous
//!   bootstrap reconcile, and spawns the background event-handling task.
//!   It must be called after the initial worker population (mesh replay,
//!   K8s discovery, etc.) has finished — same ordering rule as
//!   `WorkerManager::start`.
//! - `Drop` aborts the event task and every per-group polling loop.
//!
//! ## Group lifecycle (event-driven)
//!
//! Polling is keyed by `WorkerGroupKey = (model_id, worker_type,
//! connection_mode)`. The event loop reacts to:
//!
//! - `Registered` / `Replaced`: reconcile every group the worker
//!   participates in. New groups start a polling loop; existing groups
//!   restart with a new interval if the per-worker override changed.
//! - `Removed`: reconcile every group the removed worker participated
//!   in. Empty groups stop their loops and evict cached state.
//! - `StatusChanged`: workers leaving `Ready` are evicted from the
//!   watch-channel snapshot and the DP cache (the group loop is left
//!   alone and will skip non-Ready workers on its next tick).
//! - `RecvError::Lagged`: stop every loop, clear shared state, and
//!   rebuild from the current registry snapshot. Monitoring state is
//!   derived data; full rebuild is the recovery mechanism.
//!
//! ## Polling
//!
//! Each group runs a single `tokio::time::interval` loop. Every tick:
//!
//! 1. Fetch the group's `Ready` workers. Polling is unconditional by
//!    default; with `--disable-load-monitoring` the tick is skipped
//!    unless a load-aware policy, engine-metrics re-export, or overload
//!    protection on a group member needs the data (the original
//!    conditional gate — never "never poll").
//! 2. Fetch loads concurrently from every `Ready` worker in the group.
//! 3. Update load-aware policies and the DP cache.
//! 4. Atomically clear stale entries for the group from the watch
//!    channel and merge in the fresh loads.

use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use dashmap::DashMap;
use futures::future;
use openai_protocol::worker::{
    RuntimeType, SchedulerLoadSnapshot, WorkerGroupKey, WorkerLoadResponse, WorkerStatus,
};
use parking_lot::{Mutex, RwLock};
use reqwest::StatusCode;
use smg_grpc_client::common_proto::EngineLoad;
use tokio::{
    sync::{broadcast, Notify},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    observability::metrics::Metrics,
    policies::PolicyRegistry,
    worker::{
        event::WorkerEvent,
        liveness,
        load_state::{LoadReceiver, LoadSnapshot, LoadState},
        ConnectionMode, Worker, WorkerRegistry,
    },
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Minimal Prometheus text-format scraper: metric name -> sample values
/// (one per label-set). Only the flat `name{labels} value` / `name value`
/// gauge lines that the engine load fetchers look up are collected;
/// `#` comment lines and unparsable values are skipped, and histogram or
/// summary buckets are simply never queried by name.
struct PromScrape {
    samples: HashMap<String, Vec<f64>>,
}

impl PromScrape {
    fn parse(text: &str) -> Self {
        let mut samples: HashMap<String, Vec<f64>> = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Line shape: `name{labels} value [timestamp]`. Split the head
            // (`name{labels}`) from the tail at the end of the label block so
            // spaces inside quoted label values don't confuse the split; fall
            // back to the first whitespace for unlabeled metrics. Then take
            // the FIRST tail token as the value, ignoring any optional
            // trailing timestamp.
            let (head, tail) = match line.rfind('}') {
                Some(close) => (&line[..=close], &line[close + 1..]),
                None => match line.split_once(char::is_whitespace) {
                    Some(split) => split,
                    None => continue,
                },
            };
            let Some(value) = tail.split_whitespace().next() else {
                continue;
            };
            let Ok(value) = value.parse::<f64>() else {
                continue;
            };
            let name = head.split('{').next().unwrap_or(head).trim();
            samples.entry(name.to_string()).or_default().push(value);
        }
        Self { samples }
    }

    /// Sum across all label-set samples for `name` (0.0 if absent). Right for
    /// additive counts like running/waiting requests across DP ranks.
    fn sum(&self, name: &str) -> f64 {
        self.samples
            .get(name)
            .map(|v| v.iter().sum())
            .unwrap_or(0.0)
    }

    /// Mean across all label-set samples for `name` (0.0 if absent). Right for
    /// ratios like KV-cache usage that must not be double-counted.
    fn mean(&self, name: &str) -> f64 {
        match self.samples.get(name) {
            Some(v) if !v.is_empty() => v.iter().sum::<f64>() / v.len() as f64,
            _ => 0.0,
        }
    }

    /// True when at least one sample exists for `name`.
    fn has(&self, name: &str) -> bool {
        self.samples.get(name).is_some_and(|v| !v.is_empty())
    }
}

/// DP rank load cache used by load-aware routing policies.
///
/// Pure in-memory data structure with no I/O. `WorkerMonitor` owns the
/// shared `Arc<WorkerLoadManager>` and updates it on every successful
/// poll; routing policies read from it via
/// `select_and_increment_lowest_dp_load`.
#[derive(Debug, Default)]
pub struct WorkerLoadManager {
    /// `<worker_url, <dp_rank, load>>`
    dp_cached_loads: RwLock<HashMap<String, HashMap<isize, isize>>>,
}

impl WorkerLoadManager {
    pub fn new() -> Self {
        Self {
            dp_cached_loads: RwLock::new(HashMap::new()),
        }
    }

    /// Replace cached rank loads for the supplied workers, preserving other workers.
    pub fn update_dp_loads(&self, loads: &HashMap<String, HashMap<isize, isize>>) {
        debug!("WorkerLoadManager update_dp_loads map:{:?}", loads);
        let mut cached = self.dp_cached_loads.write();
        cached.extend(loads.iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    /// Select and increment the least-loaded eligible rank. A virtual worker
    /// can only select its pinned rank; missing telemetry returns `None`.
    pub fn select_and_increment_lowest_dp_load(
        &self,
        worker: &dyn Worker,
        increment: isize,
    ) -> Option<isize> {
        let mut cached = self.dp_cached_loads.write();
        let loads = cached.get_mut(worker.url())?;
        let (&dp_rank, _) = loads
            .iter()
            .filter(|(rank, _)| {
                worker
                    .dp_rank()
                    .is_none_or(|pinned| **rank == pinned as isize)
            })
            .min_by_key(|&(rank, load)| (*load, *rank))?;
        if let Some(v) = loads.get_mut(&dp_rank) {
            *v += increment;
        }
        Some(dp_rank)
    }

    pub fn remove_workers(&self, urls: &[String]) {
        let mut cached = self.dp_cached_loads.write();
        for url in urls {
            cached.remove(url);
        }
    }

    /// Drop a single worker's cached DP loads. Avoids the
    /// `&[String]` allocation that `remove_workers` requires when the
    /// caller only has a single `&str`.
    pub fn remove_worker(&self, url: &str) {
        self.dp_cached_loads.write().remove(url);
    }

    /// Invalidate only the virtual worker's rank after a successful poll omits it.
    fn remove_rank(&self, url: &str, rank: usize) {
        let Ok(rank) = isize::try_from(rank) else {
            return;
        };
        if let Some(loads) = self.dp_cached_loads.write().get_mut(url) {
            loads.remove(&rank);
        }
    }

    /// Drop every cached DP load entry. Used by `WorkerMonitor` during
    /// `stop_all_groups` and lag-recovery rebuilds so the cache cannot
    /// hand out stale per-rank loads after a full reconcile.
    pub fn clear(&self) {
        self.dp_cached_loads.write().clear();
    }
}

/// Per-group polling loop state.
struct GroupState {
    handle: JoinHandle<()>,
    interval: Duration,
}

/// Outcome of probing one native load endpoint.
enum NativeLoads {
    Available(WorkerLoadResponse),
    /// The backend answered, and the endpoint is not there. Memoizable.
    Absent,
    /// Nothing was learned about the endpoint — retry on the next tick.
    Inconclusive,
}

/// A route that answers with the native load schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeLoadsPath {
    /// `/loads` — an SMG gateway registered as a worker.
    Gateway,
    /// `/v1/loads` — the engine-native route.
    Engine,
}

impl NativeLoadsPath {
    /// Probe order for a worker nothing is memoized about. Gateway first:
    /// a gateway serves only `/loads`, and an engine 404s it once and is
    /// then memoized onto its own route.
    const DISCOVERY: [Self; 2] = [Self::Gateway, Self::Engine];

    fn index(self) -> usize {
        match self {
            Self::Gateway => 0,
            Self::Engine => 1,
        }
    }

    /// Where to probe this route on `worker`: its engine's address (a
    /// dp-aware rank worker's URL carries the router's `@rank` suffix, which
    /// is no part of the host).
    fn probe_url(self, worker: &Arc<dyn Worker>) -> String {
        let base = worker.base_url();
        match self {
            // Sections beyond `core` degrade gracefully: an engine that does
            // not report them omits the fields, which deserialize to `None`.
            Self::Engine => format!("{base}/v1/loads?include=core,disagg,queues,memory"),
            // A gateway serves a whole fleet, so an unscoped `/loads` blends
            // every model it fronts. A worker registered for exactly one
            // model must be asked about that model alone.
            Self::Gateway => match worker.models().as_slice() {
                [card] => {
                    let model: String =
                        url::form_urlencoded::byte_serialize(card.id.as_bytes()).collect();
                    format!("{base}/loads?model={model}")
                }
                _ => format!("{base}/loads"),
            },
        }
    }
}

/// What a worker's probes have established about its native load routes.
///
/// Absence is tracked per route, not per worker: a 404 on one route is a
/// fact that outlives a timeout on the other. Collapsing the two into a
/// single "answered nothing" flag would discard the 404 whenever the other
/// route was merely inconclusive, and the dead route would be re-asked on
/// every tick forever.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct NativeLoadsMemo {
    answered: Option<NativeLoadsPath>,
    absent: [bool; NativeLoadsPath::DISCOVERY.len()],
}

impl NativeLoadsMemo {
    /// Routes worth probing this tick, in discovery order: the route that
    /// answered last if there is one, else everything not known absent.
    fn candidates(self) -> impl Iterator<Item = NativeLoadsPath> {
        NativeLoadsPath::DISCOVERY
            .into_iter()
            .filter(move |&path| match self.answered {
                Some(answered) => path == answered,
                None => !self.absent[path.index()],
            })
    }

    fn record_answered(&mut self, path: NativeLoadsPath) {
        self.answered = Some(path);
        self.absent[path.index()] = false;
    }

    fn record_absent(&mut self, path: NativeLoadsPath) {
        self.absent[path.index()] = true;
        if self.answered == Some(path) {
            self.answered = None;
        }
    }

    #[cfg(test)]
    pub(crate) fn answered(self) -> Option<NativeLoadsPath> {
        self.answered
    }

    #[cfg(test)]
    pub(crate) fn is_absent(self, path: NativeLoadsPath) -> bool {
        self.absent[path.index()]
    }
}

/// Load monitoring service that subscribes to `WorkerRegistry` events.
pub struct WorkerMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    pub worker_load_manager: Arc<WorkerLoadManager>,
    default_interval: Duration,
    /// When set, force load polling even if no load-aware routing policy is
    /// active (`--engine-metrics`). Every successful poll is re-exported.
    engine_metrics: bool,
    /// `--disable-load-monitoring`: restore the conditional poll gate (a
    /// load-aware/dp-rank policy, engine metrics, or overload protection on a
    /// group member still forces the poll) instead of polling every group
    /// unconditionally from registration onward.
    conditional_polling: bool,
    /// Shared immutable load snapshots: group ticks and the eviction flusher
    /// publish here; routing readers grab `Arc` snapshots.
    load_state: Arc<LoadState>,
    /// What each worker's probes established about its native load routes,
    /// so discovery is not repeated every tick. A worker whose routes are
    /// all absent is served from `/metrics`. Cleared by
    /// [`Self::evict_worker_loads`], which also runs on `Replaced` — an
    /// in-place image upgrade that adds or removes an endpoint is
    /// re-discovered.
    native_loads_memo: Arc<DashMap<String, NativeLoadsMemo>>,
    group_handles: Mutex<HashMap<WorkerGroupKey, GroupState>>,
    event_task: Mutex<Option<JoinHandle<()>>>,
    eviction_flush_task: Mutex<Option<JoinHandle<()>>>,
    liveness_sweep_task: Mutex<Option<JoinHandle<()>>>,
    /// Each worker's report by rank, as its stream and the poll last left
    /// it: a record pushed on the KV-event stream replaces its rank's entry
    /// ([`Self::apply_pushed_load`]), a poll's entry replaces a rank's that
    /// pushed nothing since the poll began ([`Self::apply_polled_load`]);
    /// every push rebuilds the worker's report from it, the way a poll
    /// reports every rank.
    pushed_ranks: Mutex<HashMap<String, BTreeMap<i32, SchedulerLoadSnapshot>>>,
    /// When each rank of a worker last pushed a load record, by receipt:
    /// the tick's poll decision reads it (see [`PollMode`]).
    pushed_at: Mutex<HashMap<String, BTreeMap<i32, Instant>>>,
    /// Pushed reports waiting for the next coalesced snapshot publish.
    pending_pushed: Mutex<HashMap<String, PushedReport>>,
    pushed_notify: Arc<Notify>,
    pushed_flush_task: Mutex<Option<JoinHandle<()>>>,
}

/// A worker's report built from pushed records, with the worker it came from
/// for the publish fence.
type PushedReport = (Arc<dyn Worker>, Arc<WorkerLoadResponse>);

/// Where a worker's load comes from at a tick. An engine that reports its
/// load on its KV-event stream is not asked for it again: in gRPC mode the
/// `GetLoads` poll is the fallback, for servicers that predate the pushed
/// record, for a stream that is down and for heartbeats that stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PollMode {
    /// No pushed record on file: the poll is the only source.
    Poll,
    /// Pushed records on file, but some pushing rank's newest is older than
    /// the tick interval: the poll takes over until the records resume.
    Fallback,
    /// Every pushing rank delivered a record within the tick interval: the
    /// stream is the source, no RPC.
    Suppressed,
}

impl PollMode {
    /// The `mode` label of `smg_engine_load_polls_total`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Poll => "poll",
            Self::Fallback => "fallback",
            Self::Suppressed => "skipped_fresh_push",
        }
    }
}

/// Telemetry a record does not carry (an event batch's record is the core
/// only; a heartbeat's and a stream's first carry it all) stays what the
/// last record that carried it said, so the gauges and `GET /loads` keep
/// the engine's figures between heartbeats. Absent is absent: a section the
/// engine never reports stays unset.
fn keep_telemetry(
    fresh: &mut SchedulerLoadSnapshot,
    previous: &SchedulerLoadSnapshot,
    record: &EngineLoad,
) {
    if record.cache_hit_rate.is_none() {
        fresh.cache_hit_rate = previous.cache_hit_rate;
    }
    if record.num_used_tokens.is_none() {
        fresh.num_used_tokens = previous.num_used_tokens;
    }
    if record.max_total_num_tokens.is_none() {
        fresh.max_total_num_tokens = previous.max_total_num_tokens;
    }
    if record.memory.is_none() {
        fresh.memory.clone_from(&previous.memory);
    }
    if record.queues.is_none() {
        fresh.queues.clone_from(&previous.queues);
    }
    if record.disaggregation.is_none() {
        fresh.kv_transfer_latency_ms = previous.kv_transfer_latency_ms;
        fresh.kv_transfer_speed_gb_s = previous.kv_transfer_speed_gb_s;
        fresh.prefill_queue_reqs = previous.prefill_queue_reqs;
        fresh.decode_queue_reqs = previous.decode_queue_reqs;
        fresh.disagg_mode.clone_from(&previous.disagg_mode);
    }
}

/// A pushed record is sampled this much before its receipt on top of its own
/// `age_ms`: the servicer-to-gateway one-way latency.
const PUSHED_ONE_WAY_MARGIN: Duration = Duration::from_millis(20);

/// The `version` of a report built from a worker's pushed load records
/// ([`WorkerMonitor::apply_pushed_load`]): what tells a reader of the shared
/// snapshot a stream's report, current at the stream's cadence, from a
/// poll's, up to a tick old. An engine's own answer never carries it.
pub(crate) const PUSHED_LOAD_VERSION: &str = "pushed";

/// Pushed reports are published into the shared snapshot in one rebuild per
/// window: the rebuild is O(fleet) pointer work, the records arrive per
/// scheduler step from every engine. Routing reads the snapshot, so a record
/// steers within a window of its receipt; the policies' dispatch credit
/// covers the gap. Twenty milliseconds is about one scheduler step, and at
/// fifty rebuilds a second a thousand-worker fleet spends well under a
/// percent of a core here.
const PUSHED_PUBLISH_WINDOW: Duration = Duration::from_millis(20);

/// Debounce window for batching worker evictions into one snapshot rebuild.
/// Registry churn (a rollout, a scale-down) emits removals as a gradual
/// stream; waiting a beat lets the whole wave land in a single publish while
/// staying far below the polling cadence that refreshes live entries.
const EVICTION_DEBOUNCE: Duration = Duration::from_millis(20);

impl Debug for WorkerMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerMonitor")
            .field("default_interval", &self.default_interval)
            .finish_non_exhaustive()
    }
}

impl WorkerMonitor {
    /// Construct a `WorkerMonitor` without spawning any background work.
    ///
    /// The caller must invoke [`Self::start_event_loop`] once initial
    /// workers have been registered. Until then, `worker_load_manager`
    /// is still safe to read; it just stays empty.
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        default_interval_secs: u64,
        engine_metrics: bool,
        conditional_polling: bool,
    ) -> Self {
        let load_state = Arc::new(LoadState::new(Arc::clone(&worker_registry)));
        Self {
            worker_registry,
            policy_registry,
            worker_load_manager: Arc::new(WorkerLoadManager::new()),
            default_interval: Duration::from_secs(default_interval_secs.max(1)),
            engine_metrics,
            conditional_polling,
            load_state,
            native_loads_memo: Arc::new(DashMap::new()),
            group_handles: Mutex::new(HashMap::new()),
            event_task: Mutex::new(None),
            eviction_flush_task: Mutex::new(None),
            liveness_sweep_task: Mutex::new(None),
            pushed_ranks: Mutex::new(HashMap::new()),
            pushed_at: Mutex::new(HashMap::new()),
            pending_pushed: Mutex::new(HashMap::new()),
            pushed_notify: Arc::new(Notify::new()),
            pushed_flush_task: Mutex::new(None),
        }
    }

    /// The absolute overload verdict of one load report (`None`: a poll that
    /// brought none, which clears the veto; absent means no opinion),
    /// applied to `worker` under its effective thresholds (resolved at
    /// registration; nothing happens while they are off). An edge is an
    /// INFO line with the report's figures, the thresholds and `source`:
    /// a worker vetoed by overload reads as "unavailable" in a refusal line
    /// exactly like one with an open breaker, and the breaker and the
    /// liveness veto log their edges already. Re-asserting the verdict the
    /// worker carries writes nothing; the counters and the gauge move once
    /// per edge, in the registry.
    fn apply_overload_verdict(
        &self,
        worker: &Arc<dyn Worker>,
        load: Option<&WorkerLoadResponse>,
        source: &'static str,
    ) {
        let overload = worker.metadata().overload;
        if !overload.is_enabled() {
            return;
        }
        let verdict = load.is_some_and(|load| overload.is_overloaded(load));
        if !self.worker_registry.set_worker_overloaded(worker, verdict) {
            return;
        }
        let waiting = load.map_or(0, WorkerLoadResponse::total_waiting_reqs);
        let kv_pressure = load.map_or(0.0, WorkerLoadResponse::effective_kv_pressure);
        info!(
            worker_url = %worker.redacted_url(),
            overloaded = verdict,
            waiting_requests = waiting,
            kv_pressure,
            waiting_threshold = ?overload.waiting_requests,
            kv_pressure_threshold = ?overload.token_usage,
            source,
            "Worker overload veto {}",
            if verdict { "set" } else { "cleared" }
        );
    }

    /// A load record pushed on a worker's KV-event stream (`KvEventBatch.load`,
    /// the servicer's `GetLoads` figures for `dp_rank`, received at
    /// `received_at`): the same input as a poll of that worker. It replaces
    /// the rank's entry in the worker's report, goes to the overload verdict,
    /// the wedged rule and the `smg_engine_*` gauges at once, and into the
    /// shared snapshot, which routing reads, in the next coalesced publish.
    /// The report's `sampled_at` is the receipt less the record's age and
    /// the one-way margin, what a policy releases its dispatch credit against.
    /// Negative ranks and records for another virtual worker's rank are
    /// ignored before updating freshness, liveness, metrics or routing state.
    pub fn apply_pushed_load(
        self: &Arc<Self>,
        worker: &Arc<dyn Worker>,
        dp_rank: i32,
        record: &EngineLoad,
        received_at: Instant,
    ) {
        if dp_rank < 0
            || worker
                .dp_rank()
                .is_some_and(|rank| usize::try_from(dp_rank) != Ok(rank))
        {
            return;
        }
        let age = Duration::from_millis(u64::from(record.age_ms)) + PUSHED_ONE_WAY_MARGIN;
        let sampled_at = received_at.checked_sub(age).unwrap_or(received_at);
        let url = worker.url().to_string();
        // The stream pushes: silence from this worker is judged by the
        // stream's cadence from now on (`liveness`, the stale rule).
        worker.note_load_record();
        // The poll decision reads the receipt, not the sample's age: a record
        // that arrives is a stream that works, whatever the engine's clock.
        let mut pushed = self.pushed_ranks.lock();
        self.pushed_at
            .lock()
            .entry(url.clone())
            .or_default()
            .insert(dp_rank, received_at);
        let response = {
            let ranks = pushed.entry(url.clone()).or_default();
            let previous = ranks.get(&dp_rank).cloned().unwrap_or_default();
            // What a poll would have put in the rank's entry, with the
            // telemetry the record left out (an event batch's record is the
            // core only) kept from the last record that carried it.
            let mut fresh = SchedulerLoadSnapshot::from(record);
            fresh.dp_rank = dp_rank;
            keep_telemetry(&mut fresh, &previous, record);
            ranks.insert(dp_rank, fresh);
            Arc::new(WorkerLoadResponse {
                timestamp: chrono::Utc::now().to_rfc3339(),
                version: PUSHED_LOAD_VERSION.to_string(),
                dp_rank_count: i32::try_from(ranks.len()).unwrap_or(i32::MAX),
                loads: ranks.values().cloned().collect(),
                aggregate: None,
                sampled_at: Some(sampled_at),
            })
        };
        // What a poll does with a fresh report, minus the DP-rank token cache
        // (a record carries no absolute token counts).
        self.apply_overload_verdict(worker, Some(&response), "pushed record");
        let waiting: i64 = response
            .loads
            .iter()
            .map(|rank| i64::from(rank.num_waiting_reqs))
            .sum();
        liveness::on_load_report(
            worker,
            waiting,
            record
                .engine_silence_ms
                .map(|ms| Duration::from_millis(u64::from(ms))),
            record.engine_reports_steps.unwrap_or(false),
            record.prefill_pending_tokens.map(u64::from),
        );
        Metrics::record_engine_load(&url, worker.model_id(), &response);
        // The DP-rank token cache takes a report with absolute token counts,
        // as it takes a poll; a core-only record leaves its last-known-good
        // entry alone.
        if response.has_absolute_token_data() && response.ranks_are_dp_ranks() {
            self.worker_load_manager
                .update_dp_loads(&HashMap::from([(url.clone(), response.dp_rank_loads())]));
        }
        self.pending_pushed
            .lock()
            .insert(url, (Arc::clone(worker), response));
        drop(pushed);
        self.start_pushed_flusher();
        self.pushed_notify.notify_one();
    }

    /// The poll decision for `url` at `now` from the receipt times of its
    /// pushed records: none on file polls, a record within `interval` on
    /// every pushing rank suppresses the poll, an older one on any rank
    /// falls back to it.
    pub(crate) fn poll_mode(&self, url: &str, now: Instant, interval: Duration) -> PollMode {
        let pushed = self.pushed_at.lock();
        match pushed.get(url) {
            Some(ranks) if !ranks.is_empty() => {
                let fresh = ranks
                    .values()
                    .all(|&at| now.saturating_duration_since(at) < interval);
                if fresh {
                    PollMode::Suppressed
                } else {
                    PollMode::Fallback
                }
            }
            _ => PollMode::Poll,
        }
    }

    /// A poll's report of `url` (`load`, asked at `polled_at`) is the newer
    /// figure for every rank that pushed no record since the poll began.
    /// Under data parallelism the ranks' streams are not equally busy: a
    /// rank whose working set is cached, or that has nothing in flight,
    /// stores no blocks, and the heartbeats that keep the worker's stream
    /// alive name the last active rank, so a quiet rank's own record stops
    /// arriving while the worker's report is rebuilt on every push from
    /// the record it last sent. The tick's decision ([`Self::poll_mode`])
    /// falls back to the poll for such a worker; the poll's entry for the
    /// quiet rank replaces the one on file, so the next push rebuilds the
    /// report from it and the rank's figures are at most a tick interval
    /// old, the poll decision's own bound. A rank whose record arrived
    /// since the poll began keeps it: the stream's is the newer. The
    /// receipts (`pushed_at`) are not touched: a poll is not the stream
    /// working, and the worker stays polled until the rank's records
    /// resume. Only a rank on file is replaced: a rank the stream never
    /// named has no receipt for the poll decision to age, so a poll's
    /// figure for it would sit in the report unrefreshed once the pushing
    /// ranks are fresh again; it stays out of the pushed report, and the
    /// poll's report stands for it while the worker is polled. A worker
    /// with no pushed record on file has no report to merge into; the
    /// poll's report stands on its own, as always.
    pub(crate) fn apply_polled_load(
        &self,
        url: &str,
        load: &WorkerLoadResponse,
        polled_at: Instant,
    ) {
        // A rollup from a gateway registered as a worker is keyed by
        // downstream worker, not by rank: its entries all say rank 0 and
        // would collapse into one slot (the DP cache applies the same
        // guard).
        if !load.ranks_are_dp_ranks() {
            return;
        }
        // Both tables held together so a record landing between the two
        // reads is not undone; pushes and retractions use the same lock order.
        let mut pushed = self.pushed_ranks.lock();
        let Some(ranks) = pushed.get_mut(url) else {
            return;
        };
        let pushed_at = self.pushed_at.lock();
        let receipts = pushed_at.get(url);
        for entry in &load.loads {
            let pushed_since = receipts
                .and_then(|ranks| ranks.get(&entry.dp_rank))
                .is_some_and(|&at| at >= polled_at);
            if pushed_since {
                continue;
            }
            if let Some(slot) = ranks.get_mut(&entry.dp_rank) {
                *slot = entry.clone();
            }
        }
    }

    /// Retract pre-poll pushed ranks after an authoritative empty response.
    /// Records received since the poll began remain available for the next push.
    fn retract_pushed_load(&self, url: &str, rank: Option<usize>, polled_at: Instant) {
        let mut pushed = self.pushed_ranks.lock();
        let mut receipts = self.pushed_at.lock();
        let Some(times) = receipts.get_mut(url) else {
            return;
        };
        times.retain(|&dp_rank, at| {
            rank.is_some_and(|rank| usize::try_from(dp_rank) != Ok(rank)) || *at >= polled_at
        });
        if let Some(ranks) = pushed.get_mut(url) {
            ranks.retain(|dp_rank, _| times.contains_key(dp_rank));
        }
        let mut pending = self.pending_pushed.lock();
        if let Some((_, response)) = pending.get_mut(url) {
            let response = Arc::make_mut(response);
            response
                .loads
                .retain(|load| times.contains_key(&load.dp_rank));
            response.dp_rank_count = i32::try_from(response.loads.len()).unwrap_or(i32::MAX);
            response.aggregate = None;
            if response.loads.is_empty() {
                pending.remove(url);
            }
        }
    }

    /// The coalescing publisher of pushed reports, started on the first one:
    /// every window it moves what accumulated into the shared snapshot in one
    /// rebuild. Holds the monitor weakly, like the eviction flusher.
    fn start_pushed_flusher(self: &Arc<Self>) {
        let mut task = self.pushed_flush_task.lock();
        if task.as_ref().is_some_and(|task| !task.is_finished()) {
            return;
        }
        let monitor = Arc::downgrade(self);
        let notify = Arc::clone(&self.pushed_notify);
        #[expect(
            clippy::disallowed_methods,
            reason = "ends when the monitor is dropped; nothing waits on it at shutdown"
        )]
        let handle = tokio::spawn(async move {
            loop {
                notify.notified().await;
                tokio::time::sleep(PUSHED_PUBLISH_WINDOW).await;
                let Some(monitor) = monitor.upgrade() else {
                    return;
                };
                monitor.flush_pushed();
            }
        });
        *task = Some(handle);
    }

    /// Publish every pushed report that accumulated since the last flush.
    pub(crate) fn flush_pushed(&self) {
        let _pushed = self.pushed_ranks.lock();
        let pending: Vec<PushedReport> = self
            .pending_pushed
            .lock()
            .drain()
            .map(|(_, entry)| entry)
            .collect();
        if pending.is_empty() {
            return;
        }
        let urls: Vec<String> = pending
            .iter()
            .map(|(worker, _)| worker.url().to_string())
            .collect();
        self.load_state.publish_group(&urls, pending);
    }

    /// Pushed reports waiting to be published (tests).
    #[cfg(test)]
    pub(crate) fn pending_pushed_len(&self) -> usize {
        self.pending_pushed.lock().len()
    }

    /// The current published load snapshot — what routing is acting on, and
    /// what `GET /loads` serves.
    pub(crate) fn load_snapshot(&self) -> Arc<LoadSnapshot> {
        self.load_state.snapshot()
    }

    /// Subscribe to the shared per-worker load snapshots.
    ///
    /// Each received value is an immutable `Arc<LoadSnapshot>`: clone the
    /// `Arc` out of `borrow()` and scan without holding the watch guard.
    /// Stale entries are pruned on each tick of the relevant group and by
    /// the batched eviction flusher.
    pub(crate) fn subscribe(&self) -> LoadReceiver {
        self.load_state.subscribe()
    }

    /// Subscribe to registry events, run a synchronous bootstrap
    /// reconcile, and spawn the background event task.
    ///
    /// Subscribing **before** the bootstrap reconcile guarantees the
    /// "registered between subscribe and reconcile" race resolves
    /// idempotently: any registration that lands after this call either
    /// is already in the snapshot (because it ran on this thread) or
    /// arrives as a `Registered` event that the loop applies on top of
    /// the snapshot.
    pub fn start_event_loop(self: &Arc<Self>) {
        // Capture the receiver before reconciling so events that fire
        // during reconcile are buffered, not lost.
        let events_rx = self.worker_registry.subscribe_events();

        // Synchronous bootstrap: build initial group set from the
        // registry snapshot and start polling loops.
        self.reconcile_from_registry();

        // Hand the spawned task a `Weak<Self>` so it does not pin the
        // monitor in memory across drops. If we passed an `Arc<Self>`
        // the spawned task would form a cycle: the monitor stores the
        // task's `JoinHandle`, and the task closure holds an `Arc`
        // back to the monitor. The strong count would never reach
        // zero, `Drop` would never run, and the abort calls would be
        // unreachable outside of full runtime shutdown.
        let monitor = Arc::downgrade(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "WorkerMonitor event loop runs for the lifetime of the registry; the JoinHandle is stored on the monitor and aborted in Drop"
        )]
        let handle = tokio::spawn(async move {
            run_event_loop(monitor, events_rx).await;
        });

        *self.event_task.lock() = Some(handle);

        // The debounced eviction flusher: holds the load state strongly (it
        // is cheap and monitor-independent) and the monitor weakly, for the
        // same cycle-breaking reason as the event loop.
        let load_state = Arc::clone(&self.load_state);
        let monitor = Arc::downgrade(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "eviction flusher runs for the monitor's lifetime; the JoinHandle is stored on the monitor and aborted in Drop"
        )]
        let flush_handle = tokio::spawn(async move {
            eviction_flush_loop(load_state, monitor).await;
        });

        *self.eviction_flush_task.lock() = Some(flush_handle);

        // The liveness sweep: vetoes a worker that stayed silent for the stall
        // threshold after a transport failure (see `worker::liveness::sweep`).
        let monitor = Arc::downgrade(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "liveness sweep runs for the monitor's lifetime; the JoinHandle is stored on the monitor and aborted in Drop"
        )]
        let sweep_handle = tokio::spawn(async move {
            liveness_sweep_loop(monitor).await;
        });
        *self.liveness_sweep_task.lock() = Some(sweep_handle);
    }

    /// Stop every per-group polling loop and clear the shared load
    /// snapshot + DP-rank cache.
    ///
    /// Called from `Drop`, the bootstrap reconcile path, and the
    /// `RecvError::Lagged` rebuild. Always clears both caches even
    /// when no groups were running, so a lag-recovery rebuild cannot
    /// inherit stale per-rank loads from a previous incarnation.
    /// Idempotent.
    pub fn stop_all_groups(&self) {
        let drained: Vec<(WorkerGroupKey, GroupState)> = {
            let mut handles = self.group_handles.lock();
            handles.drain().collect()
        };

        if !drained.is_empty() {
            info!("Stopping all {} load monitor groups", drained.len());
            for (key, state) in drained {
                debug!("Stopping load monitor group: {key}");
                state.handle.abort();
            }
        }

        // Always clear every cache. Skipping when `drained.is_empty()`
        // would leave stale per-rank loads behind for any caller that
        // seeded the cache without going through a group loop, and
        // makes the function harder to reason about as a "reset".
        //
        // The probe memo is dropped with the rest rather than retained
        // per live URL: workers removed during the lag window would
        // otherwise leak entries, and the cost is one re-probe per
        // worker on a path that only runs on lag recovery.
        self.load_state.clear();
        self.worker_load_manager.clear();
        self.native_loads_memo.clear();
        // The feed that would clear the vetoes is being torn down: fail open,
        // but say so — a wiped gauge is otherwise indistinguishable from a
        // genuine recovery. With no thresholds anywhere no flag was ever
        // written, so this clears nothing and stays silent.
        let cleared = self.worker_registry.clear_all_overload_flags();
        if cleared > 0 {
            warn!(
                vetoes_cleared = cleared,
                "Overload vetoes cleared with the load feed; every worker is routable again \
                 until its next poll"
            );
        }
    }

    /// Recompute the polling state for every currently-known group.
    ///
    /// Used as the synchronous bootstrap path and as the lag-recovery
    /// rebuild after `RecvError::Lagged`. Stops every existing loop,
    /// clears the cached snapshot, then walks the registry to start
    /// fresh loops for each non-empty group.
    fn reconcile_from_registry(self: &Arc<Self>) {
        // Stop everything first so the rebuild starts from a clean slate.
        self.stop_all_groups();

        // Walk every worker once and bucket them by group.
        let workers = self.worker_registry.get_all();
        let mut group_keys: HashMap<WorkerGroupKey, ()> = HashMap::new();
        for worker in workers {
            for key in group_keys_for_worker(&worker) {
                group_keys.insert(key, ());
            }
        }

        for key in group_keys.into_keys() {
            self.reconcile_group(&key);
        }
    }

    /// Bring a single group's polling state into sync with the registry.
    ///
    /// - Empty group: stop the loop (if any), evict the group's cached
    ///   loads from the watch channel and DP cache.
    /// - Non-empty group, no current loop: spawn one with the group's
    ///   desired interval.
    /// - Non-empty group, loop exists with a stale interval: stop and
    ///   restart so the new interval takes effect.
    /// - Non-empty group, loop exists with the correct interval: no-op.
    fn reconcile_group(self: &Arc<Self>, key: &WorkerGroupKey) {
        let workers = self.worker_registry.get_workers_filtered(
            Some(&key.model_id),
            Some(key.worker_type),
            Some(key.connection_mode),
            None,
            false,
        );

        if workers.is_empty() {
            self.stop_group(key);
            return;
        }

        let desired_interval = group_interval(&workers, self.default_interval);

        let needs_start = {
            let mut handles = self.group_handles.lock();
            match handles.get(key) {
                Some(state) if state.interval == desired_interval => false,
                Some(_) => {
                    // Interval changed — stop the old loop and fall
                    // through to start a fresh one below.
                    if let Some(old) = handles.remove(key) {
                        debug!("Restarting load monitor group {key} with new interval {desired_interval:?}");
                        old.handle.abort();
                    }
                    true
                }
                None => true,
            }
        };

        if needs_start {
            self.spawn_group_loop(key.clone(), desired_interval);
        }
    }

    /// Stop a single group's polling loop.
    ///
    /// Does **not** evict per-worker cached loads from the watch
    /// channel or DP cache. Callers that know the affected URLs (e.g.
    /// the event loop on `Removed` / `Replaced` / `StatusChanged`)
    /// must invoke [`Self::evict_worker_loads`] separately. The
    /// per-group polling loop itself prunes URLs on the next tick when
    /// the worker is gone, but a stopped loop never gets a next tick.
    fn stop_group(&self, key: &WorkerGroupKey) {
        let removed = {
            let mut handles = self.group_handles.lock();
            handles.remove(key)
        };

        if let Some(state) = removed {
            info!("Stopping load monitor for empty group {key}");
            state.handle.abort();
        }
    }

    /// Evict a single worker's cached loads. Used by the event loop on
    /// `Removed`, `Replaced`, and `StatusChanged` away from `Ready`.
    ///
    /// The overload flag, DP cache, and probe memo are cleared synchronously;
    /// the shared snapshot removal is queued so a removal wave costs one
    /// batched rebuild instead of one per worker (see
    /// [`Self::flush_pending_evictions`]).
    fn evict_worker_loads(&self, worker: &Arc<dyn Worker>) {
        let url = worker.url();
        // No load feed, no verdict: a flag left set would strand the worker
        // out of routing with nothing to ever clear it.
        self.worker_registry.set_worker_overloaded(worker, false);
        self.worker_load_manager.remove_worker(url);
        self.native_loads_memo.remove(url);
        // The pushed-record state goes with the worker: a replacement is
        // polled until its own stream pushes.
        self.pushed_ranks.lock().remove(url);
        self.pushed_at.lock().remove(url);
        self.load_state.enqueue_eviction(Arc::clone(worker));
    }

    /// Apply every queued eviction in one snapshot rebuild, and sentinel the
    /// `smg_engine_*` series of workers whose own entry was removed —
    /// metrics-rs cannot delete series. A same-URL replacement that already
    /// republished keeps its entry, and its series continues uninterrupted.
    pub(crate) fn flush_pending_evictions(&self) {
        for worker in self.load_state.apply_pending_evictions() {
            // A worker can serve multiple models (one load group per model),
            // so sentinel every model's series — not just the primary.
            let dp_size = worker.dp_size().unwrap_or(1);
            for model_id in WorkerRegistry::worker_model_ids(&worker) {
                Metrics::remove_engine_load_metrics(worker.url(), &model_id, dp_size);
            }
        }
    }

    /// Spawn the polling loop for a single group.
    fn spawn_group_loop(self: &Arc<Self>, key: WorkerGroupKey, interval: Duration) {
        info!("Starting load monitor for group {key} with interval {interval:?}");

        // Same `Weak<Self>` rationale as `start_event_loop`: the
        // group loop is owned by `group_handles` on the monitor, so
        // it must not own a strong reference back to the monitor.
        let monitor = Arc::downgrade(self);
        let group_key = key.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "Group polling loop runs for the lifetime of the group; the JoinHandle is stored in group_handles and aborted on group removal or monitor drop"
        )]
        let handle = tokio::spawn(async move {
            group_monitor_loop(monitor, group_key, interval).await;
        });

        let mut handles = self.group_handles.lock();
        handles.insert(key, GroupState { handle, interval });
    }

    /// Fetch load over HTTP, preferring a native load endpoint over the
    /// runtime's Prometheus gauges.
    ///
    /// The native schema reports the same scheduler state the `/metrics` arms
    /// reconstruct from gauges, in two orders of magnitude fewer bytes, and
    /// carries fields — queued tokens, generation throughput, the disagg
    /// section — that no gauge exposes. Preferring it is both cheaper and
    /// more informative wherever it exists.
    ///
    /// Two routes serve that schema: `/loads` on an SMG gateway registered as
    /// a worker, and `/v1/loads` on an engine. Which one a worker answered on
    /// is memoized, so the extra attempt costs one 404 per worker, once.
    ///
    /// Engine responses are scoped to the virtual worker's DP rank, if any.
    /// Gateway fleet rollups retain their worker annotations. An empty response
    /// means a successful poll omitted this rank; `None` means a failed poll.
    ///
    /// `native_loads_memo` is the shared probe memo, or `None` for callers
    /// with no monitor to borrow it from (they simply always discover).
    pub(crate) async fn fetch_http_load(
        worker: &Arc<dyn Worker>,
        native_loads_memo: Option<&DashMap<String, NativeLoadsMemo>>,
    ) -> Option<WorkerLoadResponse> {
        if let Some((path, load)) = Self::fetch_native_loads(worker, native_loads_memo).await {
            liveness::on_contact(worker);
            return Some(Self::scoped_native_load(path, load, worker.dp_rank()));
        }
        Self::fetch_http_gauges(worker).await
    }

    /// The whole answer of the first native load route that has one (an
    /// engine's every rank, or a gateway's fleet rollup) with the route it
    /// came from, before it is scoped to a worker. `None` is a failed poll.
    async fn fetch_native_loads(
        worker: &Arc<dyn Worker>,
        native_loads_memo: Option<&DashMap<String, NativeLoadsMemo>>,
    ) -> Option<(NativeLoadsPath, WorkerLoadResponse)> {
        // Only workers already `Ready` are polled, so a definitive "no such
        // endpoint" cannot be a warm-up artifact and is safe to memoize.
        let known = native_loads_memo
            .and_then(|store| store.get(worker.url()).map(|hit| *hit))
            .unwrap_or_default();
        let mut memo = known;

        let mut response = None;
        for path in known.candidates() {
            match Self::probe_native_loads(worker, path).await {
                NativeLoads::Available(load) => {
                    memo.record_answered(path);
                    response = Some((path, load));
                    break;
                }
                NativeLoads::Absent => memo.record_absent(path),
                NativeLoads::Inconclusive => {}
            }
        }
        // Write back only what a probe actually settled, so a tick that
        // learned nothing leaves the memo alone and retries next time.
        if let Some(store) = native_loads_memo {
            if memo != known {
                store.insert(worker.url().to_string(), memo);
            }
        }
        response
    }

    /// A native answer as one worker's report. An engine's is scoped to the
    /// worker's DP rank, if any: a missing rank is unavailable, not a reason
    /// to substitute the backend-wide Prometheus average. A gateway's fleet
    /// rollup keeps its worker annotations.
    fn scoped_native_load(
        path: NativeLoadsPath,
        load: WorkerLoadResponse,
        dp_rank: Option<usize>,
    ) -> WorkerLoadResponse {
        match path {
            NativeLoadsPath::Engine => load.for_dp_rank(dp_rank).unwrap_or_default(),
            NativeLoadsPath::Gateway => load,
        }
    }

    /// The backend-wide gauges of a worker with no native load route.
    async fn fetch_http_gauges(worker: &Arc<dyn Worker>) -> Option<WorkerLoadResponse> {
        // Backend-wide gauges cannot be attributed to a virtual DP worker.
        if worker.dp_rank().is_some() {
            return None;
        }

        match worker.metadata().spec.runtime_type {
            RuntimeType::Vllm => Self::fetch_http_load_vllm(worker).await,
            RuntimeType::Sglang => Self::fetch_http_load_sglang(worker).await,
            // Custom engines expose no gauge schema we can parse; the native
            // routes above were their only path.
            _ => None,
        }
    }

    /// Probe one native load route.
    ///
    /// Distinguishing [`NativeLoads::Absent`] from [`NativeLoads::Inconclusive`]
    /// is what makes the memo safe. Collapsing both into "unsupported" would
    /// let one timeout demote a healthy backend to the expensive `/metrics`
    /// path permanently.
    async fn probe_native_loads(worker: &Arc<dyn Worker>, path: NativeLoadsPath) -> NativeLoads {
        let url = path.probe_url(worker);
        let resp = match Self::authed_request(worker, &url).send().await {
            Ok(resp) => resp,
            // Transport error or timeout: says nothing about the route.
            Err(_) => return NativeLoads::Inconclusive,
        };

        let status = resp.status();
        if matches!(
            status,
            StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED | StatusCode::NOT_IMPLEMENTED
        ) {
            return NativeLoads::Absent;
        }
        if !status.is_success() {
            // 5xx and auth failures are conditions, not routing facts.
            return NativeLoads::Inconclusive;
        }

        // WorkerLoadResponse defaults missing fields for compatibility. Native
        // route discovery must require an explicit loads array so an unrelated
        // JSON response is not mistaken for an authoritative empty snapshot.
        #[derive(serde::Deserialize)]
        struct NativeResponse {
            loads: Vec<SchedulerLoadSnapshot>,
            #[serde(flatten)]
            response: WorkerLoadResponse,
        }

        match resp.json::<NativeResponse>().await {
            // An empty native snapshot is still authoritative: no rank has
            // telemetry yet. Do not turn it into an aggregate fallback.
            Ok(NativeResponse {
                loads,
                mut response,
            }) => {
                response.loads = loads;
                NativeLoads::Available(response)
            }
            // A 200 that is not a load response means something else is
            // mounted here; that is as definitive as a 404.
            Err(_) => NativeLoads::Absent,
        }
    }

    /// vLLM HTTP: derive load from the Prometheus `/metrics` endpoint.
    /// The KV-cache usage ratio (0.0–1.0) maps onto `token_usage`; it is
    /// exposed as `vllm:gpu_cache_usage_perc` in vLLM v0 and renamed to
    /// `vllm:kv_cache_usage_perc` in vLLM v1, so accept either.
    async fn fetch_http_load_vllm(worker: &Arc<dyn Worker>) -> Option<WorkerLoadResponse> {
        let url = format!("{}/metrics", worker.base_url());
        let body = Self::authed_get(worker, &url).await?.text().await.ok()?;
        let m = PromScrape::parse(&body);

        // Require the KV-usage gauge: it is the signal load-aware routing acts
        // on, and without it `token_usage` would default to `0.0` and make a
        // worker of unknown pressure look idle. v0 and v1 name it differently.
        let kv_usage = ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"]
            .into_iter()
            .find(|name| m.has(name))?;

        Some(Self::single_rank(SchedulerLoadSnapshot {
            num_running_reqs: m.sum("vllm:num_requests_running") as i32,
            num_waiting_reqs: m.sum("vllm:num_requests_waiting") as i32,
            token_usage: m.mean(kv_usage),
            cache_hit_rate: m.mean("vllm:gpu_prefix_cache_hit_rate"),
            ..Default::default()
        }))
    }

    /// SGLang HTTP: try the custom `/v1/loads` endpoint first (some builds
    /// serve it), then fall back to the Prometheus `/metrics` gauges. The
    /// KV-usage ratio (0.0–1.0) is `<prefix>token_usage`, where SGLang used
    /// the `sglang:` metric prefix through v0.5.3 and switched to `sglang_`
    /// in v0.5.4+, so detect whichever is present and use it throughout.
    async fn fetch_http_load_sglang(worker: &Arc<dyn Worker>) -> Option<WorkerLoadResponse> {
        let url = format!("{}/metrics", worker.base_url());
        let body = Self::authed_get(worker, &url).await?.text().await.ok()?;
        let m = PromScrape::parse(&body);

        // Require the KV-usage gauge — the load signal routing acts on.
        // Without it `token_usage` would default to 0.0 and hide real KV
        // pressure, so report unavailable instead. Pick the prefix off it.
        let prefix = ["sglang:", "sglang_"]
            .into_iter()
            .find(|p| m.has(&format!("{p}token_usage")))?;

        Some(Self::single_rank(SchedulerLoadSnapshot {
            num_running_reqs: m.sum(&format!("{prefix}num_running_reqs")) as i32,
            num_waiting_reqs: m.sum(&format!("{prefix}num_queue_reqs")) as i32,
            token_usage: m.mean(&format!("{prefix}token_usage")),
            gen_throughput: m.mean(&format!("{prefix}gen_throughput")),
            cache_hit_rate: m.mean(&format!("{prefix}cache_hit_rate")),
            utilization: m.mean(&format!("{prefix}utilization")),
            ..Default::default()
        }))
    }

    /// Shared authenticated GET builder with the standard timeout.
    fn authed_request(worker: &Arc<dyn Worker>, url: &str) -> reqwest::RequestBuilder {
        let req = worker.http_client().get(url).timeout(REQUEST_TIMEOUT);
        match worker.api_key() {
            Some(key) => req.bearer_auth(key),
            None => req,
        }
    }

    /// Shared authenticated GET with the standard timeout. Returns `None` on
    /// transport error or non-success status.
    async fn authed_get(worker: &Arc<dyn Worker>, url: &str) -> Option<reqwest::Response> {
        match Self::authed_request(worker, url).send().await {
            Ok(r) if r.status().is_success() => Some(r),
            _ => None,
        }
    }

    /// Wrap a single scheduler snapshot as a one-rank `WorkerLoadResponse`.
    /// HTTP `/metrics` already aggregates across DP ranks, so a single
    /// synthetic rank is the correct shape.
    ///
    /// Such a snapshot carries the KV-usage ratio (`token_usage`) but no
    /// absolute token counts (`max_total_num_tokens`/`num_used_tokens` stay
    /// `0`), so `WorkerLoadResponse::has_absolute_token_data` reports `false`
    /// for it — keeping it out of the DP-rank cache.
    fn single_rank(snapshot: SchedulerLoadSnapshot) -> WorkerLoadResponse {
        WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![snapshot],
            ..Default::default()
        }
    }

    /// Fetch load via the backend client's `GetLoads` (gRPC RPC or the ZMQ
    /// engine's pushed scheduler stats) for the workers of one engine: one
    /// worker, or the rank workers of a data-parallel engine under
    /// `--dp-aware` (one address, one answer naming every rank). The first
    /// member asks; the outcome is every member's, as each one's own poll
    /// would have had it: the whole answer (the caller scopes it to each
    /// member's rank), the contact that ends an unreachable veto, a
    /// transport failure. Returns `None` on no members, a missing client or
    /// an error.
    pub(crate) async fn fetch_backend_load(
        members: &[Arc<dyn Worker>],
    ) -> Option<WorkerLoadResponse> {
        let worker = members.first()?;
        let backend_client = match worker.get_backend_client().await {
            Ok(Some(client)) => client,
            Ok(None) => {
                debug!("No backend client for worker {}", worker.redacted_url());
                return None;
            }
            Err(e) => {
                debug!(
                    "Failed to get backend client for {}: {e}",
                    worker.redacted_url()
                );
                return None;
            }
        };

        // A deadline, so one half-open connection cannot stall the whole
        // group's poll (the group awaits every worker's poll together). A
        // poll that times out is a slow worker, not a dead one: the keepalive
        // on the connection is what reports those.
        match tokio::time::timeout(LOAD_POLL_DEADLINE, backend_client.get_loads()).await {
            Ok(Ok(load)) => {
                for member in members {
                    liveness::on_contact(member);
                }
                Some(load)
            }
            Ok(Err(e)) => {
                debug!("backend GetLoads failed for {}: {e}", worker.redacted_url());
                if liveness::is_connection_code(e.code()) {
                    for member in members {
                        liveness::on_contact_failed(member, "load poll");
                    }
                }
                None
            }
            Err(_) => {
                debug!(
                    "backend GetLoads for {} did not answer within {LOAD_POLL_DEADLINE:?}",
                    worker.redacted_url()
                );
                None
            }
        }
    }

    /// One tick's fetch for the polled workers of one engine: one worker, or
    /// the rank workers of a data-parallel engine under `--dp-aware`, which
    /// share it. The engine answers once; each member's report is that
    /// answer scoped to its own rank, as its own poll would have had it (a
    /// rank the answer omits is an empty report).
    async fn fetch_engine_loads(
        members: Vec<Arc<dyn Worker>>,
        connection_mode: ConnectionMode,
        native_loads_memo: &DashMap<String, NativeLoadsMemo>,
    ) -> Vec<(Arc<dyn Worker>, Option<WorkerLoadResponse>)> {
        let mut reports = Vec::with_capacity(members.len());
        match connection_mode {
            ConnectionMode::Http => {
                if let [worker] = members.as_slice() {
                    reports.push(Self::fetch_http_load(worker, Some(native_loads_memo)).await);
                } else {
                    match Self::fetch_native_loads(&members[0], Some(native_loads_memo)).await {
                        Some((path, load)) => {
                            for member in &members {
                                liveness::on_contact(member);
                                reports.push(Some(Self::scoped_native_load(
                                    path,
                                    load.clone(),
                                    member.dp_rank(),
                                )));
                            }
                        }
                        None => {
                            for member in &members {
                                reports.push(Self::fetch_http_gauges(member).await);
                            }
                        }
                    }
                }
            }
            ConnectionMode::Grpc | ConnectionMode::Zmq => {
                let answer = Self::fetch_backend_load(&members).await;
                for member in &members {
                    reports.push(
                        answer
                            .clone()
                            .map(|load| load.for_dp_rank(member.dp_rank()).unwrap_or_default()),
                    );
                }
            }
        }
        members.into_iter().zip(reports).collect()
    }
}

impl Drop for WorkerMonitor {
    fn drop(&mut self) {
        if let Some(handle) = self.event_task.get_mut().take() {
            handle.abort();
        }
        if let Some(handle) = self.eviction_flush_task.get_mut().take() {
            handle.abort();
        }
        if let Some(handle) = self.liveness_sweep_task.get_mut().take() {
            handle.abort();
        }
        for (_, state) in self.group_handles.get_mut().drain() {
            state.handle.abort();
        }
    }
}

/// Debounced eviction flusher: waits for the first queued eviction, lets a
/// gradual removal stream accumulate for [`EVICTION_DEBOUNCE`], then applies
/// the whole backlog in one snapshot rebuild.
///
/// Holds the monitor weakly (same cycle-breaking rationale as the other
/// loops); holding the `LoadState` strongly is fine — it owns no task
/// handles back into the monitor.
async fn eviction_flush_loop(load_state: Arc<LoadState>, monitor: Weak<WorkerMonitor>) {
    loop {
        load_state.eviction_wakeup().await;
        tokio::time::sleep(EVICTION_DEBOUNCE).await;
        let Some(monitor) = monitor.upgrade() else {
            debug!("WorkerMonitor was dropped; exiting eviction flush loop");
            return;
        };
        monitor.flush_pending_evictions();
        drop(monitor);
    }
}

/// The polled workers as the fetches of one tick: one per worker, except the
/// rank workers of one data-parallel engine, which share one. Under
/// `--dp-aware` an engine of N ranks is N workers at one address, and each
/// one's poll asked the engine for the same answer (N `GetLoads`, or N reads
/// of the native load route, per engine per tick) and kept its own rank of
/// it. One member asks, the lowest rank, so the native-route memo of an
/// engine is one worker's tick after tick; the answer is scoped to each.
fn polled_engines(polled: &[Arc<dyn Worker>]) -> Vec<Vec<Arc<dyn Worker>>> {
    let mut engines: Vec<Vec<Arc<dyn Worker>>> = Vec::with_capacity(polled.len());
    let mut by_address: HashMap<&str, usize> = HashMap::new();
    for worker in polled {
        if worker.dp_rank().is_some() {
            if let Some(&engine) = by_address.get(worker.base_url()) {
                engines[engine].push(Arc::clone(worker));
                continue;
            }
            by_address.insert(worker.base_url(), engines.len());
        }
        engines.push(vec![Arc::clone(worker)]);
    }
    for members in &mut engines {
        members.sort_by_key(|worker| worker.dp_rank());
    }
    engines
}

/// Compute the set of `WorkerGroupKey`s a worker participates in.
///
/// A worker can serve multiple models (multimodel deployments), so it
/// can belong to multiple groups simultaneously. Each `(model, type,
/// connection)` triple is one group.
fn group_keys_for_worker(worker: &Arc<dyn Worker>) -> Vec<WorkerGroupKey> {
    WorkerRegistry::worker_model_ids(worker)
        .into_iter()
        .map(|model_id| WorkerGroupKey {
            model_id,
            worker_type: *worker.worker_type(),
            connection_mode: *worker.connection_mode(),
        })
        .collect()
}

/// Compute the polling interval for a group from per-worker overrides.
///
/// Uses the smallest `load_monitor_interval_secs` across the group so
/// the fastest worker's polling cadence wins. Falls back to
/// `default_interval` when no worker sets an override. Always floored
/// to one second to prevent tight-loop DoS.
fn group_interval(workers: &[Arc<dyn Worker>], default_interval: Duration) -> Duration {
    let override_secs = workers
        .iter()
        .filter_map(|w| w.metadata().spec.load_monitor_interval_secs)
        .min();

    let interval = override_secs
        .map(|s| Duration::from_secs(s.max(1)))
        .unwrap_or(default_interval);
    interval.max(Duration::from_secs(1))
}

/// Background event handler. Lives as long as the `WorkerMonitor`.
///
/// Holds a `Weak<WorkerMonitor>` rather than an `Arc<WorkerMonitor>`
/// so the spawned task does not pin the monitor in memory after every
/// strong reference has been dropped. Each iteration upgrades the
/// `Weak` only for the duration of the work and drops the temporary
/// `Arc` before the next `recv().await`, so the monitor's `Drop` can
/// fire as soon as the owning `AppContext` goes away.
async fn run_event_loop(
    monitor: Weak<WorkerMonitor>,
    mut events_rx: broadcast::Receiver<WorkerEvent>,
) {
    loop {
        let event = events_rx.recv().await;
        let Some(monitor) = monitor.upgrade() else {
            debug!("WorkerMonitor was dropped; exiting event loop");
            return;
        };
        match event {
            Ok(WorkerEvent::Registered { worker, .. }) => {
                for key in group_keys_for_worker(&worker) {
                    monitor.reconcile_group(&key);
                }
            }
            Ok(WorkerEvent::Removed { worker, .. }) => {
                // Evict the worker from both caches BEFORE reconciling
                // the group. If this was the last worker in its group,
                // `reconcile_group` will stop the polling loop, and a
                // stopped loop cannot prune the stale entry on a later
                // tick — it would persist forever otherwise.
                monitor.evict_worker_loads(&worker);
                for key in group_keys_for_worker(&worker) {
                    monitor.reconcile_group(&key);
                }
            }
            Ok(WorkerEvent::Replaced { old, new, .. }) => {
                // Registry guarantees `old.url() == new.url()` (replace
                // rejects URL changes), but the model list may have
                // shrunk so some old groups disappear. Evict the URL
                // from caches before reconciling so the disappearing
                // groups do not leak the entry; the surviving / new
                // groups will repopulate it on their next poll tick.
                monitor.evict_worker_loads(&old);
                let mut keys = group_keys_for_worker(&old);
                keys.extend(group_keys_for_worker(&new));
                keys.sort_by(|a, b| {
                    a.model_id
                        .cmp(&b.model_id)
                        .then_with(|| (a.worker_type as u8).cmp(&(b.worker_type as u8)))
                        .then_with(|| (a.connection_mode as u8).cmp(&(b.connection_mode as u8)))
                });
                keys.dedup();
                for key in keys {
                    monitor.reconcile_group(&key);
                }
            }
            Ok(WorkerEvent::StatusChanged {
                worker, new_status, ..
            }) => {
                if new_status != WorkerStatus::Ready {
                    monitor.evict_worker_loads(&worker);
                }
                // No action needed when transitioning *into* Ready: the
                // group's polling loop reads from the registry on every
                // tick and will pick the worker up automatically.
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!(
                    skipped = n,
                    "WorkerMonitor lagged behind registry events; rebuilding from snapshot"
                );
                monitor.reconcile_from_registry();
            }
            Err(broadcast::error::RecvError::Closed) => {
                debug!("WorkerMonitor event channel closed; exiting event loop");
                return;
            }
        }
        // Drop the temporary strong reference so we do not keep the
        // monitor alive while parked on the next `recv().await`.
        drop(monitor);
    }
}

/// Polling loop body for a single worker group.
///
/// Holds a `Weak<WorkerMonitor>` for the same cycle-breaking reason
/// as `run_event_loop`. The temporary `Arc` is upgraded after the
/// timer tick and dropped before the next tick so the monitor's
/// `Drop` is reachable.
/// How long a backend load poll may take before the tick moves on without it.
const LOAD_POLL_DEADLINE: Duration = Duration::from_secs(3);

/// Run [`liveness::sweep`] over every registered worker every quarter second.
async fn liveness_sweep_loop(monitor: Weak<WorkerMonitor>) {
    let mut ticker = tokio::time::interval(liveness::SWEEP_INTERVAL);
    loop {
        ticker.tick().await;
        let Some(monitor) = monitor.upgrade() else {
            return;
        };
        for worker in monitor
            .worker_registry
            .get_workers_filtered(None, None, None, None, false)
        {
            liveness::sweep(&worker);
        }
    }
}

async fn group_monitor_loop(
    monitor: Weak<WorkerMonitor>,
    group_key: WorkerGroupKey,
    interval: Duration,
) {
    let mut interval_timer = tokio::time::interval(interval);

    loop {
        interval_timer.tick().await;

        let Some(monitor) = monitor.upgrade() else {
            debug!("WorkerMonitor was dropped; exiting group loop for {group_key}");
            return;
        };
        poll_group_once(&monitor, &group_key, interval, Instant::now()).await;
        // Drop the temporary strong reference so we do not keep the
        // monitor alive across the next `interval_timer.tick().await`.
        drop(monitor);
    }
}

/// One tick of a group at `now`: the poll decision per worker, the polls of
/// those not served by their stream, and the publish of what they returned.
async fn poll_group_once(
    monitor: &Arc<WorkerMonitor>,
    group_key: &WorkerGroupKey,
    interval: Duration,
    now: Instant,
) {
    // Only poll Ready workers — Pending/NotReady/Failed do not
    // serve traffic and should not contribute load samples.
    let workers: Vec<Arc<dyn Worker>> = monitor
        .worker_registry
        .get_workers_filtered(
            Some(&group_key.model_id),
            Some(group_key.worker_type),
            Some(group_key.connection_mode),
            None,
            false,
        )
        .into_iter()
        .filter(|w| w.status() == WorkerStatus::Ready)
        .collect();

    if workers.is_empty() {
        debug!("No Ready workers in group {group_key}, skipping");
        return;
    }

    // Polling is unconditional by default so registration alone gives a
    // worker live load state. `--disable-load-monitoring` restores the
    // conditional gate: a load-aware policy, engine-metrics re-export, or
    // overload protection on any group member still forces the poll —
    // never "never poll".
    if monitor.conditional_polling {
        let routing_needs_load = monitor.policy_registry.needs_backend_loads()
            || monitor.policy_registry.get_dp_rank_policy().is_some();
        let overload_needs_load = workers.iter().any(|w| w.metadata().overload.is_enabled());
        if !routing_needs_load && !monitor.engine_metrics && !overload_needs_load {
            debug!("Load monitoring disabled and nothing needs the data, skipping load fetch for group {group_key}");
            return;
        }
    }

    // A worker whose stream pushed a load record on every pushing rank
    // within the interval is served by that stream: its report, verdicts
    // and gauges are already current from `apply_pushed_load`, and nothing
    // here touches its entry. The others are polled: no record ever
    // (`poll`) or records gone quiet (`fallback`).
    let mut polled: Vec<Arc<dyn Worker>> = Vec::with_capacity(workers.len());
    for worker in &workers {
        let mode = monitor.poll_mode(worker.url(), now, interval);
        Metrics::record_engine_load_poll(worker.url(), mode.as_str());
        if mode != PollMode::Suppressed {
            polled.push(Arc::clone(worker));
        }
    }

    let results = if polled.is_empty() {
        debug!("Every worker in group {group_key} pushed its load within {interval:?}; no poll");
        Vec::new()
    } else {
        let futures: Vec<_> = polled_engines(&polled)
            .into_iter()
            .map(|members| {
                let native_loads_memo = Arc::clone(&monitor.native_loads_memo);
                let connection_mode = group_key.connection_mode;
                async move {
                    WorkerMonitor::fetch_engine_loads(members, connection_mode, &native_loads_memo)
                        .await
                }
            })
            .collect();
        future::join_all(futures)
            .await
            .into_iter()
            .flatten()
            .collect()
    };

    let mut group_loads: HashMap<String, WorkerLoadResponse> = HashMap::new();
    let mut group_dp_loads: HashMap<String, HashMap<isize, isize>> = HashMap::new();
    let mut dp_evict: Vec<String> = Vec::new();
    for (worker, response) in results {
        let url = worker.url().to_string();
        // The overload predicate runs exactly here, once per report, never
        // on a request path, against the worker's effective thresholds
        // (resolved at registration). A failed fetch means no fresh
        // signal, which clears the flag — absent means no opinion.
        monitor.apply_overload_verdict(
            &worker,
            response.as_ref().filter(|load| !load.loads.is_empty()),
            "poll",
        );
        if let Some(mut load) = response {
            // A successful native poll can retract a rank. A failed poll
            // leaves the last-known DP cache alone; an explicit omission must
            // not keep that rank selectable or be published as an idle load.
            if load.loads.is_empty() {
                monitor.retract_pushed_load(&url, worker.dp_rank(), now);
                if let Some(rank) = worker.dp_rank() {
                    monitor.worker_load_manager.remove_rank(&url, rank);
                } else {
                    monitor.worker_load_manager.remove_worker(&url);
                }
                continue;
            }
            load.sampled_at = Some(Instant::now());
            // A worker polled because a rank's records went quiet
            // (`fallback`) takes the poll's figures for the ranks that
            // pushed nothing since the tick began.
            monitor.apply_polled_load(&url, &load, now);
            // The wedged rule reads the queue depth off every report.
            let waiting: i64 = load
                .loads
                .iter()
                .map(|rank| i64::from(rank.num_waiting_reqs))
                .sum();
            liveness::on_load_report(&worker, waiting, None, false, None);
            // Only feed the DP-rank cache from responses that carry real
            // absolute per-rank token counts. Ratio-only snapshots,
            // which would otherwise poison with a fake `{0: 0}`
            // entry and collapse DP routing onto rank 0.
            //
            // A fleet rollup from a gateway worker is keyed by downstream
            // worker, not by rank, so its repeated `dp_rank: 0` entries
            // would overwrite each other down to a single bogus rank.
            if load.has_absolute_token_data() && load.ranks_are_dp_ranks() {
                group_dp_loads.insert(url.clone(), load.dp_rank_loads());
            } else {
                dp_evict.push(url.clone());
            }
            group_loads.insert(url, load);
        }
    }

    // The poll is also the reconciliation tick: each policy trims what it
    // booked on a worker to the requests the router still holds there,
    // the safety net behind the load guard's completion report. It runs
    // for every worker, polled or served by its stream, whether or not a
    // fetch succeeded; the live count is the router's.
    for worker in &workers {
        monitor.policy_registry.reconcile_in_flight(worker.as_ref());
    }

    if polled.is_empty() {
        return;
    }

    // The URL set the publish prunes: the polled workers only, so a worker
    // served by its stream keeps its entry (the pushed path owns it).
    // Without the empty-fetch prune, a group that starts timing out keeps
    // publishing its previous tick's loads forever — subscribers see a
    // stale snapshot indefinitely.
    let polled_urls: Vec<String> = polled.iter().map(|w| w.url().to_string()).collect();

    if group_loads.is_empty() {
        debug!("No loads fetched for group {group_key}, pruning stale entries");
        monitor.load_state.publish_group(&polled_urls, Vec::new());
        // The DP cache deliberately keeps last-known-good entries
        // so routing decisions still have a hint to fall back to
        // when the upstream is briefly unreachable.
        return;
    }

    debug!(
        "Fetched loads from {}/{} polled workers in group {group_key}",
        group_loads.len(),
        polled.len()
    );

    monitor.worker_load_manager.update_dp_loads(&group_dp_loads);

    if !dp_evict.is_empty() {
        monitor.worker_load_manager.remove_workers(&dp_evict);
    }

    // Every successful load poll is also the canonical observability
    // sample. Load-aware policies already require this poll; the explicit
    // engine-metrics option only forces polling when routing does not.
    // Reusing the response avoids a second Engine RPC. (A pushed record
    // feeds the same gauges from `apply_pushed_load`.)
    for (url, load) in &group_loads {
        Metrics::record_engine_load(url, &group_key.model_id, load);
    }

    // Merge into the shared snapshot in one rebuild: clear stale entries
    // for the polled URLs first, then insert the fresh loads — each
    // paired with the worker that produced it so the incarnation fence
    // can drop reports that raced a removal or replacement. Workers that
    // failed this tick get their stale entries pruned along with the
    // rest. The responses move (no deep clones): the metrics pass above
    // took its references.
    let worker_by_url: HashMap<&str, &Arc<dyn Worker>> =
        polled.iter().map(|w| (w.url(), w)).collect();
    let fresh: Vec<(Arc<dyn Worker>, Arc<WorkerLoadResponse>)> = group_loads
        .into_iter()
        .filter_map(|(url, load)| {
            worker_by_url
                .get(url.as_str())
                .map(|worker| (Arc::clone(worker), Arc::new(load)))
        })
        .collect();
    monitor.load_state.publish_group(&polled_urls, fresh);
}

#[cfg(test)]
mod worker_load_manager_tests {
    use super::*;
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    #[test]
    fn test_new_dp_load_manager_instance() {
        let dp_load_manager = WorkerLoadManager::new();
        let cached = dp_load_manager.dp_cached_loads.read();
        assert!(cached.is_empty());
    }

    #[test]
    fn test_update_dp_load() {
        let manager = WorkerLoadManager::new();
        let mut loads = HashMap::new();

        let mut worker1_load = HashMap::new();
        worker1_load.insert(0, 2);
        worker1_load.insert(1, 1);
        loads.insert("http://worker1:8080".to_string(), worker1_load);

        let mut worker2_load = HashMap::new();
        worker2_load.insert(0, 3);
        loads.insert("http://worker2:8080".to_string(), worker2_load);

        manager.update_dp_loads(&loads);

        let cached = manager.dp_cached_loads.read();
        assert_eq!(cached.len(), 2);

        let worker2_cache = cached.get("http://worker2:8080").unwrap();
        assert_eq!(worker2_cache.get(&0), Some(&3));
    }

    #[test]
    fn test_select_and_increment_lowest_dp_load_multiple() {
        let worker = BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test_key")
            .build();

        let manager = WorkerLoadManager::new();
        let mut loads = HashMap::new();
        let mut worker_load = HashMap::new();
        worker_load.insert(0, 10);
        worker_load.insert(1, 3);
        worker_load.insert(2, 7);
        loads.insert(worker.url().to_string(), worker_load);
        manager.update_dp_loads(&loads);

        let selected = manager.select_and_increment_lowest_dp_load(&worker, 4);

        assert_eq!(selected, Some(1));
        let cached = manager.dp_cached_loads.read();
        assert_eq!(*cached.get(worker.url()).unwrap().get(&1).unwrap(), 3 + 4);
    }

    #[test]
    fn test_select_and_increment_lowest_dp_load_none_worker() {
        let worker = BasicWorkerBuilder::new("http://nonexist:8080")
            .worker_type(WorkerType::Regular)
            .api_key("test")
            .build();

        let manager = WorkerLoadManager::new();
        let result = manager.select_and_increment_lowest_dp_load(&worker, 1);
        assert_eq!(result, None);
    }
}

#[cfg(test)]
mod worker_monitor_tests;

#[cfg(test)]
mod prom_scrape_tests {
    use super::*;

    // Trimmed sample mirroring the fields observed on a live vLLM v0.7.3
    // `/metrics` endpoint (dev ORD cluster).
    const VLLM_METRICS: &str = r#"
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="llama"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama"} 5.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="llama"} 0.75
vllm:generation_tokens_total{model_name="llama"} 123456.0
"#;

    const SGLANG_METRICS: &str = r#"
sglang:num_running_reqs{model="llama"} 2.0
sglang:num_queue_reqs{model="llama"} 4.0
sglang:token_usage{model="llama"} 0.42
sglang:utilization{model="llama"} 0.9
"#;

    #[test]
    fn parses_gauges_ignoring_comments_and_labels() {
        let m = PromScrape::parse(VLLM_METRICS);
        assert!(m.has("vllm:num_requests_running"));
        assert_eq!(m.sum("vllm:num_requests_running"), 3.0);
        assert_eq!(m.sum("vllm:num_requests_waiting"), 5.0);
        assert_eq!(m.mean("vllm:gpu_cache_usage_perc"), 0.75);
        assert!(!m.has("vllm:does_not_exist"));
        assert_eq!(m.sum("vllm:does_not_exist"), 0.0);
        assert_eq!(m.mean("vllm:does_not_exist"), 0.0);
    }

    #[test]
    fn sum_adds_ranks_mean_averages_ratios() {
        // Two DP-rank series for the same gauge name.
        let text = "g{dp=\"0\"} 10\ng{dp=\"1\"} 30\nr{dp=\"0\"} 0.2\nr{dp=\"1\"} 0.8\n";
        let m = PromScrape::parse(text);
        assert_eq!(m.sum("g"), 40.0); // counts add
        assert_eq!(m.mean("r"), 0.5); // ratios average
    }

    #[test]
    fn vllm_metrics_map_onto_token_usage_snapshot() {
        let m = PromScrape::parse(VLLM_METRICS);
        let snap = SchedulerLoadSnapshot {
            num_running_reqs: m.sum("vllm:num_requests_running") as i32,
            num_waiting_reqs: m.sum("vllm:num_requests_waiting") as i32,
            token_usage: m.mean("vllm:gpu_cache_usage_perc"),
            ..Default::default()
        };
        let resp = WorkerMonitor::single_rank(snap);
        assert_eq!(resp.dp_rank_count, 1);
        assert_eq!(resp.effective_token_usage(), 0.75);
        assert_eq!(resp.loads[0].num_running_reqs, 3);
        assert_eq!(resp.loads[0].num_waiting_reqs, 5);
    }

    #[test]
    fn sglang_metrics_map_onto_token_usage_snapshot() {
        let m = PromScrape::parse(SGLANG_METRICS);
        assert_eq!(m.mean("sglang:token_usage"), 0.42);
        assert_eq!(m.sum("sglang:num_running_reqs"), 2.0);
        assert_eq!(m.sum("sglang:num_queue_reqs"), 4.0);
    }

    #[test]
    fn sglang_v054_underscore_prefix_is_recognized() {
        // SGLang v0.5.4+ renamed the metric prefix `sglang:` -> `sglang_`.
        let v054 = "sglang_token_usage{model=\"llama\"} 0.5\n\
                    sglang_num_running_reqs{model=\"llama\"} 7\n\
                    sglang_num_queue_reqs{model=\"llama\"} 1\n";
        let m = PromScrape::parse(v054);
        assert!(!m.has("sglang:token_usage"));
        let prefix = ["sglang:", "sglang_"]
            .into_iter()
            .find(|p| m.has(&format!("{p}token_usage")));
        assert_eq!(prefix, Some("sglang_"));
        assert_eq!(m.mean("sglang_token_usage"), 0.5);
        assert_eq!(m.sum("sglang_num_running_reqs"), 7.0);
    }

    #[test]
    fn parse_ignores_trailing_timestamp_and_label_spaces() {
        // Optional trailing timestamp must not be read as the value, and a
        // space inside a quoted label value must not split the head early.
        let text = "vllm:gpu_cache_usage_perc{model_name=\"my model\"} 0.75 1719849600000\n";
        let m = PromScrape::parse(text);
        assert_eq!(m.mean("vllm:gpu_cache_usage_perc"), 0.75);
    }

    #[test]
    fn vllm_v1_kv_cache_usage_metric_is_recognized() {
        // vLLM v1 renamed the gauge from `gpu_cache_usage_perc` to
        // `kv_cache_usage_perc`; the fetcher accepts either name.
        let v1 = "vllm:kv_cache_usage_perc{model_name=\"llama\"} 0.6\n";
        let m = PromScrape::parse(v1);
        assert!(!m.has("vllm:gpu_cache_usage_perc"));
        let kv_usage = ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"]
            .into_iter()
            .find(|name| m.has(name));
        assert_eq!(kv_usage, Some("vllm:kv_cache_usage_perc"));
        assert_eq!(m.mean("vllm:kv_cache_usage_perc"), 0.6);
    }

    #[test]
    fn metric_derived_snapshot_has_no_absolute_token_data() {
        // A ratio-only snapshot (from `/metrics`) must not be treated as
        // carrying absolute token counts: it stays out of the DP cache,
        // even at high KV usage.
        let resp = WorkerMonitor::single_rank(SchedulerLoadSnapshot {
            token_usage: 0.75,
            num_running_reqs: 3,
            ..Default::default()
        });
        assert!(!resp.has_absolute_token_data());
        assert_eq!(resp.total_used_tokens(), 0);
        assert_eq!(resp.effective_token_usage(), 0.75);
    }

    #[test]
    fn real_snapshot_with_capacity_has_absolute_token_data() {
        // A snapshot from gRPC/`/v1/loads` reports KV capacity, so its
        // absolute token fields are meaningful (even when idle).
        let resp = WorkerMonitor::single_rank(SchedulerLoadSnapshot {
            max_total_num_tokens: 8192,
            num_used_tokens: 1024,
            ..Default::default()
        });
        assert!(resp.has_absolute_token_data());
        assert_eq!(resp.total_used_tokens(), 1024);
    }

    #[test]
    fn fleet_rollup_is_not_dp_ranks() {
        // A gateway worker answers `/loads` with one entry per downstream
        // worker, all at `dp_rank: 0`. Keying those by rank collapses them
        // onto a single entry, so the DP cache must reject the response.
        let fleet = WorkerLoadResponse {
            dp_rank_count: 2,
            loads: vec![
                SchedulerLoadSnapshot {
                    worker: Some("http://a:8000".to_string()),
                    max_total_num_tokens: 8192,
                    num_used_tokens: 1024,
                    ..Default::default()
                },
                SchedulerLoadSnapshot {
                    worker: Some("http://b:8000".to_string()),
                    max_total_num_tokens: 8192,
                    num_used_tokens: 4096,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert!(fleet.has_absolute_token_data());
        assert!(!fleet.ranks_are_dp_ranks());
        assert_eq!(fleet.dp_rank_loads().len(), 1);

        // One engine's own ranks are distinct and stay eligible.
        let engine = WorkerLoadResponse {
            dp_rank_count: 2,
            loads: vec![
                SchedulerLoadSnapshot {
                    dp_rank: 0,
                    max_total_num_tokens: 8192,
                    num_used_tokens: 1024,
                    ..Default::default()
                },
                SchedulerLoadSnapshot {
                    dp_rank: 1,
                    max_total_num_tokens: 8192,
                    num_used_tokens: 4096,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert!(engine.ranks_are_dp_ranks());
        assert_eq!(engine.dp_rank_loads().len(), 2);
    }
}

#[cfg(test)]
mod native_loads_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use openai_protocol::{
        model_card::ModelCard,
        worker::{HealthCheckConfig, OverloadUpdate},
    };

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        policies::{LoadBalancingPolicy, SelectWorkerInfo},
        worker::{BasicWorkerBuilder, ConnectionMode, WorkerLoadGuard, WorkerType},
    };

    const VLLM_METRICS: &str = "vllm:num_requests_running{m=\"a\"} 7.0\n\
         vllm:num_requests_waiting{m=\"a\"} 11.0\n\
         vllm:kv_cache_usage_perc{m=\"a\"} 0.5\n";

    /// Carries `num_waiting_uncached_tokens`, which the `/metrics` arm has no
    /// gauge for — so its presence identifies which path answered.
    const NATIVE_BODY: &str = r#"{"loads":[{"dp_rank":0,"num_running_reqs":3,
        "num_waiting_reqs":4,"num_waiting_uncached_tokens":900,"token_usage":0.25}]}"#;

    const NATIVE_DP_BODY: &str = r#"{"dp_rank_count":2,"loads":[
        {"dp_rank":0,"num_running_reqs":3,"token_usage":0.25},
        {"dp_rank":1,"num_running_reqs":9,"token_usage":0.75}]}"#;

    const GATEWAY_FLEET_BODY: &str = r#"{"dp_rank_count":2,"loads":[
        {"worker":"http://engine-a","dp_rank":0,"num_running_reqs":3},
        {"worker":"http://engine-b","dp_rank":0,"num_running_reqs":9}]}"#;

    struct Stub {
        url: String,
        /// `/v1/loads` hits.
        probes: Arc<AtomicUsize>,
        /// `/loads` hits.
        gateway_probes: Arc<AtomicUsize>,
        /// Query string of the last `/loads` hit.
        gateway_query: Arc<Mutex<Option<String>>>,
    }

    /// Loopback backend stub. Each native route answers with the status and
    /// body it was given and counts its hits; `/metrics` always serves the
    /// vLLM gauges.
    async fn spawn_backend(
        gateway: (StatusCode, &'static str),
        engine: (StatusCode, &'static str),
    ) -> Stub {
        fn route(
            hits: &Arc<AtomicUsize>,
            query: &Arc<Mutex<Option<String>>>,
            (status, body): (StatusCode, &'static str),
        ) -> axum::routing::MethodRouter {
            let counter = Arc::clone(hits);
            let sink = Arc::clone(query);
            axum::routing::get(
                move |axum::extract::RawQuery(raw): axum::extract::RawQuery| {
                    let counter = Arc::clone(&counter);
                    let sink = Arc::clone(&sink);
                    async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                        *sink.lock() = raw;
                        (status, body)
                    }
                },
            )
        }

        let probes = Arc::new(AtomicUsize::new(0));
        let gateway_probes = Arc::new(AtomicUsize::new(0));
        let gateway_query = Arc::new(Mutex::new(None));
        let engine_query = Arc::new(Mutex::new(None));
        let app = axum::Router::new()
            .route("/loads", route(&gateway_probes, &gateway_query, gateway))
            .route("/v1/loads", route(&probes, &engine_query, engine))
            .route("/metrics", axum::routing::get(|| async { VLLM_METRICS }));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind stub");
        let addr = listener.local_addr().expect("stub address");
        #[expect(
            clippy::disallowed_methods,
            reason = "test stub server lives for the duration of the test process"
        )]
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("stub serve");
        });

        Stub {
            url: format!("http://{addr}"),
            probes,
            gateway_probes,
            gateway_query,
        }
    }

    /// An engine: serves `/v1/loads`, 404s the gateway route.
    async fn spawn_engine(status: StatusCode, body: &'static str) -> Stub {
        spawn_backend((StatusCode::NOT_FOUND, ""), (status, body)).await
    }

    /// A gateway registered as a worker: serves `/loads` only.
    async fn spawn_gateway(body: &'static str) -> Stub {
        spawn_backend((StatusCode::OK, body), (StatusCode::NOT_FOUND, "")).await
    }

    fn vllm_worker(url: &str) -> Arc<dyn Worker> {
        vllm_worker_with_overload(url, OverloadUpdate::default())
    }

    /// Build an HTTP virtual rank whose probes must use the base endpoint.
    fn vllm_dp_worker(url: &str, rank: usize, size: usize) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .runtime_type(RuntimeType::Vllm)
                .model(ModelCard::new("a"))
                .dp_config(rank, size)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// Worker carrying a per-worker overload block — protection for it is
    /// enabled by the spec alone, with no gateway thresholds anywhere.
    fn vllm_worker_with_overload(url: &str, overload: OverloadUpdate) -> Arc<dyn Worker> {
        vllm_worker_built(url, "a", overload)
    }

    fn vllm_worker_with_model(url: &str, model: &str) -> Arc<dyn Worker> {
        vllm_worker_built(url, model, OverloadUpdate::default())
    }

    fn vllm_worker_built(url: &str, model: &str, overload: OverloadUpdate) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .runtime_type(RuntimeType::Vllm)
                .model(ModelCard::new(model))
                .overload(overload)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    fn cache_aware_policy_config(overlap_decay: f32) -> PolicyConfig {
        PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 4096,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
            selection_policy: None,
            selection_accounting_ttl_ms: 0,
            balance_prefix_margin_tokens: 0,
            balance_uncached_credit: false,
        }
    }

    fn monitor_with_policy(config: PolicyConfig) -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>) {
        monitor_with(config, false)
    }

    fn monitor_with(
        config: PolicyConfig,
        conditional_polling: bool,
    ) -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>) {
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(config));
        let monitor = Arc::new(WorkerMonitor::new(
            registry.clone(),
            policy_registry,
            1,
            false,
            conditional_polling,
        ));
        (registry, monitor)
    }

    /// Poll `condition` until it holds or the timeout expires. The group loop
    /// writes the flag out of band, so there is nothing to await directly.
    async fn wait_until(label: &str, mut condition: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while !condition() {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {label}"));
    }

    /// A policy that records the reconciliation ticks it receives.
    #[derive(Debug, Default)]
    struct ReconcileRecorder(Mutex<Vec<(String, usize)>>);

    impl LoadBalancingPolicy for ReconcileRecorder {
        fn select_worker(
            &self,
            workers: &[Arc<dyn Worker>],
            _info: &SelectWorkerInfo,
        ) -> Option<usize> {
            (!workers.is_empty()).then_some(0)
        }

        fn reconcile_in_flight(&self, worker_url: &str, in_flight: usize) {
            self.0.lock().push((worker_url.to_string(), in_flight));
        }

        fn name(&self) -> &'static str {
            "reconcile_recorder"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// Every poll hands each polled worker's policy the router's live
    /// in-flight count on it, the safety net for a completion that never
    /// reached the policy.
    #[tokio::test]
    async fn load_poll_reconciles_in_flight_with_the_workers_policy() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, false);
        let recorder = Arc::new(ReconcileRecorder::default());
        monitor
            .policy_registry
            .set_decode_policy(Arc::clone(&recorder) as Arc<dyn LoadBalancingPolicy>);
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(stub.url.as_str())
                .worker_type(WorkerType::Decode)
                .connection_mode(ConnectionMode::Http)
                .runtime_type(RuntimeType::Vllm)
                .model(ModelCard::new("a"))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        worker.set_status(WorkerStatus::Ready);
        let held = WorkerLoadGuard::new(Arc::clone(&worker), None);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        wait_until("the poll to reconcile the worker's policy", || {
            recorder
                .0
                .lock()
                .iter()
                .any(|(url, in_flight)| *url == stub.url && *in_flight == 1)
        })
        .await;
        drop(held);
    }

    /// `NATIVE_BODY` reports 4 waiting requests, so a threshold of 4 vetoes and
    /// a threshold of 5 does not — the ingestion path must latch the verdict on
    /// the worker itself, under a policy (round-robin) that reads no loads at
    /// all. The threshold rides the worker's spec block, so this also proves a
    /// spec block alone enables protection with no gateway flags anywhere.
    #[tokio::test]
    async fn load_ingestion_flags_worker_over_waiting_threshold() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, false);
        let worker = vllm_worker_with_overload(
            &stub.url,
            OverloadUpdate {
                waiting_requests: Some(4),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        wait_until("worker to be flagged overloaded", || worker.is_overloaded()).await;
        assert_eq!(registry.overloaded_worker_count("a"), 1);
        assert!(
            !worker.is_available(),
            "an overloaded worker is not routable"
        );
    }

    /// A worker with real queueing that stays under the threshold must keep
    /// serving: the veto is absolute, not comparative.
    #[tokio::test]
    async fn busy_worker_under_threshold_stays_eligible() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, false);
        let worker = vllm_worker_with_overload(
            &stub.url,
            OverloadUpdate {
                waiting_requests: Some(5),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        // Wait for the feed to actually land, then assert the verdict.
        let mut rx = monitor.subscribe();
        tokio::time::timeout(Duration::from_secs(10), async {
            while !rx.borrow().contains(&stub.url) {
                rx.changed().await.expect("watch sender alive");
            }
        })
        .await
        .expect("load feed must reach the watch channel");

        assert!(!worker.is_overloaded());
        assert!(worker.is_available());
        assert_eq!(registry.overloaded_worker_count("a"), 0);
    }

    /// Overload protection alone must open the poll gate even under
    /// `--disable-load-monitoring`: without it the feature would silently
    /// never engage under a load-blind policy.
    #[tokio::test]
    async fn overload_thresholds_alone_start_load_polling() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, true);
        let worker = vllm_worker_with_overload(
            &stub.url,
            OverloadUpdate {
                waiting_requests: None,
                token_usage: Some(0.2),
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        wait_until("the engine to be polled", || {
            stub.probes.load(Ordering::SeqCst) > 0
        })
        .await;
        wait_until("token usage 0.25 to trip the 0.2 ceiling", || {
            worker.is_overloaded()
        })
        .await;
    }

    /// With protection off and no spec block the feature is off: an engine
    /// reporting a load that would trip either signal must still leave the
    /// flag, the counters and every routing verdict exactly where they were.
    #[tokio::test]
    async fn unconfigured_thresholds_never_write_the_flag() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        // cache_aware with overlap_decay on, so loads are polled and ingested
        // for reasons unrelated to overload protection.
        let (registry, monitor) = monitor_with(cache_aware_policy_config(1.0), false);
        let worker = vllm_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        let mut rx = monitor.subscribe();
        tokio::time::timeout(Duration::from_secs(10), async {
            while !rx.borrow().contains(&stub.url) {
                rx.changed().await.expect("watch sender alive");
            }
        })
        .await
        .expect("load feed must reach the watch channel");
        // A second tick, so this is not just "the first poll had not landed".
        tokio::time::sleep(Duration::from_millis(1200)).await;

        assert!(!worker.is_overloaded());
        assert!(worker.is_available());
        assert_eq!(registry.overloaded_worker_count("a"), 0);
    }

    /// A worker whose feed goes away has no verdict at all, so it must be
    /// re-admitted rather than stranded out of routing forever.
    #[tokio::test]
    async fn losing_the_load_feed_clears_the_veto() {
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, false);
        // Never-registered address: every fetch fails, so the report is absent.
        let worker = vllm_worker_with_overload(
            "http://127.0.0.1:1",
            OverloadUpdate {
                waiting_requests: Some(1),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        registry.set_worker_overloaded(&worker, true);
        assert_eq!(registry.overloaded_worker_count("a"), 1);

        monitor.start_event_loop();
        wait_until("the absent feed to clear the veto", || {
            !worker.is_overloaded()
        })
        .await;
        assert_eq!(registry.overloaded_worker_count("a"), 0);
    }

    /// `stop_all_groups` tears down the only writer that could clear the flag,
    /// so it must clear it itself — same rule the load snapshot follows.
    #[tokio::test]
    async fn stop_all_groups_clears_overload_flags() {
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, false);
        let worker = vllm_worker_with_overload(
            "http://127.0.0.1:1",
            OverloadUpdate {
                waiting_requests: Some(1),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        registry.set_worker_overloaded(&worker, true);

        monitor.stop_all_groups();

        assert!(!worker.is_overloaded());
        assert_eq!(registry.overloaded_worker_count("a"), 0);
    }

    /// A load-aware policy must still be fed with the opt-out set — the
    /// conditional gate is "poll when needed", never "never poll".
    #[tokio::test]
    async fn pressure_configured_cache_aware_starts_load_polling() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(cache_aware_policy_config(1.0), true);
        let worker = vllm_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(worker).unwrap();
        monitor.start_event_loop();

        let mut rx = monitor.subscribe();
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let waiting = rx
                    .borrow()
                    .get(&stub.url)
                    .map(|load| load.total_waiting_uncached_tokens());
                if let Some(waiting) = waiting {
                    assert_eq!(waiting, 900);
                    break;
                }
                rx.changed().await.expect("watch sender alive");
            }
        })
        .await
        .expect("cache_aware with overlap_decay must trigger load polling");
    }

    /// Load monitoring is on by default: a load-blind policy with no flags and
    /// no spec blocks still gets its workers polled from registration onward.
    #[tokio::test]
    async fn load_blind_policy_polls_by_default() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with_policy(cache_aware_policy_config(0.0));
        let worker = vllm_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        wait_until("default-on monitoring to poll the engine", || {
            stub.probes.load(Ordering::SeqCst) > 0
        })
        .await;
        // Polling alone must not write the overload flag.
        assert!(!worker.is_overloaded());
    }

    /// `--disable-load-monitoring` restores the old conditional gate: a
    /// load-blind policy with nothing needing the data is never polled.
    #[tokio::test]
    async fn disable_load_monitoring_restores_the_conditional_gate() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, true);
        let worker = vllm_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(worker).unwrap();
        monitor.start_event_loop();

        // Covers the immediate first tick plus one full 1s interval.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert_eq!(stub.probes.load(Ordering::SeqCst), 0);
        assert!(monitor.load_state.snapshot().is_empty());
    }

    #[tokio::test]
    async fn native_loads_preferred_over_metrics_on_vllm() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let worker = vllm_worker(&stub.url);
        let memo = DashMap::new();

        let resp = WorkerMonitor::fetch_http_load(&worker, Some(&memo))
            .await
            .expect("load response");

        // `/metrics` cannot report queued tokens; only the native path can.
        assert_eq!(resp.loads[0].num_waiting_uncached_tokens, 900);
        assert_eq!(resp.loads[0].num_running_reqs, 3);
        assert_eq!(
            memo.get(worker.url()).and_then(|hit| hit.answered()),
            Some(NativeLoadsPath::Engine)
        );
    }

    /// The load probe of a dp-aware HTTP worker goes to its engine's address:
    /// the `@rank` suffix of the worker's URL is the router's, not a host. The
    /// test checks the address (one probe, the engine's native report read);
    /// which entry of that report is the worker's rank is a separate question.
    #[tokio::test]
    async fn a_dp_aware_http_workers_probe_reaches_its_engine() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_DP_BODY).await;
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(stub.url.as_str())
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .runtime_type(RuntimeType::Vllm)
                .model(ModelCard::new("a"))
                .dp_config(1, 2)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        assert_eq!(worker.url(), format!("{}@1", stub.url));
        let memo = DashMap::new();
        let resp = WorkerMonitor::fetch_http_load(&worker, Some(&memo))
            .await
            .expect("the engine answers at its address");
        assert_eq!(resp.loads[0].dp_rank, 1);
        assert_eq!(resp.loads[0].num_running_reqs, 9);
        assert_eq!(stub.probes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    /// A backend-wide native response only updates the selected virtual rank.
    async fn native_loads_are_projected_to_the_virtual_worker_rank() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_DP_BODY).await;
        let worker = vllm_dp_worker(&stub.url, 1, 2);

        let resp = WorkerMonitor::fetch_http_load(&worker, None)
            .await
            .expect("rank 1 load response");

        assert_eq!(resp.dp_rank_count, 1);
        assert_eq!(resp.loads.len(), 1);
        assert_eq!(resp.loads[0].dp_rank, 1);
        assert_eq!(resp.loads[0].num_running_reqs, 9);
    }

    #[tokio::test]
    /// Missing native telemetry must not fall back to backend-wide averages.
    async fn missing_native_rank_never_uses_prometheus_average() {
        for body in [NATIVE_BODY, r#"{"loads":[]}"#] {
            let stub = spawn_engine(StatusCode::OK, body).await;
            let worker = vllm_dp_worker(&stub.url, 1, 2);
            let memo = DashMap::new();
            for _ in 0..2 {
                assert!(WorkerMonitor::fetch_http_load(&worker, Some(&memo))
                    .await
                    .expect("successful native poll")
                    .loads
                    .is_empty());
            }
            assert_eq!(stub.probes.load(Ordering::SeqCst), 2);
            assert_eq!(stub.gateway_probes.load(Ordering::SeqCst), 1);
        }
    }

    #[tokio::test]
    /// Gateway rollups retain their independent downstream worker identities.
    async fn gateway_fleet_loads_are_not_projected_as_engine_dp_ranks() {
        let stub = spawn_gateway(GATEWAY_FLEET_BODY).await;
        let worker = vllm_dp_worker(&stub.url, 1, 2);

        let resp = WorkerMonitor::fetch_http_load(&worker, None)
            .await
            .expect("gateway fleet load response");

        assert_eq!(resp.loads.len(), 2);
        assert!(resp.loads.iter().all(|load| load.worker.is_some()));
        assert_eq!(stub.gateway_probes.load(Ordering::SeqCst), 1);
        assert_eq!(stub.probes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn absent_native_endpoint_falls_back_and_is_probed_once() {
        let stub = spawn_engine(StatusCode::NOT_FOUND, "").await;
        let worker = vllm_worker(&stub.url);
        let memo = DashMap::new();

        for _ in 0..3 {
            let resp = WorkerMonitor::fetch_http_load(&worker, Some(&memo))
                .await
                .expect("metrics fallback");
            assert_eq!(resp.loads[0].num_running_reqs, 7);
            assert_eq!(resp.loads[0].num_waiting_reqs, 11);
        }

        let hit = memo.get(worker.url()).map(|hit| *hit).expect("memo entry");
        assert_eq!(hit.answered(), None);
        assert!(hit.is_absent(NativeLoadsPath::Gateway));
        assert!(hit.is_absent(NativeLoadsPath::Engine));
        assert_eq!(
            stub.probes.load(Ordering::SeqCst),
            1,
            "a 404 is definitive; the probe must not repeat every tick"
        );
        assert_eq!(stub.gateway_probes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn transient_native_failure_is_not_memoized() {
        // 503 says the backend is unwell, not that the route is missing.
        // Memoizing it would demote a healthy engine to `/metrics` forever.
        // The gateway route's 404 next to it is a fact, though, and tracking
        // absence per route is what lets the dead one be dropped while the
        // unwell one keeps being asked.
        let stub = spawn_engine(StatusCode::SERVICE_UNAVAILABLE, "").await;
        let worker = vllm_worker(&stub.url);
        let memo = DashMap::new();

        for _ in 0..3 {
            WorkerMonitor::fetch_http_load(&worker, Some(&memo))
                .await
                .expect("metrics fallback");
        }

        let hit = memo.get(worker.url()).map(|hit| *hit).expect("memo entry");
        assert!(!hit.is_absent(NativeLoadsPath::Engine));
        assert_eq!(hit.answered(), None);
        assert_eq!(stub.probes.load(Ordering::SeqCst), 3, "probe must retry");
        assert_eq!(stub.gateway_probes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    /// Unrelated JSON and malformed load fields cannot suppress metrics fallback.
    async fn non_load_body_on_200_is_treated_as_absent() {
        for body in [
            "<html>nope</html>",
            "{}",
            r#"{"status":"ok"}"#,
            r#"{"loads":null}"#,
        ] {
            let stub = spawn_engine(StatusCode::OK, body).await;
            let worker = vllm_worker(&stub.url);
            let memo = DashMap::new();

            WorkerMonitor::fetch_http_load(&worker, Some(&memo))
                .await
                .expect("metrics fallback");

            let hit = memo.get(worker.url()).map(|hit| *hit).expect("memo entry");
            assert!(hit.is_absent(NativeLoadsPath::Engine));
        }
    }

    #[tokio::test]
    /// An empty native response stays unavailable while future polls retry it.
    async fn empty_loads_array_keeps_probing() {
        // Our schema, no ranks yet — the endpoint exists, so keep asking.
        let stub = spawn_engine(StatusCode::OK, r#"{"loads":[]}"#).await;
        let worker = vllm_worker(&stub.url);
        let memo = DashMap::new();

        for _ in 0..2 {
            assert!(WorkerMonitor::fetch_http_load(&worker, Some(&memo))
                .await
                .expect("successful native poll")
                .loads
                .is_empty());
        }

        let hit = memo.get(worker.url()).map(|hit| *hit).expect("memo entry");
        assert!(!hit.is_absent(NativeLoadsPath::Engine));
        assert_eq!(stub.probes.load(Ordering::SeqCst), 2);
        assert_eq!(stub.gateway_probes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn absent_memo_is_bypassed_when_caller_has_none() {
        let stub = spawn_engine(StatusCode::NOT_FOUND, "").await;
        let worker = vllm_worker(&stub.url);

        for _ in 0..2 {
            WorkerMonitor::fetch_http_load(&worker, None)
                .await
                .expect("metrics fallback");
        }

        assert_eq!(stub.probes.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn gateway_route_answers_and_is_memoized() {
        // A gateway registered as a worker: `/loads` is the only route it
        // has, and it must be tried before the engine one.
        let stub = spawn_gateway(NATIVE_BODY).await;
        let worker = vllm_worker(&stub.url);
        let memo = DashMap::new();

        for _ in 0..3 {
            let resp = WorkerMonitor::fetch_http_load(&worker, Some(&memo))
                .await
                .expect("load response");
            assert_eq!(resp.loads[0].num_waiting_uncached_tokens, 900);
        }

        assert_eq!(
            memo.get(worker.url()).and_then(|hit| hit.answered()),
            Some(NativeLoadsPath::Gateway)
        );
        assert_eq!(stub.gateway_probes.load(Ordering::SeqCst), 3);
        assert_eq!(
            stub.probes.load(Ordering::SeqCst),
            0,
            "the engine route must not be probed once the gateway one answers"
        );
    }

    #[tokio::test]
    async fn memoized_route_that_stops_answering_is_rediscovered() {
        // An engine-only stub with the gateway route memoized: the stale
        // memo must be dropped, not treated as "serves neither".
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let worker = vllm_worker(&stub.url);
        let memo = DashMap::new();
        let mut stale = NativeLoadsMemo::default();
        stale.record_answered(NativeLoadsPath::Gateway);
        memo.insert(worker.url().to_string(), stale);

        let resp = WorkerMonitor::fetch_http_load(&worker, Some(&memo))
            .await
            .expect("metrics fallback");
        assert_eq!(resp.loads[0].num_running_reqs, 7, "served by /metrics");
        let hit = memo.get(worker.url()).map(|hit| *hit).expect("memo entry");
        assert_eq!(hit.answered(), None, "a stale memo must be forgotten");

        let resp = WorkerMonitor::fetch_http_load(&worker, Some(&memo))
            .await
            .expect("load response");
        assert_eq!(resp.loads[0].num_running_reqs, 3, "served by /v1/loads");
        assert_eq!(
            memo.get(worker.url()).and_then(|hit| hit.answered()),
            Some(NativeLoadsPath::Engine)
        );
        assert_eq!(
            stub.gateway_probes.load(Ordering::SeqCst),
            1,
            "the route that 404'd must not be probed again"
        );
    }

    #[tokio::test]
    async fn gateway_probe_is_scoped_to_the_worker_model() {
        // A gateway fronts many models; the load of the one this worker was
        // registered for is the only load that belongs to it.
        let stub = spawn_gateway(NATIVE_BODY).await;
        let worker = vllm_worker_with_model(&stub.url, "meta-llama/Llama-3.1-8B");

        WorkerMonitor::fetch_http_load(&worker, None)
            .await
            .expect("load response");

        assert_eq!(
            stub.gateway_query.lock().clone(),
            Some("model=meta-llama%2FLlama-3.1-8B".to_string())
        );
    }

    /// An omitted virtual rank is evicted; another cached rank and failed polls survive.
    #[tokio::test]
    async fn successful_missing_rank_evicts_only_that_rank_from_the_dp_cache() {
        for (status, body, missing) in [
            (StatusCode::OK, NATIVE_BODY, true),
            (StatusCode::OK, r#"{"loads":[]}"#, true),
            (StatusCode::SERVICE_UNAVAILABLE, "unavailable", false),
            (StatusCode::NOT_FOUND, "", false),
        ] {
            let stub = spawn_engine(status, body).await;
            let (registry, monitor) = monitor_with_policy(PolicyConfig::RoundRobin);
            let worker = vllm_dp_worker(&stub.url, 1, 2);
            worker.set_status(WorkerStatus::Ready);
            registry.register(Arc::clone(&worker)).unwrap();
            // Simulate a previously cached backend-wide snapshot under this URL.
            monitor
                .worker_load_manager
                .update_dp_loads(&HashMap::from([(
                    worker.url().to_string(),
                    HashMap::from([(0, 100), (1, 200)]),
                )]));
            let key = WorkerGroupKey {
                model_id: "a".to_string(),
                worker_type: WorkerType::Regular,
                connection_mode: ConnectionMode::Http,
            };
            poll_group_once(&monitor, &key, Duration::from_secs(10), Instant::now()).await;
            let cached = monitor.worker_load_manager.dp_cached_loads.read();
            let ranks = cached.get(worker.url()).expect("other cached rank remains");
            assert_eq!(ranks.get(&0), Some(&100));
            assert_eq!(ranks.get(&1), if missing { None } else { Some(&200) });
            drop(cached);
            assert!(monitor.load_state.snapshot().get(worker.url()).is_none());
            assert_eq!(
                monitor
                    .worker_load_manager
                    .select_and_increment_lowest_dp_load(worker.as_ref(), 1),
                if missing { None } else { Some(1) },
            );
        }
    }

    /// Empty polls retract old ranks without discarding concurrent stream records.
    #[tokio::test]
    async fn empty_poll_does_not_resurrect_stale_pushed_ranks() {
        for fresh_offset in [None, Some(Duration::ZERO), Some(Duration::from_millis(1))] {
            let stub = spawn_engine(StatusCode::OK, r#"{"loads":[]}"#).await;
            let (registry, monitor) = monitor_with_policy(PolicyConfig::RoundRobin);
            let worker = vllm_worker(&stub.url);
            worker.set_status(WorkerStatus::Ready);
            registry.register(Arc::clone(&worker)).unwrap();
            let interval = Duration::from_secs(10);
            let t0 = Instant::now();
            let polled_at = t0 + interval;
            for rank in 0..2 {
                monitor.apply_pushed_load(&worker, rank, &EngineLoad::default(), t0);
            }
            let absolute = WorkerLoadResponse {
                loads: (0..2)
                    .map(|dp_rank| SchedulerLoadSnapshot {
                        dp_rank,
                        num_used_tokens: 100,
                        max_total_num_tokens: 1000,
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            };
            monitor.apply_polled_load(worker.url(), &absolute, t0 + Duration::from_secs(1));
            if let Some(offset) = fresh_offset {
                monitor.apply_pushed_load(&worker, 1, &EngineLoad::default(), polled_at + offset);
            }
            let key = WorkerGroupKey {
                model_id: "a".to_string(),
                worker_type: WorkerType::Regular,
                connection_mode: ConnectionMode::Http,
            };
            poll_group_once(&monitor, &key, interval, polled_at).await;
            monitor.flush_pushed();
            let snapshot = monitor.load_state.snapshot();
            if let Some(report) = snapshot.get(worker.url()) {
                assert!(fresh_offset.is_some());
                assert_eq!(report.loads.len(), 1);
                assert_eq!(report.loads[0].dp_rank, 1);
            }
            assert_eq!(
                monitor.pushed_at.lock()[worker.url()].contains_key(&1),
                fresh_offset.is_some()
            );
            monitor.apply_pushed_load(
                &worker,
                0,
                &EngineLoad::default(),
                polled_at + Duration::from_secs(1),
            );
            monitor.flush_pushed();
            let snapshot = monitor.load_state.snapshot();
            let report = snapshot.get(worker.url()).unwrap();
            assert_eq!(
                report.loads.len(),
                if fresh_offset.is_some() { 2 } else { 1 }
            );
            if fresh_offset.is_none() {
                assert_eq!(report.loads[0].dp_rank, 0);
                let cached = monitor.worker_load_manager.dp_cached_loads.read();
                assert!(cached
                    .get(worker.url())
                    .is_none_or(|ranks| !ranks.contains_key(&1)));
            }
        }
    }

    /// Two HTTP rank workers of one engine share one read of its native load
    /// route per tick, and each one's published report is its own rank's
    /// entry of the engine's two-rank answer: the shared snapshot, the DP
    /// token cache and the overload verdict read the rank.
    #[tokio::test]
    async fn the_rank_workers_of_one_http_engine_share_one_native_read() {
        const TWO_RANKS: &str = r#"{"loads":[
            {"dp_rank":0,"num_running_reqs":3,"num_waiting_reqs":1,"num_used_tokens":300,"max_total_num_tokens":1000,"token_usage":0.3},
            {"dp_rank":1,"num_running_reqs":9,"num_waiting_reqs":4,"num_used_tokens":900,"max_total_num_tokens":1000,"token_usage":0.9}]}"#;
        let stub = spawn_engine(StatusCode::OK, TWO_RANKS).await;
        let (registry, monitor) = monitor_with_policy(PolicyConfig::RoundRobin);
        let workers = [
            vllm_dp_worker(&stub.url, 0, 2),
            vllm_dp_worker(&stub.url, 1, 2),
        ];
        for worker in &workers {
            registry.register(Arc::clone(worker)).unwrap();
        }
        let key = WorkerGroupKey {
            model_id: "a".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
        };
        // A transport failure noted on each rank worker: the one read's
        // contact must clear it on both.
        for worker in &workers {
            worker.note_transport_failure();
        }
        poll_group_once(&monitor, &key, Duration::from_secs(10), Instant::now()).await;

        let snapshot = monitor.load_state.snapshot();
        for (worker, expected) in workers.iter().zip([(0, 3, 1, 300), (1, 9, 4, 900)]) {
            assert!(
                !worker.transport_failure_pending(),
                "the contact is every member's: {}",
                worker.url()
            );
            let report = snapshot.get(worker.url()).expect("published");
            let ranks: Vec<(i32, i64, i64, i64)> = report
                .loads
                .iter()
                .map(|rank| {
                    (
                        rank.dp_rank,
                        i64::from(rank.num_running_reqs),
                        i64::from(rank.num_waiting_reqs),
                        i64::from(rank.num_used_tokens),
                    )
                })
                .collect();
            assert_eq!(ranks, vec![expected], "{}", worker.url());
            assert_eq!(report.dp_rank_count, 1, "{}", worker.url());
            assert_eq!(
                monitor
                    .worker_load_manager
                    .select_and_increment_lowest_dp_load(worker.as_ref(), 0),
                Some(expected.0 as isize),
                "{}",
                worker.url()
            );
        }
        assert_eq!(
            stub.probes.load(Ordering::SeqCst),
            1,
            "one read of the engine's native route for its two rank workers"
        );
    }

    /// The rank workers of one gRPC engine share one `GetLoads` per tick: the
    /// engine answers once, and every rank worker gets its contact and its
    /// own rank's part of the answer, as its own poll would have had it (the
    /// canned engine names rank 0 only, so rank 1 has no report).
    #[tokio::test]
    async fn the_rank_workers_of_one_grpc_engine_share_one_poll() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let cfg = Arc::new(mock_worker::config::Config {
            http_count: 0,
            grpc_base_port: port,
            grpc_count: 1,
            model_id: "a".to_string(),
            tokenizer_path: "a".to_string(),
            gen_delay: Duration::ZERO,
            output_tokens: 3,
            realistic: false,
            ..mock_worker::config::Config::default()
        });
        #[expect(
            clippy::disallowed_methods,
            reason = "test engine: the server task lives for the test process"
        )]
        let _engine = tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
        let (registry, monitor) = monitor_with_policy(PolicyConfig::RoundRobin);
        let rank_worker = |rank: usize| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::TokenSpeed)
                    .model(ModelCard::new("a"))
                    .dp_config(rank, 2)
                    .health_config(HealthCheckConfig {
                        disable_health_check: true,
                        ..Default::default()
                    })
                    .build(),
            )
        };
        let workers = [rank_worker(0), rank_worker(1)];
        for worker in &workers {
            registry.register(Arc::clone(worker)).unwrap();
        }
        let key = WorkerGroupKey {
            model_id: "a".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Grpc,
        };
        let polls_before = mock_worker::overrides::for_port(port).loads_polls();
        // A transport failure noted on each rank worker: the one GetLoads'
        // contact must clear it on both.
        for worker in &workers {
            worker.note_transport_failure();
        }
        poll_group_once(&monitor, &key, Duration::from_secs(10), Instant::now()).await;
        assert_eq!(
            mock_worker::overrides::for_port(port).loads_polls() - polls_before,
            1,
            "one GetLoads for the engine's two rank workers"
        );

        let snapshot = monitor.load_state.snapshot();
        let report = snapshot
            .get(workers[0].url())
            .expect("the rank the answer names has its report");
        assert_eq!((report.loads.len(), report.loads[0].dp_rank), (1, 0));
        assert!(
            snapshot.get(workers[1].url()).is_none(),
            "a rank the answer does not name has no report, as its own poll would have had it"
        );
        for worker in &workers {
            assert!(
                !worker.transport_failure_pending(),
                "the contact is every member's: {}",
                worker.url()
            );
        }
    }

    /// The tick's fallback poll of a worker whose stream went quiet on one
    /// rank refreshes that rank's entry in the pushed report: the next
    /// record the stream pushes rebuilds the report from the poll's figure
    /// for it, not from the rank's last pushed record.
    #[tokio::test]
    async fn a_fallback_polls_figures_reach_the_pushed_report_of_a_quiet_rank() {
        const TWO_RANKS: &str = r#"{"loads":[
            {"dp_rank":0,"num_running_reqs":3,"num_waiting_reqs":0,"token_usage":0.25,"cache_hit_rate":0.9},
            {"dp_rank":1,"num_running_reqs":8,"num_waiting_reqs":0,"token_usage":0.5,"cache_hit_rate":0.91}]}"#;
        let stub = spawn_engine(StatusCode::OK, TWO_RANKS).await;
        let (registry, monitor) = monitor_with_policy(PolicyConfig::RoundRobin);
        let worker = vllm_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let key = WorkerGroupKey {
            model_id: "a".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
        };
        let interval = Duration::from_secs(10);
        let t0 = Instant::now();
        let record = |running: u32, hit_rate: f64, sample: u64| EngineLoad {
            running_requests: running,
            cache_hit_rate: Some(hit_rate),
            sample,
            ..Default::default()
        };
        monitor.apply_pushed_load(&worker, 0, &record(2, 0.5, 1), t0);
        monitor.apply_pushed_load(&worker, 1, &record(8, 0.65, 2), t0);

        // One interval later rank 1's record is older than the interval:
        // the tick polls the worker.
        poll_group_once(&monitor, &key, interval, t0 + interval).await;
        assert_eq!(stub.probes.load(Ordering::SeqCst), 1, "one fallback poll");

        // The stream's next record names rank 0 only.
        monitor.apply_pushed_load(
            &worker,
            0,
            &record(4, 0.5, 3),
            t0 + interval + Duration::from_millis(1),
        );
        monitor.flush_pushed();
        let snapshot = monitor.load_state.snapshot();
        let report = snapshot.get(stub.url.as_str()).expect("published");
        assert_eq!(report.version, "pushed");
        let ranks: Vec<(i32, i32, f64)> = report
            .loads
            .iter()
            .map(|rank| (rank.dp_rank, rank.num_running_reqs, rank.cache_hit_rate))
            .collect();
        assert_eq!(ranks, vec![(0, 4, 0.5), (1, 8, 0.91)]);
    }
}
