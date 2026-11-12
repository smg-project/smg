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

    /// One worker's rank loads replace its cached entry: a pushed record's
    /// update, without the map of maps a poll tick hands over.
    pub(crate) fn update_worker_dp_loads(&self, url: &str, loads: HashMap<isize, isize>) {
        let mut cached = self.dp_cached_loads.write();
        if let Some(entry) = cached.get_mut(url) {
            *entry = loads;
        } else {
            cached.insert(url.to_string(), loads);
        }
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
    /// Each worker's report from its pushed records, in the poll's shape with
    /// the ranks sorted, as its stream and the poll last left it: a record
    /// pushed on the KV-event stream replaces its rank's entry in place
    /// ([`Self::apply_pushed_load`]), a poll's entry replaces a rank's that
    /// pushed nothing since the poll began ([`Self::apply_polled_load`]).
    /// A record's readers borrow it, and the coalesced publish copies it
    /// once per window, so a record builds no report of its own.
    pushed_reports: Mutex<HashMap<String, WorkerLoadResponse>>,
    /// When each rank of a worker last pushed a load record, by receipt:
    /// the tick's poll decision reads it (see [`PollMode`]).
    pushed_at: Mutex<HashMap<String, BTreeMap<i32, Instant>>>,
    /// The workers whose report changed since the last coalesced publish,
    /// each with the incarnation that pushed, for the publish fence.
    pending_pushed: Mutex<HashMap<String, Arc<dyn Worker>>>,
    pushed_notify: Arc<Notify>,
    pushed_flush_task: Mutex<Option<JoinHandle<()>>>,
    /// Each worker's rank instants under its own lock, for a worker without
    /// a stream, where the host's paths publish: a publish holds the lock
    /// from the snapshot read through the publish and arbitrates per rank
    /// by each row's own sample instant (see [`RankInstants`]).
    rank_instants: Mutex<HashMap<String, Arc<Mutex<Option<RankInstants>>>>>,
    /// A test's hook, run under a worker's read lock just before the read's
    /// report is published, so a test can see the lock held at that point.
    #[cfg(test)]
    pub(crate) before_read_publish: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}

/// A worker's published copy of its pushed report, with the worker it came
/// from for the publish fence.
type PushedReport = (Arc<dyn Worker>, Arc<WorkerLoadResponse>);

/// The sample instants of a worker's published rows, for a worker without
/// a stream, where the host's paths publish (a read of one rank, a poll
/// through the store): an instant travels with its value, so a publish
/// arbitrates per rank, the newest sample wins and an older row, a replayed
/// poll's, is dropped for its rank; the report's instant is the earliest of
/// its rows'. `published` is the report these paths wrote last, by
/// identity: a published report that is not it is the poll loop's, whose
/// rows all carry its instant, and the record follows it.
#[derive(Debug)]
pub(crate) struct RankInstants {
    ranks: BTreeMap<i32, Instant>,
    published: Option<Arc<WorkerLoadResponse>>,
}

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
/// engine never reports stays unset. `previous` is the rank's entry the
/// record replaces; what it keeps moves out of it.
fn keep_telemetry(
    fresh: &mut SchedulerLoadSnapshot,
    previous: SchedulerLoadSnapshot,
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
        fresh.memory = previous.memory;
    }
    if record.queues.is_none() {
        fresh.queues = previous.queues;
    }
    if record.disaggregation.is_none() {
        fresh.kv_transfer_latency_ms = previous.kv_transfer_latency_ms;
        fresh.kv_transfer_speed_gb_s = previous.kv_transfer_speed_gb_s;
        fresh.prefill_queue_reqs = previous.prefill_queue_reqs;
        fresh.decode_queue_reqs = previous.decode_queue_reqs;
        fresh.disagg_mode = previous.disagg_mode;
    }
}

/// A pushed record is sampled this much before its receipt on top of its own
/// `age_ms`: the servicer-to-gateway one-way latency.
const PUSHED_ONE_WAY_MARGIN: Duration = Duration::from_millis(20);
/// Pushed reports are published into the shared snapshot in one rebuild per
/// window: the rebuild is O(fleet) pointer work, the records arrive per
/// scheduler step from every engine, and a worker's poll-shaped copy is made
/// once per window, not per record. Routing reads the snapshot, so a record
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
            pushed_reports: Mutex::new(HashMap::new()),
            pushed_at: Mutex::new(HashMap::new()),
            pending_pushed: Mutex::new(HashMap::new()),
            pushed_notify: Arc::new(Notify::new()),
            pushed_flush_task: Mutex::new(None),
            rank_instants: Mutex::new(HashMap::new()),
            #[cfg(test)]
            before_read_publish: Mutex::new(None),
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
    /// the rank's entry in the worker's report in place, goes to the overload
    /// verdict and the wedged rule at once, which read the report where it
    /// is, to its own rank's `smg_engine_*` gauges, set through the handles
    /// kept for that rank, and into the shared snapshot, which routing
    /// reads, in the next coalesced publish, which copies the report once
    /// per window: a record allocates no report, key, label or timestamp of
    /// its own. The report's `sampled_at` is the receipt less the record's
    /// age and the one-way margin, what a policy releases its dispatch credit
    /// against. Negative ranks and records for another virtual worker's rank
    /// are ignored before updating freshness, liveness, metrics or routing
    /// state.
    pub fn apply_pushed_load(
        self: &Arc<Self>,
        worker: &Arc<dyn Worker>,
        dp_rank: i32,
        record: &EngineLoad,
        received_at: Instant,
    ) {
        self.merge_rank_record(worker, dp_rank, record, received_at, true);
    }

    /// One rank's record from the gateway's own read of the worker (a poll
    /// of one rank, a scrape), received at `received_at`. A worker with a
    /// report on file, its stream's, takes it in place as a pushed record,
    /// so a rank whose pushed record waits for the window keeps it; a worker
    /// without one takes the poll's path over its published rows, so the
    /// ranks a poll learned stay published and the read is published at
    /// once, with no report and no receipt made. The read is no stream: no
    /// push receipt, so the poll decision stays the poll's, no liveness
    /// cadence mark, and no one-way margin on its age; the read joins the
    /// published rows under the worker's lock, where each rank's newest
    /// sample wins and the report takes the earliest of its rows' own
    /// instants (see [`Self::arbitrated_report`]).
    pub(crate) fn apply_read_load(
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
        let url = worker.url();
        if self.pushed_reports.lock().contains_key(url) {
            self.merge_rank_record(worker, dp_rank, record, received_at, false);
            return;
        }
        let age = Duration::from_millis(u64::from(record.age_ms));
        let sampled_at = received_at.checked_sub(age).unwrap_or(received_at);
        let mut read = SchedulerLoadSnapshot::from(record);
        read.dp_rank = dp_rank;
        // The worker's lock from the snapshot read through the publish: a
        // second read of another rank waits for it, so neither publishes
        // the other's rank from a snapshot the other has replaced.
        let lock = self.rank_lock_of(url);
        let mut instants = lock.lock();
        let Some(load) = self.arbitrated_report(url, &mut instants, vec![(read, sampled_at)], None)
        else {
            // No newer than its rank's published row: nothing to publish.
            return;
        };
        #[cfg(test)]
        if let Some(hook) = self.before_read_publish.lock().as_ref() {
            hook();
        }
        self.publish_polled_report(worker, url, load);
        // The worker's lock (`instants`) drops here, after the publish.
    }

    /// The lock and rank instants of one worker.
    fn rank_lock_of(&self, url: &str) -> Arc<Mutex<Option<RankInstants>>> {
        Arc::clone(
            self.rank_instants
                .lock()
                .entry(url.to_string())
                .or_default(),
        )
    }

    /// The lock of one worker, for a test that holds it.
    #[cfg(test)]
    pub(crate) fn rank_lock(&self, url: &str) -> Arc<Mutex<Option<RankInstants>>> {
        self.rank_lock_of(url)
    }

    /// The one arbitration for a worker without a stream, under its lock:
    /// the incoming rows, each with its own sample instant (a read's its
    /// read's, a poll's rows the poll's), over the published rows with
    /// theirs. Per rank the newest sample wins, so an older row, a replayed
    /// poll's, is dropped for its rank and never merged with newer
    /// metadata; a whole worker's report (`poll_ranks`, its rank count)
    /// also drops the published ranks it does not name unless a newer
    /// sample kept them, a read keeps them all. The report's instant is the
    /// earliest of its rows', the instants follow the values into the
    /// record, and `None` says the rows brought nothing newer: no publish.
    fn arbitrated_report(
        &self,
        url: &str,
        instants: &mut Option<RankInstants>,
        incoming: Vec<(SchedulerLoadSnapshot, Instant)>,
        poll_ranks: Option<i32>,
    ) -> Option<Arc<WorkerLoadResponse>> {
        let snapshot = self.load_state.snapshot();
        let published = snapshot
            .report(url)
            .filter(|(report, _)| report.ranks_are_dp_ranks());
        let record = instants.get_or_insert_with(|| RankInstants {
            ranks: BTreeMap::new(),
            published: None,
        });
        // The record follows the published report: one these paths did not
        // write is the poll loop's, with one instant for every row.
        let ours = match (published, &record.published) {
            (Some((report, _)), Some(last)) => std::ptr::eq(report, Arc::as_ptr(last)),
            _ => false,
        };
        if !ours {
            record.ranks = published
                .map(|(report, at)| report.loads.iter().map(|row| (row.dp_rank, at)).collect())
                .unwrap_or_default();
        }
        let mut rows: BTreeMap<i32, (SchedulerLoadSnapshot, Instant)> = published
            .map(|(report, at)| {
                report
                    .loads
                    .iter()
                    .map(|row| {
                        let instant = record.ranks.get(&row.dp_rank).copied().unwrap_or(at);
                        (row.dp_rank, (row.clone(), instant))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let newest = incoming.iter().map(|(_, at)| *at).max()?;
        let named: Vec<i32> = incoming.iter().map(|(row, _)| row.dp_rank).collect();
        let mut fresh = false;
        for (row, at) in incoming {
            match rows.get(&row.dp_rank) {
                // An older or replayed sample of a rank the report has.
                Some((_, have)) if *have >= at => {}
                _ => {
                    rows.insert(row.dp_rank, (row, at));
                    fresh = true;
                }
            }
        }
        if poll_ranks.is_some() {
            let before = rows.len();
            rows.retain(|rank, (_, at)| named.contains(rank) || *at > newest);
            fresh |= rows.len() != before;
        }
        if !fresh {
            return None;
        }
        record.ranks = rows.iter().map(|(rank, (_, at))| (*rank, *at)).collect();
        let instant = rows.values().map(|(_, at)| *at).min().unwrap_or(newest);
        let loads: Vec<SchedulerLoadSnapshot> = rows.into_values().map(|(row, _)| row).collect();
        let count = i32::try_from(loads.len()).unwrap_or(i32::MAX);
        let load = Arc::new(WorkerLoadResponse {
            dp_rank_count: poll_ranks.map_or(count, |ranks| ranks.max(count)),
            loads,
            sampled_at: Some(instant),
            ..WorkerLoadResponse::default()
        });
        record.published = Some(Arc::clone(&load));
        Some(load)
    }

    /// The in-place merge behind both: `pushed` marks a stream's record,
    /// which alone carries the one-way margin, the cadence mark and the
    /// receipt.
    fn merge_rank_record(
        self: &Arc<Self>,
        worker: &Arc<dyn Worker>,
        dp_rank: i32,
        record: &EngineLoad,
        received_at: Instant,
        pushed: bool,
    ) {
        if dp_rank < 0
            || worker
                .dp_rank()
                .is_some_and(|rank| usize::try_from(dp_rank) != Ok(rank))
        {
            return;
        }
        let mut age = Duration::from_millis(u64::from(record.age_ms));
        if pushed {
            age += PUSHED_ONE_WAY_MARGIN;
            // The stream pushes: silence from this worker is judged by the
            // stream's cadence from now on (`liveness`, the stale rule).
            worker.note_load_record();
        }
        let sampled_at = received_at.checked_sub(age).unwrap_or(received_at);
        let url = worker.url();
        // The poll decision reads the receipt, not the sample's age: a record
        // that arrives is a stream that works, whatever the engine's clock.
        // A worker's entries are made on its first record; the lookups that
        // find them copy no key.
        let mut reports = self.pushed_reports.lock();
        if pushed {
            let mut receipts = self.pushed_at.lock();
            if let Some(ranks) = receipts.get_mut(url) {
                ranks.insert(dp_rank, received_at);
            } else {
                receipts.insert(url.to_string(), BTreeMap::from([(dp_rank, received_at)]));
            }
        }
        if !reports.contains_key(url) {
            reports.insert(url.to_string(), WorkerLoadResponse::default());
        }
        // On file: made just above when it was not.
        let Some(report) = reports.get_mut(url) else {
            return;
        };
        // What a poll would have put in the rank's entry, with the telemetry
        // the record left out (an event batch's record is the core only)
        // kept from the last record that carried it.
        let mut fresh = SchedulerLoadSnapshot::from(record);
        fresh.dp_rank = dp_rank;
        let at = match report
            .loads
            .binary_search_by_key(&dp_rank, |load| load.dp_rank)
        {
            Ok(at) => {
                let previous = std::mem::replace(&mut report.loads[at], fresh);
                keep_telemetry(&mut report.loads[at], previous, record);
                at
            }
            Err(at) => {
                report.loads.insert(at, fresh);
                report.dp_rank_count = i32::try_from(report.loads.len()).unwrap_or(i32::MAX);
                at
            }
        };
        report.sampled_at = Some(sampled_at);
        // What a poll does with a fresh report, read off the worker's own.
        self.apply_overload_verdict(
            worker,
            Some(&*report),
            if pushed {
                "pushed record"
            } else {
                "read record"
            },
        );
        liveness::on_load_report(
            worker,
            report.total_waiting_reqs(),
            record
                .engine_silence_ms
                .map(|ms| Duration::from_millis(u64::from(ms))),
            record.engine_reports_steps.unwrap_or(false),
            record.prefill_pending_tokens.map(u64::from),
        );
        // The record feeds its own rank's `smg_engine_*` series through the
        // handles kept for it; a poll's report feeds every rank's.
        Metrics::record_engine_rank_load(url, worker.model_id(), &report.loads[at]);
        // The DP-rank token cache takes a report with absolute token counts,
        // as it takes a poll; a core-only record leaves its last-known-good
        // entry alone.
        if report.has_absolute_token_data() && report.ranks_are_dp_ranks() {
            self.worker_load_manager
                .update_worker_dp_loads(url, report.dp_rank_loads());
        }
        // The window's publish takes the report once, under the incarnation
        // that pushed last.
        {
            let mut pending = self.pending_pushed.lock();
            if pending
                .get(url)
                .is_none_or(|source| !Arc::ptr_eq(source, worker))
            {
                pending.insert(url.to_string(), Arc::clone(worker));
            }
        }
        drop(reports);
        self.start_pushed_flusher();
        self.pushed_notify.notify_one();
    }

    /// Whether `workers`, the members of one group, are to have live load
    /// state: always by default; under `--disable-load-monitoring` only when
    /// a load-aware or dp-rank policy, the engine-metrics export or overload
    /// protection on a member needs it. The poll tick and the decision to
    /// open a worker's load stream read the same answer.
    pub(crate) fn loads_wanted(&self, workers: &[Arc<dyn Worker>]) -> bool {
        if !self.conditional_polling {
            return true;
        }
        let routing_needs_load = self.policy_registry.needs_backend_loads()
            || self.policy_registry.get_dp_rank_policy().is_some();
        let overload_needs_load = workers.iter().any(|w| w.metadata().overload.is_enabled());
        routing_needs_load || self.engine_metrics || overload_needs_load
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
    /// arriving while the worker's report keeps, for that rank, the record
    /// it last sent. The tick's decision ([`Self::poll_mode`]) falls back
    /// to the poll for such a worker; the poll's entry for the quiet rank
    /// replaces the one on file, so the report the next push publishes
    /// carries it and the rank's figures are at most a tick interval old,
    /// the poll decision's own bound. A rank whose record arrived
    /// since the poll began keeps it: the stream's is the newer. The
    /// receipts (`pushed_at`) are not touched: a poll is not the stream
    /// working, and the worker stays polled until the rank's records
    /// resume. Only a rank on file is replaced: a rank the stream never
    /// named has no receipt for the poll decision to age, so a poll's
    /// figure for it would sit in the report unrefreshed once the pushing
    /// ranks are fresh again; it stays out of the pushed report, and the
    /// poll's report stands for it while the worker is polled. A worker
    /// with no pushed record on file has no report to merge into; the
    /// poll's report stands on its own, as always. Returns the rows on file
    /// of the ranks that pushed since the poll began (the stream's, the
    /// newer) and, when there are any, the report on file's sample instant,
    /// for a publish that carries them over the poll's under their own
    /// sample boundary.
    pub(crate) fn apply_polled_load(
        &self,
        url: &str,
        load: &WorkerLoadResponse,
        polled_at: Instant,
    ) -> (Vec<SchedulerLoadSnapshot>, Option<Instant>) {
        // A rollup from a gateway registered as a worker is keyed by
        // downstream worker, not by rank: its entries all say rank 0 and
        // would collapse into one slot (the DP cache applies the same
        // guard).
        if !load.ranks_are_dp_ranks() {
            return (Vec::new(), None);
        }
        // Both tables held together so a record landing between the two
        // reads is not undone; pushes and retractions use the same lock order.
        let mut reports = self.pushed_reports.lock();
        let Some(report) = reports.get_mut(url) else {
            return (Vec::new(), None);
        };
        let pushed_at = self.pushed_at.lock();
        let receipts = pushed_at.get(url);
        let mut newer = Vec::new();
        for entry in &load.loads {
            let Ok(at) = report
                .loads
                .binary_search_by_key(&entry.dp_rank, |load| load.dp_rank)
            else {
                continue;
            };
            let pushed_since = receipts
                .and_then(|ranks| ranks.get(&entry.dp_rank))
                .is_some_and(|&at| at >= polled_at);
            if pushed_since {
                newer.push(report.loads[at].clone());
            } else {
                report.loads[at].clone_from(entry);
            }
        }
        let sampled_at = if newer.is_empty() {
            None
        } else {
            report.sampled_at
        };
        (newer, sampled_at)
    }

    /// One worker's polled report through the poll loop's own steps: an
    /// empty response's retraction, the merge over a pushed report, the
    /// overload verdict, the queue-depth read, the DP cache and the publish
    /// under the incarnation fence. The store ingress feeds polls here; a
    /// rank whose record arrived since the poll began keeps the stream's
    /// row in this publish too, as the pushed report keeps it, and a worker
    /// without a stream takes the one arbitration of its rank instants
    /// ([`Self::arbitrated_report`]), so a read through the host sees the
    /// one arbitration the sink's contract names.
    pub(crate) fn ingest_polled_report(
        &self,
        worker: &Arc<dyn Worker>,
        mut load: WorkerLoadResponse,
        polled_at: Instant,
    ) {
        let url = worker.url().to_string();
        if load.loads.is_empty() {
            self.apply_overload_verdict(worker, None, "poll");
            self.retract_pushed_load(&url, worker.dp_rank(), polled_at);
            if let Some(rank) = worker.dp_rank() {
                self.worker_load_manager.remove_rank(&url, rank);
            } else {
                self.worker_load_manager.remove_worker(&url);
            }
            self.load_state
                .publish_group(std::slice::from_ref(&url), Vec::new());
            return;
        }
        // A report that names its sample instant keeps it (the host's paths
        // do); one that does not is stamped now.
        let now = load.sampled_at.unwrap_or_else(Instant::now);
        if load.ranks_are_dp_ranks() && !self.pushed_reports.lock().contains_key(&url) {
            // No stream: the poll's rows, each at the poll's instant, over
            // the published rows under the worker's lock through the publish.
            let lock = self.rank_lock_of(&url);
            let mut instants = lock.lock();
            let ranks = load.dp_rank_count;
            let rows = load.loads.into_iter().map(|row| (row, now)).collect();
            let Some(load) = self.arbitrated_report(&url, &mut instants, rows, Some(ranks)) else {
                return;
            };
            self.publish_polled_report(worker, &url, load);
            return;
        }
        let (newer, stream_sampled_at) = self.apply_polled_load(&url, &load, polled_at);
        // One sample instant per report, the earlier of the two: the carried
        // rows keep their stream's, so the scorer releases no dispatch they
        // never saw; the poll's own rows cost their in-flight dispatches
        // once more until the next report, the safe side.
        load.sampled_at = Some(stream_sampled_at.map_or(now, |at| at.min(now)));
        for newer in newer {
            if let Some(entry) = load
                .loads
                .iter_mut()
                .find(|entry| entry.dp_rank == newer.dp_rank)
            {
                *entry = newer;
            }
        }
        self.publish_polled_report(worker, &url, Arc::new(load));
    }

    /// The publish of a polled report: the overload verdict, the queue-depth
    /// read, the DP cache, the gauges and the snapshot under the incarnation
    /// fence, every one reading the reconciled report.
    fn publish_polled_report(
        &self,
        worker: &Arc<dyn Worker>,
        url: &str,
        load: Arc<WorkerLoadResponse>,
    ) {
        self.apply_overload_verdict(worker, Some(&load), "poll");
        let waiting: i64 = load
            .loads
            .iter()
            .map(|rank| i64::from(rank.num_waiting_reqs))
            .sum();
        liveness::on_load_report(worker, waiting, None, false, None);
        let url = url.to_string();
        if load.has_absolute_token_data() && load.ranks_are_dp_ranks() {
            self.worker_load_manager
                .update_dp_loads(&HashMap::from([(url.clone(), load.dp_rank_loads())]));
        } else {
            self.worker_load_manager
                .remove_workers(std::slice::from_ref(&url));
        }
        Metrics::record_engine_load(&url, worker.model_id(), &load);
        self.load_state
            .publish_group(std::slice::from_ref(&url), vec![(Arc::clone(worker), load)]);
    }

    /// Retract pre-poll pushed ranks after an authoritative empty response.
    /// Records received since the poll began remain available for the next
    /// push; a read's rows, which carry no receipt, go with the pre-poll
    /// ones; a report left without ranks publishes nothing.
    fn retract_pushed_load(&self, url: &str, rank: Option<usize>, polled_at: Instant) {
        let mut reports = self.pushed_reports.lock();
        let mut receipts = self.pushed_at.lock();
        if let Some(times) = receipts.get_mut(url) {
            times.retain(|&dp_rank, at| {
                rank.is_some_and(|rank| usize::try_from(dp_rank) != Ok(rank)) || *at >= polled_at
            });
        }
        // A worker with no receipt at all keeps nothing: every row on file
        // is a read's.
        let times = receipts.get(url);
        if let Some(report) = reports.get_mut(url) {
            report
                .loads
                .retain(|load| times.is_some_and(|times| times.contains_key(&load.dp_rank)));
            report.dp_rank_count = i32::try_from(report.loads.len()).unwrap_or(i32::MAX);
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

    /// Publish every worker whose report changed since the last flush: the
    /// poll's wire shape is built here, once per window per worker, from the
    /// report its records updated in place. A report a poll's empty answer
    /// emptied has nothing to publish (that poll's publish pruned the entry).
    pub(crate) fn flush_pushed(&self) {
        let reports = self.pushed_reports.lock();
        let changed: Vec<(String, Arc<dyn Worker>)> = self.pending_pushed.lock().drain().collect();
        if changed.is_empty() {
            return;
        }
        let timestamp = chrono::Utc::now().to_rfc3339();
        let mut urls = Vec::with_capacity(changed.len());
        let mut fresh: Vec<PushedReport> = Vec::with_capacity(changed.len());
        for (url, worker) in changed {
            let Some(report) = reports
                .get(url.as_str())
                .filter(|report| !report.loads.is_empty())
            else {
                continue;
            };
            fresh.push((
                worker,
                Arc::new(WorkerLoadResponse {
                    timestamp: timestamp.clone(),
                    version: "pushed".to_string(),
                    dp_rank_count: report.dp_rank_count,
                    loads: report.loads.clone(),
                    aggregate: None,
                    sampled_at: report.sampled_at,
                }),
            ));
            urls.push(url);
        }
        if !fresh.is_empty() {
            self.load_state.publish_group(&urls, fresh);
        }
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
        // The rank instants go with the loads they describe: a worker whose
        // removal event the lag skipped would otherwise keep its lock and
        // instants for good.
        self.rank_instants.lock().clear();
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
    pub(crate) fn evict_worker_loads(&self, worker: &Arc<dyn Worker>) {
        let url = worker.url();
        // No load feed, no verdict: a flag left set would strand the worker
        // out of routing with nothing to ever clear it.
        self.worker_registry.set_worker_overloaded(worker, false);
        self.worker_load_manager.remove_worker(url);
        self.native_loads_memo.remove(url);
        // The pushed-record state goes with the worker: a replacement is
        // polled until its own stream pushes.
        self.pushed_reports.lock().remove(url);
        self.pushed_at.lock().remove(url);
        self.rank_instants.lock().remove(url);
        self.load_state.enqueue_eviction(Arc::clone(worker));
    }

    /// Apply every queued eviction in one snapshot rebuild, and sentinel the
    /// `smg_engine_*` series of workers whose own entry was removed. The
    /// sentinel matters for a worker that stays registered (its status left
    /// `Ready`, or a same-URL replacement has not published yet), whose series
    /// would otherwise read its last load as live. For a removed worker it is
    /// a no-op: the worker series store dropped the series with the
    /// registration. A same-URL replacement that already republished keeps
    /// its entry, and its series continues uninterrupted.
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
///
/// A tick reads the registry's immutable membership snapshot: one pointer
/// load while the fleet is stable, rebuilt once after a membership change,
/// so a worker registered after this loop started is swept on the next
/// tick. Walking the worker map instead took a read lock and boxed its guard
/// for every shard of the map on every tick, whether or not the shard held a
/// worker (the map has four shards per CPU the process sees, rounded up to a
/// power of two: 1,024 on a 192-CPU host, 4,096 small allocations a second
/// with the gateway idle).
async fn liveness_sweep_loop(monitor: Weak<WorkerMonitor>) {
    let mut ticker = tokio::time::interval(liveness::SWEEP_INTERVAL);
    loop {
        ticker.tick().await;
        let Some(monitor) = monitor.upgrade() else {
            return;
        };
        for worker in monitor.worker_registry.get_routing_workers().iter() {
            liveness::sweep(worker);
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
    if !monitor.loads_wanted(&workers) {
        debug!("Load monitoring disabled and nothing needs the data, skipping load fetch for group {group_key}");
        return;
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
mod tests;

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

    /// Recovery clears a worker's rank instants with its loads: a worker
    /// whose removal event the lag skipped keeps no lock or instants behind.
    #[tokio::test]
    async fn stop_all_groups_drops_the_rank_instants_of_a_skipped_removal() {
        let (registry, monitor) = monitor_with(PolicyConfig::RoundRobin, false);
        let worker = vllm_worker("http://w1:8000");
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let record = EngineLoad {
            running_requests: 1,
            sample: 1,
            ..Default::default()
        };
        monitor.apply_read_load(&worker, 0, &record, Instant::now());
        assert!(monitor.rank_instants.lock().contains_key(worker.url()));

        // The worker's removal event never reaches the monitor: no eviction
        // ran before the recovery.
        monitor.stop_all_groups();

        assert!(
            monitor.rank_instants.lock().is_empty(),
            "recovery drops the rank instants with the loads"
        );
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

    /// The one predicate behind the poll tick and the load stream: wanted
    /// by default; under `--disable-load-monitoring` only for a reason.
    #[tokio::test]
    async fn loads_wanted_follows_the_conditional_gate() {
        let (_registry, by_default) = monitor_with(PolicyConfig::Random, false);
        assert!(by_default.loads_wanted(&[vllm_worker("http://a:8000")]));
        let (_registry, conditional) = monitor_with(PolicyConfig::Random, true);
        assert!(!conditional.loads_wanted(&[vllm_worker("http://a:8000")]));
        assert!(
            conditional.loads_wanted(&[vllm_worker_with_overload(
                "http://a:8000",
                OverloadUpdate {
                    waiting_requests: Some(1),
                    ..Default::default()
                }
            )]),
            "overload protection on a member wants the loads"
        );
        let (_registry, load_aware) = monitor_with(
            PolicyConfig::PowerOfTwo {
                load_metric: Default::default(),
            },
            true,
        );
        assert!(load_aware.loads_wanted(&[vllm_worker("http://a:8000")]));
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
        // A port an earlier test's engine served on keeps that engine's
        // overrides; this engine starts from fresh ones.
        mock_worker::overrides::fresh_for_port(port);
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
    /// rank refreshes that rank's entry in the pushed report: the report
    /// the next record publishes carries the poll's figure for it, not
    /// the rank's last pushed record.
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

    /// The records of one window publish one report per worker, in the
    /// poll's shape: every rank on file in rank order, the `pushed` version,
    /// an RFC 3339 timestamp, the rank count and the last record's sample
    /// instant, built at the publish and not per record; a window with no
    /// record publishes nothing.
    #[tokio::test]
    async fn a_windows_records_publish_one_report_in_the_polls_shape() {
        let (registry, monitor) = monitor_with_policy(PolicyConfig::RoundRobin);
        let worker = vllm_worker("http://w1:8000");
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let t0 = Instant::now();
        let record = |running: u32, sample: u64| EngineLoad {
            running_requests: running,
            sample,
            ..Default::default()
        };
        monitor.apply_pushed_load(&worker, 1, &record(8, 1), t0);
        let last = t0 + Duration::from_millis(5);
        monitor.apply_pushed_load(&worker, 0, &record(2, 2), last);
        assert_eq!(monitor.pending_pushed_len(), 1, "one publish per worker");
        monitor.flush_pushed();
        let snapshot = monitor.load_state.snapshot();
        let (report, sampled_at) = snapshot.report(worker.url()).expect("published");
        assert_eq!(report.version, "pushed");
        assert!(
            chrono::DateTime::parse_from_rfc3339(&report.timestamp).is_ok(),
            "{}",
            report.timestamp
        );
        assert_eq!(report.dp_rank_count, 2);
        let ranks: Vec<(i32, i32)> = report
            .loads
            .iter()
            .map(|rank| (rank.dp_rank, rank.num_running_reqs))
            .collect();
        assert_eq!(ranks, vec![(0, 2), (1, 8)]);
        assert_eq!(sampled_at, last - PUSHED_ONE_WAY_MARGIN);
        assert_eq!(monitor.pending_pushed_len(), 0);
        monitor.flush_pushed();
        assert!(
            Arc::ptr_eq(&snapshot, &monitor.load_state.snapshot()),
            "no record since the publish: no rebuild"
        );
    }

    /// A poll's publish that carries a stream row over the poll's keeps the
    /// stream's sample instant: a dispatch between that sample and the poll
    /// is still in flight for the scorer, so the report is not stamped with
    /// the poll's later time.
    #[tokio::test]
    async fn a_reconciled_poll_publishes_the_carried_stream_rows_sample_instant() {
        let (registry, monitor) = monitor_with_policy(PolicyConfig::RoundRobin);
        let worker = vllm_worker("http://w1:8000");
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let t0 = Instant::now();
        let record = |running: u32, age_ms: u32, sample: u64| EngineLoad {
            running_requests: running,
            age_ms,
            sample,
            ..Default::default()
        };
        // Both ranks on file from the stream; then a poll begins.
        monitor.apply_pushed_load(&worker, 0, &record(2, 0, 1), t0);
        monitor.apply_pushed_load(&worker, 1, &record(3, 0, 2), t0);
        let polled_at = t0 + Duration::from_millis(10);
        // Rank 0's record arrives after the poll began, sampled 100 ms before
        // its receipt.
        let received = polled_at + Duration::from_millis(5);
        monitor.apply_pushed_load(&worker, 0, &record(4, 100, 3), received);
        let stream_sample = received - Duration::from_millis(100) - PUSHED_ONE_WAY_MARGIN;
        let poll = WorkerLoadResponse {
            dp_rank_count: 2,
            loads: (0..2)
                .map(|dp_rank| SchedulerLoadSnapshot {
                    dp_rank,
                    num_running_reqs: 7,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        monitor.ingest_polled_report(&worker, poll, polled_at);
        let snapshot = monitor.load_state.snapshot();
        let (report, sampled_at) = snapshot.report(worker.url()).expect("published");
        let ranks: Vec<(i32, i32)> = report
            .loads
            .iter()
            .map(|rank| (rank.dp_rank, rank.num_running_reqs))
            .collect();
        assert_eq!(
            ranks,
            vec![(0, 4), (1, 7)],
            "rank 0 keeps the stream's row, rank 1 takes the poll's"
        );
        assert_eq!(
            sampled_at, stream_sample,
            "the carried row's sample instant, not the poll's"
        );
    }

    /// A record for a rank on file replaces the rank's entry in place: the
    /// figures are the record's, the telemetry it left out stays what the
    /// last record that carried it said, the report keeps one entry per
    /// rank, and the DP-rank token cache takes the report's counts.
    #[tokio::test]
    async fn a_record_for_a_rank_on_file_replaces_its_entry_in_place() {
        let (registry, monitor) = monitor_with_policy(PolicyConfig::RoundRobin);
        let worker = vllm_worker("http://w1:8000");
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let t0 = Instant::now();
        let heartbeat = EngineLoad {
            running_requests: 2,
            cache_hit_rate: Some(0.75),
            num_used_tokens: Some(1024),
            max_total_num_tokens: Some(4096),
            sample: 1,
            ..Default::default()
        };
        monitor.apply_pushed_load(&worker, 0, &heartbeat, t0);
        let core = EngineLoad {
            running_requests: 5,
            num_used_tokens: Some(2048),
            sample: 2,
            ..Default::default()
        };
        monitor.apply_pushed_load(&worker, 0, &core, t0 + Duration::from_millis(1));
        monitor.flush_pushed();
        let snapshot = monitor.load_state.snapshot();
        let report = snapshot.get(worker.url()).expect("published");
        let entries: Vec<(i32, i32, f64, i32, i32)> = report
            .loads
            .iter()
            .map(|rank| {
                (
                    rank.dp_rank,
                    rank.num_running_reqs,
                    rank.cache_hit_rate,
                    rank.num_used_tokens,
                    rank.max_total_num_tokens,
                )
            })
            .collect();
        assert_eq!(entries, vec![(0, 5, 0.75, 2048, 4096)]);
        let cached = monitor.worker_load_manager.dp_cached_loads.read();
        assert_eq!(
            cached.get(worker.url()).and_then(|ranks| ranks.get(&0)),
            Some(&2048)
        );
    }
}
