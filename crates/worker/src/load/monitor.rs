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
//!   derived data; full rebuild is the recovery mechanism. A worker
//!   that pushed its loads and left in the lost window is evicted
//!   there, as its event would have.
//!
//! ## Polling
//!
//! Each group runs a single `tokio::time::interval` loop. Every tick:
//!
//! 1. Fetch the group's `Ready` workers, less those the routing side asks
//!    for no load ([`LoadConsumer::polls`]). Polling is unconditional by
//!    default; with `--disable-load-monitoring` the tick is skipped
//!    unless a load-aware policy, engine-metrics re-export, or overload
//!    protection on a group member needs the data (the original
//!    conditional gate — never "never poll").
//! 2. Fetch loads concurrently from each of them that its load stream
//!    does not serve.
//! 3. Update load-aware policies and the DP cache.
//! 4. Atomically clear stale entries for the group from the watch
//!    channel and merge in the fresh loads.
//!
//! A group's interval is the smallest its workers state for themselves, in
//! whole seconds, and the monitor's own where none states one
//! ([`WorkerMonitor::with_interval`]). It is never under
//! [`LEAST_POLL_INTERVAL`]. A tick that a slow poll overran is skipped: the
//! loop polls once when that poll returns, and next on the interval's own
//! beat.
//!
//! ## The port
//!
//! What a pushed record or a poll's answer means for routing crosses the
//! port of [`super::port`]: the monitor is the port's collector. It says
//! each fact to a [`LoadSink`] and keeps the report a worker, with its
//! gauges, liveness and the published snapshot. The sink
//! [`WorkerMonitor::new`] builds, in `classic_sink`, keeps the receipts of
//! the pushed records, the overload verdict, the cache of rank loads and
//! the trim of what the routing side booked.

use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use dashmap::DashMap;
use futures::future;
use parking_lot::{Mutex, RwLock};
use reqwest::StatusCode;
use rustc_hash::FxHashMap;
use tokio::{
    sync::{broadcast, Notify},
    task::JoinHandle,
    time::MissedTickBehavior,
};
use tracing::{debug, info, warn};

use crate::{
    connection::engine::LoadSource,
    health::liveness,
    identity::{
        basic::{ConnectionMode, Worker},
        endpoint::Endpoint,
    },
    load::{
        counts::{FetchFailure, PollCounts, PollTotals},
        port::{GroupTick, LoadSink, PollMode, PolledMember, PolledRanks, PolledReport},
        state::{LoadReceiver, LoadSnapshot, LoadState},
    },
    metrics::{
        intern_model_label,
        writers::{registered_engine_role, LoadTest, Metrics},
    },
    registry::{event::WorkerEvent, worker_model_ids, Generation, WorkerRegistry},
    value_bounds,
    worker::{
        LoadRecord, LoadTelemetry, RuntimeType, SchedulerLoadSnapshot, WorkerGroupKey,
        WorkerLoadResponse, WorkerStatus,
    },
};

mod classic_sink;
mod scrape;

use classic_sink::ClassicLoadSink;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// The rank loads by worker as the update line writes them: each address
/// masked. A worker's address may hold a credential in its userinfo or a
/// token in its query. The address is read when the line is written, so
/// nothing is parsed while debug logging is off.
struct MaskedLoads<'a>(&'a HashMap<String, HashMap<isize, isize>>);

impl Debug for MaskedLoads<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(
                self.0
                    .iter()
                    .map(|(url, ranks)| (Endpoint::redacted_str(url), ranks)),
            )
            .finish()
    }
}

/// DP rank load cache used by load-aware routing policies.
///
/// Pure in-memory data structure with no I/O. `WorkerMonitor` owns the
/// shared `Arc<WorkerLoadManager>` and updates it on every successful
/// poll; routing policies read from it via
/// `select_and_increment_lowest_dp_load`.
///
/// Only a poll feeds it: a worker served by its load stream is not polled,
/// so its entry is its last poll's, however old, and a reader for a worker
/// that streams brings its own feed.
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
    pub(crate) fn update_dp_loads(&self, loads: &HashMap<String, HashMap<isize, isize>>) {
        debug!(
            "WorkerLoadManager update_dp_loads map:{:?}",
            MaskedLoads(loads)
        );
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

    pub(crate) fn remove_workers(&self, urls: &[String]) {
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

    /// Drop every cached DP load entry. Called by the monitor's own sink at
    /// the monitor's reset ([`LoadSink::reset`]), so the cache cannot hand
    /// out stale per-rank loads after a full reconcile.
    pub fn clear(&self) {
        self.dp_cached_loads.write().clear();
    }
}

/// The token counts the DP-rank cache takes from a poll's `report`: one per
/// rank whose row carries token capacity. A row without it is no token
/// count of its rank, whatever the rows beside it carry: a poll's report
/// under the stream's rows holds rows of more than one origin.
fn dp_rank_token_loads(report: &WorkerLoadResponse) -> HashMap<isize, isize> {
    report
        .loads
        .iter()
        .filter(|rank| rank.max_total_num_tokens > 0)
        .map(|rank| (rank.dp_rank as isize, rank.num_used_tokens as isize))
        .collect()
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
    /// The backend answered success with a body that is no load response,
    /// or with one whose read failed: with why. What that says of the route
    /// depends on whether it ever answered.
    Unreadable(FetchFailure),
    /// Nothing was learned about the endpoint — retry on the next tick. With
    /// why: the read's own failure.
    Inconclusive(FetchFailure),
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

/// The routing side as the load monitor sees it: whoever reads the loads the
/// monitor publishes and books requests on workers. The gateway's policy
/// registry implements it; the monitor names no policy.
pub trait LoadConsumer: Send + Sync {
    /// Whether a reader wants worker loads, so the monitor has to fetch them.
    fn needs_loads(&self) -> bool;

    /// Trim what the readers booked on `worker` to the requests the router
    /// still holds there.
    fn reconcile_in_flight(&self, worker: &dyn Worker);

    /// Whether the monitor asks `worker` for its load. A worker the routing
    /// side says no of is no member of a tick: no poll goes to it, no load
    /// stream is wanted for it, and a tick judges and trims nothing on it.
    /// Yes by default. A host that cannot dial every worker of its registry
    /// says which it can. A record pushed for a worker it says no of is
    /// still taken and judged: the push asks nothing here.
    ///
    /// The answer is read at each tick. A worker whose answer turns from
    /// yes to no keeps its last published report and its last verdict: a
    /// tick prunes and judges its members alone. Both stand until the
    /// monitor evicts the worker's loads, as at its removal, or resets. A
    /// host that answers by a fact of the worker's life leaves neither
    /// behind.
    fn polls(&self, _worker: &dyn Worker) -> bool {
        true
    }
}

/// A worker's pushed report on file, with the incarnation whose record
/// started it. A reset holds the report to that worker: one that left, was
/// replaced or left `Ready` while registry events were lost is evicted
/// there ([`WorkerMonitor::evict_lost_pushers`]). The pusher was the
/// registry's worker at the URL when its record started the report,
/// `Ready` or not, and the report holds that worker's records alone: a
/// record of any other worker starts it again or is dropped.
struct PushedFile {
    report: WorkerLoadResponse,
    pusher: Arc<dyn Worker>,
    /// When each row of the report last had each of its two fixed token
    /// counts stated: one statement a row, in the rows' order. A row comes
    /// and goes with its statement ([`Self::retain_rows`]).
    counts_stated: Vec<CountsStated>,
    /// A record came while the pusher was not `Ready`: its rank is on file
    /// and its engine series is held. The file stays in the flush's queue
    /// until a flush finds the pusher `Ready`: that one writes every
    /// rank's series, publishes the report and clears the mark.
    series_held: bool,
    /// The pusher stands in the flush's queue: a record of the same window
    /// queues it once.
    queued: bool,
}

impl PushedFile {
    /// An empty report that the first record of `pusher` starts.
    fn started_by(pusher: &Arc<dyn Worker>) -> Self {
        Self {
            report: WorkerLoadResponse::default(),
            pusher: Arc::clone(pusher),
            counts_stated: Vec::new(),
            series_held: false,
            queued: false,
        }
    }

    /// Keeps the rows `keep` says to, each with the statement of its two
    /// counts: a row that goes takes its statement along, so the two stay
    /// in one order. `keep` may write the row and its statement.
    fn retain_rows(
        &mut self,
        mut keep: impl FnMut(&mut SchedulerLoadSnapshot, &mut CountsStated) -> bool,
    ) {
        let stated = &mut self.counts_stated;
        let (mut read, mut kept) = (0, 0);
        self.report.loads.retain_mut(|row| {
            let mut own = stated.get(read).copied().unwrap_or_default();
            read += 1;
            let stays = keep(row, &mut own);
            if stays {
                if let Some(slot) = stated.get_mut(kept) {
                    *slot = own;
                }
                kept += 1;
            }
            stays
        });
        stated.truncate(kept);
    }

    /// One rank of the report goes to its engine series. A file is reached
    /// through the lock of the reports alone, so every live write of a
    /// pusher's series is made under that lock, where the sentinel is
    /// written too ([`WorkerMonitor::clear_series`]).
    fn write_rank_series(&self, at: usize, model_id: &str) {
        Metrics::record_engine_rank_load(
            self.pusher.as_ref(),
            model_id,
            registered_engine_role(*self.pusher.worker_type()),
            &self.report.loads[at],
        );
    }
}

/// When each of the two fixed token counts of one rank's row was last
/// stated: the sample instant of the record of the stream that stated it,
/// or of the poll's answer laid on the row. A record that leaves a count
/// out keeps it, and says by these how old it is
/// ([`SchedulerLoadSnapshot::kept_used_tokens_age_ms`]).
///
/// Held for each row of a file, in a report of one rank and of several. A
/// reader dates a kept count by its row's own sample instant less the
/// count's age, and the row states that instant beside the age
/// ([`SchedulerLoadSnapshot::sampled_at`]): no other rank's record moves
/// it. A row that is new on the file has had nothing stated.
#[derive(Clone, Copy, Default)]
struct CountsStated {
    used: Option<Instant>,
    total: Option<Instant>,
}

impl CountsStated {
    /// Both counts stated at `at`: a poll's answer states every figure of
    /// its entry.
    fn both(at: Instant) -> Self {
        Self {
            used: Some(at),
            total: Some(at),
        }
    }

    /// Dates the two counts of `entry`, the row a record sampled at `at`
    /// has just written.
    fn date(&mut self, entry: &mut SchedulerLoadSnapshot, at: Instant) {
        let kept = entry.kept;
        let used = kept.contains(LoadTelemetry::USED_TOKENS);
        let total = kept.contains(LoadTelemetry::TOTAL_TOKENS);
        entry.kept_used_tokens_age_ms = age_at(&mut self.used, used, at);
        entry.kept_total_tokens_age_ms = age_at(&mut self.total, total, at);
    }
}

/// How old a count is at `at`, the sample of the record that wrote its
/// entry, in milliseconds. A count the record states (`kept` is false) is
/// stated at `at` from here on and has no age. A count the entry keeps is
/// as old as the time since its statement, rounded up to a whole
/// millisecond: a reader ends the count by this age, and a count must
/// never read younger than it is. The time is taken whole at each record
/// and never summed, so no rounding adds up over the records between two
/// statements. `None` where no statement is on file.
fn age_at(stated: &mut Option<Instant>, kept: bool, at: Instant) -> Option<u32> {
    if !kept {
        *stated = Some(at);
        return None;
    }
    let since = at.saturating_duration_since((*stated)?);
    Some(u32::try_from(since.as_nanos().div_ceil(1_000_000)).unwrap_or(u32::MAX))
}

/// The rank of each entry `report` holds at or past `width`, its worker's
/// own width: a report taken off the file, or the load of a row the batch
/// removed. A rollup's row names a rank once for each downstream worker
/// that holds it, so a rank can come more than once. The sentinel of the
/// worker's series covers the width and each of these ranks once, and no
/// rank between.
fn ranks_past(report: &WorkerLoadResponse, width: usize) -> impl Iterator<Item = i32> + '_ {
    report
        .loads
        .iter()
        .map(|load| load.dp_rank)
        .filter(move |rank| usize::try_from(*rank).is_ok_and(|rank| rank >= width))
}

/// Set `worker`'s `smg_engine_*` series to the sentinel, rank by rank, for
/// its own width and each rank `ended` holds past it: a rank's load series
/// under each of the worker's models, and its pair series, which carry no
/// model. `ended` is the report or the row that ends with the worker's
/// feed, where there is one. A series a live writer writes is left: the
/// writer asks `live` whether a live writer holds an entry of the rank
/// that passes the series' test, under a model's label or, given none,
/// under any ([`Metrics::remove_engine_rank_metrics`]). Its callers hold
/// the lock of the reports, under which every series is written.
#[cold]
fn write_sentinel(
    worker: &Arc<dyn Worker>,
    ended: Option<&WorkerLoadResponse>,
    live: impl Fn(Option<&str>, i32, &LoadTest<'_>) -> bool,
) {
    let url = worker.url();
    let width = worker.dp_size().unwrap_or(1).max(1);
    let mut ranks: Vec<i32> = (0..width)
        .filter_map(|rank| i32::try_from(rank).ok())
        .chain(
            ended
                .into_iter()
                .flat_map(|report| ranks_past(report, width)),
        )
        .collect();
    // An ended row can name a far rank more than once: one pass a rank.
    ranks.sort_unstable();
    ranks.dedup();
    // A worker can serve multiple models (one load group per model),
    // so sentinel every model's series — not just the primary.
    let models = worker_model_ids(worker);
    // The worker's load series carry the role it is registered in.
    let role = registered_engine_role(*worker.worker_type());
    for rank in ranks {
        Metrics::remove_engine_rank_metrics(url, &models, rank, role, |label, test| {
            live(label, rank, test)
        });
    }
}

/// Load monitoring service that subscribes to `WorkerRegistry` events.
pub struct WorkerMonitor<G> {
    worker_registry: Arc<WorkerRegistry<G>>,
    /// Where the monitor's facts go ([`LoadSink`]): the sink [`Self::new`]
    /// builds over the registry and the routing side, or the one a host
    /// hands to [`Self::with_sink`].
    sink: Arc<dyn LoadSink>,
    /// The cache of rank loads that the sink [`Self::new`] builds writes.
    /// Under a host's sink nothing writes it and it stays empty.
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
    load_state: Arc<LoadState<G>>,
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
    /// pushed nothing since the poll began ([`Self::lay_polled`]).
    /// A record's readers borrow it, and the coalesced publish copies it
    /// once per window, so a record builds no report of its own. This map
    /// is keyed by the worker's URL, a string of the gateway's own, and
    /// every record looks its worker up in it: it hashes with FxHash. The
    /// sink is called under this map's lock at a record and at a poll's
    /// answer: the lock rule of [`LoadSink`]. Each report stands with the
    /// incarnation that started it ([`PushedFile`]).
    pushed_reports: Mutex<FxHashMap<String, PushedFile>>,
    /// The workers whose report changed since the last coalesced publish,
    /// each the incarnation that pushed, for the publish fence. A file's
    /// mark keeps its pusher here once a window, so a record hashes no URL
    /// and copies no key for it.
    pending_pushed: Mutex<Vec<Arc<dyn Worker>>>,
    pushed_notify: Arc<Notify>,
    pushed_flush_task: Mutex<Option<JoinHandle<()>>>,
    /// What the polls did, for a host's read ([`Self::poll_totals`]). A
    /// group's tick adds to it, and nothing else does.
    poll_counts: Arc<PollCounts>,
    /// The sink [`Self::new`] built, for the cases that read its receipts.
    #[cfg(test)]
    classic: Option<Arc<ClassicLoadSink<G>>>,
}

/// A worker's published copy of its pushed report, with the worker it came
/// from for the publish fence.
type PushedReport = (Arc<dyn Worker>, Arc<WorkerLoadResponse>);

/// One engine's answer of a tick: its polled members, each with its own
/// report of it, `None` where the fetch failed.
type EngineAnswer = Vec<(Arc<dyn Worker>, Option<WorkerLoadResponse>)>;

/// Telemetry a record does not carry (an event batch's record is the core
/// only; a heartbeat's and a stream's first carry it all) stays what the
/// last record that carried it said, so the gauges and `GET /loads` keep
/// the engine's figures between heartbeats. Absent is absent: a section the
/// engine never reports stays unset. `previous` is the rank's entry the
/// record replaces; what it keeps moves out of it. `carried` is the
/// record's own word on what it carries. The mark of what the record did
/// not state (`SchedulerLoadSnapshot::kept`) is the fresh snapshot's own,
/// written at its making, and nothing here sets the mark. How old a kept
/// token count is, which the entry offers by name with it, is written
/// behind this call ([`CountsStated::date`]).
fn keep_telemetry(
    fresh: &mut SchedulerLoadSnapshot,
    previous: SchedulerLoadSnapshot,
    carried: LoadTelemetry,
) {
    if !carried.contains(LoadTelemetry::CACHE_HIT_RATE) {
        fresh.cache_hit_rate = previous.cache_hit_rate;
    }
    if !carried.contains(LoadTelemetry::USED_TOKENS) {
        fresh.num_used_tokens = previous.num_used_tokens;
    }
    if !carried.contains(LoadTelemetry::TOTAL_TOKENS) {
        fresh.max_total_num_tokens = previous.max_total_num_tokens;
    }
    if !carried.contains(LoadTelemetry::MEMORY) {
        fresh.memory = previous.memory;
    }
    if !carried.contains(LoadTelemetry::QUEUES) {
        fresh.queues = previous.queues;
    }
    if !carried.contains(LoadTelemetry::DISAGGREGATION) {
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

impl<G> Debug for WorkerMonitor<G> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerMonitor")
            .field("default_interval", &self.default_interval)
            .finish_non_exhaustive()
    }
}

impl<G: Generation> WorkerMonitor<G> {
    /// Construct a `WorkerMonitor` without spawning any background work.
    ///
    /// The caller must invoke [`Self::start_event_loop`] once initial
    /// workers have been registered. Until then, `worker_load_manager`
    /// is still safe to read; it just stays empty.
    ///
    /// The sink this builds writes its overload vetoes on the registry's
    /// workers. The monitor's drop says no reset, so a host that drops the
    /// monitor and keeps the registry takes the vetoes off with
    /// [`WorkerRegistry::clear_all_overload_flags`].
    pub fn new(
        worker_registry: Arc<WorkerRegistry<G>>,
        load_consumer: Arc<impl LoadConsumer + 'static>,
        default_interval_secs: u64,
        engine_metrics: bool,
        conditional_polling: bool,
    ) -> Self {
        let worker_load_manager = Arc::new(WorkerLoadManager::new());
        let classic = Arc::new(ClassicLoadSink::new(
            Arc::clone(&worker_registry),
            load_consumer,
            Arc::clone(&worker_load_manager),
        ));
        #[cfg(test)]
        let kept = Arc::clone(&classic);
        let mut monitor = Self::with_sink(
            worker_registry,
            classic,
            default_interval_secs,
            engine_metrics,
            conditional_polling,
        );
        monitor.worker_load_manager = worker_load_manager;
        #[cfg(test)]
        {
            monitor.classic = Some(kept);
        }
        monitor
    }

    /// [`Self::new`] over the sink a host hands: the monitor says its facts
    /// to `sink` ([`LoadSink`]) and keeps the report a worker, the gauges,
    /// liveness and the published snapshot, which it goes on publishing.
    /// `worker_load_manager` stays empty: it is the cache of the sink
    /// [`Self::new`] builds.
    pub fn with_sink(
        worker_registry: Arc<WorkerRegistry<G>>,
        sink: Arc<dyn LoadSink>,
        default_interval_secs: u64,
        engine_metrics: bool,
        conditional_polling: bool,
    ) -> Self {
        let load_state = Arc::new(LoadState::new(Arc::clone(&worker_registry)));
        Self {
            worker_registry,
            sink,
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
            pushed_reports: Mutex::new(FxHashMap::default()),
            pending_pushed: Mutex::new(Vec::new()),
            pushed_notify: Arc::new(Notify::new()),
            pushed_flush_task: Mutex::new(None),
            poll_counts: Arc::new(PollCounts::default()),
            #[cfg(test)]
            classic: None,
        }
    }

    /// The monitor's own interval as a duration: what a group polls at where
    /// none of its workers states an interval for itself. [`Self::new`] and
    /// [`Self::with_sink`] take it in whole seconds, raised to one. This
    /// takes any duration and raises one under [`LEAST_POLL_INTERVAL`] to
    /// it. It takes the monitor by value: the interval is said before the
    /// monitor is shared and its loops start.
    #[must_use]
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.default_interval = interval.max(LEAST_POLL_INTERVAL);
        self
    }

    /// The monitor's own interval: what a group polls at where none of its
    /// workers states one for itself.
    pub fn interval(&self) -> Duration {
        self.default_interval
    }

    /// What the monitor's polls did since it was built, as plain numbers:
    /// for a host that renders series of its own and cannot read the
    /// process recorder's. The monitor's own series are written as before,
    /// and each number here moves in a group's tick alone.
    pub fn poll_totals(&self) -> PollTotals {
        self.poll_counts.totals()
    }

    /// A load record pushed on a worker's KV-event stream (`KvEventBatch.load`,
    /// the servicer's scheduler report for `dp_rank`, received at
    /// `received_at`). Its fixed fields have the shape of a poll. It replaces
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
    /// state, and before any reader of the record is called. A record that
    /// finds no report, or the report of another worker, starts one for the
    /// registry's worker at the URL alone, whatever its status: the sink
    /// and the wedged rule hear a registered worker that is not `Ready`,
    /// and its rows and its series are held back for the flush that finds
    /// it `Ready`. Any other is dropped ahead of the report, the sink, the
    /// series and the publish. Its worker is still marked as one whose
    /// stream pushes, as for any record. A record of a rank the report does
    /// not consider is dropped the same way, its worker marked the same.
    pub fn apply_pushed_load<R: LoadRecord>(
        self: &Arc<Self>,
        worker: &Arc<dyn Worker>,
        dp_rank: i32,
        record: &R,
        received_at: Instant,
    ) {
        if dp_rank < 0
            || worker
                .dp_rank()
                .is_some_and(|rank| usize::try_from(dp_rank) != Ok(rank))
        {
            return;
        }
        // An age the margin overflows, or one older than the clock, reads as
        // sampled at the receipt.
        let sampled_at = record
            .age()
            .checked_add(PUSHED_ONE_WAY_MARGIN)
            .and_then(|age| received_at.checked_sub(age))
            .unwrap_or(received_at);
        let url = worker.url();
        // The stream pushes: silence from this worker is judged by the
        // stream's cadence from now on (`liveness`, the stale rule).
        let runtime = worker.runtime_state();
        if let Some(state) = &runtime {
            state.note_load_record();
        }
        // A rank the report does not consider: the record said the stream
        // works, above, and is dropped ahead of the report, the sink and
        // the series.
        if !value_bounds::considers(worker.dp_rank(), dp_rank) {
            return;
        }
        // A worker's entry is made on its first record; the lookup that
        // finds it copies no key.
        let mut reports = self.pushed_reports.lock();
        // The status is read under this hold, for the series below. A
        // leave's eviction takes the hold behind the status's change, and
        // writes the sentinel there with the report it removes: a record
        // behind that eviction reads the change, and one ahead of it has
        // its series under that sentinel.
        let ready = runtime
            .as_deref()
            .map_or_else(|| worker.status(), |state| state.status())
            == WorkerStatus::Ready;
        let mut on_file = reports
            .get_mut(url)
            .filter(|file| Arc::ptr_eq(&file.pusher, worker));
        if on_file.is_none() {
            // A report is on file for the registry's worker at its URL
            // alone, by identity, and holds that worker's records alone.
            // Its status is not read: a worker that is not `Ready` still
            // feeds the sink and the wedged rule, and the publish fence
            // holds its rows back. A host ends a worker's stream after the
            // registry let the worker go, and starts its successor's with
            // no word to the monitor. So a record can land behind its
            // worker's eviction or ahead of its predecessor's: it starts a
            // report for the registry's worker and is dropped for any other.
            if !self.worker_registry.is_current_handle(worker) {
                return;
            }
            let started = PushedFile::started_by(worker);
            if let Some(left) = reports.insert(url.to_string(), started) {
                // The report replaced was a worker's whose eviction is still
                // to come. The sink forgets it here, under this hold, so the
                // receipts are this worker's alone. Its series go with it,
                // every rank it held: this record writes its own rank
                // below, and the others read the sentinel until this
                // worker reports them. A series that a poll of this worker
                // wrote, with the row it published, is not cleared.
                self.sink.forget(&left.pusher);
                self.clear_series(&reports, &left.pusher, Some(&left.report));
            }
            on_file = reports.get_mut(url);
        }
        // On file: made just above when it was not.
        let Some(file) = on_file else {
            return;
        };
        let report = &mut file.report;
        // What a poll would have put in the rank's entry, with the telemetry
        // the record left out (an event batch's record is the core only)
        // kept from the last record that carried it.
        // The record's named values are its own: the entry's list is this
        // record's, and none of the last record's is kept.
        let mut fresh = record.snapshot();
        fresh.dp_rank = dp_rank;
        // The entry states its own sample. The report's instant is its
        // last record's, of whatever rank.
        fresh.sampled_at = Some(sampled_at);
        // The values the record named and its snapshot does not hold were
        // dropped at the snapshot's bounds. They are counted here, where
        // the record and its snapshot stand together.
        let dropped = record.values_sent().saturating_sub(fresh.values.len());
        if dropped != 0 {
            Metrics::record_engine_load_values_dropped(url, dropped);
        }
        let at = match report
            .loads
            .binary_search_by_key(&dp_rank, |load| load.dp_rank)
        {
            Ok(at) => {
                let previous = std::mem::replace(&mut report.loads[at], fresh);
                keep_telemetry(&mut report.loads[at], previous, record.telemetry());
                at
            }
            Err(at) => {
                report.loads.insert(at, fresh);
                report.dp_rank_count = i32::try_from(report.loads.len()).unwrap_or(i32::MAX);
                // A row that is new on the file has had nothing stated.
                file.counts_stated.insert(at, CountsStated::default());
                at
            }
        };
        // A token count the entry keeps is offered by name as old as its
        // last statement, for each rank of the report: the age counts back
        // from the entry's own sample.
        if let Some(stated) = file.counts_stated.get_mut(at) {
            stated.date(&mut report.loads[at], sampled_at);
        }
        report.sampled_at = Some(sampled_at);
        // The sink takes the receipt and judges the report where it is,
        // under this hold of the reports: its receipt and the row stand
        // together.
        self.sink
            .record(worker, dp_rank, &*report, received_at, sampled_at);
        liveness::on_load_report(
            worker,
            runtime.as_deref(),
            report.total_waiting_reqs(),
            record.engine_silence(),
            record.reports_every_step(),
            record.prefill_pending(),
        );
        // The record feeds its own rank's `smg_engine_*` series through the
        // handles kept for it; a poll's report feeds every rank's. The
        // series is held back with the rows: a worker that is not `Ready`
        // keeps the sentinel its eviction wrote, so a stream that ends
        // there leaves no load that reads as live. Its file is marked, and
        // the flush that finds the worker `Ready` writes the held series.
        if ready {
            file.write_rank_series(at, worker.model_id());
        } else {
            file.series_held = true;
        }
        // A record feeds the DP-rank token cache nothing. The cache's one
        // reader is the HTTP pair router's rank pick, and a stream is opened
        // for a gRPC worker alone: no pick reads a pushing worker's entry. A
        // poll feeds the cache (`poll_group_once`).
        //
        // The window's publish takes the report once, under the incarnation
        // that pushed last: the file's pusher, queued once a window.
        if !file.queued {
            file.queued = true;
            self.pending_pushed.lock().push(Arc::clone(worker));
        }
        drop(reports);
        self.start_pushed_flusher();
        self.pushed_notify.notify_one();
    }

    /// Whether `workers`, the members of one group, are to have live load
    /// state: always by default; under `--disable-load-monitoring` only when
    /// a load-aware or dp-rank policy, the engine-metrics export or overload
    /// protection on a member needs it. Never where the host asks none of
    /// them for its load ([`LoadConsumer::polls`]), and a member it leaves
    /// out counts for nothing: its overload protection asks for no load.
    /// The poll tick and the decision to open a worker's load stream read
    /// the same answer.
    pub fn loads_wanted(&self, workers: &[Arc<dyn Worker>]) -> bool {
        let mut asked = workers
            .iter()
            .filter(|worker| self.sink.polls(worker.as_ref()))
            .peekable();
        if !workers.is_empty() && asked.peek().is_none() {
            return false;
        }
        if !self.conditional_polling {
            return true;
        }
        let routing_needs_load = self.sink.needs_loads();
        let overload_needs_load = asked.any(|w| w.metadata().overload.is_enabled());
        routing_needs_load || self.engine_metrics || overload_needs_load
    }

    /// The poll decision of the sink [`Self::new`] built, by URL, for the
    /// cases.
    #[cfg(test)]
    pub(crate) fn poll_mode(&self, url: &str, now: Instant, interval: Duration) -> PollMode {
        self.classic.as_ref().map_or(PollMode::Poll, |classic| {
            classic.poll_mode_of(url, now, interval)
        })
    }

    /// Whether the sink [`Self::new`] built holds a receipt for `rank` of
    /// `url`, for the cases.
    #[cfg(test)]
    pub(crate) fn has_receipt(&self, url: &str, rank: i32) -> bool {
        self.classic
            .as_ref()
            .is_some_and(|classic| classic.has_receipt(url, rank))
    }

    /// A poll's answer for one worker, for the cases: the two halves the
    /// tick runs for an engine ([`Self::reconcile_engine`]), the sink's
    /// merge and the file's lay, under one hold of the reports.
    #[cfg(test)]
    pub(crate) fn apply_polled_load(
        &self,
        url: &str,
        load: &WorkerLoadResponse,
        polled_at: Instant,
    ) -> (Vec<SchedulerLoadSnapshot>, Option<Instant>) {
        let mut reports = self.pushed_reports.lock();
        let mut ranks = PolledRanks::default();
        if let Some(classic) = self.classic.as_ref() {
            let on_file = reports.get(url).map(|file| &file.report);
            classic.merge(url, load, on_file, polled_at, &mut ranks);
        }
        Self::lay_polled(&mut reports, url, load, &ranks, polled_at)
    }

    /// The file's half of a poll's answer with rows for one member (`load`,
    /// the report of `url`), by the ranks the sink named for the member's
    /// engine: the row of a rank whose stream is the newer stays and is
    /// handed back, any other takes the answer's entry for it, and the row
    /// of a rank that leaves is dropped, so the worker's next pushed record
    /// does not publish it again. A worker with no pushed record on file has
    /// no report to lay: the poll's report stands on its own. Returns the
    /// rows of the newer ranks and, when there are any, the report on file's
    /// sample instant, for a publish that carries them over the poll's under
    /// their own sample boundary. `polled_at` is the instant the poll's
    /// report is dated at: a row that takes the answer's entry is sampled
    /// then ([`SchedulerLoadSnapshot::sampled_at`]) and has its two fixed
    /// token counts stated then ([`CountsStated`]), whatever other ranks
    /// the file lists.
    fn lay_polled(
        reports: &mut FxHashMap<String, PushedFile>,
        url: &str,
        load: &WorkerLoadResponse,
        ranks: &PolledRanks,
        polled_at: Instant,
    ) -> (Vec<SchedulerLoadSnapshot>, Option<Instant>) {
        // A rollup from a gateway registered as a worker is keyed by
        // downstream worker, not by rank: its entries all say rank 0 and
        // would collapse into one slot (the sink's merge and the DP cache
        // apply the same guard).
        if !load.ranks_are_dp_ranks() {
            return (Vec::new(), None);
        }
        let Some(file) = reports.get_mut(url) else {
            return (Vec::new(), None);
        };
        let on_file = file.report.loads.len();
        let mut newer = Vec::new();
        file.retain_rows(|row, stated| {
            if ranks.newer.contains(&row.dp_rank) {
                newer.push(row.clone());
                return true;
            }
            if let Some(entry) = load.loads.iter().find(|entry| entry.dp_rank == row.dp_rank) {
                // The row is the answer's from here: sampled at the poll's
                // instant, with both counts stated then.
                row.clone_from(entry);
                row.sampled_at = Some(polled_at);
                *stated = CountsStated::both(polled_at);
                return true;
            }
            !ranks.leaving.contains(&row.dp_rank)
        });
        let report = &mut file.report;
        if report.loads.len() < on_file {
            report.dp_rank_count = i32::try_from(report.loads.len()).unwrap_or(i32::MAX);
        }
        let sampled_at = if newer.is_empty() {
            None
        } else {
            report.sampled_at
        };
        (newer, sampled_at)
    }

    /// The file's half of an answer that brought a member no row: the rows
    /// of the ranks that leave are dropped. A report left without ranks
    /// publishes nothing.
    fn drop_leaving(reports: &mut FxHashMap<String, PushedFile>, url: &str, leaving: &[i32]) {
        if let Some(file) = reports.get_mut(url) {
            file.retain_rows(|row, _| !leaving.contains(&row.dp_rank));
            let report = &mut file.report;
            report.dp_rank_count = i32::try_from(report.loads.len()).unwrap_or(i32::MAX);
        }
    }

    /// The end of a group's tick, for the sink: every Ready worker of the
    /// group, polled or served by its stream, and each polled member's
    /// report as the tick publishes it, with whether its answer carried rank
    /// capacity. It is said whether or not a fetch succeeded, and when
    /// nobody was polled.
    fn end_tick(
        &self,
        workers: &[Arc<dyn Worker>],
        group_loads: &[(Arc<dyn Worker>, WorkerLoadResponse, bool)],
    ) {
        let reports: Vec<PolledReport<'_>> = group_loads
            .iter()
            .map(|(worker, report, carries_rank_capacity)| PolledReport {
                worker,
                report,
                carries_rank_capacity: *carries_rank_capacity,
            })
            .collect();
        self.sink.tick(&GroupTick {
            workers,
            reports: &reports,
        });
    }

    /// The end of a tick that polled: each answer's `smg_engine_*` series
    /// and the group's rows, under one hold of the lock of the reports. A
    /// record writes its series under that lock and the sentinel is
    /// written there, so no record and no eviction runs between a poll's
    /// series and its row, which the sentinel's guard reads. The series
    /// are written behind the publish's own fence, read under the hold:
    /// an answer for a worker that left during the fetch writes none. A
    /// row the publish takes from a worker that left ends under the same
    /// hold ([`Self::publish_rows`]).
    fn publish_polled(
        &self,
        model_id: &str,
        polled_urls: &[String],
        group_loads: Vec<(Arc<dyn Worker>, WorkerLoadResponse, bool)>,
    ) {
        let reports = self.pushed_reports.lock();
        // Every successful load poll is also the canonical observability
        // sample. Load-aware policies already require this poll; the
        // explicit engine-metrics option only forces polling when routing
        // does not. Reusing the response avoids a second Engine RPC. (A
        // pushed record feeds the same gauges from `apply_pushed_load`.)
        for (worker, load, _) in &group_loads {
            if self.load_state.is_current_incarnation(worker) {
                Metrics::record_engine_load(
                    worker.as_ref(),
                    model_id,
                    registered_engine_role(*worker.worker_type()),
                    load,
                );
            }
        }
        // Merge into the shared snapshot in one rebuild: clear stale entries
        // for the polled URLs first, then insert the fresh loads — each
        // paired with the worker that produced it so the incarnation fence
        // can drop reports that raced a removal or replacement. Workers that
        // failed this tick get their stale entries pruned along with the
        // rest. The responses move (no deep clones): the metrics pass above
        // took its references.
        let fresh: Vec<(Arc<dyn Worker>, Arc<WorkerLoadResponse>)> = group_loads
            .into_iter()
            .map(|(worker, load, _)| (worker, Arc::new(load)))
            .collect();
        self.publish_rows(&reports, polled_urls, fresh);
    }

    /// Publish rows, and end each row the publish took away from a worker
    /// that is not its URL's `Ready` worker: a row that went under its
    /// successor's, or one a tick pruned behind its worker's leave. That
    /// worker's eviction finds no row of its own, so its batch can name no
    /// rank the row held past the worker's width, or finds no row at the
    /// URL and writes nothing. The row's series end here instead, but for
    /// the ones a live writer writes ([`Self::clear_series`]). A row taken
    /// from the URL's `Ready` worker, whose fetch failed this tick, ends
    /// nothing. `reports` is the locked map of the pushed reports.
    fn publish_rows(
        &self,
        reports: &FxHashMap<String, PushedFile>,
        group_urls: &[String],
        fresh: Vec<(Arc<dyn Worker>, Arc<WorkerLoadResponse>)>,
    ) {
        for (worker, row) in self.load_state.publish_group(group_urls, fresh) {
            self.clear_series(reports, &worker, Some(&row));
        }
    }

    /// One engine's answer of a tick, reconciled with what its members'
    /// streams pushed. The sink is told the answer whole, under the lock of
    /// the reports, and names the ranks whose stream is the newer and the
    /// ranks that leave the file; the file is laid by that answer under the
    /// same hold, so a record that lands between the two is not undone.
    /// Each member's answer then takes the rows of its newer ranks, and
    /// every step after reads the report so reconciled. A fetch that failed
    /// for every member is said with no lock held. Returns, by member,
    /// whether its answer carried rank capacity
    /// ([`PolledReport::carries_rank_capacity`]).
    fn reconcile_engine(
        &self,
        engine: &mut [(Arc<dyn Worker>, Option<WorkerLoadResponse>)],
        polled_at: Instant,
    ) -> Vec<bool> {
        // Whether the rank cache takes the answer as the worker's ranks is
        // the answer's own fact, the one the merge reads too: asked before
        // the stream's rows are laid over it, whose token capacity is not
        // the answer's. The merge reads one fact more, the answer's count
        // against its rows: a short answer feeds the cache and retracts no
        // rank from the file.
        let carries_rank_capacity: Vec<bool> = engine
            .iter()
            .map(|(_, response)| {
                response
                    .as_ref()
                    .is_some_and(|load| load.has_absolute_token_data() && load.ranks_are_dp_ranks())
            })
            .collect();
        if engine.iter().all(|(_, response)| response.is_none()) {
            let members: Vec<Arc<dyn Worker>> = engine
                .iter()
                .map(|(worker, _)| Arc::clone(worker))
                .collect();
            self.sink.fetch_failed(&members, polled_at);
            return carries_rank_capacity;
        }
        let with_rows = engine
            .iter()
            .any(|(_, response)| response.as_ref().is_some_and(|load| !load.loads.is_empty()));
        let read_at = Instant::now();
        let mut kept = Vec::with_capacity(engine.len());
        {
            let mut reports = self.pushed_reports.lock();
            let ranks = {
                let answer: Vec<PolledMember<'_>> = engine
                    .iter()
                    .map(|(worker, response)| PolledMember {
                        worker,
                        answer: response.as_ref(),
                        on_file: reports.get(worker.url()).map(|file| &file.report),
                    })
                    .collect();
                if with_rows {
                    self.sink.report(&answer, polled_at)
                } else {
                    PolledRanks {
                        newer: Vec::new(),
                        leaving: self.sink.report_empty(&answer, polled_at),
                    }
                }
            };
            for (worker, response) in engine.iter() {
                kept.push(match response {
                    Some(load) if !load.loads.is_empty() => {
                        Self::lay_polled(&mut reports, worker.url(), load, &ranks, read_at)
                    }
                    Some(_) => {
                        Self::drop_leaving(&mut reports, worker.url(), &ranks.leaving);
                        (Vec::new(), None)
                    }
                    None => (Vec::new(), None),
                });
            }
        }
        // A worker polled because a rank's records went quiet (`fallback`)
        // takes the poll's figures for the ranks that pushed nothing since
        // the tick began. A rank that pushed since keeps the stream's row,
        // in the report on file and in the report this poll publishes,
        // whether the poll's answer names the rank or not.
        for ((_, response), (newer, stream_sampled_at)) in engine.iter_mut().zip(kept) {
            let Some(load) = response.as_mut().filter(|load| !load.loads.is_empty()) else {
                continue;
            };
            // One sample instant per report, the earlier of the two: the
            // carried rows keep their stream's, so the scorer releases
            // no dispatch they never saw; the poll's own rows cost their
            // in-flight dispatches once more until the next report, the
            // safe side.
            load.sampled_at = Some(stream_sampled_at.map_or(read_at, |at| at.min(read_at)));
            let polled_rows = load.loads.len();
            for row in newer {
                if let Some(entry) = load
                    .loads
                    .iter_mut()
                    .find(|entry| entry.dp_rank == row.dp_rank)
                {
                    *entry = row;
                } else {
                    // A rank the answer does not name: the row goes in
                    // before the first entry of a greater rank, its
                    // rank's place where the answer is listed by rank.
                    let at = load
                        .loads
                        .iter()
                        .position(|entry| entry.dp_rank > row.dp_rank)
                        .unwrap_or(load.loads.len());
                    load.loads.insert(at, row);
                }
            }
            // The rank count is never below the rows the report carries; a
            // larger count of the poll's own stands, so a partial report
            // stays partial for the readers that ask.
            if load.loads.len() > polled_rows {
                let carried = i32::try_from(load.loads.len()).unwrap_or(i32::MAX);
                load.dp_rank_count = load.dp_rank_count.max(carried);
            }
        }
        carries_rank_capacity
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
            reason = "ends when the monitor is stopped or dropped; the stop alone waits on it"
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
    /// A report the fence refuses publishes nothing and prunes nothing: the
    /// row its worker published before is the eviction's to remove. A file
    /// marked while its worker was not `Ready` stays queued through the
    /// flushes that find it so. The first that finds the worker `Ready`
    /// writes its held series and publishes its row, with no record since.
    pub(crate) fn flush_pushed(&self) {
        let mut reports = self.pushed_reports.lock();
        let mut pending = self.pending_pushed.lock();
        if pending.is_empty() {
            return;
        }
        let timestamp = chrono::Utc::now().to_rfc3339();
        let mut fresh: Vec<PushedReport> = Vec::with_capacity(pending.len());
        let mut held: Vec<Arc<dyn Worker>> = Vec::new();
        for worker in pending.drain(..) {
            // A worker whose file went to the next worker at its URL has
            // nothing here: that one queued itself.
            let Some(file) = reports
                .get_mut(worker.url())
                .filter(|file| Arc::ptr_eq(&file.pusher, &worker))
            else {
                continue;
            };
            file.queued = false;
            if file.report.loads.is_empty() {
                continue;
            }
            if file.series_held {
                if !self.load_state.is_current_incarnation(&worker) {
                    // Held still, and the fence would refuse it: it stays
                    // queued for the flush that finds its worker `Ready`.
                    file.queued = true;
                    held.push(worker);
                    continue;
                }
                for at in 0..file.report.loads.len() {
                    file.write_rank_series(at, worker.model_id());
                }
                file.series_held = false;
            }
            let report = &file.report;
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
        }
        // Under the hold of the reports still: no record has queued one of
        // these workers since the drain above.
        pending.extend(held);
        drop(pending);
        if !fresh.is_empty() {
            // No URL is pruned. A report the fence admits overwrites its
            // worker's row, and a row of another worker it overwrites ends
            // there. A refused one leaves the row to the eviction, which
            // writes the sentinel of the series with it.
            self.publish_rows(&reports, &[], fresh);
        }
    }

    /// `flush_pushed` for a host's tests: the publish of the pushed reports
    /// without the window's wait.
    #[cfg(any(test, feature = "test-util"))]
    pub fn flush_pushed_for_test(&self) {
        self.flush_pushed();
    }

    /// Pushed reports waiting to be published (tests).
    #[cfg(test)]
    pub(crate) fn pending_pushed_len(&self) -> usize {
        self.pending_pushed.lock().len()
    }

    /// The current published load snapshot — what routing is acting on, and
    /// what `GET /loads` serves.
    pub fn load_snapshot(&self) -> Arc<LoadSnapshot> {
        self.load_state.snapshot()
    }

    /// Subscribe to the shared per-worker load snapshots.
    ///
    /// Each received value is an immutable `Arc<LoadSnapshot>`: clone the
    /// `Arc` out of `borrow()` and scan without holding the watch guard.
    /// Stale entries are pruned on each tick of the relevant group and by
    /// the batched eviction flusher.
    pub fn subscribe(&self) -> LoadReceiver {
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
    ///
    /// A start ends what it replaces. Where a slot still holds the handle
    /// of an earlier start's event task or liveness sweep, that task is
    /// told to end and the new task's handle takes the slot. The groups'
    /// loops end at the rebuild from the registry. An eviction flusher
    /// that still stands is kept: it may have taken the wakeup of an
    /// eviction it has not applied yet.
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
            reason = "WorkerMonitor event loop runs for the lifetime of the registry; the JoinHandle is stored on the monitor and aborted at its stop or in Drop"
        )]
        let handle = tokio::spawn(async move {
            run_event_loop(monitor, events_rx).await;
        });

        replace(&self.event_task, handle);

        // The debounced eviction flusher: holds the load state strongly (it
        // is cheap and monitor-independent) and the monitor weakly, for the
        // same cycle-breaking reason as the event loop. One that still
        // stands is kept, by the rule of the pushed flusher's start: it
        // holds nothing a start changes, and ended in its debounce it
        // would leave the eviction that woke it queued.
        let mut flusher = self.eviction_flush_task.lock();
        if flusher.as_ref().is_none_or(JoinHandle::is_finished) {
            let load_state = Arc::clone(&self.load_state);
            let monitor = Arc::downgrade(self);
            #[expect(
                clippy::disallowed_methods,
                reason = "eviction flusher runs for the monitor's lifetime; the JoinHandle is stored on the monitor and aborted at its stop or in Drop"
            )]
            let flush_handle = tokio::spawn(async move {
                eviction_flush_loop(load_state, monitor).await;
            });
            *flusher = Some(flush_handle);
        }
        drop(flusher);

        // The liveness sweep: vetoes a worker that stayed silent for the stall
        // threshold after a transport failure (see `health::liveness::sweep`).
        let monitor = Arc::downgrade(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "liveness sweep runs for the monitor's lifetime; the JoinHandle is stored on the monitor and aborted at its stop or in Drop"
        )]
        let sweep_handle = tokio::spawn(async move {
            liveness_sweep_loop(monitor).await;
        });
        replace(&self.liveness_sweep_task, sweep_handle);
    }

    /// End what [`Self::start_event_loop`] started, for a host that reads
    /// its workers' loads for a part of its life alone. The event task ends
    /// first, so no group starts behind this call. Then every group's
    /// polling loop, the eviction flusher, the liveness sweep and the
    /// publisher of pushed reports end. Each task's end is awaited, so no
    /// tick of a loop that stands publishes behind the reset, and none
    /// writes a verdict there. One kind of loop is not awaited: that of a
    /// group that ended, changed its interval or had another loop spawned
    /// over it ahead of the call. It was
    /// told to end there, and a tick of it that is past its fetch can
    /// still publish. Last the
    /// monitor resets as its rebuild from the registry does: the shared
    /// snapshot is published empty, the probe memo goes, and the sink is
    /// told the reset ([`LoadSink::reset`]). The monitor's own sink takes
    /// every overload veto off the registry's workers there.
    ///
    /// The monitor then runs no task: its snapshot is empty, its probe memo
    /// is gone and its sink was told the reset. A pusher that is still the
    /// registry's `Ready` worker at its URL keeps its report on file, and
    /// the sink its receipts. The receivers stay subscribed and read the
    /// empty snapshot, and [`Self::start_event_loop`] starts the monitor
    /// again over the registry as it stands at that call.
    ///
    /// A drop is no such end. A group's loop holds the monitor across its
    /// poll, so a monitor its host dropped lives until no poll is in
    /// flight. Where the polls of two groups each outlast the period, each
    /// loop takes the monitor again while the other's poll holds it, and
    /// the monitor polls on with no host.
    ///
    /// A stopped monitor still takes a pushed record
    /// ([`Self::apply_pushed_load`]), as one that was never started does.
    /// So a worker whose stream pushes can carry an overload veto with no
    /// poll behind it, until the next start: its rebuild from the registry
    /// tells the sink the reset.
    ///
    /// A stop dropped at its first wait puts the event task's handle back.
    /// That task was told to end, and a poll of it that still runs can
    /// start a group: the next stop waits for the task's end ahead of the
    /// groups it takes. A stop dropped at a later wait has told every other
    /// task to end. It has reset nothing: a later stop does.
    ///
    /// A host makes its stops and starts in turn. A start behind a stop
    /// that was dropped ends the event task and the liveness sweep that
    /// stop left standing, and keeps its eviction flusher
    /// ([`Self::start_event_loop`]): no task is left with no handle. A
    /// start beside a stop that still waits is no turn: that stop ends the
    /// start's other tasks and leaves its event task, with its handle in
    /// the slot.
    ///
    /// Called by no task of the monitor: the call waits for each of them.
    pub async fn stop(&self) {
        Awaited::out_of(&self.event_task).ended().await;
        // No group starts behind the event task's end, so the tasks taken
        // here are every task that stands.
        let mut ending: Vec<JoinHandle<()>> = Vec::new();
        for task in [
            &self.eviction_flush_task,
            &self.liveness_sweep_task,
            &self.pushed_flush_task,
        ] {
            ending.extend(task.lock().take());
        }
        {
            let mut handles = self.group_handles.lock();
            ending.extend(handles.drain().map(|(_, state)| state.handle));
        }
        // Each is told to end ahead of the first wait: a stop dropped there
        // drops handles of tasks that are ending.
        for task in &ending {
            task.abort();
        }
        for mut task in ending {
            ended(&mut task).await;
        }
        self.stop_all_groups();
    }

    /// Stop every per-group polling loop, evict every pusher that is no longer
    /// the registry's `Ready` worker at its URL
    /// ([`Self::evict_lost_pushers`]), clear the shared load snapshot and the
    /// probe memo, and say the reset to the sink ([`LoadSink::reset`]): the
    /// monitor's own drops its DP-rank cache and fails every overload veto
    /// open.
    ///
    /// Called from the rebuild from the registry, which is the bootstrap
    /// reconcile path and the `RecvError::Lagged` rebuild, and from
    /// [`Self::stop`] behind the end of every task. The monitor's drop does
    /// not call it: the drop aborts the tasks and says nothing to the sink.
    /// Always clears, even when no groups were running, so a lag-recovery
    /// rebuild cannot inherit stale per-rank loads from a previous
    /// incarnation. Idempotent.
    pub(crate) fn stop_all_groups(&self) {
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

        // The lost pushers go first, and every queued eviction is applied
        // at once: the sentinel of a worker's engine series follows the
        // snapshot entry an eviction removes, and the clear below leaves
        // none.
        self.evict_lost_pushers();
        self.flush_pending_evictions();

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
        self.native_loads_memo.clear();
        // What the sink keeps is void with the feed, and the sink is told:
        // the monitor's own drops its cache of rank loads and fails every
        // overload veto open.
        self.sink.reset();
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

    /// Evict the loads of every pusher that is no longer the `Ready` worker
    /// the registry holds at its URL: it left, was replaced or left `Ready`
    /// while registry events were lost, so no event evicted it. The test is
    /// the publish fence's ([`LoadState::is_current_incarnation`]): the
    /// eviction a lost event owed is made here. A pusher that stays keeps
    /// its report, and the sink its receipts: its stream is not the
    /// monitor's to stop. A worker that is still registered starts its
    /// report again at its next record, and one registered at the URL
    /// later starts a report of its own.
    fn evict_lost_pushers(&self) {
        let lost: Vec<Arc<dyn Worker>> = self
            .pushed_reports
            .lock()
            .values()
            .filter(|file| !self.load_state.is_current_incarnation(&file.pusher))
            .map(|file| Arc::clone(&file.pusher))
            .collect();
        for pusher in &lost {
            self.evict_worker_loads(pusher);
        }
    }

    /// Evict a single worker's cached loads. Used by the event loop on
    /// `Removed`, `Replaced`, and `StatusChanged` away from `Ready`, and by
    /// a reset for a pusher whose event was lost.
    ///
    /// The probe memo, the pushed report and what the sink holds of the
    /// worker (its overload flag, its receipts, its cached rank loads) are
    /// cleared synchronously; the shared snapshot removal is queued so a
    /// removal wave costs one batched rebuild instead of one per worker (see
    /// [`Self::flush_pending_evictions`]). The report and the sink's part
    /// go when the report at the URL is the worker's own, or none is there,
    /// and the series the report fed are set to the sentinel with it
    /// ([`Self::clear_series`]). A report another worker started ahead of
    /// this eviction stays, with what the sink holds at the URL: both are
    /// that worker's.
    fn evict_worker_loads(&self, worker: &Arc<dyn Worker>) {
        let url = worker.url();
        self.native_loads_memo.remove(url);
        {
            let mut reports = self.pushed_reports.lock();
            if reports
                .get(url)
                .is_none_or(|file| Arc::ptr_eq(&file.pusher, worker))
            {
                // The pushed report goes with the worker that pushed it: a
                // replacement is polled until its own stream pushes. The
                // sink forgets under this hold, so a record of the next
                // worker at the URL lands behind it and keeps its receipt.
                let report = reports.remove(url);
                self.sink.forget(worker);
                // The series go with the report, under this hold and with
                // or without a published row: a record ahead of the first
                // publish wrote them too.
                if let Some(file) = &report {
                    self.clear_series(&reports, worker, Some(&file.report));
                }
                // Its place in the flush's queue goes too, where a held
                // report waits. A worker whose file its successor took
                // over keeps its place until the next flush, which skips it.
                if report.is_some_and(|file| file.queued) {
                    self.pending_pushed
                        .lock()
                        .retain(|queued| !Arc::ptr_eq(queued, worker));
                }
            }
        }
        self.load_state.enqueue_eviction(Arc::clone(worker));
    }

    /// Apply every queued eviction in one snapshot rebuild, and sentinel the
    /// `smg_engine_*` series of workers whose own entry the batch removed
    /// (every rank that entry held), or whose entry had gone under another
    /// worker's ahead of the batch. The sentinel matters for a worker that
    /// stays registered (its status left `Ready`, or a same-URL replacement
    /// has not published yet), whose series would otherwise read its last
    /// load as live. For a removed worker it is a no-op: the worker series
    /// store dropped the series with the registration. A same-URL
    /// replacement that already republished keeps its entry, and its series
    /// continues uninterrupted. So does a worker that left `Ready` and came
    /// back ahead of the flush: the batch does not evict it, and no sentinel
    /// lands over what it wrote since. The sentinel is written under the
    /// lock of the reports, and not where the series has a live owner
    /// ([`Self::clear_series`]).
    pub(crate) fn flush_pending_evictions(&self) {
        for (worker, row) in self.load_state.apply_pending_evictions() {
            // One hold a worker: a removal wave does not keep the records
            // waiting behind every sentinel of the batch at once.
            let reports = self.pushed_reports.lock();
            self.clear_series(&reports, &worker, row.as_deref());
        }
    }

    /// Set a worker's `smg_engine_*` series to the sentinel, one by one,
    /// unless a live writer writes the series. The worker itself is one,
    /// where it is the registry's `Ready` worker again: none is set.
    /// Another `Ready` worker whose row stands at the URL is one for what
    /// the row's load carries, under each of its models: a poll writes a
    /// worker's series with the row it publishes. Another worker whose
    /// report stands at the URL with no record held is one for what that
    /// report carries, under its own model. A series is a live writer's
    /// where an entry of the rank in its load carries the series' figure,
    /// and a load can name a rank more than once: every entry is asked.
    /// The pair series carry the engine's role and no model: one is a live
    /// writer's under the role such an entry reports, whatever its model.
    /// A load series carries its model's label, and two models share one
    /// past the label cap: a writer's model is matched by that label.
    /// A report with a held record is no live writer: its worker was not
    /// `Ready` at that record, and the flush that finds it `Ready` writes
    /// its series. `reports` is the locked map of the pushed reports, and
    /// every series is written under that lock, so a sentinel and a live
    /// write do not cross. `ended` is what ends with the worker's feed,
    /// where there is one: the report an eviction or a successor's first
    /// record took off the file, or the load of the row the batch removed
    /// or a publish took away. The sentinel covers the worker's own width
    /// and each rank it held past that.
    fn clear_series(
        &self,
        reports: &FxHashMap<String, PushedFile>,
        worker: &Arc<dyn Worker>,
        ended: Option<&WorkerLoadResponse>,
    ) {
        if self.load_state.is_current_incarnation(worker) {
            return;
        }
        let row = self
            .load_state
            .another_live_workers_row(worker)
            .map(|(source, load)| {
                (
                    worker_model_ids(&source),
                    registered_engine_role(*source.worker_type()),
                    load,
                )
            });
        // Each live writer's labels are read once, and the guard compares
        // texts alone. The read is the intern the writers use, so it gives
        // the label a writer's series carry, and it admits at most the
        // label that writer's own first write admits. A second kind of
        // read of the label map would stop the compiler inlining the
        // map's lookup into the intern, a cost to every record that
        // builds its own keys.
        let row = row.as_ref().map(|(models, registered, load)| {
            let labels: Vec<Arc<str>> = models
                .iter()
                .map(|model| intern_model_label(model))
                .collect();
            (labels, *registered, load)
        });
        let other = reports
            .get(worker.url())
            .filter(|other| !Arc::ptr_eq(&other.pusher, worker) && !other.series_held)
            .map(|other| {
                (
                    intern_model_label(other.pusher.model_id()),
                    registered_engine_role(*other.pusher.worker_type()),
                    other,
                )
            });
        write_sentinel(worker, ended, |label, rank, carries| {
            // An entry is tested with its own writer's registered role,
            // the one that writer's load series carry.
            let holds = |loads: &[SchedulerLoadSnapshot], registered: &'static str| {
                loads
                    .iter()
                    .any(|load| load.dp_rank == rank && carries(load, registered))
            };
            row.as_ref().is_some_and(|(labels, registered, load)| {
                label.is_none_or(|label| labels.iter().any(|own| **own == *label))
                    && holds(&load.loads, registered)
            }) || other.as_ref().is_some_and(|(own, registered, other)| {
                label.is_none_or(|label| **own == *label) && holds(&other.report.loads, registered)
            })
        });
    }

    /// Spawn the polling loop for a single group. A loop of the group
    /// that still stands is told to end: its handle is written over.
    fn spawn_group_loop(self: &Arc<Self>, key: WorkerGroupKey, interval: Duration) {
        info!("Starting load monitor for group {key} with interval {interval:?}");

        // Same `Weak<Self>` rationale as `start_event_loop`: the
        // group loop is owned by `group_handles` on the monitor, so
        // it must not own a strong reference back to the monitor.
        let monitor = Arc::downgrade(self);
        let group_key = key.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "Group polling loop runs for the lifetime of the group; the JoinHandle is stored in group_handles and aborted on group removal, or at the monitor's stop or drop"
        )]
        let handle = tokio::spawn(async move {
            group_monitor_loop(monitor, group_key, interval).await;
        });

        let mut handles = self.group_handles.lock();
        // Two reconciles of one group can each find no loop and each spawn
        // one: the loop written over is told to end, so none polls on
        // with no handle.
        if let Some(replaced) = handles.insert(key, GroupState { handle, interval }) {
            replaced.handle.abort();
        }
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
    /// means a successful poll omitted this rank; an error means a failed
    /// poll, with the reason the gauges' read ended on, or the native read's
    /// for a worker with no gauges to read.
    ///
    /// `native_loads_memo` is the shared probe memo, or `None` for callers
    /// with no monitor to borrow it from (they simply always discover).
    pub(crate) async fn fetch_http_load(
        worker: &Arc<dyn Worker>,
        native_loads_memo: Option<&DashMap<String, NativeLoadsMemo>>,
    ) -> Result<WorkerLoadResponse, FetchFailure> {
        match Self::fetch_native_loads(worker, native_loads_memo).await {
            Ok((path, load)) => {
                liveness::on_contact(worker);
                Ok(Self::scoped_native_load(path, load, worker.dp_rank()))
            }
            Err(native) => Self::fetch_http_gauges(worker, native).await,
        }
    }

    /// The whole answer of the first native load route that has one (an
    /// engine's every rank, or a gateway's fleet rollup) with the route it
    /// came from, before it is scoped to a worker. An error is a failed poll:
    /// with the reason the last read that failed ended on, or with none
    /// where no route is there.
    async fn fetch_native_loads(
        worker: &Arc<dyn Worker>,
        native_loads_memo: Option<&DashMap<String, NativeLoadsMemo>>,
    ) -> Result<(NativeLoadsPath, WorkerLoadResponse), Option<FetchFailure>> {
        // Only workers already `Ready` are polled, so a definitive "no such
        // endpoint" cannot be a warm-up artifact and is safe to memoize.
        let known = native_loads_memo
            .and_then(|store| store.get(worker.url()).map(|hit| *hit))
            .unwrap_or_default();
        let mut memo = known;

        let mut response = Err(None);
        for path in known.candidates() {
            match Self::probe_native_loads(worker, path).await {
                NativeLoads::Available(load) => {
                    memo.record_answered(path);
                    response = Ok((path, load));
                    break;
                }
                // On a route that had answered, a body that does not read is
                // one bad answer: the memo keeps the route and the next tick
                // asks it again. On a route that never answered it means
                // something else is mounted here, as definitive as a 404.
                // The read ended on its own reason either way.
                NativeLoads::Unreadable(reason) => {
                    if known.answered != Some(path) {
                        memo.record_absent(path);
                    }
                    response = Err(Some(reason));
                }
                NativeLoads::Absent => memo.record_absent(path),
                NativeLoads::Inconclusive(reason) => response = Err(Some(reason)),
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

    /// One engine's answer as each member's report, in the members' order:
    /// a member ahead of the last copies its own rank of the answer, and the
    /// last takes the answer itself.
    fn scope_to_members(
        reports: &mut Vec<Option<WorkerLoadResponse>>,
        members: &[Arc<dyn Worker>],
        answer: WorkerLoadResponse,
    ) {
        let Some((last, ahead)) = members.split_last() else {
            return;
        };
        for member in ahead {
            let own = Self::copy_for_dp_rank(&answer, member.dp_rank());
            reports.push(Some(own.unwrap_or_default()));
        }
        reports.push(Some(answer.for_dp_rank(last.dp_rank()).unwrap_or_default()));
    }

    /// [`WorkerLoadResponse::for_dp_rank`] of a copy, for a member that
    /// leaves the answer to the members after it: the entries of its own
    /// rank alone are copied, not every rank's.
    fn copy_for_dp_rank(
        answer: &WorkerLoadResponse,
        rank: Option<usize>,
    ) -> Option<WorkerLoadResponse> {
        let Some(rank) = rank else {
            return answer.clone().for_dp_rank(None);
        };
        if !answer.ranks_are_dp_ranks() {
            return None;
        }
        let rank = i32::try_from(rank).ok()?;
        let mut of_rank = answer.loads.iter().filter(|load| load.dp_rank == rank);
        let mut loads = vec![of_rank.next()?.clone()];
        loads.extend(of_rank.cloned());
        Some(WorkerLoadResponse {
            timestamp: answer.timestamp.clone(),
            version: answer.version.clone(),
            dp_rank_count: i32::try_from(loads.len()).ok()?,
            loads,
            aggregate: None,
            sampled_at: answer.sampled_at,
        })
    }

    /// The backend-wide gauges of a worker whose native load routes brought
    /// no answer (`native`: why, or none where no route is there). A worker
    /// with no gauges to read fails as its native read did, and has no load
    /// read where no route is there.
    async fn fetch_http_gauges(
        worker: &Arc<dyn Worker>,
        native: Option<FetchFailure>,
    ) -> Result<WorkerLoadResponse, FetchFailure> {
        let no_gauges = native.unwrap_or(FetchFailure::Unsupported);
        // Backend-wide gauges cannot be attributed to a virtual DP worker.
        if worker.dp_rank().is_some() {
            return Err(no_gauges);
        }

        match worker.metadata().spec.runtime_type {
            RuntimeType::Vllm => Self::fetch_http_load_vllm(worker).await,
            RuntimeType::Sglang => Self::fetch_http_load_sglang(worker).await,
            // Custom engines expose no gauge schema we can parse; the native
            // routes above were their only path.
            _ => Err(no_gauges),
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
            Err(error) => return NativeLoads::Inconclusive(Self::request_failure(&error)),
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
            return NativeLoads::Inconclusive(FetchFailure::Answer);
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

        // The body's read and its parse are two steps, so each failure has
        // its own reason: a read that failed is the transport's or the
        // deadline's, and a body that is no load response is the engine's
        // answer. Whether the route is absent is the caller's to say: it
        // knows if the route ever answered.
        let body = match resp.bytes().await {
            Ok(body) => body,
            Err(error) => return NativeLoads::Unreadable(Self::request_failure(&error)),
        };
        match serde_json::from_slice::<NativeResponse>(&body) {
            // An empty native snapshot is still authoritative: no rank has
            // telemetry yet. Do not turn it into an aggregate fallback.
            Ok(NativeResponse {
                loads,
                mut response,
            }) => {
                response.loads = loads;
                NativeLoads::Available(response)
            }
            // Not a load response.
            Err(_) => NativeLoads::Unreadable(FetchFailure::Answer),
        }
    }

    /// The first family over HTTP: the load from the Prometheus `/metrics`
    /// endpoint, read by its gauge table in [`scrape`].
    async fn fetch_http_load_vllm(
        worker: &Arc<dyn Worker>,
    ) -> Result<WorkerLoadResponse, FetchFailure> {
        let url = format!("{}/metrics", worker.base_url());
        let body = Self::authed_text(worker, &url).await?;
        scrape::vllm(&body)
            .map(Self::single_rank)
            .ok_or(FetchFailure::NoLoad)
    }

    /// The second family over HTTP: the load from the Prometheus `/metrics`
    /// gauges, read by its gauge table in [`scrape`] (either metric
    /// prefix).
    async fn fetch_http_load_sglang(
        worker: &Arc<dyn Worker>,
    ) -> Result<WorkerLoadResponse, FetchFailure> {
        let url = format!("{}/metrics", worker.base_url());
        let body = Self::authed_text(worker, &url).await?;
        scrape::sglang(&body)
            .map(Self::single_rank)
            .ok_or(FetchFailure::NoLoad)
    }

    /// Shared authenticated GET builder with the standard timeout.
    fn authed_request(worker: &Arc<dyn Worker>, url: &str) -> reqwest::RequestBuilder {
        let req = worker.http_client().get(url).timeout(REQUEST_TIMEOUT);
        match worker.api_key() {
            Some(key) => req.bearer_auth(key),
            None => req,
        }
    }

    /// The body of a shared authenticated GET with the standard timeout. A
    /// status that is no success is the engine's answer. A request the
    /// timeout ended is a deadline, at the send or in the body. Any other
    /// failure is the transport's.
    async fn authed_text(worker: &Arc<dyn Worker>, url: &str) -> Result<String, FetchFailure> {
        let failed = |error: reqwest::Error| Self::request_failure(&error);
        let response = Self::authed_request(worker, url)
            .send()
            .await
            .map_err(failed)?;
        if !response.status().is_success() {
            return Err(FetchFailure::Answer);
        }
        response.text().await.map_err(failed)
    }

    /// Why a request brought no answer: the deadline's where its timeout
    /// ended it, the transport's otherwise.
    fn request_failure(error: &reqwest::Error) -> FetchFailure {
        if error.is_timeout() {
            FetchFailure::Deadline
        } else {
            FetchFailure::Transport
        }
    }

    /// A single scheduler snapshot as a one-rank report: the wrapper of
    /// [`scrape`], kept under this name for the readers here.
    fn single_rank(snapshot: SchedulerLoadSnapshot) -> WorkerLoadResponse {
        scrape::single_rank(snapshot)
    }

    /// Fetch load via the backend client's `GetLoads` (gRPC RPC or the ZMQ
    /// engine's pushed scheduler stats) for the workers of one engine: one
    /// worker, or the rank workers of a data-parallel engine (one address,
    /// one answer naming every rank). The first
    /// member asks; the outcome is every member's, as each one's own poll
    /// would have had it: the whole answer (the caller scopes it to each
    /// member's rank), the contact that ends an unreachable veto, a
    /// transport failure. A failure names its reason: the host's own where
    /// there is no member or no client, the transport's for a connect that
    /// failed and for a status that is a connection's, the deadline's for a
    /// connect or a call the poll's deadline ended, and the engine's answer
    /// for any other status, but for a call the engine does not implement.
    pub(crate) async fn fetch_backend_load(
        members: &[Arc<dyn Worker>],
    ) -> Result<WorkerLoadResponse, FetchFailure> {
        let worker = members.first().ok_or(FetchFailure::Client)?;
        // A first use connects through the host's connector. The connect
        // stands under the deadline as the call below does, and for its
        // reason: a connector that never returns must not stall the group's
        // poll. A connect the deadline ends is a missing report for this
        // tick. The next tick connects again, and the worker's health
        // check, its event stream and its requests drive a slow connect to
        // its end.
        let connected = tokio::time::timeout(LOAD_POLL_DEADLINE, worker.engine_connection());
        let backend_client = match connected.await {
            Ok(Ok(Some(client))) => client,
            Ok(Ok(None)) => {
                debug!("No backend client for worker {}", worker.redacted_url());
                return Err(FetchFailure::Client);
            }
            Ok(Err(e)) => {
                debug!(
                    "Failed to get backend client for {}: {e}",
                    worker.redacted_url()
                );
                return Err(FetchFailure::Transport);
            }
            Err(_) => {
                debug!(
                    "the connect of {} did not end within {LOAD_POLL_DEADLINE:?}",
                    worker.redacted_url()
                );
                return Err(FetchFailure::Deadline);
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
                Ok(load)
            }
            Ok(Err(e)) => {
                debug!("backend GetLoads failed for {}: {e}", worker.redacted_url());
                if liveness::is_connection_code(e.code()) {
                    for member in members {
                        liveness::on_contact_failed(member, "load poll");
                    }
                    return Err(FetchFailure::Transport);
                }
                Err(match e.code() {
                    tonic::Code::Unimplemented => FetchFailure::Unsupported,
                    _ => FetchFailure::Answer,
                })
            }
            Err(_) => {
                debug!(
                    "backend GetLoads for {} did not answer within {LOAD_POLL_DEADLINE:?}",
                    worker.redacted_url()
                );
                Err(FetchFailure::Deadline)
            }
        }
    }

    /// One call of a host's load source for the HTTP workers of one engine,
    /// under the poll's deadline. A failure or a late answer is a missing
    /// report: the source's answer, or the deadline's. Nothing here speaks
    /// to the worker, so no contact is noted and none is failed: an HTTP
    /// worker's health stays its health check's.
    async fn fetch_source_load(
        worker: &Arc<dyn Worker>,
        source: &dyn LoadSource,
    ) -> Result<WorkerLoadResponse, FetchFailure> {
        match tokio::time::timeout(LOAD_POLL_DEADLINE, source.get_loads()).await {
            Ok(Ok(load)) => Ok(load),
            Ok(Err(e)) => {
                debug!(
                    "the host's load source failed for {}: {e}",
                    worker.redacted_url()
                );
                Err(FetchFailure::Answer)
            }
            Err(_) => {
                debug!(
                    "the host's load source for {} did not answer within {LOAD_POLL_DEADLINE:?}",
                    worker.redacted_url()
                );
                Err(FetchFailure::Deadline)
            }
        }
    }

    /// One tick's fetch for the polled workers of one engine: one worker, or
    /// the rank workers of a data-parallel engine, which share it. The
    /// engine answers once; each member's report is that
    /// answer scoped to its own rank, as its own poll would have had it (a
    /// rank the answer omits is an empty report). The last member takes the
    /// answer itself and each member ahead of it a copy of its own rank's
    /// rows (of the whole of a gateway's rollup, which every member keeps),
    /// so an engine of one worker copies none. A report's ranks and its
    /// named values stand under the report's bounds, and the values that
    /// left them are counted for its member.
    ///
    /// Where the host keeps a load source for an HTTP worker
    /// ([`Worker::load_source`]), the source answers for the engine in the
    /// worker's place: no load request goes to the worker, and its entry in
    /// the native loads memo is never written. The connector's word on the
    /// source and the source's call each stand under the poll's deadline.
    ///
    /// Each fetch made here is counted once in `counts` by how it ended
    /// ([`PollCounts::fetched`]): one for the engine, its ranks' shared read
    /// among them.
    async fn fetch_engine_loads(
        members: Vec<Arc<dyn Worker>>,
        connection_mode: ConnectionMode,
        native_loads_memo: &DashMap<String, NativeLoadsMemo>,
        counts: &PollCounts,
    ) -> Vec<(Arc<dyn Worker>, Option<WorkerLoadResponse>)> {
        let over_http = connection_mode == ConnectionMode::Http;
        let mut reports = Vec::with_capacity(members.len());
        // The first member asks for the engine, as it does on every wire.
        // The connector's answer stands under the poll's deadline, as its
        // source's call does, so a host that is slow to say stalls no poll
        // of the group. A late answer is a missing report for this tick,
        // never a poll over HTTP in the source's place: the next tick asks
        // again.
        let source = match (connection_mode, members.first()) {
            (ConnectionMode::Http, Some(first)) => {
                let asked = tokio::time::timeout(LOAD_POLL_DEADLINE, first.load_source());
                let Ok(source) = asked.await else {
                    debug!(
                        "the host's connector did not say within {LOAD_POLL_DEADLINE:?} \
                         whether it keeps a load source for {}",
                        first.redacted_url()
                    );
                    counts.fetched::<()>(over_http, &Err(FetchFailure::Client));
                    return members.into_iter().map(|member| (member, None)).collect();
                };
                source
            }
            _ => None,
        };
        match (source, connection_mode) {
            (Some(source), _) => {
                let answer = Self::fetch_source_load(&members[0], source.as_ref()).await;
                counts.fetched(over_http, &answer);
                match answer {
                    Ok(load) => Self::scope_to_members(&mut reports, &members, load),
                    Err(_) => reports.resize(members.len(), None),
                }
            }
            (None, ConnectionMode::Http) => {
                if let [worker] = members.as_slice() {
                    let answer = Self::fetch_http_load(worker, Some(native_loads_memo)).await;
                    counts.fetched(over_http, &answer);
                    reports.push(answer.ok());
                } else {
                    match Self::fetch_native_loads(&members[0], Some(native_loads_memo)).await {
                        Ok((path, load)) => {
                            counts.fetched(over_http, &Ok(()));
                            for member in &members {
                                liveness::on_contact(member);
                            }
                            match path {
                                NativeLoadsPath::Engine => {
                                    Self::scope_to_members(&mut reports, &members, load);
                                }
                                // A gateway's rollup is every member's whole:
                                // it keeps its worker annotations.
                                NativeLoadsPath::Gateway => {
                                    reports.extend(std::iter::repeat_n(Some(load), members.len()));
                                }
                            }
                        }
                        // The members are one engine's ranks, and a rank
                        // has no gauges of its own: the engine's one read is
                        // the one count, by the reason it ended on, or as
                        // no load read where no route is there.
                        Err(native) => {
                            let ended = native.unwrap_or(FetchFailure::Unsupported);
                            counts.fetched::<()>(over_http, &Err(ended));
                            reports.resize(members.len(), None);
                        }
                    }
                }
            }
            (None, ConnectionMode::Grpc | ConnectionMode::Zmq) => {
                let answer = Self::fetch_backend_load(&members).await;
                counts.fetched(over_http, &answer);
                match answer {
                    Ok(load) => Self::scope_to_members(&mut reports, &members, load),
                    Err(_) => reports.resize(members.len(), None),
                }
            }
        }
        let mut answers: Vec<_> = members.into_iter().zip(reports).collect();
        // A polled list is held to the bounds a pushed record's is read
        // under, here: ahead of the sink and of every reader.
        for (member, report) in &mut answers {
            // The rows first. A rollup lists one row a downstream worker,
            // so it may state a rank more than once. One engine's count
            // follows the rows that left for their rank, a repeated row
            // aside: a whole listing still reads as whole.
            if let Some(answer) = report {
                let one_engine = answer.ranks_are_dp_ranks();
                let left =
                    value_bounds::bound_ranks(&mut answer.loads, member.dp_rank(), one_engine);
                if member.dp_rank().is_some() {
                    // `for_dp_rank` counted the scoped rows, a repeated one
                    // among them: a pinned member's count is its rows kept.
                    answer.dp_rank_count = i32::try_from(answer.loads.len()).unwrap_or(i32::MAX);
                } else if one_engine && left != 0 {
                    let left = i32::try_from(left).unwrap_or(i32::MAX);
                    answer.dp_rank_count = answer.dp_rank_count.saturating_sub(left).max(0);
                }
            }
            let dropped: usize = report
                .iter_mut()
                .flat_map(|answer| &mut answer.loads)
                .map(SchedulerLoadSnapshot::bound_values)
                .sum();
            if dropped != 0 {
                counts.dropped(member.url(), dropped);
            }
        }
        answers
    }
}

impl<G> Drop for WorkerMonitor<G> {
    fn drop(&mut self) {
        // The tasks are aborted and the sink is told nothing: no reset is
        // said here. A sink's code inside a drop would run on whatever
        // thread lets the last handle go, under the locks that thread
        // holds. `LoadSink::reset` says what a host does instead.
        if let Some(handle) = self.event_task.get_mut().take() {
            handle.abort();
        }
        if let Some(handle) = self.eviction_flush_task.get_mut().take() {
            handle.abort();
        }
        if let Some(handle) = self.pushed_flush_task.get_mut().take() {
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

/// A task's handle, out of its slot while [`WorkerMonitor::stop`] waits for
/// the task's end. Dropped ahead of that end, it puts the handle back: the
/// stop behind a dropped one waits for the task too. It puts it back into
/// an empty slot alone. A handle a start wrote there meanwhile stays: that
/// start's task is the one a later stop ends.
struct Awaited<'a> {
    slot: &'a Mutex<Option<JoinHandle<()>>>,
    handle: Option<JoinHandle<()>>,
}

impl<'a> Awaited<'a> {
    fn out_of(slot: &'a Mutex<Option<JoinHandle<()>>>) -> Self {
        let handle = slot.lock().take();
        Self { slot, handle }
    }

    /// Tell the task to end and wait for its end ([`ended`]).
    async fn ended(mut self) {
        if let Some(handle) = &mut self.handle {
            ended(handle).await;
        }
        self.handle = None;
    }
}

impl Drop for Awaited<'_> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let mut slot = self.slot.lock();
            if slot.is_none() {
                *slot = Some(handle);
            }
        }
    }
}

/// Write a task's handle into its slot, and tell the task the slot still
/// held to end: a start ends the event task and the liveness sweep it
/// replaces ([`WorkerMonitor::start_event_loop`]).
fn replace(slot: &Mutex<Option<JoinHandle<()>>>, handle: JoinHandle<()>) {
    if let Some(replaced) = slot.lock().replace(handle) {
        replaced.abort();
    }
}

/// Abort a task of the monitor and wait for its end ([`WorkerMonitor::stop`]).
/// An aborted task ends at its next wait, and a cancelled join is its end.
async fn ended(handle: &mut JoinHandle<()>) {
    handle.abort();
    let _ = handle.await;
}

/// Debounced eviction flusher: waits for the first queued eviction, lets a
/// gradual removal stream accumulate for [`EVICTION_DEBOUNCE`], then applies
/// the whole backlog in one snapshot rebuild.
///
/// Holds the monitor weakly (same cycle-breaking rationale as the other
/// loops); holding the `LoadState` strongly is fine — it owns no task
/// handles back into the monitor.
async fn eviction_flush_loop<G: Generation>(
    load_state: Arc<LoadState<G>>,
    monitor: Weak<WorkerMonitor<G>>,
) {
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
/// rank workers of one data-parallel engine, which share one. An engine
/// of N ranks is N workers at one address, and each
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
    worker_model_ids(worker)
        .into_iter()
        .map(|model_id| WorkerGroupKey {
            model_id,
            worker_type: *worker.worker_type(),
            connection_mode: *worker.connection_mode(),
        })
        .collect()
}

/// The least interval a group polls at. A tick asks every polled engine of
/// the group for its load and publishes one snapshot, so at this interval
/// an engine is asked ten times a second. A load that has to be newer than
/// that is pushed on the worker's event stream.
pub const LEAST_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Compute the polling interval for a group from per-worker overrides.
///
/// Uses the smallest `load_monitor_interval_secs` across the group so
/// the fastest worker's polling cadence wins: a worker's override is whole
/// seconds, raised to one. Falls back to `default_interval` when no worker
/// sets an override. Always floored to [`LEAST_POLL_INTERVAL`] to prevent
/// tight-loop DoS.
fn group_interval(workers: &[Arc<dyn Worker>], default_interval: Duration) -> Duration {
    let override_secs = workers
        .iter()
        .filter_map(|w| w.metadata().spec.load_monitor_interval_secs)
        .min();

    let interval = override_secs
        .map(|s| Duration::from_secs(s.max(1)))
        .unwrap_or(default_interval);
    interval.max(LEAST_POLL_INTERVAL)
}

/// Background event handler. Lives as long as the `WorkerMonitor`.
///
/// Holds a `Weak<WorkerMonitor>` rather than an `Arc<WorkerMonitor>`
/// so the spawned task does not pin the monitor in memory after every
/// strong reference has been dropped. Each iteration upgrades the
/// `Weak` only for the duration of the work and drops the temporary
/// `Arc` before the next `recv().await`, so the monitor's `Drop` can
/// fire as soon as its owner, the gateway's `AppContext`, goes away.
async fn run_event_loop<G: Generation>(
    monitor: Weak<WorkerMonitor<G>>,
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
                // The event can be handled behind the worker's return. A
                // worker that is the registry's `Ready` worker again keeps
                // its loads: its feed has carried on since.
                if new_status == WorkerStatus::Ready {
                    // The group's polling loop reads the registry on every
                    // tick and picks the worker up by itself. A report
                    // held while the worker was not `Ready` waits in the
                    // flush's queue: the flusher is woken to publish it.
                    monitor.pushed_notify.notify_one();
                } else if !monitor.load_state.is_current_incarnation(&worker) {
                    monitor.evict_worker_loads(&worker);
                }
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
async fn liveness_sweep_loop<G: Generation>(monitor: Weak<WorkerMonitor<G>>) {
    let mut ticker = tokio::time::interval(liveness::SWEEP_INTERVAL);
    loop {
        ticker.tick().await;
        let Some(monitor) = monitor.upgrade() else {
            return;
        };
        for worker in monitor.worker_registry.global_generation().members().iter() {
            liveness::sweep(worker);
        }
    }
}

async fn group_monitor_loop<G: Generation>(
    monitor: Weak<WorkerMonitor<G>>,
    group_key: WorkerGroupKey,
    interval: Duration,
) {
    let mut interval_timer = tokio::time::interval(interval);
    // A tick that a poll overran is skipped. The timer's own default fires
    // every such tick when the slow poll returns: one poll behind another
    // with no interval between them, thirty of them behind a poll that ran
    // to its deadline at the least interval.
    interval_timer.set_missed_tick_behavior(MissedTickBehavior::Skip);

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
async fn poll_group_once<G: Generation>(
    monitor: &Arc<WorkerMonitor<G>>,
    group_key: &WorkerGroupKey,
    interval: Duration,
    now: Instant,
) {
    // Only poll Ready workers — Pending/NotReady/Failed do not
    // serve traffic and should not contribute load samples. A worker its
    // host asks for no load is no member of the tick.
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
        .filter(|w| w.status() == WorkerStatus::Ready && monitor.sink.polls(w.as_ref()))
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
    let on_connection = group_key.connection_mode != ConnectionMode::Http;
    for worker in &workers {
        let mode = monitor.sink.poll_mode(worker, now, interval);
        monitor
            .poll_counts
            .decided(worker.url(), mode, on_connection);
        if mode != PollMode::Suppressed {
            polled.push(Arc::clone(worker));
        }
    }

    // One answer an engine, its polled members together: the sink is told
    // an engine's answer whole.
    let engines: Vec<EngineAnswer> = if polled.is_empty() {
        debug!("Every worker in group {group_key} pushed its load within {interval:?}; no poll");
        Vec::new()
    } else {
        let futures: Vec<_> = polled_engines(&polled)
            .into_iter()
            .map(|members| {
                let native_loads_memo = Arc::clone(&monitor.native_loads_memo);
                let counts = Arc::clone(&monitor.poll_counts);
                let connection_mode = group_key.connection_mode;
                async move {
                    WorkerMonitor::<G>::fetch_engine_loads(
                        members,
                        connection_mode,
                        &native_loads_memo,
                        &counts,
                    )
                    .await
                }
            })
            .collect();
        future::join_all(futures).await
    };

    let mut group_loads: Vec<(Arc<dyn Worker>, WorkerLoadResponse, bool)> = Vec::new();
    for mut engine in engines {
        let carries_rank_capacity = monitor.reconcile_engine(&mut engine, now);
        for ((worker, response), carries_rank_capacity) in
            engine.into_iter().zip(carries_rank_capacity)
        {
            let Some(load) = response.filter(|load| !load.loads.is_empty()) else {
                continue;
            };
            // The wedged rule reads the queue depth off every report.
            let waiting: i64 = load
                .loads
                .iter()
                .map(|rank| i64::from(rank.num_waiting_reqs))
                .sum();
            let state = worker.runtime_state();
            liveness::on_load_report(&worker, state.as_deref(), waiting, None, false, None);
            group_loads.push((worker, load, carries_rank_capacity));
        }
    }

    monitor.end_tick(&workers, &group_loads);

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
        monitor.publish_polled(&group_key.model_id, &polled_urls, Vec::new());
        return;
    }

    debug!(
        "Fetched loads from {}/{} polled workers in group {group_key}",
        group_loads.len(),
        polled.len()
    );

    let rows = group_loads.iter().map(|(_, load, _)| load.loads.len());
    monitor.poll_counts.published(rows.sum());
    monitor.publish_polled(&group_key.model_id, &polled_urls, group_loads);
}

/// `poll_group_once` for a host's tests: one tick over engines the host's own
/// connector reaches.
#[cfg(any(test, feature = "test-util"))]
pub async fn poll_group_once_for_test<G: Generation>(
    monitor: &Arc<WorkerMonitor<G>>,
    group_key: &WorkerGroupKey,
    interval: Duration,
    now: Instant,
) {
    poll_group_once(monitor, group_key, interval, now).await;
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod native_loads_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{tests::PushedRecord, *};

    #[path = "tests/poll_counts.rs"]
    mod poll_counts;
    use crate::{
        connection::engine::{EngineConnection, EngineConnector, ZmqLink},
        identity::basic::{ConnectionMode, WorkerLoadGuard, WorkerType},
        model_card::ModelCard,
        test_support::{
            assert_metric, render_with_recorder, BasicWorkerBuilder, LoadReaders, WorkerMonitor,
            WorkerRegistry,
        },
        worker::{HealthCheckConfig, NamedValue, OverloadUpdate, WorkerSpec},
    };

    const ENGINE_METRICS: &str = "vllm:num_requests_running{m=\"a\"} 7.0\n\
         vllm:num_requests_waiting{m=\"a\"} 11.0\n\
         vllm:kv_cache_usage_perc{m=\"a\"} 0.5\n";

    /// Carries `num_waiting_uncached_tokens`, which the `/metrics` arm has no
    /// gauge for — so its presence identifies which path answered.
    const NATIVE_BODY: &str = r#"{"loads":[{"dp_rank":0,"num_running_reqs":3,
        "num_waiting_reqs":4,"num_waiting_uncached_tokens":900,"token_usage":0.25}]}"#;

    /// A native response but for one value: a bare token no JSON reader takes.
    const UNREADABLE_NATIVE_BODY: &str = r#"{"loads":[{"dp_rank":0,"num_running_reqs":NaN}]}"#;

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

    /// Loopback backend stub. Each native route answers with the statuses and
    /// bodies it was given, in order, the last one repeating, and counts its
    /// hits; `/metrics` always serves the first family's gauges.
    async fn spawn_backend(
        gateway: &[(StatusCode, &'static str)],
        engine: &[(StatusCode, &'static str)],
    ) -> Stub {
        fn route(
            hits: &Arc<AtomicUsize>,
            query: &Arc<Mutex<Option<String>>>,
            answers: &[(StatusCode, &'static str)],
        ) -> axum::routing::MethodRouter {
            assert!(!answers.is_empty());
            let answers: Arc<[_]> = answers.into();
            let counter = Arc::clone(hits);
            let sink = Arc::clone(query);
            axum::routing::get(
                move |axum::extract::RawQuery(raw): axum::extract::RawQuery| {
                    let hit = counter.fetch_add(1, Ordering::SeqCst);
                    let answer = answers[hit.min(answers.len() - 1)];
                    let sink = Arc::clone(&sink);
                    async move {
                        *sink.lock() = raw;
                        answer
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
            .route("/metrics", axum::routing::get(|| async { ENGINE_METRICS }));

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
        spawn_backend(&[(StatusCode::NOT_FOUND, "")], &[(status, body)]).await
    }

    /// A gateway registered as a worker: serves `/loads` only.
    async fn spawn_gateway(body: &'static str) -> Stub {
        spawn_backend(&[(StatusCode::OK, body)], &[(StatusCode::NOT_FOUND, "")]).await
    }

    fn gauge_worker(url: &str) -> Arc<dyn Worker> {
        gauge_worker_with_overload(url, OverloadUpdate::default())
    }

    /// Build an HTTP virtual rank whose probes must use the base endpoint.
    fn gauge_dp_worker(url: &str, rank: usize, size: usize) -> Arc<dyn Worker> {
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
    fn gauge_worker_with_overload(url: &str, overload: OverloadUpdate) -> Arc<dyn Worker> {
        gauge_worker_built(url, "a", overload)
    }

    fn gauge_worker_with_model(url: &str, model: &str) -> Arc<dyn Worker> {
        gauge_worker_built(url, model, OverloadUpdate::default())
    }

    fn gauge_worker_built(url: &str, model: &str, overload: OverloadUpdate) -> Arc<dyn Worker> {
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

    fn monitor_with_readers(readers: LoadReaders) -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>) {
        monitor_with(readers, false)
    }

    fn monitor_with(
        readers: LoadReaders,
        conditional_polling: bool,
    ) -> (Arc<WorkerRegistry>, Arc<WorkerMonitor>) {
        let registry = Arc::new(WorkerRegistry::new());
        let monitor = Arc::new(WorkerMonitor::new(
            registry.clone(),
            Arc::new(readers),
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

    /// A routing side that records the reconciliation ticks it receives:
    /// the worker's URL and the requests the router holds on it.
    #[derive(Debug, Default)]
    struct ReconcileRecorder(Mutex<Vec<(String, usize)>>);

    impl LoadConsumer for ReconcileRecorder {
        fn needs_loads(&self) -> bool {
            false
        }

        fn reconcile_in_flight(&self, worker: &dyn Worker) {
            self.0
                .lock()
                .push((worker.url().to_string(), worker.load()));
        }
    }

    /// Every poll hands the routing side each polled worker, with the
    /// router's live in-flight count on it: the safety net for a completion
    /// the routing side never saw.
    #[tokio::test]
    async fn load_poll_reconciles_in_flight_with_the_routing_side() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let registry = Arc::new(WorkerRegistry::new());
        let recorder = Arc::new(ReconcileRecorder::default());
        let monitor = Arc::new(WorkerMonitor::new(
            registry.clone(),
            Arc::clone(&recorder),
            1,
            false,
            false,
        ));
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
        let held = WorkerLoadGuard::with_key(Arc::clone(&worker), None);
        registry.register(Arc::clone(&worker)).unwrap();
        monitor.start_event_loop();

        wait_until("the poll to reconcile the routing side", || {
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
    /// the worker itself, beside a routing side that reads no loads at
    /// all. The threshold rides the worker's spec block, so this also proves a
    /// spec block alone enables protection with no gateway flags anywhere.
    #[tokio::test]
    async fn load_ingestion_flags_worker_over_waiting_threshold() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(LoadReaders::Blind, false);
        let worker = gauge_worker_with_overload(
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
        let (registry, monitor) = monitor_with(LoadReaders::Blind, false);
        let worker = gauge_worker_with_overload(
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
        let (registry, monitor) = monitor_with(LoadReaders::Blind, true);
        let worker = gauge_worker_with_overload(
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
        // A load-aware reader, so loads are polled and ingested for reasons
        // unrelated to overload protection.
        let (registry, monitor) = monitor_with(LoadReaders::Aware, false);
        let worker = gauge_worker(&stub.url);
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
        let (registry, monitor) = monitor_with(LoadReaders::Blind, false);
        // Never-registered address: every fetch fails, so the report is absent.
        let worker = gauge_worker_with_overload(
            "http://127.0.0.1:1",
            OverloadUpdate {
                waiting_requests: Some(1),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();

        // The loop's start resets every flag, so this one is set after it
        // and the failed fetch alone is left to clear it.
        monitor.start_event_loop();
        registry.set_worker_overloaded(&worker, true);
        assert_eq!(registry.overloaded_worker_count("a"), 1);

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
        let (registry, monitor) = monitor_with(LoadReaders::Blind, false);
        let worker = gauge_worker_with_overload(
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

    /// A load-aware reader must still be fed with the opt-out set — the
    /// conditional gate is "poll when needed", never "never poll".
    #[tokio::test]
    async fn a_load_aware_reader_starts_load_polling_under_the_conditional_gate() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with(LoadReaders::Aware, true);
        let worker = gauge_worker(&stub.url);
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
        .expect("a load-aware reader must trigger load polling under the conditional gate");
    }

    /// The one predicate behind the poll tick and the load stream: wanted
    /// by default; under `--disable-load-monitoring` only for a reason.
    #[tokio::test]
    async fn loads_wanted_follows_the_conditional_gate() {
        let (_registry, by_default) = monitor_with(LoadReaders::Blind, false);
        assert!(by_default.loads_wanted(&[gauge_worker("http://a:8000")]));
        let (_registry, conditional) = monitor_with(LoadReaders::Blind, true);
        assert!(!conditional.loads_wanted(&[gauge_worker("http://a:8000")]));
        assert!(
            conditional.loads_wanted(&[gauge_worker_with_overload(
                "http://a:8000",
                OverloadUpdate {
                    waiting_requests: Some(1),
                    ..Default::default()
                }
            )]),
            "overload protection on a member wants the loads"
        );
        let (_registry, load_aware) = monitor_with(LoadReaders::Aware, true);
        assert!(load_aware.loads_wanted(&[gauge_worker("http://a:8000")]));
    }

    /// Load monitoring is on by default: a load-blind policy with no flags and
    /// no spec blocks still gets its workers polled from registration onward.
    #[tokio::test]
    async fn load_blind_policy_polls_by_default() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
        let worker = gauge_worker(&stub.url);
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
        let (registry, monitor) = monitor_with(LoadReaders::Blind, true);
        let worker = gauge_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(worker).unwrap();
        monitor.start_event_loop();

        // Covers the immediate first tick plus one full 1s interval.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert_eq!(stub.probes.load(Ordering::SeqCst), 0);
        assert!(monitor.load_state.snapshot().is_empty());
    }

    #[tokio::test]
    async fn native_loads_preferred_over_the_gauge_page() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let worker = gauge_worker(&stub.url);
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

    /// Seventy named values for rank 0, and for rank 1 a `values` key of
    /// the engine's own shape.
    fn named_body() -> &'static str {
        let entries: Vec<String> = (0..70)
            .map(|index| format!(r#"{{"name":"ready_{index}","unit":"items","value":{index}}}"#))
            .collect();
        Box::leak(
            format!(
                r#"{{"dp_rank_count":2,"loads":[
                {{"dp_rank":0,"num_running_reqs":3,"values":[{}]}},
                {{"dp_rank":1,"num_running_reqs":9,"values":{{"depth":3}}}}]}}"#,
                entries.join(",")
            )
            .into_boxed_str(),
        )
    }

    /// One tick's fetch: each member with its report.
    type Answers = Vec<(Arc<dyn Worker>, Option<WorkerLoadResponse>)>;

    /// One tick's fetch for `members` of the engine behind `body`, with
    /// the scrape its counters wrote.
    fn fetched(
        body: &'static str,
        members: impl FnOnce(&str) -> Vec<Arc<dyn Worker>>,
    ) -> (Answers, String) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime");
        let stub = runtime.block_on(spawn_engine(StatusCode::OK, body));
        let members = members(&stub.url);
        let memo = DashMap::new();
        let mut answers = Vec::new();
        let rendered = render_with_recorder(|| {
            answers = runtime.block_on(WorkerMonitor::fetch_engine_loads(
                members,
                ConnectionMode::Http,
                &memo,
                &PollCounts::default(),
            ));
        });
        (answers, rendered)
    }

    /// A host's load source that answers what it was made with, or fails
    /// where it was made with nothing, and counts its calls.
    #[derive(Debug, Default)]
    struct HostLoads {
        answer: Option<WorkerLoadResponse>,
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl LoadSource for HostLoads {
        async fn get_loads(&self) -> Result<WorkerLoadResponse, tonic::Status> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.answer
                .clone()
                .ok_or_else(|| tonic::Status::unavailable("the host's service is down"))
        }
    }

    /// A host's connector for HTTP workers: it keeps the source it was made
    /// with, or none, and notes each address it is asked a source for.
    #[derive(Debug, Default)]
    struct SourceConnector {
        source: Option<Arc<HostLoads>>,
        asked: Mutex<Vec<String>>,
        /// How many of the next asks never answer.
        silent_asks: AtomicUsize,
    }

    impl SourceConnector {
        fn keeping(answer: Option<WorkerLoadResponse>) -> Arc<Self> {
            Arc::new(Self {
                source: Some(Arc::new(HostLoads {
                    answer,
                    calls: AtomicUsize::new(0),
                })),
                asked: Mutex::new(Vec::new()),
                silent_asks: AtomicUsize::new(0),
            })
        }

        fn calls(&self) -> usize {
            self.source
                .as_ref()
                .map_or(0, |source| source.calls.load(Ordering::Relaxed))
        }
    }

    #[async_trait::async_trait]
    impl EngineConnector for SourceConnector {
        async fn connect_grpc(
            &self,
            _base_url: &str,
            _runtime: RuntimeType,
        ) -> Result<Arc<dyn EngineConnection>, String> {
            Err("an HTTP worker connects no engine".to_string())
        }

        async fn connect_zmq(&self, _link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
            Err("an HTTP worker connects no engine".to_string())
        }

        async fn dial_grpc(&self, _base_url: &str, _timeout: Duration) -> Result<(), String> {
            Err("an HTTP worker dials no engine".to_string())
        }

        async fn http_load_source(
            &self,
            base_url: &str,
            _runtime: RuntimeType,
        ) -> Option<Arc<dyn LoadSource>> {
            self.asked.lock().push(base_url.to_string());
            // One poll asks at a time in these cases: a read, then a write.
            if self.silent_asks.load(Ordering::Relaxed) > 0 {
                self.silent_asks.fetch_sub(1, Ordering::Relaxed);
                std::future::pending::<()>().await;
            }
            let source = self.source.clone()?;
            Some(source)
        }
    }

    /// An HTTP worker, or one rank of an HTTP engine, built with a host's
    /// connector.
    fn hosted_worker(
        url: &str,
        connector: &Arc<SourceConnector>,
        rank: Option<(usize, usize)>,
    ) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Http)
            .runtime_type(RuntimeType::Vllm)
            .model(ModelCard::new("a"))
            .engine_connector(Arc::clone(connector) as Arc<dyn EngineConnector>)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            });
        if let Some((rank, size)) = rank {
            builder = builder.dp_config(rank, size);
        }
        Arc::new(builder.build())
    }

    fn queue_age(value: f64) -> NamedValue {
        NamedValue {
            name: "queue_age".into(),
            unit: "ms".into(),
            value,
            age_ms: Some(37),
        }
    }

    /// What a host's service answers for an engine of `ranks` ranks: rank
    /// `k` runs `41 + 2k` requests and states one named value, beside an
    /// entry with no name that the report's bounds drop.
    fn host_answer(ranks: i32) -> WorkerLoadResponse {
        let loads = (0..ranks)
            .map(|rank| SchedulerLoadSnapshot {
                dp_rank: rank,
                num_running_reqs: 41 + 2 * rank,
                values: vec![
                    queue_age(6.5 - f64::from(rank)),
                    NamedValue {
                        name: "".into(),
                        ..queue_age(1.0)
                    },
                ],
                ..Default::default()
            })
            .collect();
        WorkerLoadResponse {
            dp_rank_count: ranks,
            loads,
            ..Default::default()
        }
    }

    async fn engine_fetch(
        members: Vec<Arc<dyn Worker>>,
        memo: &DashMap<String, NativeLoadsMemo>,
    ) -> Vec<Option<WorkerLoadResponse>> {
        WorkerMonitor::fetch_engine_loads(
            members,
            ConnectionMode::Http,
            memo,
            &PollCounts::default(),
        )
        .await
        .into_iter()
        .map(|(_, report)| report)
        .collect()
    }

    /// An HTTP worker whose host keeps a load source reports the source's
    /// answer, its named values under the report's bounds, and no load
    /// request reaches the worker. The connector is asked once, for the
    /// worker's address, and the native loads memo stays empty.
    #[tokio::test]
    async fn an_http_worker_with_a_load_source_reports_its_answer_and_is_asked_nothing() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let connector = SourceConnector::keeping(Some(host_answer(1)));
        let worker = hosted_worker(&stub.url, &connector, None);
        let memo = DashMap::new();
        for _ in 0..2 {
            let report = engine_fetch(vec![Arc::clone(&worker)], &memo)
                .await
                .pop()
                .flatten()
                .expect("the source's answer");
            assert_eq!(report.loads.len(), 1);
            assert_eq!(report.loads[0].num_running_reqs, 41);
            assert_eq!(report.loads[0].values, [queue_age(6.5)]);
        }
        assert_eq!(connector.calls(), 2);
        assert_eq!(
            connector.asked.lock().as_slice(),
            std::slice::from_ref(&stub.url)
        );
        assert_eq!(stub.probes.load(Ordering::Relaxed), 0);
        assert!(memo.is_empty());
    }

    /// A report considers the ranks 0 to 63 of a worker registered for no
    /// rank, by either path: the rows its poll's answer states and the
    /// records its stream pushes. Rank 64 is dropped on both, and of a rank
    /// an answer states twice the first row stands: the bound is by rank,
    /// not by place, and the answer counts the ranks kept. The worker
    /// registered for rank 64 keeps it by either path, counted once where
    /// the engine states it twice.
    #[tokio::test]
    async fn a_load_report_considers_the_ranks_below_64() {
        let mut answer = host_answer(65);
        answer.loads.insert(0, answer.loads[3].clone());
        let connector = SourceConnector::keeping(Some(answer));
        let worker = hosted_worker("http://engine.invalid:8000", &connector, None);
        let polled = engine_fetch(vec![Arc::clone(&worker)], &DashMap::new())
            .await
            .pop()
            .flatten()
            .expect("the source's answer");
        let mut twice = host_answer(65);
        twice.loads.push(twice.loads[64].clone());
        let connector = SourceConnector::keeping(Some(twice));
        let ranked = hosted_worker("http://ranked.invalid:8000", &connector, Some((64, 65)));
        let kept = engine_fetch(vec![Arc::clone(&ranked)], &DashMap::new())
            .await
            .pop()
            .flatten()
            .expect("the source's answer");

        let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
        registry.register(Arc::clone(&worker)).unwrap();
        registry.register(Arc::clone(&ranked)).unwrap();
        for rank in 0..65 {
            monitor.apply_pushed_load(&worker, rank, &PushedRecord::requests(3, 0), Instant::now());
        }
        monitor.apply_pushed_load(&ranked, 64, &PushedRecord::requests(3, 0), Instant::now());
        let pushed = monitor.pushed_reports.lock();
        let filed = &pushed[ranked.url()].report.loads;
        let pushed = &pushed[worker.url()].report.loads;

        let last = |loads: &[SchedulerLoadSnapshot]| loads.last().map(|load| load.dp_rank);
        assert_eq!(
            (polled.loads.len(), polled.dp_rank_count, pushed.len()),
            (64, 64, 64)
        );
        assert_eq!((last(&polled.loads), last(pushed)), (Some(63), Some(63)));
        assert_eq!((kept.loads.len(), kept.dp_rank_count), (1, 1));
        assert_eq!((last(&kept.loads), last(filed)), (Some(64), Some(64)));
    }

    /// The rank workers of one engine share one call of their source: the
    /// first member asks, for the engine's own address, and each member's
    /// report is its own rank of the answer.
    #[tokio::test]
    async fn the_rank_workers_of_one_engine_share_one_call_of_their_source() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_DP_BODY).await;
        let connector = SourceConnector::keeping(Some(host_answer(2)));
        let members = vec![
            hosted_worker(&stub.url, &connector, Some((0, 2))),
            hosted_worker(&stub.url, &connector, Some((1, 2))),
        ];
        let ranks: Vec<(i32, i32, f64)> = engine_fetch(members, &DashMap::new())
            .await
            .into_iter()
            .map(|report| {
                let report = report.expect("its rank's report");
                assert_eq!(report.loads.len(), 1);
                let rank = &report.loads[0];
                assert_eq!(rank.values.len(), 1);
                (rank.dp_rank, rank.num_running_reqs, rank.values[0].value)
            })
            .collect();
        assert_eq!(ranks, [(0, 41, 6.5), (1, 43, 5.5)]);
        assert_eq!(connector.calls(), 1);
        assert_eq!(
            connector.asked.lock().as_slice(),
            std::slice::from_ref(&stub.url)
        );
        assert_eq!(stub.probes.load(Ordering::Relaxed), 0);
    }

    /// A member's copy of its rank is the whole answer scoped to that rank,
    /// field for field: a rank of its own, one the answer states twice, one
    /// it leaves out, one past the entries' type, a rollup by worker, and
    /// no rank at all.
    #[test]
    fn a_members_copy_of_its_rank_is_the_answer_scoped_to_that_rank() {
        let mut twice = host_answer(3);
        twice.loads[2].dp_rank = 0;
        twice.timestamp = "2026-01-02T03:04:05Z".to_owned();
        twice.version = "engine-7".to_owned();
        twice.aggregate = Some(Default::default());
        twice.sampled_at = Some(Instant::now());
        let mut rollup = host_answer(2);
        rollup.loads[1].worker = Some("http://engine-b".to_owned());
        for answer in [host_answer(3), twice, rollup, WorkerLoadResponse::default()] {
            for rank in [None, Some(0), Some(1), Some(2), Some(usize::MAX)] {
                assert_eq!(
                    format!("{:?}", WorkerMonitor::copy_for_dp_rank(&answer, rank)),
                    format!("{:?}", answer.clone().for_dp_rank(rank)),
                    "rank {rank:?}"
                );
            }
        }
        // Its list has room for its own row alone: a copy of the whole
        // answer keeps room for every rank's.
        let own = WorkerMonitor::copy_for_dp_rank(&host_answer(3), Some(1));
        assert_eq!(own.map(|own| own.loads.capacity()), Some(1));
    }

    /// A connector that keeps no source is asked once, and its worker is
    /// then polled over HTTP as a worker with no connector is: one request
    /// a poll, the engine's own answer, the same memo.
    #[tokio::test]
    async fn a_worker_whose_host_keeps_no_source_is_polled_over_http_as_before() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let connector = Arc::new(SourceConnector::default());
        let memo = DashMap::new();
        let workers = [
            hosted_worker(&stub.url, &connector, None),
            gauge_worker(&stub.url),
        ];
        let mut probes = 0;
        for worker in workers {
            for _ in 0..2 {
                let report = engine_fetch(vec![Arc::clone(&worker)], &memo)
                    .await
                    .pop()
                    .flatten()
                    .expect("the engine's own answer");
                probes += 1;
                assert_eq!(report.loads[0].num_waiting_uncached_tokens, 900);
                assert_eq!(stub.probes.load(Ordering::Relaxed), probes);
            }
        }
        assert_eq!(
            connector.asked.lock().as_slice(),
            std::slice::from_ref(&stub.url)
        );
        assert_eq!(
            memo.get(&stub.url).and_then(|hit| hit.answered()),
            Some(NativeLoadsPath::Engine)
        );
    }

    /// A source that fails is a missing report. No request goes to the
    /// worker in its place, and nothing is said of the worker's health: no
    /// contact is noted and none is failed.
    #[tokio::test]
    async fn a_load_source_that_fails_is_a_missing_report_and_no_word_on_health() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let connector = SourceConnector::keeping(None);
        let worker = hosted_worker(&stub.url, &connector, None);
        let before = (worker.status(), worker.stall_reason());
        let memo = DashMap::new();
        for _ in 0..2 {
            let reports = engine_fetch(vec![Arc::clone(&worker)], &memo).await;
            assert!(matches!(reports.as_slice(), [None]));
        }
        assert_eq!(connector.calls(), 2);
        assert_eq!(stub.probes.load(Ordering::Relaxed), 0);
        assert!(memo.is_empty());
        assert_eq!((worker.status(), worker.stall_reason()), before);
    }

    /// A host's connector whose gRPC connect never returns.
    #[derive(Debug)]
    struct SilentConnect;

    #[async_trait::async_trait]
    impl EngineConnector for SilentConnect {
        async fn connect_grpc(
            &self,
            _base_url: &str,
            _runtime: RuntimeType,
        ) -> Result<Arc<dyn EngineConnection>, String> {
            std::future::pending().await
        }

        async fn connect_zmq(&self, _link: ZmqLink) -> Result<Arc<dyn EngineConnection>, String> {
            Err("a gRPC worker runs no handshake".to_string())
        }

        async fn dial_grpc(&self, _base_url: &str, _timeout: Duration) -> Result<(), String> {
            Ok(())
        }
    }

    /// A gRPC worker of `model` whose connect never returns.
    fn silent_worker(url: &str, model: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .model(ModelCard::new(model))
                .engine_connector(Arc::new(SilentConnect))
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    /// A drop alone does not end a monitor whose polls outlast its period.
    /// Each of two groups polls a worker whose connect never returns, so
    /// each poll lasts the poll's deadline, three periods. Each group's
    /// loop takes the monitor again while the other's poll holds it, and
    /// the monitor polls on with no host. A stop ends every loop, and the
    /// monitor then ends with its host's handle.
    #[tokio::test(start_paused = true)]
    async fn a_stop_ends_a_monitor_that_its_own_polls_keep_past_its_drop() {
        let (registry, monitor) = monitor_with_readers(LoadReaders::Aware);
        for (url, model) in [
            ("grpc://engine-a.invalid:50051", "a"),
            ("grpc://engine-b.invalid:50051", "b"),
        ] {
            registry.register(silent_worker(url, model)).unwrap();
        }
        monitor.start_event_loop();
        // Each group's first poll is out.
        tokio::time::sleep(Duration::from_millis(1)).await;

        let dropped = Arc::downgrade(&monitor);
        drop(monitor);
        tokio::time::sleep(LOAD_POLL_DEADLINE * 4).await;
        let monitor = dropped
            .upgrade()
            .expect("the dropped monitor's own polls keep it");

        monitor.stop().await;
        drop(monitor);
        assert!(
            dropped.upgrade().is_none(),
            "a stopped monitor outlived its host's handle"
        );
    }

    /// A stop dropped at a wait leaves no task of the monitor running: each
    /// was told to end ahead of that wait. Each group's poll is in flight
    /// and holds the monitor. The stop is polled until it has taken the
    /// groups' loops, and is dropped at its wait for the tasks' ends.
    #[tokio::test(start_paused = true)]
    async fn a_stop_dropped_at_a_wait_leaves_no_task_running() {
        let (registry, monitor) = monitor_with_readers(LoadReaders::Aware);
        for (url, model) in [
            ("grpc://engine-a.invalid:50051", "a"),
            ("grpc://engine-b.invalid:50051", "b"),
        ] {
            registry.register(silent_worker(url, model)).unwrap();
        }
        monitor.start_event_loop();
        // Each group's first poll is out, and holds the monitor.
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert!(Arc::strong_count(&monitor) > 1);

        {
            let stop = monitor.stop();
            tokio::pin!(stop);
            while !monitor.group_handles.lock().is_empty() {
                let ended = tokio::select! {
                    biased;
                    () = &mut stop => true,
                    () = tokio::task::yield_now() => false,
                };
                assert!(!ended, "the stop ended ahead of its drop");
            }
        }
        // A task told to end is gone at its next turn.
        tokio::time::sleep(Duration::from_millis(1)).await;
        assert_eq!(
            Arc::weak_count(&monitor),
            0,
            "a task of the monitor runs on"
        );
        assert_eq!(
            Arc::strong_count(&monitor),
            1,
            "a loop's poll still holds the monitor"
        );
    }

    /// A monitor told `interval` as its own, over one group of HTTP workers
    /// whose host keeps one load source for them all. Each entry of `own`
    /// is a worker: the whole seconds it states for itself, or nothing. The
    /// source's calls are the polls that reached it.
    fn polled_every(
        interval: Duration,
        own: &[Option<u64>],
    ) -> (Arc<SourceConnector>, Arc<WorkerMonitor>) {
        let connector = SourceConnector::keeping(Some(host_answer(1)));
        let registry = Arc::new(WorkerRegistry::new());
        for (at, stated) in own.iter().enumerate() {
            let mut spec = WorkerSpec::new(format!("http://engine-{at}.invalid:8000"));
            spec.load_monitor_interval_secs = *stated;
            let worker = BasicWorkerBuilder::from_spec(spec)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .runtime_type(RuntimeType::Vllm)
                .model(ModelCard::new("a"))
                .engine_connector(Arc::clone(&connector) as Arc<dyn EngineConnector>)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build();
            registry.register(Arc::new(worker)).unwrap();
        }
        let monitor = WorkerMonitor::new(registry, Arc::new(LoadReaders::Aware), 1, false, false)
            .with_interval(interval);
        (connector, Arc::new(monitor))
    }

    /// A monitor told an interval under a second polls at it: once at its
    /// start, then four times a second at a quarter second.
    #[tokio::test(start_paused = true)]
    async fn a_monitor_told_a_quarter_second_polls_four_times_a_second() {
        let (connector, monitor) = polled_every(Duration::from_millis(250), &[None]);
        assert_eq!(monitor.interval(), Duration::from_millis(250));
        monitor.start_event_loop();
        // Between the first poll and the second.
        tokio::time::sleep(Duration::from_millis(125)).await;
        assert_eq!(connector.calls(), 1);
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert_eq!(connector.calls(), 5);
    }

    /// An interval under the least is raised to it: told ten milliseconds,
    /// a monitor polls ten times a second.
    #[tokio::test(start_paused = true)]
    async fn an_interval_under_the_least_is_raised_to_it() {
        let (connector, monitor) = polled_every(Duration::from_millis(10), &[None]);
        assert_eq!(monitor.interval(), LEAST_POLL_INTERVAL);
        monitor.start_event_loop();
        tokio::time::sleep(LEAST_POLL_INTERVAL / 2).await;
        assert_eq!(connector.calls(), 1);
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert_eq!(connector.calls(), 11);
    }

    /// A worker's own interval stands over the monitor's. One worker of a
    /// group states one second and the other states nothing: a monitor told
    /// a quarter second polls the two each second.
    #[tokio::test(start_paused = true)]
    async fn a_workers_own_second_stands_over_the_monitors_quarter_second() {
        let (connector, monitor) = polled_every(Duration::from_millis(250), &[Some(1), None]);
        monitor.start_event_loop();
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(connector.calls(), 2);
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert_eq!(connector.calls(), 6);
    }

    /// The ticks a slow poll overran are skipped, at every interval. The
    /// first ask of the host's connector never answers, so the first poll
    /// lasts the poll's deadline: thirty ticks at the least interval, three
    /// at one second. One poll follows it at once and the next comes one
    /// interval later. The timer's own default sends a poll for every tick
    /// overrun, one behind another.
    #[tokio::test(start_paused = true)]
    async fn the_ticks_a_slow_poll_overran_are_skipped_at_every_interval() {
        for interval in [LEAST_POLL_INTERVAL, Duration::from_secs(1)] {
            let (connector, monitor) = polled_every(interval, &[None]);
            connector.silent_asks.store(1, Ordering::Relaxed);
            monitor.start_event_loop();
            tokio::time::sleep(LOAD_POLL_DEADLINE + interval / 2).await;
            assert_eq!(
                connector.calls(),
                1,
                "the polls behind a slow poll at {interval:?}"
            );
            tokio::time::sleep(interval).await;
            assert_eq!(connector.calls(), 2, "the next tick at {interval:?}");
            monitor.stop().await;
        }
    }

    /// What polling costs at the least interval, for a group of eight
    /// workers and of sixty-four. A tick asks every worker's engine once
    /// and publishes one snapshot: each worker is asked ten times a second,
    /// and the group publishes ten snapshots a second whatever its size.
    #[tokio::test(start_paused = true)]
    async fn a_group_at_the_least_interval_asks_each_worker_ten_times_a_second() {
        for workers in [8, 64] {
            let (connector, monitor) = polled_every(LEAST_POLL_INTERVAL, &vec![None; workers]);
            monitor.start_event_loop();
            // Between the first tick and the second.
            tokio::time::sleep(LEAST_POLL_INTERVAL / 2).await;
            let (asked, published) = (connector.calls(), monitor.load_state.publish_count());
            assert_eq!(asked, workers, "the first tick of {workers} workers");
            tokio::time::sleep(Duration::from_secs(1)).await;
            assert_eq!(
                (
                    connector.calls() - asked,
                    monitor.load_state.publish_count() - published
                ),
                (10 * workers, 10),
                "one second of {workers} workers"
            );
            monitor.stop().await;
        }
    }

    /// A routing side that asks the worker at one URL for no load, and reads
    /// no load itself.
    #[derive(Debug)]
    struct LeavesOut(String);

    impl LoadConsumer for LeavesOut {
        fn needs_loads(&self) -> bool {
            false
        }

        fn reconcile_in_flight(&self, _worker: &dyn Worker) {}

        fn polls(&self, worker: &dyn Worker) -> bool {
            worker.url() != self.0
        }
    }

    /// A worker its host asks for no load is no member of a tick: no poll
    /// goes to it, it has no published report, and no load stream is
    /// wanted for it. The group's other worker is polled as before. Under
    /// polling by need its overload protection asks for no load of its
    /// group.
    #[tokio::test]
    async fn a_worker_its_host_leaves_out_is_asked_for_no_load() {
        let asked = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let left_out = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let registry = Arc::new(WorkerRegistry::new());
        let monitor = Arc::new(WorkerMonitor::new(
            Arc::clone(&registry),
            Arc::new(LeavesOut(left_out.url.clone())),
            1,
            false,
            false,
        ));
        let workers = [gauge_worker(&asked.url), gauge_worker(&left_out.url)];
        for worker in &workers {
            registry.register(Arc::clone(worker)).unwrap();
        }
        let key = WorkerGroupKey {
            model_id: "a".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
        };
        poll_group_once(&monitor, &key, Duration::from_secs(1), Instant::now()).await;

        assert_eq!(asked.probes.load(Ordering::SeqCst), 1);
        assert_eq!(left_out.probes.load(Ordering::SeqCst), 0);
        assert_eq!(left_out.gateway_probes.load(Ordering::SeqCst), 0);
        let snapshot = monitor.load_state.snapshot();
        assert!(snapshot.get(&asked.url).is_some());
        assert!(snapshot.get(&left_out.url).is_none());
        assert!(monitor.loads_wanted(&workers[..1]));
        assert!(
            !monitor.loads_wanted(&workers[1..]),
            "a load stream is wanted for a worker left out"
        );
        assert!(monitor.loads_wanted(&workers));

        // Under polling by need a member left out counts for nothing: its
        // overload protection asks for no load of its group. Asked, it does.
        let guarded = gauge_worker_with_overload(
            &left_out.url,
            OverloadUpdate {
                waiting_requests: Some(5),
                token_usage: None,
            },
        );
        let group = [Arc::clone(&workers[0]), guarded];
        let by_need = |leaves_out: &str| {
            let routing = Arc::new(LeavesOut(leaves_out.to_string()));
            WorkerMonitor::new(Arc::clone(&registry), routing, 1, false, true)
        };
        assert!(
            !by_need(&left_out.url).loads_wanted(&group),
            "a member left out asks for its group's loads"
        );
        assert!(by_need("").loads_wanted(&group));
    }

    /// A gRPC worker whose connect never returns stalls no poll. Its engine
    /// has a missing report for the tick and holds no connection, and an
    /// engine fetched beside it has its report.
    #[tokio::test(start_paused = true)]
    async fn a_connect_that_never_returns_is_a_missing_report_and_stalls_no_other_engine() {
        let silent = silent_worker("grpc://engine-a.invalid:50051", "a");
        let prompt = SourceConnector::keeping(Some(host_answer(1)));
        let beside = hosted_worker("http://engine-b.invalid:8000", &prompt, None);
        let memo = DashMap::new();
        let counts = PollCounts::default();

        let (stalled, answered) = tokio::join!(
            WorkerMonitor::fetch_engine_loads(
                vec![Arc::clone(&silent)],
                ConnectionMode::Grpc,
                &memo,
                &counts,
            ),
            engine_fetch(vec![beside], &memo),
        );
        assert!(matches!(stalled.as_slice(), [(_, None)]));
        assert!(matches!(answered.as_slice(), [Some(_)]));
        assert!(silent.engine_connection_if_initialized().is_none());
    }

    /// A connector that does not say in time whether it keeps a source
    /// stalls no poll. Its engine has a missing report for the tick, no
    /// load request goes to the worker in the source's place, and another
    /// engine fetched beside it has its report. The next poll asks again
    /// and takes the source's answer.
    #[tokio::test(start_paused = true)]
    async fn a_connector_that_is_late_to_say_is_a_missing_report_and_asked_again() {
        let stub = spawn_engine(StatusCode::OK, NATIVE_BODY).await;
        let late = SourceConnector::keeping(Some(host_answer(1)));
        late.silent_asks.store(1, Ordering::Relaxed);
        let prompt = SourceConnector::keeping(Some(host_answer(1)));
        let worker = hosted_worker(&stub.url, &late, None);
        let beside = hosted_worker("http://engine-b.invalid:8000", &prompt, None);
        let memo = DashMap::new();

        let (stalled, answered) = tokio::join!(
            engine_fetch(vec![Arc::clone(&worker)], &memo),
            engine_fetch(vec![beside], &memo),
        );
        assert!(matches!(stalled.as_slice(), [None]));
        assert!(matches!(answered.as_slice(), [Some(_)]));
        assert_eq!(late.calls(), 0);
        assert_eq!(stub.probes.load(Ordering::Relaxed), 0);

        let report = engine_fetch(vec![worker], &memo)
            .await
            .pop()
            .flatten()
            .expect("the source's answer at the next poll");
        assert_eq!(report.loads[0].num_running_reqs, 41);
        assert_eq!(late.asked.lock().len(), 2);
        assert_eq!(late.calls(), 1);
        assert_eq!(stub.probes.load(Ordering::Relaxed), 0);
        assert!(memo.is_empty());
    }

    /// A polled answer's named values stand under the bounds a pushed
    /// record's do, and what left them is counted for the worker. A
    /// `values` key of another shape is the engine's own: its rank is
    /// read, with no list.
    #[test]
    fn a_polled_answers_named_values_are_bound_and_counted() {
        let (mut answers, rendered) = fetched(named_body(), |url| vec![gauge_worker(url)]);
        let (worker, report) = answers.pop().expect("the one member");
        let report = report.expect("the native answer");
        assert_eq!(report.loads.len(), 2);
        assert_eq!(report.loads[0].values.len(), 64);
        assert_eq!(report.loads[0].values[63].name.as_ref(), "ready_63");
        assert_eq!(report.loads[1].num_running_reqs, 9);
        assert!(report.loads[1].values.is_empty());
        assert_metric(
            &rendered,
            "smg_engine_load_values_dropped_total",
            &[&format!("worker=\"{}\"", worker.url())],
            "6",
        );
    }

    /// The rank workers of one engine share its answer. Each one's report
    /// is bound, and what left a rank's list is counted for that rank's
    /// worker alone.
    #[test]
    fn a_ranks_dropped_values_are_counted_for_its_own_worker() {
        let (answers, rendered) = fetched(named_body(), |url| {
            vec![gauge_dp_worker(url, 0, 2), gauge_dp_worker(url, 1, 2)]
        });
        let lists: Vec<usize> = answers
            .iter()
            .map(|(_, report)| {
                report.as_ref().expect("its rank's report").loads[0]
                    .values
                    .len()
            })
            .collect();
        assert_eq!(lists, [64, 0]);
        let series: Vec<&str> = rendered
            .lines()
            .filter(|line| line.starts_with("smg_engine_load_values_dropped_total{"))
            .collect();
        assert_eq!(series.len(), 1, "{rendered}");
        assert_metric(
            &rendered,
            "smg_engine_load_values_dropped_total",
            &[&format!("worker=\"{}\"", answers[0].0.url())],
            "6",
        );
    }

    /// A `values` key of another shape fails no native answer: the
    /// engine's route stays the worker's load source.
    #[tokio::test]
    async fn a_values_key_of_another_shape_keeps_the_native_route() {
        let stub = spawn_engine(
            StatusCode::OK,
            r#"{"loads":[{"dp_rank":0,"num_running_reqs":3,
                "num_waiting_uncached_tokens":900,"values":{"depth":3}}]}"#,
        )
        .await;
        let worker = gauge_worker(&stub.url);
        let memo = DashMap::new();

        let resp = WorkerMonitor::fetch_http_load(&worker, Some(&memo))
            .await
            .expect("load response");

        assert_eq!(resp.loads[0].num_waiting_uncached_tokens, 900);
        assert!(resp.loads[0].values.is_empty());
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
        let worker = gauge_dp_worker(&stub.url, 1, 2);

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
            let worker = gauge_dp_worker(&stub.url, 1, 2);
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
        let worker = gauge_dp_worker(&stub.url, 1, 2);

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
        let worker = gauge_worker(&stub.url);
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
        let worker = gauge_worker(&stub.url);
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
    async fn a_route_that_had_answered_survives_a_body_that_does_not_read() {
        let stub = spawn_backend(
            &[(StatusCode::NOT_FOUND, "")],
            &[
                (StatusCode::OK, NATIVE_BODY),
                (StatusCode::OK, UNREADABLE_NATIVE_BODY),
                (StatusCode::OK, NATIVE_BODY),
            ],
        )
        .await;
        let worker = gauge_worker(&stub.url);
        let memo = DashMap::new();
        let route = NativeLoadsPath::Engine;

        let resp = WorkerMonitor::fetch_http_load(&worker, Some(&memo))
            .await
            .ok();
        assert_eq!(
            resp.as_ref()
                .map(WorkerLoadResponse::total_waiting_uncached_tokens),
            Some(900),
            "first tick uses the native report"
        );
        assert_eq!(
            memo.get(worker.url()).and_then(|hit| hit.answered()),
            Some(route),
            "first tick records the answering route"
        );

        let resp = WorkerMonitor::fetch_http_load(&worker, Some(&memo))
            .await
            .ok();
        assert_eq!(
            resp.as_ref().map(WorkerLoadResponse::total_running_reqs),
            Some(7),
            "second tick uses the gauge report"
        );
        let hit = *memo.get(worker.url()).expect("second tick keeps the memo");
        assert_eq!(
            hit.answered(),
            Some(route),
            "second tick keeps the answering route"
        );
        assert!(!hit.is_absent(route), "route stays eligible");

        let resp = WorkerMonitor::fetch_http_load(&worker, Some(&memo))
            .await
            .ok();
        assert_eq!(
            resp.as_ref()
                .map(WorkerLoadResponse::total_waiting_uncached_tokens),
            Some(900),
            "third tick uses the native report again"
        );
        assert_eq!(
            stub.probes.load(Ordering::SeqCst),
            3,
            "the answering route is probed on all three ticks"
        );
        assert_eq!(
            stub.gateway_probes.load(Ordering::SeqCst),
            1,
            "the other route is probed only at discovery"
        );
    }

    #[tokio::test]
    /// Unrelated JSON and malformed load fields cannot suppress metrics fallback.
    async fn non_load_body_on_200_is_treated_as_absent() {
        for body in [
            UNREADABLE_NATIVE_BODY,
            "<html>nope</html>",
            "{}",
            r#"{"status":"ok"}"#,
            r#"{"loads":null}"#,
        ] {
            let stub = spawn_engine(StatusCode::OK, body).await;
            let worker = gauge_worker(&stub.url);
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
        let worker = gauge_worker(&stub.url);
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
        let worker = gauge_worker(&stub.url);

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
        let worker = gauge_worker(&stub.url);
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
        let worker = gauge_worker(&stub.url);
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
        let worker = gauge_worker_with_model(&stub.url, "meta-llama/Llama-3.1-8B");

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
            let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
            let worker = gauge_dp_worker(&stub.url, 1, 2);
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
            let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
            let worker = gauge_worker(&stub.url);
            worker.set_status(WorkerStatus::Ready);
            registry.register(Arc::clone(&worker)).unwrap();
            let interval = Duration::from_secs(10);
            let t0 = Instant::now();
            let polled_at = t0 + interval;
            for rank in 0..2 {
                monitor.apply_pushed_load(&worker, rank, &PushedRecord::default(), t0);
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
                monitor.apply_pushed_load(&worker, 1, &PushedRecord::default(), polled_at + offset);
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
            assert_eq!(monitor.has_receipt(worker.url(), 1), fresh_offset.is_some());
            monitor.apply_pushed_load(
                &worker,
                0,
                &PushedRecord::default(),
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
        let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
        let workers = [
            gauge_dp_worker(&stub.url, 0, 2),
            gauge_dp_worker(&stub.url, 1, 2),
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
            worker.runtime_state().unwrap().note_transport_failure();
        }
        poll_group_once(&monitor, &key, Duration::from_secs(10), Instant::now()).await;

        let snapshot = monitor.load_state.snapshot();
        for (worker, expected) in workers.iter().zip([(0, 3, 1, 300), (1, 9, 4, 900)]) {
            assert!(
                !worker.runtime_state().unwrap().transport_failure_pending(),
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
        let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
        let worker = gauge_worker(&stub.url);
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let key = WorkerGroupKey {
            model_id: "a".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
        };
        let interval = Duration::from_secs(10);
        let t0 = Instant::now();
        let record = |running: i32, hit_rate: f64| {
            PushedRecord::requests(running, 0).cache_hit_rate(hit_rate)
        };
        monitor.apply_pushed_load(&worker, 0, &record(2, 0.5), t0);
        monitor.apply_pushed_load(&worker, 1, &record(8, 0.65), t0);

        // One interval later rank 1's record is older than the interval:
        // the tick polls the worker.
        poll_group_once(&monitor, &key, interval, t0 + interval).await;
        assert_eq!(stub.probes.load(Ordering::SeqCst), 1, "one fallback poll");

        // The stream's next record names rank 0 only.
        monitor.apply_pushed_load(
            &worker,
            0,
            &record(4, 0.5),
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
        let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
        let worker = gauge_worker("http://w1:8000");
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let t0 = Instant::now();
        let record = |running: i32| PushedRecord::requests(running, 0);
        monitor.apply_pushed_load(&worker, 1, &record(8), t0);
        let last = t0 + Duration::from_millis(5);
        monitor.apply_pushed_load(&worker, 0, &record(2), last);
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

    /// A record's age is the engine's word. One the one-way margin overflows,
    /// or one older than the clock, keeps its record and reads as sampled at
    /// the receipt. No wire message carries such an age, so the case states
    /// both on a plain record.
    #[tokio::test]
    async fn an_age_past_the_clock_reads_as_sampled_at_the_receipt() {
        for age in [Duration::MAX, Duration::MAX - PUSHED_ONE_WAY_MARGIN] {
            let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
            let worker = gauge_worker("http://w1:8000");
            worker.set_status(WorkerStatus::Ready);
            registry.register(Arc::clone(&worker)).unwrap();
            let received = Instant::now();
            let record = PushedRecord {
                age,
                ..PushedRecord::requests(3, 0)
            };
            monitor.apply_pushed_load(&worker, 0, &record, received);
            monitor.flush_pushed();
            let snapshot = monitor.load_state.snapshot();
            let (report, sampled_at) = snapshot.report(worker.url()).expect("published");
            assert_eq!(report.loads[0].num_running_reqs, 3, "the record is kept");
            assert_eq!(sampled_at, received, "an age of {age:?}");
        }
    }

    /// A record for a rank on file replaces the rank's entry in place: the
    /// figures are the record's, the telemetry it left out stays what the
    /// last record that carried it said, and the report keeps one entry per
    /// rank.
    #[tokio::test]
    async fn a_record_for_a_rank_on_file_replaces_its_entry_in_place() {
        let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
        let worker = gauge_worker("http://w1:8000");
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let t0 = Instant::now();
        let heartbeat = PushedRecord::requests(2, 0)
            .cache_hit_rate(0.75)
            .used_tokens(1024)
            .total_tokens(4096);
        monitor.apply_pushed_load(&worker, 0, &heartbeat, t0);
        let core = PushedRecord::requests(5, 0).used_tokens(2048);
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
    }

    /// What a fallback poll leaves of a worker whose rank 0 pushed a record
    /// after the tick began: the published rows (rank, running requests),
    /// the published rank count and sample instant, the instant of the push,
    /// the overload verdict, the worker's two entries in the rank cache
    /// (`None` when the cache holds nothing of the worker), and the rows the
    /// worker's next pushed record publishes from the report on file.
    struct AfterFallbackPoll {
        rows: Vec<(i32, i32)>,
        rank_count: i32,
        sampled_at: Option<Instant>,
        pushed_at: Instant,
        overloaded: bool,
        cached: Option<(Option<isize>, Option<isize>)>,
        next_rows: Vec<(i32, i32)>,
    }

    /// Two ranks with a pushed record on file, rank 1's aged by the tick so
    /// the worker is polled. Rank 0 pushes a record after the tick's
    /// instant, before the poll's response is read: ten running and eight
    /// waiting, over the worker's threshold. Then the tick's fallback poll
    /// reads `answer`, and rank 1 pushes its next record.
    async fn fallback_poll_after_rank_0_pushed(answer: &'static str) -> AfterFallbackPoll {
        let stub = spawn_engine(StatusCode::OK, answer).await;
        let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
        let worker = gauge_worker_with_overload(
            &stub.url,
            OverloadUpdate {
                waiting_requests: Some(4),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let key = WorkerGroupKey {
            model_id: "a".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
        };
        let interval = Duration::from_secs(10);
        // The tick lies a second back, so the record pushed just after it
        // was sampled before the poll's response is read.
        let tick = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("the clock is older than a second");
        let t0 = tick
            .checked_sub(interval)
            .expect("the clock is older than the interval");
        let record = |running: i32, waiting: i32, used: i32| {
            PushedRecord::requests(running, waiting)
                .used_tokens(used)
                .total_tokens(1000)
        };
        // What an earlier poll left in the rank cache. No record writes
        // there, so the tick's poll is what replaces this entry or takes it
        // away.
        monitor
            .worker_load_manager
            .update_dp_loads(&HashMap::from([(
                stub.url.clone(),
                HashMap::from([(0, 150), (1, 250)]),
            )]));
        monitor.apply_pushed_load(&worker, 0, &record(2, 0, 200), t0);
        monitor.apply_pushed_load(&worker, 1, &record(8, 0, 500), t0);

        let pushed_at = tick + Duration::from_millis(1);
        monitor.apply_pushed_load(&worker, 0, &record(10, 8, 900), pushed_at);
        monitor.flush_pushed();
        let published = |monitor: &WorkerMonitor| {
            let snapshot = monitor.load_state.snapshot();
            let report = snapshot.get(stub.url.as_str()).expect("published");
            let rows: Vec<(i32, i32)> = report
                .loads
                .iter()
                .map(|rank| (rank.dp_rank, rank.num_running_reqs))
                .collect();
            (rows, report.dp_rank_count, report.sampled_at)
        };
        assert_eq!(
            published(&monitor).0,
            vec![(0, 10), (1, 8)],
            "the stream's publish"
        );
        assert!(worker.is_overloaded(), "the pushed record's verdict");

        poll_group_once(&monitor, &key, interval, tick).await;
        assert_eq!(stub.probes.load(Ordering::SeqCst), 1, "one fallback poll");
        let (rows, rank_count, sampled_at) = published(&monitor);
        let overloaded = worker.is_overloaded();
        let cached = {
            let cached = monitor.worker_load_manager.dp_cached_loads.read();
            cached
                .get(stub.url.as_str())
                .map(|ranks| (ranks.get(&0).copied(), ranks.get(&1).copied()))
        };
        // The worker's next record publishes the report on file.
        let next = pushed_at + Duration::from_millis(1);
        monitor.apply_pushed_load(&worker, 1, &record(8, 0, 500), next);
        monitor.flush_pushed();
        AfterFallbackPoll {
            rows,
            rank_count,
            sampled_at,
            pushed_at,
            overloaded,
            cached,
            next_rows: published(&monitor).0,
        }
    }

    /// A rank that pushed a record after the tick began keeps the stream's
    /// row in what the tick's fallback poll publishes, as the report on file
    /// keeps it: the poll's figure for that rank is the older one. The
    /// overload verdict and the DP-rank token cache read the same rows, and
    /// the report carries the earlier of the two sample instants.
    #[tokio::test]
    async fn a_row_pushed_during_a_fallback_poll_is_in_what_the_poll_publishes() {
        // The poll's answer: rank 0 idle, as it was before its record.
        const TWO_RANKS: &str = r#"{"loads":[
            {"dp_rank":0,"num_running_reqs":0,"num_waiting_reqs":0,
             "num_used_tokens":100,"max_total_num_tokens":1000},
            {"dp_rank":1,"num_running_reqs":8,"num_waiting_reqs":0,
             "num_used_tokens":500,"max_total_num_tokens":1000}]}"#;
        let after = fallback_poll_after_rank_0_pushed(TWO_RANKS).await;
        assert_eq!(
            after.rows,
            vec![(0, 10), (1, 8)],
            "rank 0 pushed since the tick began: the stream's row is the newer one"
        );
        assert_eq!(
            after.sampled_at,
            Some(after.pushed_at - PUSHED_ONE_WAY_MARGIN),
            "the earlier sample instant, the stream's"
        );
        assert!(
            after.overloaded,
            "the verdict reads the stream's row for rank 0"
        );
        assert_eq!(
            after.cached,
            Some((Some(900), Some(500))),
            "the rank cache reads the stream's row for rank 0"
        );
    }

    /// The same push where the poll's answer does not name the rank at all.
    /// The stream's row is still that rank's row in everything the poll
    /// writes: it is published at its rank's place under a rank count that
    /// covers it, and the verdict and the rank cache read it. Without it the
    /// worker lost its overload verdict on an answer older than its newest
    /// record. The row stays on file too: the worker's next pushed record
    /// publishes it again.
    #[tokio::test]
    async fn a_row_pushed_during_a_fallback_poll_is_published_though_the_answer_omits_its_rank() {
        const RANK_1_ALONE: &str = r#"{"loads":[
            {"dp_rank":1,"num_running_reqs":8,"num_waiting_reqs":0,
             "num_used_tokens":500,"max_total_num_tokens":1000}]}"#;
        let after = fallback_poll_after_rank_0_pushed(RANK_1_ALONE).await;
        assert_eq!(
            after.rows,
            vec![(0, 10), (1, 8)],
            "the stream's row, at its rank's place"
        );
        assert_eq!(after.rank_count, 2, "the count covers the rows carried");
        assert_eq!(
            after.sampled_at,
            Some(after.pushed_at - PUSHED_ONE_WAY_MARGIN),
            "the earlier sample instant, the stream's"
        );
        assert!(
            after.overloaded,
            "the verdict reads the stream's row for the omitted rank"
        );
        assert_eq!(
            after.cached,
            Some((Some(900), Some(500))),
            "the rank cache holds the omitted rank's row"
        );
        assert_eq!(
            after.next_rows,
            vec![(0, 10), (1, 8)],
            "the omitted rank pushed since the tick: its row stays on file"
        );
    }

    /// A poll's answer may count more ranks than it lists: a partial report.
    /// Carrying the stream's row of a rank the answer omits does not lower
    /// that count to the rows the report now holds, so the report stays
    /// partial for the readers that ask.
    #[tokio::test]
    async fn a_fallback_polls_larger_rank_count_stands_over_the_rows_it_carries() {
        const RANK_1_OF_FOUR: &str = r#"{"dp_rank_count":4,"loads":[
            {"dp_rank":1,"num_running_reqs":8,"num_waiting_reqs":0,
             "num_used_tokens":500,"max_total_num_tokens":1000}]}"#;
        let after = fallback_poll_after_rank_0_pushed(RANK_1_OF_FOUR).await;
        assert_eq!(
            after.rows,
            vec![(0, 10), (1, 8)],
            "the stream's row, at its rank's place"
        );
        assert_eq!(after.rank_count, 4, "the poll's own larger count stands");
    }

    /// The rank cache's test is asked of the poll's answer, not of the
    /// report the stream's rows are laid over. An answer with no token
    /// capacity leaves the cache nothing of the worker, as it did before,
    /// though the row it carries holds capacity; the poll still publishes
    /// the carried row.
    #[tokio::test]
    async fn a_fallback_poll_without_token_capacity_feeds_no_rank_cache_over_a_carried_row() {
        const RANK_1_NO_CAPACITY: &str = r#"{"loads":[
            {"dp_rank":1,"num_running_reqs":8,"num_waiting_reqs":0,
             "num_used_tokens":0,"max_total_num_tokens":0}]}"#;
        let after = fallback_poll_after_rank_0_pushed(RANK_1_NO_CAPACITY).await;
        assert_eq!(
            after.rows,
            vec![(0, 10), (1, 8)],
            "the stream's row, at its rank's place"
        );
        assert_eq!(
            after.cached, None,
            "the answer carries no capacity: the cache holds nothing of the worker"
        );
    }

    /// What the worker's next pushed record publishes after a fallback poll
    /// whose answer does not name a rank that pushed nothing since the tick
    /// began: the published rows (rank, running requests), the published
    /// rank count, the overload verdict, and the poll decision after that
    /// record. Before them, the sample instant of the poll's own report and
    /// an instant taken just before the poll. Last, the worker's entries in
    /// the rank cache (rank, token count; `None` when the cache holds
    /// nothing of the worker) right after the poll and after that record.
    struct AfterNextRecord {
        poll_sampled_at: Option<Instant>,
        before_poll: Instant,
        rows: Vec<(i32, i32)>,
        rank_count: i32,
        overloaded: bool,
        mode: PollMode,
        cached_after_poll: Option<Vec<(isize, isize)>>,
        cached: Option<Vec<(isize, isize)>>,
    }

    /// Ranks 0 and 1 with a pushed record on file, both aged by the tick:
    /// rank 0 with ten running and eight waiting, over the worker's
    /// threshold. Each rank of `also_quiet` is on file the same way, three
    /// running. The tick's fallback poll reads `answer`, which has no entry
    /// for this worker's rank 0; then rank 1 alone pushes its next record
    /// and the window publishes.
    async fn next_record_after_a_poll_omits_quiet_rank_0(
        answer: &'static str,
        also_quiet: &[i32],
    ) -> AfterNextRecord {
        let stub = spawn_engine(StatusCode::OK, answer).await;
        let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
        let worker = gauge_worker_with_overload(
            &stub.url,
            OverloadUpdate {
                waiting_requests: Some(4),
                token_usage: None,
            },
        );
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let key = WorkerGroupKey {
            model_id: "a".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
        };
        let interval = Duration::from_secs(10);
        let tick = Instant::now()
            .checked_sub(Duration::from_secs(3))
            .expect("the clock is older than three seconds");
        let t0 = tick
            .checked_sub(interval)
            .expect("the clock is older than the interval");
        let record = |running: i32, waiting: i32, used: i32| {
            PushedRecord::requests(running, waiting)
                .used_tokens(used)
                .total_tokens(1000)
        };
        // What an earlier poll left in the rank cache: a count of its own
        // for every rank on file, none of them a record's.
        let earlier_poll: Vec<(isize, isize)> = [0, 1]
            .into_iter()
            .chain(also_quiet.iter().copied())
            .map(|rank| (rank as isize, 150))
            .collect();
        monitor
            .worker_load_manager
            .update_dp_loads(&HashMap::from([(
                stub.url.clone(),
                earlier_poll.iter().copied().collect(),
            )]));
        monitor.apply_pushed_load(&worker, 0, &record(10, 8, 900), t0);
        monitor.apply_pushed_load(&worker, 1, &record(8, 0, 500), t0);
        for &rank in also_quiet {
            monitor.apply_pushed_load(&worker, rank, &record(3, 0, 300), t0);
        }
        monitor.flush_pushed();
        assert!(
            worker.is_overloaded(),
            "rank 0's record is over the threshold"
        );
        let cached = |monitor: &WorkerMonitor| {
            let loads = monitor.worker_load_manager.dp_cached_loads.read();
            loads.get(stub.url.as_str()).map(|ranks| {
                let mut ranks: Vec<(isize, isize)> = ranks
                    .iter()
                    .map(|(&rank, &tokens)| (rank, tokens))
                    .collect();
                ranks.sort_unstable();
                ranks
            })
        };
        assert_eq!(
            cached(&monitor),
            Some(earlier_poll),
            "the records feed the rank cache nothing: it holds the earlier poll's counts"
        );

        let before_poll = Instant::now();
        poll_group_once(&monitor, &key, interval, tick).await;
        assert_eq!(stub.probes.load(Ordering::SeqCst), 1, "one fallback poll");
        let cached_after_poll = cached(&monitor);
        let poll_sampled_at = monitor
            .load_state
            .snapshot()
            .get(stub.url.as_str())
            .and_then(|report| report.sampled_at);

        let next = tick + Duration::from_secs(2);
        monitor.apply_pushed_load(&worker, 1, &record(8, 0, 500), next);
        monitor.flush_pushed();
        let snapshot = monitor.load_state.snapshot();
        let report = snapshot.get(stub.url.as_str()).expect("published");
        AfterNextRecord {
            poll_sampled_at,
            before_poll,
            rows: report
                .loads
                .iter()
                .map(|rank| (rank.dp_rank, rank.num_running_reqs))
                .collect(),
            rank_count: report.dp_rank_count,
            overloaded: worker.is_overloaded(),
            mode: monitor.poll_mode(&stub.url, next, interval),
            cached_after_poll,
            cached: cached(&monitor),
        }
    }

    /// A rank on file that pushed nothing since the tick began and that the
    /// fallback poll's answer omits leaves the file with its receipt, as
    /// after an empty answer. The worker's next pushed record does not
    /// publish the old row again: the report, its count and the overload
    /// verdict read the ranks the engine still names, and the stream serves
    /// the worker again. No rank pushed since the tick began, so the poll's
    /// own report goes out under the poll's instant, not the file's.
    #[tokio::test]
    async fn a_quiet_rank_a_fallback_poll_omits_is_not_published_again_by_the_next_record() {
        const RANK_1_ALONE: &str = r#"{"loads":[
            {"dp_rank":1,"num_running_reqs":8,"num_waiting_reqs":0,
             "num_used_tokens":500,"max_total_num_tokens":1000}]}"#;
        let after = next_record_after_a_poll_omits_quiet_rank_0(RANK_1_ALONE, &[]).await;
        assert!(
            after.poll_sampled_at >= Some(after.before_poll),
            "no rank pushed since the tick began: the poll's report carries the poll's own instant"
        );
        assert_eq!(
            after.rows,
            vec![(1, 8)],
            "the omitted rank's old row is off the file"
        );
        assert_eq!(after.rank_count, 1, "the file counts the rows it holds");
        assert!(
            !after.overloaded,
            "no verdict from a row nobody stands behind"
        );
        assert_eq!(
            after.mode,
            PollMode::Suppressed,
            "the receipt went with the row: the stream serves the worker again"
        );
    }

    /// A row with no token capacity is the one synthesized from backend-wide
    /// gauges: it names one rank and lists none. A rank it does not name
    /// keeps its row and its receipt on file.
    #[tokio::test]
    async fn a_fallback_polls_row_without_token_capacity_retracts_no_rank_it_does_not_name() {
        const RANK_1_NO_CAPACITY: &str = r#"{"loads":[
            {"dp_rank":1,"num_running_reqs":8,"num_waiting_reqs":0,
             "num_used_tokens":0,"max_total_num_tokens":0}]}"#;
        let after = next_record_after_a_poll_omits_quiet_rank_0(RANK_1_NO_CAPACITY, &[]).await;
        assert_eq!(
            after.rows,
            vec![(0, 10), (1, 8)],
            "the rank the row does not name stays on file"
        );
        assert_eq!(after.rank_count, 2, "both rows are counted");
        assert_eq!(
            after.mode,
            PollMode::Fallback,
            "rank 0's receipt stays, so the worker stays polled"
        );
    }

    /// An answer that counts more ranks than it lists is short by its own
    /// word: it says nothing of the ranks it leaves out, though its row
    /// carries token capacity. A quiet rank it does not name keeps its row
    /// and its receipt on file, and the worker's next record publishes both
    /// rows. The rank cache takes the short answer's row all the same: here
    /// the merge's test and the cache's part.
    #[tokio::test]
    async fn a_fallback_polls_answer_short_of_its_rank_count_retracts_no_rank_it_does_not_name() {
        const RANK_1_OF_FOUR: &str = r#"{"dp_rank_count":4,"loads":[
            {"dp_rank":1,"num_running_reqs":8,"num_waiting_reqs":0,
             "num_used_tokens":500,"max_total_num_tokens":1000}]}"#;
        const RANK_64_OF_65: &str = r#"{"dp_rank_count":65,"loads":[
            {"dp_rank":64,"num_running_reqs":8,"num_waiting_reqs":0,
             "num_used_tokens":500,"max_total_num_tokens":1000}]}"#;
        let after = next_record_after_a_poll_omits_quiet_rank_0(RANK_1_OF_FOUR, &[]).await;
        assert_eq!(
            after.rows,
            vec![(0, 10), (1, 8)],
            "the rank a short answer does not name stays on file"
        );
        assert_eq!(after.rank_count, 2, "both rows are counted");
        assert_eq!(
            after.mode,
            PollMode::Fallback,
            "rank 0's receipt stays, so the worker stays polled"
        );
        assert_eq!(
            after.cached_after_poll,
            Some(vec![(1, 500)]),
            "the rank cache takes the short answer's row"
        );

        // The same of a short answer whose one row is of a rank the report
        // does not consider: it lists no row and still counts ranks.
        let after = next_record_after_a_poll_omits_quiet_rank_0(RANK_64_OF_65, &[]).await;
        assert_eq!(
            (after.rows, after.mode),
            (vec![(0, 10), (1, 8)], PollMode::Fallback),
            "the rows and rank 0's receipt stay on file"
        );
    }

    /// A retraction takes the retracted rank's receipt alone. Three ranks
    /// on file, each aged by the tick: the answer omits rank 0 and names
    /// ranks 1 and 2, counting the two it lists as an engine's answer does,
    /// then rank 1 alone pushes its next record. Rank 2 pushed nothing, and
    /// its receipt, kept, holds the worker in the fallback poll. Had the
    /// retraction cleared the worker's receipts, the stream would seem to
    /// serve every rank left, and rank 2's row would stay as the poll left
    /// it.
    #[tokio::test]
    async fn a_fallback_poll_that_retracts_a_rank_keeps_the_receipt_of_a_rank_that_stays() {
        const RANKS_1_AND_2: &str = r#"{"dp_rank_count":2,"loads":[
            {"dp_rank":1,"num_running_reqs":8,"num_waiting_reqs":0,
             "num_used_tokens":500,"max_total_num_tokens":1000},
            {"dp_rank":2,"num_running_reqs":3,"num_waiting_reqs":0,
             "num_used_tokens":300,"max_total_num_tokens":1000}]}"#;
        let after = next_record_after_a_poll_omits_quiet_rank_0(RANKS_1_AND_2, &[2]).await;
        assert_eq!(
            after.rows,
            vec![(1, 8), (2, 3)],
            "rank 0 alone is off the file"
        );
        assert_eq!(after.rank_count, 2, "the file counts the rows it holds");
        assert_eq!(
            after.mode,
            PollMode::Fallback,
            "rank 2 is still quiet: its receipt stays and the worker stays polled"
        );
    }

    /// An answer whose entries are annotated with their workers is a
    /// gateway's rollup: it carries token capacity, and its entries are not
    /// this worker's ranks. The rank cache takes none of them: the worker's
    /// entry, an earlier poll's, leaves the cache with the poll. The answer
    /// changes and retracts no rank on file.
    #[tokio::test]
    async fn a_fallback_polls_rollup_by_worker_changes_no_rank_on_file() {
        const ROLLUP: &str = r#"{"loads":[
            {"worker":"http://w1:8000","dp_rank":0,"num_running_reqs":1,"num_waiting_reqs":0,
             "num_used_tokens":100,"max_total_num_tokens":1000}]}"#;
        let after = next_record_after_a_poll_omits_quiet_rank_0(ROLLUP, &[]).await;
        assert_eq!(
            after.cached_after_poll, None,
            "a rollup's entries are no ranks of the worker: its entry leaves the rank cache"
        );
        assert_eq!(
            after.rows,
            vec![(0, 10), (1, 8)],
            "both ranks keep their rows"
        );
        assert_eq!(after.rank_count, 2, "both rows are counted");
        assert_eq!(
            after.mode,
            PollMode::Fallback,
            "rank 0's receipt stays, so the worker stays polled"
        );
    }

    /// A poll's row without token capacity is no token count of the rank it
    /// names. The answer feeds the rank cache nothing and takes the earlier
    /// poll's entry away. On file the row is its rank's, beside rows that
    /// carry capacity, and the worker's next record feeds the cache nothing
    /// either.
    #[tokio::test]
    async fn a_fallback_polls_row_without_token_capacity_is_no_token_count_in_the_rank_cache() {
        const RANK_2_NO_CAPACITY: &str = r#"{"loads":[
            {"dp_rank":2,"num_running_reqs":4,"num_waiting_reqs":0,
             "num_used_tokens":0,"max_total_num_tokens":0}]}"#;
        let after = next_record_after_a_poll_omits_quiet_rank_0(RANK_2_NO_CAPACITY, &[2]).await;
        assert_eq!(
            after.cached_after_poll, None,
            "an answer without token capacity feeds the rank cache nothing"
        );
        assert_eq!(
            after.rows,
            vec![(0, 10), (1, 8), (2, 4)],
            "the answer retracts no rank, and its row is rank 2's on file"
        );
        assert_eq!(
            after.cached, None,
            "the worker's next record feeds the rank cache nothing"
        );
    }

    /// The poll's feed reads its report row by row: beside a row that
    /// carries token capacity, an answer's row without it is no token
    /// count. The row with capacity is an idle rank's: its zero is a count,
    /// and it is fed. The worker's next record writes nothing over the
    /// poll's entry.
    #[tokio::test]
    async fn a_fallback_polls_answer_feeds_the_rank_cache_its_rows_with_token_capacity_alone() {
        const CAPACITY_ON_RANK_1: &str = r#"{"loads":[
            {"dp_rank":1,"num_running_reqs":0,"num_waiting_reqs":0,
             "num_used_tokens":0,"max_total_num_tokens":1000},
            {"dp_rank":2,"num_running_reqs":4,"num_waiting_reqs":0,
             "num_used_tokens":0,"max_total_num_tokens":0}]}"#;
        let after = next_record_after_a_poll_omits_quiet_rank_0(CAPACITY_ON_RANK_1, &[2]).await;
        assert_eq!(
            after.cached_after_poll,
            Some(vec![(1, 0)]),
            "the poll feeds the idle rank that carries capacity, and it alone"
        );
        assert_eq!(
            after.cached,
            Some(vec![(1, 0)]),
            "rank 1's next record writes nothing over the poll's entry"
        );
    }

    /// A pushed record feeds the rank cache nothing, whatever it carries:
    /// the worker's last entry, a poll's, stays. A record's token count
    /// reaches the worker's report, which routing reads, and no cache
    /// entry: the cache's one reader picks a rank of a worker that has no
    /// stream.
    #[tokio::test]
    async fn a_pushed_record_leaves_the_rank_caches_last_entry() {
        let (registry, monitor) = monitor_with_readers(LoadReaders::Blind);
        let worker = gauge_worker("http://engine:8000");
        worker.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&worker)).unwrap();
        let last_entry = HashMap::from([(0, 900), (1, 500)]);
        monitor
            .worker_load_manager
            .update_dp_loads(&HashMap::from([(
                worker.url().to_string(),
                last_entry.clone(),
            )]));

        // A core-only record, then one that carries a token count and its
        // capacity for a rank the entry names.
        let core_only = PushedRecord::requests(3, 0);
        let with_tokens = PushedRecord::requests(4, 0)
            .used_tokens(2048)
            .total_tokens(4096);
        for record in [&core_only, &with_tokens] {
            monitor.apply_pushed_load(&worker, 0, record, Instant::now());
            assert_eq!(
                monitor
                    .worker_load_manager
                    .dp_cached_loads
                    .read()
                    .get(worker.url())
                    .cloned(),
                Some(last_entry.clone()),
                "no record writes over the last entry"
            );
        }
        monitor.flush_pushed();
        let snapshot = monitor.load_state.snapshot();
        let report = snapshot.get(worker.url()).expect("published");
        assert_eq!(
            (
                report.loads[0].num_used_tokens,
                report.loads[0].max_total_num_tokens
            ),
            (2048, 4096),
            "the report takes the record's token count"
        );
    }
}

#[cfg(test)]
#[expect(clippy::print_stdout, reason = "a recorder says where its sheet went")]
mod classic_overload_rows;

#[cfg(test)]
#[expect(clippy::print_stdout, reason = "a recorder says where its sheet went")]
mod classic_minimum_tokens_rows;
