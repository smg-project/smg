//! The gateway as the store's host (design 3, the embedded mode): the store
//! ingress traits over the structures the gateway already holds, so the
//! controller's drivers speak one contract whether the store is the
//! controller's own or this process's registry, monitors and admission
//! counters. Built beside the engine under `--policy-engine engine`
//! ([`Decider::set_stores`](super::engine_host::Decider::set_stores)) and not
//! at all under `legacy`; nothing reads or writes through it until the step
//! that moves the ingestion drivers, when the subscription task, the prober
//! and the poll feed it instead of the structures directly.
//!
//! What stands behind each trait here:
//!
//! - [`WorkersStore`]: the worker registry. A row's id is the view's
//!   ([`worker_id`]), its incarnation the handle's number, its revision the
//!   registry's. The watch is a snapshot only: the registry's broadcast is
//!   the in-process tail, and a watcher is told to snapshot again.
//! - [`HealthSink`]: the registry's status machine under its revision guard
//!   (the prober's rule), and a table of the vetoes and completed reports
//!   other gateways would send a controller, empty in this process (this
//!   gateway's own vetoes are the handle's atomics).
//! - [`LoadSink`]: the worker monitor's pushed-record and polled-report
//!   paths; a row enters as a core-only record, which is what an event
//!   batch's record is.
//! - [`KvEventSink`]: the store's cursor in front of the model's index,
//!   applied through the monitor's own apply path.
//! - [`ReservationStore`]: a slot is the worker's load count under its
//!   window, a room the decode's admission claim.
//!
//! Sessions and leases have no host form: today's pins live inside the keyed
//! policies and the request lease is the load guard; both move in later
//! steps.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use openai_protocol::{
    model_card::ModelCard,
    worker::{
        ConnectionMode, HealthCheckConfig, OverloadUpdate, RuntimeType, SchedulerLoadSnapshot,
        WorkerLoadResponse, WorkerStatus, WorkerType,
    },
};
use parking_lot::Mutex;
use smg_grpc_client::common_proto::{EngineLoad, KvEventBatch};
use smg_policy_controller::{
    store::{
        Admission, BatchFacts, BatchOutcome, GatewayId, HealthPolicy, HealthSink, KvCursor,
        KvEventSink, LoadSample, LoadSink, LoadSource, MembershipChange, MembershipTail, Overload,
        PollMode, RankCursor, Reachability, ReservationKind, ReservationStore, ResyncReason,
        Signals, WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus as StoreStatus, WorkersStore,
    },
    LeaseId, LoadRow, Role, TransportClass, WireModes, WorkerId,
};

use super::engine_host::{incarnation, load_row, worker_id};
use crate::worker::{
    basic::WINDOW_LABELS, kv_event_monitor::WorkerIndexState, manager::compute_next_status,
    monitor::PollMode as MonitorPollMode, overload::OverloadThresholds,
    registry::WorkerId as RegistryId, BasicWorker, BasicWorkerBuilder, KvEventMonitor, KvIndex,
    Worker, WorkerMonitor, WorkerRegistry, UNKNOWN_MODEL_ID,
};

/// The store ingress over this process's structures.
pub struct HostStores {
    registry: Arc<WorkerRegistry>,
    /// The load monitor, held weakly: the monitor holds the policy registry,
    /// the registry its decider and the decider these stores, so a strong
    /// edge here would close the cycle and keep the monitor and its publish
    /// task alive after their owner let go.
    monitor: Option<Weak<WorkerMonitor>>,
    /// The event monitor, held weakly for the same reason.
    kv_monitor: Option<Weak<KvEventMonitor>>,
    /// The view's id of a worker to its registration key, filled as rows are
    /// seen; a miss scans the registry once.
    keys: DashMap<WorkerId, Arc<str>>,
    /// The writes made through this host, for the snapshot's epoch (the
    /// registry's own counter is its pools').
    epoch: AtomicU64,
    vetoes: Mutex<HashMap<(GatewayId, WorkerId), Instant>>,
    /// The latest completed dispatch of class `Ok` on a worker by each
    /// gateway: the precedence rule is per gateway.
    ok_at: Mutex<HashMap<(GatewayId, WorkerId), Instant>>,
    /// One lock per worker's feed, so the subscriptions' batches apply in
    /// parallel as they do on their own tasks today.
    feeds: DashMap<WorkerId, FeedCell>,
    reservations: Mutex<HashMap<(GatewayId, LeaseId), Vec<Claim>>>,
}

/// What one lease holds on one worker: the handle it claimed on (a same-URL
/// replacement carries the counters to its successor on the shared runtime,
/// and the view's id with it, so the release goes through the handle, never
/// through the id), the kind and the count claimed.
struct Claim {
    worker: Arc<dyn Worker>,
    kind: ReservationKind,
    count: u32,
}

/// One worker's feed into its model's index: the store's cursors in front of
/// the monitor's apply path.
struct KvFeed {
    index: Arc<KvIndex>,
    /// The registration key the slot was interned under: the index keeps
    /// one slot per name, shared with a same-URL successor.
    url: Arc<str>,
    interned: u32,
    state: WorkerIndexState,
    cursors: HashMap<u16, RankCursor>,
}

impl KvFeed {
    /// The worker's blocks leave the index with its slot's state.
    fn retire(self) {
        let KvFeed {
            index,
            interned,
            state,
            ..
        } = self;
        index.remove_worker(interned, state.blocks);
    }
}

/// A worker's feed, or `None` once removed, the tombstone: a batch that took
/// the cell before the removal, or one whose removal ran between the
/// registry and the map, finds the worker gone and lands nothing. A
/// tombstone whose worker left the registry is pruned by the next batch
/// for it and by the next removal.
type FeedCell = Arc<Mutex<Option<KvFeed>>>;

impl std::fmt::Debug for HostStores {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostStores")
            .field("workers", &self.registry.len())
            .field("epoch", &self.epoch.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl HostStores {
    /// The host over the registry and, when the context runs them, the load
    /// and KV event monitors.
    pub fn new(
        registry: Arc<WorkerRegistry>,
        monitor: Option<&Arc<WorkerMonitor>>,
        kv_monitor: Option<&Arc<KvEventMonitor>>,
    ) -> Self {
        Self {
            registry,
            monitor: monitor.map(Arc::downgrade),
            kv_monitor: kv_monitor.map(Arc::downgrade),
            keys: DashMap::new(),
            epoch: AtomicU64::new(0),
            vetoes: Mutex::new(HashMap::new()),
            ok_at: Mutex::new(HashMap::new()),
            feeds: DashMap::new(),
            reservations: Mutex::new(HashMap::new()),
        }
    }

    /// The worker behind a view id: the key map first, the registry's whole
    /// list on a miss (a worker the registration workflow added enters the
    /// map that way).
    fn resolve(&self, id: WorkerId) -> Option<Arc<dyn Worker>> {
        if let Some(key) = self.keys.get(&id) {
            if let Some(worker) = self.registry.get_by_url(&key) {
                if worker_id(worker.as_ref()) == id {
                    return Some(worker);
                }
            }
            drop(key);
            self.keys.remove(&id);
        }
        let worker = self
            .registry
            .get_all()
            .into_iter()
            .find(|worker| worker_id(worker.as_ref()) == id)?;
        self.keys.insert(id, Arc::from(worker.url()));
        Some(worker)
    }

    fn note(&self, worker: &dyn Worker) -> WorkerId {
        let id = worker_id(worker);
        self.keys.insert(id, Arc::from(worker.url()));
        id
    }

    fn registry_id(&self, worker: &dyn Worker) -> Option<RegistryId> {
        self.registry.get_id_by_url(worker.url())
    }

    fn bump(&self) {
        self.epoch.fetch_add(1, Ordering::Relaxed);
    }

    /// The load monitor while it lives; a host outliving it feeds nothing.
    fn monitor(&self) -> Option<Arc<WorkerMonitor>> {
        self.monitor.as_ref()?.upgrade()
    }

    /// The event monitor while it lives.
    fn kv_monitor(&self) -> Option<Arc<KvEventMonitor>> {
        self.kv_monitor.as_ref()?.upgrade()
    }

    /// A tombstone whose worker left the registry has nothing left to guard.
    fn prune_tombstone(&self, id: WorkerId) {
        if self.resolve(id).is_none() {
            self.feeds.remove_if(&id, |_, cell| cell.lock().is_none());
        }
    }

    /// The feed of a worker into its model's index, when the monitor holds
    /// an index for the model (an events subscription created it).
    fn feed_of(&self, id: WorkerId) -> Option<KvFeed> {
        let worker = self.resolve(id)?;
        let kv_monitor = self.kv_monitor()?;
        let model = match worker.model_id() {
            "" => UNKNOWN_MODEL_ID,
            model => model,
        };
        let index = kv_monitor.get_indexer(model)?;
        let interned = index.intern_worker(worker.url()).ok()?;
        Some(KvFeed {
            index,
            url: Arc::from(worker.url()),
            interned,
            state: WorkerIndexState::default(),
            cursors: HashMap::new(),
        })
    }

    /// A batch's feed enters the map under its id. The registry was read
    /// before the entry was taken, so a removal that ran in between swept
    /// a tombstone this entry never saw: the worker is read once more
    /// under the fresh cell, and one gone by now takes its feed out again
    /// and leaves no entry; the URL's slot goes back to the index only
    /// when the URL itself has left, since a same-URL successor and the
    /// monitor's own subscription share it. An entry
    /// already there is the one the batch takes; the feed built for it
    /// is dropped.
    fn install_feed(&self, id: WorkerId, feed: KvFeed) -> Option<FeedCell> {
        let mut fresh = false;
        let cell = Arc::clone(
            self.feeds
                .entry(id)
                .or_insert_with(|| {
                    fresh = true;
                    Arc::new(Mutex::new(Some(feed)))
                })
                .value(),
        );
        if fresh {
            let mut guard = cell.lock();
            if self.resolve(id).is_none() {
                if let Some(feed) = guard.take() {
                    if self.registry.get_by_url(&feed.url).is_none() {
                        feed.retire();
                    }
                }
                drop(guard);
                self.prune_tombstone(id);
                return None;
            }
        }
        Some(cell)
    }
}

// ------------------------------------------------------------ conversions

fn role_of(worker_type: WorkerType) -> Role {
    match worker_type {
        WorkerType::Regular => Role::Single,
        WorkerType::Prefill => Role::Prefill,
        WorkerType::Decode => Role::Decode,
        WorkerType::Encode => Role::Encode,
    }
}

fn type_of(role: Role) -> WorkerType {
    match role {
        Role::Single => WorkerType::Regular,
        Role::Prefill => WorkerType::Prefill,
        Role::Decode => WorkerType::Decode,
        Role::Encode => WorkerType::Encode,
    }
}

fn wires_of(mode: ConnectionMode) -> WireModes {
    match mode {
        ConnectionMode::Http => WireModes::HTTP,
        ConnectionMode::Grpc => WireModes::GRPC,
        ConnectionMode::Zmq => WireModes::ZMQ,
    }
}

fn mode_of(wires: WireModes) -> ConnectionMode {
    if wires.contains(WireModes::GRPC) {
        ConnectionMode::Grpc
    } else if wires.contains(WireModes::ZMQ) {
        ConnectionMode::Zmq
    } else {
        ConnectionMode::Http
    }
}

fn status_of(status: WorkerStatus) -> StoreStatus {
    match status {
        WorkerStatus::Pending => StoreStatus::Pending,
        WorkerStatus::Ready => StoreStatus::Ready,
        WorkerStatus::NotReady => StoreStatus::NotReady,
        WorkerStatus::Failed => StoreStatus::Failed,
        WorkerStatus::Draining => StoreStatus::Draining,
    }
}

fn registry_status(status: StoreStatus) -> WorkerStatus {
    match status {
        StoreStatus::Pending => WorkerStatus::Pending,
        StoreStatus::Ready => WorkerStatus::Ready,
        StoreStatus::NotReady => WorkerStatus::NotReady,
        StoreStatus::Failed => WorkerStatus::Failed,
        StoreStatus::Draining => WorkerStatus::Draining,
    }
}

fn millis(secs: u64) -> u32 {
    u32::try_from(secs.saturating_mul(1_000)).unwrap_or(u32::MAX)
}

fn health_of(config: &HealthCheckConfig) -> HealthPolicy {
    HealthPolicy {
        failure_threshold: config.failure_threshold,
        success_threshold: config.success_threshold,
        check_interval_ms: millis(config.check_interval_secs),
        timeout_ms: millis(config.timeout_secs),
        probes: !config.disable_health_check,
    }
}

fn health_config_of(policy: HealthPolicy) -> HealthCheckConfig {
    health_config_over(&HealthCheckConfig::default(), policy)
}

/// The policy's five fields over a config; the rest (the drain settle) stays.
fn health_config_over(base: &HealthCheckConfig, policy: HealthPolicy) -> HealthCheckConfig {
    HealthCheckConfig {
        failure_threshold: policy.failure_threshold.max(1),
        success_threshold: policy.success_threshold.max(1),
        timeout_secs: u64::from(policy.timeout_ms / 1_000).max(1),
        check_interval_secs: u64::from(policy.check_interval_ms / 1_000).max(1),
        disable_health_check: !policy.probes,
        ..base.clone()
    }
}

fn overload_of(thresholds: &OverloadThresholds) -> Overload {
    Overload {
        kv_usage_permille: thresholds
            .token_usage
            .map(|ratio| (ratio * 1_000.0).round().clamp(0.0, 1_000.0) as u16),
        waiting_requests: thresholds
            .waiting_requests
            .map(|n| u32::try_from(n).unwrap_or(u32::MAX)),
    }
}

fn overload_update_of(overload: Overload) -> OverloadUpdate {
    OverloadUpdate {
        waiting_requests: overload
            .waiting_requests
            .map(|n| usize::try_from(n).unwrap_or(usize::MAX)),
        token_usage: overload
            .kv_usage_permille
            .map(|permille| f64::from(permille) / 1_000.0),
    }
}

/// The worker row the handle stands for.
fn facts_of(worker: &dyn Worker) -> WorkerFacts {
    let metadata = worker.metadata();
    let spec = &metadata.spec;
    let mode = *worker.connection_mode();
    WorkerFacts {
        key: worker.url().to_string(),
        engine: spec
            .dp_base_url
            .clone()
            .unwrap_or_else(|| worker.url().to_string()),
        incarnation: incarnation(worker),
        role: role_of(*worker.worker_type()),
        wire_modes: wires_of(mode),
        engine_kind: spec.runtime_type.to_string(),
        models: worker.models().into_iter().map(|card| card.id).collect(),
        dp_rank: worker
            .dp_rank()
            .and_then(|rank| u16::try_from(rank).ok())
            .unwrap_or(0),
        dp_size: worker
            .dp_size()
            .and_then(|size| u16::try_from(size).ok())
            .unwrap_or(1),
        labels: spec
            .labels
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        zone: None,
        pairing: None,
        pair_class: 0,
        max_running: worker.max_running_requests().map(u32::from),
        kv_block_size: spec.kv_block_size.and_then(|size| u32::try_from(size).ok()),
        bootstrap_port: worker.bootstrap_port(),
        priority: i32::try_from(worker.priority()).unwrap_or(i32::MAX),
        reachable_from: if mode == ConnectionMode::Zmq {
            Reachability::Gateway(GatewayId::LOCAL)
        } else {
            Reachability::Any
        },
        signals: Signals {
            events: mode == ConnectionMode::Grpc,
            loads: true,
            scrape_interval_ms: 0,
        },
        health: health_of(&metadata.health_config),
        overload: overload_of(&metadata.overload),
    }
}

fn row_of(worker: &dyn Worker) -> WorkerRow {
    WorkerRow {
        id: worker_id(worker),
        facts: facts_of(worker),
        status: status_of(worker.status()),
        revision: worker.revision(),
        admitted_at: Instant::now()
            .checked_sub(worker.admitted_age())
            .unwrap_or_else(Instant::now),
    }
}

impl HostStores {
    /// A handle built from the facts a handshake produced, carrying the
    /// registry's connect signal. Over a live handle under the same key it
    /// starts from what registration resolved (the breaker and its state,
    /// the health endpoint, the resilience, HTTP/2, the ZMQ bounds, the
    /// discovered labels) and lays the facts over, as a patch does; the
    /// registry's replacement then hands the runtime over.
    fn handle_of(&self, facts: &WorkerFacts, live: Option<&Arc<dyn Worker>>) -> Arc<dyn Worker> {
        // A rank's key is `{engine}@{rank}` and the builder appends the rank
        // itself, so a fresh rank starts from the engine's address; a live
        // handle keeps its key, so one whose engine changed shape starts
        // over.
        let carried = live.filter(|live| {
            let was = facts_of(live.as_ref());
            (was.dp_rank, was.dp_size) == (facts.dp_rank, facts.dp_size)
        });
        // The facts' policy over the live health config: what lies outside
        // the policy (the drain settle) is what registration resolved.
        let health_config = match carried {
            Some(live) => health_config_over(&live.metadata().health_config, facts.health),
            None => health_config_of(facts.health),
        };
        let mut builder = match carried {
            Some(live) => BasicWorkerBuilder::carrying(live),
            None if facts.dp_size > 1 => BasicWorkerBuilder::new(facts.engine.clone())
                .dp_config(usize::from(facts.dp_rank), usize::from(facts.dp_size)),
            None => BasicWorkerBuilder::new(facts.key.clone()),
        };
        let mut labels: HashMap<String, String> = facts
            .labels
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        // The admission window is a label on the handle.
        if let Some(window) = facts.max_running {
            labels
                .entry("max_running_requests".to_string())
                .or_insert_with(|| window.to_string());
        }
        builder = builder
            .worker_type(type_of(facts.role))
            .connection_mode(mode_of(facts.wire_modes))
            .labels(labels)
            .health_config(health_config)
            .bootstrap_port(facts.bootstrap_port)
            .priority(u32::try_from(facts.priority).unwrap_or(0))
            .overload(overload_update_of(facts.overload))
            .kv_block_size(
                facts
                    .kv_block_size
                    .map(|tokens| usize::try_from(tokens).unwrap_or(usize::MAX)),
            )
            .connect_signal_tx(self.registry.connect_signal_sender());
        if !facts.models.is_empty() {
            // A card the live handle holds under the id keeps its aliases,
            // parsers and type; an id new to it gets a bare card.
            let live_cards = carried.map(|live| live.models()).unwrap_or_default();
            let cards: Vec<ModelCard> = facts
                .models
                .iter()
                .map(|id| {
                    live_cards
                        .iter()
                        .find(|card| card.id == *id)
                        .cloned()
                        .unwrap_or_else(|| ModelCard::new(id))
                })
                .collect();
            builder = builder.models(cards);
        }
        if let Ok(kind) = facts.engine_kind.parse::<RuntimeType>() {
            builder = builder.runtime_type(kind);
        }
        let mut worker = builder.build();
        if let Some(previous) = carried.and_then(|live| live.as_any().downcast_ref::<BasicWorker>())
        {
            worker.share_kv_engine_state(previous);
        }
        Arc::new(worker)
    }
}

/// A core-only record from a row: what an event batch's record carries,
/// which the monitor merges over the telemetry it already holds.
fn engine_load_of(row: LoadRow, role: Role) -> EngineLoad {
    let rate = f64::from(row.live_rate);
    EngineLoad {
        running_requests: row.running,
        waiting_requests: row.queued,
        waiting_uncached_tokens: Some(row.waiting_uncached_tokens),
        token_usage: f64::from(row.kv_usage_permille) / 1_000.0,
        active_token_usage: Some(f64::from(row.active_kv_permille) / 1_000.0),
        gen_throughput: rate,
        prefill_throughput: (role == Role::Prefill).then_some(rate),
        max_running_requests: row.max_running,
        age_ms: row.report_age_ms,
        num_resident_tokens: Some(u64::from(row.resident_tokens)),
        ..EngineLoad::default()
    }
}

/// One rank's entry of a polled report, from the store's row.
fn scheduler_entry_of(rank: u16, row: LoadRow, role: Role) -> SchedulerLoadSnapshot {
    let mut entry = SchedulerLoadSnapshot::from(&engine_load_of(row, role));
    entry.dp_rank = i32::from(rank);
    entry
}

/// A polled report over `loads`, sampled when the poll began.
fn polled_report_of(loads: Vec<SchedulerLoadSnapshot>, polled_at: Instant) -> WorkerLoadResponse {
    WorkerLoadResponse {
        dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
        loads,
        sampled_at: Some(polled_at),
        ..WorkerLoadResponse::default()
    }
}

// ---------------------------------------------------------------- workers

/// Two of the trait's rules bend to the registry's shape until it gains an
/// in-place patch: a same-key `upsert` and a `patch` both swap the handle
/// (each rebuilt over the live one), so each answers `Replaced` with a new
/// id where the contract says a patch keeps it, and a row's revision moves
/// on that swap alone, not on a status change. The contract tests pin the
/// rules on the crate's store; nothing reads through this host until the
/// registry keeps the handle.
impl WorkersStore for HostStores {
    fn upsert(&self, facts: WorkerFacts) -> MembershipChange {
        let live = self.registry.get_by_url(&facts.key);
        let old = live.as_ref().map(|worker| worker_id(worker.as_ref()));
        let worker = self.handle_of(&facts, live.as_ref());
        if let Some(old) = old {
            self.keys.remove(&old);
        }
        let new = self.note(worker.as_ref());
        self.registry.register_or_replace(worker);
        self.bump();
        match old {
            Some(old) => MembershipChange::Replaced { old, new },
            None => MembershipChange::Registered { id: new },
        }
    }

    /// A patch replaces the handle under its URL (the registry's property
    /// replace), so it reads as a replacement with a new id; the rebuild
    /// carries what registration resolved, as the property step's does.
    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange> {
        let worker = self.resolve(id)?;
        let rid = self.registry_id(worker.as_ref())?;
        let mut builder = BasicWorkerBuilder::carrying(&worker)
            .connect_signal_tx(self.registry.connect_signal_sender());
        if let Some(labels) = patch.labels {
            let mut labels: HashMap<String, String> = labels.into_iter().collect();
            // The window is a fact a patch cannot change (the memory store
            // keeps it): labels naming neither window key keep the handle's.
            let named = WINDOW_LABELS.iter().any(|key| labels.contains_key(*key));
            if let Some(window) = worker.max_running_requests().filter(|_| !named) {
                labels.insert("max_running_requests".to_string(), window.to_string());
            }
            builder = builder.labels(labels);
        }
        if let Some(priority) = patch.priority {
            builder = builder.priority(u32::try_from(priority).unwrap_or(0));
        }
        if let Some(overload) = patch.overload {
            builder = builder.overload(overload_update_of(overload));
        }
        let mut rebuilt = builder.build();
        if let Some(previous) = worker.as_any().downcast_ref::<BasicWorker>() {
            rebuilt.share_kv_engine_state(previous);
        }
        let replacement: Arc<dyn Worker> = Arc::new(rebuilt);
        if !self
            .registry
            .replace_claiming_local(&rid, Arc::clone(&replacement), revision)
        {
            return None;
        }
        self.keys.remove(&id);
        let new = self.note(replacement.as_ref());
        self.bump();
        Some(MembershipChange::Replaced { old: id, new })
    }

    /// The revision check and the removal run under the registry's per-worker
    /// lock, so a replacement landing in between keeps its successor.
    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        let worker = self.resolve(id)?;
        let rid = self.registry_id(worker.as_ref())?;
        self.registry.remove_if_revision(&rid, revision)?;
        self.keys.remove(&id);
        self.bump();
        Some(MembershipChange::Removed { id })
    }

    fn set_status(
        &self,
        id: WorkerId,
        status: StoreStatus,
        revision: u64,
    ) -> Option<MembershipChange> {
        let worker = self.resolve(id)?;
        let rid = self.registry_id(worker.as_ref())?;
        let (from, to) =
            self.registry
                .transition_status_if_revision(&rid, revision, registry_status(status))?;
        self.bump();
        Some(MembershipChange::StatusChanged {
            id,
            from: status_of(from),
            to: status_of(to),
        })
    }

    fn get(&self, id: WorkerId) -> Option<WorkerRow> {
        self.resolve(id).map(|worker| row_of(worker.as_ref()))
    }

    fn by_key(&self, key: &str) -> Option<WorkerRow> {
        let worker = self.registry.get_by_url(key)?;
        self.note(worker.as_ref());
        Some(row_of(worker.as_ref()))
    }

    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    fn snapshot(&self) -> (u64, Vec<WorkerRow>) {
        let mut rows: Vec<WorkerRow> = self
            .registry
            .get_all()
            .iter()
            .map(|worker| {
                self.note(worker.as_ref());
                row_of(worker.as_ref())
            })
            .collect();
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        (self.epoch(), rows)
    }

    /// The registry keeps no change log: its subscribers read its broadcast
    /// and reconcile from a snapshot when they lag, which is what a watcher
    /// here is told to do.
    fn changes_since(&self, _epoch: u64) -> MembershipTail {
        MembershipTail::SnapshotRequired
    }
}

// ----------------------------------------------------------------- health

impl HealthSink for HostStores {
    fn probe_outcome(
        &self,
        id: WorkerId,
        ok: bool,
        revision: u64,
    ) -> Option<(StoreStatus, StoreStatus)> {
        let worker = self.resolve(id)?;
        let rid = self.registry_id(worker.as_ref())?;
        let ((), transition) = self.registry.apply_if_revision(&rid, revision, |worker| {
            // The prober counts a pending worker's probes before it reads the
            // machine, so the never-reachable cap can fire.
            if worker.status() == WorkerStatus::Pending {
                worker.total_pending_probes_increment();
            }
            (
                (),
                compute_next_status(worker, ok, &worker.metadata().health_config),
            )
        })?;
        transition.map(|(from, to)| (status_of(from), status_of(to)))
    }

    /// A draining or ready worker takes no transition from a connect signal,
    /// read under the registry's lock.
    fn connected(&self, id: WorkerId, revision: u64) -> Option<(StoreStatus, StoreStatus)> {
        let worker = self.resolve(id)?;
        let rid = self.registry_id(worker.as_ref())?;
        let ((), transition) = self.registry.apply_if_revision(&rid, revision, |worker| {
            let status = worker.status();
            let promoted = status != WorkerStatus::Draining && status != WorkerStatus::Ready;
            ((), promoted.then_some(WorkerStatus::Ready))
        })?;
        transition.map(|(from, to)| (status_of(from), status_of(to)))
    }

    fn gateway_veto(&self, gateway: GatewayId, id: WorkerId, until: Option<Instant>) {
        let mut vetoes = self.vetoes.lock();
        match until {
            Some(until) => {
                vetoes.insert((gateway, id), until);
            }
            None => {
                vetoes.remove(&(gateway, id));
            }
        }
    }

    fn vetoed(&self, gateway: GatewayId, id: WorkerId, now: Instant) -> bool {
        self.vetoes
            .lock()
            .get(&(gateway, id))
            .is_some_and(|until| *until > now)
    }

    fn completed(&self, gateway: GatewayId, id: WorkerId, class: TransportClass, at: Instant) {
        if class != TransportClass::Ok {
            return;
        }
        let mut ok_at = self.ok_at.lock();
        let latest = ok_at.entry((gateway, id)).or_insert(at);
        if at > *latest {
            *latest = at;
        }
    }

    fn completed_ok_since(&self, gateway: GatewayId, id: WorkerId, since: Instant) -> bool {
        self.ok_at
            .lock()
            .get(&(gateway, id))
            .is_some_and(|at| *at >= since)
    }
}

// ------------------------------------------------------------------ loads

impl LoadSink for HostStores {
    fn record(
        &self,
        id: WorkerId,
        rank: u16,
        row: LoadRow,
        received_at: Instant,
        source: LoadSource,
    ) {
        let (Some(monitor), Some(worker)) = (self.monitor(), self.resolve(id)) else {
            return;
        };
        let record = engine_load_of(row, role_of(*worker.worker_type()));
        match source {
            LoadSource::Stream | LoadSource::Relay(_) => {
                monitor.apply_pushed_load(&worker, i32::from(rank), &record, received_at);
            }
            // The gateway's own read of one rank merges where the stream's
            // records land when the worker has them, and takes the poll's
            // path over its published rows when not; no push receipt either
            // way, so the poll stays.
            LoadSource::Poll | LoadSource::Scrape => {
                monitor.apply_read_load(&worker, i32::from(rank), &record, received_at);
            }
        }
    }

    /// A poll's rows take the poll loop's own path: merged over a pushed
    /// report for a worker whose stream fell back, published on their own
    /// for a worker with no stream.
    fn report(
        &self,
        id: WorkerId,
        rows: &[(u16, LoadRow)],
        polled_at: Instant,
        _source: LoadSource,
    ) {
        let (Some(monitor), Some(worker)) = (self.monitor(), self.resolve(id)) else {
            return;
        };
        let role = role_of(*worker.worker_type());
        // A member of a data-parallel group speaks for its own rank: a row
        // for another rank is dropped, as the stream path drops a record
        // for one.
        let own = worker.dp_rank();
        let loads: Vec<SchedulerLoadSnapshot> = rows
            .iter()
            .filter(|(rank, _)| own.is_none_or(|own| usize::from(*rank) == own))
            .map(|&(rank, row)| scheduler_entry_of(rank, row, role))
            .collect();
        if loads.is_empty() && !rows.is_empty() {
            return;
        }
        monitor.ingest_polled_report(&worker, polled_report_of(loads, polled_at), polled_at);
    }

    fn poll_mode(&self, id: WorkerId, now: Instant, interval: Duration) -> PollMode {
        let (Some(monitor), Some(worker)) = (self.monitor(), self.resolve(id)) else {
            return PollMode::Poll;
        };
        match monitor.poll_mode(worker.url(), now, interval) {
            MonitorPollMode::Poll => PollMode::Poll,
            MonitorPollMode::Fallback => PollMode::Fallback,
            MonitorPollMode::Suppressed => PollMode::Suppressed,
        }
    }

    fn load(&self, id: WorkerId, rank: u16) -> Option<LoadSample> {
        let monitor = self.monitor()?;
        let worker = self.resolve(id)?;
        let snapshot = monitor.load_snapshot();
        let (report, sampled_at) = snapshot.report(worker.url())?;
        let entry = report
            .loads
            .iter()
            .find(|entry| entry.dp_rank == i32::from(rank))?;
        let one = WorkerLoadResponse {
            dp_rank_count: 1,
            loads: vec![entry.clone()],
            sampled_at: Some(sampled_at),
            ..WorkerLoadResponse::default()
        };
        Some(LoadSample {
            row: load_row(&one, *worker.worker_type(), Instant::now(), sampled_at),
            // The published entry keeps its sample instant alone.
            received_at: sampled_at,
            sampled_at,
            source: if worker.pushes_load_records() {
                LoadSource::Stream
            } else {
                LoadSource::Poll
            },
        })
    }

    fn ranks(&self, id: WorkerId) -> usize {
        let (Some(monitor), Some(worker)) = (self.monitor(), self.resolve(id)) else {
            return 0;
        };
        monitor
            .load_snapshot()
            .report(worker.url())
            .map_or(0, |(report, _)| report.loads.len())
    }

    /// The monitor's eviction, as the registry's removal event drives it for
    /// a removed worker: the overload flag, the pushed state and the memo go
    /// now, the snapshot entry in a rebuild flushed here, so a worker that
    /// stays registered after its reports stopped reads empty at once.
    fn evict(&self, id: WorkerId) {
        let (Some(monitor), Some(worker)) = (self.monitor(), self.resolve(id)) else {
            return;
        };
        monitor.evict_worker_loads(&worker);
        monitor.flush_pending_evictions();
    }
}

// -------------------------------------------------------------- KV events

impl KvEventSink for HostStores {
    type Batch = KvEventBatch;

    /// A worker whose model has no index yet (no events subscription created
    /// one) takes no batch here.
    fn admit(
        &self,
        id: WorkerId,
        rank: u16,
        facts: BatchFacts,
        batch: &KvEventBatch,
    ) -> BatchOutcome {
        let cell = match self.feeds.get(&id) {
            Some(cell) => Arc::clone(cell.value()),
            None => {
                // Resolved before the entry is taken: the map's shard is not
                // held across the registry and the index.
                let Some(feed) = self.feed_of(id) else {
                    return BatchOutcome::Skipped;
                };
                let Some(cell) = self.install_feed(id, feed) else {
                    return BatchOutcome::Skipped;
                };
                cell
            }
        };
        let mut guard = cell.lock();
        let Some(feed) = &mut *guard else {
            drop(guard);
            self.prune_tombstone(id);
            return BatchOutcome::Skipped;
        };
        let cursor = feed.cursors.entry(rank).or_default();
        if let Some(chunk) = facts.snapshot {
            if chunk.index == 0 {
                feed.index
                    .apply_cleared(feed.interned, &mut feed.state.blocks);
            }
            cursor.resync_to(facts.sequence, chunk.unknown_before);
            KvEventMonitor::apply_events(
                &batch.events,
                feed.interned,
                &feed.index,
                &mut feed.state,
            );
            return BatchOutcome::Applied;
        }
        let admission = cursor.admit(facts.sequence, facts.clears);
        match admission {
            Admission::Stale => return BatchOutcome::Skipped,
            Admission::Replay { expected } => {
                return BatchOutcome::Gap {
                    expected,
                    received: facts.sequence,
                };
            }
            Admission::Apply | Admission::Restart | Admission::Unrecovered { .. } => {}
        }
        if admission.clears_rank() {
            feed.index
                .apply_cleared(feed.interned, &mut feed.state.blocks);
        }
        KvEventMonitor::apply_events(&batch.events, feed.interned, &feed.index, &mut feed.state);
        BatchOutcome::Applied
    }

    fn reconnected(&self, id: WorkerId) {
        if let Some(cell) = self.feeds.get(&id) {
            if let Some(feed) = &mut *cell.lock() {
                for cursor in feed.cursors.values_mut() {
                    cursor.reconnected();
                }
            }
        }
    }

    /// A rank's resync clears the worker's blocks: under one row per rank the
    /// worker's index membership is its own rank's.
    fn resync(&self, id: WorkerId, rank: Option<u16>, _reason: ResyncReason) {
        let Some(cell) = self.feeds.get(&id) else {
            return;
        };
        let mut guard = cell.lock();
        let Some(feed) = &mut *guard else {
            return;
        };
        match rank {
            Some(rank) => {
                if let Some(cursor) = feed.cursors.get_mut(&rank) {
                    cursor.reset();
                }
            }
            None => {
                for cursor in feed.cursors.values_mut() {
                    cursor.reset();
                }
            }
        }
        feed.index
            .apply_cleared(feed.interned, &mut feed.state.blocks);
    }

    fn cursor(&self, id: WorkerId, rank: u16) -> Option<KvCursor> {
        self.feeds
            .get(&id)?
            .lock()
            .as_ref()?
            .cursors
            .get(&rank)
            .map(RankCursor::view)
    }

    /// The entry becomes a tombstone under its shard, so a batch between the
    /// registry and the map finds it; a worker with no entry yet leaves one
    /// too while it still resolves.
    fn removed(&self, id: WorkerId) {
        let cell = Arc::clone(
            self.feeds
                .entry(id)
                .or_insert_with(|| Arc::new(Mutex::new(None)))
                .value(),
        );
        if let Some(feed) = cell.lock().take() {
            feed.retire();
        }
        // Tombstones whose workers left the registry go now, this one
        // included when its worker is already gone. A cell busy under a
        // batch is live and stays, without waiting on it: the shard's
        // write lock is held here.
        self.feeds.retain(|other, cell| {
            cell.try_lock().is_none_or(|guard| guard.is_some()) || self.resolve(*other).is_some()
        });
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.kv_monitor()?
            .block_size(model)
            .and_then(|size| u32::try_from(size).ok())
    }

    /// The monitor keeps a size its streams learned; a registration's seed
    /// never lowers it.
    fn set_block_size(&self, model: &str, size: u32) {
        if let Some(kv_monitor) = self.kv_monitor() {
            kv_monitor.set_block_size(model, usize::try_from(size).unwrap_or(usize::MAX));
        }
    }
}

// ----------------------------------------------------------- reservations

impl ReservationStore for HostStores {
    /// Both kinds read the engine's admission window: a decode's rooms are
    /// bounded by it today, and a prefill's slots take the same bound until
    /// the window moves into the store with its configured cap.
    fn window(&self, id: WorkerId, _kind: ReservationKind) -> Option<u32> {
        self.resolve(id)?.max_running_requests().map(u32::from)
    }

    fn try_reserve(
        &self,
        gateway: GatewayId,
        id: WorkerId,
        kind: ReservationKind,
        count: u32,
        lease: LeaseId,
    ) -> bool {
        let Some(worker) = self.resolve(id) else {
            return false;
        };
        let mut reservations = self.reservations.lock();
        let holds_kind = reservations
            .get(&(gateway, lease))
            .is_some_and(|held| held.iter().any(|claim| claim.kind == kind));
        if holds_kind {
            return false;
        }
        let window = worker.max_running_requests().map(usize::from);
        let wanted = usize::try_from(count).unwrap_or(usize::MAX);
        let claimed = match kind {
            // A decode without a window has no gate: nothing to claim.
            ReservationKind::Room => match window {
                Some(window) => {
                    if !worker.try_admit_pd(wanted, window) {
                        return false;
                    }
                    count
                }
                None => 0,
            },
            // A slot is the load guard: tracked whatever the window, bounded
            // by it when there is one.
            ReservationKind::Slot => {
                let cap = window.unwrap_or(usize::MAX);
                let mut taken = 0u32;
                while taken < count {
                    if !worker.try_increment_load(cap) {
                        for _ in 0..taken {
                            worker.decrement_load();
                        }
                        return false;
                    }
                    taken += 1;
                }
                taken
            }
        };
        reservations
            .entry((gateway, lease))
            .or_default()
            .push(Claim {
                worker,
                kind,
                count: claimed,
            });
        true
    }

    fn release(&self, gateway: GatewayId, lease: LeaseId) {
        let Some(held) = self.reservations.lock().remove(&(gateway, lease)) else {
            return;
        };
        for Claim {
            worker,
            kind,
            count,
        } in held
        {
            match kind {
                ReservationKind::Room => {
                    if count > 0 {
                        worker.release_pd(usize::try_from(count).unwrap_or(usize::MAX));
                    }
                }
                ReservationKind::Slot => {
                    for _ in 0..count {
                        worker.decrement_load();
                    }
                }
            }
        }
    }

    fn held(&self, id: WorkerId, kind: ReservationKind) -> u32 {
        let Some(worker) = self.resolve(id) else {
            return 0;
        };
        let held = match kind {
            ReservationKind::Room => worker.pd_admitted(),
            ReservationKind::Slot => worker.load(),
        };
        u32::try_from(held).unwrap_or(u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::{atomic::AtomicBool, mpsc, Arc},
        thread,
    };

    use openai_protocol::worker::RuntimeType;
    use smg_grpc_client::common_proto::{kv_cache_event, KvBlock, KvBlocksStored, KvCacheEvent};
    use smg_policy_controller::store::identity;

    use super::*;
    use crate::{
        config::PolicyConfig,
        policies::PolicyRegistry,
        worker::test_support::{no_health_check, ready_worker, worker_of},
    };

    fn facts(key: &str) -> WorkerFacts {
        WorkerFacts {
            key: key.to_string(),
            engine: key.to_string(),
            incarnation: 1,
            models: vec!["m".to_string()],
            dp_size: 1,
            ..WorkerFacts::default()
        }
    }

    fn host(registry: &Arc<WorkerRegistry>) -> HostStores {
        HostStores::new(Arc::clone(registry), None, None)
    }

    #[test]
    fn the_registry_host_registers_replaces_and_removes_under_the_views_ids() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let MembershipChange::Registered { id } = host.upsert(facts("http://w:8000")) else {
            panic!("a new key registers");
        };
        let row = WorkersStore::get(&host, id).expect("the row");
        assert_eq!(row.status, StoreStatus::Pending);
        assert_eq!(row.facts.models, vec!["m".to_string()]);
        assert_eq!(
            identity(&row.facts.key, row.facts.incarnation),
            row.id,
            "the row's id is the store's identity of its key and incarnation"
        );
        assert_eq!(
            row.id,
            worker_id(
                registry
                    .get_by_url("http://w:8000")
                    .expect("registered")
                    .as_ref()
            ),
            "and the view's id of the handle"
        );
        assert_eq!(
            WorkersStore::by_key(&host, "http://w:8000").map(|row| row.id),
            Some(id)
        );
        let (epoch, rows) = host.snapshot();
        assert_eq!((epoch, rows.len()), (1, 1));
        assert_eq!(host.changes_since(0), MembershipTail::SnapshotRequired);

        let stale = row.revision.wrapping_add(1);
        assert!(host.set_status(id, StoreStatus::Ready, stale).is_none());
        assert_eq!(
            host.set_status(id, StoreStatus::Ready, row.revision),
            Some(MembershipChange::StatusChanged {
                id,
                from: StoreStatus::Pending,
                to: StoreStatus::Ready
            })
        );
        assert_eq!(
            registry.get_by_url("http://w:8000").map(|w| w.status()),
            Some(WorkerStatus::Ready)
        );

        let MembershipChange::Replaced { old, new } = host.upsert(facts("http://w:8000")) else {
            panic!("the same key replaces");
        };
        assert_eq!(old, id);
        assert_ne!(new, old, "a new handle is a new id");
        assert!(WorkersStore::get(&host, old).is_none(), "the old id left");
        let revision = WorkersStore::get(&host, new).expect("the row").revision;
        assert_eq!(
            host.remove(new, revision),
            Some(MembershipChange::Removed { id: new })
        );
        assert!(registry.get_by_url("http://w:8000").is_none());
        assert_eq!(host.epoch(), 4);

        let mut trusted = facts("http://t:8000");
        trusted.health.probes = false;
        let MembershipChange::Registered { id: trusted_id } = host.upsert(trusted) else {
            panic!("a new key registers");
        };
        assert_eq!(
            WorkersStore::get(&host, trusted_id)
                .expect("the row")
                .status,
            StoreStatus::Ready,
            "a worker nobody probes registers ready, the builder's rule"
        );
    }

    #[test]
    fn the_registry_host_runs_the_probers_machine_under_the_revision_guard() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://p:8000")
                .model(ModelCard::new("m"))
                .health_config(HealthCheckConfig {
                    failure_threshold: 3,
                    success_threshold: 2,
                    ..HealthCheckConfig::default()
                })
                .build(),
        );
        registry.register(worker).expect("registered");
        let row = WorkersStore::by_key(&host, "http://p:8000").expect("the row");
        let (id, revision) = (row.id, row.revision);
        assert_eq!(host.probe_outcome(id, true, revision), None);
        assert_eq!(
            host.probe_outcome(id, true, revision.wrapping_add(1)),
            None,
            "a stale outcome is dropped"
        );
        assert_eq!(
            host.probe_outcome(id, true, revision),
            Some((StoreStatus::Pending, StoreStatus::Ready))
        );
        for _ in 0..2 {
            assert_eq!(host.probe_outcome(id, false, revision), None);
        }
        assert_eq!(
            host.probe_outcome(id, false, revision),
            Some((StoreStatus::Ready, StoreStatus::NotReady))
        );
        assert_eq!(
            host.connected(id, revision),
            Some((StoreStatus::NotReady, StoreStatus::Ready))
        );
        let now = Instant::now();
        host.gateway_veto(GatewayId(7), id, Some(now + Duration::from_secs(1)));
        assert!(host.vetoed(GatewayId(7), id, now));
        assert!(!host.vetoed(GatewayId::LOCAL, id, now));
        host.completed(GatewayId(7), id, TransportClass::Ok, now);
        assert!(host.completed_ok_since(GatewayId(7), id, now));
        assert!(
            !host.completed_ok_since(GatewayId::LOCAL, id, now),
            "another gateway's dispatch does not count"
        );
    }

    fn monitor_host() -> (
        tokio::runtime::Runtime,
        Arc<WorkerRegistry>,
        Arc<WorkerMonitor>,
        HostStores,
    ) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime for the monitor's publish task");
        let registry = Arc::new(WorkerRegistry::new());
        let monitor = {
            let _enter = runtime.enter();
            Arc::new(WorkerMonitor::new(
                Arc::clone(&registry),
                Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
                1,
                false,
                false,
            ))
        };
        let host = HostStores::new(Arc::clone(&registry), Some(&monitor), None);
        (runtime, registry, monitor, host)
    }

    /// The store contract's scrape rule through the host: a scrape record is
    /// the gateway's own read, so it writes no push receipt and the poll
    /// stays, while a stream record of the same worker suppresses it.
    #[test]
    fn a_scrape_record_writes_no_receipt_so_the_poll_stays_through_the_host() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let interval = Duration::from_secs(10);
        let t0 = Instant::now();
        let row = LoadRow {
            running: 4,
            live_rate: 2_000,
            ..LoadRow::default()
        };
        host.record(id, 0, row, t0, LoadSource::Scrape);
        monitor.flush_pushed();
        assert_eq!(
            host.load(id, 0).map(|sample| sample.row.running),
            Some(4),
            "the row is held"
        );
        assert_eq!(
            host.poll_mode(id, t0 + Duration::from_secs(1), interval),
            PollMode::Poll,
            "a scrape writes no receipt"
        );
        host.record(id, 0, row, t0 + Duration::from_secs(1), LoadSource::Stream);
        assert_eq!(
            host.poll_mode(id, t0 + Duration::from_secs(2), interval),
            PollMode::Suppressed,
            "the stream's receipt is the one read"
        );
    }

    /// A read of one rank merges where the stream's records land: a rank
    /// whose pushed record waits for the window keeps it through another
    /// rank's scrape, instead of taking the last publish's row back.
    #[test]
    fn a_scrape_of_one_rank_keeps_another_ranks_pending_push() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let t0 = Instant::now();
        let row = |running| LoadRow {
            running,
            ..LoadRow::default()
        };
        host.record(id, 0, row(4), t0, LoadSource::Stream);
        monitor.flush_pushed();
        assert_eq!(host.load(id, 0).map(|sample| sample.row.running), Some(4));
        host.record(
            id,
            0,
            row(9),
            t0 + Duration::from_millis(10),
            LoadSource::Stream,
        );
        host.record(
            id,
            1,
            row(2),
            t0 + Duration::from_millis(20),
            LoadSource::Scrape,
        );
        monitor.flush_pushed();
        assert_eq!(
            host.load(id, 0).map(|sample| sample.row.running),
            Some(9),
            "the pending push outlives the scrape of the other rank"
        );
        assert_eq!(host.load(id, 1).map(|sample| sample.row.running), Some(2));
        assert_eq!(host.ranks(id), 2);
    }

    /// A scrape row's sample instant is its receipt less the row's age, as
    /// a stream record's is: the scorer releases the dispatches made before
    /// the engine's state was read, not before the publish.
    #[test]
    fn a_scrape_rows_sample_instant_is_its_receipt_less_its_age() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let received_at = Instant::now();
        let row = LoadRow {
            running: 4,
            report_age_ms: 300,
            ..LoadRow::default()
        };
        host.record(id, 0, row, received_at, LoadSource::Scrape);
        monitor.flush_pushed();
        let sample = host.load(id, 0).expect("the row");
        assert_eq!(
            sample.sampled_at,
            received_at - Duration::from_millis(300),
            "the age, with no one-way margin"
        );
    }

    /// A member of a data-parallel group speaks for its own rank: a record
    /// or a report row for another rank is dropped, on the read path as on
    /// the stream's.
    #[test]
    fn a_members_record_and_report_keep_only_its_own_rank() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let member: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://m:8000")
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("m"))
                .health_config(no_health_check())
                .dp_config(1, 2)
                .build(),
        );
        member.set_status(WorkerStatus::Ready);
        registry.register(Arc::clone(&member)).expect("registered");
        let id = worker_id(member.as_ref());
        let t0 = Instant::now();
        let row = |running| LoadRow {
            running,
            ..LoadRow::default()
        };
        host.report(id, &[(0, row(5))], t0, LoadSource::Poll);
        assert_eq!(
            host.ranks(id),
            0,
            "a report of another rank alone says nothing"
        );
        host.report(id, &[(0, row(5)), (1, row(3))], t0, LoadSource::Poll);
        assert_eq!(host.load(id, 1).map(|sample| sample.row.running), Some(3));
        assert!(
            host.load(id, 0).is_none(),
            "the other rank's row is dropped"
        );
        host.record(
            id,
            0,
            row(7),
            t0 + Duration::from_secs(1),
            LoadSource::Scrape,
        );
        host.record(
            id,
            1,
            row(8),
            t0 + Duration::from_secs(1),
            LoadSource::Scrape,
        );
        monitor.flush_pushed();
        assert_eq!(host.load(id, 1).map(|sample| sample.row.running), Some(8));
        assert!(
            host.load(id, 0).is_none(),
            "a record for the other rank is dropped"
        );
        assert_eq!(host.ranks(id), 1);
    }

    /// An authoritative empty poll retracts a read's rows as it retracts the
    /// pre-poll pushed ones: a worker whose rows came from scrapes alone
    /// keeps none of them past it, and a later scrape of one rank brings no
    /// other rank back.
    #[test]
    fn an_empty_poll_retracts_scraped_rows_without_a_receipt() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let t0 = Instant::now();
        let row = |running| LoadRow {
            running,
            ..LoadRow::default()
        };
        host.record(id, 0, row(4), t0, LoadSource::Scrape);
        host.record(id, 1, row(5), t0, LoadSource::Scrape);
        monitor.flush_pushed();
        assert_eq!(host.ranks(id), 2);
        host.record(
            id,
            1,
            row(7),
            t0 + Duration::from_millis(500),
            LoadSource::Scrape,
        );
        host.report(id, &[], t0 + Duration::from_secs(1), LoadSource::Poll);
        assert_eq!(host.ranks(id), 0, "the empty poll retracts the read rows");
        monitor.flush_pushed();
        assert_eq!(
            host.ranks(id),
            0,
            "a flush after the empty poll publishes nothing"
        );
        host.record(
            id,
            0,
            row(6),
            t0 + Duration::from_secs(2),
            LoadSource::Scrape,
        );
        monitor.flush_pushed();
        assert_eq!(host.load(id, 0).map(|sample| sample.row.running), Some(6));
        assert!(
            host.load(id, 1).is_none(),
            "the retracted rank does not come back with the next read"
        );
        assert_eq!(host.ranks(id), 1);
    }

    /// A read of one rank on a worker without a stream keeps the ranks a
    /// poll learned: the read is laid over the published rows and published
    /// at once, with no report of its own for the window to publish over
    /// them.
    #[test]
    fn a_scrape_of_one_rank_keeps_the_polls_other_ranks() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let t0 = Instant::now();
        let row = |running| LoadRow {
            running,
            ..LoadRow::default()
        };
        host.report(id, &[(0, row(3)), (1, row(4))], t0, LoadSource::Poll);
        assert_eq!(host.ranks(id), 2);
        host.record(
            id,
            1,
            row(7),
            t0 + Duration::from_secs(1),
            LoadSource::Scrape,
        );
        monitor.flush_pushed();
        assert_eq!(
            host.load(id, 0).map(|sample| sample.row.running),
            Some(3),
            "the poll's other rank stays published"
        );
        assert_eq!(host.load(id, 1).map(|sample| sample.row.running), Some(7));
        assert_eq!(host.ranks(id), 2);
    }

    /// Repeated reads of a worker's ranks advance the report's sample
    /// instant: each row keeps its own read's instant, a row no read
    /// refreshed the poll's, and the report takes the earliest of them, so
    /// the scorer releases the dispatches every current row has seen.
    #[test]
    fn repeated_reads_of_two_ranks_advance_the_reports_sample_instant() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let t0 = Instant::now();
        let at = |secs| t0 + Duration::from_secs(secs);
        let row = |running| LoadRow {
            running,
            ..LoadRow::default()
        };
        host.report(id, &[(0, row(3)), (1, row(4))], t0, LoadSource::Poll);
        let instant = |rank| host.load(id, rank).map(|sample| sample.sampled_at);
        host.record(id, 0, row(5), at(1), LoadSource::Scrape);
        assert_eq!(instant(0), Some(t0), "rank 1 is still the poll's row");
        host.record(id, 1, row(6), at(2), LoadSource::Scrape);
        assert_eq!(
            instant(1),
            Some(at(1)),
            "both rows are reads: the earlier read's instant"
        );
        host.record(id, 0, row(7), at(3), LoadSource::Scrape);
        assert_eq!(
            instant(0),
            Some(at(2)),
            "rank 0 refreshed: rank 1's read is the earliest"
        );
        monitor.flush_pushed();
        assert_eq!(host.load(id, 0).map(|sample| sample.row.running), Some(7));
        assert_eq!(host.load(id, 1).map(|sample| sample.row.running), Some(6));
    }

    /// A poll after reads supersedes their instants: its rows are the
    /// poll's, so a later read of one rank takes the poll's instant for the
    /// other rank, not the stale read's.
    #[test]
    fn a_poll_after_reads_supersedes_their_sample_instants() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let t0 = Instant::now();
        let at = |secs| t0 + Duration::from_secs(secs);
        let row = |running| LoadRow {
            running,
            ..LoadRow::default()
        };
        host.report(id, &[(0, row(3)), (1, row(4))], t0, LoadSource::Poll);
        let instant = |rank| host.load(id, rank).map(|sample| sample.sampled_at);
        host.record(id, 0, row(5), at(1), LoadSource::Scrape);
        host.record(id, 1, row(6), at(2), LoadSource::Scrape);
        assert_eq!(instant(1), Some(at(1)));
        host.report(id, &[(0, row(8)), (1, row(9))], at(3), LoadSource::Poll);
        assert_eq!(instant(0), Some(at(3)), "the poll's instant for both rows");
        host.record(id, 0, row(10), at(4), LoadSource::Scrape);
        assert_eq!(
            instant(0),
            Some(at(3)),
            "rank 1 is the poll's row again, not the earlier read's"
        );
        monitor.flush_pushed();
        assert_eq!(host.load(id, 0).map(|sample| sample.row.running), Some(10));
        assert_eq!(host.load(id, 1).map(|sample| sample.row.running), Some(9));
    }

    /// A poll replayed with its old instant after a read refreshed a rank:
    /// the rank's newest sample wins, so the read's value and instant stay
    /// and the replay's older row is dropped for it, and the next read of
    /// the other rank carries the first read's value under the earlier
    /// read's instant, not the replay's value under it.
    #[test]
    fn a_replayed_poll_keeps_a_ranks_newer_read() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let t0 = Instant::now();
        let at = |secs| t0 + Duration::from_secs(secs);
        let row = |running| LoadRow {
            running,
            ..LoadRow::default()
        };
        let sample = |rank| {
            host.load(id, rank)
                .map(|sample| (sample.row.running, sample.sampled_at))
        };
        host.report(id, &[(0, row(3)), (1, row(4))], t0, LoadSource::Poll);
        host.record(id, 0, row(5), at(1), LoadSource::Scrape);
        host.report(id, &[(0, row(3)), (1, row(4))], t0, LoadSource::Poll);
        assert_eq!(
            sample(0),
            Some((5, t0)),
            "the replay's older row is dropped: rank 0 keeps the read's value, the report rank 1's instant"
        );
        host.record(id, 1, row(6), at(2), LoadSource::Scrape);
        assert_eq!(
            sample(0),
            Some((5, at(1))),
            "rank 0 is the read's value under the earlier read's instant"
        );
        assert_eq!(sample(1), Some((6, at(1))));
        let _ = monitor;
    }

    /// A read holds its worker's lock from the snapshot read through the
    /// publish: a second read of the worker waits for it, so neither
    /// publishes the other's rank from a snapshot the other has replaced.
    #[test]
    fn a_read_waits_for_the_lock_another_read_of_the_worker_holds() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let t0 = Instant::now();
        let row = |running| LoadRow {
            running,
            ..LoadRow::default()
        };
        host.report(id, &[(0, row(3)), (1, row(4))], t0, LoadSource::Poll);
        let lock = monitor.rank_lock(worker.url());
        let held = lock.lock();
        let handle = runtime.handle().clone();
        let started = AtomicBool::new(false);
        let (host, started) = (&host, &started);
        thread::scope(move |scope| {
            let reader = scope.spawn(move || {
                let _enter = handle.enter();
                started.store(true, Ordering::SeqCst);
                host.record(
                    id,
                    1,
                    row(7),
                    t0 + Duration::from_secs(1),
                    LoadSource::Scrape,
                );
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            while !started.load(Ordering::SeqCst) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            thread::sleep(Duration::from_millis(200));
            assert_eq!(
                host.load(id, 1).map(|sample| sample.row.running),
                Some(4),
                "the read waits while the lock is held"
            );
            drop(held);
            reader.join().expect("the reader thread");
        });
        monitor.flush_pushed();
        assert_eq!(host.load(id, 1).map(|sample| sample.row.running), Some(7));
        assert_eq!(host.load(id, 0).map(|sample| sample.row.running), Some(3));
    }

    /// A read holds its worker's lock through the publish: at the instant
    /// the report is published the lock is still taken, so no second read
    /// can slip its snapshot read between this read's and its publish.
    #[test]
    fn a_read_holds_its_workers_lock_through_the_publish() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let t0 = Instant::now();
        let row = |running| LoadRow {
            running,
            ..LoadRow::default()
        };
        host.report(id, &[(0, row(3)), (1, row(4))], t0, LoadSource::Poll);
        let lock = monitor.rank_lock(worker.url());
        let held_at_publish = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&held_at_publish);
        *monitor.before_read_publish.lock() = Some(Box::new(move || {
            seen.store(lock.try_lock().is_none(), Ordering::SeqCst);
        }));
        host.record(
            id,
            1,
            row(7),
            t0 + Duration::from_secs(1),
            LoadSource::Scrape,
        );
        assert!(
            held_at_publish.load(Ordering::SeqCst),
            "the worker's read lock is held when the read publishes"
        );
        *monitor.before_read_publish.lock() = None;
        monitor.flush_pushed();
        assert_eq!(host.load(id, 1).map(|sample| sample.row.running), Some(7));
    }

    /// The store contract's eviction through the host: a worker that stays
    /// registered after its reports stopped loses its samples, its rank
    /// count and its push receipt on `evict`, and its poll resumes.
    #[test]
    fn evict_clears_a_registered_workers_loads_through_the_host() {
        let (runtime, registry, monitor, host) = monitor_host();
        let _enter = runtime.enter();
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let interval = Duration::from_secs(10);
        let t0 = Instant::now();
        let row = LoadRow {
            running: 4,
            live_rate: 2_000,
            ..LoadRow::default()
        };
        host.record(id, 0, row, t0, LoadSource::Stream);
        monitor.flush_pushed();
        assert_eq!(host.ranks(id), 1);
        assert_eq!(
            host.poll_mode(id, t0 + Duration::from_secs(1), interval),
            PollMode::Suppressed
        );
        host.evict(id);
        assert_eq!(
            host.load(id, 0).map(|sample| sample.row.running),
            None,
            "the samples are gone"
        );
        assert_eq!(host.ranks(id), 0, "the rank count is gone");
        assert_eq!(
            host.poll_mode(id, t0 + Duration::from_secs(2), interval),
            PollMode::Poll,
            "the receipt is gone, the poll resumes"
        );
    }

    #[test]
    fn the_monitor_host_feeds_a_record_and_a_polls_rows_into_the_published_report() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime for the monitor's publish task");
        let _enter = runtime.enter();
        let registry = Arc::new(WorkerRegistry::new());
        let monitor = Arc::new(WorkerMonitor::new(
            Arc::clone(&registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            1,
            false,
            false,
        ));
        let host = HostStores::new(Arc::clone(&registry), Some(&monitor), None);
        let worker = ready_worker("http://w:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let interval = Duration::from_secs(10);
        let t0 = Instant::now();
        assert_eq!(host.poll_mode(id, t0, interval), PollMode::Poll);
        assert_eq!(host.ranks(id), 0);

        let row = LoadRow {
            running: 4,
            queued: 2,
            kv_usage_permille: 250,
            active_kv_permille: 250,
            live_rate: 2_000,
            max_running: 16,
            ..LoadRow::default()
        };
        host.record(id, 0, row, t0, LoadSource::Stream);
        monitor.flush_pushed();
        let sample = host.load(id, 0).expect("the published row");
        assert_eq!((sample.row.running, sample.row.queued), (4, 2));
        assert_eq!(sample.row.kv_usage_permille, 250);
        assert_eq!(sample.source, LoadSource::Stream);
        assert_eq!(host.ranks(id), 1);
        assert_eq!(
            host.poll_mode(id, t0 + Duration::from_secs(5), interval),
            PollMode::Suppressed
        );
        assert_eq!(
            host.poll_mode(id, t0 + Duration::from_secs(20), interval),
            PollMode::Fallback
        );

        let polled = LoadRow { running: 9, ..row };
        host.report(
            id,
            &[(0, polled)],
            t0 - Duration::from_millis(1),
            LoadSource::Poll,
        );
        assert_eq!(
            host.load(id, 0).expect("the row").row.running,
            4,
            "a rank that pushed since the poll began keeps the stream's row in the poll's publish"
        );
        host.record(
            id,
            1,
            row,
            t0 + Duration::from_millis(500),
            LoadSource::Stream,
        );
        monitor.flush_pushed();
        assert_eq!(
            host.load(id, 0).expect("the row").row.running,
            4,
            "a rank that pushed since the poll began keeps the stream's row on the pushed report"
        );
        host.report(
            id,
            &[(0, polled)],
            t0 + Duration::from_secs(1),
            LoadSource::Poll,
        );
        assert_eq!(
            host.load(id, 0).expect("the row").row.running,
            9,
            "a quiet rank takes the poll's row at once"
        );
        host.record(id, 1, row, t0 + Duration::from_secs(2), LoadSource::Stream);
        monitor.flush_pushed();
        assert_eq!(
            host.load(id, 0).expect("the row").row.running,
            9,
            "and keeps it at the next publish"
        );
        assert_eq!(host.ranks(id), 2);
    }

    fn stored(sequence: u64, hash: i64) -> KvEventBatch {
        let first = u32::try_from(hash).expect("a small hash") * 16;
        KvEventBatch {
            sequence_number: sequence,
            events: vec![KvCacheEvent {
                event_id: sequence,
                data: Some(kv_cache_event::Data::Stored(KvBlocksStored {
                    blocks: vec![KvBlock {
                        block_hash: hash,
                        token_ids: (first..first + 16).collect(),
                        block_size: 16,
                        ..KvBlock::default()
                    }],
                    ..KvBlocksStored::default()
                })),
            }],
            ..KvEventBatch::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_kv_host_runs_the_cursor_in_front_of_the_models_index() {
        let registry = Arc::new(WorkerRegistry::new());
        let kv_monitor = Arc::new(KvEventMonitor::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:1")
                .model(ModelCard::new("m"))
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .health_config(no_health_check())
                .build(),
        );
        registry.register(Arc::clone(&worker)).expect("registered");
        let host = HostStores::new(Arc::clone(&registry), None, Some(&kv_monitor));
        let id = worker_id(worker.as_ref());
        let facts = |sequence: u64| BatchFacts {
            sequence,
            ..BatchFacts::default()
        };
        assert_eq!(
            host.admit(id, 0, facts(1), &stored(1, 11)),
            BatchOutcome::Skipped,
            "no index for the model yet"
        );
        kv_monitor
            .on_worker_added(&worker, crate::worker::SubscriptionKind::Events)
            .await;
        let index = kv_monitor.get_indexer("m").expect("the model's index");
        assert_eq!(
            host.admit(id, 0, facts(1), &stored(1, 11)),
            BatchOutcome::Applied
        );
        assert_eq!(
            host.admit(id, 0, facts(2), &stored(2, 12)),
            BatchOutcome::Applied
        );
        let interned = index.worker_id(worker.url()).expect("interned");
        assert_eq!(index.worker_block_count(interned), 2);
        assert_eq!(
            host.admit(id, 0, facts(2), &stored(2, 12)),
            BatchOutcome::Skipped,
            "a duplicate"
        );
        assert_eq!(
            host.admit(id, 0, facts(9), &stored(9, 19)),
            BatchOutcome::Gap {
                expected: 3,
                received: 9
            }
        );
        assert_eq!(host.cursor(id, 0).expect("heard").applied, 2);
        host.resync(id, Some(0), ResyncReason::OutOfRange);
        assert_eq!(
            index.worker_block_count(interned),
            0,
            "the resync cleared the blocks"
        );
        assert_eq!(host.cursor(id, 0).expect("heard").applied, 0);
        assert_eq!(
            host.admit(id, 0, facts(40), &stored(40, 50)),
            BatchOutcome::Applied
        );
        assert_eq!(index.worker_block_count(interned), 1);
        host.set_block_size("m", 16);
        assert_eq!(host.block_size("m"), Some(16));
        host.removed(id);
        assert_eq!(host.cursor(id, 0), None);
        assert_eq!(index.worker_block_count(interned), 0);
        kv_monitor.stop().await;
    }

    /// A removal between the feed map and the feed lock frees the worker's
    /// slot: a batch that took the feed before it lands nothing, and a
    /// resync after it clears nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_batch_that_took_the_feed_before_the_removal_lands_on_nothing() {
        let registry = Arc::new(WorkerRegistry::new());
        let kv_monitor = Arc::new(KvEventMonitor::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:2")
                .model(ModelCard::new("m"))
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .health_config(no_health_check())
                .build(),
        );
        registry.register(Arc::clone(&worker)).expect("registered");
        let host = HostStores::new(Arc::clone(&registry), None, Some(&kv_monitor));
        let id = worker_id(worker.as_ref());
        kv_monitor
            .on_worker_added(&worker, crate::worker::SubscriptionKind::Events)
            .await;
        let index = kv_monitor.get_indexer("m").expect("the model's index");
        let facts = |sequence: u64| BatchFacts {
            sequence,
            ..BatchFacts::default()
        };
        assert_eq!(
            host.admit(id, 0, facts(1), &stored(1, 21)),
            BatchOutcome::Applied
        );
        let interned = index.worker_id(worker.url()).expect("interned");
        assert_eq!(index.worker_block_count(interned), 1);
        // The batch in flight resolved its cell before the removal.
        let in_flight = Arc::clone(host.feeds.get(&id).expect("the feed").value());
        host.removed(id);
        assert_eq!(index.worker_block_count(interned), 0);
        assert!(
            in_flight.lock().is_none(),
            "the removal leaves a tombstone under the cell's lock"
        );
        assert_eq!(
            host.admit(id, 0, facts(2), &stored(2, 22)),
            BatchOutcome::Skipped,
            "the late batch lands nothing"
        );
        host.resync(id, None, ResyncReason::OutOfRange);
        assert_eq!(
            index.worker_block_count(interned),
            0,
            "no phantom blocks on the freed slot"
        );
        assert!(
            host.feeds.contains_key(&id),
            "the tombstone stays while the worker resolves"
        );
        let rid = registry.get_id_by_url(worker.url()).expect("registered");
        registry.remove(&rid);
        assert_eq!(
            host.admit(id, 0, facts(3), &stored(3, 23)),
            BatchOutcome::Skipped
        );
        assert!(
            !host.feeds.contains_key(&id),
            "and goes with the next batch once the worker left"
        );
        kv_monitor.stop().await;
    }

    /// A removal that runs between a batch's registry lookup and its map
    /// insert leaves the tombstone the insert finds: no feed, nothing
    /// applied, no cursor.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_removal_before_the_first_batch_leaves_the_tombstone_the_batch_finds() {
        let registry = Arc::new(WorkerRegistry::new());
        let kv_monitor = Arc::new(KvEventMonitor::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:3")
                .model(ModelCard::new("m"))
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .health_config(no_health_check())
                .build(),
        );
        registry.register(Arc::clone(&worker)).expect("registered");
        let host = HostStores::new(Arc::clone(&registry), None, Some(&kv_monitor));
        let id = worker_id(worker.as_ref());
        kv_monitor
            .on_worker_added(&worker, crate::worker::SubscriptionKind::Events)
            .await;
        let index = kv_monitor.get_indexer("m").expect("the model's index");
        host.removed(id);
        assert!(
            host.feeds.contains_key(&id),
            "a tombstone while the worker resolves"
        );
        let facts = BatchFacts {
            sequence: 1,
            ..BatchFacts::default()
        };
        assert_eq!(
            host.admit(id, 0, facts, &stored(1, 31)),
            BatchOutcome::Skipped
        );
        assert!(
            host.feeds.get(&id).expect("the cell").lock().is_none(),
            "the batch found the tombstone, not a fresh feed"
        );
        assert_eq!(
            index
                .worker_id(worker.url())
                .map_or(0, |slot| index.worker_block_count(slot)),
            0
        );
        assert_eq!(host.cursor(id, 0), None);
        kv_monitor.stop().await;
    }

    /// A removal that runs after a batch's registry lookup and after the
    /// registry dropped the worker sweeps its own tombstone before the
    /// batch's insert, so the insert reads the registry once more under
    /// its cell: the feed of a worker gone by then leaves the map and the
    /// index at once instead of outliving the worker.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_feed_built_before_a_removal_after_the_registry_drop_is_retired_at_its_insert() {
        let registry = Arc::new(WorkerRegistry::new());
        let kv_monitor = Arc::new(KvEventMonitor::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:3")
                .model(ModelCard::new("m"))
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .health_config(no_health_check())
                .build(),
        );
        registry.register(Arc::clone(&worker)).expect("registered");
        let host = HostStores::new(Arc::clone(&registry), None, Some(&kv_monitor));
        let id = worker_id(worker.as_ref());
        // The index placed without a stream: the host is the only writer that
        // interns the URL, so the slot's return below is deterministic (a
        // subscription's task would intern it again with no ordering against
        // this body).
        let index = Arc::new(KvIndex::chain());
        kv_monitor.set_index("m", Arc::clone(&index));
        // The batch's lookup: the worker resolves and its slot is interned.
        let feed = host.feed_of(id).expect("the feed of a live worker");
        // Then the registry drops the worker and the removal sweeps.
        let rid = registry
            .get_id_by_url("grpc://127.0.0.1:3")
            .expect("the registry id");
        registry.remove(&rid);
        host.removed(id);
        assert!(
            !host.feeds.contains_key(&id),
            "the removal swept its own tombstone"
        );
        assert!(
            host.install_feed(id, feed).is_none(),
            "a worker gone by the insert takes no feed"
        );
        assert!(
            !host.feeds.contains_key(&id),
            "no entry outlives the worker"
        );
        let facts = BatchFacts {
            sequence: 1,
            ..BatchFacts::default()
        };
        assert_eq!(
            host.admit(id, 0, facts, &stored(1, 31)),
            BatchOutcome::Skipped
        );
        assert_eq!(
            index.worker_id(worker.url()),
            None,
            "the URL's slot went back to the index with the retired feed"
        );
        kv_monitor.stop().await;
    }

    /// The sweep a removal runs holds the shard's write lock: a cell busy
    /// under a batch is kept without waiting on it, so one worker's long
    /// apply does not stall another worker's removal and the shard's reads.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_removal_keeps_a_busy_feed_without_waiting_on_it() {
        let registry = Arc::new(WorkerRegistry::new());
        let kv_monitor = Arc::new(KvEventMonitor::new());
        let mut ids = Vec::new();
        for url in ["grpc://127.0.0.1:3", "grpc://127.0.0.1:4"] {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(url)
                    .model(ModelCard::new("m"))
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::TokenSpeed)
                    .health_config(no_health_check())
                    .build(),
            );
            registry.register(Arc::clone(&worker)).expect("registered");
            kv_monitor
                .on_worker_added(&worker, crate::worker::SubscriptionKind::Events)
                .await;
            ids.push(worker_id(worker.as_ref()));
        }
        let host = Arc::new(HostStores::new(
            Arc::clone(&registry),
            None,
            Some(&kv_monitor),
        ));
        let (busy, other) = (ids[0], ids[1]);
        for id in [busy, other] {
            let facts = BatchFacts {
                sequence: 1,
                ..BatchFacts::default()
            };
            assert_eq!(
                host.admit(id, 0, facts, &stored(1, 31)),
                BatchOutcome::Applied
            );
        }
        // A batch holds the busy worker's cell while the other is removed;
        // the busy worker has left the registry by then, so only the lock's
        // state keeps its cell in the sweep.
        let cell = Arc::clone(host.feeds.get(&busy).expect("the busy cell").value());
        let held = cell.lock();
        let rid = registry
            .get_id_by_url("grpc://127.0.0.1:3")
            .expect("the busy worker's registry id");
        registry.remove(&rid);
        let (done, swept) = mpsc::channel();
        let remover = Arc::clone(&host);
        thread::spawn(move || {
            remover.removed(other);
            let _ = done.send(());
        });
        assert!(
            swept.recv_timeout(Duration::from_secs(5)).is_ok(),
            "the removal returned while the other cell stayed busy"
        );
        assert!(
            host.feeds.contains_key(&busy),
            "the busy feed stays while the batch holds it, registered or not"
        );
        drop(held);
        assert!(
            host.feeds.get(&other).expect("the cell").lock().is_none(),
            "the removed worker's cell is a tombstone while it resolves"
        );
        kv_monitor.stop().await;
    }

    /// A same-URL replacement ends the old id before a batch's feed is
    /// installed, but the URL's interned slot now belongs to the live
    /// successor: the fresh feed is dropped without releasing it, and the
    /// successor's batches land on that slot.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_fresh_feed_of_a_replaced_id_leaves_the_urls_slot_to_the_successor() {
        let registry = Arc::new(WorkerRegistry::new());
        let kv_monitor = Arc::new(KvEventMonitor::new());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:3")
                .model(ModelCard::new("m"))
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::TokenSpeed)
                .health_config(no_health_check())
                .build(),
        );
        registry.register(Arc::clone(&worker)).expect("registered");
        let host = HostStores::new(Arc::clone(&registry), None, Some(&kv_monitor));
        let id = worker_id(worker.as_ref());
        kv_monitor
            .on_worker_added(&worker, crate::worker::SubscriptionKind::Events)
            .await;
        let index = kv_monitor.get_indexer("m").expect("the model's index");
        let feed = host.feed_of(id).expect("the feed of a live worker");
        let slot = index.worker_id(worker.url()).expect("the URL's slot");
        // A refresh under the same URL replaces the handle: a new id under
        // the same URL and the same slot.
        let MembershipChange::Replaced { old, new } = host.upsert(facts("grpc://127.0.0.1:3"))
        else {
            panic!("a live key refreshes");
        };
        assert_eq!(old, id);
        assert!(
            host.install_feed(id, feed).is_none(),
            "the replaced id takes no feed"
        );
        assert_eq!(
            index.worker_id(worker.url()),
            Some(slot),
            "the URL's slot stays with its live owner"
        );
        let facts = BatchFacts {
            sequence: 1,
            ..BatchFacts::default()
        };
        assert_eq!(
            host.admit(new, 0, facts, &stored(1, 31)),
            BatchOutcome::Applied
        );
        assert_eq!(
            index.worker_block_count(slot),
            1,
            "the successor writes under the same slot"
        );
        kv_monitor.stop().await;
    }

    /// The monitor holds the policy registry, the registry its decider and
    /// the decider these stores: a strong edge back to the monitors would
    /// close the cycle and keep them alive after their owner let go.
    #[test]
    fn the_host_holds_the_monitors_weakly_so_the_decider_keeps_none_alive() {
        use crate::config::PolicyEngineMode;

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime for the monitors");
        let _enter = runtime.enter();
        let registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(
            PolicyRegistry::new(PolicyConfig::RoundRobin).with_policy_engine(
                PolicyEngineMode::Engine,
                &PolicyConfig::RoundRobin,
                &PolicyConfig::RoundRobin,
            ),
        );
        let monitor = Arc::new(WorkerMonitor::new(
            Arc::clone(&registry),
            Arc::clone(&policy_registry),
            1,
            false,
            false,
        ));
        let kv_monitor = Arc::new(KvEventMonitor::new());
        let decider = policy_registry.decider().expect("the engine");
        decider.set_stores(Arc::new(HostStores::new(
            Arc::clone(&registry),
            Some(&monitor),
            Some(&kv_monitor),
        )));
        let stores = decider.stores().expect("the host's stores");
        let (load_monitor, event_monitor) = (Arc::downgrade(&monitor), Arc::downgrade(&kv_monitor));
        drop(monitor);
        drop(kv_monitor);
        assert!(
            load_monitor.upgrade().is_none(),
            "the decider's stores do not keep the load monitor alive"
        );
        assert!(event_monitor.upgrade().is_none(), "nor the event monitor");
        assert_eq!(
            stores.block_size("m"),
            None,
            "a host outliving its monitors feeds nothing"
        );
    }

    #[test]
    fn the_admission_host_claims_slots_and_rooms_within_the_windows() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://d:8000")
                .worker_type(WorkerType::Decode)
                .label("max_running_requests", "2")
                .health_config(no_health_check())
                .build(),
        );
        registry.register(Arc::clone(&decode)).expect("registered");
        let id = worker_id(decode.as_ref());
        assert_eq!(host.window(id, ReservationKind::Room), Some(2));
        let (a, b, c) = (LeaseId(1), LeaseId(2), LeaseId(3));
        assert!(host.try_reserve(GatewayId::LOCAL, id, ReservationKind::Room, 2, a));
        assert!(
            !host.try_reserve(GatewayId::LOCAL, id, ReservationKind::Room, 1, b),
            "the window is full"
        );
        assert!(
            !host.try_reserve(GatewayId::LOCAL, id, ReservationKind::Room, 1, a),
            "one room claim per lease"
        );
        assert_eq!(host.held(id, ReservationKind::Room), 2);
        host.release(GatewayId::LOCAL, a);
        assert_eq!(host.held(id, ReservationKind::Room), 0);
        assert!(host.try_reserve(GatewayId::LOCAL, id, ReservationKind::Slot, 1, b));
        assert!(host.try_reserve(GatewayId::LOCAL, id, ReservationKind::Slot, 1, c));
        assert!(
            !host.try_reserve(GatewayId::LOCAL, id, ReservationKind::Slot, 1, LeaseId(4)),
            "two slots fit"
        );
        assert_eq!(host.held(id, ReservationKind::Slot), 2);
        assert_eq!(decode.load(), 2, "a slot is the load guard");
        host.release(GatewayId::LOCAL, b);
        host.release(GatewayId::LOCAL, b);
        assert_eq!(decode.load(), 1);

        let plain = worker_of("http://r:8000", WorkerType::Regular);
        registry.register(Arc::clone(&plain)).expect("registered");
        let plain_id = worker_id(plain.as_ref());
        assert_eq!(host.window(plain_id, ReservationKind::Room), None);
        assert!(
            host.try_reserve(
                GatewayId::LOCAL,
                plain_id,
                ReservationKind::Room,
                5,
                LeaseId(5)
            ),
            "no window: the gate abstains"
        );
        assert_eq!(
            host.held(plain_id, ReservationKind::Room),
            0,
            "and nothing is claimed"
        );
        assert!(host.try_reserve(
            GatewayId::LOCAL,
            plain_id,
            ReservationKind::Slot,
            3,
            LeaseId(6)
        ));
        assert_eq!(plain.load(), 3);
        host.release(GatewayId::LOCAL, LeaseId(6));
        assert_eq!(plain.load(), 0);
        // The contract's other half: a worker the host does not know is
        // refused, the same answer the memory store gives.
        assert!(
            !host.try_reserve(
                GatewayId::LOCAL,
                WorkerId(77),
                ReservationKind::Room,
                1,
                LeaseId(7)
            ),
            "a worker the host does not know is refused"
        );
    }

    #[test]
    fn two_gateways_sharing_a_lease_number_hold_separate_reservations() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let decode = decode_worker("grpc://d:8000", "2");
        registry.register(Arc::clone(&decode)).expect("registered");
        let id = worker_id(decode.as_ref());
        let (g1, g2) = (GatewayId(1), GatewayId(2));
        assert!(host.try_reserve(g1, id, ReservationKind::Room, 1, LeaseId(1)));
        assert!(
            host.try_reserve(g2, id, ReservationKind::Room, 1, LeaseId(1)),
            "the other gateway's lease of the same number is its own"
        );
        assert_eq!(host.held(id, ReservationKind::Room), 2);
        host.release(g1, LeaseId(1));
        assert_eq!(
            host.held(id, ReservationKind::Room),
            1,
            "one gateway's release frees its own claim alone"
        );
        host.release(g2, LeaseId(1));
        assert_eq!(host.held(id, ReservationKind::Room), 0);
    }

    fn decode_worker(url: &str, window: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Decode)
                .label("max_running_requests", window)
                .health_config(no_health_check())
                .build(),
        )
    }

    /// A same-URL replacement carries the admission counters to its
    /// successor on the shared runtime; a claim taken before it releases on
    /// them, not on the id the replacement left behind.
    #[test]
    fn a_claim_taken_before_a_replacement_releases_on_the_successors_counters() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let decode = decode_worker("grpc://d:8000", "2");
        registry.register(Arc::clone(&decode)).expect("registered");
        let id = worker_id(decode.as_ref());
        assert!(host.try_reserve(GatewayId::LOCAL, id, ReservationKind::Room, 1, LeaseId(1)));
        assert!(host.try_reserve(GatewayId::LOCAL, id, ReservationKind::Slot, 1, LeaseId(1)));
        let successor = decode_worker("grpc://d:8000", "2");
        registry.register_or_replace(Arc::clone(&successor));
        assert_eq!(
            (successor.pd_admitted(), successor.load()),
            (1, 1),
            "the counters moved with the runtime"
        );
        host.release(GatewayId::LOCAL, LeaseId(1));
        assert_eq!(
            (successor.pd_admitted(), successor.load()),
            (0, 0),
            "released on the successor"
        );
    }

    /// The revision check and the removal share the registry's per-worker
    /// lock: a stale revision removes nothing.
    #[test]
    fn a_removal_under_a_stale_revision_keeps_the_worker() {
        let registry = Arc::new(WorkerRegistry::new());
        let worker = ready_worker("http://g:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let rid = registry.get_id_by_url("http://g:8000").expect("the id");
        assert!(registry
            .remove_if_revision(&rid, worker.revision().wrapping_add(1))
            .is_none());
        assert!(registry.get_by_url("http://g:8000").is_some());
        assert!(registry
            .remove_if_revision(&rid, worker.revision())
            .is_some());
        assert!(registry.get_by_url("http://g:8000").is_none());
    }

    /// A rank's key is the engine's address with the rank appended once, and
    /// a handle serves every model its facts name.
    #[test]
    fn a_rank_row_keeps_its_key_and_every_model_through_a_refresh() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let rank = || {
            let mut rank = facts("http://w:8000@1");
            rank.engine = "http://w:8000".to_string();
            rank.dp_rank = 1;
            rank.dp_size = 2;
            rank.models = vec!["a".to_string(), "b".to_string()];
            rank.max_running = Some(6);
            rank.kv_block_size = Some(16);
            rank
        };
        let MembershipChange::Registered { id } = host.upsert(rank()) else {
            panic!("a new key registers");
        };
        let row = WorkersStore::get(&host, id).expect("the row");
        assert_eq!(
            (row.facts.key.as_str(), row.facts.engine.as_str()),
            ("http://w:8000@1", "http://w:8000")
        );
        assert_eq!(row.facts.models, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(
            row.facts.max_running,
            Some(6),
            "the window is a label on the handle"
        );
        assert_eq!(
            row.facts.kv_block_size,
            Some(16),
            "the block size the registration states"
        );
        let MembershipChange::Replaced { new, .. } = host.upsert(rank()) else {
            panic!("the same key refreshes");
        };
        let row = WorkersStore::get(&host, new).expect("the refreshed row");
        assert_eq!(
            (row.facts.max_running, row.facts.kv_block_size),
            (Some(6), Some(16)),
            "and both survive the refresh"
        );
        assert!(
            registry.get_by_url("http://w:8000@1@1").is_none(),
            "no second rank suffix"
        );
    }

    /// A refresh under a live key rebuilds from the live handle as a patch
    /// does: the endpoint and HTTP/2 registration resolved stay, the old id
    /// leaves, and the new handle signals its connection to the registry.
    #[test]
    fn a_same_key_upsert_keeps_what_registration_resolved_and_signals_connections() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let mut signals = registry
            .take_connect_signal_receiver()
            .expect("the registry's connect signal");
        let card = ModelCard {
            aliases: vec!["m-alias".to_string()],
            reasoning_parser: Some("plain".to_string()),
            ..ModelCard::new("m")
        };
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://u:8000")
                .model(card)
                .health_config(HealthCheckConfig {
                    drain_settle_secs: 30,
                    ..no_health_check()
                })
                .health_endpoint("/healthz")
                .http2(true)
                .build(),
        );
        registry.register(Arc::clone(&worker)).expect("registered");
        // The first worker is read through the store once, so its id enters
        // the key map through the miss path and the upsert has an entry to
        // remove.
        let first = worker_id(worker.as_ref());
        assert!(
            WorkersStore::get(&host, first).is_some(),
            "the first worker resolves"
        );
        assert!(
            host.keys.contains_key(&first),
            "and its key entry is on file"
        );
        let mut refresh = facts("http://u:8000");
        refresh.kv_block_size = Some(32);
        let MembershipChange::Replaced { old, new } = host.upsert(refresh) else {
            panic!("a live key refreshes");
        };
        assert_eq!(old, worker_id(worker.as_ref()));
        let successor = registry.get_by_url("http://u:8000").expect("the successor");
        assert_eq!(worker_id(successor.as_ref()), new);
        assert_eq!(
            successor.metadata().health_endpoint,
            "/healthz",
            "the endpoint registration resolved"
        );
        assert!(successor.http2(), "and HTTP/2");
        let kept = successor.models();
        assert_eq!(
            (
                kept.len(),
                kept[0].aliases.as_slice(),
                kept[0].reasoning_parser.as_deref()
            ),
            (1, ["m-alias".to_string()].as_slice(), Some("plain")),
            "the live card under the id, with its aliases and parser"
        );
        assert_eq!(
            successor.metadata().health_config.drain_settle_secs,
            30,
            "the health config outside the policy's five fields"
        );
        assert_eq!(
            WorkersStore::get(&host, new)
                .expect("the row")
                .facts
                .kv_block_size,
            Some(32)
        );
        assert!(
            !host.keys.contains_key(&old),
            "the old key entry left at the upsert, not at the next lazy resolve"
        );
        assert!(
            WorkersStore::get(&host, old).is_none(),
            "and the old id with it"
        );
        successor.signal_connected();
        assert!(
            signals.try_recv().is_ok(),
            "the handle carries the registry's connect signal"
        );
    }

    /// The window is a fact a patch cannot change: labels naming neither
    /// window key keep the handle's, labels naming one move it.
    #[test]
    fn a_labels_patch_naming_no_window_key_keeps_the_admission_window() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let mut window = facts("http://l:8000");
        window.max_running = Some(6);
        let MembershipChange::Registered { id } = host.upsert(window) else {
            panic!("a new key registers");
        };
        let revision = WorkersStore::get(&host, id).expect("the row").revision;
        let relabel = WorkerPatch {
            labels: Some(BTreeMap::from([("zone".to_string(), "a".to_string())])),
            ..WorkerPatch::default()
        };
        let Some(MembershipChange::Replaced { new, .. }) = host.patch(id, relabel, revision) else {
            panic!("a patch replaces the handle");
        };
        let row = WorkersStore::get(&host, new).expect("the relabelled row");
        assert_eq!(row.facts.labels.get("zone").map(String::as_str), Some("a"));
        assert_eq!(row.facts.max_running, Some(6), "the window stays");
        let resize = WorkerPatch {
            labels: Some(BTreeMap::from([(
                "max_running_requests".to_string(),
                "9".to_string(),
            )])),
            ..WorkerPatch::default()
        };
        let Some(MembershipChange::Replaced { new, .. }) = host.patch(new, resize, row.revision)
        else {
            panic!("a patch replaces the handle");
        };
        assert_eq!(
            WorkersStore::get(&host, new)
                .expect("the resized row")
                .facts
                .max_running,
            Some(9),
            "labels naming a window key move it"
        );
    }

    /// A patch rebuilds the handle as the property step does: what
    /// registration resolved stays.
    #[test]
    fn a_priority_patch_keeps_the_workers_resolved_configuration() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://p:8000")
                .model(ModelCard::new("m"))
                .health_config(no_health_check())
                .health_endpoint("/healthz")
                .label("max_running_requests", "4")
                .build(),
        );
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let revision = WorkersStore::get(&host, id).expect("the row").revision;
        let patch = WorkerPatch {
            priority: Some(7),
            ..WorkerPatch::default()
        };
        let Some(MembershipChange::Replaced { new, .. }) = host.patch(id, patch, revision) else {
            panic!("a patch replaces the handle");
        };
        let successor = registry.get_by_url("http://p:8000").expect("the successor");
        assert_eq!(worker_id(successor.as_ref()), new);
        assert_eq!(successor.priority(), 7);
        assert_eq!(
            successor.metadata().health_endpoint,
            "/healthz",
            "the endpoint registration resolved"
        );
        assert!(successor.metadata().health_config.disable_health_check);
        assert_eq!(successor.max_running_requests(), Some(4), "and the window");
    }

    /// The prober counts a pending worker's probes, so one that never
    /// answers fails at ten times the failure threshold.
    #[test]
    fn a_pending_worker_that_never_answers_fails_at_the_cap() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://n:8000")
                .model(ModelCard::new("m"))
                .health_config(HealthCheckConfig {
                    failure_threshold: 3,
                    success_threshold: 2,
                    ..HealthCheckConfig::default()
                })
                .build(),
        );
        registry.register(worker).expect("registered");
        let row = WorkersStore::by_key(&host, "http://n:8000").expect("the row");
        let failed = (0..40).find_map(|_| {
            host.probe_outcome(row.id, false, row.revision)
                .filter(|(_, to)| *to == StoreStatus::Failed)
        });
        assert_eq!(
            failed,
            Some((StoreStatus::Pending, StoreStatus::Failed)),
            "ten times the failure threshold of unanswered probes"
        );
    }

    /// A connect signal never takes a draining worker back into service.
    #[test]
    fn a_connect_signal_never_undrains_a_worker() {
        let registry = Arc::new(WorkerRegistry::new());
        let host = host(&registry);
        let worker = ready_worker("http://d:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let rid = registry.get_id_by_url("http://d:8000").expect("the id");
        let revision = worker.revision();
        assert!(registry
            .transition_status_if_revision(&rid, revision, WorkerStatus::Draining)
            .is_some());
        assert_eq!(host.connected(id, revision), None, "draining stands");
        assert_eq!(worker.status(), WorkerStatus::Draining);
    }

    /// A worker with no stream is published by the poll's own path, so the
    /// sink's report is readable back at once.
    #[test]
    fn a_poll_only_worker_is_published_through_the_loops_path() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime for the monitor");
        let _enter = runtime.enter();
        let registry = Arc::new(WorkerRegistry::new());
        let monitor = Arc::new(WorkerMonitor::new(
            Arc::clone(&registry),
            Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin)),
            1,
            false,
            false,
        ));
        let host = HostStores::new(Arc::clone(&registry), Some(&monitor), None);
        let worker = ready_worker("http://q:8000", "m");
        registry.register(Arc::clone(&worker)).expect("registered");
        let id = worker_id(worker.as_ref());
        let row = LoadRow {
            running: 3,
            ..LoadRow::default()
        };
        host.report(id, &[(0, row)], Instant::now(), LoadSource::Poll);
        assert_eq!(
            host.load(id, 0).expect("published").row.running,
            3,
            "a worker with no stream is the poll's own"
        );
        assert_eq!(host.ranks(id), 1);
    }
}
