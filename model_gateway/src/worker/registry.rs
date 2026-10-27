//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing.
//!
//! # Performance Optimizations
//! The model index uses immutable Arc snapshots instead of RwLock for lock-free reads.
//! This is critical for high-concurrency scenarios where many requests query the same model.
//!
//! # Consistent Hash Ring
//! The registry maintains a pre-computed [`HashRing`] per model for O(log n)
//! consistent hashing. The ring is rebuilt only when workers are added or
//! removed, not per request. See [`crate::worker::hash_ring`] for the ring
//! itself — this file only wires registry events to ring rebuilds.

#[cfg(test)]
use std::sync::mpsc as test_mpsc;
use std::{
    collections::{BTreeSet, HashSet},
    ops::Deref,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

use arc_swap::ArcSwap;
use dashmap::{mapref::entry::Entry, DashMap};
use openai_protocol::worker::WorkerStatus;
use tokio::sync::{broadcast, mpsc};
use uuid::Uuid;

use crate::{
    config::types::RetryConfig,
    observability::metrics::{metrics_labels, Metrics},
    worker::{
        circuit_breaker::CircuitState,
        event::{WorkerConnected, WorkerEvent},
        hash_ring::HashRing,
        pd_pair_index::{PdPairIndex, PdWire},
        pd_pairing::PdPairingMode,
        worker::{RuntimeType, WorkerType},
        ConnectionMode, Worker, DEFAULT_SAMPLING_PARAMS_LABEL, UNKNOWN_MODEL_ID,
    },
};

/// Unique identifier for a worker
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new() -> Self {
        Self(Uuid::now_v7().to_string())
    }

    /// Create a worker ID from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Where a worker's registration came from. `Local` workers are owned by
/// this node (their state is published to the mesh); `Mesh` workers were
/// imported from a peer's published state and must never be re-published.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerOrigin {
    Local,
    Mesh,
}

/// Side-effect-free worker snapshot for subscriber bootstrap or lag recovery.
#[derive(Debug, Clone)]
pub struct WorkerDescriptor {
    pub worker_id: WorkerId,
    pub status: WorkerStatus,
    pub disable_health_check: bool,
    pub check_interval_secs: u64,
}

/// Static routing pool inside a model snapshot.
///
/// Availability is deliberately not encoded here: health, circuit-breaker and
/// overload state change independently of membership and remain lock-free on
/// each worker. Policies validate that live state while selecting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoutingPool {
    HttpRegular,
    GrpcPipelineRegular,
    /// gRPC workers, and direct-ZMQ TokenSpeed engines, unlike
    /// [`Self::GrpcPipelineRegular`]'s every ZMQ worker: the other ZMQ wires
    /// carry no KV-transfer rendezvous, so such a worker must never join a PD
    /// pool even though it rides the gRPC pipeline elsewhere.
    GrpcPrefill,
    /// The members [`Self::GrpcPrefill`] admits.
    GrpcDecode,
    /// The members [`Self::GrpcPrefill`] admits: a TokenSpeed encode server
    /// takes its encode request on the ZMQ lane; the other runtimes' encode
    /// dispatch is a gRPC encoder RPC.
    GrpcEncode,
    /// HTTP-transport prefill pool for the HTTP PD router: it proxies plain
    /// HTTP to the selected worker's URL, so a gRPC or ZMQ worker must never
    /// be selectable (the same rule `select_first_worker` follows).
    HttpPrefill,
    /// HTTP-transport decode pool (same contract as [`Self::HttpPrefill`]).
    HttpDecode,
    /// Workers that front a third-party provider, whatever their type.
    External,
}

type WorkerSnapshot = Arc<[Arc<dyn Worker>]>;
type LazyRoutingPool = OnceLock<WorkerSnapshot>;

impl RoutingPool {
    const COUNT: usize = 8;

    const fn index(self) -> usize {
        match self {
            Self::HttpRegular => 0,
            Self::GrpcPipelineRegular => 1,
            Self::GrpcPrefill => 2,
            Self::GrpcDecode => 3,
            Self::GrpcEncode => 4,
            Self::HttpPrefill => 5,
            Self::HttpDecode => 6,
            Self::External => 7,
        }
    }

    fn matches(self, worker: &Arc<dyn Worker>) -> bool {
        match self {
            Self::HttpRegular => {
                *worker.worker_type() == WorkerType::Regular
                    && *worker.connection_mode() == ConnectionMode::Http
            }
            Self::GrpcPipelineRegular => {
                *worker.worker_type() == WorkerType::Regular
                    && worker.connection_mode().uses_grpc_pipeline()
            }
            Self::GrpcPrefill => {
                *worker.worker_type() == WorkerType::Prefill && serves_disaggregated_role(worker)
            }
            Self::GrpcDecode => {
                *worker.worker_type() == WorkerType::Decode && serves_disaggregated_role(worker)
            }
            Self::GrpcEncode => {
                *worker.worker_type() == WorkerType::Encode && serves_disaggregated_role(worker)
            }
            Self::HttpPrefill => {
                *worker.worker_type() == WorkerType::Prefill
                    && *worker.connection_mode() == ConnectionMode::Http
            }
            Self::HttpDecode => {
                *worker.worker_type() == WorkerType::Decode
                    && *worker.connection_mode() == ConnectionMode::Http
            }
            Self::External => worker.metadata().spec.runtime_type == RuntimeType::External,
        }
    }
}

/// Whether a worker of a disaggregated role rides the gRPC pipeline's
/// encode/prefill/decode dispatch: a gRPC worker, or a direct-ZMQ TokenSpeed
/// engine, whose wire carries the rendezvous and the encode request (the
/// other ZMQ wires carry neither, and registration refuses them a role).
fn serves_disaggregated_role(worker: &Arc<dyn Worker>) -> bool {
    match worker.connection_mode() {
        ConnectionMode::Grpc => true,
        ConnectionMode::Zmq => worker.metadata().spec.runtime_type == RuntimeType::TokenSpeed,
        ConnectionMode::Http => false,
    }
}

/// Immutable membership snapshot. Route-specific projections are populated
/// lazily once and then shared by every request until membership changes
/// publish a replacement snapshot.
///
/// Multi-role selection (PD/EPD) must derive every role from ONE snapshot: two
/// separate registry lookups can straddle a concurrent membership change and
/// pair workers that never coexisted.
#[derive(Debug)]
pub(crate) struct ModelWorkerSnapshot {
    all: WorkerSnapshot,
    pools: [LazyRoutingPool; RoutingPool::COUNT],
    /// The compatible prefill/decode pairs per wire and pairing mode, built
    /// once from the pools above (see [`PdPairIndex`]).
    pd_pairs: [[OnceLock<Arc<PdPairIndex>>; PdPairingMode::COUNT]; PdWire::COUNT],
}

impl ModelWorkerSnapshot {
    fn new(all: WorkerSnapshot) -> Self {
        Self {
            all,
            pools: std::array::from_fn(|_| OnceLock::new()),
            pd_pairs: std::array::from_fn(|_| std::array::from_fn(|_| OnceLock::new())),
        }
    }

    /// The compatible prefill/decode pairs of `wire` under `mode`: descriptor
    /// comparisons happen here, once per membership snapshot, never on the
    /// request path.
    pub(crate) fn pd_pairs(&self, wire: PdWire, mode: PdPairingMode) -> Arc<PdPairIndex> {
        self.pd_pairs[wire.index()][mode.index()]
            .get_or_init(|| {
                let (prefill, decode) = wire.pools();
                Arc::new(PdPairIndex::build(
                    self.pool(prefill),
                    self.pool(decode),
                    mode,
                ))
            })
            .clone()
    }

    pub(crate) fn pool(&self, pool: RoutingPool) -> WorkerSnapshot {
        self.pools[pool.index()]
            .get_or_init(|| {
                if self.all.iter().all(|worker| pool.matches(worker)) {
                    return Arc::clone(&self.all);
                }
                self.all
                    .iter()
                    .filter(|worker| pool.matches(worker))
                    .cloned()
                    .collect()
            })
            .clone()
    }
}

impl Deref for ModelWorkerSnapshot {
    type Target = [Arc<dyn Worker>];

    fn deref(&self) -> &Self::Target {
        &self.all
    }
}

/// The global membership snapshot together with the epoch it was built at.
///
/// Carrying the epoch inside the published object makes the read fast path
/// self-validating: load the object, compare its epoch to the live counter.
/// A bump installs a marker with `epoch: usize::MAX`, which can never match
/// and therefore can never be served.
#[derive(Debug)]
struct GlobalRoutingSnapshot {
    epoch: usize,
    snapshot: Arc<ModelWorkerSnapshot>,
}

/// Model index using immutable snapshots for lock-free reads.
/// Updates create new snapshots (copy-on-write semantics); route projections
/// are cached lazily inside each immutable snapshot.
type ModelIndex = Arc<DashMap<String, Arc<ModelWorkerSnapshot>>>;

/// Model alias to canonical model ID.
type ModelAliasIndex = Arc<DashMap<String, Arc<str>>>;

#[cfg(test)]
#[derive(Debug, Default)]
struct HashRingTestHooks {
    before_rebuild: Option<test_mpsc::Sender<()>>,
    after_snapshot: Option<(test_mpsc::Sender<()>, test_mpsc::Receiver<()>)>,
}

/// Pauses the next status transition after whatever its caller did outside
/// the worker's mutation lock and before the lock is taken: the transition
/// reports in, then waits to be let go. The interleaving proofs place
/// another writer's transition in that window.
#[cfg(test)]
#[derive(Debug, Default)]
struct TransitionTestHooks {
    before_lock: Option<(test_mpsc::Sender<()>, test_mpsc::Receiver<()>)>,
}

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Model index for O(1) lookups using immutable snapshots.
    /// Uses `Arc<[T]>` instead of `Arc<RwLock<Vec<T>>>` for lock-free reads.
    model_index: ModelIndex,

    /// Immutable snapshot spanning every registered worker, tagged with the
    /// `global_epoch` it was built at. Model-less routes use its lazy route
    /// projections instead of scanning the worker map and cloning every
    /// matching `Arc` on each request.
    ///
    /// Rebuilt lazily: membership mutations only bump `global_epoch` and
    /// install the shared empty marker (O(1) amortized), and the first read
    /// after a change rebuilds once from the worker map. A registration or
    /// removal storm therefore costs one rebuild total, not one full-fleet
    /// copy per worker.
    global_routing_snapshot: ArcSwap<GlobalRoutingSnapshot>,

    /// Monotonic membership generation; bumped on every register / replace /
    /// remove.
    global_epoch: AtomicUsize,

    /// Orders membership writes against rebuild scans. Mutators hold the
    /// (shared, O(1)) read side across the worker-map write and the epoch
    /// bump; the rebuild scan holds the write side, so a published snapshot
    /// is always a map state that truly existed — DashMap iteration alone
    /// locks shard by shard and can capture a mixture of generations.
    global_membership_order: parking_lot::RwLock<()>,

    /// Serializes the cold rebuild path so concurrent stale readers rebuild
    /// once. Mutations never take it.
    global_routing_update: parking_lot::Mutex<()>,

    /// Alias index kept separate from `model_index` so aliases do not appear as
    /// models in discovery or statistics.
    ///
    /// Lock order: code that holds an entry of this map may then take a
    /// `model_index` lock, never the reverse. Every alias write below reads
    /// `model_index` while holding its own entry, so taking them in the other
    /// order somewhere else would deadlock.
    model_alias_index: ModelAliasIndex,

    /// Consistent hash rings per model for O(log n) routing.
    /// Rebuilt on worker add/remove (copy-on-write).
    hash_rings: Arc<DashMap<String, Arc<HashRing>>>,

    /// Orders snapshots and publication for both per-model and wildcard rings.
    /// Membership writers release their index guards before taking this lock;
    /// routing readers never take it. Hold it through the wildcard rebuild so
    /// a late publication cannot restore a superseded membership snapshot.
    hash_ring_update: parking_lot::Mutex<()>,

    #[cfg(test)]
    hash_ring_test_hooks: parking_lot::Mutex<HashRingTestHooks>,

    #[cfg(test)]
    transition_test_hooks: parking_lot::Mutex<TransitionTestHooks>,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping
    url_to_id: Arc<DashMap<String, WorkerId>>,

    /// Per-worker-ID locks for serializing replace() operations.
    /// Only held during the in-memory model index diff (no I/O, microseconds).
    worker_mutation_locks: Arc<DashMap<WorkerId, Arc<parking_lot::Mutex<()>>>>,

    /// Count of workers currently flagged overloaded, per model-index key.
    /// Moved only on flag transitions and on registry membership changes, so
    /// reading it costs one map probe and no worker walk.
    model_overloaded: Arc<DashMap<String, AtomicUsize>>,

    /// `--worker-overload-shed`: refuse a request with a 503 when every
    /// candidate is overloaded instead of steering it to the least-loaded
    /// one. Read by every router at its overload decision points.
    overload_shed: Arc<AtomicBool>,

    /// Serializes overload *edges* so a flag flip and its counter adjustment
    /// land as one step. Two group loops flipping the same worker in opposite
    /// directions would otherwise be free to apply their deltas in the reverse
    /// order and leave the counter disagreeing with the flag. Never taken when
    /// the verdict is unchanged, which is every poll but the crossing ones, and
    /// never taken on a request path.
    overload_transitions: Arc<parking_lot::Mutex<()>>,

    /// Per-model retry config (last write wins).
    /// Updated when a worker with non-empty retry overrides registers.
    /// Cleaned up when the last worker for a model is removed.
    /// When retries are disabled, max_retries is set to 1.
    model_retry_configs: Arc<DashMap<String, RetryConfig>>,

    /// Registration origin per worker (local vs mesh-imported). Written
    /// under the per-worker mutation lock before the `Registered` event,
    /// removed in `remove()` teardown.
    worker_origins: Arc<DashMap<WorkerId, WorkerOrigin>>,

    /// Broadcast channel for worker state change events.
    event_tx: broadcast::Sender<WorkerEvent>,

    /// Sender handed to workers so a completed backend connection can wake
    /// the manager for immediate promotion (see [`WorkerConnected`]). Lives
    /// on the registry because workers are built before the manager starts,
    /// and dynamically-registered workers need it long after.
    connect_signal_tx: mpsc::UnboundedSender<WorkerConnected>,

    /// Receiver drained by the manager's health loop. Taken once via
    /// [`Self::take_connect_signal_receiver`]; `None` thereafter (and when
    /// no manager runs, e.g. health checks globally disabled).
    connect_signal_rx: parking_lot::Mutex<Option<mpsc::UnboundedReceiver<WorkerConnected>>>,
}

impl WorkerRegistry {
    // ───────────────────────────────────────────────────────────────────
    // 1. Construction & subscription
    // ───────────────────────────────────────────────────────────────────

    /// Create an empty worker registry.
    ///
    /// Initialises all indexes and a broadcast channel with capacity 1024
    /// for `WorkerEvent` delivery. Holds no locks. Emits no events.
    pub fn new() -> Self {
        // Unbounded so a worker's detached handshake task never blocks on a
        // busy manager; signals are one-per-connect and rare, so the queue
        // cannot grow without bound in practice.
        let (connect_signal_tx, connect_signal_rx) = mpsc::unbounded_channel();
        Self {
            workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::new()),
            global_routing_snapshot: ArcSwap::from_pointee(GlobalRoutingSnapshot {
                epoch: 0,
                snapshot: Self::empty_routing_snapshot(),
            }),
            global_epoch: AtomicUsize::new(0),
            global_membership_order: parking_lot::RwLock::new(()),
            global_routing_update: parking_lot::Mutex::new(()),
            model_alias_index: Arc::new(DashMap::new()),
            hash_rings: Arc::new(DashMap::new()),
            hash_ring_update: parking_lot::Mutex::new(()),
            #[cfg(test)]
            hash_ring_test_hooks: parking_lot::Mutex::new(HashRingTestHooks::default()),
            #[cfg(test)]
            transition_test_hooks: parking_lot::Mutex::new(TransitionTestHooks::default()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
            worker_mutation_locks: Arc::new(DashMap::new()),
            model_overloaded: Arc::new(DashMap::new()),
            overload_shed: Arc::new(AtomicBool::new(false)),
            overload_transitions: Arc::new(parking_lot::Mutex::new(())),
            model_retry_configs: Arc::new(DashMap::new()),
            worker_origins: Arc::new(DashMap::new()),
            // Sized for fleet-scale bursts (startup registration, probe
            // storms): a lagged subscriber forces a full state resync, so
            // the capacity should comfortably exceed realistic worker
            // counts. ~100 B per slot; fixed cost ~100 KB.
            event_tx: broadcast::Sender::new(1024),
            connect_signal_tx,
            connect_signal_rx: parking_lot::Mutex::new(Some(connect_signal_rx)),
        }
    }

    /// Registration origin for a worker, if it is currently registered.
    /// `Local` workers are owned by this node; `Mesh` workers were
    /// imported from a peer's published state.
    pub fn origin_of(&self, worker_id: &WorkerId) -> Option<WorkerOrigin> {
        self.worker_origins.get(worker_id).map(|entry| *entry)
    }

    /// Subscribe to the `WorkerEvent` broadcast stream.
    ///
    /// Returns a `broadcast::Receiver` that observes every future mutation
    /// event emitted by `register` / `replace` / `remove` / `transition_status`.
    /// Late subscribers miss past events — callers that need historical
    /// state should combine this with [`Self::reconcile_snapshot`] on startup
    /// and on `RecvError::Lagged`. Holds no locks. Emits no events.
    pub fn subscribe_events(&self) -> broadcast::Receiver<WorkerEvent> {
        self.event_tx.subscribe()
    }

    /// Clone the connect-signal sender for a worker to fire once its backend
    /// connection completes. See [`WorkerConnected`]. Holds no locks.
    pub fn connect_signal_sender(&self) -> mpsc::UnboundedSender<WorkerConnected> {
        self.connect_signal_tx.clone()
    }

    /// Take the connect-signal receiver. Returns `Some` exactly once (the
    /// manager takes it at startup); `None` on any later call. Holds the
    /// receiver lock briefly.
    pub fn take_connect_signal_receiver(&self) -> Option<mpsc::UnboundedReceiver<WorkerConnected>> {
        self.connect_signal_rx.lock().take()
    }

    /// True if any registered worker is Pending and depends on the connect
    /// signal (rather than health polling) to reach Ready. Such a worker is
    /// promoted only by the manager consuming [`WorkerConnected`], so the
    /// manager must run even when health checks are otherwise disabled.
    pub fn has_workers_awaiting_connect_signal(&self) -> bool {
        self.get_by_connection(ConnectionMode::Zmq)
            .iter()
            .any(|w| w.status() == WorkerStatus::Pending)
    }

    // ───────────────────────────────────────────────────────────────────
    // 2. Read — single worker
    // ───────────────────────────────────────────────────────────────────

    /// Look up a worker by its ID.
    ///
    /// Returns `Some(worker)` when the ID exists, `None` otherwise.
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Look up a worker by its URL.
    ///
    /// Returns `Some(worker)` when a worker with this URL is registered,
    /// `None` otherwise.
    ///
    /// A scheme-less `host:port` input is canonicalized against the registered
    /// `http://`/`grpc://` form — see `Self::resolve_url_to_id`.
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        self.resolve_url_to_id(url).and_then(|id| self.get(&id))
    }

    /// Look up a worker's ID by its URL.
    ///
    /// Returns `Some(id)` when a worker with this URL is registered,
    /// `None` otherwise. Useful for callers that need to invoke
    /// `transition_status_if_revision` with the current worker revision.
    ///
    /// A scheme-less `host:port` input is canonicalized against the registered
    /// `http://`/`grpc://` form — see `Self::resolve_url_to_id`.
    pub fn get_id_by_url(&self, url: &str) -> Option<WorkerId> {
        self.resolve_url_to_id(url)
    }

    /// Resolve a URL string to the `WorkerId` of a live worker.
    ///
    /// Tries exact match first. If the input has no recognized scheme
    /// (`http://`, `https://`, `grpc://`, `grpcs://`), retries with
    /// `http://` and `grpc://` prefixes — service discovery synthesizes
    /// bare `host:port` URLs and the workflow normalizes them to either
    /// scheme based on probe results, so removal-by-url needs to match
    /// either canonical form.
    ///
    /// Skips ids that are present in `url_to_id` but missing from `workers`.
    /// [`Self::reserve_id_for_url`] writes the URL→ID mapping before the
    /// worker is built, so a bare URL submitted to `WorkerService::create_worker`
    /// can shadow the canonical `http://`/`grpc://` entry that the
    /// AddWorker workflow later registers under.
    fn resolve_url_to_id(&self, url: &str) -> Option<WorkerId> {
        let is_live = |id: &WorkerId| self.workers.contains_key(id);

        if let Some(id) = self.url_to_id.get(url) {
            let id = id.clone();
            if is_live(&id) {
                return Some(id);
            }
        }
        let has_scheme = url.starts_with("http://")
            || url.starts_with("https://")
            || url.starts_with("grpc://")
            || url.starts_with("grpcs://");
        if has_scheme {
            return None;
        }
        for scheme in ["http://", "grpc://"] {
            let candidate = format!("{scheme}{url}");
            if let Some(id) = self.url_to_id.get(&candidate) {
                let id = id.clone();
                if is_live(&id) {
                    return Some(id);
                }
            }
        }
        None
    }

    /// Reverse-lookup the URL for a given worker ID.
    ///
    /// Prefers the URL stored on the live worker object; falls back to
    /// scanning `url_to_id` so pre-reserved IDs (from
    /// [`Self::reserve_id_for_url`]) can still be resolved before a worker
    /// is installed.
    pub fn get_url_by_id(&self, worker_id: &WorkerId) -> Option<String> {
        if let Some(worker) = self.get(worker_id) {
            return Some(worker.url().to_string());
        }
        self.url_to_id
            .iter()
            .find_map(|entry| (entry.value() == worker_id).then(|| entry.key().clone()))
    }

    /// Get the consistent hash ring for a model (O(1) lookup).
    ///
    /// Returns `Some(ring)` if any workers are registered for this model,
    /// `None` otherwise. The ring is pre-built and updated on worker add
    /// or remove, so reads are allocation-free apart from the Arc clone.
    ///
    /// Keyed by canonical model ID only, like every other per-model map on
    /// this registry except [`Self::get_by_model`]. A caller holding a
    /// client-supplied name resolves it with [`Self::resolve_model_alias`]
    /// once at request entry and passes the canonical ID from there on.
    ///
    /// [`UNKNOWN_MODEL_ID`] returns the wildcard ring spanning every worker,
    /// matching the candidate set a request that names no model is routed
    /// against.
    pub fn get_hash_ring(&self, model_id: &str) -> Option<Arc<HashRing>> {
        self.hash_rings.get(model_id).map(|r| Arc::clone(&r))
    }

    // ───────────────────────────────────────────────────────────────────
    // 3. Read — collections
    // ───────────────────────────────────────────────────────────────────

    /// Empty worker slice constant returned when a lookup has no matches.
    const EMPTY_WORKERS: &'static [Arc<dyn Worker>] = &[];

    /// Return all workers serving a canonical model or alias.
    ///
    /// This is the fastest possible read path: the model index already
    /// stores the slice as an `Arc<[_]>`, so the return value is just an
    /// atomic refcount bump with zero contention. Returns an empty shared
    /// slice when the model is unknown.
    pub fn get_by_model(&self, model_id: &str) -> Arc<[Arc<dyn Worker>]> {
        if let Some(workers) = self.model_index.get(model_id) {
            return Arc::clone(&workers.all);
        }
        self.model_alias_index
            .get(model_id)
            .and_then(|canonical_id| {
                self.model_index
                    .get(canonical_id.as_ref())
                    .map(|workers| Arc::clone(&workers.all))
            })
            .unwrap_or_else(|| Arc::from(Self::EMPTY_WORKERS))
    }

    /// Return a shared route-specific candidate snapshot for a model.
    ///
    /// The first lookup after a membership change builds the projection once;
    /// subsequent requests only clone the outer `Arc`. Dynamic availability is
    /// intentionally checked by the policy so health, circuit-breaker and
    /// overload changes never require rebuilding this snapshot. The wildcard
    /// model ID uses the global registry snapshot.
    pub(crate) fn get_routing_pool(
        &self,
        model_id: &str,
        pool: RoutingPool,
    ) -> Arc<[Arc<dyn Worker>]> {
        if model_id == UNKNOWN_MODEL_ID {
            return self.current_global_routing_snapshot().pool(pool);
        }
        self.get_model_routing_pool(model_id, pool)
    }

    /// Like [`Self::get_routing_pool`] but never widens to the global
    /// snapshot: the wildcard id resolves through the literal `"unknown"`
    /// model-index entry (untagged workers index there), yielding an empty
    /// slice when no such entry exists. The HTTP PD router uses this to keep
    /// its conditional wildcard fallback — untagged workers win when present,
    /// and only an empty entry widens to the global pool.
    pub(crate) fn get_model_routing_pool(
        &self,
        model_id: &str,
        pool: RoutingPool,
    ) -> Arc<[Arc<dyn Worker>]> {
        self.model_routing_snapshot(model_id).map_or_else(
            || Arc::from(Self::EMPTY_WORKERS),
            |snapshot| snapshot.pool(pool),
        )
    }

    /// One membership snapshot for a model (alias-aware), from which several
    /// role pools can be derived atomically: a PD/EPD selection must draw all
    /// its roles from a single generation, and two separate pool lookups could
    /// straddle a concurrent membership change.
    pub(crate) fn model_routing_snapshot(
        &self,
        model_id: &str,
    ) -> Option<Arc<ModelWorkerSnapshot>> {
        // Bind the clone via a `let` so the shard `Ref` drops at the end of
        // the statement (edition 2021 keeps `if let` scrutinee temporaries
        // alive through the body): the caller's first post-change `pool()`
        // call does an O(workers) projection build that must not run under
        // the shard read guard.
        let snapshot = self
            .model_index
            .get(model_id)
            .map(|workers| Arc::clone(workers.value()));
        if snapshot.is_some() {
            return snapshot;
        }
        let canonical_id = self
            .model_alias_index
            .get(model_id)
            .map(|canonical_id| Arc::clone(canonical_id.value()));
        canonical_id.and_then(|canonical_id| {
            self.model_index
                .get(canonical_id.as_ref())
                .map(|workers| Arc::clone(workers.value()))
        })
    }

    /// One membership snapshot for multi-role selection, with the wildcard
    /// model mapping to the global snapshot and an unknown model to a shared
    /// empty one.
    pub(crate) fn get_routing_snapshot(&self, model_id: &str) -> Arc<ModelWorkerSnapshot> {
        if model_id == UNKNOWN_MODEL_ID {
            return self.current_global_routing_snapshot();
        }
        self.model_routing_snapshot(model_id)
            .unwrap_or_else(Self::empty_routing_snapshot)
    }

    /// Shared empty snapshot for unknown models.
    fn empty_routing_snapshot() -> Arc<ModelWorkerSnapshot> {
        static EMPTY: OnceLock<Arc<ModelWorkerSnapshot>> = OnceLock::new();
        Arc::clone(
            EMPTY
                .get_or_init(|| Arc::new(ModelWorkerSnapshot::new(Arc::from(Self::EMPTY_WORKERS)))),
        )
    }

    /// Return the immutable global membership snapshot used by routing paths
    /// whose model/provider filters are evaluated dynamically.
    pub(crate) fn get_routing_workers(&self) -> Arc<[Arc<dyn Worker>]> {
        Arc::clone(&self.current_global_routing_snapshot().all)
    }

    /// Whether an all-overloaded candidate pool is refused (`true`) or
    /// steered to its least-loaded worker (the default).
    pub fn overload_shed_enabled(&self) -> bool {
        self.overload_shed.load(Ordering::Relaxed)
    }

    /// Set `--worker-overload-shed` for every router reading this registry.
    pub fn set_overload_shed(&self, shed: bool) {
        self.overload_shed.store(shed, Ordering::Relaxed);
    }

    /// Apply the absolute overload veto to `worker`, returning `true` when the
    /// flag transitioned. The only sanctioned writer of
    /// [`Worker::set_overloaded`]: counters and gauge move once per edge.
    ///
    /// Writes through a handle the registry no longer holds are dropped — the
    /// monitor writes its verdict after a multi-second fetch await, by which
    /// time the worker may have been removed or replaced, and a stale write
    /// would move a counter no reset path can reach again.
    pub fn set_worker_overloaded(&self, worker: &Arc<dyn Worker>, overloaded: bool) -> bool {
        // Lock-free fast path for the common case: the poll re-asserted a
        // verdict the worker already carries, so there is no edge to record.
        if worker.is_overloaded() == overloaded {
            return false;
        }
        // Allocate the keys before taking the edge lock; only the flag flip,
        // the counter deltas and the gauge publish go inside it. The publish
        // must share the critical section: published after release, two edges
        // on one model can land their gauge writes in the reverse order and
        // pin a stale value until the next transition. Counter GC re-locks a
        // `model_overloaded` shard, so it stays outside.
        let model_ids = Self::worker_model_ids(worker);
        let mut updates = Vec::with_capacity(model_ids.len());
        {
            let _edge = self.overload_transitions.lock();
            if !self.is_current_handle(worker) {
                return false;
            }
            if !worker.set_overloaded(overloaded) {
                return false;
            }
            for model_id in model_ids {
                let updated = self.adjust_model_overloaded(&model_id, overloaded);
                Metrics::set_workers_overloaded(&model_id, updated);
                updates.push((model_id, updated));
            }
        }
        for (model_id, updated) in updates {
            if updated == 0 {
                self.model_overloaded
                    .remove_if(&model_id, |_, count| count.load(Ordering::Acquire) == 0);
            }
        }
        true
    }

    /// Whether `worker` is the `Arc` the registry currently holds for its URL.
    ///
    /// The identity check is by pointer, not by URL: a `replace()` keeps the URL
    /// and the shared runtime but installs a new worker object with a possibly
    /// different model card, and the replaced handle must not be allowed to
    /// attribute a counter move to the model set it used to carry.
    fn is_current_handle(&self, worker: &Arc<dyn Worker>) -> bool {
        let Some(worker_id) = self.url_to_id.get(worker.url()).map(|id| id.clone()) else {
            return false;
        };
        self.workers
            .get(&worker_id)
            .is_some_and(|current| Arc::ptr_eq(current.value(), worker))
    }

    /// Clear the veto on every registered worker, returning how many were
    /// flagged. Used by the load monitor's reset paths, where the feed that
    /// would otherwise clear the flags is itself being torn down.
    pub fn clear_all_overload_flags(&self) -> usize {
        let workers: Vec<Arc<dyn Worker>> = self
            .workers
            .iter()
            .map(|entry| Arc::clone(entry.value()))
            .collect();
        workers
            .iter()
            .filter(|worker| self.set_worker_overloaded(worker, false))
            .count()
    }

    /// Workers of `model_id` currently flagged overloaded. O(1).
    pub fn overloaded_worker_count(&self, model_id: &str) -> usize {
        if let Some(count) = self.model_overloaded.get(model_id) {
            return count.load(Ordering::Acquire);
        }
        self.model_alias_index
            .get(model_id)
            .and_then(|canonical_id| {
                self.model_overloaded
                    .get(canonical_id.as_ref())
                    .map(|count| count.load(Ordering::Acquire))
            })
            .unwrap_or(0)
    }

    /// Move one worker in or out of `model_id`'s overloaded count, returning
    /// the updated count. Feeds the gauge only — the shed verdict walks the
    /// caller's candidate pool, since a per-model counter cannot describe a
    /// type/transport-narrowed sub-pool.
    fn adjust_model_overloaded(&self, model_id: &str, overloaded: bool) -> usize {
        let entry = self
            .model_overloaded
            .entry(model_id.to_string())
            .or_default();
        if overloaded {
            entry.fetch_add(1, Ordering::AcqRel) + 1
        } else {
            // Saturating: a double-clear must never wrap the counter into a
            // permanent all-overloaded verdict for the model.
            entry
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    Some(current.saturating_sub(1))
                })
                .unwrap_or(0)
                .saturating_sub(1)
        }
    }

    /// Resolve an alias to its canonical model ID without copying the string.
    ///
    /// Returns `None` for canonical model IDs, the `unknown` wildcard, and
    /// unknown names. Callers can use the original input when this returns
    /// `None`.
    ///
    /// # Rewriting an outbound request with the result
    ///
    /// Safe only where the alias is a second name for the same model, which is
    /// what a self-hosted worker declares. It is NOT safe for external
    /// providers: `workflow::steps::external::discover_models` groups a
    /// provider's date-stamped variants under the shortest name, so `gpt-4o`
    /// becomes canonical and `gpt-4o-2024-08-06` becomes its alias. Those name
    /// different models — one floats to the provider's current release, the
    /// other is pinned — and substituting one for the other would silently run
    /// a model the client did not ask for.
    ///
    /// External workers reach only the provider routers
    /// ([`Gateway::select_router_for_model`] gives them priority, and
    /// single-router mode picks by routing mode), none of which rewrite the
    /// outbound model. Keep it that way: canonicalize registry lookups there
    /// if needed, never the request body.
    ///
    /// [`Gateway::select_router_for_model`]: crate::routers::gateway::Gateway
    pub fn resolve_model_alias(&self, model_id: &str) -> Option<Arc<str>> {
        if model_id == UNKNOWN_MODEL_ID || self.model_index.contains_key(model_id) {
            return None;
        }

        let canonical_id = self.model_alias_index.get(model_id)?;
        self.model_index
            .contains_key(canonical_id.as_ref())
            .then(|| Arc::clone(&canonical_id))
    }

    /// Return all workers of a given type as an immutable shared slice.
    ///
    /// Unified with [`Self::get_by_model`] on `Arc<[_]>` so callers can
    /// treat all worker collections uniformly. Builds a fresh slice per
    /// call (one boxed-slice allocation).
    pub fn get_by_type(&self, worker_type: WorkerType) -> Arc<[Arc<dyn Worker>]> {
        let workers: Vec<Arc<dyn Worker>> = self
            .type_workers
            .get(&worker_type)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default();
        Arc::from(workers.into_boxed_slice())
    }

    /// Return all workers using a given connection mode (HTTP or gRPC).
    ///
    /// Returned as an immutable shared slice for uniformity with the other
    /// collection getters. Builds a fresh slice per call.
    pub fn get_by_connection(&self, connection_mode: ConnectionMode) -> Arc<[Arc<dyn Worker>]> {
        let workers: Vec<Arc<dyn Worker>> = self
            .connection_workers
            .get(&connection_mode)
            .map(|ids| ids.iter().filter_map(|id| self.get(id)).collect())
            .unwrap_or_default();
        Arc::from(workers.into_boxed_slice())
    }

    /// Return every prefill worker, regardless of which model they serve.
    ///
    /// Thin wrapper over [`Self::get_by_type`] with `WorkerType::Prefill`.
    pub fn get_prefill_workers(&self) -> Arc<[Arc<dyn Worker>]> {
        self.get_by_type(WorkerType::Prefill)
    }

    /// Return every decode worker, regardless of which model they serve.
    ///
    /// Thin wrapper over [`Self::get_by_type`] with `WorkerType::Decode`.
    pub fn get_decode_workers(&self) -> Arc<[Arc<dyn Worker>]> {
        self.get_by_type(WorkerType::Decode)
    }

    /// Return workers matching every supplied filter.
    ///
    /// Filters:
    /// - `model_id`: scope to a single model (uses the O(1) model index)
    /// - `worker_type`: `Regular` / `Prefill` / `Decode`
    /// - `connection_mode`: `Http` / `Grpc`
    /// - `runtime_type`: `Sglang` / `Vllm` / `External` / …
    /// - `healthy_only`: skip workers whose `is_healthy()` is false
    ///
    /// Only workers that pass every filter are cloned; the source collection
    /// (the per-model index slice, or the worker map when `model_id` is `None`)
    /// is iterated by reference rather than cloned wholesale first. Returns an
    /// owned `Vec` because each call applies a unique filter combination.
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        runtime_type: Option<RuntimeType>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        let matches = |w: &Arc<dyn Worker>| {
            if let Some(ref wtype) = worker_type {
                if *w.worker_type() != *wtype {
                    return false;
                }
            }
            if let Some(ref conn) = connection_mode {
                if w.connection_mode() != conn {
                    return false;
                }
            }
            if let Some(ref rt) = runtime_type {
                if w.metadata().spec.runtime_type != *rt {
                    return false;
                }
            }
            !healthy_only || w.is_healthy()
        };

        // Clone only the workers that pass: scope to the O(1) model index when a
        // model is given, otherwise iterate the worker map directly. Avoids the
        // per-request O(total workers) clone that fetching the full set first
        // would incur.
        if let Some(model) = model_id {
            self.get_by_model(model)
                .iter()
                .filter(|w| matches(w))
                .cloned()
                .collect()
        } else {
            self.workers
                .iter()
                .filter(|entry| matches(entry.value()))
                .map(|entry| entry.value().clone())
                .collect()
        }
    }

    /// Return an owned snapshot of every registered worker.
    ///
    /// Allocates a fresh `Vec` by cloning each Arc. Intended for cold
    /// paths (admin endpoints, diagnostics). Hot routing paths should
    /// prefer [`Self::get_by_model`].
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Return every worker paired with its ID.
    ///
    /// Used by bootstrap/reconcile paths that need to correlate workers
    /// with their IDs.
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Return every worker's URL as a freshly allocated `Vec`.
    ///
    /// Used by admin endpoints and tests.
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    /// Return every worker's URL paired with its optional API key.
    ///
    /// Used by the gateway when proxying to upstream workers that require
    /// per-worker credentials.
    pub fn get_all_urls_with_api_key(&self) -> Vec<(String, Option<String>)> {
        self.workers
            .iter()
            .map(|entry| {
                (
                    entry.value().url().to_string(),
                    entry.value().api_key().cloned(),
                )
            })
            .collect()
    }

    /// Return a side-effect-free descriptor snapshot for reconcile paths.
    ///
    /// Each `WorkerDescriptor` captures the fields a subscriber needs to
    /// rebuild its in-memory state from scratch (e.g. health scheduling
    /// after `RecvError::Lagged`) without re-reading the worker objects.
    pub fn reconcile_snapshot(&self) -> Vec<WorkerDescriptor> {
        self.workers
            .iter()
            .map(|entry| {
                let worker = entry.value();
                WorkerDescriptor {
                    worker_id: entry.key().clone(),
                    status: worker.status(),
                    disable_health_check: worker.metadata().health_config.disable_health_check,
                    check_interval_secs: worker.metadata().health_config.check_interval_secs,
                }
            })
            .collect()
    }

    /// Return the set of model IDs that currently have at least one
    /// worker serving them.
    ///
    /// Skips model entries whose worker slice has become empty (those are
    /// eventually evicted by the removal path).
    pub fn get_models(&self) -> Vec<String> {
        self.model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Distinct canonical models with at least one worker serving them.
    /// Wildcard workers carry no model and contribute nothing.
    pub fn model_count(&self) -> usize {
        self.model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .count()
    }

    /// Whether at least one worker serves this name, as a canonical model ID
    /// or as an alias.
    ///
    /// [`Self::get_models`] lists canonical IDs only, so a membership test
    /// written against it rejects every alias. Endpoints that gate on "is
    /// this model servable" use this instead. The `unknown` wildcard is not a
    /// registered name and stays rejected.
    pub fn contains_model(&self, model_id: &str) -> bool {
        if self.model_has_workers(model_id) {
            return true;
        }
        self.model_alias_index
            .get(model_id)
            .is_some_and(|canonical_id| self.model_has_workers(canonical_id.as_ref()))
    }

    fn model_has_workers(&self, canonical_id: &str) -> bool {
        self.model_index
            .get(canonical_id)
            .is_some_and(|workers| !workers.is_empty())
    }

    /// The value of the `model` label on the request metric families for a
    /// request that names `model_id`: the id itself while a worker serves it
    /// (an alias included), else [`metrics_labels::MODEL_OTHER`], so a client
    /// cannot mint label values with names the gateway does not serve.
    /// [`UNKNOWN_MODEL_ID`], the gateway's own wildcard for a request that
    /// names no model and is routed to every worker, keeps its own value: it
    /// is one fixed label, and the recorders of that traffic that run after
    /// routing write it as it is. Routing keeps the request's own id; only
    /// the label is bounded.
    pub fn metric_model_label<'a>(&self, model_id: &'a str) -> &'a str {
        if model_id == UNKNOWN_MODEL_ID || self.contains_model(model_id) {
            model_id
        } else {
            metrics_labels::MODEL_OTHER
        }
    }

    /// Return the number of registered workers.
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    /// Return `true` when no workers are registered.
    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// Return a consolidated snapshot of registry statistics.
    ///
    /// Iterates the `workers` map once, counting totals per worker type,
    /// connection mode, circuit-breaker state, and health status. Used by
    /// `/v1/stats` and monitoring dashboards.
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        // Count models directly instead of allocating Vec via get_models() (lock-free)
        let total_models = self
            .model_index
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .count();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut encode_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;
        let mut http_count = 0;
        let mut grpc_count = 0;
        let mut zmq_count = 0;
        let mut cb_open_count = 0;
        let mut cb_half_open_count = 0;

        // Iterate DashMap directly to avoid cloning all workers via get_all()
        for entry in self.workers.iter() {
            let worker = entry.value();
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Encode => encode_count += 1,
                WorkerType::Prefill => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }

            match worker.connection_mode() {
                ConnectionMode::Http => http_count += 1,
                ConnectionMode::Grpc => grpc_count += 1,
                ConnectionMode::Zmq => zmq_count += 1,
            }

            match worker.circuit_breaker_state() {
                CircuitState::Open => cb_open_count += 1,
                CircuitState::HalfOpen => cb_half_open_count += 1,
                CircuitState::Closed => {}
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            unhealthy_workers: total_workers.saturating_sub(healthy_count),
            total_load,
            regular_workers: regular_count,
            encode_workers: encode_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
            http_workers: http_count,
            grpc_workers: grpc_count,
            zmq_workers: zmq_count,
            circuit_breaker_open: cb_open_count,
            circuit_breaker_half_open: cb_half_open_count,
        }
    }

    /// Return `(regular_count, pd_count)` using the type index directly.
    ///
    /// Avoids allocating the full worker list the way [`Self::stats`] does.
    /// `pd_count` is any worker that is not `Regular`.
    pub fn get_worker_distribution(&self) -> (usize, usize) {
        // Use the existing type_workers index for O(1) lookup
        let regular_count = self
            .type_workers
            .get(&WorkerType::Regular)
            .map(|v| v.len())
            .unwrap_or(0);

        let total_workers = self.workers.len();
        let pd_count = total_workers.saturating_sub(regular_count);

        (regular_count, pd_count)
    }

    // ───────────────────────────────────────────────────────────────────
    // 4. Read — config
    // ───────────────────────────────────────────────────────────────────

    /// Get the per-model retry config override, if any.
    ///
    /// Returns `None` when no worker in this model group set a retry
    /// override. When retries are disabled for the group, the stored
    /// `max_retries` is always 1.
    pub fn get_retry_config(&self, model_id: &str) -> Option<RetryConfig> {
        self.model_retry_configs
            .get(model_id)
            .map(|entry| entry.value().clone())
    }

    /// True when any per-model retry override still allows retries; the
    /// content-blind streamed path cannot know the model, so it must assume
    /// the strictest override.
    pub fn any_model_retry_override_enables_retries(&self) -> bool {
        self.model_retry_configs
            .iter()
            .any(|entry| entry.value().max_retries > 1)
    }

    // ───────────────────────────────────────────────────────────────────
    // 5. Write — mutation primitives
    //
    // Every method in this section holds the per-worker mutation lock
    // (`worker_mutation_locks`) and emits exactly one `WorkerEvent` before
    // releasing the lock. New mutation methods MUST follow this pattern.
    // Manual publish at each call site is intentional — there are only a
    // handful of mutation methods, and the simplicity beats a generic
    // helper layer.
    //
    // Status. A registered worker's status is written under the worker's
    // mutation lock by two primitives. `transition_under_lock` is the one
    // transition: it looks the worker up, checks the revision when the
    // caller names one, writes a status that differs from the current one,
    // runs `on_status_changed` and publishes `StatusChanged`, so no two
    // writers interleave and no subscriber sees one worker's events out of
    // order. `demote_before_removal` is the one other write, the removal's
    // silent NotReady over Ready, a step of `remove_inner` under the same
    // lock; it publishes nothing. The registry has no opinion on which
    // edges are legal: the callers own the rules (the registry enforces
    // none). A rule that depends on the worker's current status is decided
    // in the primitive's closure, under the lock, so the status it reads is
    // the one the write lands on; decided outside, another writer's status
    // landing between the read and the write was written over (a probe's
    // `Failed` by the mesh hint's `Ready`, a probe's `NotReady` by the drain
    // step's `Draining`). A writer whose target does not depend on the
    // current status names it outright. This table is the whole set of
    // them:
    //
    //   writer                     edge(s)                          through
    //   health probes              Pending/NotReady/Failed -> Ready `apply_if_revision`
    //   (WorkerManager)            Pending/NotReady -> Failed       (decided under the lock)
    //                              Ready -> NotReady
    //   connect signal             any -> Ready                     `transition_status_if_revision`
    //   (WorkerManager)            (a ZMQ handshake, or a worker
    //                              without health checks; a probed
    //                              worker's signal moves its probe)
    //   mesh health hint           Pending/NotReady -> Ready        `transition_under_lock`
    //   (`on_remote_worker_state`) Ready -> NotReady                (decided under the lock)
    //   drain step                 Ready -> Draining                `apply_if_revision`
    //   (removal workflow)                                          (decided under the lock)
    //   activation step            not Ready -> Ready               `transition_status`
    //   (registration workflow)
    //   property update that       any -> Ready                     `transition_status`
    //   turns health checks off
    //   removal (`remove_inner`)   Ready -> NotReady, no event      `demote_before_removal`
    //
    // Two status writes are not transitions and stay outside: a mesh
    // import's status is set at its construction, before registration, and
    // the activation step writes directly to a worker the registry no
    // longer holds. The revision-checked writers discard an outcome that
    // lost a race with a same-URL `replace()`.
    // ───────────────────────────────────────────────────────────────────

    /// Register a new worker (create-only).
    ///
    /// Returns the new `WorkerId` on success, or `None` if a worker with
    /// the same URL is already registered and active. A URL that was
    /// pre-reserved via [`Self::reserve_id_for_url`] but has no worker yet
    /// is treated as a new registration (reuses the reserved ID).
    ///
    /// Emits [`WorkerEvent::Registered`] on success. Holds the per-worker
    /// mutation lock for the entire `register_inner` call — the index
    /// updates, origin record, and event broadcast all run under the same
    /// lock so subscribers cannot observe `Removed` / `Replaced` /
    /// `StatusChanged` events before the `Registered` event for a
    /// concurrent same-ID operation.
    pub fn register(&self, worker: Arc<dyn Worker>) -> Option<WorkerId> {
        self.register_inner(worker, WorkerOrigin::Local)
    }

    /// Register or replace a worker (upsert).
    ///
    /// Returns the resulting `WorkerId`. Used by internal callers (K8s
    /// discovery, startup) that need idempotent registration. If the URL
    /// is new (or pre-reserved), behaves like [`Self::register`] and emits
    /// [`WorkerEvent::Registered`]. If the URL already has an active
    /// worker, delegates to [`Self::replace`] and emits
    /// [`WorkerEvent::Replaced`].
    ///
    /// Holds the per-worker mutation lock for the duration of the
    /// underlying `register` or `replace` call.
    pub fn register_or_replace(&self, worker: Arc<dyn Worker>) -> WorkerId {
        // Try to create first — succeeds for fresh URLs and pre-reserved IDs
        // (where url_to_id has an entry but workers does not).
        if let Some(id) = self.register(worker.clone()) {
            return id;
        }

        // URL exists with an active worker — replace it. This is the "node
        // claims this URL" path (startup workflows, K8s discovery): if a
        // mesh import won the race for the URL, the local registration must
        // take ownership back, else the worker is never published and a peer
        // tombstone could delete a locally-configured worker. The promotion
        // happens inside replace_inner, under the per-worker mutation lock.
        if let Some(existing_id) = self.url_to_id.get(worker.url()).map(|e| e.clone()) {
            if !self.replace_inner(&existing_id, worker, Some(WorkerOrigin::Local), None) {
                // replace() returned false — worker was removed concurrently.
                // The mutation lock prevents stale indexes, so this is safe to ignore.
                tracing::warn!(
                    "register_or_replace: worker {} was removed during replace",
                    existing_id.as_str()
                );
            }
            return existing_id;
        }

        // Should not reach here: register() returned None means URL is in url_to_id.
        // Recover by clearing the stale entry and retrying full registration.
        tracing::error!(
            "register_or_replace: inconsistent state for URL {}, clearing stale entry",
            worker.url()
        );
        self.url_to_id.remove(worker.url());
        // register() will now succeed since we cleared the entry.
        // If it still fails, something is deeply wrong — return a default ID.
        self.register(worker).unwrap_or_default()
    }

    /// Replace an existing worker with a new one (overwrite-then-diff).
    ///
    /// Updates the worker object in-place and diffs the model index to avoid
    /// a transient gap where the worker is missing from indexes. Leaves the
    /// origin untouched; callers that claim the worker for this node use
    /// [`Self::replace_claiming_local`] instead.
    ///
    /// Returns `true` if the worker was replaced, `false` if the ID was
    /// not found or the URL would change (URL changes require
    /// remove + register instead).
    ///
    /// Emits [`WorkerEvent::Replaced`] on success. Holds the per-worker
    /// mutation lock for the entire diff + broadcast sequence.
    pub fn replace(&self, worker_id: &WorkerId, new_worker: Arc<dyn Worker>) -> bool {
        self.replace_inner(worker_id, new_worker, None, None)
    }

    /// Replace an existing worker and claim local ownership of its URL.
    ///
    /// Same as [`Self::replace`], plus the origin promotion that
    /// [`Self::register_or_replace`] performs: the caller configures this
    /// worker on this node, so the worker must be published to the mesh and
    /// must not be deleted by a peer tombstone.
    ///
    /// Used by `PUT /workers/{worker_id}`, which answers 202 and registers
    /// later, so other writes for the same URL can land in between. Both
    /// preconditions are checked under the per-worker mutation lock and the
    /// call returns `false` if either fails:
    ///
    /// - `worker_id` still exists — a `DELETE` (or `DELETE` + `POST`) makes
    ///   the late write fail instead of resurrecting a deleted worker or
    ///   overwriting a newly created one.
    /// - the worker is still at `expected_revision` — a second `PUT` that
    ///   finished first makes the older, slower one fail instead of silently
    ///   restoring the older specification.
    pub fn replace_claiming_local(
        &self,
        worker_id: &WorkerId,
        new_worker: Arc<dyn Worker>,
        expected_revision: u64,
    ) -> bool {
        self.replace_inner(
            worker_id,
            new_worker,
            Some(WorkerOrigin::Local),
            Some(expected_revision),
        )
    }

    /// Core replacement shared by [`Self::replace`] and
    /// [`Self::register_or_replace`]. `promote_origin` is applied under the
    /// per-worker mutation lock, before the `Replaced` event, so consumers
    /// observing the event see the final origin (the outbound mesh sync
    /// publishes a claimed worker immediately) and a concurrent `remove()`
    /// cannot orphan the entry.
    fn replace_inner(
        &self,
        worker_id: &WorkerId,
        new_worker: Arc<dyn Worker>,
        promote_origin: Option<WorkerOrigin>,
        expected_revision: Option<u64>,
    ) -> bool {
        // Serialize concurrent replacements for the same worker ID.
        // Lock is held only during the in-memory diff (no I/O, microseconds).
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        let old_worker = match self.workers.get(worker_id) {
            Some(entry) => entry.clone(),
            None => return false,
        };

        // Each replacement bumps the revision (see `inherit_shared_state_from`
        // below), so a caller that captured the revision before it started can
        // detect that another replacement landed first and give up.
        if let Some(expected) = expected_revision {
            let current = old_worker.revision();
            if current != expected {
                tracing::warn!(
                    worker_id = %worker_id.as_str(),
                    worker_url = old_worker.url(),
                    expected,
                    current,
                    "Aborting replacement: worker was replaced before the lock was acquired"
                );
                return false;
            }
        }

        let old_models: HashSet<String> = Self::worker_model_ids(&old_worker).into_iter().collect();
        let new_models: HashSet<String> = Self::worker_model_ids(&new_worker).into_iter().collect();

        // URL changes are not supported via replace — use remove + register instead
        if old_worker.url() != new_worker.url() {
            tracing::error!(
                old_url = old_worker.url(),
                new_url = new_worker.url(),
                "replace() does not support URL changes"
            );
            return false;
        }

        // Release the verdict while `old_worker` is still the current handle:
        // the replacement inherits the shared runtime, and a flag carried onto
        // a possibly different model set would leak its counter. The next poll
        // re-derives the verdict for the URL.
        self.set_worker_overloaded(&old_worker, false);

        if !new_worker.inherit_shared_state_from(&*old_worker) {
            tracing::warn!(
                worker_id = %worker_id.as_str(),
                worker_url = old_worker.url(),
                "replace() did not preserve shared mutable worker state"
            );
        }

        // Overwrite worker object atomically. The membership-order read
        // guard keeps the rebuild scan from interleaving with this write.
        {
            let _order = self.global_membership_order.read();
            self.workers.insert(worker_id.clone(), new_worker.clone());
            self.bump_global_routing_epoch();
        }

        // The replacement carries its own backend-client slot, so the old
        // instance's ZMQ handshake driver can now only hold its socket binds
        // against the new worker's connect.
        old_worker.abort_background_tasks();

        // Diff model indexes: remove stale, add new.
        for removed_model in old_models.difference(&new_models) {
            self.remove_worker_from_model_index(removed_model, old_worker.url());
            // Mirror `remove()`: drop any per-model retry override when
            // the replacement leaves the model with no workers. Without
            // this, `get_retry_config()` would keep returning a stale
            // override for a model that is no longer served.
            let model_empty = self
                .model_index
                .get(removed_model)
                .is_none_or(|workers| workers.is_empty());
            if model_empty {
                self.model_retry_configs.remove(removed_model);
            }
        }
        for added_model in new_models.difference(&old_models) {
            self.add_worker_to_model_index(added_model, new_worker.clone());
            self.rebuild_hash_ring(added_model);
            self.drop_alias_shadowed_by_model(added_model);
        }
        // For models that stayed the same, update the worker reference in the index
        for kept_model in old_models.intersection(&new_models) {
            self.add_worker_to_model_index(kept_model, new_worker.clone());
            self.rebuild_hash_ring(kept_model);
        }

        // Update aliases after canonical indexes so alias removal can inspect
        // the final worker set for each canonical model.
        for model in old_worker.models() {
            for alias in model.aliases {
                self.remove_model_alias_if_unused(&alias, &model.id);
            }
        }
        for model in new_worker.models() {
            for alias in model.aliases {
                self.add_model_alias(&alias, &model.id);
            }
        }

        self.warn_on_sampling_defaults_divergence_for_worker(&new_worker);

        if old_worker.worker_type() != new_worker.worker_type() {
            if let Some(mut type_workers) = self.type_workers.get_mut(old_worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }
            self.type_workers
                .entry(*new_worker.worker_type())
                .or_default()
                .push(worker_id.clone());
        }

        if old_worker.connection_mode() != new_worker.connection_mode() {
            if let Some(mut conn_workers) = self
                .connection_workers
                .get_mut(old_worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }
            self.connection_workers
                .entry(*new_worker.connection_mode())
                .or_default()
                .push(worker_id.clone());
        }

        if let Some(origin) = promote_origin {
            self.worker_origins.insert(worker_id.clone(), origin);
        }

        let _ = self.event_tx.send(WorkerEvent::Replaced {
            worker_id: worker_id.clone(),
            old: old_worker,
            new: new_worker,
        });

        true
    }

    /// Atomically transition a worker's lifecycle status and emit a
    /// `StatusChanged` event if it actually changed.
    ///
    /// This is a pure mutation primitive — the registry has no opinion on
    /// when a worker should transition. The caller (typically
    /// `WorkerManager`) owns the state machine logic.
    ///
    /// The per-worker mutation lock guarantees:
    ///   1. The status read, write, and event emission are atomic per
    ///      worker.
    ///   2. Two concurrent calls cannot interleave to publish events out
    ///      of order for the same worker.
    ///
    /// Returns `Some((old, new))` if the status changed, `None` if the
    /// worker is gone or the status was already `new_status`.
    ///
    /// Emits [`WorkerEvent::StatusChanged`] on transition.
    pub fn transition_status(
        &self,
        worker_id: &WorkerId,
        new_status: WorkerStatus,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        self.transition_status_inner(worker_id, None, new_status)
    }

    /// Same as [`Self::transition_status`], but becomes a no-op if the
    /// currently installed worker revision no longer matches
    /// `expected_revision`.
    ///
    /// Used by health probes that must discard stale probe outcomes
    /// after a same-URL `replace()`.
    ///
    /// Emits [`WorkerEvent::StatusChanged`] on transition. Holds the
    /// per-worker mutation lock.
    pub fn transition_status_if_revision(
        &self,
        worker_id: &WorkerId,
        expected_revision: u64,
        new_status: WorkerStatus,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        self.transition_status_inner(worker_id, Some(expected_revision), new_status)
    }

    /// Apply a worker-local mutation while holding the per-worker lock
    /// and optionally emit a `StatusChanged` event under the same lock.
    ///
    /// Used by `WorkerManager` so counter mutation and revision-checked
    /// status transitions cannot race a same-URL `replace()`, and by the
    /// drain step, whose Ready check is the closure's. The closure returns
    /// `(result, Option<new_status>)`; a transition is emitted only when
    /// the candidate status differs from the current one. A caller whose
    /// target depends on the current status reads it in the closure: that
    /// read is under the lock, so no other writer lands between it and the
    /// write.
    ///
    /// Returns `None` when the worker is gone or the revision no longer
    /// matches. Otherwise returns `Some((result, transition))` where
    /// `transition` is `Some((old, new))` if a `StatusChanged` event was
    /// emitted.
    ///
    /// Emits [`WorkerEvent::StatusChanged`] only when the candidate
    /// status differs. Holds the per-worker mutation lock for the whole
    /// closure.
    pub fn apply_if_revision<T, F>(
        &self,
        worker_id: &WorkerId,
        expected_revision: u64,
        f: F,
    ) -> Option<(T, Option<(WorkerStatus, WorkerStatus)>)>
    where
        F: FnOnce(&Arc<dyn Worker>) -> (T, Option<WorkerStatus>),
    {
        self.transition_under_lock(worker_id, Some(expected_revision), f)
    }

    /// The one place a registered worker's status transitions. Under the
    /// worker's mutation lock: the worker is looked up (gone -> `None`), its
    /// revision checked when the caller names one (moved on -> `None`), `f`
    /// runs with the worker and answers a result and the status it wants;
    /// a status that differs from the current one is written, followed by
    /// [`Self::on_status_changed`] and one `StatusChanged` event, all before
    /// the lock is released. Returns the result and the transition, if any.
    /// A caller whose target depends on the current status reads it in
    /// `f`, under the lock, never before the call: a status read outside
    /// the lock may be gone by the time the write lands.
    /// The removal's [`Self::demote_before_removal`] is the only other
    /// status write on a registered worker; it publishes nothing.
    fn transition_under_lock<T, F>(
        &self,
        worker_id: &WorkerId,
        expected_revision: Option<u64>,
        f: F,
    ) -> Option<(T, Option<(WorkerStatus, WorkerStatus)>)>
    where
        F: FnOnce(&Arc<dyn Worker>) -> (T, Option<WorkerStatus>),
    {
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();

        #[cfg(test)]
        {
            let pause = self.transition_test_hooks.lock().before_lock.take();
            if let Some((reached, resume)) = pause {
                reached.send(()).expect("transition observer is listening");
                resume
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .expect("test releases the paused transition");
            }
        }

        let _guard = lock.lock();

        let worker = self.workers.get(worker_id)?.clone();
        if expected_revision.is_some_and(|revision| worker.revision() != revision) {
            return None;
        }

        let old_status = worker.status();
        let (result, candidate_status) = f(&worker);
        let transition = match candidate_status {
            Some(new_status) if new_status != old_status => {
                worker.set_status(new_status);
                Self::on_status_changed(&worker, new_status);
                let _ = self.event_tx.send(WorkerEvent::StatusChanged {
                    worker_id: worker_id.clone(),
                    worker: worker.clone(),
                    old_status,
                    new_status,
                });
                Some((old_status, new_status))
            }
            _ => None,
        };

        Some((result, transition))
    }

    /// The removal's demotion, under the worker's mutation lock in
    /// `remove_inner`: a Ready worker about to leave `self.workers` is made
    /// ineligible first, so routing snapshots that still hold it see it so
    /// by the time the removal becomes visible. No event: the `Removed`
    /// event that follows is the one subscribers act on, and a
    /// `StatusChanged` beside it would be a second notice of the same exit.
    fn demote_before_removal(worker: &Arc<dyn Worker>) {
        if worker.status() == WorkerStatus::Ready {
            worker.set_status(WorkerStatus::NotReady);
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // 6. Update — config (no event)
    // ───────────────────────────────────────────────────────────────────

    /// Update the retry config for a model group (last write wins).
    ///
    /// Called during worker registration when the worker carries non-empty
    /// retry overrides. If `enabled` is false, `max_retries` is normalised
    /// to 1 before storage. Holds no registry locks. Emits no events.
    pub fn set_model_retry_config(&self, model_id: &str, mut config: RetryConfig, enabled: bool) {
        if !enabled {
            config.max_retries = 1;
        }
        self.model_retry_configs
            .insert(model_id.to_string(), config);
    }

    /// Reserve (or retrieve) a stable UUID for a worker URL.
    ///
    /// Used by `WorkerService::create_worker()` to return a worker ID in
    /// the 202 response before the async workflow runs. The workflow's
    /// `register_or_replace()` call will find the pre-reserved entry and
    /// create the worker under this ID. Idempotent — repeated calls for
    /// the same URL return the same ID. Emits no events.
    pub fn reserve_id_for_url(&self, url: &str) -> WorkerId {
        self.url_to_id.entry(url.to_string()).or_default().clone()
    }

    /// Drop a reservation from [`Self::reserve_id_for_url`] whose worker never
    /// registered, so a failed AddWorker does not orphan the entry (#1533).
    /// A URL that already has a live worker is left untouched.
    pub fn release_reservation(&self, url: &str) {
        self.url_to_id
            .remove_if(url, |_, id| !self.workers.contains_key(id));
    }

    // ───────────────────────────────────────────────────────────────────
    // 7. Remove
    // ───────────────────────────────────────────────────────────────────

    /// Whether a registration at another URL renders to `worker`'s label
    /// (`Worker::redacted_url`): two registrations at one address that
    /// differ only by their credentials share one, and a leaving
    /// registration then leaves the label's series to the one that stays
    /// (the retirement in `remove_inner`, the load monitor's eviction
    /// sentinel). By URL: the worker's own object and a same-URL successor
    /// (a `replace`, whose old object is evicted while the successor sits in
    /// the map) are not another registration. In `remove_inner` the leaving
    /// URL still maps to the leaving id under the per-id lock, so no other
    /// registration at that URL is in the map.
    pub(crate) fn label_shared_with(&self, worker: &Arc<dyn Worker>) -> bool {
        let label = worker.redacted_url();
        self.workers.iter().any(|other| {
            other.value().url() != worker.url() && other.value().redacted_url() == label
        })
    }

    /// Remove a worker by ID and clean up every index entry.
    ///
    /// Returns `Some(worker)` if the ID existed, `None` otherwise. Tears
    /// down the URL mapping, per-worker mutation lock, origin record,
    /// model/type/connection indexes, and per-model retry config when the
    /// last worker for a model is removed. Clears per-worker Prometheus
    /// metrics; mesh tombstoning rides the `Removed` event.
    ///
    /// Emits [`WorkerEvent::Removed`] on success. Holds the per-worker
    /// mutation lock for the whole teardown so it cannot race a
    /// concurrent `replace()`.
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.remove_inner(worker_id, None)
    }

    /// Core removal shared by [`Self::remove`] and [`Self::remove_remote`].
    /// When `expect_origin` is set, the origin is re-checked after the
    /// per-worker mutation lock is acquired and the removal aborts on a
    /// mismatch — a lock-free pre-check alone races `replace_inner`'s
    /// promotion (check Mesh, block on the lock, promotion lands, then
    /// delete a now locally-owned worker).
    fn remove_inner(
        &self,
        worker_id: &WorkerId,
        expect_origin: Option<WorkerOrigin>,
    ) -> Option<Arc<dyn Worker>> {
        // Acquire the same per-worker lock used by replace() to prevent
        // remove racing with a concurrent replace that has already snapshot
        // the old worker and is about to re-insert.
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        if let Some(expected) = expect_origin {
            if self.origin_of(worker_id) != Some(expected) {
                tracing::warn!(
                    worker_id = %worker_id.as_str(),
                    "Aborting removal: worker origin changed before the lock was acquired"
                );
                return None;
            }
        }

        // Release the overload count and demote the worker before it leaves
        // `self.workers` — stale-handle writes are dropped after removal, so
        // clearing afterwards would leak the count, and routing snapshots
        // that still hold this worker must already see it ineligible by the
        // time the removal becomes visible.
        if let Some(entry) = self.workers.get(worker_id) {
            let worker = Arc::clone(entry.value());
            drop(entry);
            self.set_worker_overloaded(&worker, false);
            Self::demote_before_removal(&worker);
        }

        // The membership-order read guard keeps the rebuild scan from
        // interleaving with this write.
        let removed = {
            let _order = self.global_membership_order.read();
            let removed = self.workers.remove(worker_id);
            if removed.is_some() {
                self.bump_global_routing_epoch();
            }
            removed
        };
        if let Some((_, worker)) = removed {
            // Retire the label's series while the URL still maps to this id: a
            // registration of the same URL cannot slip in between and have its
            // fresh series retired under it. A registration at another URL that
            // renders to the same label (the same address under rotated
            // credentials, registered before this one leaves) keeps the label's
            // series: they are retired with the last registration rendering to it.
            let label = worker.redacted_url();
            {
                // The scan and the retirement under the membership-order write
                // guard: a registration of another URL that renders to this
                // label inserts under the read side, so it lands before the
                // scan (the label reads shared) or after the retirement (its
                // `worker_registered` brings the label back), never between
                // them with its fresh series retired under it.
                let _order = self.global_membership_order.write();
                if self.label_shared_with(&worker) {
                    // The leaving registration's demotion wrote the gauges the
                    // label shares and left its state pair at 1: every pair of
                    // the label is lowered once, then the registrations that
                    // stay write their gauges and publish their pair again, so
                    // the family ends one-hot on theirs.
                    super::worker::lower_routing_state_series(label);
                    for other in self.workers.iter() {
                        let other = other.value();
                        if other.redacted_url() == label {
                            Metrics::set_worker_health(label, other.is_healthy());
                            Metrics::set_worker_cb_state(
                                label,
                                other.circuit_breaker_state().as_int(),
                            );
                            Metrics::set_worker_requests_active(label, other.load());
                            other.republish_state();
                        }
                    }
                } else {
                    Metrics::remove_worker_metrics(label);
                }
            }
            self.url_to_id.remove(worker.url());
            // We hold _guard; drop the DashMap entry but the Mutex stays alive via Arc.
            self.worker_mutation_locks.remove(worker_id);
            self.worker_origins.remove(worker_id);

            for model_id in Self::worker_model_ids(&worker) {
                self.remove_worker_from_model_index(&model_id, worker.url());
                // Drop the per-model retry config when the last worker leaves.
                let model_empty = self.model_index.get(&model_id).is_none_or(|w| w.is_empty());
                if model_empty {
                    self.model_retry_configs.remove(&model_id);
                }
            }
            for model in worker.models() {
                for alias in model.aliases {
                    self.remove_model_alias_if_unused(&alias, &model.id);
                }
            }
            if let Some(mut type_workers) = self.type_workers.get_mut(worker.worker_type()) {
                type_workers.retain(|id| id != worker_id);
            }

            if let Some(mut conn_workers) =
                self.connection_workers.get_mut(worker.connection_mode())
            {
                conn_workers.retain(|id| id != worker_id);
            }

            // Release background work owned by this instance — notably the ZMQ
            // handshake driver, whose bound sockets would otherwise block a
            // re-registration at the same URL until it times out.
            worker.abort_background_tasks();

            // Mesh tombstoning rides the `Removed` event below: the
            // outbound sync loop deletes `worker:{id}` for local workers.

            let _ = self.event_tx.send(WorkerEvent::Removed {
                worker_id: worker_id.clone(),
                worker: worker.clone(),
            });

            Some(worker)
        } else {
            None
        }
    }

    /// Remove a worker by URL.
    ///
    /// Thin wrapper over [`Self::remove`] that first resolves the URL to
    /// a `WorkerId`. Returns `None` if no worker is registered at this
    /// URL. Emits [`WorkerEvent::Removed`] on success via the underlying
    /// `remove()` call.
    ///
    /// Only *reads* the `url_to_id` mapping here — the actual removal is
    /// performed inside `remove()` while the per-worker mutation lock is
    /// held. Pre-removing the mapping would open a race where a
    /// concurrent `register()` could reclaim the URL under a new
    /// `WorkerId` before `remove()` takes the lock, and the subsequent
    /// teardown would then delete the new mapping.
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        let worker_id = self.resolve_url_to_id(url)?;
        self.remove(&worker_id)
    }

    /// Remove a mesh-imported worker in response to a remote tombstone.
    ///
    /// Removes only when the worker's origin is [`WorkerOrigin::Mesh`],
    /// re-verified under the per-worker mutation lock so a concurrent
    /// local claim (`register_or_replace` promoting the origin) cannot
    /// slip between the check and the removal. A locally-owned worker is
    /// never removed by a peer's tombstone — only the owning node retires
    /// its own key, so a remote tombstone for a local worker is anomalous
    /// and is refused with a warning. Returns the removed worker, or
    /// `None` if the id is unknown or locally owned.
    pub fn remove_remote(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        match self.origin_of(worker_id) {
            Some(WorkerOrigin::Mesh) => self.remove_inner(worker_id, Some(WorkerOrigin::Mesh)),
            Some(WorkerOrigin::Local) => {
                tracing::warn!(
                    worker_id = %worker_id.as_str(),
                    "Refusing remote tombstone for locally-owned worker"
                );
                None
            }
            None => None,
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // 8. Internal helpers
    // ───────────────────────────────────────────────────────────────────

    /// Collect the unique model IDs advertised by a worker.
    ///
    /// Public so workflow steps can share the same de-duplication rule
    /// the registry uses internally when building the model index. Falls
    /// back to the worker's primary `model_id()` if the richer
    /// `models()` list is empty. Does not touch the registry; emits no
    /// events.
    pub fn worker_model_ids(worker: &Arc<dyn Worker>) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut model_ids: Vec<String> = worker
            .models()
            .into_iter()
            .map(|model| model.id)
            .filter(|model_id| seen.insert(model_id.clone()))
            .collect();

        if model_ids.is_empty() {
            model_ids.push(worker.model_id().to_string());
        }

        model_ids
    }

    fn sampling_defaults_label(worker: &Arc<dyn Worker>) -> Option<&str> {
        worker
            .metadata()
            .spec
            .labels
            .get(DEFAULT_SAMPLING_PARAMS_LABEL)
            .map(String::as_str)
            .filter(|value| !value.is_empty())
    }

    fn sampling_defaults_values_for_group(
        &self,
        model_id: &str,
        worker_type: WorkerType,
    ) -> Vec<String> {
        self.get_workers_filtered(
            Some(model_id),
            Some(worker_type),
            Some(ConnectionMode::Grpc),
            None,
            false,
        )
        .into_iter()
        .filter_map(|worker| Self::sampling_defaults_label(&worker).map(str::to_owned))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
    }

    fn warn_on_sampling_defaults_divergence_for_worker(&self, worker: &Arc<dyn Worker>) {
        if *worker.connection_mode() != ConnectionMode::Grpc {
            return;
        }

        if !matches!(
            *worker.worker_type(),
            WorkerType::Regular | WorkerType::Decode
        ) {
            return;
        }

        if Self::sampling_defaults_label(worker).is_none() {
            return;
        }

        let worker_type = *worker.worker_type();
        for model_id in Self::worker_model_ids(worker) {
            let values = self.sampling_defaults_values_for_group(&model_id, worker_type);
            if values.len() > 1 {
                tracing::warn!(
                    model_id = %model_id,
                    worker_type = %worker_type,
                    connection_mode = %ConnectionMode::Grpc,
                    worker_url = %worker.url(),
                    observed_values = ?values,
                    "Divergent default sampling params reported by workers in the same routing group"
                );
            }
        }
    }

    /// Core registration logic shared by local and mesh paths.
    ///
    /// Acquires the per-worker mutation lock before making the worker
    /// visible in any index, and holds it for the full sequence — origin
    /// record, insert, index updates, and the `Registered` event
    /// broadcast. Releasing the lock only after the event is sent
    /// guarantees subscribers cannot observe a mutation event for this
    /// `WorkerId` before the `Registered` event that created it.
    ///
    /// `origin` records whether this is a local workflow registration or a
    /// mesh import; the outbound mesh sync consults it so imported workers
    /// are never re-published (which would version-bump the CRDT in a loop).
    fn register_inner(&self, worker: Arc<dyn Worker>, origin: WorkerOrigin) -> Option<WorkerId> {
        // Resolve (or reserve) the worker_id from url_to_id. The entry
        // API is atomic per bucket, so concurrent callers either reuse
        // the same existing_id or serialize on vacant insertion.
        let worker_id = match self.url_to_id.entry(worker.url().to_string()) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let new_id = WorkerId::new();
                entry.insert(new_id.clone());
                new_id
            }
        };

        // Acquire the per-worker mutation lock BEFORE making the worker
        // visible in `workers`. The lock is keyed on `worker_id`, so
        // concurrent registrations for the same URL serialize here.
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        // Under the lock, reject if the URL already has an active
        // worker. A pre-reserved ID (from `reserve_id_for_url`) or a
        // same-ID re-entry from a racing caller both hit this check.
        if self.workers.contains_key(&worker_id) {
            return None;
        }

        // Record origin BEFORE the worker becomes visible in `workers`:
        // lock-free readers resolve workers by URL the moment the insert
        // lands, and a visible worker with no origin would be treated as
        // mesh-imported (peer state could mutate a local worker's status).
        self.worker_origins.insert(worker_id.clone(), origin);

        // The membership-order read guard keeps the rebuild scan from
        // interleaving with this write.
        {
            let _order = self.global_membership_order.read();
            self.workers.insert(worker_id.clone(), worker.clone());
            self.bump_global_routing_epoch();
        }

        // The worker's series are kept for the time its address is
        // registered (see `observability::worker_metrics`). The builder's
        // writes for health, HTTP/2 and breaker state preceded this
        // registration and were dropped if the address was retired by an
        // earlier removal, so they are repeated here.
        Metrics::worker_registered(worker.redacted_url());
        Metrics::set_worker_health(worker.redacted_url(), worker.is_healthy());
        Metrics::set_worker_http2(worker.redacted_url(), worker.http2());
        Metrics::set_worker_cb_state(
            worker.redacted_url(),
            worker.circuit_breaker_state().as_int(),
        );
        worker.republish_state();

        // The worker's request counter starts at zero with its registration,
        // so a worker that gets no traffic shows as such.
        Metrics::init_worker_requests(worker.redacted_url(), worker.model_id());

        // Update model index for O(1) lookups using copy-on-write.
        for model_id in Self::worker_model_ids(&worker) {
            self.add_worker_to_model_index(&model_id, worker.clone());
            self.rebuild_hash_ring(&model_id);
            // Run after the model index so `add_model_alias` below sees the
            // model IDs this worker just contributed, and never while an index
            // entry is held (see the lock order on `model_alias_index`).
            self.drop_alias_shadowed_by_model(&model_id);
        }
        for model in worker.models() {
            for alias in model.aliases {
                self.add_model_alias(&alias, &model.id);
            }
        }
        self.warn_on_sampling_defaults_divergence_for_worker(&worker);

        // Update type index (clone needed for DashMap key ownership)
        self.type_workers
            .entry(*worker.worker_type())
            .or_default()
            .push(worker_id.clone());

        // Update connection mode index (clone needed for DashMap key ownership)
        self.connection_workers
            .entry(*worker.connection_mode())
            .or_default()
            .push(worker_id.clone());

        // Broadcast under the lock so event order per worker_id is
        // strictly: Registered → (Replaced | StatusChanged | Removed).
        let _ = self.event_tx.send(WorkerEvent::Registered {
            worker_id: worker_id.clone(),
            worker: worker.clone(),
        });

        Some(worker_id)
    }

    /// Reconcile the hash ring for a model to the current model index.
    ///
    /// Diffs against the cached ring so only changed URLs are rehashed: a
    /// full rebuild is O(workers) hashing plus a sort on every mutation,
    /// which at fleet scale turns a registration wave quadratic and
    /// saturates the runtime.
    fn rebuild_hash_ring(&self, model_id: &str) {
        #[cfg(test)]
        if let Some(started) = self.hash_ring_test_hooks.lock().before_rebuild.take() {
            started.send(()).expect("rebuild observer is listening");
        }

        // Different worker IDs may mutate the same model concurrently. Lock
        // before reading membership, not just before inserting the ring: an
        // older snapshot must never publish after a newer one has completed.
        let _update = self.hash_ring_update.lock();

        // Clone the copy-on-write slice out so the ring build never runs
        // under the index shard guard.
        let workers = self.model_index.get(model_id).map(|entry| entry.clone());

        #[cfg(test)]
        {
            let pause = self.hash_ring_test_hooks.lock().after_snapshot.take();
            if let Some((started, resume)) = pause {
                started.send(()).expect("snapshot observer is listening");
                resume
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .expect("test releases the paused snapshot");
            }
        }

        match workers {
            Some(workers) => {
                let previous = self.hash_rings.get(model_id).map(|ring| Arc::clone(&ring));
                let urls = workers.iter().map(|w| w.url());
                let ring = match previous {
                    Some(previous) => previous.updated(urls),
                    None => HashRing::new(urls),
                };
                self.hash_rings.insert(model_id.to_string(), Arc::new(ring));
            }
            None => {
                // No workers for this model, remove the ring
                self.hash_rings.remove(model_id);
            }
        }

        self.rebuild_wildcard_hash_ring();
    }

    /// Rebuild the ring stored under [`UNKNOWN_MODEL_ID`], which requests that
    /// name no model are routed against. Those requests may land on any worker,
    /// so the ring spans every model's workers, deduplicated by URL.
    /// Called only by `rebuild_hash_ring` while holding `hash_ring_update`.
    fn rebuild_wildcard_hash_ring(&self) {
        let model_ids: Vec<String> = self
            .model_index
            .iter()
            .map(|entry| entry.key().clone())
            .collect();

        match model_ids.as_slice() {
            [] => {
                self.hash_rings.remove(UNKNOWN_MODEL_ID);
            }
            // A single model already covers every worker, so share its ring
            // instead of hashing the same URLs a second time.
            [only] => {
                let ring = self.hash_rings.get(only).map(|ring| Arc::clone(&ring));
                match ring {
                    Some(ring) => {
                        self.hash_rings.insert(UNKNOWN_MODEL_ID.to_string(), ring);
                    }
                    None => {
                        self.hash_rings.remove(UNKNOWN_MODEL_ID);
                    }
                }
            }
            _ => {
                let mut urls: HashSet<String> = HashSet::new();
                for entry in self.model_index.iter() {
                    urls.extend(entry.value().iter().map(|w| w.url().to_string()));
                }
                let previous = self
                    .hash_rings
                    .get(UNKNOWN_MODEL_ID)
                    .map(|ring| Arc::clone(&ring));
                let ring = match previous {
                    Some(previous) => previous.updated(&urls),
                    None => HashRing::new(&urls),
                };
                self.hash_rings
                    .insert(UNKNOWN_MODEL_ID.to_string(), Arc::new(ring));
            }
        }
    }

    /// Append `worker` to the copy-on-write model index slice for `model_id`.
    /// Replaces any existing entry with the same URL so updates via replace()
    /// do not leave duplicate rows.
    fn add_worker_to_model_index(&self, model_id: &str, worker: Arc<dyn Worker>) {
        // The replaced snapshot (and its cached projections) must not drop
        // under the shard write guard: capture it and let it fall after the
        // entry expression releases the lock.
        let mut previous = None;
        self.model_index
            .entry(model_id.to_string())
            .and_modify(|existing| {
                let mut new_workers: Vec<Arc<dyn Worker>> = existing
                    .iter()
                    .filter(|w| w.url() != worker.url())
                    .cloned()
                    .collect();
                new_workers.push(worker.clone());
                previous = Some(std::mem::replace(
                    existing,
                    Arc::new(ModelWorkerSnapshot::new(Arc::from(
                        new_workers.into_boxed_slice(),
                    ))),
                ));
            })
            .or_insert_with(|| {
                Arc::new(ModelWorkerSnapshot::new(Arc::from(
                    vec![worker].into_boxed_slice(),
                )))
            });
        drop(previous);
    }

    /// Mark the global routing snapshot stale and release the superseded
    /// generation. O(1) amortized: the first bump after a published
    /// generation drops it (otherwise model-scoped-only traffic would retain
    /// a full-fleet generation until a model-less read that may never come),
    /// and every further bump in a storm drops only the shared marker. The
    /// marker's `usize::MAX` epoch can never match the live counter, so a
    /// racing fast-path reader can never serve it.
    ///
    /// Callers must hold a `global_membership_order` read guard across the
    /// worker-map mutation and this bump, so the rebuild scan cannot observe
    /// a half-applied write.
    fn bump_global_routing_epoch(&self) {
        self.global_epoch.fetch_add(1, Ordering::Release);
        let previous = self
            .global_routing_snapshot
            .swap(Arc::new(GlobalRoutingSnapshot {
                epoch: usize::MAX,
                snapshot: Self::empty_routing_snapshot(),
            }));
        drop(previous);
    }

    /// The current global membership snapshot, rebuilt first when membership
    /// changed since it was last published.
    ///
    /// The fast path is self-validating: the published object carries the
    /// epoch it was built at, so a snapshot superseded between the counter
    /// load and the pointer load fails the comparison and falls through to
    /// the rebuild. The rebuild scans the worker map under the
    /// `global_membership_order` write side, so every published snapshot is
    /// a membership state that truly existed at one instant — never a
    /// mixture of generations from interleaved shard reads.
    fn current_global_routing_snapshot(&self) -> Arc<ModelWorkerSnapshot> {
        let published = self.global_routing_snapshot.load_full();
        if published.epoch == self.global_epoch.load(Ordering::Acquire) {
            return Arc::clone(&published.snapshot);
        }
        drop(published);
        let _update = self.global_routing_update.lock();
        // Re-check under the rebuild lock: a concurrent reader may have
        // already rebuilt for the current epoch.
        let published = self.global_routing_snapshot.load_full();
        if published.epoch == self.global_epoch.load(Ordering::Acquire) {
            return Arc::clone(&published.snapshot);
        }
        drop(published);
        let (epoch, fresh) = {
            let _order = self.global_membership_order.write();
            let epoch = self.global_epoch.load(Ordering::Acquire);
            let members: Vec<Arc<dyn Worker>> = self
                .workers
                .iter()
                .map(|entry| Arc::clone(entry.value()))
                .collect();
            let fresh = Arc::new(ModelWorkerSnapshot::new(Arc::from(
                members.into_boxed_slice(),
            )));
            (epoch, fresh)
        };
        let previous = self
            .global_routing_snapshot
            .swap(Arc::new(GlobalRoutingSnapshot {
                epoch,
                snapshot: Arc::clone(&fresh),
            }));
        // Superseded generations can own large projection slices; drop them
        // outside the membership-order guard.
        drop(previous);
        fresh
    }

    /// Drop `worker_url` from the copy-on-write model index slice for `model_id`
    /// and rebuild the hash ring. Evicts the whole model entry when empty.
    fn remove_worker_from_model_index(&self, model_id: &str, worker_url: &str) {
        let mut should_remove_entry = false;
        // As in add_worker_to_model_index: the replaced snapshot drops after
        // the shard guard, not under it.
        let mut previous = None;

        if let Some(mut entry) = self.model_index.get_mut(model_id) {
            let new_workers: Vec<Arc<dyn Worker>> = entry
                .iter()
                .filter(|w| w.url() != worker_url)
                .cloned()
                .collect();

            if new_workers.is_empty() {
                previous = Some(std::mem::replace(
                    &mut *entry,
                    Arc::new(ModelWorkerSnapshot::new(Arc::from(
                        Vec::<Arc<dyn Worker>>::new().into_boxed_slice(),
                    ))),
                ));
                should_remove_entry = true;
            } else {
                previous = Some(std::mem::replace(
                    &mut *entry,
                    Arc::new(ModelWorkerSnapshot::new(Arc::from(
                        new_workers.into_boxed_slice(),
                    ))),
                ));
            }
        }
        drop(previous);

        if should_remove_entry {
            self.model_index
                .remove_if(model_id, |_, workers| workers.is_empty());
        }

        self.rebuild_hash_ring(model_id);
    }

    fn add_model_alias(&self, alias: &str, canonical_id: &str) {
        if alias == canonical_id {
            return;
        }
        if alias == UNKNOWN_MODEL_ID {
            tracing::warn!(
                alias,
                canonical_id,
                "Ignoring model alias reserved for wildcard routing"
            );
            return;
        }
        // A name is either a canonical model ID or an alias, never both.
        // Registered models win, so the alias is dropped rather than kept as a
        // shadow that would take over once the model's last worker leaves.
        //
        // The collision check runs while the alias entry is held (alias
        // before model, per the lock order on `model_alias_index`). Checked
        // before taking the entry, a concurrent registration of a model
        // named `alias` could slip between the check and the insert, run its
        // `drop_alias_shadowed_by_model` against a still-absent entry, and
        // leave the alias inserted here in place. Holding the entry makes
        // that cleanup wait until the insert is visible.
        let entry = self.model_alias_index.entry(alias.to_string());
        if self.model_index.contains_key(alias) {
            tracing::warn!(
                alias,
                canonical_id,
                "Ignoring model alias that collides with a registered model ID"
            );
            // An occupied entry here is stale: the invariant above says the
            // name cannot be both. Remove it instead of leaving it to take
            // over once the model's last worker leaves.
            if let Entry::Occupied(entry) = entry {
                entry.remove();
            }
            return;
        }

        match entry {
            Entry::Occupied(entry) if entry.get().as_ref() != canonical_id => tracing::warn!(
                alias,
                existing_canonical_id = entry.get().as_ref(),
                ignored_canonical_id = canonical_id,
                "Model alias already maps to a different canonical model"
            ),
            Entry::Occupied(_) => {}
            Entry::Vacant(entry) => {
                entry.insert(Arc::from(canonical_id));
            }
        }
    }

    /// Drop an alias entry that a newly registered model ID now shadows.
    ///
    /// Registration order decides which of the two arrives first, so the
    /// collision has to be resolved from both sides: [`Self::add_model_alias`]
    /// refuses an alias that names an existing model, and this refuses to keep
    /// an existing alias that a new model now names.
    fn drop_alias_shadowed_by_model(&self, model_id: &str) {
        if let Some((alias, shadowed_canonical_id)) = self.model_alias_index.remove(model_id) {
            tracing::warn!(
                alias,
                shadowed_canonical_id = shadowed_canonical_id.as_ref(),
                "Dropping model alias that collides with a registered model ID"
            );
        }
    }

    /// Release `canonical_id`'s claim on `alias` after its declaring workers
    /// are gone.
    ///
    /// Two models may declare the same alias; [`Self::add_model_alias`] keeps
    /// whichever registered first and warns about the rest. Deleting the alias
    /// outright when that winner leaves would strand the losers, which still
    /// advertise it, so the alias is handed to a remaining model that declares
    /// it and only deleted when none does.
    ///
    /// Holds the alias entry across the whole decision. Checking outside the
    /// entry would let a concurrent registration insert the same alias between
    /// the check and the delete, and the delete would erase it.
    fn remove_model_alias_if_unused(&self, alias: &str, canonical_id: &str) {
        let Entry::Occupied(mut entry) = self.model_alias_index.entry(alias.to_string()) else {
            return;
        };
        if entry.get().as_ref() != canonical_id {
            return;
        }
        if self.model_declares_alias(canonical_id, alias) {
            return;
        }

        match self.find_model_declaring_alias(alias, canonical_id) {
            Some(next_canonical_id) => {
                tracing::debug!(
                    alias,
                    previous_canonical_id = canonical_id,
                    next_canonical_id = %next_canonical_id,
                    "Handing model alias to another model that declares it"
                );
                entry.insert(Arc::from(next_canonical_id.as_str()));
            }
            None => {
                entry.remove();
            }
        }
    }

    /// Whether any worker still serving `canonical_id` declares `alias`.
    fn model_declares_alias(&self, canonical_id: &str, alias: &str) -> bool {
        self.model_index.get(canonical_id).is_some_and(|workers| {
            workers
                .iter()
                .any(|worker| Self::worker_declares_alias(worker, canonical_id, alias))
        })
    }

    /// Find a registered model other than `exclude` that declares `alias`.
    fn find_model_declaring_alias(&self, alias: &str, exclude: &str) -> Option<String> {
        self.model_index.iter().find_map(|entry| {
            let canonical_id = entry.key();
            if canonical_id == exclude {
                return None;
            }
            entry
                .value()
                .iter()
                .any(|worker| Self::worker_declares_alias(worker, canonical_id, alias))
                .then(|| canonical_id.clone())
        })
    }

    fn worker_declares_alias(worker: &Arc<dyn Worker>, canonical_id: &str, alias: &str) -> bool {
        worker.models().into_iter().any(|model| {
            model.id == canonical_id && model.aliases.iter().any(|candidate| candidate == alias)
        })
    }

    /// What goes with a status change besides the status. Run by
    /// [`Self::transition_under_lock`] for every write, so a worker comes
    /// back the same way whichever path promoted it (a connect signal, an
    /// admin path, a health probe).
    fn on_status_changed(worker: &Arc<dyn Worker>, new_status: WorkerStatus) {
        if new_status != WorkerStatus::Ready {
            return;
        }
        // The warm-up slice (cache_aware) counts from here.
        worker.note_admitted();
        // A worker promoted to Ready does not wait out its circuit breaker's
        // timeout: the failures that opened the breaker belong to the outage
        // its probes just saw end. An open breaker goes half-open now, so
        // the worker takes traffic at once, closes on the usual success
        // threshold and reopens on the first failure if the engine is not
        // back after all; a breaker the outage left closed forgets the
        // failure streak it banked, so the first failure after the return
        // does not open it on the outage's account. A flapping engine gets
        // one trial per promotion, and the next promotion costs it a full
        // demotion first. The breaker logs the transition itself.
        if worker.readmit_circuit_breaker() {
            Metrics::record_worker_cb_health_readmission(worker.redacted_url());
        }
    }

    /// Shared backend for [`Self::transition_status`] and
    /// [`Self::transition_status_if_revision`]: the write of
    /// [`Self::transition_under_lock`] with a fixed target status.
    fn transition_status_inner(
        &self,
        worker_id: &WorkerId,
        expected_revision: Option<u64>,
        new_status: WorkerStatus,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        self.transition_under_lock(worker_id, expected_revision, |_| ((), Some(new_status)))?
            .1
    }
}

#[cfg(test)]
impl WorkerRegistry {
    /// Test support for the interleaving proofs. Runs `transition` on its
    /// own thread and pauses it at the next status transition's seam: after
    /// whatever the caller did outside the worker's mutation lock, before
    /// the lock is taken. `other_write` then runs on this thread (another
    /// writer's transition, through the registry, so it lands under the
    /// lock), after which `transition` is let go and its result returned.
    /// No sleep determines the ordering.
    pub(crate) fn with_another_write_before_the_lock<R: Send>(
        &self,
        transition: impl FnOnce(&WorkerRegistry) -> R + Send,
        other_write: impl FnOnce(&WorkerRegistry),
    ) -> R {
        use std::time::Duration;

        let (reached_tx, reached_rx) = test_mpsc::channel();
        let (resume_tx, resume_rx) = test_mpsc::channel();
        self.transition_test_hooks.lock().before_lock = Some((reached_tx, resume_rx));

        std::thread::scope(|scope| {
            let paused = scope.spawn(|| transition(self));
            reached_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("the transition reaches the lock");
            other_write(self);
            resume_tx
                .send(())
                .expect("the paused transition is waiting");
            paused.join().expect("the transition did not panic")
        })
    }
}

// `Default` delegates to `new()` so there is a single source of truth.
// We cannot `#[derive(Default)]` on `WorkerRegistry` because
// `broadcast::Sender` has no `Default` impl — it needs an explicit
// capacity.
impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerRegistry {
    /// Sink for inbound mesh worker-state updates. The v2
    /// `WorkerSyncAdapter` calls this for each entry it pulls from
    /// the `worker:` CRDT namespace. Behaviour matches the prior
    /// `WorkerStateSubscriber` impl: URL-dedupe a remote update
    /// against an existing local worker (refresh health only), or
    /// register a new worker from the embedded `WorkerSpec` (falling
    /// back to a minimal builder if the spec is absent or invalid).
    pub fn on_remote_worker_state(&self, state: &smg_mesh::WorkerState) {
        use openai_protocol::model_card::ModelCard;

        // ZMQ is a same-host transport: its `ipc://` endpoint names a socket
        // on the publisher's machine, so importing it here would advertise a
        // route that can never reach the engine. Publishers filter these out;
        // this guard also covers peers running older builds.
        if ConnectionMode::from_url(&state.url) == Some(ConnectionMode::Zmq) {
            tracing::debug!(
                url = %state.url,
                "Ignoring mesh state for host-local ZMQ worker"
            );
            return;
        }

        // If worker already exists at this URL, update its health
        // status from the mesh state. Don't re-register — the existing
        // worker has full config from its creation workflow.
        // `true` promotes `Pending`/`NotReady` to `Ready`; `false` only
        // demotes from `Ready` to `NotReady`. `Failed` and `Draining`
        // are owned by the local state machine, never by mesh hints — a
        // dead owner's stale `health=true` key replayed by the periodic
        // reconcile would otherwise flap a probe-failed import back into
        // rotation every pass.
        if let Some(existing_id) = self.get_id_by_url(&state.url) {
            // A locally-owned worker's state is published BY this node;
            // a peer's echo of it must never mutate local status — the
            // local health state machine is the single writer.
            if self.origin_of(&existing_id) == Some(WorkerOrigin::Local) {
                tracing::debug!(
                    url = %state.url,
                    "Ignoring mesh state for locally-owned worker"
                );
                return;
            }
            // The status is read and the write decided under the worker's
            // mutation lock, in one step with the write: decided outside
            // it, a probe's `Failed` (or the drain step's `Draining`)
            // landing between the read and the write was written over
            // with the hint's `Ready` (or `NotReady`). The origin is read
            // again under the lock: a local registration claiming the URL
            // in between (`register_or_replace`) replaces the worker and
            // promotes its origin under this same lock, and the hint must
            // not write the replacement's status either. A worker that is
            // gone by the time the lock is taken (`None`) is imported anew
            // below, as before.
            let health = state.health;
            let outcome = self.transition_under_lock(&existing_id, None, |existing| {
                if self.origin_of(&existing_id) == Some(WorkerOrigin::Local) {
                    return ((), None);
                }
                let target = match (health, existing.status()) {
                    (true, WorkerStatus::Pending | WorkerStatus::NotReady) => {
                        Some(WorkerStatus::Ready)
                    }
                    (false, WorkerStatus::Ready) => Some(WorkerStatus::NotReady),
                    _ => None,
                };
                ((), target)
            });
            if outcome.is_some() {
                tracing::debug!(
                    url = %state.url,
                    healthy = state.health,
                    "Updated health for existing mesh-synced worker"
                );
                return;
            }
        }

        // Decode the spec (and run the transport gate it declares) BEFORE
        // touching any index: a rejected state must leave no trace, or the
        // id reservation below would outlive it and a legitimate worker
        // later arriving at this URL would silently inherit the rejected
        // publisher's id — breaking tombstone routing for it.
        let spec = if state.spec.is_empty() {
            None
        } else {
            match serde_json::from_slice::<openai_protocol::worker::WorkerSpec>(&state.spec) {
                Ok(spec) => {
                    // Same-host transport declared by the spec rather than by
                    // the URL scheme — not routable from this node.
                    if spec.connection_mode == ConnectionMode::Zmq {
                        tracing::debug!(
                            url = %state.url,
                            "Ignoring mesh state for host-local ZMQ worker"
                        );
                        return;
                    }
                    Some(spec)
                }
                Err(err) => {
                    tracing::warn!(
                        url = %state.url,
                        %err,
                        "undecodable WorkerSpec in mesh state; importing minimal worker"
                    );
                    None
                }
            }
        };

        // Adopt the publisher's worker id for the import so a later
        // tombstone for `worker:{id}` (which carries no value, only the
        // key) resolves to this worker. A pre-existing reservation for
        // the URL wins; the import then lives under the local id and a
        // remote tombstone for it will not resolve directly (rare; the
        // adapter's reconcile pass removes the import once its backing
        // key is gone).
        if !state.worker_id.is_empty() {
            self.url_to_id
                .entry(state.url.clone())
                .or_insert_with(|| WorkerId::from_string(state.worker_id.clone()));
        }

        // New worker — build from the full WorkerSpec if it decoded,
        // otherwise fall back to the minimal builder.
        let spec_applied = spec.is_some();
        let worker = match spec {
            Some(spec) => super::builder::BasicWorkerBuilder::from_spec(spec).build(),
            None => super::builder::BasicWorkerBuilder::new(&state.url)
                .model(ModelCard::new(&state.model_id))
                .build(),
        };

        // An explicitly-unhealthy import must not be routable: the builder
        // defaults `disable_health_check` workers to `Ready`, so the
        // `false` case needs a forced demotion, not just no promotion.
        if state.health {
            worker.set_status(WorkerStatus::Ready);
        } else {
            worker.set_status(WorkerStatus::NotReady);
        }

        // A `Mesh` origin keeps the outbound sync from re-publishing the
        // imported state (which would version-bump the CRDT in a loop),
        // but still publishes the local `Registered` event under the
        // per-worker mutation lock so in-process subscribers
        // (WorkerManager's health scheduler, etc.) pick up mesh-imported
        // workers via the same event path as any other registration.
        let worker: Arc<dyn Worker> = Arc::new(worker);
        if let Some(id) = self.register_inner(worker, WorkerOrigin::Mesh) {
            tracing::info!(
                worker_id = %id.as_str(),
                url = %state.url,
                model = %state.model_id,
                healthy = state.health,
                spec_applied,
                "Registered mesh-synced worker into local registry"
            );
        }
    }
}

/// Statistics for the worker registry
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    /// Total number of registered workers
    pub total_workers: usize,
    /// Number of unique models served
    pub total_models: usize,
    /// Number of workers passing health checks
    pub healthy_workers: usize,
    /// Number of workers failing health checks
    pub unhealthy_workers: usize,
    /// Sum of current load across all workers
    pub total_load: usize,
    /// Number of regular (non-PD) workers
    pub regular_workers: usize,
    /// Number of encode workers (EPD mode)
    pub encode_workers: usize,
    /// Number of prefill workers (PD mode)
    pub prefill_workers: usize,
    /// Number of decode workers (PD mode)
    pub decode_workers: usize,
    /// Number of HTTP-connected workers
    pub http_workers: usize,
    /// Number of gRPC-connected workers
    pub grpc_workers: usize,
    /// Number of ZMQ-connected workers (direct-backend transport)
    pub zmq_workers: usize,
    /// Number of workers with circuit breaker in Open state (not accepting requests)
    pub circuit_breaker_open: usize,
    /// Number of workers with circuit breaker in HalfOpen state (testing recovery)
    pub circuit_breaker_half_open: usize,
}

#[cfg(test)]
mod tests;
