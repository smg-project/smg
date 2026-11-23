//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing.
//!
//! # Performance Optimizations
//! The model index uses immutable Arc snapshots instead of RwLock for lock-free reads.
//! This is critical for high-concurrency scenarios where many requests query the same model.

#[cfg(any(test, feature = "test-util"))]
use std::sync::mpsc as test_mpsc;
use std::{
    collections::{BTreeSet, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use arc_swap::ArcSwap;
use dashmap::{mapref::entry::Entry, DashMap};
use rustc_hash::FxBuildHasher;
use tokio::sync::{broadcast, mpsc, Notify};
use uuid::Uuid;

use crate::{
    connection::engine::EngineConnector,
    health::circuit_breaker::CircuitState,
    identity::{
        basic::{ConnectionMode, RuntimeType, Worker, WorkerType},
        endpoint::Endpoint,
        sampling_defaults::DEFAULT_SAMPLING_PARAMS_LABEL,
    },
    metrics::writers::Metrics,
    metrics_labels,
    model_card::ModelCard,
    registry::event::{WorkerEvent, WorkerSignal},
    settings::RetryConfig,
    worker::WorkerStatus,
    UNKNOWN_MODEL_ID,
};

pub mod event;
mod pool;

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

/// What a host keeps of one membership generation.
///
/// The registry ends a generation at every membership change: a
/// registration, a removal, a property rebuild and a refresh of a worker's
/// cards each publish a new one, and so does the first read of the
/// model-less generation after any of them. At each publish it builds the
/// host's value from the generation's members, and a reader that asks for a
/// generation gets that value. What a host derives from one generation
/// lives in its value and falls with it, and a selection of several roles
/// reads every role from the one value it took.
///
/// A value falls when its generation ends: a model's once the write that
/// replaced it let the index go, the model-less one inside the next
/// membership change, under the membership order. Its drop keeps the
/// build's rule: it takes no lock of the registry and calls nothing of it.
pub trait Generation: std::fmt::Debug + Send + Sync + 'static {
    /// The value of the generation with `members`.
    ///
    /// Runs at every publish, under a registry guard: a model's shard of the
    /// index, or for the model-less generation the membership-order write
    /// guard that every registration and removal waits on. It makes empty
    /// cells and returns. It takes no lock, calls nothing of the registry
    /// and reads no member.
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self;

    /// The generation's members, as [`Self::of`] took them.
    fn members(&self) -> &Arc<[Arc<dyn Worker>]>;
}

/// The global membership snapshot together with the epoch it was built at.
///
/// Carrying the epoch inside the published object makes the read fast path
/// self-validating: load the object, compare its epoch to the live counter.
/// A bump installs a marker with `epoch: usize::MAX`, which can never match
/// and therefore can never be served.
#[derive(Debug)]
struct GlobalRoutingSnapshot<G> {
    epoch: usize,
    snapshot: Arc<G>,
}

/// Model index using immutable snapshots for lock-free reads.
/// Updates create new snapshots (copy-on-write semantics), each with its
/// host's value of the generation ([`Generation`]).
///
/// The maps a request's model id or a worker's URL is looked up in hash with
/// [`FxBuildHasher`]: the keys are the server's own (registration and
/// configuration name them; a request's model string is looked up, never
/// inserted), so the std hasher's flooding resistance buys nothing, and its
/// SipHash-1-3 over a 20-40 byte key was 0.7-1.5 % of a request's user
/// instructions with the map lookups on top. The kv index hashes its tenant
/// and node maps the same way.
type ModelIndex<G> = Arc<DashMap<String, Arc<G>, FxBuildHasher>>;

/// Shards of the maps that hold one entry a model and are iterated on
/// request paths, on any host: the model index
/// ([`WorkerRegistry::model_count`]) and the per-model retry overrides
/// ([`WorkerRegistry::any_model_retry_override_enables_retries`]). Iterating
/// a map costs one lock and one allocation a shard, whether the shard holds
/// an entry or not: the count must not follow the CPUs the process sees, as
/// the library's default does (four shards a CPU).
const MODEL_MAP_SHARDS: usize = 16;

/// Model alias to canonical model ID.
type ModelAliasIndex = Arc<DashMap<String, Arc<str>, FxBuildHasher>>;

/// The read a conditional removal makes of the worker as it stands, under its
/// lock ([`WorkerRegistry::remove_if`]).
type RemovalCondition<'a> = &'a dyn Fn(&Arc<dyn Worker>) -> bool;

/// Pauses the next status transition after whatever its caller did outside
/// the worker's mutation lock and before the lock is taken: the transition
/// reports in, then waits to be let go. The interleaving proofs place
/// another writer's transition in that window.
#[cfg(any(test, feature = "test-util"))]
#[derive(Debug, Default)]
struct TransitionTestHooks {
    before_lock: Option<(test_mpsc::Sender<()>, test_mpsc::Receiver<()>)>,
}

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry<G> {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Model index for O(1) lookups using immutable snapshots.
    /// Uses `Arc<[T]>` instead of `Arc<RwLock<Vec<T>>>` for lock-free reads.
    model_index: ModelIndex<G>,

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
    global_routing_snapshot: ArcSwap<GlobalRoutingSnapshot<G>>,

    /// The generation without a member: what an unknown model reads, and
    /// what the marker of a stale `global_routing_snapshot` carries. Built
    /// once, with the registry.
    empty_generation: Arc<G>,

    /// Membership generation; bumped on every register / replace / remove.
    /// `usize::MAX` is reserved for the stale snapshot marker.
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

    #[cfg(any(test, feature = "test-util"))]
    transition_test_hooks: parking_lot::Mutex<TransitionTestHooks>,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping
    url_to_id: Arc<DashMap<String, WorkerId, FxBuildHasher>>,

    /// Per-worker-ID locks for serializing replace() operations.
    /// Only held during the in-memory model index diff (no I/O, microseconds).
    worker_mutation_locks: Arc<DashMap<WorkerId, Arc<parking_lot::Mutex<()>>>>,

    /// Count of workers currently flagged overloaded, per model-index key.
    /// Moved only on flag transitions and on registry membership changes, so
    /// reading it costs one map probe and no worker walk.
    model_overloaded: Arc<DashMap<String, AtomicUsize, FxBuildHasher>>,

    /// Count of workers in each pool group, which the pool-size gauge reads.
    /// Moved by a delta where a worker enters, is replaced or leaves, with
    /// the gauge's write in the same hold.
    pool_sizes: pool::PoolSizes,

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

    /// How many overload edges the registry applied ([`Self::overload_edges`]):
    /// one add an edge, made under the lock above.
    overload_edges: Arc<AtomicU64>,

    /// Per-model retry config (last write wins).
    /// Updated when a worker with non-empty retry overrides registers.
    /// Cleaned up when the last worker for a model is removed.
    /// When retries are disabled, max_retries is set to 1.
    model_retry_configs: Arc<DashMap<String, RetryConfig, FxBuildHasher>>,

    /// Registration origin per worker (local vs mesh-imported). Written
    /// under the per-worker mutation lock before the `Registered` event,
    /// removed in `remove()` teardown.
    worker_origins: Arc<DashMap<WorkerId, WorkerOrigin>>,

    /// The publication a mesh import was last built from (its owner's
    /// `version`, and whether the import was built from the spec it carried
    /// or is the minimal worker), written with the import and removed with
    /// it. A state on the import's own key with a higher version is its
    /// owner's in-place replacement and rebuilds the import
    /// (`on_remote_worker_state`); the same version replayed by the
    /// reconcile pass is a health hint only.
    import_records: Arc<DashMap<WorkerId, ImportRecord>>,

    /// Broadcast channel for worker state change events.
    event_tx: broadcast::Sender<WorkerEvent>,

    /// Sender handed to workers so a completed backend connection can wake
    /// the manager for immediate promotion, and a move of the liveness veto
    /// reaches the health sink (see [`WorkerSignal`]). Lives on the registry
    /// because workers are built before the manager starts, and
    /// dynamically-registered workers need it long after.
    connect_signal_tx: mpsc::UnboundedSender<WorkerSignal>,

    /// Receiver drained by the manager's health loop. One loop holds it at
    /// a time: [`Self::take_connect_signal_receiver`] hands it out, and the
    /// loop hands it back at its end
    /// ([`Self::return_connect_signal_receiver`]). It waits here while no
    /// loop runs: health checks globally disabled, or between two loops of
    /// a host that starts one for each term of its lead.
    connect_signal_rx: parking_lot::Mutex<Option<mpsc::UnboundedReceiver<WorkerSignal>>>,

    /// Wakes a loop that started while another held the receiver: told each
    /// time the receiver comes back.
    connect_signal_home: Notify,

    /// What makes the engine connection of a worker this registry builds
    /// from a peer's published state (see [`Self::with_engine_connector`]).
    engine_connector: Option<Arc<dyn EngineConnector>>,
}

impl<G: Generation> WorkerRegistry<G> {
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
        let empty_generation = Arc::new(G::of(Arc::from(Self::EMPTY_WORKERS)));
        Self {
            workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::with_hasher_and_shard_amount(
                FxBuildHasher,
                MODEL_MAP_SHARDS,
            )),
            global_routing_snapshot: ArcSwap::from_pointee(GlobalRoutingSnapshot {
                epoch: 0,
                snapshot: Arc::clone(&empty_generation),
            }),
            empty_generation,
            global_epoch: AtomicUsize::new(0),
            global_membership_order: parking_lot::RwLock::new(()),
            global_routing_update: parking_lot::Mutex::new(()),
            model_alias_index: Arc::new(DashMap::with_hasher(FxBuildHasher)),
            #[cfg(any(test, feature = "test-util"))]
            transition_test_hooks: parking_lot::Mutex::new(TransitionTestHooks::default()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::with_hasher(FxBuildHasher)),
            worker_mutation_locks: Arc::new(DashMap::new()),
            model_overloaded: Arc::new(DashMap::with_hasher(FxBuildHasher)),
            pool_sizes: pool::PoolSizes::default(),
            overload_shed: Arc::new(AtomicBool::new(false)),
            overload_transitions: Arc::new(parking_lot::Mutex::new(())),
            overload_edges: Arc::new(AtomicU64::new(0)),
            model_retry_configs: Arc::new(DashMap::with_hasher_and_shard_amount(
                FxBuildHasher,
                MODEL_MAP_SHARDS,
            )),
            worker_origins: Arc::new(DashMap::new()),
            import_records: Arc::new(DashMap::new()),
            // Sized for fleet-scale bursts (startup registration, probe
            // storms): a lagged subscriber forces a full state resync, so
            // the capacity should comfortably exceed realistic worker
            // counts. ~100 B per slot; fixed cost ~100 KB.
            event_tx: broadcast::Sender::new(1024),
            connect_signal_tx,
            connect_signal_rx: parking_lot::Mutex::new(Some(connect_signal_rx)),
            connect_signal_home: Notify::new(),
            engine_connector: None,
        }
    }

    /// The registry with the connector its mesh imports connect through.
    /// The host names it where it builds the registry: a worker imported
    /// from a peer's published state is built here, and a gRPC or ZMQ
    /// worker built without a connector refuses its connect.
    #[must_use]
    pub fn with_engine_connector(mut self, connector: Arc<dyn EngineConnector>) -> Self {
        self.engine_connector = Some(connector);
        self
    }

    /// Registration origin for a worker, if it is currently registered.
    /// `Local` workers are owned by this node; `Mesh` workers were
    /// imported from a peer's published state.
    pub fn origin_of(&self, worker_id: &WorkerId) -> Option<WorkerOrigin> {
        self.worker_origins.get(worker_id).map(|entry| *entry)
    }

    /// Whether a worker this node owns serves the canonical model
    /// `model_id`. Mesh imports do not count: a model's policy is built from
    /// a local registration's labels and lives as long as one does. An alias
    /// does not count either: policies are keyed by canonical ids, which the
    /// routers resolve every alias to first.
    ///
    /// The worker at each of the model's index entries counts while it lists
    /// the model: a refresh of an endpoint's model list rewrites the worker's
    /// models and leaves the model index as the registration wrote it. A
    /// worker registered without model cards is indexed under its primary
    /// model alone, and counts for that.
    pub fn serves_model_locally(&self, model_id: &str) -> bool {
        let Some(indexed) = self
            .model_index
            .get(model_id)
            .map(|workers| Arc::clone(workers.members()))
        else {
            return false;
        };
        indexed.iter().any(|entry| {
            self.get_id_by_url(entry.url()).is_some_and(|id| {
                self.origin_of(&id) == Some(WorkerOrigin::Local)
                    && self.get(&id).is_some_and(|worker| {
                        worker.any_model(&mut |card| card.id == model_id)
                            || (worker.metadata().spec.models.is_wildcard()
                                && worker.model_id() == model_id)
                    })
            })
        })
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

    /// Clone the signal sender for a worker. It says there that its backend
    /// connection completed, and that its liveness veto moved. See
    /// [`WorkerSignal`]. Holds no locks.
    pub fn connect_signal_sender(&self) -> mpsc::UnboundedSender<WorkerSignal> {
        self.connect_signal_tx.clone()
    }

    /// Take the connect-signal receiver: `Some` for the one loop that holds
    /// it, `None` while another does. A loop hands it back at its end
    /// ([`Self::return_connect_signal_receiver`]), and the next take gets it
    /// with every signal sent meanwhile. The take is the crate's, as the
    /// hand-back is: a taker that could not hand it back would leave every
    /// later loop without signals. Holds the receiver lock briefly.
    pub(crate) fn take_connect_signal_receiver(
        &self,
    ) -> Option<mpsc::UnboundedReceiver<WorkerSignal>> {
        self.connect_signal_rx.lock().take()
    }

    /// Hand the connect-signal receiver back, at the end of the loop that
    /// took it, and wake a loop that waits for it
    /// ([`Self::connect_signal_receiver_returned`]). Holds the receiver lock
    /// briefly.
    pub(crate) fn return_connect_signal_receiver(
        &self,
        receiver: mpsc::UnboundedReceiver<WorkerSignal>,
    ) {
        *self.connect_signal_rx.lock() = Some(receiver);
        self.connect_signal_home.notify_one();
    }

    /// Resolves once the receiver was handed back. A loop that started
    /// without it awaits this, then takes. A hand-back nobody awaited is
    /// kept for the next await, so a loop that takes nothing on waking
    /// awaits again.
    pub(crate) async fn connect_signal_receiver_returned(&self) {
        self.connect_signal_home.notified().await;
    }

    /// True if any registered worker is Pending and depends on the connect
    /// signal (rather than health polling) to reach Ready. Such a worker is
    /// promoted only by the manager consuming [`WorkerSignal::Connected`], so the
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

    /// The id `url` maps to, a reservation from [`Self::reserve_id_for_url`]
    /// included: a `POST`'s registration job owns its reserved id before any
    /// worker stands behind it, where [`Self::get_id_by_url`] answers for
    /// live workers alone. The exact URL, no scheme canonicalisation.
    pub fn get_id_by_url_or_reservation(&self, url: &str) -> Option<WorkerId> {
        self.url_to_id.get(url).map(|id| id.clone())
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
            return Arc::clone(workers.members());
        }
        self.model_alias_index
            .get(model_id)
            .and_then(|canonical_id| {
                self.model_index
                    .get(canonical_id.as_ref())
                    .map(|workers| Arc::clone(workers.members()))
            })
            .unwrap_or_else(|| Arc::from(Self::EMPTY_WORKERS))
    }

    /// One generation for a model (alias-aware). A selection of several
    /// roles derives every role from the one value it took: two separate
    /// lookups could straddle a concurrent membership change.
    pub fn model_generation(&self, model_id: &str) -> Option<Arc<G>> {
        // Bind the clone via a `let` so the shard `Ref` drops at the end of
        // the statement (edition 2021 keeps `if let` scrutinee temporaries
        // alive through the body): what a host builds from the value on its
        // first read after a change can walk every member, and must not run
        // under the shard read guard.
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

    /// One generation for a model, with the wildcard model mapping to the
    /// global generation and an unknown model to the shared empty one.
    pub fn generation(&self, model_id: &str) -> Arc<G> {
        if model_id == UNKNOWN_MODEL_ID {
            return self.global_generation();
        }
        self.model_generation(model_id)
            .unwrap_or_else(|| Arc::clone(&self.empty_generation))
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
        let model_ids = worker_model_ids(worker);
        let mut updates = Vec::with_capacity(model_ids.len());
        {
            let _edge = self.overload_transitions.lock();
            if !self.is_current_handle(worker) {
                return false;
            }
            if !worker.set_overloaded(overloaded) {
                return false;
            }
            self.overload_edges.fetch_add(1, Ordering::Relaxed);
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

    /// How many times the overload veto of a worker changed since the
    /// registry was built, either way: one an edge
    /// ([`Self::set_worker_overloaded`]). A clear of every flag
    /// ([`Self::clear_all_overload_flags`]) counts each flag it lowers, and
    /// the removal or the replacement of an overloaded worker lowers its
    /// flag and counts that edge. A verdict that is said again counts
    /// nothing. For a host that renders series of its own.
    pub fn overload_edges(&self) -> u64 {
        self.overload_edges.load(Ordering::Relaxed)
    }

    /// Whether `worker` is the `Arc` the registry currently holds for its URL.
    ///
    /// The identity check is by pointer, not by URL: a `replace()` keeps the URL
    /// and the shared runtime but installs a new worker object with a possibly
    /// different model card, and the replaced handle must not be allowed to
    /// attribute a counter move to the model set it used to carry.
    pub(crate) fn is_current_handle(&self, worker: &Arc<dyn Worker>) -> bool {
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
                .try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
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
    /// External workers reach only the provider routers (the gateway's
    /// router selection gives them priority, and single-router mode picks by
    /// routing mode), none of which rewrite the outbound model. Keep it that
    /// way: canonicalize registry lookups there if needed, never the request
    /// body.
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
    /// - `runtime_type`: one `RuntimeType` variant
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
            .filter(|entry| !entry.value().members().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Distinct canonical models with at least one worker serving them.
    /// Wildcard workers carry no model and contribute nothing.
    pub fn model_count(&self) -> usize {
        self.model_index
            .iter()
            .filter(|entry| !entry.value().members().is_empty())
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
            .is_some_and(|workers| !workers.members().is_empty())
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
            .filter(|entry| !entry.value().members().is_empty())
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
            if !self.replace_inner(&existing_id, worker, Some(WorkerOrigin::Local), None, None) {
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
            worker.redacted_url()
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
        self.replace_inner(worker_id, new_worker, None, None, None)
    }

    /// Replace an existing worker and claim local ownership of its URL.
    ///
    /// Same as [`Self::replace`], plus the origin promotion that
    /// [`Self::register_or_replace`] performs: the caller configures this
    /// worker on this node, so the worker must be published to the mesh and
    /// must not be deleted by a peer tombstone.
    ///
    /// Used by a metadata update of `/workers/{worker_id}`, which answers 202
    /// and registers later, so other writes for the same URL can land in
    /// between (a `PUT` goes through
    /// [`Self::replace_claiming_local_unless_draining`]). Both
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
            Some(Expected {
                revision: expected_revision,
                refuse_draining: false,
            }),
            None,
        )
    }

    /// [`Self::replace_claiming_local`] for `PUT /workers/{worker_id}`: it
    /// also returns `false` when the worker reads `Draining`, checked under
    /// the same lock. A removal's mark leaves the revision as it is, so a
    /// `PUT` answered before the mark would otherwise write its replacement
    /// onto a worker that is leaving.
    pub fn replace_claiming_local_unless_draining(
        &self,
        worker_id: &WorkerId,
        new_worker: Arc<dyn Worker>,
        expected_revision: u64,
    ) -> bool {
        self.replace_inner(
            worker_id,
            new_worker,
            Some(WorkerOrigin::Local),
            Some(Expected {
                revision: expected_revision,
                refuse_draining: true,
            }),
            None,
        )
    }

    /// Core replacement shared by [`Self::replace`] and
    /// [`Self::register_or_replace`]. `promote_origin` is applied under the
    /// per-worker mutation lock, before the `Replaced` event, so consumers
    /// observing the event see the final origin (the outbound mesh sync
    /// publishes a claimed worker immediately) and a concurrent `remove()`
    /// cannot orphan the entry. `import` names the publication a mesh
    /// import is rebuilt to: the import's origin and record are read again
    /// under the lock (`import_rebuild_wanted`: a local claim, or a
    /// publication at or past this version, that landed in between aborts
    /// the replacement, as a removal does for `remove_inner`; a replacement
    /// without a usable spec keeps an import built from one), and the record
    /// is written before the lock is released, so the compare and the write
    /// are one step. `expected` is what the caller asks of the worker it
    /// replaces ([`Expected`]), tested under the same lock: a miss aborts
    /// the replacement.
    fn replace_inner(
        &self,
        worker_id: &WorkerId,
        new_worker: Arc<dyn Worker>,
        promote_origin: Option<WorkerOrigin>,
        expected: Option<Expected>,
        import: Option<ImportRecord>,
    ) -> bool {
        // Serialize concurrent replacements for the same worker ID.
        // Lock is held only during the in-memory diff (no I/O, microseconds).
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        if let Some(record) = import {
            if !self.import_rebuild_wanted(worker_id, record) {
                drop(_guard);
                self.forget_lock_of_missing(worker_id, lock);
                return false;
            }
        }

        let old_worker = match self.workers.get(worker_id) {
            Some(entry) => entry.clone(),
            None => {
                drop(_guard);
                self.forget_lock_of_missing(worker_id, lock);
                return false;
            }
        };

        // Each replacement bumps the revision (see `inherit_shared_state_from`
        // below), so a caller that captured the revision before it started can
        // detect that another replacement landed first and give up.
        if let Some(Expected {
            revision: expected,
            refuse_draining,
        }) = expected
        {
            let current = old_worker.revision();
            if current != expected {
                tracing::warn!(
                    worker_id = %worker_id.as_str(),
                    worker_url = old_worker.redacted_url(),
                    expected,
                    current,
                    "Aborting replacement: worker was replaced before the lock was acquired"
                );
                return false;
            }
            // Under the lock the removal's drain writes its mark under.
            if refuse_draining && old_worker.status() == WorkerStatus::Draining {
                tracing::warn!(
                    worker_id = %worker_id.as_str(),
                    worker_url = old_worker.redacted_url(),
                    "Aborting replacement: a removal is draining the worker"
                );
                return false;
            }
        }

        let old_models: HashSet<String> = self.models_left_by(&old_worker).into_iter().collect();
        let new_models: HashSet<String> = worker_model_ids(&new_worker).into_iter().collect();

        // URL changes are not supported via replace — use remove + register instead
        if old_worker.url() != new_worker.url() {
            tracing::error!(
                old_url = old_worker.redacted_url(),
                new_url = new_worker.redacted_url(),
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
                worker_url = old_worker.redacted_url(),
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

        // The old instance stays alive in the handles that hold it (a load
        // guard taken before the replacement) and writes under the URL until
        // they go: the series store keeps its witness, so a later removal of
        // the URL drops those writes until the old instance is gone too.
        Metrics::worker_replaced(old_worker.url(), Arc::downgrade(&old_worker));
        // A replacement of another type, mode or primary model leaves one
        // pool group and joins another, in this one step.
        self.pool_sizes
            .replaced(old_worker.as_ref(), new_worker.as_ref());

        // The replacement's series are its own from here: its HTTP/2 flag,
        // and the breaker it ends with, the adopted one or one of its own.
        // Its build wrote none, so a replacement that never got here left
        // the old worker's series as they stood.
        new_worker.publish_series();

        // The replacement owns its handshake driver. Release any binds the
        // retired instance still holds while waiting for the engine.
        old_worker.abort_background_tasks();

        // Diff model indexes: remove stale, add new.
        let mut emptied = Vec::new();
        for removed_model in old_models.difference(&new_models) {
            self.remove_worker_from_model_index(removed_model, &old_worker);
            // Mirror `remove()`: drop any per-model retry override when
            // the replacement leaves the model with no workers. Without
            // this, `get_retry_config()` would keep returning a stale
            // override for a model that is no longer served.
            let model_empty = self
                .model_index
                .get(removed_model)
                .is_none_or(|workers| workers.members().is_empty());
            if model_empty {
                self.model_retry_configs.remove(removed_model);
                emptied.push(removed_model.clone());
            }
        }
        for added_model in new_models.difference(&old_models) {
            self.add_worker_to_model_index(added_model, new_worker.clone());
            self.drop_alias_shadowed_by_model(added_model);
        }
        // For models that stayed the same, update the worker reference in the index
        for kept_model in old_models.intersection(&new_models) {
            self.add_worker_to_model_index(kept_model, new_worker.clone());
        }

        // Update aliases after canonical indexes so alias removal can inspect
        // the final worker set for each canonical model.
        for (alias, canonical_id) in self.aliases_left_by(&old_worker, &emptied) {
            self.remove_model_alias_if_unused(&alias, &canonical_id);
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
            // A claimed import is this node's worker from here on; the
            // publication it was built from no longer matters, except for
            // its version. The claim publishes the key at the shared
            // runtime's revision, which restarts at zero with the process
            // (a node back under its name claims its earlier life's key at
            // revision 1, whatever that life reached), and a peer follows a
            // publication only above the version it recorded for the
            // import. The revision is raised past the record's version
            // before the `Replaced` event, so the claim outranks every
            // version this node saw for the key: the record's version is
            // the latest it imported. A later publication of the earlier
            // life that a peer saw first reaches this node after the claim
            // and is answered by the worker-sync adapter, which raises the
            // revision past it and publishes the worker again.
            if let Some((_, record)) = self.import_records.remove(worker_id) {
                new_worker.raise_revision_to(record.version + 1);
            }
        }
        if let Some(record) = import {
            // The rebuild says what the import is built from now. A claim or
            // a removal in between dropped the record, and it must not come
            // back: `alter` writes an existing entry only.
            self.import_records.alter(worker_id, |_, _| record);
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

        #[cfg(any(test, feature = "test-util"))]
        #[expect(
            clippy::expect_used,
            reason = "test support: a hook nobody observes must fail the test that set it"
        )]
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

        let Some(worker) = self.workers.get(worker_id).map(|entry| entry.clone()) else {
            drop(_guard);
            self.forget_lock_of_missing(worker_id, lock);
            return None;
        };
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

    /// Write the model cards a refresh read for `worker`, on the same handle.
    ///
    /// The cards the worker holds already write nothing and publish nothing:
    /// a refresh that repeats them leaves every answer a snapshot keeps as
    /// it is. Other cards are stored in place, so rotations and tables keyed
    /// by the handle stand, and each model snapshot that holds the worker is
    /// published again over the same list, with the global one: what a host's
    /// value of a generation keeps from its workers' cards is read again by
    /// the next request.
    /// The model index stays as the registration wrote it, so the worker is
    /// found by its handle, not by its cards.
    ///
    /// Returns whether the cards changed. Emits no events.
    pub fn refresh_models(&self, worker: &Arc<dyn Worker>, models: Vec<ModelCard>) -> bool {
        if worker.models() == models {
            return false;
        }
        // Stored before the publish: a snapshot published from here on reads
        // the new cards.
        worker.set_models(models);
        let holding = self.indexed_under(worker);
        // As in add_worker_to_model_index: a replaced snapshot drops after
        // its shard guard, not under it.
        let mut previous = Vec::with_capacity(holding.len());
        for model_id in &holding {
            if let Some(mut entry) = self.model_index.get_mut(model_id) {
                let members = Arc::clone(entry.members());
                previous.push(std::mem::replace(&mut *entry, Arc::new(G::of(members))));
            }
        }
        drop(previous);
        let _order = self.global_membership_order.read();
        self.bump_global_routing_epoch();
        true
    }

    /// The models whose index entry holds `worker`, by its handle.
    ///
    /// The index is written from a worker's model list as it registers or
    /// takes another's place, and a refresh of the list leaves the index as
    /// it is ([`Self::refresh_models`]). So the entries that hold a worker
    /// are read from the index, and not from its list of the minute.
    fn indexed_under(&self, worker: &Arc<dyn Worker>) -> Vec<String> {
        self.model_index
            .iter()
            .filter(|entry| {
                entry
                    .members()
                    .iter()
                    .any(|member| Arc::ptr_eq(member, worker))
            })
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// The models a removal or a replacement takes `worker` out of: the ones
    /// it lists now, and every one whose index entry still holds it. A
    /// refresh of its list can leave out a model the index keeps, and the
    /// lookup by that model would hand the worker out past its removal.
    fn models_left_by(&self, worker: &Arc<dyn Worker>) -> Vec<String> {
        let mut model_ids = worker_model_ids(worker);
        for held in self.indexed_under(worker) {
            if !model_ids.contains(&held) {
                model_ids.push(held);
            }
        }
        model_ids
    }

    /// The aliases that resolve to one of `model_ids`, each with its model,
    /// read from the alias index: an alias can outlive the card that
    /// declared it, where a refresh dropped the card.
    fn aliases_of(&self, model_ids: &[String]) -> Vec<(String, String)> {
        self.model_alias_index
            .iter()
            .filter(|entry| {
                model_ids
                    .iter()
                    .any(|id| id.as_str() == entry.value().as_ref())
            })
            .map(|entry| (entry.key().clone(), entry.value().to_string()))
            .collect()
    }

    /// The aliases a removal or a replacement asks after: the ones the
    /// cards of `worker` declare, and every alias of a model the write
    /// `emptied`.
    ///
    /// The second part is read from the alias index, since a refresh can
    /// drop the card that declared an alias. It reads the emptied models
    /// alone. A refresh writes cards with no alias, so where peers still
    /// serve a model, their cards of the minute cannot tell an alias they
    /// registered from a stale one, and the alias stays.
    fn aliases_left_by(
        &self,
        worker: &Arc<dyn Worker>,
        emptied: &[String],
    ) -> Vec<(String, String)> {
        let mut aliases = self.aliases_of(emptied);
        for model in worker.models() {
            aliases.extend(
                model
                    .aliases
                    .into_iter()
                    .map(|alias| (alias, model.id.clone())),
            );
        }
        aliases
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
        self.remove_inner(worker_id, None, None)
    }

    /// Remove the worker only while `condition` reads true of the worker as
    /// it stands, the read and the removal under the same per-worker lock.
    /// The removal workflow's final step removes the incarnation it drained
    /// (the pinned revision, or a rebuild of it that still reads `Draining`:
    /// that state is the removal's own mark, inherited through a rebuild
    /// nothing re-activated) and leaves a replacement that stands on its own.
    /// Returns `None` when the worker is gone or the condition reads false.
    pub fn remove_if<F>(&self, worker_id: &WorkerId, condition: F) -> Option<Arc<dyn Worker>>
    where
        F: Fn(&Arc<dyn Worker>) -> bool,
    {
        self.remove_inner(worker_id, None, Some(&condition))
    }

    /// Core removal shared by [`Self::remove`], [`Self::remove_remote`] and
    /// [`Self::remove_if`]. When `expect_origin` is set, the origin is
    /// re-checked after the per-worker mutation lock is acquired and the
    /// removal aborts on a mismatch — a lock-free pre-check alone races
    /// `replace_inner`'s promotion (check Mesh, block on the lock, promotion
    /// lands, then delete a now locally-owned worker). When `condition` is
    /// set, it is read from the worker under the same lock and the removal
    /// returns `None` unless it holds.
    fn remove_inner(
        &self,
        worker_id: &WorkerId,
        expect_origin: Option<WorkerOrigin>,
        condition: Option<RemovalCondition<'_>>,
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

        if let Some(condition) = condition {
            let holds = self
                .workers
                .get(worker_id)
                .map(|entry| condition(entry.value()));
            if holds != Some(true) {
                drop(_guard);
                self.forget_lock_of_missing(worker_id, lock);
                return None;
            }
        }

        if let Some(expected) = expect_origin {
            if self.origin_of(worker_id) != Some(expected) {
                tracing::warn!(
                    worker_id = %worker_id.as_str(),
                    "Aborting removal: worker origin changed before the lock was acquired"
                );
                drop(_guard);
                self.forget_lock_of_missing(worker_id, lock);
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
            // Retire the URL's series while the URL still maps to this id: a
            // registration of the same URL cannot slip in between and have its
            // fresh series retired under it.
            Metrics::remove_worker_metrics(worker.url(), Arc::downgrade(&worker));
            // Its pool group gives it up in the same step.
            self.pool_sizes.left(worker.as_ref());
            self.url_to_id.remove(worker.url());
            // We hold _guard; drop the DashMap entry but the Mutex stays alive via Arc.
            self.worker_mutation_locks.remove(worker_id);
            self.worker_origins.remove(worker_id);
            self.import_records.remove(worker_id);

            let mut emptied = Vec::new();
            for model_id in self.models_left_by(&worker) {
                self.remove_worker_from_model_index(&model_id, &worker);
                // Drop the per-model retry config when the last worker leaves.
                let model_empty = self
                    .model_index
                    .get(&model_id)
                    .is_none_or(|w| w.members().is_empty());
                if model_empty {
                    self.model_retry_configs.remove(&model_id);
                    emptied.push(model_id);
                }
            }
            for (alias, canonical_id) in self.aliases_left_by(&worker, &emptied) {
                self.remove_model_alias_if_unused(&alias, &canonical_id);
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
            drop(_guard);
            self.forget_lock_of_missing(worker_id, lock);
            None
        }
    }

    /// Gives up `lock`, the mutation lock of `worker_id` that a call has
    /// released, and takes its entry out of the map when no worker stands
    /// under the id and no other call holds it. Only the removal that finds
    /// a worker drops its entry, so one a later call makes would stay.
    fn forget_lock_of_missing(&self, worker_id: &WorkerId, lock: Arc<parking_lot::Mutex<()>>) {
        // Dropped first: of several late calls the last one out then finds
        // the map's reference alone.
        drop(lock);
        if self.workers.contains_key(worker_id) {
            return;
        }
        // References are made under the map's entry lock only, so the map's
        // alone means no call holds this lock or waits on it.
        self.worker_mutation_locks
            .remove_if(worker_id, |_, current| Arc::strong_count(current) == 1);
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
            Some(WorkerOrigin::Mesh) => {
                self.remove_inner(worker_id, Some(WorkerOrigin::Mesh), None)
            }
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

    /// The models whose policy `worker` keeps: every model it serves
    /// ([`worker_model_ids`]), every model its registration listed,
    /// which a refresh of its model list can leave out while the model
    /// index keeps them, and its primary model, which names the policy its
    /// registration built. A model's policy goes once no worker of this node
    /// serves it (see [`Self::serves_model_locally`]), so a workflow that
    /// takes `worker` away asks after each of them.
    pub fn policy_model_ids(worker: &Arc<dyn Worker>) -> Vec<String> {
        let mut model_ids = worker_model_ids(worker);
        let registered = worker.metadata().spec.models.all().iter();
        for model_id in registered
            .map(|card| card.id.as_str())
            .chain([worker.model_id()])
        {
            if !model_ids.iter().any(|known| known == model_id) {
                model_ids.push(model_id.to_string());
            }
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
        for model_id in worker_model_ids(worker) {
            let values = self.sampling_defaults_values_for_group(&model_id, worker_type);
            if values.len() > 1 {
                tracing::warn!(
                    model_id = %model_id,
                    worker_type = %worker_type,
                    connection_mode = %ConnectionMode::Grpc,
                    worker_url = %worker.redacted_url(),
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
        // registered (see `metrics::series`), and building a worker writes
        // none of them: they start here, where the worker enters.
        Metrics::worker_registered(worker.url());
        worker.publish_series();
        worker.republish_state();
        // Its pool group counts it from here.
        self.pool_sizes.entered(worker.as_ref());

        // The worker's request counter starts at zero with its registration,
        // so a worker that gets no traffic shows as such.
        Metrics::init_worker_requests(worker.url(), worker.model_id());

        // Update model index for O(1) lookups using copy-on-write.
        for model_id in worker_model_ids(&worker) {
            self.add_worker_to_model_index(&model_id, worker.clone());
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
                    .members()
                    .iter()
                    .filter(|w| w.url() != worker.url())
                    .cloned()
                    .collect();
                new_workers.push(worker.clone());
                previous = Some(std::mem::replace(
                    existing,
                    Arc::new(G::of(Arc::from(new_workers.into_boxed_slice()))),
                ));
            })
            .or_insert_with(|| Arc::new(G::of(Arc::from(vec![worker].into_boxed_slice()))));
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
        self.global_epoch
            .update(Ordering::Release, Ordering::Relaxed, |epoch| {
                let next = epoch.wrapping_add(1);
                if next == usize::MAX {
                    0
                } else {
                    next
                }
            });
        let previous = self
            .global_routing_snapshot
            .swap(Arc::new(GlobalRoutingSnapshot {
                epoch: usize::MAX,
                snapshot: Arc::clone(&self.empty_generation),
            }));
        drop(previous);
    }

    /// The global generation, over every registered worker, rebuilt first
    /// when membership changed since it was last published.
    ///
    /// The fast path is self-validating: the published object carries the
    /// epoch it was built at, so a snapshot superseded between the counter
    /// load and the pointer load fails the comparison and falls through to
    /// the rebuild. The rebuild scans the worker map under the
    /// `global_membership_order` write side, so every published snapshot is
    /// a membership state that truly existed at one instant — never a
    /// mixture of generations from interleaved shard reads.
    pub fn global_generation(&self) -> Arc<G> {
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
            let fresh = Arc::new(G::of(Arc::from(members.into_boxed_slice())));
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

    /// Drop `worker` from the copy-on-write model index slice for
    /// `model_id`, by its handle. Evicts the whole model entry when empty.
    ///
    /// By handle and not by URL: a removal frees its URL ahead of its walk
    /// of the index, so a registration of the same URL can stand in the
    /// slice already, and it keeps its place.
    fn remove_worker_from_model_index(&self, model_id: &str, worker: &Arc<dyn Worker>) {
        let mut should_remove_entry = false;
        // As in add_worker_to_model_index: the replaced snapshot drops after
        // the shard guard, not under it.
        let mut previous = None;

        if let Some(mut entry) = self.model_index.get_mut(model_id) {
            let new_workers: Vec<Arc<dyn Worker>> = entry
                .members()
                .iter()
                .filter(|w| !Arc::ptr_eq(w, worker))
                .cloned()
                .collect();

            if new_workers.is_empty() {
                previous = Some(std::mem::replace(
                    &mut *entry,
                    Arc::new(G::of(Arc::from(
                        Vec::<Arc<dyn Worker>>::new().into_boxed_slice(),
                    ))),
                ));
                should_remove_entry = true;
            } else {
                previous = Some(std::mem::replace(
                    &mut *entry,
                    Arc::new(G::of(Arc::from(new_workers.into_boxed_slice()))),
                ));
            }
        }
        drop(previous);

        if should_remove_entry {
            self.model_index
                .remove_if(model_id, |_, workers| workers.members().is_empty());
        }
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
                .members()
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
                .members()
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
        if let Some(state) = worker.runtime_state() {
            state.note_admitted();
        }
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
            Metrics::record_worker_cb_health_readmission(worker.url());
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

#[cfg(any(test, feature = "test-util"))]
impl<G: Generation> WorkerRegistry<G> {
    /// Test support for the interleaving proofs. Runs `transition` on its
    /// own thread and pauses it at the next status transition's seam: after
    /// whatever the caller did outside the worker's mutation lock, before
    /// the lock is taken. `other_write` then runs on this thread (another
    /// writer's transition, through the registry, so it lands under the
    /// lock), after which `transition` is let go and its result returned.
    /// No sleep determines the ordering.
    #[expect(
        clippy::expect_used,
        reason = "test support: a hook nobody observes must fail the test that set it"
    )]
    pub fn with_another_write_before_the_lock<R: Send>(
        &self,
        transition: impl FnOnce(&Self) -> R + Send,
        other_write: impl FnOnce(&Self),
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
impl<G: Generation> Default for WorkerRegistry<G> {
    fn default() -> Self {
        Self::new()
    }
}

/// What a replacement asks of the worker it replaces, tested under the
/// worker's mutation lock.
struct Expected {
    /// The revision the caller read: it moves when another replacement lands.
    revision: u64,
    /// Whether a worker that reads `Draining` is refused too (a `PUT`).
    refuse_draining: bool,
}

/// What a mesh import was last built from: its owner's publication
/// `version`, and whether the spec that publication carried was decoded
/// into the import (`from_spec`) or the import is the minimal worker.
#[derive(Clone, Copy, Debug)]
struct ImportRecord {
    version: u64,
    from_spec: bool,
}

/// What a published state carries as its worker's specification.
enum PublishedSpec {
    /// A `Zmq` connection mode declared by the spec rather than by the URL
    /// scheme: the worker is host-local to its publisher and is not imported.
    HostLocal,
    /// No spec at all: a publisher that could not encode its spec, or one
    /// that predates the field.
    Absent,
    /// A spec this build cannot decode: a newer publisher's field.
    Undecodable(serde_json::Error),
    /// The spec, boxed: it is the one large variant.
    Spec(Box<crate::worker::WorkerSpec>),
}

/// A worker as another node published it: what the registry reads of a
/// remote record. The host's record of a peer's worker may carry more (a
/// published load, the publishing node); the registry reads neither, and
/// the host hands these fields over at its call of
/// [`WorkerRegistry::on_remote_worker_state`].
#[derive(Debug, Clone)]
pub struct RemoteWorkerState {
    /// The publisher's id for the worker, which an import adopts; empty
    /// when the publisher sent none.
    pub worker_id: String,
    /// The model the worker serves: the model of an import built without
    /// a usable spec.
    pub model_id: String,
    /// The worker's address: a record for an address the registry holds
    /// refreshes that worker, one for a new address imports it.
    pub url: String,
    /// Whether the publisher holds the worker healthy.
    pub health: bool,
    /// The publication's version: a higher one under an import's own id
    /// is its owner's replacement of the worker.
    pub version: u64,
    /// The JSON bytes of the worker's `WorkerSpec`; empty when the
    /// publisher sent none.
    pub spec: Vec<u8>,
}

impl<G: Generation> WorkerRegistry<G> {
    /// Sink for inbound mesh worker-state updates. The v2
    /// `WorkerSyncAdapter` calls this for each entry it pulls from
    /// the `worker:` CRDT namespace. A state for a URL this registry
    /// holds refreshes the worker's health, unless it is a newer
    /// publication of a mesh import's own key: then its owner replaced
    /// the worker in place, and the import is rebuilt from the embedded
    /// `WorkerSpec` under its id before the health is read. A state for
    /// a URL this registry does not hold registers a new worker from the
    /// spec (falling back to a minimal builder if the spec is absent or
    /// invalid).
    pub fn on_remote_worker_state(&self, state: &RemoteWorkerState) {
        // ZMQ is a same-host transport: its `ipc://` endpoint names a socket
        // on the publisher's machine, so importing it here would advertise a
        // route that can never reach the engine. Publishers filter these out;
        // this guard also covers peers running older builds.
        if ConnectionMode::from_url(&state.url) == Some(ConnectionMode::Zmq) {
            tracing::debug!(
                url = %Endpoint::redacted_str(&state.url),
                "Ignoring mesh state for host-local ZMQ worker"
            );
            return;
        }

        // If worker already exists at this URL, update its health
        // status from the mesh state. Don't re-register — the existing
        // worker has full config from its creation workflow (an import
        // whose owner replaced the worker is the exception, below).
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
                    url = %Endpoint::redacted_str(&state.url),
                    "Ignoring mesh state for locally-owned worker"
                );
                return;
            }
            // An owner that replaces a worker in place (a new model, role,
            // bootstrap port or labels under the same address) publishes
            // the replacement under the same key with a higher version.
            // An import built from the earlier state would otherwise keep
            // that specification for as long as it lives and route what
            // the worker no longer serves. The rebuild goes through the
            // replacement path, so the id, the origin and the shared
            // runtime state (status, breaker, counters) stay; the health
            // hint below then applies to the rebuilt import. The check here
            // is the cheap one, outside the lock; `replace_inner` decides
            // again under the worker's lock (still an import, the recorded
            // version still below this one) and writes the record there, so
            // two publications applied on two tasks leave the import at the
            // higher one, and a local claim in between is never written
            // over.
            if self.is_newer_publication_of_import(&existing_id, state) {
                match Self::importable_spec(state) {
                    PublishedSpec::HostLocal => return,
                    PublishedSpec::Spec(spec) => {
                        let replacement: Arc<dyn Worker> =
                            Arc::new(self.import_from(state, Some(*spec)));
                        let record = ImportRecord {
                            version: state.version,
                            from_spec: true,
                        };
                        if self.replace_inner(&existing_id, replacement, None, None, Some(record)) {
                            tracing::info!(
                                worker_id = %existing_id.as_str(),
                                url = %Endpoint::redacted_str(&state.url),
                                model = %state.model_id,
                                version = state.version,
                                "Rebuilt mesh-synced worker from its owner's replacement"
                            );
                        }
                    }
                    // A replacement without a spec this node can build from
                    // (none at all, or one a newer owner publishes during a
                    // rolling upgrade) rebuilds the import when it is the
                    // minimal worker already (its first state carried no
                    // usable spec either): a minimal-to-minimal rebuild
                    // loses nothing and follows the owner's new model. An
                    // import built from a spec keeps it, the minimal
                    // fallback being a downgrade for it; `replace_inner`
                    // tells the two apart under the lock and records the
                    // version either way, so the pass does not read these
                    // bytes again every interval.
                    PublishedSpec::Absent => self.rebuild_minimal_import(&existing_id, state),
                    PublishedSpec::Undecodable(err) => {
                        tracing::warn!(
                            worker_id = %existing_id.as_str(),
                            url = %Endpoint::redacted_str(&state.url),
                            version = state.version,
                            %err,
                            "the owner's replacement of a mesh-synced worker carries a spec this node cannot decode"
                        );
                        self.rebuild_minimal_import(&existing_id, state);
                    }
                }
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
                    url = %Endpoint::redacted_str(&state.url),
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
        let spec = match Self::importable_spec(state) {
            PublishedSpec::HostLocal => return,
            PublishedSpec::Spec(spec) => Some(*spec),
            PublishedSpec::Absent => None,
            PublishedSpec::Undecodable(err) => {
                tracing::warn!(
                    url = %Endpoint::redacted_str(&state.url),
                    %err,
                    "undecodable WorkerSpec in mesh state; importing minimal worker"
                );
                None
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
        let worker: Arc<dyn Worker> = Arc::new(self.import_from(state, spec));

        // A `Mesh` origin keeps the outbound sync from re-publishing the
        // imported state (which would version-bump the CRDT in a loop),
        // but still publishes the local `Registered` event under the
        // per-worker mutation lock so in-process subscribers
        // (WorkerManager's health scheduler, etc.) pick up mesh-imported
        // workers via the same event path as any other registration.
        if let Some(id) = self.register_inner(worker, WorkerOrigin::Mesh) {
            let record = ImportRecord {
                version: state.version,
                from_spec: spec_applied,
            };
            self.import_records.insert(id.clone(), record);
            tracing::info!(
                worker_id = %id.as_str(),
                url = %Endpoint::redacted_str(&state.url),
                model = %state.model_id,
                healthy = state.health,
                spec_applied,
                "Registered mesh-synced worker into local registry"
            );
        }
    }

    /// Whether `state` is the owner's newer publication of the key a mesh
    /// import was built from: the import's own key (the import adopted the
    /// publisher's id) carrying a higher version than the one recorded with
    /// the import. A state from another key for the same address (a second
    /// publisher, an orphan key of the owner's earlier life) is not, and
    /// neither is a replay of the version the import already has.
    fn is_newer_publication_of_import(
        &self,
        import_id: &WorkerId,
        state: &RemoteWorkerState,
    ) -> bool {
        state.worker_id == import_id.as_str()
            && self
                .import_records
                .get(import_id)
                .is_some_and(|built_from| state.version > built_from.version)
    }

    /// Rebuild a mesh import from its owner's publication without a usable
    /// spec: the minimal worker, through the replacement path
    /// (`import_rebuild_wanted` keeps an import built from a spec instead).
    fn rebuild_minimal_import(&self, import_id: &WorkerId, state: &RemoteWorkerState) {
        let replacement: Arc<dyn Worker> = Arc::new(self.import_from(state, None));
        let record = ImportRecord {
            version: state.version,
            from_spec: false,
        };
        if self.replace_inner(import_id, replacement, None, None, Some(record)) {
            tracing::info!(
                worker_id = %import_id.as_str(),
                url = %Endpoint::redacted_str(&state.url),
                model = %state.model_id,
                version = state.version,
                "Rebuilt the minimal mesh-synced worker from its owner's replacement"
            );
        }
    }

    /// Under the worker's lock: whether a mesh import is rebuilt to the
    /// publication `record` is for. It is not when the worker was claimed
    /// (no longer an import) or removed, when it has no record (removed and imported
    /// again since the check outside the lock: the new import's record
    /// follows its registration), when a publication at or past that
    /// version rebuilt it first, or when the publication carries no spec this node
    /// can build from and the import was built from one: the minimal
    /// fallback is a downgrade for it, so it keeps its specification and the
    /// version is recorded (the bytes never decode later on this node, and
    /// the owner's next replacement carries a higher version, so the pass
    /// does not read them again every interval). A minimal import is
    /// rebuilt: nothing is lost, and the owner's new model is gained.
    fn import_rebuild_wanted(&self, worker_id: &WorkerId, record: ImportRecord) -> bool {
        match self.origin_of(worker_id) {
            Some(WorkerOrigin::Mesh) => {}
            Some(WorkerOrigin::Local) => {
                tracing::warn!(
                    worker_id = %worker_id.as_str(),
                    "Aborting the rebuild of a mesh-synced worker: it was claimed before the lock was acquired"
                );
                return false;
            }
            // Removed under this lock since the check outside it: the
            // ordinary removal race, and the replacement would find no
            // worker anyway.
            None => {
                tracing::debug!(
                    worker_id = %worker_id.as_str(),
                    "Skipping the rebuild of a mesh-synced worker: it was removed before the lock was acquired"
                );
                return false;
            }
        }
        let Some(built_from) = self
            .import_records
            .get(worker_id)
            .map(|built_from| *built_from)
        else {
            tracing::warn!(
                worker_id = %worker_id.as_str(),
                "Aborting the rebuild of a mesh-synced worker: it was removed and imported again before the lock was acquired"
            );
            return false;
        };
        if built_from.version >= record.version {
            tracing::debug!(
                worker_id = %worker_id.as_str(),
                version = record.version,
                "Skipping the rebuild of a mesh-synced worker: a publication at or past this version rebuilt it before the lock was acquired"
            );
            return false;
        }
        if !record.from_spec && built_from.from_spec {
            self.record_import_version(worker_id, record.version);
            tracing::warn!(
                worker_id = %worker_id.as_str(),
                version = record.version,
                "keeping the mesh-synced worker as built: its owner's replacement carries no spec this node can build from"
            );
            return false;
        }
        true
    }

    /// Record `version` as the publication a mesh import was last kept at,
    /// what it was built from unchanged, when the import holds a record (a
    /// claim or a removal drops it, and a record must not come back for a
    /// worker that is no longer an import): the record never goes backwards,
    /// so two publications applied out of order leave the higher one.
    fn record_import_version(&self, import_id: &WorkerId, version: u64) {
        self.import_records
            .alter(import_id, |_, built_from| ImportRecord {
                version: built_from.version.max(version),
                ..built_from
            });
    }

    /// The `WorkerSpec` a published state carries, decoded, under the
    /// transport gate it declares. The caller says what it does without one:
    /// the import path builds the minimal worker, the rebuild keeps an
    /// import built from a spec as built and rebuilds a minimal one.
    fn importable_spec(state: &RemoteWorkerState) -> PublishedSpec {
        if state.spec.is_empty() {
            return PublishedSpec::Absent;
        }
        match serde_json::from_slice::<crate::worker::WorkerSpec>(&state.spec) {
            Ok(spec) => {
                // Same-host transport declared by the spec rather than by
                // the URL scheme — not routable from this node.
                if spec.connection_mode == ConnectionMode::Zmq {
                    tracing::debug!(
                        url = %Endpoint::redacted_str(&state.url),
                        "Ignoring mesh state for host-local ZMQ worker"
                    );
                    return PublishedSpec::HostLocal;
                }
                PublishedSpec::Spec(Box::new(spec))
            }
            Err(err) => PublishedSpec::Undecodable(err),
        }
    }

    /// The worker a published state imports as: built from its spec when it
    /// has one, from the minimal builder otherwise, with the status the
    /// state's health says. An explicitly-unhealthy import must not be
    /// routable: the builder defaults `disable_health_check` workers to
    /// `Ready`, so the `false` case needs a forced demotion, not just no
    /// promotion. It connects through the registry's connector.
    fn import_from(
        &self,
        state: &RemoteWorkerState,
        spec: Option<crate::worker::WorkerSpec>,
    ) -> super::identity::basic::BasicWorker {
        use crate::model_card::ModelCard;

        // The builder without a default: an import connects through the
        // registry's own connector.
        let builder: super::identity::builder::BasicWorkerBuilder = match spec {
            Some(spec) => super::identity::builder::BasicWorkerBuilder::from_spec(spec),
            None => super::identity::builder::BasicWorkerBuilder::new(&state.url)
                .model(ModelCard::new(&state.model_id)),
        };
        let worker = match &self.engine_connector {
            Some(connector) => builder.engine_connector(Arc::clone(connector)),
            None => builder,
        }
        .build();
        worker.set_status(if state.health {
            WorkerStatus::Ready
        } else {
            WorkerStatus::NotReady
        });
        worker
    }
}

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
mod generation_tests;
#[cfg(test)]
mod tests;
