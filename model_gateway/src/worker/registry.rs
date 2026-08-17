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

use std::{
    collections::{BTreeSet, HashSet},
    sync::Arc,
};

use dashmap::{mapref::entry::Entry, DashMap};
use openai_protocol::worker::WorkerStatus;
use tokio::sync::{broadcast, mpsc};
use uuid::Uuid;

use crate::{
    config::types::RetryConfig,
    observability::metrics::Metrics,
    worker::{
        circuit_breaker::CircuitState,
        event::{WorkerConnected, WorkerEvent},
        hash_ring::HashRing,
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

/// Model index using immutable snapshots for lock-free reads.
/// Each model maps to an Arc'd slice of workers that can be read without locking.
/// Updates create new snapshots (copy-on-write semantics).
type ModelIndex = Arc<DashMap<String, Arc<[Arc<dyn Worker>]>>>;

/// Model alias to canonical model ID.
type ModelAliasIndex = Arc<DashMap<String, Arc<str>>>;

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry {
    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Model index for O(1) lookups using immutable snapshots.
    /// Uses Arc<[T]> instead of Arc<RwLock<Vec<T>>> for lock-free reads.
    model_index: ModelIndex,

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

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping
    url_to_id: Arc<DashMap<String, WorkerId>>,

    /// Per-worker-ID locks for serializing replace() operations.
    /// Only held during the in-memory model index diff (no I/O, microseconds).
    worker_mutation_locks: Arc<DashMap<WorkerId, Arc<parking_lot::Mutex<()>>>>,

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
            model_alias_index: Arc::new(DashMap::new()),
            hash_rings: Arc::new(DashMap::new()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
            worker_mutation_locks: Arc::new(DashMap::new()),
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
    /// `http://`/`grpc://` form — see [`Self::resolve_url_to_id`].
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
    /// `http://`/`grpc://` form — see [`Self::resolve_url_to_id`].
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
            return Arc::clone(&workers);
        }
        self.model_alias_index
            .get(model_id)
            .and_then(|canonical_id| {
                self.model_index
                    .get(canonical_id.as_ref())
                    .map(|workers| Arc::clone(&workers))
            })
            .unwrap_or_else(|| Arc::from(Self::EMPTY_WORKERS))
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
    /// External workers reach only the OpenAI, Anthropic and Gemini routers
    /// ([`RouterManager::select_router_for_workers`] gives them priority, and
    /// single-router mode picks by routing mode), none of which rewrite the
    /// outbound model. Keep it that way: canonicalize registry lookups there
    /// if needed, never the request body.
    ///
    /// [`RouterManager::select_router_for_workers`]: crate::routers::RouterManager
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
                ConnectionMode::Grpc | ConnectionMode::Zmq => grpc_count += 1,
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

    // ───────────────────────────────────────────────────────────────────
    // 5. Write — mutation primitives
    //
    // Every method in this section holds the per-worker mutation lock
    // (`worker_mutation_locks`) and emits exactly one `WorkerEvent` before
    // releasing the lock. New mutation methods MUST follow this pattern.
    // Manual publish at each call site is intentional — there are only a
    // handful of mutation methods, and the simplicity beats a generic
    // helper layer.
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

        if !new_worker.inherit_shared_state_from(&*old_worker) {
            tracing::warn!(
                worker_id = %worker_id.as_str(),
                worker_url = old_worker.url(),
                "replace() did not preserve shared mutable worker state"
            );
        }

        // Overwrite worker object atomically
        self.workers.insert(worker_id.clone(), new_worker.clone());

        // Diff model indexes: remove stale, add new
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
    /// status transitions cannot race a same-URL `replace()`. The closure
    /// returns `(result, Option<new_status>)`; a transition is emitted
    /// only when the candidate status differs from the current one.
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
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        let worker = self.workers.get(worker_id)?.clone();
        if worker.revision() != expected_revision {
            return None;
        }

        let old_status = worker.status();
        let (result, candidate_status) = f(&worker);
        let transition = match candidate_status {
            Some(new_status) if new_status != old_status => {
                worker.set_status(new_status);
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

        if let Some((_, worker)) = self.workers.remove(worker_id) {
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

            // Mark the worker as not-ready before tearing down its
            // metrics so any in-flight `is_healthy()` callers that
            // still hold an `Arc` see the correct state. Skip the
            // transition for `Pending` (hasn't proven itself) and
            // `Failed` (already terminal); only Ready warrants the
            // explicit demotion. Mirrors the legacy `set_healthy(false)`
            // semantics without going through the deprecated shim.
            if worker.status() == WorkerStatus::Ready {
                worker.set_status(WorkerStatus::NotReady);
            }
            Metrics::remove_worker_metrics(worker.url());

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

        self.workers.insert(worker_id.clone(), worker.clone());

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

    /// Rebuild the hash ring for a model based on current workers in the model index.
    fn rebuild_hash_ring(&self, model_id: &str) {
        let ring = self
            .model_index
            .get(model_id)
            .map(|workers| Arc::new(HashRing::new(workers.value().iter().map(|w| w.url()))));

        match ring {
            Some(ring) => {
                self.hash_rings.insert(model_id.to_string(), ring);
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
                self.hash_rings
                    .insert(UNKNOWN_MODEL_ID.to_string(), Arc::new(HashRing::new(urls)));
            }
        }
    }

    /// Append `worker` to the copy-on-write model index slice for `model_id`.
    /// Replaces any existing entry with the same URL so updates via replace()
    /// do not leave duplicate rows.
    fn add_worker_to_model_index(&self, model_id: &str, worker: Arc<dyn Worker>) {
        self.model_index
            .entry(model_id.to_string())
            .and_modify(|existing| {
                let mut new_workers: Vec<Arc<dyn Worker>> = existing
                    .iter()
                    .filter(|w| w.url() != worker.url())
                    .cloned()
                    .collect();
                new_workers.push(worker.clone());
                *existing = Arc::from(new_workers.into_boxed_slice());
            })
            .or_insert_with(|| Arc::from(vec![worker].into_boxed_slice()));
    }

    /// Drop `worker_url` from the copy-on-write model index slice for `model_id`
    /// and rebuild the hash ring. Evicts the whole model entry when empty.
    fn remove_worker_from_model_index(&self, model_id: &str, worker_url: &str) {
        let mut should_remove_entry = false;

        if let Some(mut entry) = self.model_index.get_mut(model_id) {
            let new_workers: Vec<Arc<dyn Worker>> = entry
                .iter()
                .filter(|w| w.url() != worker_url)
                .cloned()
                .collect();

            if new_workers.is_empty() {
                *entry = Arc::from(Vec::<Arc<dyn Worker>>::new().into_boxed_slice());
                should_remove_entry = true;
            } else {
                *entry = Arc::from(new_workers.into_boxed_slice());
            }
        }

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

    /// Shared backend for [`Self::transition_status`] and
    /// [`Self::transition_status_if_revision`]. Holds the per-worker
    /// mutation lock for the full read-modify-emit sequence.
    fn transition_status_inner(
        &self,
        worker_id: &WorkerId,
        expected_revision: Option<u64>,
        new_status: WorkerStatus,
    ) -> Option<(WorkerStatus, WorkerStatus)> {
        let lock = self
            .worker_mutation_locks
            .entry(worker_id.clone())
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone();
        let _guard = lock.lock();

        let worker = self.workers.get(worker_id)?.clone();
        if expected_revision.is_some_and(|revision| worker.revision() != revision) {
            return None;
        }

        let old_status = worker.status();
        if old_status == new_status {
            return None;
        }

        worker.set_status(new_status);

        let _ = self.event_tx.send(WorkerEvent::StatusChanged {
            worker_id: worker_id.clone(),
            worker: worker.clone(),
            old_status,
            new_status,
        });

        Some((old_status, new_status))
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
            if let Some(existing) = self.get(&existing_id) {
                let status = existing.status();
                if state.health {
                    if matches!(status, WorkerStatus::Pending | WorkerStatus::NotReady) {
                        existing.set_status(WorkerStatus::Ready);
                    }
                } else if status == WorkerStatus::Ready {
                    existing.set_status(WorkerStatus::NotReady);
                }
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
    /// Number of workers with circuit breaker in Open state (not accepting requests)
    pub circuit_breaker_open: usize,
    /// Number of workers with circuit breaker in HalfOpen state (testing recovery)
    pub circuit_breaker_half_open: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use openai_protocol::model_card::ModelCard;

    use super::*;
    use crate::worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        BasicWorkerBuilder, WorkerLoadGuard,
    };

    fn no_health_check() -> openai_protocol::worker::HealthCheckConfig {
        openai_protocol::worker::HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn worker_with_sampling_defaults(
        url: &str,
        model_id: &str,
        worker_type: WorkerType,
        connection_mode: ConnectionMode,
        defaults: Option<&str>,
    ) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model_id))
            .worker_type(worker_type)
            .connection_mode(connection_mode);

        if let Some(defaults) = defaults {
            let mut labels = HashMap::new();
            labels.insert(
                DEFAULT_SAMPLING_PARAMS_LABEL.to_string(),
                defaults.to_string(),
            );
            builder = builder.labels(labels);
        }

        Arc::new(builder.build())
    }

    fn worker_with_model_aliases(
        url: &str,
        canonical_id: &str,
        aliases: &[&str],
        worker_type: WorkerType,
    ) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new(canonical_id).with_aliases(aliases.iter().copied()))
                .worker_type(worker_type)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn assert_sampling_defaults_group_values(
        registry: &WorkerRegistry,
        model_id: &str,
        worker_type: WorkerType,
        expected: &[&str],
    ) {
        let expected: Vec<String> = expected.iter().map(|value| value.to_string()).collect();
        assert_eq!(
            registry.sampling_defaults_values_for_group(model_id, worker_type),
            expected
        );
    }

    #[test]
    fn connect_signal_receiver_is_taken_exactly_once() {
        let registry = WorkerRegistry::new();
        assert!(
            registry.take_connect_signal_receiver().is_some(),
            "first take (the manager) must get the receiver"
        );
        assert!(
            registry.take_connect_signal_receiver().is_none(),
            "a second take must return None so a second manager runs poll-only"
        );
    }

    #[tokio::test]
    async fn connect_signal_sender_delivers_to_the_receiver() {
        let registry = WorkerRegistry::new();
        let tx = registry.connect_signal_sender();
        let mut rx = registry
            .take_connect_signal_receiver()
            .expect("receiver present");

        tx.send(WorkerConnected {
            url: "ipc:///tmp/w.ipc".to_string(),
            revision: 7,
        })
        .expect("send on a live channel");

        let got = rx.recv().await.expect("a delivered signal");
        assert_eq!(got.url, "ipc:///tmp/w.ipc");
        assert_eq!(got.revision, 7);
    }

    #[test]
    fn has_workers_awaiting_connect_signal_tracks_pending_zmq_workers() {
        let registry = WorkerRegistry::new();
        assert!(
            !registry.has_workers_awaiting_connect_signal(),
            "empty registry awaits nothing"
        );

        // An HTTP worker never waits on the connect signal.
        let http: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w:1")
                .connection_mode(ConnectionMode::Http)
                .build(),
        );
        registry.register(http).unwrap();
        assert!(!registry.has_workers_awaiting_connect_signal());

        // A Pending ZMQ worker is promoted only by the connect signal.
        let zmq: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
                .connection_mode(ConnectionMode::Zmq)
                .build(),
        );
        assert_eq!(zmq.status(), WorkerStatus::Pending);
        let zmq_url = zmq.url().to_string();
        registry.register(zmq).unwrap();
        assert!(registry.has_workers_awaiting_connect_signal());

        // Once promoted to Ready it no longer awaits the signal.
        registry
            .get_by_url(&zmq_url)
            .unwrap()
            .set_status(WorkerStatus::Ready);
        assert!(!registry.has_workers_awaiting_connect_signal());
    }

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register worker
        let worker_id = registry.register(Arc::from(worker)).unwrap();

        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(ConnectionMode::Http).len(), 1);

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_stats_counts_encode_workers() {
        let registry = WorkerRegistry::new();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://encode-worker:8080")
                .worker_type(WorkerType::Encode)
                .connection_mode(ConnectionMode::Grpc)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );

        registry.register(worker).unwrap();

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.encode_workers, 1);
        assert_eq!(stats.prefill_workers, 0);
        assert_eq!(stats.decode_workers, 0);
        assert_eq!(stats.regular_workers, 0);
        assert_eq!(stats.grpc_workers, 1);
    }

    #[test]
    fn origin_tracks_local_and_mesh_registrations() {
        let registry = WorkerRegistry::new();

        let local: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://local:8080")
                .model(ModelCard::new("llama-3"))
                .build(),
        );
        let local_id = registry.register(local).unwrap();
        assert_eq!(registry.origin_of(&local_id), Some(WorkerOrigin::Local));

        registry.on_remote_worker_state(&smg_mesh::WorkerState {
            worker_id: "peer-w1".to_string(),
            model_id: "llama-3".to_string(),
            url: "http://remote:8080".to_string(),
            health: true,
            load: 0.0,
            version: 1,
            spec: vec![],
        });
        let mesh_id = registry.get_id_by_url("http://remote:8080").unwrap();
        assert_eq!(registry.origin_of(&mesh_id), Some(WorkerOrigin::Mesh));
    }

    fn remote_state(
        worker_id: &str,
        url: &str,
        health: bool,
        spec: Vec<u8>,
    ) -> smg_mesh::WorkerState {
        smg_mesh::WorkerState {
            worker_id: worker_id.to_string(),
            model_id: "llama-3".to_string(),
            url: url.to_string(),
            health,
            load: 0.0,
            version: 1,
            spec,
        }
    }

    #[test]
    fn mesh_import_adopts_publisher_worker_id() {
        let registry = WorkerRegistry::new();
        registry.on_remote_worker_state(&remote_state(
            "peer-w1",
            "http://remote:8080",
            true,
            vec![],
        ));
        assert_eq!(
            registry.get_id_by_url("http://remote:8080"),
            Some(WorkerId::from_string("peer-w1".to_string())),
            "import keys under the publisher's id so its tombstone resolves"
        );
    }

    #[test]
    fn mesh_state_for_zmq_worker_is_never_imported() {
        // ZMQ is same-host: an `ipc://` endpoint published by a peer names a
        // socket path on that peer's machine, so importing it would advertise
        // an unroutable worker.
        let registry = WorkerRegistry::new();
        registry.on_remote_worker_state(&remote_state(
            "peer-w1",
            "ipc:///tmp/smg-peer.sock",
            true,
            vec![],
        ));
        assert!(
            registry.get_by_url("ipc:///tmp/smg-peer.sock").is_none(),
            "a host-local ZMQ worker must not be imported from the mesh"
        );

        // Same rejection when the transport is declared only by the spec.
        let spec: openai_protocol::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
            "url": "http://remote:8080",
            "connection_mode": "zmq"
        }))
        .unwrap();
        registry.on_remote_worker_state(&remote_state(
            "peer-w2",
            "http://remote:8080",
            true,
            serde_json::to_vec(&spec).unwrap(),
        ));
        assert!(
            registry.get_by_url("http://remote:8080").is_none(),
            "a spec-declared ZMQ worker must not be imported from the mesh"
        );
    }

    #[test]
    fn rejected_zmq_state_leaves_no_url_to_id_residue() {
        // Both transport gates run before the id reservation. A leftover
        // `url_to_id` entry would be invisible to `get_id_by_url` (which
        // skips ids with no live worker) yet still win the `Entry::Occupied`
        // arm in `register_inner`, handing the next legitimate worker at
        // this URL the rejected publisher's id — so a peer tombstone for
        // that id would delete a worker that never came from the mesh.
        let registry = WorkerRegistry::new();

        // Rejected by the URL scheme.
        registry.on_remote_worker_state(&remote_state(
            "peer-w1",
            "ipc:///tmp/smg-peer.sock",
            true,
            vec![],
        ));
        assert!(
            registry.url_to_id.get("ipc:///tmp/smg-peer.sock").is_none(),
            "a URL-scheme rejection must not reserve an id"
        );

        // Rejected by the spec's declared connection mode.
        let spec: openai_protocol::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
            "url": "http://remote:8080",
            "connection_mode": "zmq"
        }))
        .unwrap();
        registry.on_remote_worker_state(&remote_state(
            "peer-w2",
            "http://remote:8080",
            true,
            serde_json::to_vec(&spec).unwrap(),
        ));
        assert!(
            registry.url_to_id.get("http://remote:8080").is_none(),
            "a spec rejection must not reserve an id"
        );

        // A legitimate worker later arriving at the same URL gets its own id.
        let local: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://remote:8080")
                .model(ModelCard::new("llama-3"))
                .build(),
        );
        let local_id = registry.register(local).expect("registers");
        assert_ne!(
            local_id,
            WorkerId::from_string("peer-w2".to_string()),
            "a later worker must not inherit the rejected publisher's id"
        );
        assert_eq!(registry.get_id_by_url("http://remote:8080"), Some(local_id));
    }

    #[test]
    fn mesh_state_never_mutates_locally_owned_worker() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://local:8080")
                .model(ModelCard::new("m"))
                .build(),
        );
        let id = registry.register(worker.clone()).unwrap();
        worker.set_status(WorkerStatus::Ready);

        registry.on_remote_worker_state(&remote_state(
            "peer-x",
            "http://local:8080",
            false,
            vec![],
        ));
        assert_eq!(
            registry.get(&id).unwrap().status(),
            WorkerStatus::Ready,
            "a peer's echo must not demote a locally-owned worker"
        );
    }

    #[test]
    fn unhealthy_import_with_disabled_health_check_is_not_ready() {
        // The builder defaults disable_health_check workers to Ready; an
        // explicitly-unhealthy import must still land unroutable.
        let spec: openai_protocol::worker::WorkerSpec = serde_json::from_value(serde_json::json!({
            "url": "http://remote:8080",
            "health": { "disable_health_check": true }
        }))
        .unwrap();
        let registry = WorkerRegistry::new();
        registry.on_remote_worker_state(&remote_state(
            "peer-w1",
            "http://remote:8080",
            false,
            serde_json::to_vec(&spec).unwrap(),
        ));
        let worker = registry.get_by_url("http://remote:8080").expect("imported");
        assert_ne!(
            worker.status(),
            WorkerStatus::Ready,
            "an explicitly-unhealthy import must not be routable"
        );
    }

    #[test]
    fn stale_healthy_state_never_resurrects_probe_failed_import() {
        // A dead owner's key keeps health=true forever; the periodic
        // reconcile replays it. A probe-failed import must stay failed,
        // not flap back into rotation every pass.
        let registry = WorkerRegistry::new();
        let state = remote_state("peer-w1", "http://remote:8080", true, vec![]);
        registry.on_remote_worker_state(&state);
        let worker = registry.get_by_url("http://remote:8080").unwrap();
        assert_eq!(worker.status(), WorkerStatus::Ready);

        worker.set_status(WorkerStatus::Failed);
        registry.on_remote_worker_state(&state);
        assert_eq!(
            registry.get_by_url("http://remote:8080").unwrap().status(),
            WorkerStatus::Failed,
            "mesh hints must not resurrect probe-owned terminal states"
        );

        // NotReady is still promotable: the owner saying healthy again
        // is the legitimate recovery signal.
        worker.set_status(WorkerStatus::NotReady);
        registry.on_remote_worker_state(&state);
        assert_eq!(
            registry.get_by_url("http://remote:8080").unwrap().status(),
            WorkerStatus::Ready
        );
    }

    #[test]
    fn remove_remote_only_removes_mesh_origin_workers() {
        let registry = WorkerRegistry::new();

        registry.on_remote_worker_state(&remote_state(
            "peer-w1",
            "http://remote:8080",
            true,
            vec![],
        ));
        let mesh_id = registry.get_id_by_url("http://remote:8080").unwrap();
        assert!(registry.remove_remote(&mesh_id).is_some());
        assert!(registry.get_by_url("http://remote:8080").is_none());

        let local: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://local:8080")
                .model(ModelCard::new("m"))
                .build(),
        );
        let local_id = registry.register(local).unwrap();
        assert!(registry.remove_remote(&local_id).is_none());
        assert!(
            registry.get(&local_id).is_some(),
            "a locally-owned worker survives a remote tombstone"
        );
    }

    #[test]
    fn register_or_replace_promotes_mesh_origin_to_local() {
        // Restart race: a mesh import wins the URL before the local
        // workflow registers it. The local claim must take ownership back,
        // else the node never publishes its own worker and a peer tombstone
        // could delete it.
        let registry = WorkerRegistry::new();
        registry.on_remote_worker_state(&remote_state("peer-w1", "http://w:8080", true, vec![]));
        let id = registry.get_id_by_url("http://w:8080").unwrap();
        assert_eq!(registry.origin_of(&id), Some(WorkerOrigin::Mesh));

        let local: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w:8080")
                .model(ModelCard::new("llama-3"))
                .build(),
        );
        let claimed_id = registry.register_or_replace(local);
        assert_eq!(claimed_id, id, "claim reuses the adopted id");
        assert_eq!(
            registry.origin_of(&id),
            Some(WorkerOrigin::Local),
            "local registration over a mesh import takes ownership"
        );
        assert!(
            registry.remove_remote(&id).is_none(),
            "a peer tombstone can no longer delete the claimed worker"
        );
    }

    #[test]
    fn remove_inner_aborts_when_origin_changed_before_lock() {
        // The TOCTOU guard: remove_remote's pre-check can observe Mesh,
        // then lose the lock race to a local claim that promotes the
        // origin. The under-lock recheck must abort the removal.
        let registry = WorkerRegistry::new();
        let local: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w:8080")
                .model(ModelCard::new("m"))
                .build(),
        );
        let id = registry.register(local).unwrap();

        // Models the post-promotion state: expecting Mesh, finding Local.
        assert!(
            registry
                .remove_inner(&id, Some(WorkerOrigin::Mesh))
                .is_none(),
            "removal must abort when the origin no longer matches"
        );
        assert!(
            registry.get(&id).is_some(),
            "the claimed worker survives the racing tombstone"
        );
    }

    #[test]
    fn origin_survives_replace_and_clears_on_remove() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w:8080")
                .model(ModelCard::new("m"))
                .build(),
        );
        let id = registry.register(worker).unwrap();

        let replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w:8080")
                .model(ModelCard::new("m2"))
                .build(),
        );
        assert!(registry.replace(&id, replacement));
        assert_eq!(
            registry.origin_of(&id),
            Some(WorkerOrigin::Local),
            "replace keeps the same id, so origin is untouched"
        );

        registry.remove(&id);
        assert_eq!(registry.origin_of(&id), None, "remove clears the origin");
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels1)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels2)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://worker3:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels3)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        // Register workers
        registry.register(Arc::from(worker1)).unwrap();
        registry.register(Arc::from(worker2)).unwrap();
        registry.register(Arc::from(worker3)).unwrap();

        let llama_workers = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        let gpt_workers = registry.get_by_model("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        let unknown_workers = registry.get_by_model("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }

    // Health-checker integration tests moved to worker/manager.rs along with
    // the loop itself. The registry is now a pure collection — see
    // `worker::manager::WorkerManager` tests.

    #[test]
    fn test_transition_status_emits_event_and_changes_status() {
        let registry = WorkerRegistry::new();
        let mut rx = registry.subscribe_events();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let worker_id = registry.register(worker.clone()).unwrap();
        // Drain Registered event
        let _ = rx.try_recv().unwrap();

        // Initial status is Ready (disable_health_check). Transition to NotReady.
        let result = registry.transition_status(&worker_id, WorkerStatus::NotReady);
        assert_eq!(result, Some((WorkerStatus::Ready, WorkerStatus::NotReady)));
        assert_eq!(worker.status(), WorkerStatus::NotReady);

        let event = rx.try_recv().unwrap();
        match event {
            WorkerEvent::StatusChanged {
                old_status,
                new_status,
                ..
            } => {
                assert_eq!(old_status, WorkerStatus::Ready);
                assert_eq!(new_status, WorkerStatus::NotReady);
            }
            other => panic!("Expected StatusChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_transition_status_no_op_when_status_unchanged() {
        let registry = WorkerRegistry::new();
        let mut rx = registry.subscribe_events();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let worker_id = registry.register(worker).unwrap();
        let _ = rx.try_recv().unwrap();

        // Already Ready — transition to Ready is a no-op
        assert_eq!(
            registry.transition_status(&worker_id, WorkerStatus::Ready),
            None
        );
        assert!(rx.try_recv().is_err(), "no event should be emitted");
    }

    #[test]
    fn test_transition_status_returns_none_for_missing_worker() {
        let registry = WorkerRegistry::new();
        let missing = WorkerId::from_string("nonexistent".to_string());
        assert_eq!(
            registry.transition_status(&missing, WorkerStatus::Ready),
            None
        );
    }

    #[test]
    fn test_get_id_by_url_returns_id_for_registered_worker() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w-by-url:8080")
                .worker_type(WorkerType::Regular)
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let worker_id = registry.register(worker).unwrap();
        assert_eq!(
            registry.get_id_by_url("http://w-by-url:8080"),
            Some(worker_id)
        );
    }

    #[test]
    fn test_get_id_by_url_returns_none_for_unknown_url() {
        let registry = WorkerRegistry::new();
        assert!(registry.get_id_by_url("http://missing:8080").is_none());
    }

    #[test]
    fn test_url_lookup_canonicalizes_bare_host_port_to_http_form() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://canon:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        );
        let worker_id = registry.register(worker).unwrap();

        assert_eq!(registry.get_id_by_url("canon:8080"), Some(worker_id));
        assert!(registry.get_by_url("canon:8080").is_some());
    }

    #[test]
    fn test_url_lookup_canonicalizes_bare_host_port_to_grpc_form() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://canon-grpc:8080")
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        );
        let worker_id = registry.register(worker).unwrap();

        assert_eq!(registry.get_id_by_url("canon-grpc:8080"), Some(worker_id));
        assert!(registry.get_by_url("canon-grpc:8080").is_some());
    }

    #[test]
    fn test_remove_by_url_canonicalizes_bare_host_port() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://to-remove:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        );
        let worker_id = registry.register(worker).unwrap();

        assert!(registry.remove_by_url("to-remove:8080").is_some());
        assert!(registry.get(&worker_id).is_none());
        assert!(registry.get_by_url("http://to-remove:8080").is_none());
    }

    #[test]
    fn test_url_lookup_does_not_cross_match_explicit_schemes() {
        let registry = WorkerRegistry::new();
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://strict:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        );
        registry.register(worker).unwrap();

        assert!(registry.get_by_url("grpc://strict:8080").is_none());
        assert!(registry.get_id_by_url("grpcs://strict:8080").is_none());
    }

    #[test]
    fn test_url_lookup_returns_none_for_bare_host_port_with_no_match() {
        let registry = WorkerRegistry::new();
        assert!(registry.get_by_url("nothing:8080").is_none());
        assert!(registry.get_id_by_url("nothing:8080").is_none());
        assert!(registry.remove_by_url("nothing:8080").is_none());
    }

    /// Regression for the bare-URL reservation shadowing the canonical
    /// scheme-prefixed entry. `WorkerService::create_worker` reserves an
    /// ID against `config.url` *before* `normalize_url` runs in the
    /// AddWorker workflow, so when service discovery submits bare
    /// `host:port` the registry briefly holds two `url_to_id` entries:
    /// the bare reservation (no live worker) and the canonical
    /// scheme-prefixed entry (live worker). Lookups for the bare URL
    /// must skip the reservation and resolve to the live worker.
    #[test]
    fn test_url_lookup_skips_orphan_reservation_for_bare_host_port() {
        let registry = WorkerRegistry::new();
        let _reserved = registry.reserve_id_for_url("orphan:8080");

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://orphan:8080")
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(no_health_check())
                .build(),
        );
        let live_id = registry.register(worker).unwrap();

        assert_eq!(registry.get_id_by_url("orphan:8080"), Some(live_id.clone()));
        assert!(registry
            .get_by_url("orphan:8080")
            .is_some_and(|w| w.url() == "grpc://orphan:8080"));
        assert!(registry.remove_by_url("orphan:8080").is_some());
        assert!(registry.get(&live_id).is_none());
    }

    #[test]
    fn test_url_lookup_returns_none_when_only_reservation_exists() {
        let registry = WorkerRegistry::new();
        registry.reserve_id_for_url("pending:8080");
        assert!(registry.get_by_url("pending:8080").is_none());
        assert!(registry.get_id_by_url("pending:8080").is_none());
        assert!(registry.remove_by_url("pending:8080").is_none());
    }

    /// `transition_status_inner` (the shared backend used by both
    /// `transition_status` and `transition_status_if_revision`) emits a
    /// single `WorkerEvent::StatusChanged` event for every status mutation,
    /// regardless of which target status is being installed. The mesh
    /// adapter subscribes to that event stream, so this proves Draining
    /// transitions propagate through the same path as Ready/NotReady.
    #[test]
    fn test_transition_to_draining_emits_status_changed_event() {
        let registry = WorkerRegistry::new();
        let mut rx = registry.subscribe_events();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w-drain:8080")
                .worker_type(WorkerType::Regular)
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        let worker_id = registry.register(worker.clone()).unwrap();
        let _ = rx.try_recv().unwrap();

        let result = registry.transition_status(&worker_id, WorkerStatus::Draining);
        assert_eq!(result, Some((WorkerStatus::Ready, WorkerStatus::Draining)));
        assert_eq!(worker.status(), WorkerStatus::Draining);

        match rx.try_recv().unwrap() {
            WorkerEvent::StatusChanged {
                old_status,
                new_status,
                ..
            } => {
                assert_eq!(old_status, WorkerStatus::Ready);
                assert_eq!(new_status, WorkerStatus::Draining);
            }
            other => panic!("expected StatusChanged, got {other:?}"),
        }
    }

    #[test]
    fn test_transition_status_if_revision_rejects_stale_transition() {
        let registry = WorkerRegistry::new();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        );
        let worker_id = registry.register(worker.clone()).unwrap();
        let stale_revision = worker.revision();

        let replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://w1:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .priority(99)
                .build(),
        );
        assert!(registry.replace(&worker_id, replacement));

        assert_eq!(
            registry.transition_status_if_revision(
                &worker_id,
                stale_revision,
                WorkerStatus::NotReady
            ),
            None
        );

        let current = registry.get(&worker_id).unwrap();
        assert_eq!(current.status(), WorkerStatus::Ready);
        assert_eq!(current.priority(), 99);
        assert_eq!(current.revision(), stale_revision + 1);
    }

    #[test]
    fn test_multi_model_worker_is_indexed_for_each_model() {
        let registry = WorkerRegistry::new();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.openai.com")
                .worker_type(WorkerType::Regular)
                .models(vec![
                    ModelCard::new("gpt-4o"),
                    ModelCard::new("text-embedding-3-large"),
                ])
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );

        let worker_id = registry.register(worker).unwrap();

        assert!(registry.get(&worker_id).is_some());
        assert_eq!(registry.get_by_model("gpt-4o").len(), 1);
        assert_eq!(registry.get_by_model("text-embedding-3-large").len(), 1);
        assert_eq!(
            registry.get_by_model("gpt-4o")[0].url(),
            "https://api.openai.com"
        );
        assert_eq!(
            registry.get_by_model("text-embedding-3-large")[0].url(),
            "https://api.openai.com"
        );

        let mut models = registry.get_models();
        models.sort();
        assert_eq!(
            models,
            vec!["gpt-4o".to_string(), "text-embedding-3-large".to_string()]
        );

        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 2);
    }

    #[test]
    fn test_model_alias_resolves_to_canonical_model() {
        let registry = WorkerRegistry::new();
        let aliased = worker_with_model_aliases(
            "http://aliased:8080",
            "GLM-5.2",
            &["GLM-5.2-Coding"],
            WorkerType::Prefill,
        );
        let canonical_only =
            worker_with_model_aliases("http://canonical:8080", "GLM-5.2", &[], WorkerType::Decode);

        registry.register(aliased).unwrap();
        registry.register(canonical_only).unwrap();

        let canonical_workers = registry.get_by_model("GLM-5.2");
        assert_eq!(canonical_workers.len(), 2);
        let alias_workers = registry.get_by_model("GLM-5.2-Coding");
        assert_eq!(alias_workers.len(), 2);
        assert_eq!(
            registry
                .get_workers_filtered(
                    Some("GLM-5.2-Coding"),
                    Some(WorkerType::Prefill),
                    None,
                    None,
                    false,
                )
                .len(),
            1
        );
        assert_eq!(
            registry
                .get_workers_filtered(
                    Some("GLM-5.2-Coding"),
                    Some(WorkerType::Decode),
                    None,
                    None,
                    false,
                )
                .len(),
            1
        );

        assert_eq!(
            registry.resolve_model_alias("GLM-5.2-Coding").as_deref(),
            Some("GLM-5.2")
        );
        assert!(registry.resolve_model_alias("GLM-5.2").is_none());
        assert!(registry.get_by_model("glm-5.2-coding").is_empty());
        assert_eq!(registry.get_models(), vec!["GLM-5.2"]);
        assert_eq!(registry.stats().total_models, 1);
    }

    #[test]
    fn test_model_alias_tracks_shared_workers_through_removal() {
        let registry = WorkerRegistry::new();
        let first = worker_with_model_aliases(
            "http://first:8080",
            "GLM-5.2",
            &["GLM-5.2-Coding"],
            WorkerType::Prefill,
        );
        let second = worker_with_model_aliases(
            "http://second:8080",
            "GLM-5.2",
            &["GLM-5.2-Coding"],
            WorkerType::Decode,
        );
        let first_id = registry.register(first).unwrap();
        let second_id = registry.register(second).unwrap();

        assert_eq!(registry.get_by_model("GLM-5.2-Coding").len(), 2);
        registry.remove(&first_id).unwrap();
        let remaining = registry.get_by_model("GLM-5.2-Coding");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].url(), "http://second:8080");

        registry.remove(&second_id).unwrap();
        assert!(registry.get_by_model("GLM-5.2-Coding").is_empty());
        assert!(registry.resolve_model_alias("GLM-5.2-Coding").is_none());
    }

    #[test]
    fn test_model_alias_replace_refreshes_kept_and_changed_aliases() {
        let registry = WorkerRegistry::new();
        let original: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .model(ModelCard::new("GLM-5.2").with_alias("old"))
                .build(),
        );
        let worker_id = registry.register(original).unwrap();
        let replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .model(ModelCard::new("GLM-5.2").with_alias("new"))
                .build(),
        );

        assert!(registry.replace(&worker_id, replacement));

        assert!(registry.get_by_model("old").is_empty());
        assert_eq!(registry.get_by_model("new").len(), 1);
    }

    #[test]
    fn test_contains_model_accepts_canonical_id_and_alias() {
        let registry = WorkerRegistry::new();
        let worker = worker_with_model_aliases(
            "http://worker:8080",
            "GLM-5.2",
            &["GLM-5.2-Coding"],
            WorkerType::Regular,
        );
        let worker_id = registry.register(worker).unwrap();

        assert!(registry.contains_model("GLM-5.2"));
        assert!(registry.contains_model("GLM-5.2-Coding"));
        assert!(!registry.contains_model("GLM-5.2-Unknown"));
        // The wildcard is a routing sentinel, never a registered name.
        assert!(!registry.contains_model(UNKNOWN_MODEL_ID));

        assert!(registry.remove(&worker_id).is_some());
        assert!(!registry.contains_model("GLM-5.2"));
        assert!(!registry.contains_model("GLM-5.2-Coding"));
    }

    #[test]
    fn test_model_alias_is_handed_over_when_its_owner_leaves() {
        // Two models declare the same alias. `add_model_alias` keeps the
        // first, so removing the first must not strand the second, which
        // still advertises the alias.
        let registry = WorkerRegistry::new();
        let owner = worker_with_model_aliases(
            "http://owner:8080",
            "GLM-5.2",
            &["shared-alias"],
            WorkerType::Regular,
        );
        let loser = worker_with_model_aliases(
            "http://loser:8080",
            "Qwen3",
            &["shared-alias"],
            WorkerType::Regular,
        );
        let owner_id = registry.register(owner).unwrap();
        registry.register(loser).unwrap();

        assert_eq!(
            registry.resolve_model_alias("shared-alias").as_deref(),
            Some("GLM-5.2")
        );

        assert!(registry.remove(&owner_id).is_some());

        assert_eq!(
            registry.resolve_model_alias("shared-alias").as_deref(),
            Some("Qwen3"),
            "alias must move to the remaining model that declares it"
        );
        let workers = registry.get_by_model("shared-alias");
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].url(), "http://loser:8080");
    }

    #[test]
    fn test_model_alias_never_shadows_a_registered_model_id() {
        // Registered later than the model it collides with.
        let registry = WorkerRegistry::new();
        let canonical =
            worker_with_model_aliases("http://canonical:8080", "GLM-5.2", &[], WorkerType::Regular);
        let colliding = worker_with_model_aliases(
            "http://colliding:8080",
            "Qwen3",
            &["GLM-5.2"],
            WorkerType::Regular,
        );
        let canonical_id = registry.register(canonical).unwrap();
        registry.register(colliding).unwrap();

        assert!(registry.resolve_model_alias("GLM-5.2").is_none());
        // The real model has gone, so the name must resolve to nothing at all
        // rather than start pointing at Qwen3.
        assert!(registry.remove(&canonical_id).is_some());
        assert!(registry.resolve_model_alias("GLM-5.2").is_none());
        assert!(registry.get_by_model("GLM-5.2").is_empty());
    }

    #[test]
    fn test_registered_model_id_evicts_a_colliding_alias() {
        // Registered earlier than the model it collides with: same invariant,
        // resolved from the other side.
        let registry = WorkerRegistry::new();
        let colliding = worker_with_model_aliases(
            "http://colliding:8080",
            "Qwen3",
            &["GLM-5.2"],
            WorkerType::Regular,
        );
        registry.register(colliding).unwrap();
        assert_eq!(
            registry.resolve_model_alias("GLM-5.2").as_deref(),
            Some("Qwen3")
        );

        let canonical =
            worker_with_model_aliases("http://canonical:8080", "GLM-5.2", &[], WorkerType::Regular);
        let canonical_id = registry.register(canonical).unwrap();

        assert!(registry.resolve_model_alias("GLM-5.2").is_none());
        assert_eq!(registry.get_by_model("GLM-5.2").len(), 1);
        assert_eq!(
            registry.get_by_model("GLM-5.2")[0].url(),
            "http://canonical:8080"
        );

        assert!(registry.remove(&canonical_id).is_some());
        assert!(registry.get_by_model("GLM-5.2").is_empty());
    }

    #[test]
    fn test_replace_same_url_refreshes_all_model_indexes() {
        let registry = WorkerRegistry::new();

        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.openai.com")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("gpt-4o")])
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );
        let second: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.openai.com")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("o3"), ModelCard::new("o4-mini")])
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );

        // First registration creates the worker
        let first_id = registry.register(first).unwrap();

        // Second registration with same URL should be rejected
        assert!(registry.register(second.clone()).is_none());

        // Use replace() to update the worker
        assert!(registry.replace(&first_id, second));

        assert_eq!(registry.len(), 1);
        assert!(registry.get_by_model("gpt-4o").is_empty());
        assert_eq!(registry.get_by_model("o3").len(), 1);
        assert_eq!(registry.get_by_model("o4-mini").len(), 1);
        assert_eq!(registry.get_by_type(WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(ConnectionMode::Http).len(), 1);

        let mut models = registry.get_models();
        models.sort();
        assert_eq!(models, vec!["o3".to_string(), "o4-mini".to_string()]);
    }

    #[test]
    fn test_register_or_replace_upsert() {
        let registry = WorkerRegistry::new();

        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.openai.com")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("gpt-4o")])
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );
        let second: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("https://api.openai.com")
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("o3"), ModelCard::new("o4-mini")])
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .build(),
        );

        let first_id = registry.register_or_replace(first);
        let second_id = registry.register_or_replace(second);

        // Same URL → same ID (upsert)
        assert_eq!(first_id, second_id);
        assert_eq!(registry.len(), 1);
        // Old model gone, new models present
        assert!(registry.get_by_model("gpt-4o").is_empty());
        assert_eq!(registry.get_by_model("o3").len(), 1);
        assert_eq!(registry.get_by_model("o4-mini").len(), 1);
    }

    #[test]
    fn test_builder_status_override_on_replace() {
        // Regression test: metadata-only updates must not kick a healthy
        // worker back to Pending. The builder exposes a `.status()` setter
        // so callers (e.g. UpdateWorkerPropertiesStep) can pass the old
        // worker's status when constructing the replacement.
        let registry = WorkerRegistry::new();

        // First worker starts Pending (health checks enabled by default),
        // then gets promoted to Ready (simulating what the health checker does).
        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("llama-3"))
                .build(),
        );
        assert_eq!(first.status(), WorkerStatus::Pending);
        first.set_status(WorkerStatus::Ready);

        let first_id = registry.register(first.clone()).unwrap();
        assert_eq!(
            registry.get(&first_id).unwrap().status(),
            WorkerStatus::Ready
        );

        // Caller (e.g. UpdateWorkerPropertiesStep) reads the old status and
        // passes it to the builder. The builder honors the override instead
        // of defaulting to Pending.
        let preserved_status = first.status();
        let second: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("llama-3"))
                .priority(99)
                .status(preserved_status)
                .build(),
        );
        assert_eq!(
            second.status(),
            WorkerStatus::Ready,
            "builder must honor explicit status override"
        );

        assert!(registry.replace(&first_id, second));

        let after = registry.get(&first_id).unwrap();
        assert_eq!(after.status(), WorkerStatus::Ready);
        assert_eq!(after.priority(), 99, "new priority should be applied");
    }

    #[test]
    fn test_replace_preserves_runtime_state_and_circuit_breaker() {
        let registry = WorkerRegistry::new();
        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", "sticky-key".parse().unwrap());

        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        );
        let first_id = registry.register(first.clone()).unwrap();
        let initial_revision = first.revision();

        first.set_status(WorkerStatus::NotReady);
        first.increment_processed();
        let load_guard = WorkerLoadGuard::new(first.clone(), Some(&headers));

        for _ in 0..5 {
            first.record_outcome(503);
        }
        assert_eq!(first.circuit_breaker_state(), CircuitState::Open);

        let replacement: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker:8080")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .priority(99)
                .build(),
        );
        assert!(registry.replace(&first_id, replacement));

        let current = registry.get(&first_id).unwrap();
        assert_eq!(current.priority(), 99);
        assert_eq!(current.status(), WorkerStatus::NotReady);
        assert_eq!(current.load(), 1);
        assert_eq!(current.routing_key_load(), 1);
        assert_eq!(current.processed_requests(), 1);
        assert_eq!(current.circuit_breaker_state(), CircuitState::Open);
        assert_eq!(current.revision(), initial_revision + 1);

        first.increment_processed();
        assert_eq!(current.processed_requests(), 2);

        drop(load_guard);
        assert_eq!(current.load(), 0);
        assert_eq!(current.routing_key_load(), 0);
    }

    #[test]
    fn test_builder_default_status_is_pending() {
        // Without an explicit override, health-checked workers start Pending.
        let worker = BasicWorkerBuilder::new("http://worker:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(worker.status(), WorkerStatus::Pending);
    }

    #[test]
    fn test_register_rejects_duplicate_url() {
        let registry = WorkerRegistry::new();

        let first: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        let second: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert!(registry.register(first).is_some());
        assert!(registry.register(second).is_none());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_model_retry_config_last_write_wins() {
        let registry = WorkerRegistry::new();

        assert!(registry.get_retry_config("llama-3").is_none());

        let config1 = RetryConfig {
            max_retries: 3,
            ..RetryConfig::default()
        };
        registry.set_model_retry_config("llama-3", config1, true);

        let stored = registry.get_retry_config("llama-3").unwrap();
        assert_eq!(stored.max_retries, 3);

        // Last write wins — overwrite with different config
        let config2 = RetryConfig {
            max_retries: 10,
            initial_backoff_ms: 200,
            ..RetryConfig::default()
        };
        registry.set_model_retry_config("llama-3", config2, true);

        let stored = registry.get_retry_config("llama-3").unwrap();
        assert_eq!(stored.max_retries, 10);
        assert_eq!(stored.initial_backoff_ms, 200);

        // Disable retries — max_retries should be set to 1
        let config3 = RetryConfig {
            max_retries: 10,
            ..RetryConfig::default()
        };
        registry.set_model_retry_config("llama-3", config3, false);

        let stored = registry.get_retry_config("llama-3").unwrap();
        assert_eq!(stored.max_retries, 1); // disabled → max_retries = 1
    }

    #[test]
    fn test_model_retry_config_cleanup_on_last_worker_removal() {
        let registry = WorkerRegistry::new();

        let worker1 = Arc::new(
            BasicWorkerBuilder::new("http://worker1:8080")
                .model(ModelCard::new("llama-3"))
                .build(),
        ) as Arc<dyn Worker>;

        let worker2 = Arc::new(
            BasicWorkerBuilder::new("http://worker2:8080")
                .model(ModelCard::new("llama-3"))
                .build(),
        ) as Arc<dyn Worker>;

        let id1 = registry.register(worker1).unwrap();
        let id2 = registry.register(worker2).unwrap();

        registry.set_model_retry_config(
            "llama-3",
            RetryConfig {
                max_retries: 5,
                ..RetryConfig::default()
            },
            true,
        );
        assert!(registry.get_retry_config("llama-3").is_some());

        // Remove first worker — config should still exist
        registry.remove(&id1);
        assert!(registry.get_retry_config("llama-3").is_some());

        // Remove last worker — config should be cleaned up
        registry.remove(&id2);
        assert!(registry.get_retry_config("llama-3").is_none());
    }

    #[test]
    fn test_sampling_defaults_group_warning_scan_tracks_distinct_values() {
        let registry = WorkerRegistry::new();
        let defaults_a = r#"{"temperature":0.6}"#;
        let defaults_b = r#"{"temperature":0.7}"#;

        let id1 = registry
            .register(worker_with_sampling_defaults(
                "http://worker1:8080",
                "llama-3",
                WorkerType::Regular,
                ConnectionMode::Grpc,
                Some(defaults_a),
            ))
            .unwrap();
        let id2 = registry
            .register(worker_with_sampling_defaults(
                "http://worker2:8080",
                "llama-3",
                WorkerType::Regular,
                ConnectionMode::Grpc,
                Some(defaults_a),
            ))
            .unwrap();

        assert_sampling_defaults_group_values(
            &registry,
            "llama-3",
            WorkerType::Regular,
            &[defaults_a],
        );

        let id3 = registry
            .register(worker_with_sampling_defaults(
                "http://worker3:8080",
                "llama-3",
                WorkerType::Regular,
                ConnectionMode::Grpc,
                Some(defaults_b),
            ))
            .unwrap();

        assert_sampling_defaults_group_values(
            &registry,
            "llama-3",
            WorkerType::Regular,
            &[defaults_a, defaults_b],
        );

        registry.remove(&id3);
        assert_sampling_defaults_group_values(
            &registry,
            "llama-3",
            WorkerType::Regular,
            &[defaults_a],
        );

        registry.remove(&id1);
        registry.remove(&id2);
        assert_sampling_defaults_group_values(&registry, "llama-3", WorkerType::Regular, &[]);
    }

    #[test]
    fn test_sampling_defaults_group_warning_scan_updates_on_replace() {
        let registry = WorkerRegistry::new();
        let defaults_a = r#"{"temperature":0.6}"#;
        let defaults_b = r#"{"temperature":0.7}"#;

        let id = registry
            .register(worker_with_sampling_defaults(
                "http://worker:8080",
                "llama-3",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                Some(defaults_a),
            ))
            .unwrap();

        assert!(registry.replace(
            &id,
            worker_with_sampling_defaults(
                "http://worker:8080",
                "llama-3",
                WorkerType::Decode,
                ConnectionMode::Grpc,
                Some(defaults_b),
            ),
        ));

        assert_sampling_defaults_group_values(
            &registry,
            "llama-3",
            WorkerType::Decode,
            &[defaults_b],
        );
    }

    #[test]
    fn test_sampling_defaults_group_warning_scan_ignores_non_applicable_workers() {
        let registry = WorkerRegistry::new();
        let defaults = r#"{"temperature":0.6}"#;

        registry.register(worker_with_sampling_defaults(
            "http://prefill:8080",
            "llama-3",
            WorkerType::Prefill,
            ConnectionMode::Grpc,
            Some(defaults),
        ));
        registry.register(worker_with_sampling_defaults(
            "http://http-worker:8080",
            "llama-3",
            WorkerType::Regular,
            ConnectionMode::Http,
            Some(defaults),
        ));
        registry.register(worker_with_sampling_defaults(
            "http://missing-label:8080",
            "llama-3",
            WorkerType::Regular,
            ConnectionMode::Grpc,
            None,
        ));

        assert_sampling_defaults_group_values(&registry, "llama-3", WorkerType::Regular, &[]);
    }

    #[test]
    fn test_worker_event_broadcast() {
        let registry = WorkerRegistry::new();
        let mut rx = registry.subscribe_events();

        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "test-model".to_string());

        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://event-worker:8080")
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .circuit_breaker_config(CircuitBreakerConfig::default())
                .api_key("test_api_key")
                .build(),
        );

        let worker_id = registry.register(Arc::from(worker)).unwrap();

        // Should receive Registered event
        let event = rx.try_recv().unwrap();
        match event {
            WorkerEvent::Registered { worker, .. } => {
                assert_eq!(worker.url(), "http://event-worker:8080");
            }
            other => panic!("Expected Registered event, got: {other:?}"),
        }

        registry.remove(&worker_id);

        // Should receive Removed event
        let event = rx.try_recv().unwrap();
        match event {
            WorkerEvent::Removed { worker, .. } => {
                assert_eq!(worker.url(), "http://event-worker:8080");
            }
            other => panic!("Expected Removed event, got: {other:?}"),
        }
    }

    fn worker_serving(url: &str, model_ids: &[&str]) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .models(
                    model_ids
                        .iter()
                        .map(|id| ModelCard::new(*id))
                        .collect::<Vec<_>>(),
                )
                .health_config(no_health_check())
                .build(),
        )
    }

    #[test]
    fn test_wildcard_hash_ring_matches_the_only_model() {
        let registry = WorkerRegistry::new();
        registry
            .register(worker_serving("http://w1:8080", &["llama-3"]))
            .unwrap();
        registry
            .register(worker_serving("http://w2:8080", &["llama-3"]))
            .unwrap();

        let wildcard = registry
            .get_hash_ring(UNKNOWN_MODEL_ID)
            .expect("requests naming no model need a ring");
        let model_ring = registry.get_hash_ring("llama-3").expect("per-model ring");

        assert_eq!(wildcard.worker_count(), 2);
        for key in ["alpha", "beta", "gamma"] {
            assert_eq!(
                wildcard.find_healthy_url(key, |_| true),
                model_ring.find_healthy_url(key, |_| true),
                "wildcard and single-model rings must agree on {key}"
            );
        }
    }

    #[test]
    fn test_wildcard_hash_ring_unions_models() {
        let registry = WorkerRegistry::new();
        registry
            .register(worker_serving("http://w1:8080", &["llama-3"]))
            .unwrap();
        registry
            .register(worker_serving("http://w2:8080", &["gpt-4"]))
            .unwrap();

        let wildcard = registry.get_hash_ring(UNKNOWN_MODEL_ID).expect("ring");
        assert_eq!(wildcard.worker_count(), 2);
        assert_eq!(
            wildcard.find_healthy_url("key", |url| url == "http://w1:8080"),
            Some("http://w1:8080")
        );
        assert_eq!(
            wildcard.find_healthy_url("key", |url| url == "http://w2:8080"),
            Some("http://w2:8080")
        );

        // Per-model rings stay scoped to their own workers.
        assert_eq!(
            registry
                .get_hash_ring("llama-3")
                .expect("ring")
                .worker_count(),
            1
        );
    }

    #[test]
    fn test_wildcard_hash_ring_weights_multi_model_worker_once() {
        let registry = WorkerRegistry::new();
        registry
            .register(worker_serving("http://w1:8080", &["llama-3", "gpt-4"]))
            .unwrap();
        registry
            .register(worker_serving("http://w2:8080", &["gpt-4"]))
            .unwrap();

        let wildcard = registry.get_hash_ring(UNKNOWN_MODEL_ID).expect("ring");
        assert_eq!(
            wildcard.worker_count(),
            2,
            "a worker serving two models must not take a double share of the ring"
        );
    }

    #[test]
    fn test_wildcard_hash_ring_follows_removals() {
        let registry = WorkerRegistry::new();
        registry
            .register(worker_serving("http://w1:8080", &["llama-3"]))
            .unwrap();
        registry
            .register(worker_serving("http://w2:8080", &["gpt-4"]))
            .unwrap();

        registry.remove_by_url("http://w2:8080");
        let wildcard = registry.get_hash_ring(UNKNOWN_MODEL_ID).expect("ring");
        assert_eq!(wildcard.worker_count(), 1);
        assert_eq!(
            wildcard.find_healthy_url("key", |_| true),
            Some("http://w1:8080")
        );

        registry.remove_by_url("http://w1:8080");
        assert!(
            registry.get_hash_ring(UNKNOWN_MODEL_ID).is_none(),
            "an empty registry has no ring to route against"
        );
    }
}
