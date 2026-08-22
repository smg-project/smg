/*
    Cache-Aware Load Balancing Router

    When load is balanced, uses cache-aware routing. When imbalanced, uses
    shortest-queue. A system is imbalanced when both:
        (max - min) > abs_threshold  AND  max > rel_threshold * min

    Three types of cache-aware routing (mutually exclusive, selected by
    worker connection mode and KV event availability):

    1. Event-Driven (gRPC + KV events)
    -------------------------------------------
    Uses PositionalIndexer overlap scoring from KvEventMonitor. Routes based
    on actual backend KV cache state. Selects the worker with the highest
    overlap count; tie-breaks by load (lower) then tree size (smaller).
    Falls back to min-load when no cache overlap exists.

    2. Approximate Token Tree (gRPC, no KV events)
    -------------------------------------------
    Maintains a TokenTree per model tracking which token prefixes were routed
    where. If match_rate > cache_threshold, routes to the best-matching worker.
    Otherwise routes to the worker with the smallest tree (most cache capacity).

    3. Approximate String Tree (HTTP)
    -------------------------------------------
    Same algorithm as (2) but operates on raw text characters instead of
    token IDs, avoiding tokenization overhead.

    Load Balancing (Shortest Queue)
    -------------------------------------------
    When the system is imbalanced, routes to the least busy worker regardless
    of cache affinity.

    Hash Index Under-Layer (cache_index = hash)
    -------------------------------------------
    Replaces all three tree modes with a TTL'd exact-match placement map
    keyed on request heads at the cache_boundaries token positions; the
    radix trees are neither consulted nor populated. Selection probes
    boundaries deepest-first for a live holder and records the dispatched
    worker at every applicable boundary.

    Configuration Parameters:
    ------------------------
    cache_threshold:         Min prefix match ratio for highest-match routing (0.0-1.0)
    balance_abs_threshold:   Absolute load diff threshold for imbalance detection
    balance_rel_threshold:   Relative load ratio threshold for imbalance detection
    eviction_interval_secs:  Interval between LRU eviction / TTL sweep cycles
    max_tree_size:           Max total size (chars/tokens) of each model's approximate tree,
                             shared across all workers; enforced by eviction
    block_size:              Backend KV cache block size for event-driven routing
    cache_index:             Under-layer: tree (radix trees) or hash (placement map)
    cache_ttl_secs:          Seconds a hash-index placement stays routable
    cache_boundaries:        Ascending token positions for hash-index keying
*/

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use dashmap::DashMap;
use kv_index::{compute_request_content_hashes, PositionalIndexer, TenantId, TokenTree, Tree};
use openai_protocol::worker::WorkerLoadResponse;
use parking_lot::RwLock;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tracing::{debug, warn};

use super::{
    normalize_model_key, utils::PeriodicTask, CacheAwareConfig, LoadBalancingPolicy,
    SelectWorkerInfo,
};
use crate::{
    config::CacheIndexKind,
    mesh::adapters::tree_sync::{RepairEntry, TreeDelta, TreeRepairPage, TreeSyncAdapter},
    observability::metrics::Metrics,
    worker::{KvEventMonitor, Worker},
};

/// Latest per-worker backend load snapshot stream, keyed by worker URL.
pub(crate) type LoadReceiver = watch::Receiver<HashMap<String, WorkerLoadResponse>>;

/// Hint about the uncached prefill portion from a partial cache match that
/// fell below `cache_threshold`. Lets the no-cache strategy classify by
/// actual prefill work instead of the full request size.
#[derive(Debug, Clone, Copy)]
pub(crate) enum UncachedHint {
    /// Uncached token count (gRPC / token-tree path).
    Tokens(usize),
    /// Uncached character count (HTTP / string-tree path); the strategy
    /// converts to tokens via its `chars_per_token` setting.
    Chars(usize),
}

/// Strategy for the no-cache branch: when a request does not hit the
/// cache tree (or KV events / hash index), this trait selects which worker
/// receives the request. The default behavior (no strategy set) routes to
/// the least-loaded healthy worker (`min_load_idx`). `CacheAwareLengthPolicy`
/// injects a strategy that splits workers into long/short pools by
/// uncached prefill tokens.
pub(crate) trait NoCacheStrategy: Send + Sync + std::fmt::Debug {
    /// Select a worker for the no-cache (miss) branch. `min_load_idx` is
    /// the pre-computed least-loaded healthy worker the caller would use
    /// by default; the strategy may return it or a pool-selected alternative.
    ///
    /// `uncached_hint` carries the estimated uncached-prefill size when the
    /// call originates from a partial cache match that fell below
    /// `cache_threshold`. It is `None` when no matching was attempted (e.g.
    /// imbalanced fallback, event-driven no-overlap, hash-path fallback), in
    /// which case the strategy should estimate from `info` as before.
    fn select_no_cache(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        avg_load: f64,
        model_id: &str,
        uncached_hint: Option<UncachedHint>,
    ) -> Option<usize>;
}

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to shortest-queue routing when load is imbalanced.
/// Maintains separate trees per model for multi-model support.
/// Supports mesh synchronization of tree operations across cluster nodes.
/// When mesh is not enabled, the policy works independently without synchronization.
///
/// Supports both HTTP (string-based) and gRPC (token-based) connections:
/// - HTTP requests use StringTree (character-based prefix matching)
/// - gRPC requests use TokenTree (token-based prefix matching, page-aligned)
#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: CacheAwareConfig,
    /// String-based trees for HTTP connections (text input)
    string_trees: Arc<DashMap<String, Arc<Tree>>>,
    /// Token-based trees for gRPC connections (pre-tokenized input)
    token_trees: Arc<DashMap<String, Arc<TokenTree>>>,
    _eviction_task: Option<PeriodicTask>,
    /// Event-driven KV cache monitor for overlap scoring (gRPC workers only).
    kv_monitor: RwLock<Option<Arc<KvEventMonitor>>>,
    /// Latest per-worker backend load snapshot (keyed by worker URL) from the
    /// `WorkerMonitor` load poll. Read on the hot path for the KV-usage imbalance
    /// trigger. `None` until wired by the registry (then the policy stays
    /// count-only, preserving current behavior).
    load_rx: RwLock<Option<LoadReceiver>>,
    /// Model-scoped hash indexes for resolving tenant delta hashes.
    /// Outer key is the normalized model_id; inner maps hold
    /// `hash → reconstructable prefix/tokens` per tree kind.
    /// Spec §7.1 mandates model scoping: the same hash can refer
    /// to different prefixes in different models, so a global
    /// index mis-routes multi-model deployments. Bounded by
    /// eviction at `max_tree_size` total entries.
    ///
    /// Per-entry value semantics differ by populate site:
    /// - `select_worker_*` (request hot paths) store the prior
    ///   shared prefix from a pre-insert match. Bytes/entry is
    ///   bounded by tree depth, not input size — a 32K-token
    ///   request costs O(matched-prefix), not O(input).
    /// - `apply_repair_page` (cold-start replay) stores the full
    ///   inserted path because the canonical path is required to
    ///   attach remote tenants at the correct node. This path
    ///   runs at replay frequency, not request rate.
    hash_index: Arc<DashMap<String, PerModelHashIndex>>,
    /// Gate request-hot-path `hash_index` writes. The index's only
    /// consumers are mesh paths (`apply_known_remote_insert` reads,
    /// `apply_repair_page` writes). When mesh is disabled the
    /// hot-path writes accumulate with no reader and OOM the
    /// gateway. Off by default; the mesh wiring code flips it on
    /// when it attaches.
    populate_hash_index: AtomicBool,
    /// Outbound bridge into the mesh `td:` broadcast namespace.
    /// `Some` after [`Self::set_mesh_tree_sync`] (called by mesh wiring
    /// at startup); `None` when mesh is disabled, in which case
    /// `sync_local_insert` is a no-op. The setter also toggles
    /// [`Self::populate_hash_index`] to match adapter presence so the
    /// two never drift apart. Note the pairing is best-effort at a
    /// point-in-time — later eviction of a hash-index entry can leave
    /// a still-in-flight delta with no local resolution; peers that
    /// repair against us will simply see the gap the next tick.
    mesh_tree_sync: RwLock<Option<Arc<TreeSyncAdapter>>>,
    /// Hash-mode placement index (`cache_index = hash`): model →
    /// (boundary, head hash) → live holders. Empty in tree mode.
    /// Inner maps are Arc'd and model entries are never removed, so a
    /// cloned inner handle stays canonical and walking it never holds
    /// an outer-shard guard.
    placement_index: Arc<DashMap<String, Arc<PlacementMap>>>,
    /// Optional no-cache strategy. When `None`, the no-cache fallback
    /// routes to `min_load_idx` (the default behavior). When `Some`,
    /// the strategy selects the worker instead — used by
    /// `CacheAwareLengthPolicy` for long/short pool split.
    no_cache_strategy: Option<Arc<dyn NoCacheStrategy>>,
}

/// Hash-mode per-model placement map: (boundary position, xxh3 of the token
/// head up to that boundary) → workers recently routed that exact head.
type PlacementMap = DashMap<(usize, u64), Vec<PlacementHolder>>;

/// One worker's most recent dispatch of a head; live while `last_touch`
/// is within `cache_ttl_secs`.
#[derive(Debug, Clone)]
struct PlacementHolder {
    worker_url: String,
    last_touch: Instant,
}

/// Max workers remembered per (boundary, head) key; recording a fourth
/// evicts the stalest.
const PLACEMENT_HOLDER_CAP: usize = 3;

fn hash_token_head(head: &[u32]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(head))
}

/// Per-model inner container for [`CacheAwarePolicy::hash_index`].
/// Keeping both kinds in one struct per model makes the
/// "separate model-scoped hash indexes for string and token
/// trees" invariant from spec §7.1 explicit in the type.
#[derive(Debug, Default)]
struct PerModelHashIndex {
    /// path hash → matched prefix (reconstructs the string-tree node).
    string_tree: DashMap<u64, String>,
    /// token-path hash → tokens (reconstructs the token-tree node).
    token_tree: DashMap<u64, Vec<u32>>,
}

/// Total cached characters across tenants and the tenant count for one
/// model's string tree. O(tenants): sums the tree's maintained counters.
fn string_tree_totals(tree: &Tree) -> (usize, usize) {
    let counts = tree.get_tenant_char_count();
    (counts.values().sum(), counts.len())
}

/// Total cached tokens across tenants and the tenant count for one
/// model's token tree. O(tenants): sums the tree's maintained counters.
fn token_tree_totals(tree: &TokenTree) -> (usize, usize) {
    let counts = tree.get_tenant_token_counts();
    (counts.values().sum(), counts.len())
}

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(mut config: CacheAwareConfig) -> Self {
        // Deepest-first probing assumes sorted, deduped, non-zero boundaries.
        config.cache_boundaries.retain(|&p| p > 0);
        config.cache_boundaries.sort_unstable();
        config.cache_boundaries.dedup();

        let string_trees = Arc::new(DashMap::<String, Arc<Tree>>::new());
        let token_trees = Arc::new(DashMap::<String, Arc<TokenTree>>::new());
        let hash_index = Arc::new(DashMap::<String, PerModelHashIndex>::new());
        let placement_index = Arc::new(DashMap::<String, Arc<PlacementMap>>::new());

        // Start background eviction thread if configured
        let eviction_task = if config.cache_index == CacheIndexKind::Hash {
            (config.eviction_interval_secs > 0).then(|| {
                let placement_clone = Arc::clone(&placement_index);
                let ttl = Duration::from_secs(config.cache_ttl_secs);
                PeriodicTask::spawn(config.eviction_interval_secs, "PlacementSweep", move || {
                    Self::sweep_placement_index(&placement_clone, ttl, Instant::now());
                })
            })
        } else if config.eviction_interval_secs > 0 {
            let string_trees_clone = Arc::clone(&string_trees);
            let token_trees_clone = Arc::clone(&token_trees);
            let hash_index_clone = Arc::clone(&hash_index);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "Eviction",
                move || {
                    // Evict string trees (HTTP)
                    let mut total_chars: usize = 0;
                    for tree_ref in string_trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        let (chars, tenants) = string_tree_totals(tree);
                        total_chars += chars;
                        Metrics::set_cache_tree_chars(model_id, chars);
                        Metrics::set_cache_tree_tenants(model_id, "string", tenants);

                        debug!(
                            "String tree eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                    // Evict token trees (gRPC)
                    let mut total_tokens: usize = 0;
                    for tree_ref in token_trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        let (tokens, tenants) = token_tree_totals(tree);
                        total_tokens += tokens;
                        Metrics::set_cache_tree_tokens(model_id, tokens);
                        Metrics::set_cache_tree_tenants(model_id, "token", tenants);

                        debug!(
                            "Token tree eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                    // Evict hash index per model: `max_tree_size` is a
                    // per-tree bound, so clearing one model's overflow
                    // must not wipe other models' still-valid metadata.
                    // Each tree kind is checked independently.
                    let mut hash_total: usize = 0;
                    for entry in hash_index_clone.iter() {
                        let per_model = entry.value();
                        if per_model.string_tree.len() > max_tree_size {
                            per_model.string_tree.clear();
                            debug!(
                                model_id = entry.key(),
                                "String hash index cleared (exceeded max_tree_size: {})",
                                max_tree_size
                            );
                        }
                        if per_model.token_tree.len() > max_tree_size {
                            per_model.token_tree.clear();
                            debug!(
                                model_id = entry.key(),
                                "Token hash index cleared (exceeded max_tree_size: {})",
                                max_tree_size
                            );
                        }
                        hash_total += per_model.string_tree.len() + per_model.token_tree.len();
                    }

                    // Log tree sizes — model counts, aggregate sizes +
                    // hash-index total, from the per-tenant counters.
                    // DO NOT call tree.snapshot() here — it clones all
                    // edge text (~170 MB) every cycle.
                    tracing::info!(
                        "Tree memory: string_trees={} models / {} chars, \
                         token_trees={} models / {} tokens, \
                         hash_index={} models / {} entries",
                        string_trees_clone.len(),
                        total_chars,
                        token_trees_clone.len(),
                        total_tokens,
                        hash_index_clone.len(),
                        hash_total,
                    );
                },
            ))
        } else {
            None
        };

        Self {
            config,
            string_trees,
            token_trees,
            _eviction_task: eviction_task,
            kv_monitor: RwLock::new(None),
            load_rx: RwLock::new(None),
            hash_index,
            populate_hash_index: AtomicBool::new(false),
            mesh_tree_sync: RwLock::new(None),
            placement_index,
            no_cache_strategy: None,
        }
    }

    /// Attach a no-cache strategy, enabling custom worker selection on the
    /// cache-miss branch. Used by `CacheAwareLengthPolicy` for long/short
    /// pool split. Returns `self` for chaining.
    pub(crate) fn with_no_cache_strategy(mut self, strategy: Arc<dyn NoCacheStrategy>) -> Self {
        self.no_cache_strategy = Some(strategy);
        self
    }

    /// Resolve the no-cache branch: if a strategy is attached, delegate to
    /// it; otherwise fall back to `min_load_idx` (the default behavior).
    /// Callers still own tree update + `increment_processed` for the
    /// returned index.
    ///
    /// `uncached_hint` is the estimated uncached-prefill token count from a
    /// partial cache match (below `cache_threshold`); `None` when no matching
    /// was attempted.
    fn resolve_no_cache(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        avg_load: f64,
        model_id: &str,
        uncached_hint: Option<UncachedHint>,
    ) -> Option<usize> {
        match &self.no_cache_strategy {
            Some(strategy) => strategy.select_no_cache(
                workers,
                info,
                healthy_indices,
                min_load_idx,
                avg_load,
                model_id,
                uncached_hint,
            ),
            None => min_load_idx,
        }
    }

    /// Enable request-hot-path `hash_index` population without attaching
    /// an adapter. Only exists so unit tests can seed the populate flag
    /// without the ceremony of wiring in a real [`TreeSyncAdapter`];
    /// production code goes through [`Self::set_mesh_tree_sync`], which
    /// flips both fields together.
    #[cfg(test)]
    fn set_populate_hash_index(&self, enabled: bool) {
        self.populate_hash_index.store(enabled, Ordering::Relaxed);
    }

    /// Token tree sized to the backend's KV page (`block_size`): affinity
    /// below one backend page is unusable by the engine.
    fn new_token_tree(&self) -> TokenTree {
        TokenTree::with_config(self.config.block_size.max(1), Default::default())
    }

    fn should_populate_hash_index(&self) -> bool {
        self.populate_hash_index.load(Ordering::Relaxed)
    }

    /// Test-only view of the effective config so registry tests can
    /// assert operator tunables propagated.
    #[cfg(test)]
    pub(crate) fn config_for_test(&self) -> &CacheAwareConfig {
        &self.config
    }

    /// Test-only: whether a KV event monitor is attached, so registry
    /// tests can assert injection at publication.
    #[cfg(test)]
    pub(crate) fn kv_event_monitor_is_set_for_test(&self) -> bool {
        self.kv_monitor.read().is_some()
    }

    /// Test-only view onto the populate flag so integration tests
    /// outside this file can assert wiring flipped it. Not part of
    /// the public API.
    #[cfg(test)]
    pub fn should_populate_hash_index_for_test(&self) -> bool {
        self.should_populate_hash_index()
    }

    /// Test-only: flip populate on without going through the mesh
    /// wiring path. Used by bridge tests that need to seed
    /// `hash_index` directly.
    #[cfg(test)]
    pub fn set_populate_hash_index_for_test_true(&self) {
        self.set_populate_hash_index(true);
    }

    /// Test-only: seed a single hash-index entry so bridge tests
    /// can exercise the inbound resolution path without driving a
    /// full request through `select_worker`. `matched` is the
    /// matched-prefix shape the populate site would normally store
    /// (full text for string / full token vec for token) — for a
    /// unit test that only asserts the lookup succeeded, any
    /// non-empty value works because the underlying tree seeds
    /// itself in `apply_known_remote_insert`.
    #[cfg(test)]
    pub fn seed_hash_index_for_test(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
        node_hash: u64,
        matched: &str,
    ) {
        let entry = self.hash_index.entry(model_id.to_string()).or_default();
        match tree_kind {
            TreeKind::String => {
                entry.string_tree.insert(node_hash, matched.to_string());
                // Ensure the string_tree map has a matching tree so
                // apply_known_remote_insert doesn't hit the
                // populate-site invariant warning.
                self.string_trees
                    .entry(model_id.to_string())
                    .or_insert_with(|| Arc::new(Tree::new()));
            }
            TreeKind::Token => {
                entry
                    .token_tree
                    .insert(node_hash, matched.bytes().map(u32::from).collect());
                self.token_trees
                    .entry(model_id.to_string())
                    .or_insert_with(|| Arc::new(self.new_token_tree()));
            }
        }
    }

    /// Attach the mesh outbound bridge and enable hash-index population
    /// in one atomic step; pass `None` to detach and disable both. The
    /// pair moves together because the hash-index has no non-mesh
    /// readers — enabling population without an adapter attached would
    /// waste memory, and the producer-side `sync_local_insert` calls
    /// only fire while population is on.
    ///
    /// Interior-mutability setter so it composes with policies stored
    /// behind `Arc<dyn LoadBalancingPolicy>` after construction, matching
    /// `set_kv_event_monitor` / `set_load_receiver`.
    pub fn set_mesh_tree_sync(&self, adapter: Option<Arc<TreeSyncAdapter>>) {
        let populate = adapter.is_some();
        let mut guard = self.mesh_tree_sync.write();
        *guard = adapter;
        // Store under the guard so no observer can see the pair
        // split (adapter attached ↔ populate flag on).
        self.populate_hash_index.store(populate, Ordering::Relaxed);
    }

    /// Publish one local tree change to the mesh outbound buffer.
    /// No-op when no adapter is attached — cheap check on the hot path.
    /// The `Arc` is cloned out before invoking `on_local_insert` so the
    /// adapter callback never runs under our read lock (avoids a future
    /// deadlock if the adapter path ever wants to write back into any
    /// policy state).
    fn sync_local_insert(&self, model_id: &str, delta: TreeDelta) {
        let adapter = self.mesh_tree_sync.read().as_ref().map(Arc::clone);
        if let Some(adapter) = adapter {
            adapter.on_local_insert(model_id, delta);
        }
    }

    /// Set event-driven KV cache monitor (thread-safe, can be called after construction).
    /// Uses interior mutability so this works on policies behind `Arc<dyn LoadBalancingPolicy>`.
    pub fn set_kv_event_monitor(&self, monitor: Option<Arc<KvEventMonitor>>) {
        *self.kv_monitor.write() = monitor;
    }

    /// Set the backend load-snapshot receiver (thread-safe, after construction).
    /// Wired from the `WorkerMonitor` via the `PolicyRegistry` so the KV-usage
    /// imbalance trigger can read fresh per-worker `token_usage`.
    pub fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        *self.load_rx.write() = rx;
    }

    #[cfg(test)]
    pub(crate) fn has_load_receiver_for_test(&self) -> bool {
        self.load_rx.read().is_some()
    }

    /// True when backend KV pressure demands abandoning cache affinity.
    ///
    /// Two triggers, OR'd together, both requiring a backend `token_usage`
    /// snapshot and disabled at their `1.0` default (utilization and spread
    /// are both `<= 1.0`, so `> 1.0` never fires):
    ///
    /// - **overload** (`overload_token_usage_threshold`): the hottest engine's
    ///   KV utilization exceeds the ceiling — a critically-saturated engine,
    ///   shed regardless of balance. Set high (e.g. 0.9) as a safety valve.
    /// - **KV spread** (`balance_token_usage_threshold`): the hottest engine is
    ///   materially more KV-saturated than the coldest, i.e. a cooler engine
    ///   exists to spill toward. This is the true balance signal for long-context
    ///   workloads, and — unlike request counts, which each gateway sees only
    ///   locally — it is invariant to the number of gateway replicas.
    ///
    /// Request-count dispersion is deliberately NOT a trigger here: a global
    /// spread check is either noise-triggered (small thresholds fire on
    /// steady-state variance and disable affinity outright) or blind (large
    /// thresholds admit a single deep queue sitting under them). Count
    /// pressure is instead applied per request, to the selected candidate,
    /// in [`Self::gate_selected_candidate`].
    fn is_kv_imbalanced(&self, workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> bool {
        // KV-based triggers — need a load snapshot; both default 1.0 = disabled.
        if let Some((min_usage, max_usage)) =
            self.backend_token_usage_bounds(workers, healthy_indices)
        {
            // Overload: a single engine is critically saturated.
            if max_usage > f64::from(self.config.overload_token_usage_threshold) {
                return true;
            }
            // KV imbalance: a hot engine with a materially cooler home.
            if max_usage - min_usage > f64::from(self.config.balance_token_usage_threshold) {
                return true;
            }
        }
        false
    }

    /// Min and max backend KV-cache utilization (0.0–1.0) across healthy workers
    /// that have a `WorkerMonitor` snapshot entry, as `(min, max)`. `None` when
    /// no receiver is wired or no healthy worker has a load entry (→ caller
    /// relies on the request-count spread).
    fn backend_token_usage_bounds(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
    ) -> Option<(f64, f64)> {
        let guard = self.load_rx.read();
        let rx = guard.as_ref()?;
        let loads = rx.borrow();
        let mut bounds: Option<(f64, f64)> = None;
        for &idx in healthy_indices {
            if let Some(load) = loads.get(workers[idx].url()) {
                let usage = load.effective_token_usage();
                bounds = Some(match bounds {
                    Some((min, max)) => (min.min(usage), max.max(usage)),
                    None => (usage, usage),
                });
            }
        }
        bounds
    }

    /// Initialize the trees with worker URLs (used only during initial setup)
    /// Initializes both string trees (HTTP) and token trees (gRPC) for each model.
    pub fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        // Hash mode keeps no tree state.
        if self.config.cache_index == CacheIndexKind::Hash {
            return;
        }
        // Group workers by model
        let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        // Initialize trees for each model (both string and token trees)
        for (tree_key, model_workers) in model_workers {
            // Initialize string tree (HTTP)
            let string_tree = self
                .string_trees
                .entry(tree_key.clone())
                .or_insert_with(|| Arc::new(Tree::new()));
            // Initialize token tree (gRPC)
            let token_tree = self
                .token_trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(self.new_token_tree()));

            for worker in model_workers {
                string_tree.insert_text("", worker.url());
                token_tree.insert_tokens(&[], worker.url());
            }
        }
    }

    /// Add a single worker to the trees (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
        if self.config.cache_index == CacheIndexKind::Hash {
            return;
        }
        let tree_key = normalize_model_key(worker.model_id()).to_string();
        // Add to string tree (HTTP)
        let string_tree = self
            .string_trees
            .entry(tree_key.clone())
            .or_insert_with(|| Arc::new(Tree::new()));
        string_tree.insert_text("", worker.url());
        // Add to token tree (gRPC)
        let token_tree = self
            .token_trees
            .entry(tree_key)
            .or_insert_with(|| Arc::new(self.new_token_tree()));
        token_tree.insert_tokens(&[], worker.url());
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
        if self.config.cache_index == CacheIndexKind::Hash {
            return;
        }
        let model_id_string = model_id.to_string();
        // Add to string tree (HTTP)
        let string_tree = self
            .string_trees
            .entry(model_id_string.clone())
            .or_insert_with(|| Arc::new(Tree::new()));
        string_tree.insert_text("", url);
        // Add to token tree (gRPC)
        let token_tree = self
            .token_trees
            .entry(model_id_string)
            .or_insert_with(|| Arc::new(self.new_token_tree()));
        token_tree.insert_tokens(&[], url);
    }

    /// Remove a worker from the trees
    pub fn remove_worker(&self, worker: &dyn Worker) {
        self.remove_worker_by_url(worker.url());
    }

    /// Remove a worker by URL, purging its tenant from every model's string
    /// and token tree. A removed worker's tenant count never grows again, so
    /// size-based eviction alone would retain its subtree forever.
    pub fn remove_worker_by_url(&self, url: &str) {
        let tenant: TenantId = Arc::from(url);
        for tree_ref in self.string_trees.iter() {
            tree_ref.value().remove_tenant_all(&tenant);
        }
        for tree_ref in self.token_trees.iter() {
            tree_ref.value().remove_tenant_all(&tenant);
        }
        let placement_maps: Vec<Arc<PlacementMap>> = self
            .placement_index
            .iter()
            .map(|model| Arc::clone(model.value()))
            .collect();
        for placements in placement_maps {
            placements.retain(|_, holders| {
                holders.retain(|h| h.worker_url != url);
                !holders.is_empty()
            });
        }
    }

    /// Run cache eviction to prevent unbounded growth
    pub fn evict_cache(&self, max_size: usize) {
        // Evict string trees (HTTP)
        for tree_ref in self.string_trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "String tree eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
        // Evict token trees (gRPC)
        for tree_ref in self.token_trees.iter() {
            let model_id = tree_ref.key();
            let tree = tree_ref.value();
            tree.evict_tenant_by_size(max_size);
            debug!(
                "Token tree eviction for model {}, max_size: {}",
                model_id, max_size
            );
        }
        // Evict hash index per model per tree kind. `max_size` is a
        // per-tree bound; clearing one model's overflow must not wipe
        // other models' still-valid metadata.
        for entry in self.hash_index.iter() {
            let per_model = entry.value();
            if per_model.string_tree.len() > max_size {
                per_model.string_tree.clear();
                debug!(
                    model_id = entry.key(),
                    "String hash index cleared (exceeded max_size: {})", max_size
                );
            }
            if per_model.token_tree.len() > max_size {
                per_model.token_tree.clear();
                debug!(
                    model_id = entry.key(),
                    "Token hash index cleared (exceeded max_size: {})", max_size
                );
            }
        }
    }

    /// Select worker with minimum load (used when load is imbalanced)
    /// Handles both HTTP (text-based) and gRPC (token-based) requests.
    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        avg_load: f64,
        model_id: &str,
    ) -> Option<usize> {
        // Shortest queue when imbalanced. The min-load index is gathered upstream
        // in select_worker with the (load, processed_requests, idx) tie-break
        // from #1714 (spreads load when decode outpaces prefill). When a
        // no-cache strategy is attached, it may override the selection.
        let min_load_idx = self.resolve_no_cache(
            workers,
            info,
            healthy_indices,
            min_load_idx,
            avg_load,
            model_id,
            None,
        )?;

        let worker_url = workers[min_load_idx].url().to_string();

        // Even in imbalanced mode, update the appropriate tree to maintain cache state
        // Prefer token tree for gRPC requests, fall back to string tree for HTTP
        if let Some(tokens) = info.tokens {
            // gRPC request: update token tree
            let tree = self
                .token_trees
                .get(model_id)
                .map(|entry| entry.value().clone());
            if let Some(tree) = tree {
                // We need the match result (the prior shared prefix) BEFORE the
                // insert so the hash_index stores only that bounded prefix, not
                // the full path that exists post-insert (32K tokens × 4 bytes ×
                // max_tree_size = multi-GB/model). `match_and_insert` resolves
                // the match against the pre-insert tree and inserts in the SAME
                // descent, so `result.matched_token_count` is the same prior
                // prefix length the standalone match returned. When we don't
                // populate the index, a plain insert (no match) suffices.
                if self.should_populate_hash_index() {
                    let result = tree.match_and_insert(tokens, &worker_url);
                    let matched_prefix: Vec<u32> = tokens[..result.matched_token_count].to_vec();
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .token_tree
                        .insert(kv_index::hash_token_path(tokens), matched_prefix);
                } else {
                    tree.insert_tokens(tokens, &worker_url);
                }
            }
        } else if let Some(text) = info.request_text {
            // HTTP request: update string tree
            let tree = self
                .string_trees
                .get(model_id)
                .map(|entry| entry.value().clone());

            if let Some(tree) = tree {
                // Match BEFORE insert so the hash_index stores only the prior
                // shared prefix (~50-200 chars), not the full prompt (20KB+)
                // that exists post-insert. `match_and_insert` does both in a
                // single descent; `result.matched_char_count` is the same prior
                // prefix length the standalone match returned. When we don't
                // populate the index, a plain insert (no match) suffices.
                if self.should_populate_hash_index() {
                    let result = tree.match_and_insert(text, &worker_url);
                    let matched_prefix: String =
                        text.chars().take(result.matched_char_count).collect();
                    let path_hash = kv_index::hash_node_path(text);
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .string_tree
                        .insert(path_hash, matched_prefix);
                } else {
                    tree.insert_text(text, &worker_url);
                }
            } else {
                debug!(
                    "Warning: No string tree found for model '{}', skipping cache update",
                    model_id
                );
            }
        }

        // Increment processed counter
        workers[min_load_idx].increment_processed();

        debug!(
            branch = "kv_pressure_min_load",
            worker = %worker_url,
            model_id,
            "Cache-aware selection"
        );
        Some(min_load_idx)
    }
}

/// Which of the two local trees a hash query targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TreeKind {
    String,
    Token,
}

/// Handle the policy exposes so mesh-adjacent consumers can apply
/// remote tenant inserts against the local tree without reaching
/// into private fields. Defined here (not in the adapter) to keep
/// the dependency direction `adapter → policy`.
pub trait TreeHandle: Send + Sync + std::fmt::Debug {
    /// If `node_hash` is known locally (resolvable to a stored
    /// matched-prefix), record `worker_url` as a tenant of the
    /// matched node and return `true`. Returns `false` if the
    /// hash isn't known — the caller is expected to request
    /// repair so the path can be reconstructed from a peer.
    ///
    /// This subsumes "is the hash known?" plus "apply the
    /// insert": the adapter doesn't need separate read+write
    /// trips, and we never expose the matched value across the
    /// trait boundary (it stays inside the policy where
    /// eviction owns its lifecycle).
    fn apply_known_remote_insert(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
        node_hash: u64,
        worker_url: &str,
    ) -> bool;

    /// Open a stream of `RepairEntry` for one `(model_id,
    /// tree_kind)`, in the deterministic pre-order produced by
    /// the underlying tree's `iter_entries`. Returns `None` if
    /// no tree exists locally for that model. Paging is wire
    /// shape and lives in the adapter, not on this trait — the
    /// stream just yields entries one at a time.
    fn open_repair_stream(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
    ) -> Option<Box<dyn Iterator<Item = RepairEntry> + Send>>;

    /// Apply every entry in `page` to the local `(model_id,
    /// tree_kind)` tree, creating the tree if it doesn't yet
    /// exist locally. Returns the number of entries successfully
    /// applied (entries whose variant doesn't match `tree_kind`
    /// are logged and skipped, not applied). Idempotent —
    /// reapplying the same page is a no-op on the tree state
    /// because the underlying radix tree's `insert_text` /
    /// `insert_tokens` are themselves idempotent for the same
    /// `(path, tenant)` pair.
    fn apply_repair_page(&self, page: &TreeRepairPage) -> usize;
}

impl TreeHandle for CacheAwarePolicy {
    fn apply_known_remote_insert(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
        node_hash: u64,
        worker_url: &str,
    ) -> bool {
        // Normalize empty → UNKNOWN_MODEL_ID so lookups match the
        // key shape every populate site already uses.
        let model_id = normalize_model_key(model_id);
        let Some(model_entry) = self.hash_index.get(model_id) else {
            return false;
        };
        match tree_kind {
            TreeKind::String => {
                let Some(path) = model_entry.string_tree.get(&node_hash) else {
                    return false;
                };
                let Some(tree) = self.string_trees.get(model_id) else {
                    // Hash index entry without a corresponding
                    // tree means a populate site mutated
                    // `hash_index` without creating the tree
                    // (or eviction dropped the tree but left the
                    // index). Returning false here masks the
                    // invariant violation as a spurious repair
                    // request, so log loudly.
                    warn!(
                        model_id,
                        node_hash,
                        "string hash_index entry without matching string_trees entry; populate-site invariant violated",
                    );
                    return false;
                };
                tree.insert_text(path.value(), worker_url);
                true
            }
            TreeKind::Token => {
                let Some(tokens) = model_entry.token_tree.get(&node_hash) else {
                    return false;
                };
                let Some(tree) = self.token_trees.get(model_id) else {
                    warn!(
                        model_id,
                        node_hash,
                        "token hash_index entry without matching token_trees entry; populate-site invariant violated",
                    );
                    return false;
                };
                tree.insert_tokens(tokens.value(), worker_url);
                true
            }
        }
    }

    fn open_repair_stream(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
    ) -> Option<Box<dyn Iterator<Item = RepairEntry> + Send>> {
        let model_id = normalize_model_key(model_id);
        match tree_kind {
            TreeKind::String => {
                let tree = self.string_trees.get(model_id)?.value().clone();
                Some(Box::new(tree.iter_entries().map(|(path, tenants)| {
                    RepairEntry::String { path, tenants }
                })))
            }
            TreeKind::Token => {
                let tree = self.token_trees.get(model_id)?.value().clone();
                Some(Box::new(tree.iter_entries().map(|(tokens, tenants)| {
                    RepairEntry::Token { tokens, tenants }
                })))
            }
        }
    }

    fn apply_repair_page(&self, page: &TreeRepairPage) -> usize {
        let model_id = normalize_model_key(&page.model_id);
        let mut applied: usize = 0;
        match page.tree_kind {
            TreeKind::String => {
                // Create the tree on first repair page if it
                // doesn't exist yet locally — repair is the
                // primary cold-start path for a fresh peer.
                let tree = self
                    .string_trees
                    .entry(model_id.to_string())
                    .or_insert_with(|| Arc::new(Tree::new()))
                    .clone();
                for entry in &page.entries {
                    match entry {
                        RepairEntry::String { path, tenants } => {
                            for (tenant, _epoch) in tenants {
                                tree.insert_text(path, tenant);
                            }
                            self.hash_index
                                .entry(model_id.to_string())
                                .or_default()
                                .string_tree
                                .insert(kv_index::hash_node_path(path), path.clone());
                            applied += 1;
                        }
                        RepairEntry::Token { .. } => {
                            warn!(
                                model_id,
                                session_id = %page.session_id,
                                page_index = page.page_index,
                                "RepairEntry variant mismatch: page kind=String but entry kind=Token; skipping",
                            );
                        }
                    }
                }
            }
            TreeKind::Token => {
                let tree = self
                    .token_trees
                    .entry(model_id.to_string())
                    .or_insert_with(|| Arc::new(self.new_token_tree()))
                    .clone();
                for entry in &page.entries {
                    match entry {
                        RepairEntry::Token { tokens, tenants } => {
                            for (tenant, _epoch) in tenants {
                                tree.insert_tokens(tokens, tenant);
                            }
                            self.hash_index
                                .entry(model_id.to_string())
                                .or_default()
                                .token_tree
                                .insert(kv_index::hash_token_path(tokens), tokens.clone());
                            applied += 1;
                        }
                        RepairEntry::String { .. } => {
                            warn!(
                                model_id,
                                session_id = %page.session_id,
                                page_index = page.page_index,
                                "RepairEntry variant mismatch: page kind=Token but entry kind=String; skipping",
                            );
                        }
                    }
                }
            }
        }
        applied
    }
}

/// One positive-overlap candidate in event-driven scoring: slice index,
/// (possibly decayed) score, and the in-flight-count tie-break key.
struct OverlapCandidate {
    idx: usize,
    effective_score: f64,
    load: usize,
}

/// Pressure-tuning inputs for [`CacheAwarePolicy::score_overlap`]: the two
/// config knobs plus a waiting-prefill backlog snapshot (worker URL → queued
/// uncached tokens, clamped non-negative) captured from the load receiver at
/// selection time. `waiting_prefill_tokens` is `None` when decay is off or no
/// load receiver is wired; workers absent from the map are never decayed.
struct OverlapTuning<'a> {
    overlap_decay: f32,
    selection_temperature: f32,
    waiting_prefill_tokens: Option<&'a HashMap<String, i64>>,
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let request_text = info.request_text;
        let request_tokens = info.tokens;

        // Single O(workers) gather: read each worker once via routing_state()
        // (status + load + processed + overload veto under one ArcSwap guard),
        // replacing the former separate passes whose per-worker guard traffic
        // dominated routing CPU at scale. Collects eligible indices, the load
        // sum (for the per-request pressure gate), and the min-load index.
        let mut healthy_indices: Vec<usize> = Vec::with_capacity(workers.len());
        let mut load_sum = 0usize;
        // Min-load worker, (load, processed_requests, idx) tie-break (#1714);
        // `processed` rides the same guard as `load`, so it is free here.
        let mut min_key: Option<(usize, usize, usize)> = None;
        let mut min_load_idx: Option<usize> = None;
        for (idx, worker) in workers.iter().enumerate() {
            let state = worker.routing_state();
            // The overload veto costs nothing here: `state` is the word this
            // pass already loaded for health, circuit breaker and load.
            if state.eligible() {
                healthy_indices.push(idx);
                load_sum += state.load;
                let key = (state.load, state.processed, idx);
                match min_key {
                    Some(best) if key >= best => {}
                    _ => {
                        min_key = Some(key);
                        min_load_idx = Some(idx);
                    }
                }
            }
        }

        if healthy_indices.is_empty() {
            return None;
        }
        let avg_load = load_sum as f64 / healthy_indices.len() as f64;

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Hash mode: TTL'd exact-match placement index; the radix trees are
        // neither consulted nor populated.
        if self.config.cache_index == CacheIndexKind::Hash {
            return self.select_worker_hash(
                workers,
                info,
                &healthy_indices,
                min_load_idx,
                avg_load,
                model_id,
            );
        }

        // Abandon cache affinity fleet-wide only under backend KV pressure;
        // request-count pressure is applied per request to the selected
        // candidate inside each affinity path.
        if self.is_kv_imbalanced(workers, &healthy_indices) {
            return self.select_worker_min_load(
                workers,
                info,
                &healthy_indices,
                min_load_idx,
                avg_load,
                model_id,
            );
        }

        // Cache-aware routing when balanced — three types (mutually exclusive):
        //   1. Event-driven: PositionalIndexer overlap scoring (gRPC + KV events)
        //   2. Approximate token tree: TokenTree prefix matching (gRPC, no events)
        //   3. Approximate string tree: Tree prefix matching (HTTP)
        if let Some(tokens) = request_tokens {
            if self.has_event_indexer(model_id) {
                self.select_worker_event_driven(
                    workers,
                    info,
                    tokens,
                    &healthy_indices,
                    min_load_idx,
                    avg_load,
                    model_id,
                )
            } else {
                self.select_worker_with_tokens(
                    workers,
                    info,
                    tokens,
                    &healthy_indices,
                    min_load_idx,
                    avg_load,
                    model_id,
                )
            }
        } else {
            let text = request_text.unwrap_or("");
            self.select_worker_with_text(
                workers,
                info,
                text,
                &healthy_indices,
                min_load_idx,
                avg_load,
                model_id,
            )
        }
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        // Could track success rates per worker for more intelligent routing
        if !success {
            // Optionally reduce affinity for failed requests
            tracing::debug!(
                "Request to {} completed with success={}",
                worker_url,
                success
            );
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware"
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    /// Backend loads feed the KV-usage gate and the waiting-prefill decay;
    /// both are disabled by default, so only configured policies poll.
    fn needs_backend_loads(&self) -> bool {
        self.config.overlap_decay > 0.0
            || self.config.balance_token_usage_threshold < 1.0
            || self.config.overload_token_usage_threshold < 1.0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Private helper methods for select_worker
impl CacheAwarePolicy {
    /// Check if an event-driven indexer exists with data for this model.
    /// Returns false when the indexer is empty (startup, reconnect) so
    /// routing falls through to the approximate token tree instead of
    /// taking the event-driven path with no data and landing on min-load.
    fn has_event_indexer(&self, model_id: &str) -> bool {
        let guard = self.kv_monitor.read();
        guard
            .as_ref()
            .and_then(|m| m.get_indexer(model_id))
            .is_some_and(|indexer| indexer.current_size() > 0)
    }

    /// Waiting-prefill backlog snapshot (worker URL → queued uncached tokens),
    /// or `None` when decay is off or no load receiver is wired. The clone is
    /// per-selection; with decay off the map is never read.
    fn waiting_prefill_snapshot(&self) -> Option<HashMap<String, i64>> {
        if self.config.overlap_decay <= 0.0 {
            return None;
        }
        let guard = self.load_rx.read();
        guard.as_ref().map(|rx| {
            rx.borrow()
                .iter()
                .map(|(url, load)| (url.clone(), load.total_waiting_uncached_tokens().max(0)))
                .collect::<HashMap<String, i64>>()
        })
    }

    /// Per-request count-pressure gate on the selected candidate: over
    /// `balance_rel_threshold` times the healthy-fleet mean load AND
    /// `balance_abs_threshold` requests above it, the request spills to the
    /// least-loaded worker instead. Affinity paths insert for the spill
    /// target, so a prefix whose home saturates gains an additional tenant —
    /// hot prefixes replicate instead of queueing behind one engine. Both
    /// margins must clear so the gate neither fires on steady-state variance
    /// (relative alone would, at low means) nor stays blind to a deep queue
    /// (absolute alone would, at high means).
    fn gate_selected_candidate(
        &self,
        workers: &[Arc<dyn Worker>],
        selected: usize,
        avg_load: f64,
        min_load_idx: Option<usize>,
    ) -> Option<usize> {
        let load = workers[selected].load() as f64;
        if load > avg_load * f64::from(self.config.balance_rel_threshold)
            && load > avg_load + self.config.balance_abs_threshold as f64
        {
            return min_load_idx.or(Some(selected));
        }
        Some(selected)
    }

    /// Pressure-select among the tenants holding the matched prefix.
    ///
    /// Every matched tenant serves the same prefix, so raw overlap cannot
    /// discriminate; the waiting-prefill decay and the load tie-break do.
    /// With default tuning this reduces to the least-loaded holding tenant
    /// (uniform among ties). Returns `None` when no matched tenant is
    /// healthy, preserving the caller's no-insert fallback.
    fn select_matched_candidate(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        matched_tenants: &[TenantId],
        request_units: usize,
        avg_load: f64,
        min_load_idx: Option<usize>,
    ) -> Option<usize> {
        let mut candidates: Vec<OverlapCandidate> = Vec::new();
        for &idx in healthy_indices {
            let url = workers[idx].url();
            if matched_tenants.iter().any(|tenant| tenant.as_ref() == url) {
                candidates.push(OverlapCandidate {
                    idx,
                    effective_score: 1.0,
                    load: workers[idx].load(),
                });
            }
        }
        if candidates.is_empty() {
            return None;
        }

        let waiting = self.waiting_prefill_snapshot();
        let tuning = OverlapTuning {
            overlap_decay: self.config.overlap_decay,
            selection_temperature: self.config.selection_temperature,
            waiting_prefill_tokens: waiting.as_ref(),
        };
        let request_blocks = (request_units / self.config.block_size).max(1);
        Self::apply_overlap_decay(
            workers,
            &mut candidates,
            request_blocks,
            self.config.block_size,
            &tuning,
        );
        let selected = if tuning.selection_temperature > 0.0 {
            Self::sample_by_temperature(&candidates, tuning.selection_temperature)
        } else {
            Self::argmax_with_random_ties(&candidates)
        }?;
        self.gate_selected_candidate(workers, selected, avg_load, min_load_idx)
    }

    /// Event-driven routing: PositionalIndexer overlap scoring (Type 1).
    ///
    /// Self-contained — when overlap is found, selects the worker with the best
    /// cache match. When no overlap (cold start, novel tokens, short request),
    /// falls back to min-load. Does NOT fall back to approximate token tree.
    fn select_worker_event_driven(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        tokens: &[u32],
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        avg_load: f64,
        model_id: &str,
    ) -> Option<usize> {
        let guard = self.kv_monitor.read();
        let monitor = guard.as_ref()?;
        let indexer = monitor.get_indexer(model_id)?;

        // Per-model block_size: learned from events > config default
        let block_size = monitor
            .block_size(model_id)
            .unwrap_or(self.config.block_size);

        let waiting_prefill_tokens = self.waiting_prefill_snapshot();
        let tuning = OverlapTuning {
            overlap_decay: self.config.overlap_decay,
            selection_temperature: self.config.selection_temperature,
            waiting_prefill_tokens: waiting_prefill_tokens.as_ref(),
        };

        if let Some(idx) = Self::score_overlap(
            workers,
            tokens,
            healthy_indices,
            &indexer,
            block_size,
            &tuning,
        ) {
            return self.gate_selected_candidate(workers, idx, avg_load, min_load_idx);
        }

        // No cache overlap — delegate to no-cache strategy (default: min-load).
        let min_idx = self.resolve_no_cache(
            workers,
            info,
            healthy_indices,
            min_load_idx,
            avg_load,
            model_id,
            None,
        )?;
        debug!(
            worker = workers[min_idx].url(),
            model_id, "Event-driven routing: no overlap, min-load fallback"
        );
        workers[min_idx].increment_processed();
        Some(min_idx)
    }

    /// Score healthy workers by PositionalIndexer overlap and select the best.
    ///
    /// Returns `Some(idx)` if at least one worker has cached blocks matching the
    /// request. Returns `None` if the request is too short for a full block or
    /// no workers have matching data.
    ///
    /// With default tuning (decay 0, temperature 0) selection is exactly the
    /// historical behavior: max raw overlap, then lower load, then uniformly
    /// at random among exact ties.
    fn score_overlap(
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        indexer: &PositionalIndexer,
        block_size: usize,
        tuning: &OverlapTuning<'_>,
    ) -> Option<usize> {
        let content_hashes = compute_request_content_hashes(tokens, block_size);
        if content_hashes.is_empty() {
            return None;
        }

        let overlap = indexer.find_matches(&content_hashes, false);
        if overlap.scores.is_empty() {
            return None;
        }

        // Gather the positive-overlap candidates once; both selection modes
        // and the decay's fleet-floor computation need the full set.
        let mut candidates: Vec<OverlapCandidate> = Vec::new();
        for &idx in healthy_indices {
            let Some(score) = indexer
                .worker_id(workers[idx].url())
                .and_then(|id| overlap.scores.get(&id))
                .copied()
                .filter(|&s| s > 0)
            else {
                continue;
            };
            candidates.push(OverlapCandidate {
                idx,
                effective_score: f64::from(score),
                load: workers[idx].load(),
            });
        }
        if candidates.is_empty() {
            return None;
        }

        Self::apply_overlap_decay(
            workers,
            &mut candidates,
            content_hashes.len(),
            block_size,
            tuning,
        );

        let best_idx = if tuning.selection_temperature > 0.0 {
            Self::sample_by_temperature(&candidates, tuning.selection_temperature)
        } else {
            Self::argmax_with_random_ties(&candidates)
        }?;

        debug!(
            worker = workers[best_idx].url(),
            score = indexer
                .worker_id(workers[best_idx].url())
                .and_then(|id| overlap.scores.get(&id))
                .copied()
                .unwrap_or(0),
            "Event-driven routing: overlap match"
        );
        workers[best_idx].increment_processed();
        Some(best_idx)
    }

    /// Anti-hotspot decay: divide each candidate's overlap score by
    /// `1 + overlap_decay * x`, where `x` is the candidate's waiting-prefill
    /// backlog in blocks, in excess of the minimum among candidates WITH load
    /// data, normalized by the request's own block count ("how many of *this*
    /// request's prefills is the worker already behind by"). The rational form
    /// keeps the multiplier in (0, 1] with no clamping: exactly 1 at the
    /// fleet floor, asymptotic to 0 under extreme backlog. Candidates without
    /// a load entry are never decayed — missing data must not punish.
    fn apply_overlap_decay(
        workers: &[Arc<dyn Worker>],
        candidates: &mut [OverlapCandidate],
        request_blocks: usize,
        block_size: usize,
        tuning: &OverlapTuning<'_>,
    ) {
        let (Some(waiting), true) = (tuning.waiting_prefill_tokens, tuning.overlap_decay > 0.0)
        else {
            return;
        };
        let backlog_of = |c: &OverlapCandidate| waiting.get(workers[c.idx].url()).copied();
        let Some(min_backlog) = candidates.iter().filter_map(&backlog_of).min() else {
            return;
        };
        // request_blocks >= 1 (empty-hash requests returned earlier);
        // block_size > 0 is config-validated.
        for candidate in candidates.iter_mut() {
            let Some(backlog) = backlog_of(candidate) else {
                continue;
            };
            let excess_blocks = (backlog - min_backlog) as f64 / block_size as f64;
            let x = excess_blocks / request_blocks as f64;
            candidate.effective_score /= 1.0 + f64::from(tuning.overlap_decay) * x;
        }
    }

    /// Historical selection: max effective score, then lower load, then
    /// uniformly at random among exact ties. Tie-break rationale: the old
    /// final key (smaller total tree size, then slice order) was a spreading
    /// proxy but deterministic — equal-overlap equal-load workers herd onto
    /// one index until the global tree-size ordering flips, and tree size
    /// tracks event-stream health, so an event-lagged worker looked "small"
    /// and attracted the whole tie. Uniform random gives the same spreading
    /// goal memorylessly; the next request's overlap scores restore affinity
    /// to whichever worker actually cached the prefix.
    fn argmax_with_random_ties(candidates: &[OverlapCandidate]) -> Option<usize> {
        let mut best: Option<(f64, usize)> = None;
        let mut tied: Vec<usize> = Vec::new();
        for candidate in candidates {
            let key = (candidate.effective_score, candidate.load);
            match best {
                Some((best_score, best_load)) => {
                    if key.0 > best_score || (key.0 == best_score && key.1 < best_load) {
                        best = Some(key);
                        tied.clear();
                        tied.push(candidate.idx);
                    } else if key.0 == best_score && key.1 == best_load {
                        tied.push(candidate.idx);
                    }
                }
                None => {
                    best = Some(key);
                    tied.push(candidate.idx);
                }
            }
        }
        match tied.len() {
            0 => None,
            1 => Some(tied[0]),
            n => Some(tied[rand::rng().random_range(0..n)]),
        }
    }

    /// Softmax selection over min-max normalized effective scores. The
    /// normalization makes temperature scale-free: only a candidate's
    /// relative position within the current score spread matters, so one
    /// temperature setting behaves the same whether overlaps span 2 blocks
    /// or 2000. The best candidate's exponent is exactly 0 (overflow-safe);
    /// a degenerate spread (all equal) is a uniform draw. Inverse-CDF
    /// sampling with a last-row fallback against floating-point drift.
    fn sample_by_temperature(candidates: &[OverlapCandidate], temperature: f32) -> Option<usize> {
        let first = candidates.first()?;
        let (min, max) = candidates.iter().fold(
            (first.effective_score, first.effective_score),
            |(min, max), c| (min.min(c.effective_score), max.max(c.effective_score)),
        );
        let range = max - min;
        if range <= 0.0 {
            return Some(candidates[rand::rng().random_range(0..candidates.len())].idx);
        }
        let weights: Vec<f64> = candidates
            .iter()
            .map(|c| (((c.effective_score - min) / range - 1.0) / f64::from(temperature)).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let draw = rand::rng().random::<f64>() * total;
        let mut cumulative = 0.0;
        for (candidate, weight) in candidates.iter().zip(&weights) {
            cumulative += weight;
            if cumulative >= draw {
                return Some(candidate.idx);
            }
        }
        candidates.last().map(|c| c.idx)
    }

    /// One decision line per tree-routed request. `selected_url == None`
    /// means the caller fell back to `fallback_url` (first healthy).
    fn log_tree_decision(
        &self,
        selected_url: Option<&str>,
        fallback_url: Option<&str>,
        matched_units: usize,
        input_units: usize,
        matched_tenants: &[TenantId],
        model_id: &str,
    ) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let matched_ratio = if input_units == 0 {
            0.0
        } else {
            matched_units as f32 / input_units as f32
        };
        let branch = match selected_url {
            None => "first_healthy_fallback",
            Some(_) if matched_ratio <= self.config.cache_threshold => "min_load_fallback",
            Some(url) => {
                if matched_tenants.iter().any(|tenant| tenant.as_ref() == url) {
                    "tree_match"
                } else {
                    "spill"
                }
            }
        };
        debug!(
            index = "tree",
            branch,
            worker = selected_url.or(fallback_url).unwrap_or("none"),
            model_id,
            matched_ratio = f64::from(matched_ratio),
            threshold = f64::from(self.config.cache_threshold),
            "Cache-aware selection"
        );
    }

    /// One decision line per hash-mode selection. `level` is the matched
    /// boundary (0 for the fallback branches).
    fn log_hash_decision(branch: &'static str, level: usize, worker: &str, model_id: &str) {
        debug!(
            index = "hash",
            branch, level, worker, model_id, "Cache-aware selection"
        );
    }

    /// Hash-mode selection: probe the placement index deepest-boundary-first
    /// for a live holder of this request's head, then record the dispatch at
    /// every applicable boundary.
    fn select_worker_hash(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        avg_load: f64,
        model_id: &str,
    ) -> Option<usize> {
        let now = Instant::now();

        // Hash mode keys on token ids; untokenized requests stay load-balanced.
        let Some(tokens) = info.tokens.filter(|t| !t.is_empty()) else {
            return self.hash_min_load(
                workers,
                info,
                healthy_indices,
                min_load_idx,
                avg_load,
                model_id,
                "min_load_fallback",
                &[],
                &[],
                now,
            );
        };

        let applicable_end = self
            .config
            .cache_boundaries
            .partition_point(|&p| p <= tokens.len());
        let applicable = &self.config.cache_boundaries[..applicable_end];

        // Head-only traffic must stay load-balanced.
        if applicable.is_empty() {
            return self.hash_min_load(
                workers,
                info,
                healthy_indices,
                min_load_idx,
                avg_load,
                model_id,
                "short_request",
                tokens,
                applicable,
                now,
            );
        }

        if self.is_kv_imbalanced(workers, healthy_indices) {
            return self.hash_min_load(
                workers,
                info,
                healthy_indices,
                min_load_idx,
                avg_load,
                model_id,
                "kv_pressure_min_load",
                tokens,
                applicable,
                now,
            );
        }

        let url_to_idx = Self::healthy_url_index(workers, healthy_indices);
        for &boundary in applicable.iter().rev() {
            let key = (boundary, hash_token_head(&tokens[..boundary]));
            let Some(holder_idx) =
                self.live_holder_min_load(workers, &url_to_idx, model_id, key, now)
            else {
                continue;
            };
            let selected =
                self.gate_selected_candidate(workers, holder_idx, avg_load, min_load_idx)?;
            let branch = if selected == holder_idx {
                "hash_hit"
            } else {
                "hash_spill"
            };
            self.record_placement(model_id, tokens, applicable, workers[selected].url(), now);
            Self::log_hash_decision(branch, boundary, workers[selected].url(), model_id);
            workers[selected].increment_processed();
            return Some(selected);
        }

        self.hash_min_load(
            workers,
            info,
            healthy_indices,
            min_load_idx,
            avg_load,
            model_id,
            "min_load_fallback",
            tokens,
            applicable,
            now,
        )
    }

    /// Min-load dispatch for the hash-path fallback branches; records the
    /// placement when boundaries apply.
    #[expect(clippy::too_many_arguments, reason = "hot-path plumbing, not state")]
    fn hash_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        avg_load: f64,
        model_id: &str,
        branch: &'static str,
        tokens: &[u32],
        applicable: &[usize],
        now: Instant,
    ) -> Option<usize> {
        let idx = self.resolve_no_cache(
            workers,
            info,
            healthy_indices,
            min_load_idx,
            avg_load,
            model_id,
            None,
        )?;
        if !applicable.is_empty() {
            self.record_placement(model_id, tokens, applicable, workers[idx].url(), now);
        }
        Self::log_hash_decision(branch, 0, workers[idx].url(), model_id);
        workers[idx].increment_processed();
        Some(idx)
    }

    /// URL → healthy worker index; duplicate URLs resolve to the index
    /// min-load selection would pick.
    fn healthy_url_index<'a>(
        workers: &'a [Arc<dyn Worker>],
        healthy_indices: &[usize],
    ) -> HashMap<&'a str, usize> {
        let mut url_to_idx: HashMap<&str, usize> = HashMap::with_capacity(healthy_indices.len());
        for &idx in healthy_indices {
            url_to_idx
                .entry(workers[idx].url())
                .and_modify(|cur| {
                    if (workers[idx].load(), idx) < (workers[*cur].load(), *cur) {
                        *cur = idx;
                    }
                })
                .or_insert(idx);
        }
        url_to_idx
    }

    /// Least-loaded live holder of `key` among healthy workers, or `None`.
    /// Expired holders are pruned in place (lazy expiry on read). Guarded
    /// work is O(holder cap): resolution goes through `url_to_idx`.
    fn live_holder_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        url_to_idx: &HashMap<&str, usize>,
        model_id: &str,
        key: (usize, u64),
        now: Instant,
    ) -> Option<usize> {
        let ttl = Duration::from_secs(self.config.cache_ttl_secs);
        let model = Arc::clone(self.placement_index.get(model_id)?.value());
        let mut holders = model.get_mut(&key)?;
        holders.retain(|h| now.duration_since(h.last_touch) <= ttl);
        holders
            .iter()
            .filter_map(|h| url_to_idx.get(h.worker_url.as_str()).copied())
            .min_by_key(|&idx| (workers[idx].load(), idx))
    }

    /// Record/touch `worker_url` at every applicable boundary of this
    /// request; above the holder cap the stalest holder is evicted.
    fn record_placement(
        &self,
        model_id: &str,
        tokens: &[u32],
        boundaries: &[usize],
        worker_url: &str,
        now: Instant,
    ) {
        let ttl = Duration::from_secs(self.config.cache_ttl_secs);
        let model = if let Some(entry) = self.placement_index.get(model_id) {
            Arc::clone(entry.value())
        } else {
            Arc::clone(
                self.placement_index
                    .entry(model_id.to_string())
                    .or_default()
                    .value(),
            )
        };
        for &boundary in boundaries {
            let key = (boundary, hash_token_head(&tokens[..boundary]));
            let mut holders = model.entry(key).or_default();
            holders.retain(|h| now.duration_since(h.last_touch) <= ttl);
            if let Some(holder) = holders.iter_mut().find(|h| h.worker_url == worker_url) {
                holder.last_touch = now;
                continue;
            }
            if holders.len() >= PLACEMENT_HOLDER_CAP {
                if let Some(stalest) = holders
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, h)| h.last_touch)
                    .map(|(i, _)| i)
                {
                    holders.swap_remove(stalest);
                }
            }
            holders.push(PlacementHolder {
                worker_url: worker_url.to_string(),
                last_touch: now,
            });
        }
    }

    /// Drop expired holders and empty keys; refresh the per-model entry
    /// gauge. Returns total live entries across models.
    fn sweep_placement_index(
        index: &DashMap<String, Arc<PlacementMap>>,
        ttl: Duration,
        now: Instant,
    ) -> usize {
        let models: Vec<(String, Arc<PlacementMap>)> = index
            .iter()
            .map(|model| (model.key().clone(), Arc::clone(model.value())))
            .collect();
        let mut total = 0usize;
        for (model_id, placements) in models {
            placements.retain(|_, holders| {
                holders.retain(|h| now.duration_since(h.last_touch) <= ttl);
                !holders.is_empty()
            });
            let entries = placements.len();
            total += entries;
            Metrics::set_cache_placement_entries(&model_id, entries);
        }
        total
    }

    /// Select worker using token-based tree (gRPC path)
    fn select_worker_with_tokens(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        tokens: &[u32],
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        avg_load: f64,
        model_id: &str,
    ) -> Option<usize> {
        let tree = self
            .token_trees
            .get(model_id)
            .map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            // Single tree descent: match, pick the worker from the match
            // result, then insert for it — replacing the former
            // match_prefix_with_counts + insert_tokens pair (two full descents
            // over the same prefix). The selection closure runs once, after the
            // match, mirroring the previous branch exactly:
            //   * cache hit  (match_rate > threshold): route to the matched
            //     worker if it is still healthy — insert for it;
            //   * cache miss (match_rate <= threshold): route to the least-loaded
            //     worker — insert for it;
            //   * matched worker gone/unhealthy: select nothing and DON'T insert
            //     (closure returns None), falling back to first-healthy below.
            let mut selected_idx: Option<usize> = None;
            // Hinted token prefixes (x-smg-routing-tokens) insert as-is: the
            // tree page-aligns internally, so a truncated prefix still trains
            // affinity.
            let result = tree.match_and_insert_with(tokens, |result| {
                let match_rate = if result.input_token_count == 0 {
                    0.0
                } else {
                    result.matched_token_count as f32 / result.input_token_count as f32
                };

                selected_idx = if match_rate > self.config.cache_threshold {
                    // Cache hit: pressure-select among the tenants holding the
                    // matched prefix. "Healthy" excludes circuit-broken workers,
                    // so a fully CB-tripped tenant set falls through to
                    // first-healthy without inserting (intended). A gated
                    // selection lands on min-load, and the insert below makes
                    // the spill target a new tenant of this prefix.
                    self.select_matched_candidate(
                        workers,
                        healthy_indices,
                        &result.matched_tenants,
                        tokens.len(),
                        avg_load,
                        min_load_idx,
                    )
                } else {
                    // Partial match below threshold: pass the uncached
                    // portion (input - matched) so the length strategy can
                    // classify by actual prefill work, not full size.
                    // When matched == 0 (no match at all), pass None so the
                    // strategy falls through to its normal priority chain
                    // (header → tokens → char estimate).
                    let uncached_hint = (result.matched_token_count > 0).then(|| {
                        UncachedHint::Tokens(
                            result
                                .input_token_count
                                .saturating_sub(result.matched_token_count),
                        )
                    });
                    self.resolve_no_cache(
                        workers,
                        info,
                        healthy_indices,
                        min_load_idx,
                        avg_load,
                        model_id,
                        uncached_hint,
                    )
                };

                // Insert for the selected worker (None => no insert, exactly
                // like the old `if let Some(idx)` guard around insert_tokens).
                selected_idx.map(|idx| workers[idx].url())
            });

            self.log_tree_decision(
                selected_idx.map(|idx| workers[idx].url()),
                healthy_indices.first().map(|&idx| workers[idx].url()),
                result.matched_token_count,
                result.input_token_count,
                &result.matched_tenants,
                model_id,
            );

            if let Some(idx) = selected_idx {
                // Record hash(full_tokens)→matched_prefix tokens.
                // The hash key matches what sync_tree_operation
                // sends on the wire (hash of full sequence). The
                // VALUE is only the matched prefix — not the full
                // sequence (32K tokens × 4 bytes = 128 KB worst
                // case). v1 never populated a token hash index;
                // v2's `TreeHandle` impl consults this map per
                // incoming token delta, so maintain it alongside
                // the tree. Mirrors the string side at the
                // analogous block; reuses the match `result`
                // returned by match_and_insert_with.
                if self.should_populate_hash_index() {
                    let matched_prefix: Vec<u32> = tokens[..result.matched_token_count].to_vec();
                    let node_hash = kv_index::hash_token_path(tokens);
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .token_tree
                        .insert(node_hash, matched_prefix);
                    // Publish only what peers can already resolve:
                    // the node_hash keys the hash_index entry we
                    // just wrote, so a receiver that repairs against
                    // us will land the same worker onto the same
                    // tree node. `epoch: 0` — the field is a reserved
                    // slot the current receiver does not consult
                    // (see `TreeDelta::epoch`).
                    self.sync_local_insert(
                        model_id,
                        TreeDelta {
                            tree_kind: TreeKind::Token,
                            node_hash,
                            worker_url: workers[idx].url().to_string(),
                            epoch: 0,
                        },
                    );
                }
                workers[idx].increment_processed();
                return Some(idx);
            }

            // Selected worker no longer exists or unhealthy - fall back to first healthy
            // Stale entries will be cleaned up by LRU eviction
            healthy_indices.first().copied()
        } else {
            let idx = healthy_indices[rand::rng().random_range(0..healthy_indices.len())];
            debug!(
                branch = "no_tree_random",
                worker = workers[idx].url(),
                model_id,
                "Cache-aware selection"
            );
            Some(idx)
        }
    }

    /// Select worker using string-based tree (HTTP path)
    fn select_worker_with_text(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        text: &str,
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        avg_load: f64,
        model_id: &str,
    ) -> Option<usize> {
        let tree = self
            .string_trees
            .get(model_id)
            .map(|entry| entry.value().clone());

        if let Some(tree) = tree {
            // Single tree descent: match, pick the worker from the match result,
            // then insert for it — replacing the former match_prefix_with_counts
            // + insert_text pair. Selection logic is unchanged (see the token
            // path for the per-branch rationale).
            let mut selected_idx: Option<usize> = None;
            let result = tree.match_and_insert_with(text, |result| {
                let match_rate = if result.input_char_count == 0 {
                    0.0
                } else {
                    result.matched_char_count as f32 / result.input_char_count as f32
                };

                selected_idx = if match_rate > self.config.cache_threshold {
                    // Cache hit: pressure-select among the tenants holding the
                    // matched prefix. "Healthy" excludes circuit-broken workers,
                    // so a fully CB-tripped tenant set falls through to
                    // first-healthy without inserting (intended). A gated
                    // selection lands on min-load, and the insert below makes
                    // the spill target a new tenant of this prefix.
                    self.select_matched_candidate(
                        workers,
                        healthy_indices,
                        &result.matched_tenants,
                        text.chars().count(),
                        avg_load,
                        min_load_idx,
                    )
                } else {
                    // Partial match below threshold: pass the uncached
                    // char count (input - matched) so the length strategy
                    // can classify by actual prefill work, not full size.
                    // When matched == 0 (no match at all), pass None so the
                    // strategy falls through to its normal priority chain.
                    let uncached_hint = (result.matched_char_count > 0).then(|| {
                        UncachedHint::Chars(
                            result
                                .input_char_count
                                .saturating_sub(result.matched_char_count),
                        )
                    });
                    self.resolve_no_cache(
                        workers,
                        info,
                        healthy_indices,
                        min_load_idx,
                        avg_load,
                        model_id,
                        uncached_hint,
                    )
                };

                // Insert for the selected worker (None => no insert, exactly
                // like the old `if let Some(idx)` guard around insert_text).
                selected_idx.map(|idx| workers[idx].url())
            });

            self.log_tree_decision(
                selected_idx.map(|idx| workers[idx].url()),
                healthy_indices.first().map(|&idx| workers[idx].url()),
                result.matched_char_count,
                result.input_char_count,
                &result.matched_tenants,
                model_id,
            );

            if let Some(idx) = selected_idx {
                // Record hash(full_text)→matched_prefix for mesh tenant delta
                // resolution. The hash key matches what sync_tree_operation sends
                // on the wire (hash of full text). The VALUE is only the matched
                // prefix (~50-200 chars), not the full prompt (20KB+). When a
                // remote delta arrives, we look up the hash and call
                // insert_text(matched_prefix, worker) which routes to the same
                // tree node. This keeps the index memory-bounded.
                if self.should_populate_hash_index() {
                    let matched_prefix: String =
                        text.chars().take(result.matched_char_count).collect();
                    let path_hash = kv_index::hash_node_path(text);
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .string_tree
                        .insert(path_hash, matched_prefix);
                    self.sync_local_insert(
                        model_id,
                        TreeDelta {
                            tree_kind: TreeKind::String,
                            node_hash: path_hash,
                            worker_url: workers[idx].url().to_string(),
                            epoch: 0,
                        },
                    );
                }

                workers[idx].increment_processed();
                return Some(idx);
            }

            // Selected worker no longer exists or unhealthy - fall back to first healthy
            // Stale entries will be cleaned up by LRU eviction
            healthy_indices.first().copied()
        } else {
            let idx = healthy_indices[rand::rng().random_range(0..healthy_indices.len())];
            debug!(
                branch = "no_tree_random",
                worker = workers[idx].url(),
                model_id,
                "Cache-aware selection"
            );
            Some(idx)
        }
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use kv_index::{compute_content_hash, SequenceHash, StoredBlock, WorkerBlockMap};
    use openai_protocol::worker::{HealthCheckConfig, SchedulerLoadSnapshot, WorkerStatus};
    use tracing_test::traced_test;

    use super::*;

    /// Neutral tuning: decay and temperature off, no load snapshot — the
    /// historical selection behavior.
    fn default_tuning() -> OverlapTuning<'static> {
        OverlapTuning {
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            waiting_prefill_tokens: None,
        }
    }
    use crate::worker::{BasicWorkerBuilder, WorkerType};

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_cache_aware_with_balanced_load() {
        // Create policy without eviction thread for testing
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .api_key("test_api_key")
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        // Initialize the policy with workers
        policy.init_workers(&workers);

        // First request should be distributed
        let idx1 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .unwrap();

        // Same request should go to same worker (cache hit)
        let idx2 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello world"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx1, idx2);

        // Similar request should also go to same worker
        let idx3 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("hello"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx1, idx3);
    }

    /// `RoutingState::overloaded` has exactly one production reader — the fused
    /// gather below `select_worker`. Without this test nothing would fail if
    /// `state.eligible()` were reverted to health + circuit breaker, or if
    /// `BasicWorker::routing_state` stopped populating the field.
    #[test]
    fn overloaded_worker_is_vetoed_by_the_fused_gather() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = ["http://w1:8000", "http://w2:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(url)
                        .worker_type(WorkerType::Regular)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        policy.init_workers(&workers);

        let info = SelectWorkerInfo {
            request_text: Some("a stable prefix that pins one worker"),
            ..Default::default()
        };
        let owner = policy.select_worker(&workers, &info).unwrap();
        assert_eq!(
            policy.select_worker(&workers, &info),
            Some(owner),
            "cache affinity holds while the worker is eligible"
        );

        workers[owner].set_overloaded(true);
        assert_ne!(
            policy.select_worker(&workers, &info),
            Some(owner),
            "affinity must not outrank the absolute veto"
        );

        // Every worker vetoed leaves nothing to select.
        for worker in &workers {
            worker.set_overloaded(true);
        }
        assert_eq!(policy.select_worker(&workers, &info), None);

        for worker in &workers {
            worker.set_overloaded(false);
        }
        assert!(policy.select_worker(&workers, &info).is_some());
    }

    #[test]
    fn test_cache_aware_with_imbalanced_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0, // Disable eviction thread
            max_tree_size: 10000,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            ..Default::default()
        });

        let worker1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        let worker2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();

        // Create significant load imbalance
        for _ in 0..20 {
            worker1.increment_load();
        }
        // worker2 has load 0

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];
        policy.init_workers(&workers);

        // Should select worker2 (lower load) despite cache affinity
        let info = SelectWorkerInfo {
            request_text: Some("test"),
            ..Default::default()
        };
        for _ in 0..5 {
            let idx = policy.select_worker(&workers, &info).unwrap();
            assert_eq!(idx, 1); // Should always pick worker2
        }
    }

    // ---- tree size aggregation (per-model gauges + eviction-cycle log) ----

    #[test]
    fn string_tree_totals_sums_tenant_chars() {
        let tree = Tree::new();
        assert_eq!(string_tree_totals(&tree), (0, 0));

        // Worker registration (empty insert) registers a zero-char tenant.
        tree.insert_text("", "http://w1:8000");
        assert_eq!(string_tree_totals(&tree), (0, 1));

        tree.insert_text("hello", "http://w1:8000");
        assert_eq!(string_tree_totals(&tree), (5, 1));

        // A shared path counts its chars for every tenant on it.
        tree.insert_text("hello", "http://w2:8000");
        assert_eq!(string_tree_totals(&tree), (10, 2));

        // "help!" shares "hel", adds only "p!" for w1.
        tree.insert_text("help!", "http://w1:8000");
        assert_eq!(string_tree_totals(&tree), (12, 2));
    }

    #[test]
    fn token_tree_totals_sums_tenant_tokens() {
        let tree = TokenTree::new();
        assert_eq!(token_tree_totals(&tree), (0, 0));

        // Worker registration (empty insert) aligns to zero pages —
        // no tenant entry.
        tree.insert_tokens(&[], "http://w1:8000");
        assert_eq!(token_tree_totals(&tree), (0, 0));

        let page = tree.page_size();
        let page_a: Vec<u32> = (0..page as u32).collect();
        tree.insert_tokens(&page_a, "http://w1:8000");
        assert_eq!(token_tree_totals(&tree), (page, 1));

        // A shared page counts its tokens for every tenant on it.
        tree.insert_tokens(&page_a, "http://w2:8000");
        assert_eq!(token_tree_totals(&tree), (2 * page, 2));

        // A disjoint page adds only for its tenant.
        let page_b: Vec<u32> = (1000..1000 + page as u32).collect();
        tree.insert_tokens(&page_b, "http://w1:8000");
        assert_eq!(token_tree_totals(&tree), (3 * page, 2));
    }

    // ---- is_kv_imbalanced: KV triggers (overload ∨ KV-spread) ----

    /// Single-DP load snapshot reporting the given KV utilization (0.0–1.0).
    fn kv_load(token_usage: f64) -> WorkerLoadResponse {
        WorkerLoadResponse {
            loads: vec![SchedulerLoadSnapshot {
                token_usage,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// Healthy workers (health checks disabled) for the given URLs.
    fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|u| {
                Arc::new(
                    BasicWorkerBuilder::new(*u)
                        .worker_type(WorkerType::Regular)
                        .health_config(no_health_check())
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    /// Inject a backend KV snapshot (utilization per worker, by index). Returns
    /// the sender; bind it (`let _tx = ...`) to keep the watch channel open.
    fn inject_kv(
        policy: &CacheAwarePolicy,
        workers: &[Arc<dyn Worker>],
        usages: &[f64],
    ) -> watch::Sender<HashMap<String, WorkerLoadResponse>> {
        let map: HashMap<String, WorkerLoadResponse> = workers
            .iter()
            .zip(usages)
            .map(|(w, &u)| (w.url().to_string(), kv_load(u)))
            .collect();
        let (tx, rx) = watch::channel(map);
        policy.set_load_receiver(Some(rx));
        tx
    }

    /// Config isolating the KV triggers (count effectively disabled): `balance`
    /// is the spread threshold, `overload` the ceiling.
    fn kv_only_config(balance_spread: f32, overload_ceiling: f32) -> CacheAwareConfig {
        CacheAwareConfig {
            balance_abs_threshold: usize::MAX,
            eviction_interval_secs: 0,
            balance_token_usage_threshold: balance_spread,
            overload_token_usage_threshold: overload_ceiling,
            ..Default::default()
        }
    }

    fn all_healthy(workers: &[Arc<dyn Worker>]) -> Vec<usize> {
        (0..workers.len()).collect()
    }

    fn imbalanced(policy: &CacheAwarePolicy, workers: &[Arc<dyn Worker>]) -> bool {
        policy.is_kv_imbalanced(workers, &all_healthy(workers))
    }

    #[test]
    fn is_kv_imbalanced_uniform_high_kv_does_not_fire() {
        // All engines equally saturated: high utilization, zero spread.
        let policy = CacheAwarePolicy::with_config(kv_only_config(0.3, 0.95));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let _tx = inject_kv(&policy, &workers, &[0.9, 0.9, 0.9]);
        // max 0.9 < 0.95 ceiling, spread 0.0 < 0.3 → keep cache affinity.
        assert!(
            !imbalanced(&policy, &workers),
            "uniform-high KV (no cooler home) must not abandon cache affinity"
        );
    }

    #[test]
    fn is_kv_imbalanced_one_hot_rest_idle_fires_via_spread() {
        // Same hottest engine (0.9) as the uniform case, but neighbors are idle.
        let policy = CacheAwarePolicy::with_config(kv_only_config(0.3, 0.95));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let _tx = inject_kv(&policy, &workers, &[0.9, 0.15, 0.15]);
        // spread 0.75 > 0.3 → spill toward a cooler engine.
        assert!(
            imbalanced(&policy, &workers),
            "a hot engine with idle neighbors (large KV spread) must rebalance"
        );
    }

    #[test]
    fn is_kv_imbalanced_overload_ceiling_fires_below_spread() {
        // Critically hot engine, but the spread is under the balance threshold.
        let policy = CacheAwarePolicy::with_config(kv_only_config(0.3, 0.95));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let _tx = inject_kv(&policy, &workers, &[0.97, 0.80]);
        // spread 0.17 < 0.3 (balance quiet) but 0.97 > 0.95 ceiling → shed.
        assert!(
            imbalanced(&policy, &workers),
            "a critically-saturated engine must shed even below the spread threshold"
        );
    }

    #[test]
    fn is_kv_imbalanced_ignores_request_count_spread() {
        // Count dispersion alone never abandons affinity fleet-wide; count
        // pressure is applied per request by the candidate gate instead.
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0,
            balance_token_usage_threshold: 0.3,
            overload_token_usage_threshold: 0.95,
            ..Default::default()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let _tx = inject_kv(&policy, &workers, &[0.3, 0.3]);
        for _ in 0..20 {
            workers[0].increment_load();
        }
        assert!(
            !imbalanced(&policy, &workers),
            "count spread alone must not disable cache affinity fleet-wide"
        );
    }

    #[test]
    fn is_kv_imbalanced_kv_disabled_by_default_ignores_snapshot() {
        // Default config: both KV thresholds 1.0 (disabled).
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        // A massive KV spread that WOULD fire if KV balancing were enabled...
        let _tx = inject_kv(&policy, &workers, &[0.95, 0.05]);
        // ...is ignored at the 1.0 default; counts balanced → no rebalance.
        assert!(
            !imbalanced(&policy, &workers),
            "default thresholds (1.0) must ignore KV usage entirely"
        );
    }

    // ---- per-request candidate gate + matched-tenant selection ----

    /// Route once so `model_key`-scoped trees exist, then seed the token tree
    /// with the given tenants for `tokens`.
    fn seed_token_tenants(
        policy: &CacheAwarePolicy,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        tenant_urls: &[&str],
    ) -> Arc<TokenTree> {
        policy.init_workers(workers);
        let model_key = normalize_model_key(workers[0].model_id()).to_string();
        let tree = Arc::clone(policy.token_trees.get(&model_key).unwrap().value());
        for url in tenant_urls {
            tree.insert_tokens(tokens, url);
        }
        tree
    }

    #[test]
    fn cache_hit_prefers_least_loaded_matched_tenant() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let tokens: Vec<u32> = (0..32).collect();
        seed_token_tenants(
            &policy,
            &workers,
            &tokens,
            &["http://w1:8000", "http://w2:8000"],
        );
        for _ in 0..5 {
            workers[0].increment_load();
        }

        // w2 and w3 are equally idle, but only w1/w2 hold the prefix: the
        // selection must stay within the matched tenants and take the less
        // loaded one.
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1, "least-loaded matched tenant, not any idle worker");
    }

    #[test]
    fn gated_hot_tenant_spills_to_min_load_and_replicates() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let tokens: Vec<u32> = (0..32).collect();
        let tree = seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
        // avg = 50; w1 clears both margins (100 > 50 * 1.1 and 100 > 50 + 32).
        for _ in 0..100 {
            workers[0].increment_load();
        }

        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1, "gated selection must spill to the min-load worker");

        // The spill inserted for w2, so the prefix now has a second tenant.
        let result = tree.match_prefix_with_counts(&tokens);
        assert!(
            result
                .matched_tenants
                .iter()
                .any(|tenant| tenant.as_ref() == "http://w2:8000"),
            "spill target must become a tenant of the hot prefix"
        );
    }

    #[test]
    #[traced_test]
    fn token_tree_selection_emits_decision_line() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let tokens: Vec<u32> = (0..32).collect();
        seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);

        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(logs_contain("Cache-aware selection"));
        assert!(logs_contain("tree_match"));

        let novel: Vec<u32> = (1000..1064).collect();
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&novel),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(logs_contain("min_load_fallback"));
    }

    #[test]
    fn count_spread_elsewhere_keeps_cache_affinity() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let tokens: Vec<u32> = (0..32).collect();
        seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
        // Fleet-wide count spread (50 vs 0) that formerly disabled affinity
        // outright — but the loaded worker is not the request's tenant.
        for _ in 0..50 {
            workers[1].increment_load();
        }

        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(
            idx, 0,
            "a deep queue on another worker must not break this request's affinity"
        );
    }

    #[test]
    fn candidate_gate_requires_both_margins() {
        // w1 load 10 vs avg 5: over the relative margin (10 > 5 * 1.1) but
        // under the absolute one (10 < 5 + 32) — affinity holds.
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let tokens: Vec<u32> = (0..32).collect();
        seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
        for _ in 0..10 {
            workers[0].increment_load();
        }
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        assert_eq!(policy.select_worker(&workers, &info).unwrap(), 0);

        // Same loads with a small absolute margin (10 > 5 + 2): spill.
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            balance_abs_threshold: 2,
            ..test_config()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
        for _ in 0..10 {
            workers[0].increment_load();
        }
        let info = SelectWorkerInfo {
            tokens: Some(&tokens),
            ..Default::default()
        };
        assert_eq!(policy.select_worker(&workers, &info).unwrap(), 1);
    }

    #[test]
    fn kv_pressure_still_forces_min_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            block_size: 4,
            ..kv_only_config(0.3, 0.95)
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let tokens: Vec<u32> = (0..32).collect();
        seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
        workers[0].increment_load();
        // KV spread 0.8 > 0.3: shed fleet-wide despite the w1 cache hit.
        let _tx = inject_kv(&policy, &workers, &[0.9, 0.1]);

        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1, "KV pressure must still override cache affinity");
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0, // Disable eviction thread
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        policy.init_workers(&workers);

        // Route some requests
        policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("test1"),
                ..Default::default()
            },
        );
        policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("test2"),
                ..Default::default()
            },
        );

        // Remove a worker
        policy.remove_worker_by_url("http://w1:8000");
        workers[0].set_status(WorkerStatus::NotReady);

        // All requests should now go to worker2
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test1"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_remove_worker_purges_tree_tenants() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        policy.init_workers(&workers);

        // Distinct cold inputs spread across both workers (min-load
        // tie-breaks by processed count), populating both trees.
        for text in ["purge me please", "keep me around"] {
            policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        request_text: Some(text),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        let tokens_a: Vec<u32> = (0..32).collect();
        let tokens_b: Vec<u32> = (1000..1032).collect();
        for tokens in [&tokens_a, &tokens_b] {
            policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(tokens),
                        ..Default::default()
                    },
                )
                .unwrap();
        }

        let model_key = normalize_model_key(workers[0].model_id()).to_string();
        let string_tree = Arc::clone(policy.string_trees.get(&model_key).unwrap().value());
        let token_tree = Arc::clone(policy.token_trees.get(&model_key).unwrap().value());

        let char_counts = string_tree.get_tenant_char_count();
        let token_counts = token_tree.get_tenant_token_counts();
        for url in ["http://w1:8000", "http://w2:8000"] {
            assert!(char_counts.contains_key(url), "precondition: {url} routed");
            assert!(token_counts.contains_key(url), "precondition: {url} routed");
        }

        policy.remove_worker(workers[0].as_ref());

        let char_counts = string_tree.get_tenant_char_count();
        assert!(!char_counts.contains_key("http://w1:8000"));
        assert!(char_counts.contains_key("http://w2:8000"));
        let token_counts = token_tree.get_tenant_token_counts();
        assert!(!token_counts.contains_key("http://w1:8000"));
        assert!(token_counts.contains_key("http://w2:8000"));

        policy.remove_worker_by_url("http://w2:8000");
        assert!(string_tree.get_tenant_char_count().is_empty());
        assert!(token_tree.get_tenant_token_counts().is_empty());
    }

    #[test]
    fn test_apply_known_remote_insert_round_trip() {
        // Seed both kinds via `apply_repair_page` (the v2 cold-start
        // path that populates hash_index), then verify
        // `apply_known_remote_insert` resolves the hash and returns
        // true. Unknown hashes return false. Wrong-kind lookups
        // against the same hash return false (model + kind scope
        // the index).
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let text = "remote_text";
        let tokens = vec![1u32, 2, 3, 4];
        let string_page = TreeRepairPage {
            session_id: uuid::Uuid::now_v7(),
            model_id: "model1".to_string(),
            tree_kind: TreeKind::String,
            page_index: 0,
            entries: vec![RepairEntry::String {
                path: text.to_string(),
                tenants: vec![(Arc::from("http://w1"), 1)],
            }],
            next_cursor: None,
            is_last: true,
        };
        assert_eq!(policy.apply_repair_page(&string_page), 1);

        let token_page = TreeRepairPage {
            session_id: uuid::Uuid::now_v7(),
            model_id: "model1".to_string(),
            tree_kind: TreeKind::Token,
            page_index: 0,
            entries: vec![RepairEntry::Token {
                tokens: tokens.clone(),
                tenants: vec![(Arc::from("http://w1"), 1)],
            }],
            next_cursor: None,
            is_last: true,
        };
        assert_eq!(policy.apply_repair_page(&token_page), 1);

        let text_hash = kv_index::hash_node_path(text);
        let token_hash = kv_index::hash_token_path(&tokens);

        // Known hashes apply for the matching kind.
        assert!(policy.apply_known_remote_insert(
            "model1",
            TreeKind::String,
            text_hash,
            "http://w2",
        ));
        assert!(policy.apply_known_remote_insert(
            "model1",
            TreeKind::Token,
            token_hash,
            "http://w2",
        ));

        // Same hash but wrong kind doesn't alias.
        assert!(!policy.apply_known_remote_insert(
            "model1",
            TreeKind::Token,
            text_hash,
            "http://w2",
        ));

        // Unknown hash, unknown model → false.
        assert!(!policy.apply_known_remote_insert(
            "model1",
            TreeKind::String,
            0xDEAD_BEEF,
            "http://w2",
        ));
        assert!(!policy.apply_known_remote_insert(
            "unknown_model",
            TreeKind::String,
            text_hash,
            "http://w2",
        ));
    }

    #[test]
    fn test_apply_repair_page_seeds_hash_index() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);
        let text = "repaired text";
        let tokens = vec![1u32; 16];

        let string_page = TreeRepairPage {
            session_id: uuid::Uuid::now_v7(),
            model_id: "model1".to_string(),
            tree_kind: TreeKind::String,
            page_index: 0,
            entries: vec![RepairEntry::String {
                path: text.to_string(),
                tenants: vec![(Arc::from("http://w1"), 1)],
            }],
            next_cursor: None,
            is_last: true,
        };
        assert_eq!(policy.apply_repair_page(&string_page), 1);
        assert!(policy.apply_known_remote_insert(
            "model1",
            TreeKind::String,
            kv_index::hash_node_path(text),
            "http://w2",
        ));

        let token_page = TreeRepairPage {
            session_id: uuid::Uuid::now_v7(),
            model_id: "model1".to_string(),
            tree_kind: TreeKind::Token,
            page_index: 0,
            entries: vec![RepairEntry::Token {
                tokens: tokens.clone(),
                tenants: vec![(Arc::from("http://w1"), 1)],
            }],
            next_cursor: None,
            is_last: true,
        };
        assert_eq!(policy.apply_repair_page(&token_page), 1);
        assert!(policy.apply_known_remote_insert(
            "model1",
            TreeKind::Token,
            kv_index::hash_token_path(&tokens),
            "http://w2",
        ));
    }

    #[test]
    fn test_apply_known_remote_insert_from_request_hot_path() {
        // Companion to `test_apply_known_remote_insert_round_trip`.
        // That test seeds via `apply_repair_page`, which stores
        // full text/tokens. The local request hot path
        // (`select_worker_with_text` / `_with_tokens` plus the
        // imbalanced fallback) stores the *matched prefix* shape
        // instead. A regression on the matched-prefix apply path
        // would still pass the full-path test, so seed via
        // `select_worker` here and assert apply succeeds.
        //
        // Opt into request-hot-path hash_index population — without
        // this the populate sites are no-ops and the apply call
        // below would have nothing to resolve. In production this
        // flag is flipped by the mesh wiring code; here we set it
        // directly because the test mimics the mesh consumer.
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        policy.set_populate_hash_index(true);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        policy.init_workers(&workers);

        // Drive a string request through select_worker — populates
        // the string-side hash_index with a matched-prefix value.
        let text = "the quick brown fox jumps over the lazy dog";
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some(text),
                    ..Default::default()
                },
            )
            .unwrap();
        let text_hash = kv_index::hash_node_path(text);

        // Drive a token request — populates the token-side
        // hash_index. select_worker uses the model_id from the
        // first worker's `model_id()`, which the builder leaves
        // empty → UNKNOWN_MODEL_ID after normalization.
        let tokens: Vec<u32> = (0..32).collect();
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        let token_hash = kv_index::hash_token_path(&tokens);

        // Both populate sites use UNKNOWN_MODEL_ID for these
        // workers (no model_id set on the builder), and the
        // resolver normalizes empty → UNKNOWN_MODEL_ID, so an
        // empty model_id resolves the same entries the populate
        // sites wrote.
        assert!(policy.apply_known_remote_insert("", TreeKind::String, text_hash, "http://remote",));
        assert!(policy.apply_known_remote_insert("", TreeKind::Token, token_hash, "http://remote",));
    }

    #[test]
    fn test_cache_aware_without_mesh() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwarePolicy::with_config(config);

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .health_config(no_health_check())
                .build(),
        )];

        policy.init_workers(&workers);

        // Should work without mesh
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("test request"),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 0);
    }

    // -----------------------------------------------------------------------
    // Event-driven routing tests (Type 1: PositionalIndexer overlap scoring)
    // -----------------------------------------------------------------------

    /// Helper: create a PositionalIndexer and store blocks for a worker.
    /// `token_chunks` is a list of token-id slices — each becomes one block.
    fn setup_indexer_with_blocks(
        worker_url: &str,
        token_chunks: &[&[u32]],
        jump_size: usize,
    ) -> Arc<PositionalIndexer> {
        let indexer = Arc::new(PositionalIndexer::new(jump_size));
        let worker_id = indexer.intern_worker(worker_url).unwrap();
        let mut wb = WorkerBlockMap::default();
        let blocks: Vec<StoredBlock> = token_chunks
            .iter()
            .enumerate()
            .map(|(i, tokens)| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash: compute_content_hash(tokens),
            })
            .collect();
        indexer
            .apply_stored(worker_id, &blocks, None, &mut wb)
            .unwrap();
        indexer
    }

    fn test_config() -> CacheAwareConfig {
        CacheAwareConfig {
            eviction_interval_secs: 0,
            block_size: 4, // small block size for easy test setup
            ..Default::default()
        }
    }

    // -- score_overlap unit tests (scoring helper) --

    #[test]
    fn test_score_overlap_selects_best_match() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        policy.init_workers(&workers);

        // Store 4 blocks for w1: tokens [1..16] in blocks of 4
        let indexer = setup_indexer_with_blocks(
            "http://w1:8000",
            &[
                &[1, 2, 3, 4],
                &[5, 6, 7, 8],
                &[9, 10, 11, 12],
                &[13, 14, 15, 16],
            ],
            4,
        );

        // Query with matching tokens — should select w1
        let result = CacheAwarePolicy::score_overlap(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            &[0, 1],
            &indexer,
            4,
            &default_tuning(),
        );
        assert_eq!(result, Some(0)); // w1
    }

    #[test]
    fn test_score_overlap_random_tie_break_spreads_equal_workers() {
        // Two workers with identical cached blocks and equal load: the pick
        // must not be deterministic (equal candidates herded onto one worker
        // before), so across many draws both must be selected.
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];

        let chunks: [&[u32]; 2] = [&[1, 2, 3, 4], &[5, 6, 7, 8]];
        let indexer = setup_indexer_with_blocks("http://w1:8000", &chunks, 4);
        // Same content cached on w2 under distinct backend seq hashes.
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb2 = WorkerBlockMap::default();
        let blocks: Vec<StoredBlock> = chunks
            .iter()
            .enumerate()
            .map(|(i, tokens)| StoredBlock {
                seq_hash: SequenceHash(100 + i as u64),
                content_hash: compute_content_hash(tokens),
            })
            .collect();
        indexer.apply_stored(w2, &blocks, None, &mut wb2).unwrap();

        let mut seen = [false; 2];
        for _ in 0..200 {
            let idx = CacheAwarePolicy::score_overlap(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &default_tuning(),
            )
            .expect("both workers fully match");
            seen[idx] = true;
            if seen[0] && seen[1] {
                break;
            }
        }
        assert!(
            seen[0] && seen[1],
            "equal-overlap equal-load tie must spread across workers, saw only one"
        );
    }

    /// Two workers with identical cached blocks (the tie-test topology): both
    /// fully match the request.
    fn equal_overlap_fixture() -> (Vec<Arc<dyn Worker>>, Arc<PositionalIndexer>) {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        let chunks: [&[u32]; 2] = [&[1, 2, 3, 4], &[5, 6, 7, 8]];
        let indexer = setup_indexer_with_blocks("http://w1:8000", &chunks, 4);
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb2 = WorkerBlockMap::default();
        let blocks: Vec<StoredBlock> = chunks
            .iter()
            .enumerate()
            .map(|(i, tokens)| StoredBlock {
                seq_hash: SequenceHash(100 + i as u64),
                content_hash: compute_content_hash(tokens),
            })
            .collect();
        indexer.apply_stored(w2, &blocks, None, &mut wb2).unwrap();
        (workers, indexer)
    }

    #[test]
    fn test_overlap_decay_prefers_less_backlogged_worker() {
        // Equal overlap, equal load — but w2 carries waiting-prefill backlog.
        // With decay on, the fleet-floor worker keeps full credit and must
        // win every draw (previously this tie was a coin flip).
        let (workers, indexer) = equal_overlap_fixture();
        let waiting = HashMap::from([
            ("http://w1:8000".to_string(), 0),
            ("http://w2:8000".to_string(), 8),
        ]);
        let tuning = OverlapTuning {
            overlap_decay: 4.0,
            selection_temperature: 0.0,
            waiting_prefill_tokens: Some(&waiting),
        };
        for _ in 0..50 {
            let idx = CacheAwarePolicy::score_overlap(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &tuning,
            )
            .expect("both workers match");
            assert_eq!(idx, 0, "backlogged worker must lose its credit edge");
        }
    }

    #[test]
    fn test_overlap_decay_missing_load_data_never_decays() {
        // Only w1 reports load (and holds the floor at zero backlog); w2 has
        // no entry. Neither may be decayed, so the equal-score tie — and its
        // random spreading — must survive.
        let (workers, indexer) = equal_overlap_fixture();
        let waiting = HashMap::from([("http://w1:8000".to_string(), 0)]);
        let tuning = OverlapTuning {
            overlap_decay: 4.0,
            selection_temperature: 0.0,
            waiting_prefill_tokens: Some(&waiting),
        };
        let mut seen = [false; 2];
        for _ in 0..200 {
            let idx = CacheAwarePolicy::score_overlap(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &tuning,
            )
            .expect("both workers match");
            seen[idx] = true;
            if seen[0] && seen[1] {
                break;
            }
        }
        assert!(
            seen[0] && seen[1],
            "workers without load data must not be decayed (tie must survive)"
        );
    }

    /// w1 caches both request blocks (score 2), w2 only the first (score 1).
    fn unequal_overlap_fixture() -> (Vec<Arc<dyn Worker>>, Arc<PositionalIndexer>) {
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        let indexer =
            setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]], 4);
        let w2 = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb2 = WorkerBlockMap::default();
        let blocks = vec![StoredBlock {
            seq_hash: SequenceHash(100),
            content_hash: compute_content_hash(&[1, 2, 3, 4]),
        }];
        indexer.apply_stored(w2, &blocks, None, &mut wb2).unwrap();
        (workers, indexer)
    }

    #[test]
    fn test_selection_temperature_spreads_but_favors_better_score() {
        // At temperature 0 the better scorer wins every draw; at temperature
        // 1.0 the weaker scorer must be sampled sometimes, while the better
        // one keeps the majority (p(best) = 1/(1+e^-1) ≈ 0.73).
        let (workers, indexer) = unequal_overlap_fixture();
        for _ in 0..50 {
            let idx = CacheAwarePolicy::score_overlap(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &default_tuning(),
            )
            .expect("both workers match");
            assert_eq!(idx, 0, "temperature 0 must be exact argmax");
        }

        let tuning = OverlapTuning {
            overlap_decay: 0.0,
            selection_temperature: 1.0,
            waiting_prefill_tokens: None,
        };
        let mut counts = [0usize; 2];
        for _ in 0..300 {
            let idx = CacheAwarePolicy::score_overlap(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &tuning,
            )
            .expect("both workers match");
            counts[idx] += 1;
        }
        assert!(
            counts[1] > 0,
            "temperature must spread picks to the weaker scorer"
        );
        assert!(
            counts[0] > counts[1],
            "better score must keep the majority: {counts:?}"
        );
    }

    #[test]
    fn test_selection_temperature_uniform_on_equal_scores() {
        // Degenerate spread (all candidates equal): the draw is uniform, so
        // both workers must appear.
        let (workers, indexer) = equal_overlap_fixture();
        let tuning = OverlapTuning {
            overlap_decay: 0.0,
            selection_temperature: 0.5,
            waiting_prefill_tokens: None,
        };
        let mut seen = [false; 2];
        for _ in 0..200 {
            let idx = CacheAwarePolicy::score_overlap(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &tuning,
            )
            .expect("both workers match");
            seen[idx] = true;
            if seen[0] && seen[1] {
                break;
            }
        }
        assert!(
            seen[0] && seen[1],
            "equal scores under temperature must draw uniformly"
        );
    }

    #[test]
    fn test_score_overlap_no_match_returns_none() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        )];
        policy.init_workers(&workers);

        let indexer =
            setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]], 4);

        // Completely different tokens — no overlap → None
        let result = CacheAwarePolicy::score_overlap(
            &workers,
            &[100, 200, 300, 400, 500, 600, 700, 800],
            &[0],
            &indexer,
            4,
            &default_tuning(),
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_score_overlap_load_tiebreak() {
        let policy = CacheAwarePolicy::with_config(test_config());

        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();

        // Give w1 higher load
        for _ in 0..10 {
            w1.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        // Store same blocks for both workers (equal overlap)
        let indexer = Arc::new(PositionalIndexer::new(4));
        let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
        let w2_id = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();
        let blocks = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4]),
        }];
        indexer
            .apply_stored(w1_id, &blocks, None, &mut wb1)
            .unwrap();
        let blocks2 = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4]),
        }];
        indexer
            .apply_stored(w2_id, &blocks2, None, &mut wb2)
            .unwrap();

        // Equal overlap → tie-break by load → w2 wins (lower load)
        let result = CacheAwarePolicy::score_overlap(
            &workers,
            &[1, 2, 3, 4],
            &[0, 1],
            &indexer,
            4,
            &default_tuning(),
        );
        assert_eq!(result, Some(1)); // w2 (lower load)
    }

    #[test]
    fn test_score_overlap_tree_size_not_a_tiebreak() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        policy.init_workers(&workers);

        let indexer = Arc::new(PositionalIndexer::new(4));
        let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
        let w2_id = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();

        // Both workers have block [1,2,3,4] (equal overlap, equal load)
        let block = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4]),
        }];
        indexer.apply_stored(w1_id, &block, None, &mut wb1).unwrap();

        // w2 has the same block plus extra blocks → larger tree
        let block2 = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4]),
        }];
        indexer
            .apply_stored(w2_id, &block2, None, &mut wb2)
            .unwrap();
        let extra = vec![StoredBlock {
            seq_hash: SequenceHash(2),
            content_hash: compute_content_hash(&[5, 6, 7, 8]),
        }];
        indexer
            .apply_stored(w2_id, &extra, Some(SequenceHash(1)), &mut wb2)
            .unwrap();

        // Equal overlap, equal load, different tree sizes: tree size is no
        // longer a tie-break key — the pick is uniform over the tie, so both
        // workers must appear across draws (the old smaller-tree preference
        // herded every tie onto w1 until the global size ordering flipped).
        let mut seen = [false; 2];
        for _ in 0..200 {
            let idx = CacheAwarePolicy::score_overlap(
                &workers,
                &[1, 2, 3, 4],
                &[0, 1],
                &indexer,
                4,
                &default_tuning(),
            )
            .expect("both workers match");
            seen[idx] = true;
            if seen[0] && seen[1] {
                break;
            }
        }
        assert!(
            seen[0] && seen[1],
            "tie must spread over both workers regardless of tree size, saw {seen:?}"
        );
    }

    #[test]
    fn test_score_overlap_short_request_returns_none() {
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        )];

        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);

        // Request shorter than block_size → no full blocks → None
        let result = CacheAwarePolicy::score_overlap(
            &workers,
            &[1, 2, 3],
            &[0],
            &indexer,
            4,
            &default_tuning(),
        );
        assert_eq!(result, None);
    }

    #[test]
    fn test_score_overlap_partial_match() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        policy.init_workers(&workers);

        let indexer = Arc::new(PositionalIndexer::new(4));
        let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
        let w2_id = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlockMap::default();
        let mut wb2 = WorkerBlockMap::default();

        // w1 has 4 blocks cached
        let blocks_w1: Vec<StoredBlock> = (0..4)
            .map(|i| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash: compute_content_hash(&[
                    (i * 4 + 1) as u32,
                    (i * 4 + 2) as u32,
                    (i * 4 + 3) as u32,
                    (i * 4 + 4) as u32,
                ]),
            })
            .collect();
        indexer
            .apply_stored(w1_id, &blocks_w1, None, &mut wb1)
            .unwrap();

        // w2 has only the first 2 blocks (partial overlap with same request)
        let blocks_w2: Vec<StoredBlock> = (0..2)
            .map(|i| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash: compute_content_hash(&[
                    (i * 4 + 1) as u32,
                    (i * 4 + 2) as u32,
                    (i * 4 + 3) as u32,
                    (i * 4 + 4) as u32,
                ]),
            })
            .collect();
        indexer
            .apply_stored(w2_id, &blocks_w2, None, &mut wb2)
            .unwrap();

        // Query with all 4 blocks worth of tokens → w1 wins (higher overlap: 4 vs 2)
        let result = CacheAwarePolicy::score_overlap(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            &[0, 1],
            &indexer,
            4,
            &default_tuning(),
        );
        assert_eq!(result, Some(0)); // w1 (higher overlap)
    }

    // -- select_worker_event_driven integration tests --

    #[test]
    fn test_event_driven_overlap_selects_cached_worker() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        policy.init_workers(&workers);

        // Set up monitor with indexer data for "unknown" model
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer =
            setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        // Full dispatch: should use event-driven and select w1
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 0); // w1 (has cached blocks)
    }

    #[test]
    fn test_event_driven_no_overlap_uses_min_load() {
        let policy = CacheAwarePolicy::with_config(test_config());

        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        // Give w1 higher load so min-load picks w2
        for _ in 0..3 {
            w1.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        // Monitor has indexer with data, but tokens don't match
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        // No overlap → event-driven falls back to min-load (not token tree)
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[100, 200, 300, 400]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1); // w2 (min load), NOT token tree result
    }

    #[test]
    fn test_event_driven_gated_hot_winner_spills_to_min_load() {
        let policy = CacheAwarePolicy::with_config(test_config());

        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        // avg = 50; w1 clears both gate margins (100 > 50 * 1.1, 100 > 50 + 32).
        for _ in 0..100 {
            w1.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        // w1 wins the overlap score but is over both load margins: spill.
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1, "gated overlap winner must spill to min load");
    }

    #[test]
    fn test_event_driven_short_request_uses_min_load() {
        let policy = CacheAwarePolicy::with_config(test_config()); // block_size=4

        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        for _ in 0..3 {
            w1.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        // Request shorter than block_size → no full blocks → min-load fallback
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1); // w2 (min load)
    }

    #[test]
    fn test_no_monitor_uses_token_tree() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        policy.init_workers(&workers);

        // No kv_monitor → has_event_indexer returns false → uses token tree
        assert!(!policy.has_event_indexer("unknown"));

        // Should still route (via token tree, not event-driven)
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(idx < 2); // valid worker selected
    }

    #[test]
    fn test_set_kv_event_monitor() {
        let policy = CacheAwarePolicy::with_config(test_config());

        // Initially no monitor
        assert!(policy.kv_monitor.read().is_none());

        // Set monitor (works via &self thanks to interior mutability)
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        policy.set_kv_event_monitor(Some(Arc::clone(&monitor)));
        assert!(policy.kv_monitor.read().is_some());

        // get_indexer returns None for unknown model
        assert!(monitor.get_indexer("nonexistent").is_none());

        // Clear monitor
        policy.set_kv_event_monitor(None);
        assert!(policy.kv_monitor.read().is_none());
    }

    #[test]
    fn test_event_driven_uses_monitor_block_size() {
        // Test that event-driven routing uses monitor's learned block_size
        // instead of config default when available.
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            block_size: 4, // config default
            eviction_interval_secs: 0,
            ..Default::default()
        });

        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        policy.init_workers(&workers);

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));

        // Store blocks using block_size=8 (tokens chunked in groups of 8)
        let indexer = Arc::new(PositionalIndexer::new(4));
        let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlockMap::default();
        let block = vec![StoredBlock {
            seq_hash: SequenceHash(1),
            content_hash: compute_content_hash(&[1, 2, 3, 4, 5, 6, 7, 8]),
        }];
        indexer.apply_stored(w1_id, &block, None, &mut wb).unwrap();
        monitor
            .indexers
            .insert("unknown".to_string(), indexer.clone());

        // Set block_size=8 in monitor (simulating learned from events)
        monitor.set_block_size("unknown", 8);

        policy.set_kv_event_monitor(Some(monitor));

        // Query with 8 tokens — with block_size=8, this is one full block
        // With config block_size=4, this would be two blocks and wouldn't match
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 0); // w1 has the cached block
    }

    #[test]
    fn test_imbalanced_skips_event_driven() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 0,
            block_size: 4,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            ..Default::default()
        });

        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();

        // Create heavy imbalance: w1 has 20 load, w2 has 0
        for _ in 0..20 {
            w1.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        // Even though we set up event monitor, imbalance check fires first
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        policy.set_kv_event_monitor(Some(monitor));

        // With imbalance, select_worker should pick min-load (w2), not event-driven
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1); // w2 (min load), regardless of event data
    }

    #[test]
    fn test_empty_indexer_falls_through_to_token_tree() {
        // When the monitor has an indexer for a model but the indexer is empty
        // (startup, reconnect), routing should fall through to the token tree
        // instead of taking the event-driven path and landing on min-load.
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ),
        ];
        policy.init_workers(&workers);

        // Set up monitor with an empty indexer
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let empty_indexer = Arc::new(PositionalIndexer::new(4));
        monitor
            .indexers
            .insert("unknown".to_string(), empty_indexer);
        policy.set_kv_event_monitor(Some(monitor));

        // Empty indexer → has_event_indexer returns false → falls through to token tree
        assert!(!policy.has_event_indexer("unknown"));

        // Tokens must fill at least one tree page (the policy's block_size)
        // to populate the tree; shorter sequences are uncacheable and fall
        // through to min-load.
        let tokens: Vec<u32> = (1..=16).collect();

        // First request populates the token tree for the selected worker.
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(idx < 2); // valid worker via token tree

        // Same tokens again — token-tree cache hit routes to the same worker.
        let idx2 = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, idx2); // token tree cache affinity preserved
    }

    // ---- hash placement index (cache_index = hash) ----

    fn hash_config(boundaries: &[usize]) -> CacheAwareConfig {
        CacheAwareConfig {
            eviction_interval_secs: 0,
            cache_index: CacheIndexKind::Hash,
            cache_boundaries: boundaries.to_vec(),
            ..Default::default()
        }
    }

    fn route_tokens(
        policy: &CacheAwarePolicy,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
    ) -> usize {
        policy
            .select_worker(
                workers,
                &SelectWorkerInfo {
                    tokens: Some(tokens),
                    ..Default::default()
                },
            )
            .unwrap()
    }

    #[test]
    fn hash_mode_never_touches_trees() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        assert!(policy.string_trees.is_empty());
        assert!(policy.token_trees.is_empty());

        let tokens: Vec<u32> = (0..32).collect();
        route_tokens(&policy, &workers, &tokens);
        policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("a text prompt long enough to insert"),
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(policy.string_trees.is_empty());
        assert!(policy.token_trees.is_empty());
        assert!(!policy.placement_index.is_empty());
    }

    #[test]
    fn hash_mode_repeat_head_sticks_to_holder() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        let tokens: Vec<u32> = (0..32).collect();
        let first = route_tokens(&policy, &workers, &tokens);
        // Mild load on the holder must not break affinity (under the gate).
        workers[first].increment_load();
        workers[first].increment_load();
        for _ in 0..5 {
            assert_eq!(route_tokens(&policy, &workers, &tokens), first);
        }

        // A different head load-balances away from the loaded holder.
        let other: Vec<u32> = (1000..1032).collect();
        assert_ne!(route_tokens(&policy, &workers, &other), first);
    }

    #[test]
    fn hash_mode_probes_deepest_boundary_first() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16, 32]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let model = normalize_model_key(workers[0].model_id());
        let tokens: Vec<u32> = (0..40).collect();
        let now = Instant::now();

        // w2 holds the 16-token head, w1 the deeper 32-token head.
        policy.record_placement(model, &tokens, &[16], "http://w2:8000", now);
        policy.record_placement(model, &tokens, &[32], "http://w1:8000", now);
        // Even with the shallow holder strictly less loaded, the deeper
        // boundary must win.
        workers[0].increment_load();

        assert_eq!(route_tokens(&policy, &workers, &tokens), 0);
    }

    #[test]
    fn hash_mode_records_at_every_applicable_boundary() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16, 32, 64]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let model = normalize_model_key(workers[0].model_id());

        let tokens: Vec<u32> = (0..40).collect();
        let selected = route_tokens(&policy, &workers, &tokens);

        let placements = policy.placement_index.get(model).unwrap();
        // 64 exceeds the request length: only the two applicable levels.
        assert_eq!(placements.len(), 2);
        for boundary in [16usize, 32] {
            let key = (boundary, hash_token_head(&tokens[..boundary]));
            let holders = placements.get(&key).unwrap();
            assert_eq!(holders.len(), 1);
            assert_eq!(holders[0].worker_url, workers[selected].url());
        }
    }

    #[test]
    fn hash_mode_short_request_stays_min_load() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        for _ in 0..3 {
            workers[0].increment_load();
        }

        let tokens: Vec<u32> = (0..8).collect();
        for _ in 0..5 {
            assert_eq!(route_tokens(&policy, &workers, &tokens), 1);
        }
        assert!(policy.placement_index.is_empty());
    }

    #[test]
    fn hash_mode_untokenized_text_stays_min_load() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        for _ in 0..3 {
            workers[0].increment_load();
        }

        let info = SelectWorkerInfo {
            request_text: Some("system prompt plus a user question"),
            ..Default::default()
        };
        for _ in 0..5 {
            assert_eq!(policy.select_worker(&workers, &info), Some(1));
        }
        assert!(policy.placement_index.is_empty());
        assert!(policy.string_trees.is_empty());
    }

    #[test]
    fn hash_mode_ttl_expires_holders_on_read() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let model = normalize_model_key(workers[0].model_id());
        let tokens: Vec<u32> = (0..16).collect();
        let t0 = Instant::now();
        let key = (16usize, hash_token_head(&tokens[..16]));

        policy.record_placement(model, &tokens, &[16], "http://w1:8000", t0);

        let url_to_idx = CacheAwarePolicy::healthy_url_index(&workers, &[0, 1]);
        // Just inside the 180s default TTL: live.
        let live_at = t0 + Duration::from_secs(179);
        assert_eq!(
            policy.live_holder_min_load(&workers, &url_to_idx, model, key, live_at),
            Some(0)
        );
        // Just past it: expired and pruned.
        let expired_at = t0 + Duration::from_secs(181);
        assert_eq!(
            policy.live_holder_min_load(&workers, &url_to_idx, model, key, expired_at),
            None
        );
        assert!(policy
            .placement_index
            .get(model)
            .unwrap()
            .get(&key)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn hash_mode_expired_holder_falls_back_to_min_load() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            cache_ttl_secs: 1,
            ..hash_config(&[16])
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);

        let tokens: Vec<u32> = (0..16).collect();
        let first = route_tokens(&policy, &workers, &tokens);
        assert_eq!(route_tokens(&policy, &workers, &tokens), first);

        // Let the placement lapse, then load the former holder: a live
        // placement would still win, an expired one must load-balance away.
        std::thread::sleep(Duration::from_millis(1300));
        for _ in 0..3 {
            workers[first].increment_load();
        }
        assert_ne!(route_tokens(&policy, &workers, &tokens), first);
    }

    #[test]
    fn hash_mode_holder_cap_evicts_stalest() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let tokens: Vec<u32> = (0..16).collect();
        let t0 = Instant::now();
        let key = (16usize, hash_token_head(&tokens[..16]));

        for (i, url) in [
            "http://w1:8000",
            "http://w2:8000",
            "http://w3:8000",
            "http://w4:8000",
        ]
        .iter()
        .enumerate()
        {
            policy.record_placement("m", &tokens, &[16], url, t0 + Duration::from_secs(i as u64));
        }

        let placements = policy.placement_index.get("m").unwrap();
        let holders = placements.get(&key).unwrap();
        assert_eq!(holders.len(), PLACEMENT_HOLDER_CAP);
        assert!(!holders.iter().any(|h| h.worker_url == "http://w1:8000"));
    }

    #[test]
    fn hash_mode_gate_spills_and_replicates_placement() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let model = normalize_model_key(workers[0].model_id());
        let tokens: Vec<u32> = (0..16).collect();
        let key = (16usize, hash_token_head(&tokens[..16]));

        policy.record_placement(model, &tokens, &[16], "http://w1:8000", Instant::now());
        // Past both gate margins (rel 1.1 of avg 50, abs avg+32): spill.
        for _ in 0..100 {
            workers[0].increment_load();
        }

        assert_eq!(route_tokens(&policy, &workers, &tokens), 1);
        // The spill target becomes an additional holder of this head.
        let placements = policy.placement_index.get(model).unwrap();
        let holders = placements.get(&key).unwrap();
        assert!(holders.iter().any(|h| h.worker_url == "http://w2:8000"));
    }

    #[test]
    fn hash_mode_kv_pressure_abandons_affinity() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            balance_token_usage_threshold: 0.3,
            overload_token_usage_threshold: 0.95,
            ..hash_config(&[16])
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let model = normalize_model_key(workers[0].model_id());
        let tokens: Vec<u32> = (0..16).collect();

        policy.record_placement(model, &tokens, &[16], "http://w1:8000", Instant::now());
        workers[0].increment_load();
        let _tx = inject_kv(&policy, &workers, &[0.9, 0.1]);

        // KV spread 0.8 > 0.3: shortest queue wins over the placement.
        assert_eq!(route_tokens(&policy, &workers, &tokens), 1);
    }

    #[test]
    fn hash_mode_co_hashes_short_and_long_requests() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[2048]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);

        // 3k- and 17k-token requests sharing the 2048-token head must land
        // on the same worker via the shared boundary key.
        let short: Vec<u32> = (0..3000).collect();
        let mut long: Vec<u32> = (0..2048).collect();
        long.extend(9_000_000..9_014_952);
        assert_eq!(long.len(), 17_000);

        let first = route_tokens(&policy, &workers, &short);
        assert_eq!(route_tokens(&policy, &workers, &long), first);
    }

    #[test]
    fn hash_mode_removed_worker_is_purged_from_placements() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let tokens: Vec<u32> = (0..16).collect();
        let now = Instant::now();
        policy.record_placement("m", &tokens, &[16], "http://w1:8000", now);
        policy.record_placement("m", &tokens, &[16], "http://w2:8000", now);

        policy.remove_worker_by_url("http://w1:8000");

        let placements = policy.placement_index.get("m").unwrap();
        let holders = placements
            .get(&(16usize, hash_token_head(&tokens[..16])))
            .unwrap();
        assert_eq!(holders.len(), 1);
        assert_eq!(holders[0].worker_url, "http://w2:8000");
    }

    #[test]
    fn sweep_placement_index_drops_expired_and_empty_keys() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let fresh: Vec<u32> = (0..16).collect();
        let stale: Vec<u32> = (500..516).collect();
        let t0 = Instant::now();
        policy.record_placement("m", &stale, &[16], "http://w1:8000", t0);
        policy.record_placement(
            "m",
            &fresh,
            &[16],
            "http://w1:8000",
            t0 + Duration::from_secs(120),
        );

        let live = CacheAwarePolicy::sweep_placement_index(
            &policy.placement_index,
            Duration::from_secs(180),
            t0 + Duration::from_secs(200),
        );
        assert_eq!(live, 1);
        let placements = policy.placement_index.get("m").unwrap();
        assert_eq!(placements.len(), 1);
        assert!(placements
            .get(&(16usize, hash_token_head(&fresh[..16])))
            .is_some());
    }

    #[test]
    fn live_holder_resolution_skips_dead_holders_and_unknown_keys() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let tokens: Vec<u32> = (0..16).collect();
        let now = Instant::now();
        let key = (16usize, hash_token_head(&tokens[..16]));
        policy.record_placement("m", &tokens, &[16], "http://w1:8000", now);
        policy.record_placement("m", &tokens, &[16], "http://w2:8000", now);

        // w1 unhealthy: its holder entry must not resolve.
        let url_to_idx = CacheAwarePolicy::healthy_url_index(&workers, &[1]);
        assert_eq!(
            policy.live_holder_min_load(&workers, &url_to_idx, "m", key, now),
            Some(1)
        );
        // No healthy workers: no candidate.
        let empty = CacheAwarePolicy::healthy_url_index(&workers, &[]);
        assert_eq!(
            policy.live_holder_min_load(&workers, &empty, "m", key, now),
            None
        );
        // Unknown key: no candidate.
        assert_eq!(
            policy.live_holder_min_load(&workers, &url_to_idx, "m", (32, 7), now),
            None
        );
    }

    #[test]
    fn healthy_url_index_duplicate_urls_resolve_to_least_loaded() {
        let workers = make_workers(&["http://w1:8000", "http://w1:8000", "http://w2:8000"]);
        workers[0].increment_load();

        let url_to_idx = CacheAwarePolicy::healthy_url_index(&workers, &[0, 1, 2]);
        assert_eq!(url_to_idx.len(), 2);
        assert_eq!(url_to_idx["http://w1:8000"], 1);
        assert_eq!(url_to_idx["http://w2:8000"], 2);

        // Equal loads: the lower index wins, matching min-load tie-break.
        let tied = make_workers(&["http://w1:8000", "http://w1:8000"]);
        assert_eq!(
            CacheAwarePolicy::healthy_url_index(&tied, &[0, 1])["http://w1:8000"],
            0
        );
    }

    #[test]
    fn concurrent_record_and_sweep_keeps_live_placements() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let now = Instant::now();
        let ttl = Duration::from_secs(180);

        std::thread::scope(|s| {
            for t in 0..8u32 {
                let policy = &policy;
                s.spawn(move || {
                    for i in 0..50u32 {
                        let base = t * 1000 + i * 16;
                        let tokens: Vec<u32> = (base..base + 16).collect();
                        policy.record_placement("m", &tokens, &[16], "http://w1:8000", now);
                        CacheAwarePolicy::sweep_placement_index(&policy.placement_index, ttl, now);
                    }
                });
            }
        });

        let placements = policy.placement_index.get("m").unwrap();
        assert_eq!(placements.len(), 400);
        for entry in placements.iter() {
            assert_eq!(entry.value().len(), 1);
            assert_eq!(entry.value()[0].worker_url, "http://w1:8000");
        }
    }

    #[test]
    fn with_config_normalizes_boundaries() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[64, 16, 16, 0]));
        assert_eq!(policy.config.cache_boundaries, vec![16, 64]);
    }
}
