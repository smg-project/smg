/*
    Cache-Aware Load Balancing Router

    When load is balanced, uses cache-aware routing. When imbalanced, uses
    LeastLoad expected-wait routing. A system is imbalanced when both:
        (max - min) > abs_threshold  AND  max > rel_threshold * min

    Three types of cache-aware routing (mutually exclusive, selected by
    worker connection mode and KV event availability):

    1. Event-Driven (gRPC + KV events)
    -------------------------------------------
    Uses the KV index's overlap scoring from KvEventMonitor. Routes based
    on actual backend KV cache state. Selects the worker with the highest
    overlap count; LeastLoad breaks equal-affinity ties atomically.
    Falls back to LeastLoad when no cache overlap exists. Over a pool larger
    than 32 workers the decision reads the holders of the request's blocks
    and a sample of 16 others, not the pool (see `FLEET_SAMPLE`).

    2. Approximate Token Tree (gRPC, no KV events)
    -------------------------------------------
    Maintains a TokenTree per model tracking which token prefixes were routed
    where. If match_rate > cache_threshold, routes to the best-matching worker.
    Otherwise routes with LeastLoad's expected-wait algorithm.

    3. Approximate String Tree (HTTP)
    -------------------------------------------
    Same algorithm as (2) but operates on raw text characters instead of
    token IDs, avoiding tokenization overhead.

    Cache Namespaces (cache_salt / extra_key / LoRA)
    -------------------------------------------
    A request that carries cache-partition fields keys the approximate trees
    and the hash index under a namespace marker (see cache_namespace.rs), so
    requests the engine cannot serve from one cache never match each other.
    Every live namespace holds its own copy of a shared prefix: max_tree_size
    bounds the SUM across namespaces, so size it for tenants x working set,
    and a per-request salt makes every request a unique path. The
    event-driven mode is unpartitioned (both sides re-hash token ids), and
    the typed gRPC/ZMQ wires stay unpartitioned until they forward the fields
    to the engine.

    Load Balancing (Expected Wait)
    -------------------------------------------
    When the system is imbalanced, routes to LeastLoad's atomic expected-wait
    winner regardless of cache affinity.

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
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use dashmap::DashMap;
use kv_index::{
    compute_request_content_hashes, request_prefix_hashes, salt::request_content_hashes_with_seed,
    ContentHash, OverlapScores, TenantId, TokenTree, Tree,
};
use openai_protocol::worker::WorkerLoadResponse;
use parking_lot::RwLock;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

use super::{
    cost::{
        self, CandidateInputs, OptimisticAccounting, Pick, RequestInputs, WorkerSelectionPolicy,
    },
    normalize_model_key,
    utils::PeriodicTask,
    CacheAwareConfig, CacheNamespace, LeastLoadPolicy, LoadBalancingPolicy, SelectWorkerInfo,
    TEXT_MARKER_LEN,
};
/// Latest per-worker backend load snapshot stream, keyed by worker URL.
pub(crate) use crate::worker::load_state::{LoadReceiver, LoadSnapshot};
use crate::{
    config::CacheIndexKind,
    mesh::adapters::tree_sync::{RepairEntry, TreeDelta, TreeRepairPage, TreeSyncAdapter},
    observability::{cache_trace, metrics::Metrics},
    worker::{liveness, KvEventMonitor, KvIndex, Worker},
};

/// An overlap of at most this many blocks counts as a miss for the warm-up
/// slice: a chat template's head is this long, shared by every holder, and
/// recomputing it costs less than keeping a returned worker idle.
const WARMUP_MISS_BLOCKS: f64 = 4.0;

/// A hit whose best overlap is at most this share of the request is a
/// shallow one: diverting it to a thin worker recomputes little, so these go
/// first (see [`CacheAwarePolicy::warmup_divert`]).
const DIVERT_SHALLOW_SHARE: f64 = 0.5;

/// A thin worker with requests in flight gets no second diversion sooner than
/// this after its last one.
const DIVERT_WINDOW_MS: u64 = 2_000;

/// How many eligible workers an event-driven decision reads beside the
/// holders of the request's blocks once the pool is larger than twice this:
/// they are the miss path's choices, the spill targets, the rows an
/// all-workers selection policy ranks, and the sample the count-pressure
/// gate's fleet mean is taken from. Sixteen choices balance as a full scan
/// does (two already do, the power of d choices), and cost the same at 128
/// workers as at 10,000. A pool of at most twice this many workers is read
/// whole, so nothing changes for a small fleet.
const FLEET_SAMPLE: usize = 16;

/// How long a pool table serves before a decision rebuilds it: a worker the
/// index interned after the build gains affinity within this, and a worker
/// that entered the warm-up window is sliced to within this.
const POOL_TABLE_REFRESH_MS: u64 = 1_000;

/// Pool tables kept at once: one per routing pool the policy sees (a model's
/// regular pool, the PD legs' pools); past this the oldest goes.
const POOL_TABLES_KEPT: usize = 8;

/// Cache-aware routing policy
///
/// Routes requests based on cache affinity when load is balanced,
/// switches to LeastLoad expected-wait routing when load is imbalanced.
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
    /// Expected-wait selector used after cache affinity has narrowed the
    /// candidate set. It owns backend snapshots and since-poll dispatch credit,
    /// keeping selection and credit atomic under concurrent arrivals.
    load_scorer: LeastLoadPolicy,
    /// Selection policy run over the per-worker inputs gathered for each
    /// request (`cost` module); the default reproduces the affinity-group
    /// decision exactly and the ports of published cost functions replace it.
    selection: WorkerSelectionPolicy,
    /// Optimistic self-accounting of dispatches the engines have not yet
    /// reported; `None` unless `selection_accounting_ttl_ms > 0`.
    accounting: Option<OptimisticAccounting>,
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
    /// Misses seen by the warm-up slice; every `period`th goes to a warming
    /// worker (see [`liveness::Warmup`]).
    warmup_misses: AtomicU64,
    /// Hits seen since the last diversion to a thin worker, and whether a
    /// shallow one was among them (see [`Self::warmup_divert`]).
    divert_hits: AtomicU64,
    divert_shallow_seen: AtomicBool,
    /// Each routing pool's workers by KV index id and position, built once
    /// per pool snapshot (see [`PoolTable`]).
    pool_tables: ArcSwap<Vec<Arc<PoolTable>>>,
    /// Set while one decision rebuilds a pool table so the others keep
    /// serving the one they have.
    pool_table_building: AtomicBool,
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

fn hash_token_head(namespace: Option<CacheNamespace>, head: &[u32]) -> u64 {
    match namespace {
        // An unpartitioned head hashes exactly as before.
        None => xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(head)),
        // A partitioned head hashes as marker ‖ head (the placement index has
        // no pages, so the bare marker suffices): a differing namespace is a
        // different key, and a same-namespace head keys as before.
        Some(namespace) => {
            let mut hasher = xxhash_rust::xxh3::Xxh3::new();
            hasher.update(bytemuck::cast_slice(&namespace.token_marker()));
            hasher.update(bytemuck::cast_slice(head));
            hasher.digest()
        }
    }
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

/// The per-model tree, created on first sight.
///
/// Reads first: `DashMap::entry` takes the shard's *exclusive* lock (and a
/// `String` key allocation) on every call, and every request for a model lands
/// on the same shard, so the selection hot path would serialize on it. A shared
/// read plus an `Arc` clone is uncontended; only a model's first request pays
/// for the insert.
fn tree_for_model<V>(
    trees: &DashMap<String, Arc<V>>,
    model_id: &str,
    new_tree: impl FnOnce() -> V,
) -> Arc<V> {
    if let Some(tree) = trees.get(model_id) {
        return Arc::clone(tree.value());
    }
    trees
        .entry(model_id.to_string())
        .or_insert_with(|| Arc::new(new_tree()))
        .value()
        .clone()
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

        let selection_policy_name = config
            .selection_policy
            .as_deref()
            .unwrap_or(cost::DEFAULT_POLICY);
        let selection = cost::build(selection_policy_name, config.selection_temperature)
            .unwrap_or_else(|err| {
                // Configuration validation rejects this before a policy is built;
                // a policy constructed outside that path still routes, with the
                // default decision, rather than failing every request.
                error!(%err, "Invalid selection policy; using {}", cost::DEFAULT_POLICY);
                cost::default_policy(config.selection_temperature)
            });
        let accounting = (config.selection_accounting_ttl_ms > 0).then(|| {
            OptimisticAccounting::new(Duration::from_millis(config.selection_accounting_ttl_ms))
        });

        Self {
            config,
            load_scorer: LeastLoadPolicy::new(),
            selection,
            accounting,
            string_trees,
            token_trees,
            _eviction_task: eviction_task,
            kv_monitor: RwLock::new(None),
            load_rx: RwLock::new(None),
            warmup_misses: AtomicU64::new(0),
            divert_hits: AtomicU64::new(0),
            divert_shallow_seen: AtomicBool::new(false),
            pool_tables: ArcSwap::from_pointee(Vec::new()),
            pool_table_building: AtomicBool::new(false),
            hash_index,
            populate_hash_index: AtomicBool::new(false),
            mesh_tree_sync: RwLock::new(None),
            placement_index,
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
    pub(crate) fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
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
    /// in [`Self::candidate_requires_spill`].
    fn is_kv_imbalanced(&self, workers: &[Arc<dyn Worker>], healthy_indices: &[usize]) -> bool {
        // Both defaults are 1.0, above the maximum clamped utilization and
        // spread. CacheAware now polls loads for LeastLoad even at defaults,
        // so return before touching the snapshot or scanning the fleet.
        if !self.kv_pressure_gate_configured() {
            return false;
        }

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
        // One Arc clone of the immutable snapshot; the receiver guard and the
        // watch borrow are both released before the scan so publishers are
        // never blocked by it.
        let loads = {
            let guard = self.load_rx.read();
            let rx = guard.as_ref()?;
            let snapshot = rx.borrow().clone();
            snapshot
        };
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
        self.load_scorer.remove_worker(url);
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

    /// Remove several workers from one model. Tree cleanup remains precise per
    /// tenant, while hash-placement state is scanned only once for the batch.
    pub(crate) fn remove_workers_from_model(&self, model_id: &str, worker_urls: &HashSet<String>) {
        let model_id = normalize_model_key(model_id);
        let tenants: Vec<TenantId> = worker_urls
            .iter()
            .map(|worker_url| TenantId::from(worker_url.as_str()))
            .collect();
        let string_tree = self
            .string_trees
            .get(model_id)
            .map(|entry| Arc::clone(entry.value()));
        if let Some(tree) = string_tree {
            for tenant in &tenants {
                tree.remove_tenant_all(tenant);
            }
        }
        let token_tree = self
            .token_trees
            .get(model_id)
            .map(|entry| Arc::clone(entry.value()));
        if let Some(tree) = token_tree {
            for tenant in &tenants {
                tree.remove_tenant_all(tenant);
            }
        }
        let placements = self
            .placement_index
            .get(model_id)
            .map(|entry| Arc::clone(entry.value()));
        if let Some(placements) = placements {
            placements.retain(|_, holders| {
                holders.retain(|holder| !worker_urls.contains(&holder.worker_url));
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

    /// Select the expected-wait worker (used when KV pressure abandons
    /// affinity), then record only that final dispatch in the local tree.
    /// Handles both HTTP (text-based) and gRPC (token-based) requests.
    fn select_worker_fallback(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        healthy_indices: &[usize],
        model_id: &str,
    ) -> Option<usize> {
        let selected = self.select_expected_wait(workers, healthy_indices, info)?;
        let worker_url = workers[selected].url();

        // Even in imbalanced mode, update the appropriate tree to maintain cache
        // state, under the request's cache namespace so a partitioned request
        // never seeds affinity for other partitions. Prefer the token tree for
        // gRPC requests, fall back to the string tree for HTTP.
        if let Some(tokens) = info.tokens {
            let prefixed;
            let tokens: &[u32] = match info.cache_namespace {
                Some(namespace) => {
                    prefixed = namespace.prefixed_tokens(tokens, self.config.block_size.max(1));
                    &prefixed
                }
                None => CacheNamespace::unpartitioned_tokens(tokens),
            };
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
                    let result = tree.match_and_insert(tokens, worker_url);
                    let matched_prefix: Vec<u32> = tokens[..result.matched_token_count].to_vec();
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .token_tree
                        .insert(kv_index::hash_token_path(tokens), matched_prefix);
                } else {
                    tree.insert_tokens(tokens, worker_url);
                }
            }
        } else if let Some(text) = info.request_text {
            let prefixed;
            let text: &str = match info.cache_namespace {
                Some(namespace) => {
                    prefixed = namespace.prefixed_text(text);
                    &prefixed
                }
                None => CacheNamespace::unpartitioned_text(text),
            };
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
                    let result = tree.match_and_insert(text, worker_url);
                    let matched_prefix: String =
                        text.chars().take(result.matched_char_count).collect();
                    let path_hash = kv_index::hash_node_path(text);
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .string_tree
                        .insert(path_hash, matched_prefix);
                } else {
                    tree.insert_text(text, worker_url);
                }
            } else {
                debug!(
                    "Warning: No string tree found for model '{}', skipping cache update",
                    model_id
                );
            }
        }

        debug!(
            branch = "kv_pressure_expected_wait",
            worker = worker_url,
            model_id,
            "Cache-aware selection"
        );
        Some(selected)
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

/// One positive-overlap candidate: slice index, undecayed overlap in blocks
/// (the tree paths report matched units over the block size) and the possibly
/// decayed score the affinity-group decision ranks on.
struct OverlapCandidate {
    idx: usize,
    raw_score: f64,
    effective_score: f64,
}

/// One routing pool's workers by their KV index ids, built once per pool
/// snapshot and refreshed every [`POOL_TABLE_REFRESH_MS`]. The registry hands
/// the policy the same slice until membership changes, so an overlap lookup's
/// `index id -> blocks` result maps to slice positions through this table
/// instead of a string lookup per healthy worker per request, and the warm-up
/// slice reads the workers inside its window from it instead of scanning the
/// pool on every thin miss.
#[derive(Debug)]
struct PoolTable {
    /// Address and length of the slice the table was built from.
    slice: (usize, usize),
    /// Gateway clock at the build, for the refresh.
    built_ms: u64,
    /// Address of the worker at each position at the build: a pool rebuilt
    /// at the same address with other workers fails this check per holder.
    worker_at: Vec<usize>,
    /// The index id of the worker at each position; `None` while it is not
    /// interned.
    id_at: Vec<Option<u32>>,
    /// Index id -> position.
    position_of: HashMap<u32, u32>,
    /// The fleet's index level at the build: the upper median of the
    /// workers' index sizes, what a thin worker is thin against.
    fleet_level: usize,
    /// Positions of the warm-up slice's candidates at the build: the workers
    /// thinner than the fleet (an index emptied by a resync, or never fed)
    /// whatever their age, else the workers inside the warm-up window
    /// (admitted within it, index still growing) unless every worker was (a
    /// young fleet slices nothing).
    warming: Vec<u32>,
    /// Positions of the thin workers alone, the targets of the hit diversion
    /// (see [`CacheAwarePolicy::warmup_divert`]).
    thin: Vec<u32>,
}

impl PoolTable {
    fn slice_key(workers: &[Arc<dyn Worker>]) -> (usize, usize) {
        (workers.as_ptr() as usize, workers.len())
    }

    fn worker_address(worker: &Arc<dyn Worker>) -> usize {
        Arc::as_ptr(worker).cast::<()>() as usize
    }

    /// One pass over the pool: the string lookup per worker happens here,
    /// once per snapshot and refresh, not per request.
    fn build(workers: &[Arc<dyn Worker>], indexer: &KvIndex, now_ms: u64) -> Self {
        let warmup = liveness::warmup();
        let mut worker_at = Vec::with_capacity(workers.len());
        let mut id_at = Vec::with_capacity(workers.len());
        let mut position_of = HashMap::with_capacity(workers.len());
        let mut sized: Vec<Option<(usize, usize)>> = Vec::with_capacity(workers.len());
        for (position, worker) in workers.iter().enumerate() {
            worker_at.push(Self::worker_address(worker));
            let id = indexer.worker_id(worker.url());
            id_at.push(id);
            if let Some(id) = id {
                position_of.insert(id, position as u32);
            }
            // Index size, and its growth since the admission for the age rule
            // (the baseline restarts when the count drops); thinness is read
            // against the fleet's level below, whatever the growth.
            sized.push(id.map(|id| {
                let indexed = indexer.worker_block_count(id);
                (indexed, worker.warmup_growth(indexed))
            }));
        }
        // The fleet's level: the upper median of the sizes, so one emptied
        // worker does not drag it down.
        let mut sizes: Vec<usize> = sized
            .iter()
            .map(|sized| sized.map_or(0, |(indexed, _)| indexed))
            .collect();
        sizes.sort_unstable();
        let fleet_level = sizes.get(sizes.len() / 2).copied().unwrap_or(0);
        let mut thin = Vec::new();
        let mut young = Vec::new();
        if warmup.share > 0.0 {
            for (position, (worker, sized)) in workers.iter().zip(&sized).enumerate() {
                let (indexed, grown) = match sized {
                    Some((indexed, grown)) => (*indexed, Some(*grown)),
                    None => (0, None),
                };
                if !warmup.applies(worker.admitted_age(), grown, indexed, fleet_level) {
                    continue;
                }
                if warmup.is_thin(indexed, fleet_level) {
                    thin.push(position as u32);
                } else {
                    young.push(position as u32);
                }
            }
        }
        // A thin worker is a candidate whatever the fleet's age; workers
        // warming only because they are young are candidates only when the
        // fleet is not all young.
        let warming = if !thin.is_empty() {
            thin.clone()
        } else if young.len() < workers.len() {
            young
        } else {
            Vec::new()
        };
        Self {
            slice: Self::slice_key(workers),
            built_ms: now_ms,
            worker_at,
            id_at,
            position_of,
            fleet_level,
            warming,
            thin,
        }
    }

    /// Whether the table was built from `workers` (the same slice).
    fn describes(&self, workers: &[Arc<dyn Worker>]) -> bool {
        self.slice == Self::slice_key(workers)
    }

    /// Whether the table is within its refresh period.
    fn fresh(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.built_ms) < POOL_TABLE_REFRESH_MS
    }

    /// The position holding index worker `id`, when the worker there is
    /// still the one the table was built from.
    fn position(&self, id: u32, workers: &[Arc<dyn Worker>]) -> Option<usize> {
        let position = *self.position_of.get(&id)? as usize;
        let same_worker = workers
            .get(position)
            .is_some_and(|worker| Self::worker_address(worker) == self.worker_at[position]);
        same_worker.then_some(position)
    }
}

/// What the sampled decision made of a request.
enum Sampled {
    /// The event-driven decision over the holders and a sample of the pool:
    /// the selection, or `None` when it declined every candidate.
    Decided(Option<usize>),
    /// Not a request for the sampled decision: the pool is small, the request
    /// carries no tokens or has no event index, the configuration wants the
    /// KV-pressure gate (a reading of the whole fleet), or the sample held no
    /// eligible worker. The scan decides.
    Scan,
}

/// Which holders a decision gathers from the overlap scores. A shared chat
/// template puts a shallow overlap on most of a fleet, so the scores name
/// nearly every worker; the decision must not pay per named worker.
#[derive(Clone, Copy)]
enum Gather {
    /// The eligible holders of the deepest overlap only: what the exact
    /// maximum-group decision reads (the default policy at temperature zero,
    /// no decay in effect, no accounting). A tie wider than `cap` (a fleet
    /// sharing a chat template's head, a prompt reaching no further) is a
    /// uniform draw of `cap` of them: the expected-wait selector breaks the
    /// tie among those, as a miss is placed among a sample of the pool.
    TopGroup { cap: Option<usize> },
    /// Every eligible holder in slice order, the `cap` deepest when set.
    All { cap: Option<usize> },
}

/// Pressure-tuning inputs for [`CacheAwarePolicy::overlap_candidates`]: the two
/// config knobs plus the immutable load snapshot captured from the load
/// receiver at selection time, from which each worker's waiting-prefill
/// backlog (queued uncached tokens, clamped non-negative) is derived at
/// lookup. `waiting_prefill_tokens` is `None` when decay is off or no load
/// receiver is wired; workers absent from the snapshot are never decayed.
struct OverlapTuning<'a> {
    overlap_decay: f32,
    /// Zero means the exact maximum-overlap group; the policy carries the
    /// temperature for its draw, the host reads it to know which holders to
    /// gather (see [`CacheAwarePolicy::exact_maximum_decision`]).
    selection_temperature: f32,
    waiting_prefill_tokens: Option<&'a LoadSnapshot>,
}

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        // The event-driven decision over a large pool reads the holders of
        // the request's blocks and a bounded sample of the rest; every other
        // path, and a pool the sample would cover anyway, reads the pool.
        if let Sampled::Decided(selected) = self.select_worker_sampled(workers, info) {
            return selected;
        }
        self.select_worker_scanned(workers, info)
    }

    fn on_request_complete(&self, worker_url: &str, success: bool) {
        if let Some(accounting) = &self.accounting {
            accounting.release(worker_url);
        }
        self.selection.on_request_complete(worker_url);
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

    fn update_loads(&self, loads: &HashMap<String, WorkerLoadResponse>) {
        // WorkerMonitor invokes this immediately before publishing its complete
        // immutable snapshot (with no await between the two operations). Advance
        // ExpectedWait here so each successful worker's new load and credit
        // reset stay atomic. KV-pressure and overlap decay intentionally keep
        // using the last fully published snapshot during that handoff; scoring
        // ExpectedWait from its old value would instead pair an
        // old queue with a new reset and reopen the incast race.
        self.load_scorer.update_loads(loads);
    }

    /// Expected-wait selection needs backend snapshots for every CacheAware
    /// configuration; KV pressure and overlap decay consume them as well.
    fn needs_backend_loads(&self) -> bool {
        true
    }

    fn remove_worker(&self, url: &str) {
        // The CacheAware-specific removal path already prunes trees and hash
        // placements before the registry invokes this generic load-aware
        // hook. Keep this hook scoped to LeastLoad state so worker churn does
        // not repeat the full all-model cache scan.
        self.load_scorer.remove_worker(url);
        if let Some(accounting) = &self.accounting {
            accounting.forget_worker(url);
        }
        self.selection.on_worker_removed(url);
    }

    fn reconcile_in_flight(&self, worker_url: &str, in_flight: usize) {
        let released = self
            .accounting
            .as_ref()
            .map_or(0, |accounting| accounting.reconcile(worker_url, in_flight));
        self.selection.reconcile_in_flight(worker_url, in_flight);
        if released > 0 {
            Metrics::record_policy_inflight_reconciled(self.name(), released);
            debug!(
                worker = worker_url,
                in_flight, released, "Released bookings whose completion never arrived"
            );
        }
    }

    fn reset(&self) {
        self.load_scorer.reset();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The event-driven index resolved once per request for the selection.
struct EventIndex<'a> {
    indexer: Arc<KvIndex>,
    block_size: usize,
    model_id: &'a str,
}

// Private helper methods for select_worker
impl CacheAwarePolicy {
    /// The decision over the whole pool: one eligibility read per worker,
    /// then the hash, tree or event-driven path. The sampled decision covers
    /// the event-driven path over a large pool; this is everything else.
    fn select_worker_scanned(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let request_text = info.request_text;
        let request_tokens = info.tokens;

        // Single O(workers) gather: read each worker once via routing_state()
        // (status + load + processed + overload veto under one ArcSwap guard),
        // replacing the former separate passes whose per-worker guard traffic
        // dominated routing CPU at scale. Collects eligible indices, the load
        // sum (for the per-request pressure gate). Final selection and credit
        // are delegated atomically to LeastLoad after affinity is resolved.
        let mut healthy_indices: Vec<usize> = Vec::with_capacity(workers.len());
        let mut load_sum = 0usize;
        for (idx, worker) in workers.iter().enumerate() {
            let state = worker.routing_state();
            if cache_trace::enabled() && !state.eligible() {
                cache_trace::gate(serde_json::json!({
                    "phase": "eligibility", "worker": worker.url(),
                    "load": state.load, "overloaded": state.overloaded,
                    "eligible": false,
                }));
            }
            // The overload veto costs nothing here: `state` is the word this
            // pass already loaded for health, circuit breaker and load.
            if state.eligible() {
                healthy_indices.push(idx);
                load_sum += state.load;
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
            return self.select_worker_hash(workers, info, &healthy_indices, avg_load, model_id);
        }

        // Abandon cache affinity fleet-wide only under backend KV pressure;
        // request-count pressure is applied per request to the selected
        // candidate inside each affinity path.
        if self.is_kv_imbalanced(workers, &healthy_indices) {
            return self.select_worker_fallback(workers, info, &healthy_indices, model_id);
        }

        // Cache-aware routing when balanced — three types (mutually exclusive):
        //   1. Event-driven: KV index overlap scoring (gRPC + KV events)
        //   2. Approximate token tree: TokenTree prefix matching (gRPC, no events)
        //   3. Approximate string tree: Tree prefix matching (HTTP)
        if let Some(tokens) = request_tokens {
            // Event-driven mode re-hashes engine-reported blocks from their
            // token ids on both sides, so it keys a namespace through the
            // hash seed rather than a marker (see cache_namespace.rs); the
            // approximate and hash modes below key under the marker.
            if let Some(index) = self.event_index_for(model_id) {
                self.select_worker_event_driven(
                    workers,
                    tokens,
                    &healthy_indices,
                    avg_load,
                    &index,
                    info,
                )
            } else {
                self.select_worker_with_tokens(
                    workers,
                    tokens,
                    &healthy_indices,
                    avg_load,
                    model_id,
                    info,
                )
            }
        } else {
            let text = request_text.unwrap_or("");
            self.select_worker_with_text(workers, text, &healthy_indices, avg_load, model_id, info)
        }
    }

    /// The event-driven decision for a pool larger than twice
    /// [`FLEET_SAMPLE`]: eligibility is read for a uniform sample of the pool
    /// (the miss path's choices, the spill targets, the count-pressure gate's
    /// fleet mean) and for the holders the index names, never for the whole
    /// pool. A sample without one eligible worker defers to the scan, which
    /// finds any that remain.
    fn select_worker_sampled(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
    ) -> Sampled {
        let Some(tokens) = info.tokens else {
            return Sampled::Scan;
        };
        if workers.len() <= 2 * FLEET_SAMPLE
            || self.config.cache_index == CacheIndexKind::Hash
            || self.kv_pressure_gate_configured()
        {
            return Sampled::Scan;
        }
        let mut sample = [0usize; FLEET_SAMPLE];
        let mut sampled = 0usize;
        let mut load_sum = 0usize;
        let mut rng = rand::rng();
        // Draws with replacement, repeats dropped: above twice the sample
        // size a repeat costs a slot and nothing else.
        for _ in 0..2 * FLEET_SAMPLE {
            if sampled == FLEET_SAMPLE {
                break;
            }
            let idx = rng.random_range(0..workers.len());
            if sample[..sampled].contains(&idx) {
                continue;
            }
            let state = workers[idx].routing_state();
            if state.eligible() {
                sample[sampled] = idx;
                sampled += 1;
                load_sum += state.load;
            }
        }
        if sampled == 0 {
            return Sampled::Scan;
        }
        let sample = &mut sample[..sampled];
        sample.sort_unstable();
        let avg_load = load_sum as f64 / sampled as f64;
        let model_id = normalize_model_key(workers[sample[0]].model_id());
        let Some(index) = self.event_index_for(model_id) else {
            return Sampled::Scan;
        };
        Sampled::Decided(
            self.select_worker_event_driven(workers, tokens, sample, avg_load, &index, info),
        )
    }

    /// Whether the decision is the exact maximum-overlap group: the default
    /// selection policy at temperature zero, no decay in effect and no
    /// accounting. Then only the deepest holders are gathered; every other
    /// configuration reads each holder.
    fn exact_maximum_decision(&self, tuning: &OverlapTuning<'_>) -> bool {
        self.selection.name() == cost::DEFAULT_POLICY
            && tuning.selection_temperature <= 0.0
            && self.accounting.is_none()
            && (tuning.overlap_decay <= 0.0 || tuning.waiting_prefill_tokens.is_none())
    }

    /// Whether the KV-pressure gate is configured: it reads the whole
    /// fleet's token usage, so the decision reads the pool.
    fn kv_pressure_gate_configured(&self) -> bool {
        self.config.balance_token_usage_threshold < 1.0
            || self.config.overload_token_usage_threshold < 1.0
    }

    /// The pool table for `workers`: the one kept when it is fresh, a stale
    /// one while another decision rebuilds, else built here and published.
    fn pool_table(
        &self,
        workers: &[Arc<dyn Worker>],
        indexer: &KvIndex,
        now_ms: u64,
    ) -> Arc<PoolTable> {
        {
            let tables = self.pool_tables.load();
            if let Some(table) = tables.iter().find(|table| table.describes(workers)) {
                if table.fresh(now_ms) || self.pool_table_building.swap(true, Ordering::AcqRel) {
                    return Arc::clone(table);
                }
            }
        }
        let table = Arc::new(PoolTable::build(workers, indexer, now_ms));
        self.pool_tables.rcu(|tables| {
            let mut next: Vec<Arc<PoolTable>> = tables
                .iter()
                .filter(|kept| kept.slice != table.slice)
                .cloned()
                .collect();
            if next.len() >= POOL_TABLES_KEPT {
                if let Some(oldest) = next
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, kept)| kept.built_ms)
                    .map(|(position, _)| position)
                {
                    next.swap_remove(oldest);
                }
            }
            next.push(Arc::clone(&table));
            next
        });
        self.pool_table_building.store(false, Ordering::Release);
        table
    }

    /// The union of two ascending position lists, ascending: the eligible
    /// workers a decision read and the holders, which a sampled decision's
    /// list need not contain.
    fn merge_rows(eligible: &[usize], candidates: &[OverlapCandidate]) -> Vec<usize> {
        let mut rows = Vec::with_capacity(eligible.len() + candidates.len());
        let (mut i, mut j) = (0, 0);
        loop {
            let next = match (eligible.get(i), candidates.get(j)) {
                (Some(&idx), Some(candidate)) if idx < candidate.idx => {
                    i += 1;
                    idx
                }
                (Some(&idx), Some(candidate)) if idx == candidate.idx => {
                    i += 1;
                    j += 1;
                    idx
                }
                (_, Some(candidate)) => {
                    j += 1;
                    candidate.idx
                }
                (Some(&idx), None) => {
                    i += 1;
                    idx
                }
                (None, None) => break,
            };
            rows.push(next);
        }
        rows
    }

    /// The event-driven index for this model: its indexer and block size.
    /// `None` when there is no monitor or indexer, or the indexer is empty
    /// (startup, reconnect), so routing falls through to the approximate
    /// token tree instead of taking the event-driven path with no data and
    /// landing on a fallback. One monitor lock and one indexer lookup serve
    /// both this check and the selection that follows.
    fn event_index_for<'a>(&self, model_id: &'a str) -> Option<EventIndex<'a>> {
        let guard = self.kv_monitor.read();
        let monitor = guard.as_ref()?;
        let indexer = monitor.get_indexer(model_id)?;
        if indexer.is_empty() {
            return None;
        }
        // Per-model block_size: learned from events > config default
        let block_size = monitor
            .block_size(model_id)
            .unwrap_or(self.config.block_size);
        Some(EventIndex {
            indexer,
            block_size,
            model_id,
        })
    }

    /// Check if an event-driven indexer exists with data for this model.
    #[cfg(test)]
    fn has_event_indexer(&self, model_id: &str) -> bool {
        self.event_index_for(model_id).is_some()
    }

    /// The shared load snapshot for waiting-prefill decay, or `None` when
    /// decay is off or no load receiver is wired. One `Arc` clone per
    /// selection — no per-request map is built; backlog values are derived
    /// at lookup from the same frozen snapshot.
    fn waiting_prefill_snapshot(&self) -> Option<Arc<LoadSnapshot>> {
        if self.config.overlap_decay <= 0.0 {
            return None;
        }
        let guard = self.load_rx.read();
        guard.as_ref().map(|rx| rx.borrow().clone())
    }

    /// Request inputs for the tree paths: units are tokens (token tree) or
    /// chars (string tree); the trees carry no prefix hashes.
    fn tree_request(&self, units: usize, avg_load: f64) -> RequestInputs<'static> {
        let block_size = self.config.block_size.max(1);
        RequestInputs {
            prompt_tokens: units,
            block_size,
            request_blocks: (units / block_size).max(1),
            avg_load,
            prefix_hashes: None,
        }
    }

    /// Select and credit one final worker atomically with LeastLoad's
    /// expected-wait algorithm. CacheAware must call this exactly once per
    /// successful dispatch, after cache affinity and spill logic have fixed
    /// the candidate set.
    fn select_expected_wait(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        // Clone the immutable snapshot out of both guards before scoring. The
        // snapshot is only a freshness fence; LeastLoad's poll-fed cache stays
        // the score source so publication cannot split a load update from its
        // matching since-poll credit reset.
        let complete_snapshot = {
            let receiver_guard = self.load_rx.read();
            receiver_guard
                .as_ref()
                .map(|receiver| receiver.borrow().clone())
        };
        self.load_scorer.select_min_expected_wait_with_freshness(
            workers,
            candidates,
            info,
            self.name(),
            complete_snapshot.as_deref(),
        )
    }

    /// Per-request count-pressure predicate: over
    /// `balance_rel_threshold` times the healthy-fleet mean load AND
    /// `balance_abs_threshold` requests above it, the request spills to the
    /// expected-wait worker instead. Affinity paths insert for the spill
    /// target, so a prefix whose home saturates gains an additional tenant —
    /// hot prefixes replicate instead of queueing behind one engine. Both
    /// margins must clear so the gate neither fires on steady-state variance
    /// (relative alone would, at low means) nor stays blind to a deep queue
    /// (absolute alone would, at high means).
    fn candidate_requires_spill(
        &self,
        workers: &[Arc<dyn Worker>],
        selected: usize,
        avg_load: f64,
    ) -> bool {
        let load = workers[selected].load() as f64;
        let spill = load > avg_load * f64::from(self.config.balance_rel_threshold)
            && load > avg_load + self.config.balance_abs_threshold as f64;
        if cache_trace::enabled() {
            cache_trace::gate(serde_json::json!({
                "worker": workers[selected].url(), "load": load, "average_load": avg_load,
                "relative_threshold": self.config.balance_rel_threshold,
                "absolute_threshold": self.config.balance_abs_threshold, "spill": spill,
            }));
        }
        spill
    }

    /// The warm-up slice: one cache miss in `1 / share` goes to the
    /// least-loaded warming worker, so a returned, new or emptied worker
    /// builds a cache instead of idling behind the fleet's affinity (on the
    /// real fleet a restarted worker went a minute without a request; in the
    /// soaks a worker whose index a publisher-restart resync had emptied never
    /// saw a request again, every prompt holding an overlap elsewhere). The
    /// candidates are the pool table's, chosen at its build (refreshed every
    /// second): workers thinner than the fleet (see
    /// [`liveness::Warmup::is_thin`]) whatever the fleet's age, else the
    /// workers inside the warm-up window unless every worker is (a young
    /// fleet slices nothing, a miss getting a load-balanced pick anyway); a
    /// settled fleet pays nothing here. Each candidate is checked live before
    /// the pick, and the pick is credited through the expected-wait selector
    /// like any other.
    fn warmup_slice(
        &self,
        workers: &[Arc<dyn Worker>],
        table: &PoolTable,
        indexer: &KvIndex,
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        if table.warming.is_empty() {
            return None;
        }
        let warmup = liveness::warmup();
        if warmup.share <= 0.0 {
            return None;
        }
        let mut pick: Option<(usize, usize)> = None;
        let mut tied = 0u32;
        let mut rng = rand::rng();
        for &position in &table.warming {
            let idx = position as usize;
            let Some(worker) = workers.get(idx) else {
                continue;
            };
            let state = worker.routing_state();
            if !state.eligible() {
                continue;
            }
            let indexed = table.id_at[idx].map(|id| indexer.worker_block_count(id));
            let grown = indexed.map(|indexed| worker.warmup_growth(indexed));
            if !warmup.applies(
                worker.admitted_age(),
                grown,
                indexed.unwrap_or(0),
                table.fleet_level,
            ) {
                continue;
            }
            match pick {
                Some((_, load)) if state.load > load => {}
                Some((_, load)) if state.load == load => {
                    // Equal loads draw uniformly: an idle warming fleet must
                    // not hand every slice to its lowest position.
                    tied += 1;
                    if rng.random_range(0..=tied) == 0 {
                        pick = Some((idx, state.load));
                    }
                }
                _ => {
                    pick = Some((idx, state.load));
                    tied = 0;
                }
            }
        }
        let (idx, _) = pick?;
        if !self
            .warmup_misses
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(warmup.period())
        {
            return None;
        }
        self.select_expected_wait(workers, &[idx], info)
    }

    /// The thin worker's share of hits. On a replay where every request has
    /// a holder (the Mooncake trace: a system prompt or a chat template in
    /// front of everything) the miss path never runs, so a worker whose
    /// index a resync emptied gets nothing from the slice and idles for good
    /// (the churn runs c1 and c2, the soaks s6 and s7: one worker lost per
    /// publisher restart, for half an hour). One hit in
    /// `Warmup::divert_every` therefore goes to the least-loaded thin worker
    /// although another holds its prefix, the recompute accepted: shallow
    /// overlaps first (at most [`DIVERT_SHALLOW_SHARE`] of the request), any
    /// hit once a window of `divert_every` hits passed without a shallow
    /// one; never while the thin worker has anything in flight unless its
    /// last diversion is older than [`DIVERT_WINDOW_MS`] (the stamp lives on
    /// the worker, so it survives the table's rebuilds), so a hot fleet is
    /// not disturbed; and only until its index crosses the thinness ratio of
    /// the fleet's level. Equal loads draw uniformly like the slice. A fleet
    /// with no thin worker returns before any clock read or counter: this
    /// costs the steady state nothing. Protection and the liveness vetoes
    /// apply through the routing state as everywhere; the pick is credited
    /// through the expected-wait selector like the slice's.
    fn warmup_divert(
        &self,
        workers: &[Arc<dyn Worker>],
        table: &PoolTable,
        indexer: &KvIndex,
        info: &SelectWorkerInfo,
        overlap_share: f64,
    ) -> Option<usize> {
        if table.thin.is_empty() {
            return None;
        }
        let warmup = liveness::warmup();
        if warmup.divert_every == 0 || warmup.share <= 0.0 {
            return None;
        }
        let shallow = overlap_share <= DIVERT_SHALLOW_SHARE;
        if shallow {
            self.divert_shallow_seen.store(true, Ordering::Relaxed);
        }
        let hits = self.divert_hits.fetch_add(1, Ordering::Relaxed) + 1;
        let shallow_seen = self.divert_shallow_seen.load(Ordering::Relaxed);
        let due = (hits >= warmup.divert_every && (shallow || !shallow_seen))
            || hits >= 2 * warmup.divert_every;
        if !due {
            return None;
        }
        let now = liveness::now_ms();
        let mut pick: Option<(usize, usize)> = None;
        let mut tied = 0u32;
        let mut rng = rand::rng();
        for &position in &table.thin {
            let idx = position as usize;
            let Some(worker) = workers.get(idx) else {
                continue;
            };
            let state = worker.routing_state();
            if !state.eligible() {
                continue;
            }
            let indexed = table.id_at[idx].map_or(0, |id| indexer.worker_block_count(id));
            if !warmup.is_thin(indexed, table.fleet_level) {
                continue;
            }
            if state.load > 0 && now < worker.divert_until_ms() {
                continue;
            }
            match pick {
                Some((_, load)) if state.load > load => {}
                Some((_, load)) if state.load == load => {
                    tied += 1;
                    if rng.random_range(0..=tied) == 0 {
                        pick = Some((idx, state.load));
                    }
                }
                _ => {
                    pick = Some((idx, state.load));
                    tied = 0;
                }
            }
        }
        let (idx, _) = pick?;
        self.divert_hits.store(0, Ordering::Relaxed);
        self.divert_shallow_seen.store(false, Ordering::Relaxed);
        workers[idx].note_diverted(now + DIVERT_WINDOW_MS);
        self.select_expected_wait(workers, &[idx], info)
    }

    /// Resolve an affinity score group to one final worker. Safe affinity
    /// candidates retain priority; only when every tied holder trips the
    /// pressure gate do we scan the healthy fleet for non-gated spill targets.
    /// The expected-wait selector both chooses and credits the result under one
    /// lock, so no provisional holder is ever accounted.
    fn select_final_from_affinity(
        &self,
        workers: &[Arc<dyn Worker>],
        affinity_candidates: &[usize],
        healthy_indices: &[usize],
        avg_load: f64,
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let safe_affinity: Vec<usize> = affinity_candidates
            .iter()
            .copied()
            .filter(|&idx| !self.candidate_requires_spill(workers, idx, avg_load))
            .collect();
        if !safe_affinity.is_empty() {
            return self.select_expected_wait(workers, &safe_affinity, info);
        }

        // A miss has no affinity candidates and is not a spill: every
        // eligible worker the decision read participates. A true spill
        // excludes other workers that trip the same gate, preventing a
        // fallback from reselecting a hot holder.
        if affinity_candidates.is_empty() {
            return self.select_expected_wait(workers, healthy_indices, info);
        }
        let spill_candidates: Vec<usize> = healthy_indices
            .iter()
            .copied()
            .filter(|&idx| !self.candidate_requires_spill(workers, idx, avg_load))
            .collect();
        let candidates = if spill_candidates.is_empty() {
            healthy_indices
        } else {
            &spill_candidates
        };
        self.select_expected_wait(workers, candidates, info)
    }

    /// Run the selection policy over this request's inputs and resolve its
    /// pick with the host's gate, expected-wait selector and credit.
    ///
    /// `candidates` are the positive-overlap workers with their decayed
    /// scores. Policies that also want cold workers or backend loads declare
    /// it in their `Needs`; the default policy declares neither, so with no
    /// accounting its hot path gathers exactly what the affinity-group
    /// decision did and reaches the same `select_final_from_affinity` call.
    ///
    /// - `Pick::Group`: the affinity group, resolved as before (pressure
    ///   gate, then expected wait, which credits the result).
    /// - `Pick::None`: the miss path (expected wait over the healthy fleet).
    /// - `Pick::Final`: that worker, credited through the expected-wait
    ///   selector; the policy owns the load trade-off, so the count-pressure
    ///   gate does not apply. If the waiting-queue veto drops it, the fleet
    ///   fallback runs.
    fn resolve_selection(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        candidates: &[OverlapCandidate],
        request: &RequestInputs<'_>,
        avg_load: f64,
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let needs = self.selection.needs();
        let accounting = self.accounting.as_ref();
        let wants_all = needs.all_workers || accounting.is_some();
        if candidates.is_empty() && !wants_all {
            return self.select_final_from_affinity(workers, &[], healthy_indices, avg_load, info);
        }

        // With `all_workers` the rows are the eligible workers the decision
        // read merged with the holders, in slice order (a sampled decision
        // reads eligibility for a bounded sample of the pool beside the
        // holders, so the two lists differ); otherwise the candidates are
        // the rows and no list is built.
        let all_rows: Vec<usize> = if wants_all {
            Self::merge_rows(healthy_indices, candidates)
        } else {
            Vec::new()
        };
        let predicted = match (accounting, request.prefix_hashes) {
            (Some(accounting), Some(hashes)) => accounting.predicted_overlaps(hashes),
            _ => Vec::new(),
        };
        let gather = |idx: usize, raw: f64, effective: f64| {
            let url = workers[idx].url();
            let predicted_blocks = predicted
                .iter()
                .find(|(predicted_url, _)| &**predicted_url == url)
                .map_or(0.0, |(_, blocks)| *blocks);
            // A prediction deeper than the index's view stands in for both
            // scores: the blocks are expected to be resident by the time the
            // request lands, so no decay is applied to them.
            let (device_blocks, effective_score) = if predicted_blocks > raw {
                (predicted_blocks, predicted_blocks)
            } else {
                (raw, effective)
            };
            CandidateInputs {
                idx,
                url,
                device_blocks,
                effective_score,
            }
        };
        let inputs: Vec<CandidateInputs<'_>> = if wants_all {
            // Rows and candidates are both in slice order, so one merge pass
            // pairs them.
            let mut next = candidates.iter().peekable();
            all_rows
                .iter()
                .map(|&idx| {
                    let (raw, effective) = next
                        .next_if(|candidate| candidate.idx == idx)
                        .map_or((0.0, 0.0), |candidate| {
                            (candidate.raw_score, candidate.effective_score)
                        });
                    gather(idx, raw, effective)
                })
                .collect()
        } else {
            candidates
                .iter()
                .map(|candidate| {
                    gather(
                        candidate.idx,
                        candidate.raw_score,
                        candidate.effective_score,
                    )
                })
                .collect()
        };
        if cache_trace::enabled() {
            if inputs.len() > 32 {
                cache_trace::mark_truncated();
            }
            for candidate in inputs.iter().take(32) {
                cache_trace::score(serde_json::json!({
                    "source": "policy_affinity", "worker": candidate.url,
                    "device_blocks": candidate.device_blocks, "effective_score": candidate.effective_score,
                    "request_blocks": request.request_blocks, "block_size": request.block_size,
                    "selection_policy": self.selection.name(),
                }));
            }
        }
        let selected = match self.selection.select(request, &inputs) {
            Pick::None => {
                self.select_final_from_affinity(workers, &[], healthy_indices, avg_load, info)
            }
            Pick::Group(rows) => {
                let group: Vec<usize> = rows.iter().map(|&row| inputs[row].idx).collect();
                self.select_final_from_affinity(workers, &group, healthy_indices, avg_load, info)
            }
            Pick::Final(row) => {
                let idx = inputs[row].idx;
                self.select_expected_wait(workers, &[idx], info)
                    .or_else(|| self.select_expected_wait(workers, healthy_indices, info))
            }
        }?;

        if let Some(dispatched) = inputs.iter().find(|candidate| candidate.idx == selected) {
            self.selection.on_dispatch(request, dispatched);
            if let Some(accounting) = accounting {
                let uncached = dispatched.uncached_prompt_tokens(request) as u64;
                accounting.record_dispatch(
                    dispatched.url,
                    uncached,
                    request.prefix_hashes.unwrap_or(&[]),
                );
            }
        }
        Some(selected)
    }

    /// Pick an effective-affinity score group while preserving temperature.
    /// At temperature zero this is the exact maximum. With temperature, the
    /// existing softmax samples one worker; expanding that draw back to every
    /// equal-score worker preserves each score group's aggregate probability,
    /// then LeastLoad breaks the tie inside the sampled group.
    ///
    /// Kept as the reference for the default selection policy, which must
    /// reproduce it (see the pin tests); production goes through
    /// `resolve_selection`.
    #[cfg(test)]
    fn affinity_score_group(
        candidates: &[OverlapCandidate],
        selection_temperature: f32,
    ) -> Vec<usize> {
        let selected_idx = if selection_temperature > 0.0 {
            Self::sample_by_temperature(candidates, selection_temperature)
        } else {
            candidates
                .iter()
                .max_by(|a, b| a.effective_score.total_cmp(&b.effective_score))
                .map(|candidate| candidate.idx)
        };
        let Some(selected_score) = selected_idx.and_then(|idx| {
            candidates
                .iter()
                .find(|candidate| candidate.idx == idx)
                .map(|candidate| candidate.effective_score)
        }) else {
            return Vec::new();
        };
        candidates
            .iter()
            .filter(|candidate| candidate.effective_score == selected_score)
            .map(|candidate| candidate.idx)
            .collect()
    }

    /// Pressure-select among the tenants holding the matched prefix.
    ///
    /// Every matched tenant serves the same prefix, so raw overlap cannot
    /// discriminate; waiting-prefill decay can split affinity score bands,
    /// then LeastLoad resolves only the selected equal-score group.
    fn select_matched_candidate(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        matched_tenants: &[TenantId],
        matched_units: usize,
        request: &RequestInputs<'_>,
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let avg_load = request.avg_load;
        let request_units = request.prompt_tokens;
        let matched_blocks = (matched_units / self.config.block_size.max(1)) as f64;
        let mut candidates: Vec<OverlapCandidate> = Vec::new();
        for &idx in healthy_indices {
            let url = workers[idx].url();
            if matched_tenants.iter().any(|tenant| tenant.as_ref() == url) {
                candidates.push(OverlapCandidate {
                    idx,
                    raw_score: matched_blocks,
                    effective_score: 1.0,
                });
            }
        }
        let waiting = self.waiting_prefill_snapshot();
        let tuning = OverlapTuning {
            overlap_decay: self.config.overlap_decay,
            selection_temperature: self.config.selection_temperature,
            waiting_prefill_tokens: waiting.as_deref(),
        };
        let request_blocks = (request_units / self.config.block_size).max(1);
        if cache_trace::enabled() {
            if healthy_indices.len() > 32 {
                cache_trace::mark_truncated();
            }
            for &idx in healthy_indices.iter().take(32) {
                let holder = matched_tenants
                    .iter()
                    .any(|tenant| tenant.as_ref() == workers[idx].url());
                cache_trace::score(serde_json::json!({
                    "source": "approximate_tree", "worker": workers[idx].url(),
                    "matched_units": holder.then_some(matched_units),
                    "match_status": if holder { "deepest_holder" } else { "shallower_match_unknown" },
                    "input_units": request.prompt_tokens, "block_size": self.config.block_size,
                }));
            }
        }
        Self::apply_overlap_decay(
            workers,
            &mut candidates,
            request_blocks,
            self.config.block_size,
            &tuning,
        );
        self.resolve_selection(
            workers,
            healthy_indices,
            &candidates,
            request,
            avg_load,
            info,
        )
    }

    /// Event-driven routing: KV index overlap scoring (Type 1).
    ///
    /// Self-contained — when overlap is found, selects the worker with the best
    /// cache match. When no overlap (cold start, novel tokens, short request),
    /// falls back to expected wait. Does NOT fall back to approximate token tree.
    fn select_worker_event_driven(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        avg_load: f64,
        index: &EventIndex<'_>,
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let EventIndex {
            indexer,
            block_size,
            model_id,
        } = index;
        let block_size = *block_size;

        let waiting_prefill_tokens = self.waiting_prefill_snapshot();
        let tuning = OverlapTuning {
            overlap_decay: self.config.overlap_decay,
            selection_temperature: self.config.selection_temperature,
            waiting_prefill_tokens: waiting_prefill_tokens.as_deref(),
        };

        // The engines fold the LoRA name and cache salt into their block
        // hashes and the monitor recomputes stored blocks under the same
        // seed, so a request in a namespace is hashed under it and matches
        // only its own blocks; a plain request keeps the plain hash.
        let content_hashes = match info.cache_namespace {
            Some(namespace) => {
                request_content_hashes_with_seed(tokens, block_size, namespace.event_seed())
            }
            None => compute_request_content_hashes(tokens, block_size),
        };
        let table = self.pool_table(workers, indexer, liveness::now_ms());
        // Over a sampled pool the candidates read are bounded as the pool
        // is: a tie among the deepest holders is drawn from, and an
        // all-workers policy ranks the sample and the deepest holders; a
        // shallow shared prefix is not a reason to read a thousand workers.
        let cap = (workers.len() > 2 * FLEET_SAMPLE).then_some(FLEET_SAMPLE);
        let gather = if self.exact_maximum_decision(&tuning) {
            Gather::TopGroup { cap }
        } else {
            let wants_all = self.selection.needs().all_workers || self.accounting.is_some();
            Gather::All {
                cap: cap.filter(|_| wants_all),
            }
        };
        let candidates = Self::overlap_candidates_for_hashes(
            workers,
            &content_hashes,
            &table,
            indexer,
            block_size,
            &tuning,
            gather,
        );
        // Chain hashes are computed only for policies (or accounting) keyed
        // on prefixes; the default decision never pays for them.
        let prefix_hashes: Vec<u64> =
            if self.selection.needs().prefix_hashes || self.accounting.is_some() {
                request_prefix_hashes(&content_hashes)
                    .into_iter()
                    .map(|hash| hash.0)
                    .collect()
            } else {
                Vec::new()
            };
        let request = RequestInputs {
            prompt_tokens: tokens.len(),
            block_size,
            request_blocks: content_hashes.len().max(1),
            avg_load,
            prefix_hashes: (!prefix_hashes.is_empty()).then_some(prefix_hashes.as_slice()),
        };
        // A miss for the warm-up slice is no overlap or a thin one: chat requests
        // share their template's head with every holder, which is affinity in
        // name only. Thin is at most the tree-mode `cache_threshold` share of
        // the request, or a few blocks outright when they are under half of
        // it (the head of a short request; a short request cached whole is a
        // hit and stays with its holder).
        let best_overlap = candidates
            .iter()
            .map(|candidate| candidate.raw_score)
            .fold(0.0_f64, f64::max);
        let request_blocks = request.request_blocks as f64;
        let thin_overlap = best_overlap / request_blocks <= f64::from(self.config.cache_threshold)
            || (best_overlap <= WARMUP_MISS_BLOCKS && best_overlap * 2.0 < request_blocks);
        if thin_overlap {
            if let Some(idx) = self.warmup_slice(workers, &table, indexer, info) {
                Metrics::record_worker_cache_aware_policy_branch("warmup_slice");
                debug!(
                    worker = workers[idx].url(),
                    branch = "warmup_slice",
                    request_blocks = content_hashes.len(),
                    "Cache miss routed to a warming worker"
                );
                return Some(idx);
            }
        } else if let Some(idx) = self.warmup_divert(
            workers,
            &table,
            indexer,
            info,
            best_overlap / request_blocks,
        ) {
            Metrics::record_worker_cache_aware_policy_branch("warmup_divert");
            debug!(
                worker = workers[idx].url(),
                branch = "warmup_divert",
                overlap_blocks = best_overlap as u64,
                request_blocks = content_hashes.len(),
                "Cache hit diverted to a thin worker"
            );
            return Some(idx);
        }
        let had_overlap = !candidates.is_empty();
        let idx = self.resolve_selection(
            workers,
            healthy_indices,
            &candidates,
            &request,
            avg_load,
            info,
        )?;
        // `overlap_blocks` is the chosen worker's undecayed overlap, so a
        // per-decision join against the engine's `cached_tokens` compares
        // blocks and not only the branch.
        let overlap_blocks = candidates
            .iter()
            .find(|candidate| candidate.idx == idx)
            .map_or(0, |candidate| candidate.raw_score as u64);
        let branch = if !had_overlap {
            "event_miss"
        } else if overlap_blocks > 0 {
            "event_hit"
        } else {
            "event_spill"
        };
        Metrics::record_worker_cache_aware_policy_branch(branch);
        debug!(
            worker = workers[idx].url(),
            branch,
            overlap_blocks,
            request_blocks = content_hashes.len(),
            policy = self.selection.name(),
            model_id,
            "Event-driven routing"
        );
        if cache_trace::enabled() {
            cache_trace::prediction(serde_json::json!({
                "source": "event_index_overlap", "branch": branch,
                "overlap_blocks": overlap_blocks, "request_blocks": content_hashes.len(), "block_size": block_size,
            }));
        }
        Some(idx)
    }

    /// Build positive-overlap candidates for event-driven routing.
    ///
    /// Returns each eligible worker with a positive, optionally decayed
    /// overlap score. This helper neither selects nor credits a worker:
    /// `affinity_score_group` chooses the effective-score group (exact maximum
    /// at temperature zero; a softmax-sampled group otherwise), and
    /// `select_final_from_affinity` uses LeastLoad to choose and credit one
    /// final worker from that group. An empty result means no full-block
    /// overlap. The tests name the eligible set themselves.
    #[cfg(test)]
    fn overlap_candidates(
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        indexer: &KvIndex,
        block_size: usize,
        tuning: &OverlapTuning<'_>,
    ) -> Vec<OverlapCandidate> {
        let content_hashes = compute_request_content_hashes(tokens, block_size);
        let table = PoolTable::build(workers, indexer, liveness::now_ms());
        let mut candidates = Self::overlap_holders(
            workers,
            &content_hashes,
            &table,
            indexer,
            Gather::All { cap: None },
        );
        candidates.retain(|candidate| healthy_indices.contains(&candidate.idx));
        if !candidates.is_empty() {
            Self::apply_overlap_decay(
                workers,
                &mut candidates,
                content_hashes.len(),
                block_size,
                tuning,
            );
        }
        candidates
    }

    /// `overlap_candidates` over already-computed block hashes.
    fn overlap_candidates_for_hashes(
        workers: &[Arc<dyn Worker>],
        content_hashes: &[ContentHash],
        table: &PoolTable,
        indexer: &KvIndex,
        block_size: usize,
        tuning: &OverlapTuning<'_>,
        gather: Gather,
    ) -> Vec<OverlapCandidate> {
        let mut candidates = Self::overlap_holders(workers, content_hashes, table, indexer, gather);
        if candidates.is_empty() {
            return candidates;
        }

        Self::apply_overlap_decay(
            workers,
            &mut candidates,
            content_hashes.len(),
            block_size,
            tuning,
        );

        candidates
    }

    /// The eligible holders of the request's blocks with their undecayed
    /// overlap, in slice order: the index names them by id, the pool table
    /// places them, and eligibility is read for them alone, so the cost
    /// follows the holders gathered and not the pool.
    fn overlap_holders(
        workers: &[Arc<dyn Worker>],
        content_hashes: &[ContentHash],
        table: &PoolTable,
        indexer: &KvIndex,
        gather: Gather,
    ) -> Vec<OverlapCandidate> {
        if content_hashes.is_empty() {
            return Vec::new();
        }

        let started = Instant::now();
        let overlap = indexer.find_matches(content_hashes, false);
        Metrics::record_kv_index_lookup(indexer.name(), started.elapsed().as_secs_f64());
        if overlap.scores.is_empty() {
            return Vec::new();
        }

        let mut candidates = match gather {
            Gather::TopGroup { cap } => Self::top_group(workers, table, &overlap, cap),
            Gather::All { cap } => Self::all_holders(workers, table, &overlap, cap),
        };
        // Downstream reads candidates in slice order: the all-workers merge
        // in `resolve_selection` and the tie-breaks.
        candidates.sort_unstable_by_key(|candidate| candidate.idx);
        candidates
    }

    /// The eligible holders of the deepest overlap: one pass over the
    /// scores for the depth, one to place and read those holders. When none
    /// of them is in this pool and routable (a PD leg's pool sharing the
    /// model's index with the other leg, a holder down), the full gather
    /// decides, so the result is always the deepest group a scan of the pool
    /// would have found.
    fn top_group(
        workers: &[Arc<dyn Worker>],
        table: &PoolTable,
        overlap: &OverlapScores,
        cap: Option<usize>,
    ) -> Vec<OverlapCandidate> {
        let deepest = overlap.scores.values().copied().max().unwrap_or(0);
        if deepest == 0 {
            return Vec::new();
        }
        let group = Self::holders_at(workers, table, overlap, deepest, cap);
        if !group.is_empty() {
            return group;
        }
        let mut all = Self::all_holders(workers, table, overlap, None);
        let best = all
            .iter()
            .map(|candidate| candidate.raw_score)
            .fold(0.0_f64, f64::max);
        all.retain(|candidate| candidate.raw_score == best);
        Self::draw(&mut all, cap);
        all
    }

    /// The eligible holders in this pool whose overlap is exactly `depth`
    /// blocks; `cap` of the tied holders, drawn uniformly, when they are more.
    fn holders_at(
        workers: &[Arc<dyn Worker>],
        table: &PoolTable,
        overlap: &OverlapScores,
        depth: u32,
        cap: Option<usize>,
    ) -> Vec<OverlapCandidate> {
        let mut tied: Vec<u32> = overlap
            .scores
            .iter()
            .filter(|(_, &score)| score == depth)
            .map(|(&id, _)| id)
            .collect();
        Self::draw(&mut tied, cap);
        let mut group = Vec::with_capacity(tied.len());
        for id in tied {
            let Some(idx) = table.position(id, workers) else {
                continue;
            };
            if !workers[idx].routing_state().eligible() {
                continue;
            }
            group.push(OverlapCandidate {
                idx,
                raw_score: f64::from(depth),
                effective_score: f64::from(depth),
            });
        }
        group
    }

    /// Keep a uniform draw of `cap` items when there are more.
    fn draw<T>(items: &mut Vec<T>, cap: Option<usize>) {
        let Some(cap) = cap else {
            return;
        };
        if items.len() <= cap {
            return;
        }
        let mut rng = rand::rng();
        for i in 0..cap {
            let j = rng.random_range(i..items.len());
            items.swap(i, j);
        }
        items.truncate(cap);
    }

    /// Every eligible holder in this pool with a positive overlap; the `cap`
    /// deepest of them when set.
    fn all_holders(
        workers: &[Arc<dyn Worker>],
        table: &PoolTable,
        overlap: &OverlapScores,
        cap: Option<usize>,
    ) -> Vec<OverlapCandidate> {
        let mut candidates: Vec<OverlapCandidate> = Vec::with_capacity(overlap.scores.len());
        for (&id, &score) in &overlap.scores {
            if score == 0 {
                continue;
            }
            let Some(idx) = table.position(id, workers) else {
                continue;
            };
            if !workers[idx].routing_state().eligible() {
                continue;
            }
            candidates.push(OverlapCandidate {
                idx,
                raw_score: f64::from(score),
                effective_score: f64::from(score),
            });
        }
        if let Some(cap) = cap {
            if candidates.len() > cap {
                candidates
                    .select_nth_unstable_by(cap - 1, |a, b| b.raw_score.total_cmp(&a.raw_score));
                candidates.truncate(cap);
            }
        }
        candidates
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
        let backlog_of = |c: &OverlapCandidate| {
            waiting
                .get(workers[c.idx].url())
                .map(|load| load.total_waiting_uncached_tokens().max(0))
        };
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

    /// Softmax selection over min-max normalized effective scores. The
    /// normalization makes temperature scale-free: only a candidate's
    /// relative position within the current score spread matters, so one
    /// temperature setting behaves the same whether overlaps span 2 blocks
    /// or 2000. The best candidate's exponent is exactly 0 (overflow-safe);
    /// a degenerate spread (all equal) is a uniform draw. Inverse-CDF
    /// sampling with a last-row fallback against floating-point drift.
    #[cfg(test)]
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

    /// One decision per tree-routed request: the branch counter and match
    /// ratio histogram always, the debug line when enabled. `selected_url ==
    /// None` means the caller fell back to `fallback_url` (first healthy).
    fn log_tree_decision(
        &self,
        selected_url: Option<&str>,
        fallback_url: Option<&str>,
        matched_units: usize,
        input_units: usize,
        matched_tenants: &[TenantId],
        model_id: &str,
    ) {
        // The branch mirrors the selection's f32 arithmetic; the histogram
        // divides in f64 so exact deciles stay in their bucket (widening the
        // f32 ratio would turn 1/10 into 0.100000001 > 0.1).
        let (matched_ratio, histogram_ratio) = if input_units == 0 {
            (0.0, 0.0)
        } else {
            (
                matched_units as f32 / input_units as f32,
                matched_units as f64 / input_units as f64,
            )
        };
        let branch = match selected_url {
            None => "first_healthy_fallback",
            Some(_) if matched_ratio <= self.config.cache_threshold => "expected_wait_fallback",
            Some(url) => {
                if matched_tenants.iter().any(|tenant| tenant.as_ref() == url) {
                    "tree_match"
                } else {
                    "spill"
                }
            }
        };
        Metrics::record_worker_cache_aware_policy_branch(branch);
        Metrics::record_cache_aware_match_ratio(histogram_ratio);
        // `credited_units` is what the served worker is expected to have
        // cached: the matched prefix only when it serves the request. On the
        // fallback and spill branches the match belongs to another tenant, so
        // a join against the engine's cached tokens must not count it.
        let credited_units = if branch == "tree_match" {
            matched_units
        } else {
            0
        };
        if cache_trace::enabled() {
            cache_trace::prediction(serde_json::json!({
                "source": "approximate_tree", "branch": branch,
                "matched_units": matched_units, "input_units": input_units, "credited_units": credited_units,
            }));
        }
        debug!(
            index = "tree",
            branch,
            worker = selected_url.or(fallback_url).unwrap_or("none"),
            model_id,
            matched_ratio = f64::from(matched_ratio),
            threshold = f64::from(self.config.cache_threshold),
            matched_units,
            input_units,
            credited_units,
            "Cache-aware selection"
        );
    }

    /// One decision line per hash-mode selection. `level` is the matched
    /// boundary (0 for the fallback branches).
    fn log_hash_decision(branch: &'static str, level: usize, worker: &str, model_id: &str) {
        if cache_trace::enabled() {
            cache_trace::prediction(serde_json::json!({
                "source": "approximate_hash_index", "branch": branch, "level": level,
            }));
        }
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
        avg_load: f64,
        model_id: &str,
    ) -> Option<usize> {
        let now = Instant::now();

        // Hash mode keys on token ids; untokenized requests stay load-balanced.
        // Unpartitioned heads are stripped of marker material like the tree
        // keys, so the placement key spaces stay disjoint by construction too.
        let tokens = match info.cache_namespace {
            Some(_) => info.tokens,
            None => info.tokens.map(CacheNamespace::unpartitioned_tokens),
        };
        let Some(tokens) = tokens.filter(|t| !t.is_empty()) else {
            return self.hash_expected_wait(
                workers,
                info,
                healthy_indices,
                model_id,
                "expected_wait_fallback",
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
            return self.hash_expected_wait(
                workers,
                info,
                healthy_indices,
                model_id,
                "short_request",
                tokens,
                applicable,
                now,
            );
        }

        if self.is_kv_imbalanced(workers, healthy_indices) {
            return self.hash_expected_wait(
                workers,
                info,
                healthy_indices,
                model_id,
                "kv_pressure_expected_wait",
                tokens,
                applicable,
                now,
            );
        }

        let url_to_idx = Self::healthy_url_index(workers, healthy_indices);
        for &boundary in applicable.iter().rev() {
            let key = (
                boundary,
                hash_token_head(info.cache_namespace, &tokens[..boundary]),
            );
            let holder_candidates = self.live_holder_candidates(&url_to_idx, model_id, key, now);
            if holder_candidates.is_empty() {
                continue;
            }
            let selected = self.select_final_from_affinity(
                workers,
                &holder_candidates,
                healthy_indices,
                avg_load,
                info,
            )?;
            let branch = if holder_candidates.contains(&selected) {
                "hash_hit"
            } else {
                "hash_spill"
            };
            self.record_placement(
                model_id,
                info.cache_namespace,
                tokens,
                applicable,
                workers[selected].url(),
                now,
            );
            Self::log_hash_decision(branch, boundary, workers[selected].url(), model_id);
            return Some(selected);
        }

        self.hash_expected_wait(
            workers,
            info,
            healthy_indices,
            model_id,
            "expected_wait_fallback",
            tokens,
            applicable,
            now,
        )
    }

    /// Expected-wait dispatch for hash-path fallback branches; records only
    /// the final worker when boundaries apply.
    #[expect(clippy::too_many_arguments, reason = "hot-path plumbing, not state")]
    fn hash_expected_wait(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        healthy_indices: &[usize],
        model_id: &str,
        branch: &'static str,
        tokens: &[u32],
        applicable: &[usize],
        now: Instant,
    ) -> Option<usize> {
        let idx = self.select_expected_wait(workers, healthy_indices, info)?;
        if !applicable.is_empty() {
            self.record_placement(
                model_id,
                info.cache_namespace,
                tokens,
                applicable,
                workers[idx].url(),
                now,
            );
        }
        Self::log_hash_decision(branch, 0, workers[idx].url(), model_id);
        Some(idx)
    }

    /// URL → healthy worker indices. Duplicate URLs represent equal-affinity
    /// candidates and remain available to LeastLoad's expected-wait tie-break.
    fn healthy_url_index<'a>(
        workers: &'a [Arc<dyn Worker>],
        healthy_indices: &[usize],
    ) -> HashMap<&'a str, Vec<usize>> {
        let mut url_to_idx: HashMap<&str, Vec<usize>> =
            HashMap::with_capacity(healthy_indices.len());
        for &idx in healthy_indices {
            url_to_idx.entry(workers[idx].url()).or_default().push(idx);
        }
        url_to_idx
    }

    /// Every healthy worker whose URL is a live holder of `key`.
    /// Expired holders are pruned in place (lazy expiry on read). Guarded
    /// work is O(holder cap): resolution goes through `url_to_idx`.
    fn live_holder_candidates(
        &self,
        url_to_idx: &HashMap<&str, Vec<usize>>,
        model_id: &str,
        key: (usize, u64),
        now: Instant,
    ) -> Vec<usize> {
        let ttl = Duration::from_secs(self.config.cache_ttl_secs);
        let Some(model) = self
            .placement_index
            .get(model_id)
            .map(|entry| Arc::clone(entry.value()))
        else {
            return Vec::new();
        };
        let Some(mut holders) = model.get_mut(&key) else {
            return Vec::new();
        };
        holders.retain(|h| now.duration_since(h.last_touch) <= ttl);
        holders
            .iter()
            .filter_map(|h| url_to_idx.get(h.worker_url.as_str()))
            .flatten()
            .copied()
            .collect()
    }

    /// Record/touch `worker_url` at every applicable boundary of this
    /// request; above the holder cap the stalest holder is evicted.
    fn record_placement(
        &self,
        model_id: &str,
        namespace: Option<CacheNamespace>,
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
            let key = (boundary, hash_token_head(namespace, &tokens[..boundary]));
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
        tokens: &[u32],
        healthy_indices: &[usize],
        avg_load: f64,
        model_id: &str,
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let tree = tree_for_model(&self.token_trees, model_id, || self.new_token_tree());

        // A partitioned request keys under its namespace marker: another
        // namespace diverges at position 0 and can never match, and the
        // marker never counts toward the match ratio. The marker fills whole
        // pages, so the prompt's page grid is the unpartitioned one.
        let page_size = self.config.block_size.max(1);
        let prefixed;
        let (keyed, marker_len): (&[u32], usize) = match info.cache_namespace {
            Some(namespace) => {
                prefixed = namespace.prefixed_tokens(tokens, page_size);
                (&prefixed, CacheNamespace::token_marker_len(page_size))
            }
            None => (CacheNamespace::unpartitioned_tokens(tokens), 0),
        };

        // One fused descent: match first, atomically choose+credit the final
        // worker from the affinity/spill candidate set, then insert only that
        // worker before the closure returns.
        let mut selected_idx: Option<usize> = None;
        let result = tree.match_and_insert_with(keyed, |result| {
            let matched = result.matched_token_count.saturating_sub(marker_len);
            let input = result.input_token_count.saturating_sub(marker_len);
            let match_rate = if input == 0 {
                0.0
            } else {
                matched as f32 / input as f32
            };
            let request = self.tree_request(tokens.len(), avg_load);
            selected_idx = if match_rate > self.config.cache_threshold {
                self.select_matched_candidate(
                    workers,
                    healthy_indices,
                    &result.matched_tenants,
                    matched,
                    &request,
                    info,
                )
            } else {
                self.resolve_selection(workers, healthy_indices, &[], &request, avg_load, info)
            };
            selected_idx.map(|idx| workers[idx].url())
        });

        self.log_tree_decision(
            selected_idx.map(|idx| workers[idx].url()),
            healthy_indices.first().map(|&idx| workers[idx].url()),
            result.matched_token_count.saturating_sub(marker_len),
            result.input_token_count.saturating_sub(marker_len),
            &result.matched_tenants,
            model_id,
        );

        let idx = selected_idx?;
        if self.should_populate_hash_index() {
            let matched_prefix: Vec<u32> = keyed[..result.matched_token_count].to_vec();
            let node_hash = kv_index::hash_token_path(keyed);
            self.hash_index
                .entry(model_id.to_string())
                .or_default()
                .token_tree
                .insert(node_hash, matched_prefix);
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
        Some(idx)
    }

    /// Select worker using string-based tree (HTTP path)
    fn select_worker_with_text(
        &self,
        workers: &[Arc<dyn Worker>],
        text: &str,
        healthy_indices: &[usize],
        avg_load: f64,
        model_id: &str,
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let tree = tree_for_model(&self.string_trees, model_id, Tree::new);

        // Same partition rule as the token tree, in chars.
        let prefixed;
        let (keyed, marker_len): (&str, usize) = match info.cache_namespace {
            Some(namespace) => {
                prefixed = namespace.prefixed_text(text);
                (&prefixed, TEXT_MARKER_LEN)
            }
            None => (CacheNamespace::unpartitioned_text(text), 0),
        };

        let mut selected_idx: Option<usize> = None;
        let result = tree.match_and_insert_with(keyed, |result| {
            let matched = result.matched_char_count.saturating_sub(marker_len);
            let input = result.input_char_count.saturating_sub(marker_len);
            let match_rate = if input == 0 {
                0.0
            } else {
                matched as f32 / input as f32
            };
            let request = self.tree_request(input, avg_load);
            selected_idx = if match_rate > self.config.cache_threshold {
                self.select_matched_candidate(
                    workers,
                    healthy_indices,
                    &result.matched_tenants,
                    matched,
                    &request,
                    info,
                )
            } else {
                self.resolve_selection(workers, healthy_indices, &[], &request, avg_load, info)
            };
            selected_idx.map(|idx| workers[idx].url())
        });

        self.log_tree_decision(
            selected_idx.map(|idx| workers[idx].url()),
            healthy_indices.first().map(|&idx| workers[idx].url()),
            result.matched_char_count.saturating_sub(marker_len),
            result.input_char_count.saturating_sub(marker_len),
            &result.matched_tenants,
            model_id,
        );

        let idx = selected_idx?;
        if self.should_populate_hash_index() {
            let matched_prefix: String = keyed.chars().take(result.matched_char_count).collect();
            let path_hash = kv_index::hash_node_path(keyed);
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
        Some(idx)
    }
}

impl Default for CacheAwarePolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use kv_index::{compute_content_hash, SequenceHash, StoredBlock};
    use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};
    use openai_protocol::worker::{
        HealthCheckConfig, SchedulerLoadSnapshot, WorkerLoadResponse, WorkerStatus,
    };
    use tokio::sync::watch;
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

    /// Exercise the same affinity-candidate and score-group stages used by
    /// production immediately before LeastLoad resolves the final worker.
    fn overlap_affinity_group(
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        indexer: &KvIndex,
        block_size: usize,
        tuning: &OverlapTuning<'_>,
    ) -> Vec<usize> {
        let candidates = CacheAwarePolicy::overlap_candidates(
            workers,
            tokens,
            healthy_indices,
            indexer,
            block_size,
            tuning,
        );
        CacheAwarePolicy::affinity_score_group(&candidates, tuning.selection_temperature)
    }
    use crate::{
        observability::metrics::CACHE_AWARE_MATCH_RATIO_BUCKETS,
        worker::{BasicWorkerBuilder, WorkerBlocks, WorkerType},
    };

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

    /// Single-DP load snapshot with the given waiting-prefill backlog.
    fn waiting_load(waiting_uncached: i32) -> WorkerLoadResponse {
        WorkerLoadResponse {
            loads: vec![SchedulerLoadSnapshot {
                num_waiting_uncached_tokens: waiting_uncached,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

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

    /// Single-DP snapshot used to make live request count and LeastLoad's
    /// expected-wait score disagree deterministically.
    fn expected_wait_load(
        queued_tokens: i32,
        token_usage: f64,
        gen_throughput: f64,
    ) -> WorkerLoadResponse {
        WorkerLoadResponse {
            loads: vec![SchedulerLoadSnapshot {
                num_waiting_uncached_tokens: queued_tokens,
                token_usage,
                gen_throughput,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn complete_load_snapshot(loads: HashMap<String, WorkerLoadResponse>) -> Arc<LoadSnapshot> {
        LoadSnapshot::from_loads_for_test(loads.into_iter().collect())
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

    fn update_expected_wait_loads(
        policy: &CacheAwarePolicy,
        workers: &[Arc<dyn Worker>],
        queued_tokens: &[i32],
    ) {
        let loads = workers
            .iter()
            .zip(queued_tokens)
            .map(|(worker, &queued)| {
                (
                    worker.url().to_string(),
                    expected_wait_load(queued, 0.1, 100.0),
                )
            })
            .collect();
        policy.update_loads(&loads);
    }

    fn assert_only_final_worker_credited(
        policy: &CacheAwarePolicy,
        workers: &[Arc<dyn Worker>],
        final_idx: usize,
        expected_tokens: u64,
    ) {
        for (idx, worker) in workers.iter().enumerate() {
            assert_eq!(
                worker.processed_requests(),
                usize::from(idx == final_idx),
                "worker {idx} processed count must reflect exactly one final dispatch"
            );
            let (_, credited_tokens, credited_requests) =
                policy.load_scorer.load_state_for_test(worker.url());
            assert_eq!(
                (credited_tokens, credited_requests),
                if idx == final_idx {
                    (expected_tokens, 1)
                } else {
                    (0, 0)
                },
                "worker {idx} LeastLoad credit must reflect exactly one final dispatch"
            );
        }
    }

    /// Inject a backend KV snapshot (utilization per worker, by index). Returns
    /// the sender; bind it (`let _tx = ...`) to keep the watch channel open.
    fn inject_kv(
        policy: &CacheAwarePolicy,
        workers: &[Arc<dyn Worker>],
        usages: &[f64],
    ) -> watch::Sender<Arc<LoadSnapshot>> {
        let snapshot = LoadSnapshot::from_loads_for_test(
            workers
                .iter()
                .zip(usages)
                .map(|(w, &u)| (w.url().to_string(), kv_load(u)))
                .collect(),
        );
        let (tx, rx) = watch::channel(snapshot);
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

    #[test]
    fn cache_aware_always_requests_backend_load_updates() {
        // Cache misses, affinity ties, and spills all use the embedded
        // expected-wait scorer, so even default CacheAware configuration must
        // participate in WorkerMonitor load polling.
        let policy = CacheAwarePolicy::with_config(test_config());
        assert!(policy.needs_backend_loads());
    }

    #[test]
    fn cache_aware_load_update_resets_atomic_inflight_credit() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        workers[1].increment_load();
        update_expected_wait_loads(&policy, &workers, &[0, 500]);

        let route_miss = |text| {
            policy.select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some(text),
                    ..Default::default()
                },
            )
        };
        assert_eq!(route_miss("alpha miss"), Some(0));
        assert_eq!(
            policy.load_scorer.load_state_for_test(workers[0].url()),
            (true, 1024, 1)
        );
        assert_eq!(
            route_miss("zulu miss"),
            Some(1),
            "first dispatch credit must steer the second miss away"
        );
        assert_eq!(
            policy.load_scorer.load_state_for_test(workers[1].url()),
            (true, 1024, 1)
        );

        update_expected_wait_loads(&policy, &workers, &[0, 500]);
        for worker in &workers {
            assert_eq!(
                policy.load_scorer.load_state_for_test(worker.url()),
                (true, 0, 0),
                "fresh load update must clear exactly the since-poll credit"
            );
        }
        assert_eq!(
            route_miss("omega miss"),
            Some(0),
            "fresh backend load must reset since-poll dispatch credit"
        );
    }

    #[test]
    fn cache_aware_ignores_cached_load_after_complete_snapshot_drops_worker() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);

        let old_loads = HashMap::from([
            (
                workers[0].url().to_string(),
                expected_wait_load(0, 0.1, 100.0),
            ),
            (
                workers[1].url().to_string(),
                expected_wait_load(1_000, 0.1, 100.0),
            ),
        ]);
        policy.update_loads(&old_loads);
        let (load_tx, load_rx) = watch::channel(complete_load_snapshot(old_loads));
        policy.set_load_receiver(Some(load_rx));

        // The next complete WorkerMonitor snapshot drops w1 after its load
        // fetch fails. Its old cached report says idle, but its five live
        // requests make the documented missing-snapshot estimate (51.2s)
        // worse than w2's reported 10s queue.
        for _ in 0..5 {
            workers[0].increment_load();
        }
        load_tx
            .send(complete_load_snapshot(HashMap::from([(
                workers[1].url().to_string(),
                expected_wait_load(1_000, 0.1, 100.0),
            )])))
            .unwrap();

        let selected = policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("snapshot-pruning miss"),
                ..Default::default()
            },
        );
        assert_eq!(
            selected,
            Some(1),
            "a report absent from the complete snapshot must not remain scoreable"
        );
        assert_only_final_worker_credited(&policy, &workers, 1, 1024);
    }

    #[test]
    fn complete_snapshot_masks_absence_without_replacing_poll_fed_scores() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);

        // The poll-fed scorer prefers w2. Both URLs remain present in the
        // complete snapshot, whose deliberately contradictory values prefer
        // w1. The watch map is an absence mask, not a second score source.
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);
        let current_loads = HashMap::from([
            (
                workers[0].url().to_string(),
                expected_wait_load(0, 0.1, 100.0),
            ),
            (
                workers[1].url().to_string(),
                expected_wait_load(10_000, 0.1, 100.0),
            ),
        ]);
        let (_load_tx, load_rx) = watch::channel(complete_load_snapshot(current_loads));
        policy.set_load_receiver(Some(load_rx));

        let selected = policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("coherent-snapshot miss"),
                ..Default::default()
            },
        );
        assert_eq!(selected, Some(1));
        assert_only_final_worker_credited(&policy, &workers, 1, 1024);
    }

    #[test]
    fn monitor_update_before_watch_publish_keeps_load_and_credit_paired() {
        use std::sync::mpsc::sync_channel;

        let policy = Arc::new(CacheAwarePolicy::with_config(test_config()));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);

        let revision_a = HashMap::from([
            (
                workers[0].url().to_string(),
                expected_wait_load(500, 0.1, 100.0),
            ),
            (
                workers[1].url().to_string(),
                expected_wait_load(0, 0.1, 100.0),
            ),
        ]);
        policy.update_loads(&revision_a);
        let (load_tx, load_rx) = watch::channel(complete_load_snapshot(revision_a));
        policy.set_load_receiver(Some(load_rx));

        assert_eq!(
            policy.select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some("alpha revision-a miss"),
                    ..Default::default()
                },
            ),
            Some(1)
        );
        assert_eq!(
            policy.load_scorer.load_state_for_test(workers[1].url()),
            (true, 1024, 1)
        );

        // Match WorkerMonitor's production order: push successful reports and
        // reset their credits, pause, then publish the pruned complete watch
        // snapshot. During the pause, w2's new heavy load must stay paired
        // with its reset instead of scoring the old watch value (idle).
        let revision_b = HashMap::from([(
            workers[1].url().to_string(),
            expected_wait_load(10_000, 0.1, 100.0),
        )]);
        let (updated_tx, updated_rx) = sync_channel::<()>(0);
        let (publish_tx, publish_rx) = sync_channel::<()>(0);
        let updater_policy = Arc::clone(&policy);
        let updater = std::thread::spawn(move || {
            updater_policy.update_loads(&revision_b);
            updated_tx.send(()).unwrap();
            publish_rx.recv().unwrap();
            load_tx.send(complete_load_snapshot(revision_b)).unwrap();
        });

        updated_rx.recv().unwrap();
        let reset_state = policy.load_scorer.load_state_for_test(workers[1].url());
        let selected_during_publication_gap = policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("zulu publication-gap miss"),
                ..Default::default()
            },
        );
        publish_tx.send(()).unwrap();
        updater.join().unwrap();

        assert_eq!(reset_state, (true, 0, 0));
        assert_eq!(
            selected_during_publication_gap,
            Some(0),
            "selection must pair w2's new report with its newly reset credit"
        );
    }

    #[test]
    fn cache_aware_reset_discards_expected_wait_state() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[100_000, 0]);
        let reset_snapshot = HashMap::from([
            (
                workers[0].url().to_string(),
                expected_wait_load(100_000, 0.1, 100.0),
            ),
            (
                workers[1].url().to_string(),
                expected_wait_load(0, 0.1, 100.0),
            ),
        ]);
        let (_load_tx, load_rx) = watch::channel(complete_load_snapshot(reset_snapshot));
        policy.set_load_receiver(Some(load_rx));

        let first = SelectWorkerInfo {
            request_text: Some("alpha reset miss"),
            ..Default::default()
        };
        assert_eq!(policy.select_worker(&workers, &first), Some(1));
        assert_eq!(
            policy.load_scorer.load_state_for_test(workers[1].url()),
            (true, 1024, 1)
        );

        policy.reset();
        for worker in &workers {
            assert_eq!(
                policy.load_scorer.load_state_for_test(worker.url()),
                (false, 0, 0),
                "reset must clear cached backend load and in-flight credit"
            );
        }
        let second = SelectWorkerInfo {
            request_text: Some("zulu reset miss"),
            ..Default::default()
        };
        assert_eq!(
            policy.select_worker(&workers, &second),
            Some(0),
            "reset must return the scorer to its dark-fleet live-load fallback"
        );
    }

    #[test]
    fn cache_aware_worker_removal_prunes_expected_wait_state() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[100_000, 0]);

        let first = SelectWorkerInfo {
            request_text: Some("alpha removal miss"),
            ..Default::default()
        };
        assert_eq!(policy.select_worker(&workers, &first), Some(1));

        LoadBalancingPolicy::remove_worker(&policy, workers[0].url());
        assert_eq!(
            policy.load_scorer.load_state_for_test(workers[0].url()),
            (false, 0, 0),
            "worker removal must prune cached backend load and in-flight credit"
        );
        assert_eq!(
            policy.load_scorer.load_state_for_test(workers[1].url()),
            (true, 1024, 1),
            "worker removal must preserve other workers' state"
        );
        // Scored like its reporting peer once the stale queue is gone, the
        // removed worker is picked again instead of shunned.
        let picked_removed = (0..50).any(|i| {
            let text = format!("{i} removal miss");
            let miss = SelectWorkerInfo {
                request_text: Some(&text),
                ..Default::default()
            };
            policy.select_worker(&workers, &miss) == Some(0)
        });
        assert!(
            picked_removed,
            "removed worker's stale heavy snapshot must not survive cleanup"
        );
    }

    // ---- per-request candidate gate + matched-tenant selection ----

    fn seed_string_tenants(
        policy: &CacheAwarePolicy,
        workers: &[Arc<dyn Worker>],
        text: &str,
        tenant_urls: &[&str],
    ) -> Arc<Tree> {
        policy.init_workers(workers);
        let model_key = normalize_model_key(workers[0].model_id()).to_string();
        let tree = Arc::clone(policy.string_trees.get(&model_key).unwrap().value());
        for url in tenant_urls {
            tree.insert_text(text, url);
        }
        tree
    }

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
    fn string_tree_unique_hit_keeps_affinity_and_credits_expected_wait() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let cached = "alpha unique cached prompt";
        seed_string_tenants(&policy, &workers, cached, &["http://w1:8000"]);
        workers[1].increment_load();
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

        let hit = SelectWorkerInfo {
            request_text: Some(cached),
            ..Default::default()
        };
        assert_eq!(policy.select_worker(&workers, &hit), Some(0));
        assert_only_final_worker_credited(&policy, &workers, 0, 1024);
    }

    #[test]
    fn string_tree_miss_uses_expected_wait_and_records_final_worker() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);
        let text = "novel string-tree miss with no shared prefix";

        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some(text),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(selected, 1, "expected wait must beat raw request count");
        assert_only_final_worker_credited(&policy, &workers, 1, 1024);

        let model = normalize_model_key(workers[0].model_id());
        let tree = policy.string_trees.get(model).unwrap();
        let matched = tree.match_prefix_with_counts(text);
        assert!(matched
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == workers[1].url()));
        assert!(!matched
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == workers[0].url()));
    }

    #[test]
    fn string_tree_spill_uses_expected_wait_and_records_only_final_worker() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let text = "shared string prefix that must spill";
        let tree = seed_string_tenants(&policy, &workers, text, &["http://w1:8000"]);
        for _ in 0..100 {
            workers[0].increment_load();
        }
        for _ in 0..10 {
            workers[2].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[20_000, 10_000, 0]);

        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some(text),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_only_final_worker_credited(&policy, &workers, 2, 1024);
        assert_eq!(selected, 2, "spill must use expected wait over the fleet");

        let matched = tree.match_prefix_with_counts(text);
        assert!(matched
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == workers[2].url()));
        assert!(!matched
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == workers[1].url()));
    }

    #[test]
    fn string_tree_equal_affinity_tie_uses_expected_wait_only_among_holders() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let text = "equal string affinity";
        seed_string_tenants(
            &policy,
            &workers,
            text,
            &["http://w1:8000", "http://w2:8000"],
        );
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[10_000, 0, 0]);

        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    request_text: Some(text),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(selected, 1);
        assert_only_final_worker_credited(&policy, &workers, 1, 1024);
    }

    #[test]
    fn token_tree_unique_hit_keeps_affinity_and_credits_expected_wait() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let cached: Vec<u32> = (0..8).collect();
        seed_token_tenants(&policy, &workers, &cached, &["http://w1:8000"]);
        workers[1].increment_load();
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

        let hit = SelectWorkerInfo {
            tokens: Some(&cached),
            ..Default::default()
        };
        assert_eq!(policy.select_worker(&workers, &hit), Some(0));
        assert_only_final_worker_credited(&policy, &workers, 0, 8);
    }

    #[test]
    fn token_tree_miss_uses_expected_wait_and_records_final_worker() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);
        let tokens: Vec<u32> = (100..108).collect();

        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(selected, 1, "expected wait must beat raw request count");
        assert_only_final_worker_credited(&policy, &workers, 1, 8);

        let model = normalize_model_key(workers[0].model_id());
        let tree = policy.token_trees.get(model).unwrap();
        let matched = tree.match_prefix_with_counts(&tokens);
        assert!(matched
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == workers[1].url()));
        assert!(!matched
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == workers[0].url()));
    }

    #[test]
    fn token_tree_spill_uses_expected_wait_and_records_only_final_worker() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let tokens: Vec<u32> = (0..8).collect();
        let tree = seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
        for _ in 0..100 {
            workers[0].increment_load();
        }
        for _ in 0..10 {
            workers[2].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[20_000, 10_000, 0]);

        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(selected, 2, "spill must use expected wait over the fleet");
        assert_only_final_worker_credited(&policy, &workers, 2, 8);

        let matched = tree.match_prefix_with_counts(&tokens);
        assert!(matched
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == workers[2].url()));
        assert!(!matched
            .matched_tenants
            .iter()
            .any(|tenant| tenant.as_ref() == workers[1].url()));
    }

    #[test]
    fn token_tree_equal_affinity_tie_uses_expected_wait_only_among_holders() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let tokens: Vec<u32> = (0..8).collect();
        seed_token_tenants(
            &policy,
            &workers,
            &tokens,
            &["http://w1:8000", "http://w2:8000"],
        );
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[10_000, 0, 0]);

        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(selected, 1);
        assert_only_final_worker_credited(&policy, &workers, 1, 8);
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
    fn gated_hot_tenant_spills_to_expected_wait_and_replicates() {
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
        assert_eq!(idx, 1, "gated selection must spill to expected wait");

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
        assert!(logs_contain("expected_wait_fallback"));
    }

    /// The branch counter and match-ratio histogram are recorded on every
    /// tree decision, independent of whether the debug line is enabled, and
    /// the ratio is bucketed the way `start_prometheus` registers it: an
    /// exact decile (32 of 320) must land in `le="0.1"`, not widen past it.
    #[test]
    fn token_tree_selection_records_branch_and_match_ratio_metrics() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let tokens: Vec<u32> = (0..32).collect();
        seed_token_tenants(&policy, &workers, &tokens, &["http://w1:8000"]);
        let novel: Vec<u32> = (1000..1064).collect();
        let decile: Vec<u32> = (0..320).collect();

        let recorder = PrometheusBuilder::new()
            .set_buckets_for_metric(
                Matcher::Full(String::from("smg_cache_aware_match_ratio")),
                CACHE_AWARE_MATCH_RATIO_BUCKETS,
            )
            .unwrap()
            .build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            // A full-prefix hit (1.0), a novel prompt (0.0), and a prompt
            // sharing only the seeded 32 tokens of 320 (0.1).
            for request in [&tokens, &novel, &decile] {
                policy
                    .select_worker(
                        &workers,
                        &SelectWorkerInfo {
                            tokens: Some(request),
                            ..Default::default()
                        },
                    )
                    .unwrap();
            }
        });
        let rendered = handle.render();

        for series in [
            "smg_cache_aware_policy_branch_total{branch=\"tree_match\"} 1",
            "smg_cache_aware_policy_branch_total{branch=\"expected_wait_fallback\"} 2",
            "smg_cache_aware_match_ratio_bucket{le=\"0\"} 1",
            "smg_cache_aware_match_ratio_bucket{le=\"0.1\"} 2",
            "smg_cache_aware_match_ratio_bucket{le=\"0.9\"} 2",
            "smg_cache_aware_match_ratio_bucket{le=\"1\"} 3",
            "smg_cache_aware_match_ratio_count 3",
        ] {
            assert!(
                rendered.lines().any(|l| l == series),
                "{series} missing; rendered:\n{rendered}"
            );
        }
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
    fn kv_pressure_still_forces_expected_wait_fallback() {
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

        // Seed both tenants explicitly; worker cleanup is the behavior under
        // test, independent of random tie-breaking in a fully dark fleet.
        let model_key = normalize_model_key(workers[0].model_id()).to_string();
        let string_tree = Arc::clone(policy.string_trees.get(&model_key).unwrap().value());
        let token_tree = Arc::clone(policy.token_trees.get(&model_key).unwrap().value());
        string_tree.insert_text("purge me please", workers[0].url());
        string_tree.insert_text("keep me around", workers[1].url());
        let tokens_a: Vec<u32> = (0..32).collect();
        let tokens_b: Vec<u32> = (1000..1032).collect();
        token_tree.insert_tokens(&tokens_a, workers[0].url());
        token_tree.insert_tokens(&tokens_b, workers[1].url());

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
    fn test_remove_workers_from_model_does_not_mutate_other_models() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            cache_boundaries: vec![16],
            ..Default::default()
        });
        let worker_url = "http://shared-url:8000";
        let tokens: Vec<u32> = (0..32).collect();

        for model_id in ["retired-model", "active-model"] {
            let string_tree = Arc::new(Tree::new());
            string_tree.insert_text("shared prefix", worker_url);
            policy
                .string_trees
                .insert(model_id.to_string(), string_tree);

            let token_tree = Arc::new(policy.new_token_tree());
            token_tree.insert_tokens(&tokens, worker_url);
            policy.token_trees.insert(model_id.to_string(), token_tree);

            policy.record_placement(model_id, None, &tokens, &[16], worker_url, Instant::now());
        }

        policy.remove_workers_from_model("retired-model", &HashSet::from([worker_url.to_string()]));

        let retired_string = policy.string_trees.get("retired-model").unwrap();
        let retired_token = policy.token_trees.get("retired-model").unwrap();
        assert!(!retired_string
            .get_tenant_char_count()
            .contains_key(worker_url));
        assert!(!retired_token
            .get_tenant_token_counts()
            .contains_key(worker_url));
        assert!(policy
            .placement_index
            .get("retired-model")
            .is_some_and(|placements| placements.is_empty()));

        let active_string = policy.string_trees.get("active-model").unwrap();
        let active_token = policy.token_trees.get("active-model").unwrap();
        assert!(active_string
            .get_tenant_char_count()
            .contains_key(worker_url));
        assert!(active_token
            .get_tenant_token_counts()
            .contains_key(worker_url));
        assert!(policy
            .placement_index
            .get("active-model")
            .is_some_and(|placements| !placements.is_empty()));
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
    // Event-driven routing tests (Type 1: KV index overlap scoring)
    // -----------------------------------------------------------------------

    /// Helper: create a positional KV index and store blocks for a worker.
    /// `token_chunks` is a list of token-id slices — each becomes one block.
    fn setup_indexer_with_blocks(
        worker_url: &str,
        token_chunks: &[&[u32]],
        jump_size: usize,
    ) -> Arc<KvIndex> {
        let indexer = Arc::new(KvIndex::positional(jump_size));
        let worker_id = indexer.intern_worker(worker_url).unwrap();
        let mut wb = WorkerBlocks::default();
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

    // -- overlap affinity-group unit tests --

    #[test]
    fn test_overlap_affinity_group_selects_best_match() {
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
        let result = overlap_affinity_group(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            &[0, 1],
            &indexer,
            4,
            &default_tuning(),
        );
        assert_eq!(result, vec![0]); // w1
    }

    #[test]
    fn test_equal_overlap_forms_one_affinity_group() {
        // Equal overlap remains one affinity group so LeastLoad, rather than
        // raw load or slice order, owns the final tie-break.
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
        let mut wb2 = WorkerBlocks::default();
        let blocks: Vec<StoredBlock> = chunks
            .iter()
            .enumerate()
            .map(|(i, tokens)| StoredBlock {
                seq_hash: SequenceHash(100 + i as u64),
                content_hash: compute_content_hash(tokens),
            })
            .collect();
        indexer.apply_stored(w2, &blocks, None, &mut wb2).unwrap();

        assert_eq!(
            overlap_affinity_group(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &default_tuning(),
            ),
            vec![0, 1]
        );
    }

    /// Two workers with identical cached blocks (the tie-test topology): both
    /// fully match the request.
    fn equal_overlap_fixture() -> (Vec<Arc<dyn Worker>>, Arc<KvIndex>) {
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
        let mut wb2 = WorkerBlocks::default();
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
        let waiting = LoadSnapshot::from_loads_for_test(vec![
            ("http://w1:8000".to_string(), waiting_load(0)),
            ("http://w2:8000".to_string(), waiting_load(8)),
        ]);
        let tuning = OverlapTuning {
            overlap_decay: 4.0,
            selection_temperature: 0.0,
            waiting_prefill_tokens: Some(&waiting),
        };
        assert_eq!(
            overlap_affinity_group(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &tuning,
            ),
            vec![0],
            "backlogged worker must lose its affinity edge"
        );
    }

    #[test]
    fn test_overlap_decay_missing_load_data_never_decays() {
        // Only w1 reports load (and holds the floor at zero backlog); w2 has
        // no entry. Neither may be decayed, so the equal-score tie — and its
        // random spreading — must survive.
        let (workers, indexer) = equal_overlap_fixture();
        let waiting = LoadSnapshot::from_loads_for_test(vec![(
            "http://w1:8000".to_string(),
            waiting_load(0),
        )]);
        let tuning = OverlapTuning {
            overlap_decay: 4.0,
            selection_temperature: 0.0,
            waiting_prefill_tokens: Some(&waiting),
        };
        assert_eq!(
            overlap_affinity_group(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &tuning,
            ),
            vec![0, 1],
            "workers without load data must not be decayed"
        );
    }

    /// w1 caches both request blocks (score 2), w2 only the first (score 1).
    fn unequal_overlap_fixture() -> (Vec<Arc<dyn Worker>>, Arc<KvIndex>) {
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
        let mut wb2 = WorkerBlocks::default();
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
            let group = overlap_affinity_group(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &default_tuning(),
            );
            assert_eq!(group, vec![0], "temperature 0 must be exact argmax");
        }

        let tuning = OverlapTuning {
            overlap_decay: 0.0,
            selection_temperature: 1.0,
            waiting_prefill_tokens: None,
        };
        let mut counts = [0usize; 2];
        for _ in 0..300 {
            let group = overlap_affinity_group(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &tuning,
            );
            assert_eq!(group.len(), 1, "unequal scores select one score band");
            counts[group[0]] += 1;
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
    fn test_selection_temperature_preserves_equal_score_group() {
        // A temperature draw chooses an affinity score band. Equal-score
        // workers stay together so LeastLoad can resolve the tie.
        let (workers, indexer) = equal_overlap_fixture();
        let tuning = OverlapTuning {
            overlap_decay: 0.0,
            selection_temperature: 0.5,
            waiting_prefill_tokens: None,
        };
        assert_eq!(
            overlap_affinity_group(
                &workers,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[0, 1],
                &indexer,
                4,
                &tuning,
            ),
            vec![0, 1]
        );
    }

    #[test]
    fn test_overlap_affinity_group_no_match_is_empty() {
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
        let result = overlap_affinity_group(
            &workers,
            &[100, 200, 300, 400, 500, 600, 700, 800],
            &[0],
            &indexer,
            4,
            &default_tuning(),
        );
        assert!(result.is_empty());
    }

    #[test]
    fn test_raw_load_does_not_break_equal_affinity() {
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
        let indexer = Arc::new(KvIndex::positional(4));
        let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
        let w2_id = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlocks::default();
        let mut wb2 = WorkerBlocks::default();
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

        // Equal overlap remains one group despite different raw loads;
        // production delegates the final tie-break to LeastLoad.
        let result = overlap_affinity_group(
            &workers,
            &[1, 2, 3, 4],
            &[0, 1],
            &indexer,
            4,
            &default_tuning(),
        );
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn test_tree_size_does_not_break_equal_affinity() {
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

        let indexer = Arc::new(KvIndex::positional(4));
        let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
        let w2_id = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlocks::default();
        let mut wb2 = WorkerBlocks::default();

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

        // Equal overlap, equal load, different tree sizes: both stay in the
        // affinity group for LeastLoad to resolve.
        assert_eq!(
            overlap_affinity_group(
                &workers,
                &[1, 2, 3, 4],
                &[0, 1],
                &indexer,
                4,
                &default_tuning(),
            ),
            vec![0, 1]
        );
    }

    #[test]
    fn test_overlap_affinity_group_short_request_is_empty() {
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .health_config(no_health_check())
                .build(),
        )];

        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);

        // Request shorter than block_size → no full blocks → None
        let result =
            overlap_affinity_group(&workers, &[1, 2, 3], &[0], &indexer, 4, &default_tuning());
        assert!(result.is_empty());
    }

    #[test]
    fn test_overlap_affinity_group_prefers_deeper_partial_match() {
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

        let indexer = Arc::new(KvIndex::positional(4));
        let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
        let w2_id = indexer.intern_worker("http://w2:8000").unwrap();
        let mut wb1 = WorkerBlocks::default();
        let mut wb2 = WorkerBlocks::default();

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
        let result = overlap_affinity_group(
            &workers,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            &[0, 1],
            &indexer,
            4,
            &default_tuning(),
        );
        assert_eq!(result, vec![0]); // w1 (higher overlap)
    }

    // -- select_worker_event_driven integration tests --

    #[test]
    fn event_unique_hit_keeps_affinity_and_credits_expected_wait() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        workers[1].increment_load();
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer =
            setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        let cached = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            policy.select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&cached),
                    ..Default::default()
                }
            ),
            Some(0)
        );
        assert_only_final_worker_credited(&policy, &workers, 0, 8);
    }

    #[test]
    fn a_worker_with_an_emptied_index_gets_the_warm_up_slice_whatever_its_age() {
        // The soaks' case in miniature: w1 holds a fleet-sized cache (1,100
        // blocks), w2's index was cleared by a resync (nothing indexed), and
        // every request shares a head with w1, so affinity alone would never
        // send w2 a request again. The slice does, on the thin overlap: w2 is
        // thin against the fleet's level whatever its age.
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        update_expected_wait_loads(&policy, &workers, &[0, 0]);
        let held: Vec<u32> = (1..=4_400).collect();
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&held], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));
        // w1's first block (the shared head) followed by eleven novel blocks.
        let mut request: Vec<u32> = (1..=4).collect();
        request.extend(90_000..90_044);
        let picks: Vec<usize> = (0..8)
            .map(|_| {
                policy
                    .select_worker(
                        &workers,
                        &SelectWorkerInfo {
                            tokens: Some(&request),
                            ..Default::default()
                        },
                    )
                    .unwrap()
            })
            .collect();
        assert!(
            picks.contains(&1),
            "the emptied worker got its slice of the thin-overlap requests: {picks:?}"
        );
        assert!(picks.contains(&0), "the holder kept the rest: {picks:?}");
    }

    /// Store `tokens` for `worker` as blocks of `block` tokens, with sequence
    /// hashes from `seq_base` (distinct per call).
    fn store_blocks(indexer: &KvIndex, worker: u32, tokens: &[u32], block: usize, seq_base: u64) {
        let mut wb = WorkerBlocks::default();
        let blocks: Vec<StoredBlock> = tokens
            .chunks(block)
            .enumerate()
            .map(|(i, chunk)| StoredBlock {
                seq_hash: SequenceHash(seq_base + i as u64),
                content_hash: compute_content_hash(chunk),
            })
            .collect();
        indexer
            .apply_stored(worker, &blocks, None, &mut wb)
            .unwrap();
    }

    /// A fleet for the diversion tests: the policy, its eight workers, the
    /// index they share and each holder's head (its first 240 tokens, 60
    /// blocks); a request made of a head and a four-block tail of its own is
    /// a deep hit on that holder (60 of 64 blocks) that no other request
    /// repeats.
    struct HolderFleet {
        policy: CacheAwarePolicy,
        workers: Vec<Arc<dyn Worker>>,
        indexer: Arc<KvIndex>,
        heads: Vec<Vec<u32>>,
    }

    /// Eight workers; the first `holders` hold 1,100 blocks each of their own
    /// 4,400-token sequence, the rest nothing.
    fn fleet_with_holders(holders: usize) -> HolderFleet {
        fleet_with_holders_of(holders, 1_100)
    }

    /// Eight workers; the first `holders` hold `blocks` blocks each of their
    /// own sequence, the rest nothing.
    fn fleet_with_holders_of(holders: usize, blocks: usize) -> HolderFleet {
        let policy = CacheAwarePolicy::with_config(test_config());
        let urls: Vec<String> = (0..8).map(|i| format!("http://w{i}:8000")).collect();
        let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
        let workers = make_workers(&refs);
        policy.init_workers(&workers);
        update_expected_wait_loads(&policy, &workers, &[0, 0, 0, 0, 0, 0, 0, 0]);
        let indexer = Arc::new(KvIndex::positional(4));
        let mut prefixes = Vec::new();
        for (h, url) in urls.iter().enumerate() {
            let id = indexer.intern_worker(url).unwrap();
            if h >= holders {
                continue;
            }
            let tokens: Vec<u32> = (0..blocks as u32 * 4)
                .map(|i| (h as u32 + 1) * 100_000 + i)
                .collect();
            store_blocks(&indexer, id, &tokens, 4, (h as u64 + 1) * 1_000_000);
            prefixes.push(tokens[..240].to_vec());
        }
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        monitor
            .indexers
            .insert("unknown".to_string(), Arc::clone(&indexer));
        policy.set_kv_event_monitor(Some(monitor));
        HolderFleet {
            policy,
            workers,
            indexer,
            heads: prefixes,
        }
    }

    #[test]
    fn an_emptied_worker_gets_its_share_of_hits_until_it_refills() {
        // The churn runs' case: every request is cached whole on one of seven
        // holders (a deep hit, never a miss), the eighth worker's index was
        // emptied by a resync. Affinity alone never sends it anything; the
        // diversion hands it one hit in eight (no shallow hit ever comes, so
        // the fallback at the window's end), and each diverted request lands
        // in its index, until it crosses half the fleet's level (1,100 -> 550).
        let HolderFleet {
            policy,
            workers,
            indexer,
            heads,
        } = fleet_with_holders(7);
        let emptied = indexer.worker_id("http://w7:8000").unwrap();
        // A request: a holder's 60-block head and a four-block tail of its own.
        let request = |i: usize| -> Vec<u32> {
            let mut tokens = heads[i % 7].clone();
            tokens.extend((0..16).map(|t| 90_000_000 + i as u32 * 16 + t));
            tokens
        };
        let mut decisions = 0usize;
        let mut first = None;
        let mut received = 0usize;
        while decisions < 1_000 && indexer.worker_block_count(emptied) < 550 {
            let tokens = request(decisions);
            let idx = policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(&tokens),
                        ..Default::default()
                    },
                )
                .unwrap();
            decisions += 1;
            if idx == 7 {
                received += 1;
                first.get_or_insert(decisions);
                // The request prefills on w7: its blocks join w7's index.
                store_blocks(
                    &indexer,
                    emptied,
                    &tokens,
                    4,
                    9_000_000 + decisions as u64 * 100,
                );
            }
        }
        assert!(
            first.is_some_and(|first| first <= 8),
            "the emptied worker is served within the first eight hits, was {first:?}"
        );
        assert!(
            indexer.worker_block_count(emptied) >= 550,
            "refilled to the ratio: {} blocks after {decisions} decisions",
            indexer.worker_block_count(emptied)
        );
        assert!(
            decisions <= 40 * 8 + 16,
            "about forty requests of 64 blocks (seven heads of 60, then tails of 4) at one in eight: \
             {decisions} decisions, {received} received"
        );
        // At the ratio the diversion stops: the hit counter runs on without a
        // reset (w7 now holds the heads too and may win a tie as a holder,
        // which is affinity, not a diversion).
        let hits_before = policy.divert_hits.load(Ordering::Relaxed);
        for i in 0..32 {
            let tokens = request(decisions + i);
            policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(&tokens),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        assert_eq!(
            policy.divert_hits.load(Ordering::Relaxed),
            hits_before + 32,
            "no diversion once the worker is no longer thin"
        );
    }

    #[test]
    fn a_thin_worker_keeps_its_share_past_the_warm_up_blocks_until_the_ratio() {
        // Churn c3 (6ce84c9e): the fleet's level was 32,767 blocks a worker,
        // far above the warm-up blocks. Within a minute of the publisher
        // restart the emptied worker's index had regrown to 1,765 from the
        // decode blocks of its requests in flight, and the pool table,
        // applying the age rule's growth cap to a thin worker, dropped it
        // there after one diversion: idle for the 25 minutes to the next
        // fault. Here the level is 4,096 (the ratio at 2,048), the worker has
        // regrown to 1,100 before any diversion, every request is a deep hit
        // (60 of 64 blocks, held elsewhere), each diverted request stays in
        // flight for the rest of the test, and the table is rebuilt every
        // eight decisions as the second's refresh would.
        let HolderFleet {
            policy,
            workers,
            indexer,
            heads,
        } = fleet_with_holders_of(7, 4_096);
        let thin = indexer.worker_id("http://w7:8000").unwrap();
        // The growth baselines date from the first table, built on the empty
        // fleet as on the run; the regrowth comes after.
        policy.pool_table(&workers, &indexer, liveness::now_ms());
        let regrown: Vec<u32> = (0..4_400).map(|i| 80_000_000 + i).collect();
        store_blocks(&indexer, thin, &regrown, 4, 8_000_000);
        assert_eq!(indexer.worker_block_count(thin), 1_100);
        let request = |i: usize| -> Vec<u32> {
            let mut tokens = heads[i % 7].clone();
            tokens.extend((0..16).map(|t| 90_000_000 + i as u32 * 16 + t));
            tokens
        };
        let select = |tokens: &[u32]| -> usize {
            policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(tokens),
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        let mut rebuilds = 1u64;
        let mut decisions = 0usize;
        let mut first = None;
        let mut received = 0usize;
        while decisions < 2_000 && indexer.worker_block_count(thin) < 2_048 {
            if decisions.is_multiple_of(8) {
                rebuilds += 1;
                policy.pool_table(
                    &workers,
                    &indexer,
                    liveness::now_ms() + rebuilds * POOL_TABLE_REFRESH_MS,
                );
            }
            let tokens = request(decisions);
            let idx = select(&tokens);
            decisions += 1;
            if idx != 7 {
                continue;
            }
            received += 1;
            store_blocks(
                &indexer,
                thin,
                &tokens,
                4,
                9_000_000 + decisions as u64 * 100,
            );
            // The diverted request runs on (tens of seconds on the trace).
            workers[7].increment_load();
            if first.is_none() {
                first = Some(decisions);
                // Inside the window, with that request in flight, no second
                // diversion: the hit counter runs on without a reset.
                let hits = policy.divert_hits.load(Ordering::Relaxed);
                for i in 0..16 {
                    select(&request(10_000 + i));
                }
                assert_eq!(
                    policy.divert_hits.load(Ordering::Relaxed),
                    hits + 16,
                    "no second diversion inside the window while the first is in flight"
                );
            }
            // The window elapses; the request is still in flight.
            workers[7].note_diverted(0);
        }
        assert!(
            first.is_some_and(|first| first <= 8),
            "the thin worker is served within the first eight hits although it \
             regrew past the warm-up blocks, was {first:?}"
        );
        assert!(
            indexer.worker_block_count(thin) >= 2_048,
            "refilled to the ratio across the rebuilds: {} blocks after {decisions} decisions",
            indexer.worker_block_count(thin)
        );
        assert!(
            received >= 15 && decisions <= 140 * 8 + 32,
            "one hit in eight all the way (seven heads of 60 blocks, then tails of 4): \
             {decisions} decisions, {received} received, {} in flight",
            workers[7].load()
        );
        // At the ratio the table rebuilt lists no thin worker, and with no
        // thin worker the diversion costs nothing: the hits are not even
        // counted.
        let table = policy.pool_table(
            &workers,
            &indexer,
            liveness::now_ms() + (rebuilds + 1) * POOL_TABLE_REFRESH_MS,
        );
        assert!(table.thin.is_empty(), "no longer thin: {:?}", table.thin);
        let hits = policy.divert_hits.load(Ordering::Relaxed);
        for i in 0..32 {
            select(&request(20_000 + i));
        }
        assert_eq!(
            policy.divert_hits.load(Ordering::Relaxed),
            hits,
            "no hit counted, let alone diverted, once no worker is thin"
        );
    }

    #[test]
    fn a_fleet_without_a_thin_worker_sees_no_diversion() {
        let fleet = fleet_with_holders(8);
        for i in 0..64 {
            let holder = i % 8;
            let idx = fleet
                .policy
                .select_worker(
                    &fleet.workers,
                    &SelectWorkerInfo {
                        tokens: Some(&fleet.heads[holder]),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(
                idx, holder,
                "every hit stays with its holder (decision {i})"
            );
        }
    }

    #[test]
    fn event_no_overlap_uses_expected_wait() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        let novel = [100, 200, 300, 400];
        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&novel),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(selected, 1, "expected wait must beat raw request count");
        assert_only_final_worker_credited(&policy, &workers, 1, 4);
    }

    #[test]
    fn event_spill_credits_only_expected_wait_final_worker() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        policy.init_workers(&workers);
        for _ in 0..100 {
            workers[0].increment_load();
        }
        for _ in 0..10 {
            workers[2].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[20_000, 10_000, 0]);

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        let tokens = [1, 2, 3, 4];
        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_only_final_worker_credited(&policy, &workers, 2, 4);
        assert_eq!(selected, 2, "spill must use expected wait over the fleet");
    }

    #[test]
    fn event_equal_overlap_tie_uses_expected_wait_only_among_holders() {
        let policy = CacheAwarePolicy::with_config(test_config());
        let (mut workers, indexer) = equal_overlap_fixture();
        workers.push(make_workers(&["http://w3:8000"]).pop().unwrap());
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[10_000, 0, 0]);

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(selected, 1);
        assert_only_final_worker_credited(&policy, &workers, 1, 8);
    }

    #[test]
    fn event_decay_keeps_strict_affinity_ahead_of_expected_wait() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            overlap_decay: 4.0,
            ..test_config()
        });
        let (workers, indexer) = equal_overlap_fixture();
        policy.init_workers(&workers);

        // The same coherent snapshot makes LeastLoad prefer w2 (fast, empty KV)
        // while its slightly larger waiting-prefill backlog decays w2 into a
        // strictly lower affinity score band. Affinity must remain primary.
        let overlap_loads = HashMap::from([
            (
                workers[0].url().to_string(),
                expected_wait_load(0, 0.9, 100.0),
            ),
            (
                workers[1].url().to_string(),
                expected_wait_load(8, 0.0, 1_000.0),
            ),
        ]);
        policy.update_loads(&overlap_loads);
        let (_tx, rx) = watch::channel(complete_load_snapshot(overlap_loads));
        policy.set_load_receiver(Some(rx));

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(selected, 0, "strict decayed affinity must remain primary");
        assert_only_final_worker_credited(&policy, &workers, 0, 8);
    }

    #[test]
    fn event_temperature_keeps_equal_score_group_for_expected_wait() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            selection_temperature: 0.5,
            ..test_config()
        });
        let (workers, indexer) = equal_overlap_fixture();
        policy.init_workers(&workers);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(selected, 1, "LeastLoad must break the sampled-score tie");
        assert_only_final_worker_credited(&policy, &workers, 1, 8);
    }

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
    fn test_event_driven_no_overlap_uses_dark_fleet_expected_wait() {
        let policy = CacheAwarePolicy::with_config(test_config());

        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        // Give w1 higher live load so dark-fleet expected wait picks w2.
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

        // No overlap → event-driven falls back to expected wait (not token tree).
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[100, 200, 300, 400]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1); // w2 (dark-fleet expected wait), not token tree
    }

    #[test]
    fn test_event_driven_gated_hot_winner_spills_to_expected_wait() {
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
        assert_eq!(idx, 1, "gated overlap winner must spill to expected wait");
    }

    #[test]
    fn test_event_driven_short_request_uses_dark_fleet_expected_wait() {
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

        // Request shorter than block_size → no full blocks → expected-wait fallback.
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1); // w2 (dark-fleet expected wait)
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
        let indexer = Arc::new(KvIndex::positional(4));
        let w1_id = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlocks::default();
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

        // With imbalance, select_worker should pick expected wait (w2), not event-driven.
        let idx = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4]),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(idx, 1); // w2 (expected wait), regardless of event data
    }

    #[test]
    fn test_empty_indexer_falls_through_to_token_tree() {
        // When the monitor has an indexer for a model but the indexer is empty
        // (startup, reconnect), routing should fall through to the token tree
        // instead of taking the event-driven path and landing on a fallback.
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
        let empty_indexer = Arc::new(KvIndex::positional(4));
        monitor
            .indexers
            .insert("unknown".to_string(), empty_indexer);
        policy.set_kv_event_monitor(Some(monitor));

        // Empty indexer → has_event_indexer returns false → falls through to token tree
        assert!(!policy.has_event_indexer("unknown"));

        // Tokens must fill at least one tree page (the policy's block_size)
        // to populate the tree; shorter sequences are uncacheable and fall
        // through to expected wait.
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
    fn hash_unique_hit_keeps_affinity_and_credits_expected_wait() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let model = normalize_model_key(workers[0].model_id());
        let cached: Vec<u32> = (0..16).collect();
        policy.record_placement(
            model,
            None,
            &cached,
            &[16],
            workers[0].url(),
            Instant::now(),
        );
        workers[1].increment_load();
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);

        assert_eq!(route_tokens(&policy, &workers, &cached), 0);
        assert_only_final_worker_credited(&policy, &workers, 0, 16);
    }

    #[test]
    fn hash_miss_uses_expected_wait_and_records_final_holder() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);
        let tokens: Vec<u32> = (0..16).collect();

        let selected = route_tokens(&policy, &workers, &tokens);
        assert_eq!(selected, 1, "expected wait must beat raw request count");
        assert_only_final_worker_credited(&policy, &workers, 1, 16);

        let model = normalize_model_key(workers[0].model_id());
        let placements = policy.placement_index.get(model).unwrap();
        let holders = placements
            .get(&(16, hash_token_head(None, &tokens)))
            .expect("final dispatch must be recorded");
        assert_eq!(holders.len(), 1);
        assert_eq!(holders[0].worker_url, workers[1].url());
    }

    #[test]
    fn hash_spill_uses_expected_wait_and_records_only_final_holder() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let model = normalize_model_key(workers[0].model_id());
        let tokens: Vec<u32> = (0..16).collect();
        policy.record_placement(
            model,
            None,
            &tokens,
            &[16],
            workers[0].url(),
            Instant::now(),
        );
        for _ in 0..100 {
            workers[0].increment_load();
        }
        for _ in 0..10 {
            workers[2].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[20_000, 10_000, 0]);

        let selected = route_tokens(&policy, &workers, &tokens);
        assert_eq!(selected, 2, "spill must use expected wait over the fleet");
        assert_only_final_worker_credited(&policy, &workers, 2, 16);

        let placements = policy.placement_index.get(model).unwrap();
        let holders = placements
            .get(&(16, hash_token_head(None, &tokens)))
            .unwrap();
        let holder_urls: HashSet<&str> = holders
            .iter()
            .map(|holder| holder.worker_url.as_str())
            .collect();
        assert_eq!(
            holder_urls,
            HashSet::from([workers[0].url(), workers[2].url()])
        );
    }

    #[test]
    fn hash_short_request_uses_expected_wait_without_recording() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[10_000, 0]);
        let tokens: Vec<u32> = (0..8).collect();

        let selected = route_tokens(&policy, &workers, &tokens);
        assert_eq!(selected, 1, "short fallback must use expected wait");
        assert_only_final_worker_credited(&policy, &workers, 1, 8);
        assert!(policy.placement_index.is_empty());
    }

    #[test]
    fn hash_multiple_holders_use_expected_wait_only_within_affinity_set() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let model = normalize_model_key(workers[0].model_id());
        let tokens: Vec<u32> = (0..16).collect();
        let now = Instant::now();
        policy.record_placement(model, None, &tokens, &[16], workers[0].url(), now);
        policy.record_placement(model, None, &tokens, &[16], workers[1].url(), now);
        for _ in 0..5 {
            workers[1].increment_load();
        }
        update_expected_wait_loads(&policy, &workers, &[10_000, 0, 0]);

        let selected = route_tokens(&policy, &workers, &tokens);
        assert_eq!(selected, 1, "non-holder w3 must not break cache affinity");
        assert_only_final_worker_credited(&policy, &workers, 1, 16);
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
        policy.record_placement(model, None, &tokens, &[16], "http://w2:8000", now);
        policy.record_placement(model, None, &tokens, &[32], "http://w1:8000", now);
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
            let key = (boundary, hash_token_head(None, &tokens[..boundary]));
            let holders = placements.get(&key).unwrap();
            assert_eq!(holders.len(), 1);
            assert_eq!(holders[0].worker_url, workers[selected].url());
        }
    }

    #[test]
    fn hash_mode_short_request_stays_dark_fleet_expected_wait() {
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
    fn hash_mode_untokenized_text_stays_dark_fleet_expected_wait() {
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
        let key = (16usize, hash_token_head(None, &tokens[..16]));

        policy.record_placement(model, None, &tokens, &[16], "http://w1:8000", t0);

        let url_to_idx = CacheAwarePolicy::healthy_url_index(&workers, &[0, 1]);
        // Just inside the 180s default TTL: live.
        let live_at = t0 + Duration::from_secs(179);
        assert_eq!(
            policy.live_holder_candidates(&url_to_idx, model, key, live_at),
            vec![0]
        );
        // Just past it: expired and pruned.
        let expired_at = t0 + Duration::from_secs(181);
        assert_eq!(
            policy.live_holder_candidates(&url_to_idx, model, key, expired_at),
            Vec::<usize>::new()
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
    fn hash_mode_expired_holder_falls_back_to_expected_wait() {
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
        let key = (16usize, hash_token_head(None, &tokens[..16]));

        for (i, url) in [
            "http://w1:8000",
            "http://w2:8000",
            "http://w3:8000",
            "http://w4:8000",
        ]
        .iter()
        .enumerate()
        {
            policy.record_placement(
                "m",
                None,
                &tokens,
                &[16],
                url,
                t0 + Duration::from_secs(i as u64),
            );
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
        let key = (16usize, hash_token_head(None, &tokens[..16]));

        policy.record_placement(
            model,
            None,
            &tokens,
            &[16],
            "http://w1:8000",
            Instant::now(),
        );
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

        policy.record_placement(
            model,
            None,
            &tokens,
            &[16],
            "http://w1:8000",
            Instant::now(),
        );
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
        policy.record_placement("m", None, &tokens, &[16], "http://w1:8000", now);
        policy.record_placement("m", None, &tokens, &[16], "http://w2:8000", now);

        policy.remove_worker_by_url("http://w1:8000");

        let placements = policy.placement_index.get("m").unwrap();
        let holders = placements
            .get(&(16usize, hash_token_head(None, &tokens[..16])))
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
        policy.record_placement("m", None, &stale, &[16], "http://w1:8000", t0);
        policy.record_placement(
            "m",
            None,
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
            .get(&(16usize, hash_token_head(None, &fresh[..16])))
            .is_some());
    }

    #[test]
    fn live_holder_resolution_skips_dead_holders_and_unknown_keys() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[16]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        let tokens: Vec<u32> = (0..16).collect();
        let now = Instant::now();
        let key = (16usize, hash_token_head(None, &tokens[..16]));
        policy.record_placement("m", None, &tokens, &[16], "http://w1:8000", now);
        policy.record_placement("m", None, &tokens, &[16], "http://w2:8000", now);

        // w1 unhealthy: its holder entry must not resolve.
        let url_to_idx = CacheAwarePolicy::healthy_url_index(&workers, &[1]);
        assert_eq!(
            policy.live_holder_candidates(&url_to_idx, "m", key, now),
            vec![1]
        );
        // No healthy workers: no candidate.
        let empty = CacheAwarePolicy::healthy_url_index(&workers, &[]);
        assert_eq!(
            policy.live_holder_candidates(&empty, "m", key, now),
            Vec::<usize>::new()
        );
        // Unknown key: no candidate.
        assert_eq!(
            policy.live_holder_candidates(&url_to_idx, "m", (32, 7), now),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn healthy_url_index_preserves_duplicate_equal_affinity_candidates() {
        let workers = make_workers(&["http://w1:8000", "http://w1:8000", "http://w2:8000"]);
        workers[0].increment_load();

        let url_to_idx = CacheAwarePolicy::healthy_url_index(&workers, &[0, 1, 2]);
        assert_eq!(url_to_idx.len(), 2);
        assert_eq!(url_to_idx["http://w1:8000"], vec![0, 1]);
        assert_eq!(url_to_idx["http://w2:8000"], vec![2]);

        // Equal loads stay tied so LeastLoad can resolve them atomically.
        let tied = make_workers(&["http://w1:8000", "http://w1:8000"]);
        assert_eq!(
            CacheAwarePolicy::healthy_url_index(&tied, &[0, 1])["http://w1:8000"],
            vec![0, 1]
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
                        policy.record_placement("m", None, &tokens, &[16], "http://w1:8000", now);
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

    // ---- cache namespaces (cache_salt / extra_key / LoRA partitioning) ----

    fn salted(salt: &str) -> Option<CacheNamespace> {
        CacheNamespace::derive(&openai_protocol::common::CachePartition {
            cache_salt: Some(salt),
            extra_key: None,
            lora_path: None,
        })
    }

    fn route_namespaced(
        policy: &CacheAwarePolicy,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        cache_namespace: Option<CacheNamespace>,
    ) -> usize {
        policy
            .select_worker(
                workers,
                &SelectWorkerInfo {
                    tokens: Some(tokens),
                    cache_namespace,
                    ..Default::default()
                },
            )
            .unwrap()
    }

    #[test]
    fn head_hash_is_unchanged_without_a_namespace_and_keys_like_the_tree_with_one() {
        let head = [5u32, 6, 7, 8];
        assert_eq!(
            hash_token_head(None, &head),
            xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(&head))
        );
        let tenant_a = salted("tenant-a").unwrap();
        let tenant_b = salted("tenant-b").unwrap();
        // A namespaced head hashes marker ‖ head (hash mode has no pages).
        let mut keyed = tenant_a.token_marker().to_vec();
        keyed.extend_from_slice(&head);
        assert_eq!(
            hash_token_head(Some(tenant_a), &head),
            xxhash_rust::xxh3::xxh3_64(bytemuck::cast_slice(&keyed))
        );
        assert_ne!(
            hash_token_head(Some(tenant_a), &head),
            hash_token_head(None, &head)
        );
        assert_ne!(
            hash_token_head(Some(tenant_a), &head),
            hash_token_head(Some(tenant_b), &head)
        );
    }

    #[test]
    fn hash_mode_keys_placements_under_the_cache_namespace() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[4]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        let tokens = [1u32, 2, 3, 4, 5];
        let tenant_a = salted("tenant-a");
        let tenant_b = salted("tenant-b");

        // w1 is busier: tenant A's first request misses onto w2 and is
        // recorded there under its namespace.
        workers[0].increment_load();
        assert_eq!(route_namespaced(&policy, &workers, &tokens, tenant_a), 1);
        // A hash hit keeps tenant A on w2 once w2 is the busier worker...
        workers[1].increment_load();
        workers[1].increment_load();
        assert_eq!(route_namespaced(&policy, &workers, &tokens, tenant_a), 1);
        // ...while tenant B and an unpartitioned request key differently,
        // miss, and take the least-loaded w1.
        assert_eq!(route_namespaced(&policy, &workers, &tokens, tenant_b), 0);
        assert_eq!(route_namespaced(&policy, &workers, &tokens, None), 0);
    }

    #[test]
    fn imbalance_fallback_inserts_under_the_cache_namespace() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        // One full page (the tree is page-aligned at the default block size).
        let prompt: Vec<u32> = (100..116).collect();
        let tenant_a = salted("tenant-a").unwrap();
        let model_id = normalize_model_key(workers[0].model_id());
        // Materialize the model's token tree so the fallback path has one to
        // update.
        let seed: Vec<u32> = (1..17).collect();
        route_namespaced(&policy, &workers, &seed, None);

        let info = SelectWorkerInfo {
            tokens: Some(&prompt),
            cache_namespace: Some(tenant_a),
            ..Default::default()
        };
        policy
            .select_worker_fallback(&workers, &info, &[0, 1], model_id)
            .unwrap();

        let tree = policy.token_trees.get(model_id).unwrap().value().clone();
        // The unpartitioned prompt cannot see the namespaced insert...
        assert!(tree.prefix_match_legacy(&prompt).0.is_empty());
        // ...while the namespaced key matches in full.
        let keyed = tenant_a.prefixed_tokens(&prompt, policy.config.block_size);
        assert_eq!(tree.prefix_match_legacy(&keyed).0.len(), keyed.len());
    }

    #[test]
    fn an_unpartitioned_request_cannot_forge_its_way_into_a_namespace() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            cache_threshold: 0.5,
            ..Default::default()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        let tenant_a = salted("tenant-a").unwrap();
        let prompt: Vec<u32> = (100..132).collect();

        // Tenant A's prompt lands on w2 (w1 is busier).
        workers[0].increment_load();
        assert_eq!(
            route_namespaced(&policy, &workers, &prompt, Some(tenant_a)),
            1
        );
        workers[1].increment_load();
        workers[1].increment_load();

        // An unpartitioned request whose token ids spell tenant A's marker
        // must not hit tenant A's entry: it misses and takes the least-loaded
        // w1. Same for text that spells the marker on the string tree.
        let forged = tenant_a.prefixed_tokens(&prompt, policy.config.block_size);
        assert_eq!(route_namespaced(&policy, &workers, &forged, None), 0);

        let text_policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            eviction_interval_secs: 0,
            cache_threshold: 0.5,
            ..Default::default()
        });
        let text_workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        text_policy.init_workers(&text_workers);
        text_workers[0].increment_load();
        let text_info = SelectWorkerInfo {
            request_text: Some("shared system prompt"),
            cache_namespace: Some(tenant_a),
            ..Default::default()
        };
        assert_eq!(
            text_policy.select_worker(&text_workers, &text_info),
            Some(1)
        );
        text_workers[1].increment_load();
        text_workers[1].increment_load();
        let forged_text = tenant_a.prefixed_text("shared system prompt");
        let forged_info = SelectWorkerInfo {
            request_text: Some(&forged_text),
            ..Default::default()
        };
        assert_eq!(
            text_policy.select_worker(&text_workers, &forged_info),
            Some(0)
        );
    }

    #[test]
    fn hash_mode_strips_marker_material_from_unpartitioned_heads() {
        let policy = CacheAwarePolicy::with_config(hash_config(&[4]));
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        let tokens = [1u32, 2, 3, 4, 5];
        let tenant_a = salted("tenant-a").unwrap();

        // Tenant A's placement lands on w2 (w1 is busier).
        workers[0].increment_load();
        assert_eq!(
            route_namespaced(&policy, &workers, &tokens, Some(tenant_a)),
            1
        );
        workers[1].increment_load();
        workers[1].increment_load();

        // An unpartitioned head that spells tenant A's marker keys as the
        // bare prompt: no holder, so it takes the least-loaded w1.
        let mut forged = tenant_a.token_marker().to_vec();
        forged.extend_from_slice(&tokens);
        assert_eq!(route_namespaced(&policy, &workers, &forged, None), 0);
    }

    // -- selection policy layer --

    /// Random positive-overlap candidate sets, as `overlap_candidates` would
    /// produce them (healthy order, scores from a small set so ties occur).
    fn random_overlap_candidates(rng: &mut impl RngExt, workers: usize) -> Vec<OverlapCandidate> {
        let mut candidates = Vec::new();
        for idx in 0..workers {
            if rng.random::<f64>() >= 0.6 {
                continue;
            }
            let score = f64::from(rng.random_range(1..6u32)) * 1.5;
            let decayed = if rng.random::<f64>() < 0.3 {
                score / 1.5
            } else {
                score
            };
            candidates.push(OverlapCandidate {
                idx,
                raw_score: score,
                effective_score: decayed,
            });
        }
        candidates
    }

    fn policy_inputs<'a>(
        candidates: &'a [OverlapCandidate],
        urls: &'a [String],
    ) -> Vec<CandidateInputs<'a>> {
        candidates
            .iter()
            .map(|candidate| CandidateInputs {
                idx: candidate.idx,
                url: &urls[candidate.idx],
                device_blocks: candidate.raw_score,
                effective_score: candidate.effective_score,
            })
            .collect()
    }

    #[test]
    fn default_policy_reproduces_affinity_score_group_at_zero_temperature() {
        let policy = cost::build(cost::DEFAULT_POLICY, 0.0).unwrap();
        let urls: Vec<String> = (0..16).map(|i| format!("http://w{i:02}:8000")).collect();
        let request = RequestInputs {
            prompt_tokens: 64,
            block_size: 4,
            request_blocks: 16,
            avg_load: 0.0,
            prefix_hashes: None,
        };
        let mut rng = rand::rng();
        for _ in 0..2_000 {
            let candidates = random_overlap_candidates(&mut rng, urls.len());
            let mut reference = CacheAwarePolicy::affinity_score_group(&candidates, 0.0);
            reference.sort_unstable();
            let inputs = policy_inputs(&candidates, &urls);
            let mut group = match policy.select(&request, &inputs) {
                Pick::Group(rows) => rows.iter().map(|&row| inputs[row].idx).collect::<Vec<_>>(),
                Pick::None => Vec::new(),
                Pick::Final(_) => panic!("the default policy never returns a single pick"),
            };
            group.sort_unstable();
            assert_eq!(group, reference);
        }
    }

    #[test]
    fn default_policy_temperature_groups_are_score_groups() {
        let policy = cost::build(cost::DEFAULT_POLICY, 0.5).unwrap();
        let urls: Vec<String> = (0..8).map(|i| format!("http://w{i}:8000")).collect();
        let candidates: Vec<OverlapCandidate> = (0..8)
            .map(|idx| OverlapCandidate {
                idx,
                raw_score: if idx < 4 { 8.0 } else { 2.0 },
                effective_score: if idx < 4 { 8.0 } else { 2.0 },
            })
            .collect();
        let inputs = policy_inputs(&candidates, &urls);
        let request = RequestInputs {
            prompt_tokens: 32,
            block_size: 4,
            request_blocks: 8,
            avg_load: 0.0,
            prefix_hashes: None,
        };
        let mut saw_high = false;
        let mut saw_low = false;
        for _ in 0..400 {
            let Pick::Group(rows) = policy.select(&request, &inputs) else {
                panic!("expected a group");
            };
            let score = inputs[rows[0]].effective_score;
            assert!(rows.iter().all(|&row| inputs[row].effective_score == score));
            assert_eq!(rows.len(), 4, "a score group is every equal-score worker");
            if score == 8.0 {
                saw_high = true;
            } else {
                saw_low = true;
            }
        }
        assert!(
            saw_high && saw_low,
            "temperature must reach both score groups"
        );
    }

    #[test]
    fn every_catalog_policy_routes_event_driven_hits_and_misses() {
        for name in cost::POLICY_NAMES {
            let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
                selection_policy: Some((*name).to_string()),
                ..test_config()
            });
            let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
            policy.init_workers(&workers);
            let monitor = Arc::new(KvEventMonitor::new(Some(4)));
            let indexer =
                setup_indexer_with_blocks("http://w2:8000", &[&[1, 2, 3, 4], &[5, 6, 7, 8]], 4);
            monitor.indexers.insert("unknown".to_string(), indexer);
            policy.set_kv_event_monitor(Some(monitor));

            let hit = policy.select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
                    ..Default::default()
                },
            );
            assert_eq!(hit, Some(1), "{name}: the holder of every block must win");
            policy.on_request_complete("http://w2:8000", true);

            let miss = policy.select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&[100, 200, 300, 400]),
                    ..Default::default()
                },
            );
            assert!(miss.is_some(), "{name}: a miss must still route");
        }
    }

    #[test]
    fn every_catalog_policy_routes_tree_matches() {
        for name in cost::POLICY_NAMES {
            let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
                selection_policy: Some((*name).to_string()),
                ..test_config()
            });
            let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
            policy.init_workers(&workers);
            let text = "a shared system prompt that is long enough to match";
            let first = policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        request_text: Some(text),
                        ..Default::default()
                    },
                )
                .unwrap();
            policy.on_request_complete(workers[first].url(), true);
            let second = policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        request_text: Some(text),
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(first, second, "{name}: an idle holder keeps its prefix");
        }
    }

    #[test]
    fn accounting_books_the_dispatch_until_completion() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            selection_accounting_ttl_ms: 60_000,
            ..test_config()
        });
        let workers = make_workers(&["http://w1:8000", "http://w2:8000"]);
        policy.init_workers(&workers);
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        let indexer = setup_indexer_with_blocks("http://w1:8000", &[&[1, 2, 3, 4]], 4);
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));
        let accounting = policy.accounting.as_ref().expect("accounting enabled");

        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let selected = policy
            .select_worker(
                &workers,
                &SelectWorkerInfo {
                    tokens: Some(&tokens),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(selected, 0);
        // One block cached, one block (4 tokens) booked as uncached prefill.
        assert_eq!(accounting.pending_prefill_tokens("http://w1:8000"), 4);
        let hashes: Vec<u64> = request_prefix_hashes(&compute_request_content_hashes(&tokens, 4))
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        let predicted = accounting.predicted_overlaps(&hashes);
        assert_eq!(predicted.len(), 1);
        assert_eq!(&*predicted[0].0, "http://w1:8000");
        assert_eq!(
            predicted[0].1, 2.0,
            "the whole two-block prefix is predicted resident"
        );

        // A sibling that shares the first block but not the second is predicted one block on w1.
        let sibling: Vec<u64> = request_prefix_hashes(&compute_request_content_hashes(
            &[1, 2, 3, 4, 9, 9, 9, 9],
            4,
        ))
        .into_iter()
        .map(|hash| hash.0)
        .collect();
        assert_eq!(accounting.predicted_overlaps(&sibling)[0].1, 1.0);

        policy.on_request_complete("http://w1:8000", true);
        assert_eq!(accounting.pending_prefill_tokens("http://w1:8000"), 0);
    }

    #[test]
    fn reconciliation_releases_bookings_whose_completion_never_arrived() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            selection_accounting_ttl_ms: 60_000,
            ..test_config()
        });
        let accounting = policy.accounting.as_ref().expect("accounting enabled");
        accounting.record_dispatch("http://w1:8000", 100, &[]);
        accounting.record_dispatch("http://w1:8000", 200, &[]);
        accounting.record_dispatch("http://w1:8000", 400, &[]);

        // The router still holds all three: nothing to release.
        policy.reconcile_in_flight("http://w1:8000", 3);
        assert_eq!(accounting.pending_prefill_tokens("http://w1:8000"), 700);

        // Two requests ended without a completion report: the two oldest
        // bookings go, the one the router still holds stays.
        policy.reconcile_in_flight("http://w1:8000", 1);
        assert_eq!(accounting.pending_prefill_tokens("http://w1:8000"), 400);

        // A worker without bookings is a no-op, with or without accounting.
        policy.reconcile_in_flight("http://w2:8000", 0);
        CacheAwarePolicy::with_config(test_config()).reconcile_in_flight("http://w1:8000", 0);
    }

    #[test]
    fn invalid_selection_policy_falls_back_to_the_default() {
        let policy = CacheAwarePolicy::with_config(CacheAwareConfig {
            selection_policy: Some("no-such-policy".to_string()),
            ..test_config()
        });
        assert_eq!(policy.selection.name(), cost::DEFAULT_POLICY);
    }

    #[test]
    fn event_driven_salted_request_matches_only_its_namespace() {
        use kv_index::salt::{content_hash_with_seed, namespace_seed};
        use openai_protocol::common::CachePartition;

        let policy = CacheAwarePolicy::with_config(test_config());
        let w1 = BasicWorkerBuilder::new("http://w1:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        let w2 = BasicWorkerBuilder::new("http://w2:8000")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        // w1 carries more live load, so every miss resolves to w2.
        for _ in 0..3 {
            w1.increment_load();
        }
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(w1), Arc::new(w2)];
        policy.init_workers(&workers);

        // Blocks stored on w1 under (lora "adapter", salt "tenant-a"), as the
        // monitor hashes a salted KvBlocksStored event.
        let indexer = Arc::new(KvIndex::positional(4));
        let worker_id = indexer.intern_worker("http://w1:8000").unwrap();
        let mut wb = WorkerBlocks::default();
        let seed = namespace_seed(Some("adapter"), Some("tenant-a"));
        let blocks: Vec<StoredBlock> = [[1u32, 2, 3, 4], [5, 6, 7, 8]]
            .iter()
            .enumerate()
            .map(|(i, tokens)| StoredBlock {
                seq_hash: SequenceHash(i as u64 + 1),
                content_hash: content_hash_with_seed(tokens, seed),
            })
            .collect();
        indexer
            .apply_stored(worker_id, &blocks, None, &mut wb)
            .unwrap();
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        monitor.indexers.insert("unknown".to_string(), indexer);
        policy.set_kv_event_monitor(Some(monitor));

        let namespace = |salt: &'static str| {
            CacheNamespace::derive(&CachePartition {
                cache_salt: Some(salt),
                extra_key: None,
                lora_path: Some("adapter"),
            })
        };
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let route = |cache_namespace: Option<CacheNamespace>| {
            policy
                .select_worker(
                    &workers,
                    &SelectWorkerInfo {
                        tokens: Some(&tokens),
                        cache_namespace,
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        assert_eq!(route(namespace("tenant-a")), 0, "same namespace hits w1");
        assert_eq!(route(namespace("tenant-b")), 1, "another salt misses");
        assert_eq!(route(None), 1, "a plain request misses salted blocks");
    }

    // -----------------------------------------------------------------------
    // Pool tables and the sampled decision over a large pool
    // -----------------------------------------------------------------------

    /// `count` workers, all interned in an event index that holds `chunks`
    /// for the worker at `holder`, behind a policy built from `config`.
    fn large_pool(
        count: usize,
        holder: usize,
        chunks: &[&[u32]],
        config: CacheAwareConfig,
    ) -> (CacheAwarePolicy, Vec<Arc<dyn Worker>>) {
        let urls: Vec<String> = (0..count).map(|i| format!("http://w{i}:8000")).collect();
        let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
        let workers = make_workers(&refs);
        let indexer = setup_indexer_with_blocks(&urls[holder], chunks, 4);
        for (i, url) in urls.iter().enumerate() {
            if i != holder {
                indexer.intern_worker(url).unwrap();
            }
        }
        let policy = CacheAwarePolicy::with_config(config);
        policy.init_workers(&workers);
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        monitor.indexers.insert("unknown".to_string(), indexer);
        monitor.set_block_size("unknown", 4);
        policy.set_kv_event_monitor(Some(monitor));
        (policy, workers)
    }

    fn tokens_info(tokens: &[u32]) -> SelectWorkerInfo<'_> {
        SelectWorkerInfo {
            tokens: Some(tokens),
            ..Default::default()
        }
    }

    #[test]
    fn a_large_pool_routes_to_the_holder_the_index_names() {
        let (policy, workers) = large_pool(40, 37, &[&[1, 2, 3, 4], &[5, 6, 7, 8]], test_config());
        for _ in 0..8 {
            let idx = policy
                .select_worker(&workers, &tokens_info(&[1, 2, 3, 4, 5, 6, 7, 8]))
                .unwrap();
            assert_eq!(
                idx, 37,
                "the holder sits outside any sample of {FLEET_SAMPLE}; the index places it"
            );
        }
    }

    #[test]
    fn a_large_pool_miss_finds_the_one_eligible_worker_the_sample_missed() {
        let (policy, workers) = large_pool(40, 0, &[&[1, 2, 3, 4]], test_config());
        for worker in &workers[..39] {
            worker.set_status(WorkerStatus::NotReady);
        }
        let idx = policy.select_worker(&workers, &tokens_info(&[9, 9, 9, 9, 8, 8, 8, 8]));
        assert_eq!(idx, Some(39));
    }

    #[test]
    fn a_large_pool_miss_spreads_over_the_pool() {
        let (policy, workers) = large_pool(40, 0, &[&[1, 2, 3, 4]], test_config());
        let mut picked = HashSet::new();
        for turn in 0..200u32 {
            let tokens = [turn + 100, 9, 9, 9, 8, 8, 8, 8];
            let idx = policy
                .select_worker(&workers, &tokens_info(&tokens))
                .unwrap();
            picked.insert(idx);
            policy.on_request_complete(workers[idx].url(), true);
        }
        assert!(
            picked.len() > 2 * FLEET_SAMPLE,
            "200 misses reached {} workers; a fresh sample per decision reaches the pool",
            picked.len()
        );
    }

    #[test]
    fn pool_table_places_ids_and_rejects_a_worker_swapped_in_at_the_position() {
        let workers = make_workers(&["http://w1:8000", "http://w2:8000", "http://w3:8000"]);
        let indexer = setup_indexer_with_blocks("http://w3:8000", &[&[1, 2, 3, 4]], 4);
        let w3 = indexer.worker_id("http://w3:8000").unwrap();
        let table = PoolTable::build(&workers, &indexer, liveness::now_ms());
        assert!(table.describes(&workers));
        assert!(table.fresh(liveness::now_ms()));
        assert_eq!(table.position(w3, &workers), Some(2));
        assert_eq!(table.id_at, vec![None, None, Some(w3)]);
        assert_eq!(
            table.position(w3 + 100, &workers),
            None,
            "an id never interned"
        );
        // The same url behind a new worker object at the same position: the
        // table was built from the old one and says so.
        let mut swapped = workers.clone();
        swapped[2] = make_workers(&["http://w3:8000"]).remove(0);
        assert_eq!(table.position(w3, &swapped), None);
        // Every worker was admitted just now: a young fleet lists nothing to slice.
        assert!(table.warming.is_empty());
        assert!(!table.fresh(liveness::now_ms() + POOL_TABLE_REFRESH_MS));
    }

    /// `count` workers that all hold the shared block `[1, 2, 3, 4]` (a chat
    /// template's head), the one at `deep` also `[5, 6, 7, 8]`.
    fn shared_prefix_pool(
        count: usize,
        deep: usize,
        config: CacheAwareConfig,
    ) -> (CacheAwarePolicy, Vec<Arc<dyn Worker>>) {
        let urls: Vec<String> = (0..count).map(|i| format!("http://w{i}:8000")).collect();
        let refs: Vec<&str> = urls.iter().map(String::as_str).collect();
        let workers = make_workers(&refs);
        let indexer = Arc::new(KvIndex::positional(4));
        for (i, url) in urls.iter().enumerate() {
            let id = indexer.intern_worker(url).unwrap();
            let mut wb = WorkerBlocks::default();
            let mut blocks = vec![StoredBlock {
                seq_hash: SequenceHash(1),
                content_hash: compute_content_hash(&[1, 2, 3, 4]),
            }];
            if i == deep {
                blocks.push(StoredBlock {
                    seq_hash: SequenceHash(2),
                    content_hash: compute_content_hash(&[5, 6, 7, 8]),
                });
            }
            indexer.apply_stored(id, &blocks, None, &mut wb).unwrap();
        }
        let policy = CacheAwarePolicy::with_config(config);
        policy.init_workers(&workers);
        let monitor = Arc::new(KvEventMonitor::new(Some(4)));
        monitor.indexers.insert("unknown".to_string(), indexer);
        monitor.set_block_size("unknown", 4);
        policy.set_kv_event_monitor(Some(monitor));
        (policy, workers)
    }

    #[test]
    fn the_default_decision_takes_the_deepest_holder_among_a_fleet_sharing_the_head() {
        let (policy, workers) = shared_prefix_pool(64, 50, test_config());
        for _ in 0..4 {
            let idx = policy
                .select_worker(&workers, &tokens_info(&[1, 2, 3, 4, 5, 6, 7, 8]))
                .unwrap();
            assert_eq!(idx, 50);
        }
        // The deepest holder down: the decision falls to the holders of the
        // head, never to a worker without the blocks.
        workers[50].set_status(WorkerStatus::NotReady);
        let idx = policy
            .select_worker(&workers, &tokens_info(&[1, 2, 3, 4, 5, 6, 7, 8]))
            .unwrap();
        assert_ne!(idx, 50);
        assert!(workers[idx].is_healthy());
    }

    #[test]
    fn a_fleet_wide_tie_is_drawn_from_not_scored_whole() {
        let (policy, workers) = shared_prefix_pool(64, 50, test_config());
        let mut picked = HashSet::new();
        for _ in 0..200 {
            let idx = policy
                .select_worker(&workers, &tokens_info(&[1, 2, 3, 4]))
                .unwrap();
            picked.insert(idx);
            policy.on_request_complete(workers[idx].url(), true);
        }
        assert!(
            picked.len() > 2 * FLEET_SAMPLE,
            "200 decisions on a head every worker holds reached {} workers",
            picked.len()
        );
    }

    #[test]
    fn merge_rows_unions_the_eligible_sample_with_the_holders_in_slice_order() {
        let candidates: Vec<OverlapCandidate> = [3usize, 7, 20]
            .iter()
            .map(|&idx| OverlapCandidate {
                idx,
                raw_score: 1.0,
                effective_score: 1.0,
            })
            .collect();
        assert_eq!(
            CacheAwarePolicy::merge_rows(&[1, 3, 9, 30], &candidates),
            vec![1, 3, 7, 9, 20, 30]
        );
        assert_eq!(
            CacheAwarePolicy::merge_rows(&[], &candidates),
            vec![3, 7, 20]
        );
        assert_eq!(CacheAwarePolicy::merge_rows(&[2, 4], &[]), vec![2, 4]);
    }
}
