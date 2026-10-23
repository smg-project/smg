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
    balance_abs_threshold:   Absolute margin of the spill gate (requests above the fleet mean); capped at
                             the larger of the in-flight count's natural swing at the pool's mean and
                             the held prefix's price, never under four unless the configured value
                             itself is lower (0: the relative part alone)
    balance_prefix_margin_tokens: Price of a held prefix in the absolute margin: a holder's margin is at
                             least its held tokens over this many tokens per request, up to the
                             configured absolute threshold (0, the default: the held prefix does not
                             count; decode picks are never priced)
    balance_uncached_credit: Credit a pick on a holder with the prompt it still has to prefill (never
                             under the mean prefill nor over the whole prompt) instead of the whole
                             prompt (default off; decode picks keep the whole prompt)
    balance_rel_threshold:   Relative margin of the spill gate (multiple of the fleet mean)
    eviction_interval_secs:  Interval between LRU eviction / TTL sweep cycles
    max_tree_size:           Max total size (chars/tokens) of each model's approximate tree,
                             shared across all workers; enforced by eviction
    block_size:              Backend KV cache block size for event-driven routing
    cache_index:             Under-layer: tree (radix trees) or hash (placement map)
    cache_ttl_secs:          Seconds a hash-index placement stays routable
    cache_boundaries:        Ascending token positions for hash-index keying
*/

use std::{
    borrow::Cow,
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
use parking_lot::RwLock;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

use super::{
    cost::{
        self, CandidateInputs, OptimisticAccounting, Pick, RequestInputs, WorkerSelectionPolicy,
    },
    normalize_model_key,
    prefix_hash::CHARS_PER_TOKEN,
    utils::PeriodicTask,
    CacheAwareConfig, CacheNamespace, LeastLoadPolicy, LoadBalancingPolicy, SelectWorkerInfo,
    WorkerRole, TEXT_MARKER_LEN,
};
use crate::{
    config::CacheIndexKind,
    mesh::adapters::tree_sync::{RepairEntry, TreeDelta, TreeRepairPage, TreeSyncAdapter},
    observability::{cache_trace, metrics::Metrics},
    worker::{
        liveness,
        load_state::{LoadReceiver, LoadSnapshot},
        KvEventMonitor, KvIndex, Worker,
    },
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
/// regular pool, the PD roles' pools); past this the oldest goes.
const POOL_TABLES_KEPT: usize = 8;

/// The floor of the swing-sized absolute margin of the spill gate: with a
/// configured `balance_abs_threshold` of at least this, a worker at most this
/// many requests above the fleet mean is never over the gate, so a handful of
/// requests on an otherwise idle fleet keep their cache affinity; a lower
/// configured value lowers the margin below it (see
/// [`CacheAwarePolicy::absolute_margin`]).
const MIN_ABSOLUTE_MARGIN: f64 = 4.0;

/// What a decision's prefix lengths count: the token tree and the event
/// index count tokens, the string tree characters. The spill gate prices a
/// held prefix in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefixUnit {
    Tokens,
    Chars,
}

impl PrefixUnit {
    /// `units` of this kind in tokens: characters at [`CHARS_PER_TOKEN`] a
    /// token, as the prefix-hash policy hashes untokenized text.
    fn tokens(self, units: f64) -> f64 {
        match self {
            Self::Tokens => units,
            Self::Chars => units / CHARS_PER_TOKEN as f64,
        }
    }
}

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
    /// The worker monitor's load feed, read on the hot path for the KV-usage
    /// imbalance trigger and the waiting-prefill decay; the expected-wait
    /// scorer holds its own handle to the same feed. `None` until wired by
    /// the registry (then the policy stays count-only).
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

    // ---- The four readings of a request's cache namespace ----
    //
    // A request in a cache namespace is keyed under the namespace's marker
    // and an unpartitioned request under its plain key with any marker
    // material stripped, so the two key spaces are disjoint by construction
    // (see cache_namespace.rs). The approximate trees read the rule through
    // `keyed_tokens` / `keyed_text`, hash mode through `placement_tokens`,
    // the event-driven index through `event_content_hashes`.

    /// The token tree's key for `tokens` and the length of the marker at its
    /// head. A partitioned request keys under its namespace marker: another
    /// namespace diverges at position 0 and can never match, and the marker
    /// never counts toward the match ratio, so a match count read from the
    /// tree has the marker length subtracted. The marker fills whole pages,
    /// so the prompt's page grid is the unpartitioned one. An unpartitioned
    /// request keys under its tokens stripped of marker material, marker
    /// length 0.
    #[inline]
    fn keyed_tokens<'a>(
        &self,
        info: &SelectWorkerInfo,
        tokens: &'a [u32],
    ) -> (Cow<'a, [u32]>, usize) {
        let page_size = self.config.block_size.max(1);
        match info.cache_namespace {
            Some(namespace) => (
                Cow::Owned(namespace.prefixed_tokens(tokens, page_size)),
                CacheNamespace::token_marker_len(page_size),
            ),
            None => (
                Cow::Borrowed(CacheNamespace::unpartitioned_tokens(tokens)),
                0,
            ),
        }
    }

    /// The string tree's key for `text` and the length of the marker at its
    /// head: the rule of [`Self::keyed_tokens`], in chars.
    #[inline]
    fn keyed_text<'a>(info: &SelectWorkerInfo, text: &'a str) -> (Cow<'a, str>, usize) {
        match info.cache_namespace {
            Some(namespace) => (Cow::Owned(namespace.prefixed_text(text)), TEXT_MARKER_LEN),
            None => (Cow::Borrowed(CacheNamespace::unpartitioned_text(text)), 0),
        }
    }

    /// Hash mode's key material: a namespaced request's tokens as sent (its
    /// head is hashed under the namespace, see `hash_token_head`); an
    /// unpartitioned head stripped of marker material like the tree keys, so
    /// the placement key spaces stay disjoint by construction too. `None`
    /// for an untokenized request.
    #[inline]
    fn placement_tokens<'a>(info: &SelectWorkerInfo<'a>) -> Option<&'a [u32]> {
        match info.cache_namespace {
            Some(_) => info.tokens,
            None => info.tokens.map(CacheNamespace::unpartitioned_tokens),
        }
    }

    /// The event-driven index's block hashes for `tokens`. The engines fold
    /// the LoRA name and cache salt into their block hashes and the monitor
    /// recomputes stored blocks under the same seed, so a request in a
    /// namespace is hashed under it and matches only its own blocks; a plain
    /// request keeps the plain hash.
    #[inline]
    fn event_content_hashes(
        info: &SelectWorkerInfo,
        tokens: &[u32],
        block_size: usize,
    ) -> Vec<ContentHash> {
        match info.cache_namespace {
            Some(namespace) => {
                request_content_hashes_with_seed(tokens, block_size, namespace.event_seed())
            }
            None => compute_request_content_hashes(tokens, block_size),
        }
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
                let usage = load.effective_kv_pressure();
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
            let (keyed, _marker_len) = self.keyed_tokens(info, tokens);
            let tokens: &[u32] = &keyed;
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
            let (keyed, _marker_len) = Self::keyed_text(info, text);
            let text: &str = &keyed;
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

    /// The count-pressure gate moves a request off a holder loaded well past
    /// the pool's mean (`candidate_requires_spill`; with a prefix price set,
    /// the deeper the prefix it holds, the further past), so the holder of a
    /// request's prefix keeps it until then.
    fn balances_load_against_affinity(&self) -> bool {
        true
    }

    fn needs_request_text(&self) -> bool {
        true // Cache-aware policy needs request text for cache affinity
    }

    /// With token ids the selection never reads the text (`select_worker`
    /// takes the ids first and the text only without them), so a tokenized
    /// request keeps no copy of its prompt for a retry's re-selection.
    fn needs_request_text_with_tokens(&self) -> bool {
        false
    }

    fn set_load_receiver(&self, rx: Option<LoadReceiver>) {
        // The scorer reads the feed for the expected wait; this policy reads
        // it for KV pressure and the waiting-prefill decay.
        self.load_scorer.set_load_receiver(rx.clone());
        *self.load_rx.write() = rx;
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
            let eligible = info.admits(state);
            if cache_trace::enabled() && !eligible {
                cache_trace::gate(serde_json::json!({
                    "phase": "eligibility", "worker": worker.url(),
                    "load": state.load, "overloaded": state.overloaded,
                    "eligible": false,
                }));
            }
            // The overload veto costs nothing here: `state` is the word this
            // pass already loaded for health, circuit breaker and load.
            if eligible {
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
            if info.admits(state) {
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
        self.load_scorer
            .select_min_expected_wait(workers, candidates, info, self.name())
    }

    /// [`Self::select_expected_wait`] over holders of the request's prefix,
    /// `held` giving the tokens each holds. With `balance_uncached_credit`
    /// the winner is credited the prompt it still has to prefill, its token
    /// count less the held tokens, never under the mean prefill nor over the
    /// whole prompt (a miss's credit), instead of the whole prompt: a hit on
    /// a long cached conversation is not booked as the full context's work,
    /// so the next miss is not steered away from that worker as if it were.
    /// Requests without token ids keep the mean-prefill credit, and decode
    /// picks the whole prompt (a decode worker prefills nothing).
    fn select_expected_wait_on_holders(
        &self,
        workers: &[Arc<dyn Worker>],
        candidates: &[usize],
        held: &[(usize, f64)],
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let credited = self.config.balance_uncached_credit && info.role != WorkerRole::Decode;
        let Some(tokens) = info.tokens.filter(|_| credited) else {
            return self.select_expected_wait(workers, candidates, info);
        };
        let prompt = tokens.len() as u64;
        let floor = u64::from(self.load_scorer.mean_prefill_tokens());
        let uncached = |idx: usize| {
            let held_tokens = held
                .iter()
                .find(|&&(holder, _)| holder == idx)
                .map_or(0.0, |&(_, held_tokens)| held_tokens);
            ((prompt as f64 - held_tokens).max(0.0) as u64)
                .max(floor)
                .min(prompt)
        };
        self.load_scorer.select_min_expected_wait_credited(
            workers,
            candidates,
            info,
            self.name(),
            Some(&uncached),
        )
    }

    /// The count-pressure gate's comparison alone, with no trace record: a
    /// worker holding more than `balance_rel_threshold` times the mean load
    /// AND the absolute margin ([`Self::absolute_margin`]) above it is over
    /// the gate. `held_tokens` is the request's prefix the worker already
    /// holds (0 for a worker that is not a holder). The affinity path
    /// records its decision through [`Self::candidate_requires_spill`]; the
    /// warm-up slice checks its candidates here, so a thin-overlap miss
    /// writes no gate entries for workers it did not pick.
    fn over_count_pressure_gate(&self, load: f64, avg_load: f64, held_tokens: f64) -> bool {
        load > avg_load * f64::from(self.config.balance_rel_threshold)
            && load > avg_load + self.absolute_margin(avg_load, held_tokens)
    }

    /// The gate's absolute margin at `avg_load` for a worker holding
    /// `held_tokens` of the request's prefix: `balance_abs_threshold`
    /// requests above the mean, capped at the larger of the in-flight
    /// count's natural swing at this mean (two standard deviations of a
    /// Poisson count, `2 * sqrt(avg_load)`, rounded up to whole requests)
    /// and the held prefix's price ([`Self::held_prefix_margin`]), and never
    /// under [`MIN_ABSOLUTE_MARGIN`] unless the configured value is lower
    /// still (0 leaves the relative margin to decide alone). A fixed
    /// absolute margin sized for a large pool cannot clear in a pool whose
    /// per-worker in-flight count stays below it (two workers at 32 in
    /// flight: the busier one would need 80 at a margin of 64), so cache
    /// affinity there was never balanced at all, whatever the skew. Sized to
    /// the swing, the margin still tells steady-state variance from pressure
    /// at every mean: a worker two standard deviations above the fleet is
    /// not noise (a mean of 16: eight requests; a mean of 4: four), the
    /// configured value caps it, and the floor keeps a handful of requests
    /// on an otherwise idle fleet from counting as pressure. A spill gives
    /// the held prefix up, and the spill target prefills it again (a long
    /// conversation's follow-up holds nearly all of its context): with a
    /// price set, the holder keeps the request until its excess load is
    /// worth that recompute, so a deep prefix stays put up to the configured
    /// margin and a shallow one balances as the swing allows. The price
    /// counts the holder's whole match, even when the spill target holds
    /// part of it. The margin is not tied
    /// to the relative one, so a tight relative threshold (1.1) does not
    /// turn one idle worker in a loaded fleet into a spill of every busy
    /// holder.
    fn absolute_margin(&self, avg_load: f64, held_tokens: f64) -> f64 {
        let configured = self.config.balance_abs_threshold as f64;
        let swing = (2.0 * avg_load.sqrt()).ceil();
        configured.min(
            swing
                .max(MIN_ABSOLUTE_MARGIN)
                .max(self.held_prefix_margin(held_tokens)),
        )
    }

    /// A held prefix in requests' worth of prefill: `held_tokens` over
    /// `balance_prefix_margin_tokens`, rounded up. 0 when the price is 0 (the
    /// default), which leaves the margin to the swing alone.
    fn held_prefix_margin(&self, held_tokens: f64) -> f64 {
        let price = self.config.balance_prefix_margin_tokens;
        if price == 0 || held_tokens <= 0.0 {
            return 0.0;
        }
        (held_tokens / f64::from(price)).ceil()
    }

    /// Per-request count-pressure predicate: over
    /// `balance_rel_threshold` times the healthy-fleet mean load AND the
    /// absolute margin above it (which weighs the `held_tokens` of the
    /// request's prefix the worker holds), the request spills to the
    /// expected-wait worker instead. Affinity paths insert for the spill
    /// target, so a prefix whose home saturates gains an additional tenant —
    /// hot prefixes replicate instead of queueing behind one engine. Both
    /// margins must clear so the gate neither fires on steady-state variance
    /// (relative alone would, at low means) nor stays blind to a deep queue
    /// (absolute alone would, at high means). Outside the fleet-wide
    /// KV-pressure fallback, a request with no holder is routed among the
    /// workers under the same gate ([`Self::ungated_candidates`]), so a
    /// burst of misses is bounded by the margins a hit spills at.
    fn candidate_requires_spill(
        &self,
        workers: &[Arc<dyn Worker>],
        selected: usize,
        avg_load: f64,
        held_tokens: f64,
    ) -> bool {
        let load = workers[selected].load() as f64;
        let spill = self.over_count_pressure_gate(load, avg_load, held_tokens);
        if cache_trace::enabled() {
            cache_trace::gate(serde_json::json!({
                "worker": workers[selected].url(), "load": load, "average_load": avg_load,
                "relative_threshold": self.config.balance_rel_threshold,
                "absolute_threshold": self.config.balance_abs_threshold,
                "held_tokens": held_tokens,
                "prefix_margin_tokens": self.config.balance_prefix_margin_tokens,
                "absolute_margin": self.absolute_margin(avg_load, held_tokens), "spill": spill,
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
    /// the pick: still eligible, still warming, and under the count-pressure
    /// gate an affinity pick obeys (`balance_rel_threshold` times the mean
    /// in-flight and the absolute margin above it), so a thin worker that
    /// cannot admit what it is sent (an in-transit backlog, a throttled
    /// admission) is not fed one miss in `1 / share` while the rest of the
    /// pool idles. The pick is credited through the expected-wait selector
    /// like any other.
    fn warmup_slice(
        &self,
        workers: &[Arc<dyn Worker>],
        table: &PoolTable,
        indexer: &KvIndex,
        info: &SelectWorkerInfo,
        avg_load: f64,
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
            if !info.admits(state) {
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
            if self.over_count_pressure_gate(workers[idx].load() as f64, avg_load, 0.0) {
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
    /// one, but a deep hit only away from a holder with more in flight than
    /// the thin worker (the least loaded of the request's deepest holders in
    /// `candidates`, where the request goes otherwise): recomputing a deep
    /// overlap is worth it where the request would queue, not on an idle
    /// fleet, where a single client's multi-turn continuations were
    /// recomputed from scratch on every worker joining a scale-out, one
    /// continuation in eight losing its whole context; never while the thin
    /// worker has anything in flight unless its last diversion is older than
    /// [`DIVERT_WINDOW_MS`] (the stamp lives on the worker, so it survives
    /// the table's rebuilds), so a hot fleet is not disturbed; and only until
    /// its index crosses the thinness ratio of the fleet's level. Equal loads
    /// draw uniformly like the slice. A fleet with no thin worker returns
    /// before any clock read or counter: this costs the steady state nothing.
    /// Protection and the liveness vetoes apply through the routing state as
    /// everywhere; the pick is credited through the expected-wait selector
    /// like the slice's.
    fn warmup_divert(
        &self,
        workers: &[Arc<dyn Worker>],
        table: &PoolTable,
        indexer: &KvIndex,
        info: &SelectWorkerInfo,
        candidates: &[OverlapCandidate],
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
        // What a deep hit's thin worker is measured against: the least
        // loaded of the request's deepest holders, read only once a deep hit
        // is due. A thin worker among them recomputes nothing and is exempt.
        let deepest: Vec<usize> = if shallow {
            Vec::new()
        } else {
            let best = candidates
                .iter()
                .map(|candidate| candidate.raw_score)
                .fold(0.0_f64, f64::max);
            candidates
                .iter()
                .filter(|candidate| candidate.raw_score >= best)
                .map(|candidate| candidate.idx)
                .collect()
        };
        let holder_load = deepest.iter().map(|&holder| workers[holder].load()).min();
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
            if !info.admits(state) {
                continue;
            }
            let indexed = table.id_at[idx].map_or(0, |id| indexer.worker_block_count(id));
            if !warmup.is_thin(indexed, table.fleet_level) {
                continue;
            }
            if state.load > 0 && now < worker.divert_until_ms() {
                continue;
            }
            if holder_load.is_some_and(|holder_load| state.load >= holder_load)
                && !deepest.contains(&idx)
            {
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

    /// Resolve an affinity score group to one final worker. Each affinity
    /// candidate comes with the tokens of the request's prefix it holds,
    /// which its pressure gate weighs, except on a decode pick: a decode
    /// worker prefills nothing again, so it keeps the swing's margin. Safe
    /// affinity candidates retain
    /// priority; only when every tied holder trips the pressure gate do we
    /// scan the healthy fleet for non-gated spill targets. The expected-wait
    /// selector both chooses and credits the result under one lock, so no
    /// provisional holder is ever accounted.
    fn select_final_from_affinity(
        &self,
        workers: &[Arc<dyn Worker>],
        affinity_candidates: &[(usize, f64)],
        healthy_indices: &[usize],
        avg_load: f64,
        info: &SelectWorkerInfo,
    ) -> Option<usize> {
        let priced =
            self.config.balance_prefix_margin_tokens > 0 && info.role != WorkerRole::Decode;
        let safe_affinity: Vec<usize> = affinity_candidates
            .iter()
            .filter(|&&(idx, held_tokens)| {
                let held_tokens = if priced { held_tokens } else { 0.0 };
                !self.candidate_requires_spill(workers, idx, avg_load, held_tokens)
            })
            .map(|&(idx, _)| idx)
            .collect();
        if !safe_affinity.is_empty() {
            // Kept only by the held prefix: the swing-sized margin alone
            // would have spilled every safe holder.
            if priced
                && safe_affinity.iter().all(|&idx| {
                    self.over_count_pressure_gate(workers[idx].load() as f64, avg_load, 0.0)
                })
            {
                Metrics::record_cache_aware_prefix_margin_held();
            }
            return self.select_expected_wait_on_holders(
                workers,
                &safe_affinity,
                affinity_candidates,
                info,
            );
        }

        // A miss has no affinity candidates and is not a spill, but it goes
        // to the expected-wait worker among the eligible workers under the
        // same count-pressure gate, so a burst of misses is bounded by the
        // margins a hit spills at. Expected wait alone keeps no count bound:
        // its queue and rate signals are a poll old, and a burst routed on
        // them piles onto one worker while another has free slots. A true
        // spill excludes other workers that trip the same gate, preventing a
        // fallback from reselecting a hot holder.
        if affinity_candidates.is_empty() {
            let candidates = self.ungated_candidates(workers, healthy_indices, avg_load);
            return self.select_expected_wait(workers, &candidates, info);
        }
        // A spill target gives no prefix up: its gate is the swing's.
        let spill_candidates: Vec<usize> = healthy_indices
            .iter()
            .copied()
            .filter(|&idx| !self.candidate_requires_spill(workers, idx, avg_load, 0.0))
            .collect();
        let candidates = if spill_candidates.is_empty() {
            healthy_indices
        } else {
            &spill_candidates
        };
        self.select_expected_wait(workers, candidates, info)
    }

    /// The eligible workers under the count-pressure gate at the swing's
    /// margin (a request with no holder gives no prefix up), compared
    /// without a trace record like the warm-up slice's check, so a miss
    /// writes no gate entries for workers it did not pick. The lightest
    /// worker never clears the snapshot mean, but `avg_load` and the
    /// per-worker load re-read can disagree under concurrent dispatch, so
    /// the guard keeps a selection from returning no worker.
    fn ungated_candidates(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        avg_load: f64,
    ) -> Vec<usize> {
        let ungated: Vec<usize> = healthy_indices
            .iter()
            .copied()
            .filter(|&idx| {
                !self.over_count_pressure_gate(workers[idx].load() as f64, avg_load, 0.0)
            })
            .collect();
        if ungated.is_empty() {
            healthy_indices.to_vec()
        } else {
            ungated
        }
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
    /// - `Pick::None`: the miss path (expected wait over the eligible workers
    ///   under the count-pressure gate).
    /// - `Pick::Final`: that worker, credited through the expected-wait
    ///   selector; the policy owns the load trade-off, so the count-pressure
    ///   gate does not apply. If the waiting-queue veto drops it, the fleet
    ///   fallback runs.
    ///
    /// `unit` is what the request's lengths count (tokens, or the string
    /// tree's characters); the gate prices each holder's prefix in tokens.
    #[expect(clippy::too_many_arguments, reason = "hot-path plumbing, not state")]
    fn resolve_selection(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        candidates: &[OverlapCandidate],
        request: &RequestInputs<'_>,
        unit: PrefixUnit,
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
                // Each holder with the prefix it holds, in tokens.
                let group: Vec<(usize, f64)> = rows
                    .iter()
                    .map(|&row| {
                        let candidate = &inputs[row];
                        let held_units =
                            candidate.device_blocks.max(0.0) * request.block_size as f64;
                        (candidate.idx, unit.tokens(held_units))
                    })
                    .collect();
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
    #[expect(clippy::too_many_arguments, reason = "hot-path plumbing, not state")]
    fn select_matched_candidate(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        matched_tenants: &[TenantId],
        matched_units: usize,
        request: &RequestInputs<'_>,
        unit: PrefixUnit,
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
            unit,
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

        let content_hashes = Self::event_content_hashes(info, tokens, block_size);
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
            info,
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
            if let Some(idx) = self.warmup_slice(workers, &table, indexer, info, avg_load) {
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
            &candidates,
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
            PrefixUnit::Tokens,
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
        // The predicted overlap of every event-driven decision, so the
        // histogram reads the production ratio and not the tree path's
        // alone (`request_blocks` is at least one).
        Metrics::record_cache_aware_match_ratio(overlap_blocks as f64 / request_blocks);
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
            &SelectWorkerInfo::default(),
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
    #[expect(
        clippy::too_many_arguments,
        reason = "the holder gather threads the selection's inputs through"
    )]
    fn overlap_candidates_for_hashes(
        workers: &[Arc<dyn Worker>],
        content_hashes: &[ContentHash],
        table: &PoolTable,
        indexer: &KvIndex,
        block_size: usize,
        tuning: &OverlapTuning<'_>,
        gather: Gather,
        info: &SelectWorkerInfo,
    ) -> Vec<OverlapCandidate> {
        let mut candidates =
            Self::overlap_holders(workers, content_hashes, table, indexer, gather, info);
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
        info: &SelectWorkerInfo,
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
            Gather::TopGroup { cap } => Self::top_group(workers, table, &overlap, cap, info),
            Gather::All { cap } => Self::all_holders(workers, table, &overlap, cap, info),
        };
        // Downstream reads candidates in slice order: the all-workers merge
        // in `resolve_selection` and the tie-breaks.
        candidates.sort_unstable_by_key(|candidate| candidate.idx);
        candidates
    }

    /// The eligible holders of the deepest overlap: one pass over the
    /// scores for the depth, one to place and read those holders. When none
    /// of them is in this pool and routable (a PD role's pool sharing the
    /// model's index with the other role, a holder down), the full gather
    /// decides, so the result is always the deepest group a scan of the pool
    /// would have found.
    fn top_group(
        workers: &[Arc<dyn Worker>],
        table: &PoolTable,
        overlap: &OverlapScores,
        cap: Option<usize>,
        info: &SelectWorkerInfo,
    ) -> Vec<OverlapCandidate> {
        let deepest = overlap.scores.values().copied().max().unwrap_or(0);
        if deepest == 0 {
            return Vec::new();
        }
        let group = Self::holders_at(workers, table, overlap, deepest, cap, info);
        if !group.is_empty() {
            return group;
        }
        let mut all = Self::all_holders(workers, table, overlap, None, info);
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
        info: &SelectWorkerInfo,
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
            if !info.admits(workers[idx].routing_state()) {
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
        info: &SelectWorkerInfo,
    ) -> Vec<OverlapCandidate> {
        let mut candidates: Vec<OverlapCandidate> = Vec::with_capacity(overlap.scores.len());
        for (&id, &score) in &overlap.scores {
            if score == 0 {
                continue;
            }
            let Some(idx) = table.position(id, workers) else {
                continue;
            };
            if !info.admits(workers[idx].routing_state()) {
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
        let tokens: &[u32] = Self::placement_tokens(info).unwrap_or(&[]);
        let applicable_end = self
            .config
            .cache_boundaries
            .partition_point(|&p| p <= tokens.len());
        let applicable = &self.config.cache_boundaries[..applicable_end];

        // Backend KV pressure sheds fleet-wide ahead of any count gate, as the
        // tree path does for every request it routes.
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

        // Untokenized or head-only traffic has no holder to find: it stays
        // load-balanced, bounded by the count-pressure gate like a tree miss.
        if applicable.is_empty() {
            let candidates = self.ungated_candidates(workers, healthy_indices, avg_load);
            let branch = if tokens.is_empty() {
                "expected_wait_fallback"
            } else {
                "short_request"
            };
            return self.hash_expected_wait(
                workers,
                info,
                &candidates,
                model_id,
                branch,
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
            // A holder of this boundary holds `boundary` tokens of the prompt.
            let holders: Vec<(usize, f64)> = holder_candidates
                .iter()
                .map(|&idx| (idx, boundary as f64))
                .collect();
            let selected = self.select_final_from_affinity(
                workers,
                &holders,
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

        // No live holder at any boundary: bounded by the count-pressure gate
        // like a tree miss.
        let candidates = self.ungated_candidates(workers, healthy_indices, avg_load);
        self.hash_expected_wait(
            workers,
            info,
            &candidates,
            model_id,
            "expected_wait_fallback",
            tokens,
            applicable,
            now,
        )
    }

    /// Expected-wait dispatch over `candidates` for hash-path fallback
    /// branches; records only the final worker when boundaries apply.
    #[expect(clippy::too_many_arguments, reason = "hot-path plumbing, not state")]
    fn hash_expected_wait(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo,
        candidates: &[usize],
        model_id: &str,
        branch: &'static str,
        tokens: &[u32],
        applicable: &[usize],
        now: Instant,
    ) -> Option<usize> {
        let idx = self.select_expected_wait(workers, candidates, info)?;
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

        // The namespaced key (see `keyed_tokens`); a match count read from
        // the tree has `marker_len` subtracted.
        let (keyed, marker_len) = self.keyed_tokens(info, tokens);
        let keyed: &[u32] = &keyed;

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
                    PrefixUnit::Tokens,
                    info,
                )
            } else {
                self.resolve_selection(
                    workers,
                    healthy_indices,
                    &[],
                    &request,
                    PrefixUnit::Tokens,
                    avg_load,
                    info,
                )
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
        let (keyed, marker_len) = Self::keyed_text(info, text);
        let keyed: &str = &keyed;

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
                    PrefixUnit::Chars,
                    info,
                )
            } else {
                self.resolve_selection(
                    workers,
                    healthy_indices,
                    &[],
                    &request,
                    PrefixUnit::Chars,
                    avg_load,
                    info,
                )
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
mod tests;
