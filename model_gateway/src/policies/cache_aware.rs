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

    Configuration Parameters:
    ------------------------
    cache_threshold:         Min prefix match ratio for highest-match routing (0.0-1.0)
    balance_abs_threshold:   Absolute load diff threshold for imbalance detection
    balance_rel_threshold:   Relative load ratio threshold for imbalance detection
    eviction_interval_secs:  Interval between LRU eviction cycles
    max_tree_size:           Max nodes per approximate tree before eviction
    block_size:              Backend KV cache block size for event-driven routing
*/

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use dashmap::DashMap;
use kv_index::{compute_request_content_hashes, PositionalIndexer, TokenTree, Tree};
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
    mesh::adapters::tree_sync::{RepairEntry, TreeRepairPage},
    worker::{KvEventMonitor, Worker},
};

/// Latest per-worker backend load snapshot stream, keyed by worker URL.
pub(crate) type LoadReceiver = watch::Receiver<HashMap<String, WorkerLoadResponse>>;

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

impl CacheAwarePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let string_trees = Arc::new(DashMap::<String, Arc<Tree>>::new());
        let token_trees = Arc::new(DashMap::<String, Arc<TokenTree>>::new());
        let hash_index = Arc::new(DashMap::<String, PerModelHashIndex>::new());

        // Start background eviction thread if configured
        let eviction_task = if config.eviction_interval_secs > 0 {
            let string_trees_clone = Arc::clone(&string_trees);
            let token_trees_clone = Arc::clone(&token_trees);
            let hash_index_clone = Arc::clone(&hash_index);
            let max_tree_size = config.max_tree_size;

            Some(PeriodicTask::spawn(
                config.eviction_interval_secs,
                "Eviction",
                move || {
                    // Evict string trees (HTTP)
                    for tree_ref in string_trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

                        debug!(
                            "String tree eviction completed for model {}, max_size: {}",
                            model_id, max_tree_size
                        );
                    }
                    // Evict token trees (gRPC)
                    for tree_ref in token_trees_clone.iter() {
                        let model_id = tree_ref.key();
                        let tree = tree_ref.value();
                        tree.evict_tenant_by_size(max_tree_size);

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

                    // Log tree sizes — model counts + hash-index total.
                    // DO NOT call tree.snapshot() here — it clones all
                    // edge text (~170 MB) every cycle.
                    tracing::info!(
                        "Tree memory: string_trees={} models, token_trees={} models, \
                         hash_index={} models / {} entries",
                        string_trees_clone.len(),
                        token_trees_clone.len(),
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
        }
    }

    /// Enable request-hot-path `hash_index` population. Called by mesh
    /// wiring when the policy is attached to a mesh adapter; otherwise
    /// the index stays empty (its only readers are mesh-only paths).
    pub fn set_populate_hash_index(&self, enabled: bool) {
        self.populate_hash_index.store(enabled, Ordering::Relaxed);
    }

    fn should_populate_hash_index(&self) -> bool {
        self.populate_hash_index.load(Ordering::Relaxed)
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

    /// True when the pool is imbalanced enough to abandon cache affinity.
    ///
    /// Three independent triggers, OR'd together. The two KV-based triggers
    /// require a backend `token_usage` snapshot and are disabled at their `1.0`
    /// default (utilization and spread are both `<= 1.0`, so `> 1.0` never
    /// fires):
    ///
    /// - **overload** (`overload_token_usage_threshold`): the hottest engine's
    ///   KV utilization exceeds the ceiling — a critically-saturated engine,
    ///   shed regardless of balance. Set high (e.g. 0.9) as a safety valve.
    /// - **KV spread** (`balance_token_usage_threshold`): the hottest engine is
    ///   materially more KV-saturated than the coldest, i.e. a cooler engine
    ///   exists to spill toward. This is the true balance signal for long-context
    ///   workloads, and — unlike request counts, which each gateway sees only
    ///   locally — it is invariant to the number of gateway replicas.
    /// - **count spread**: request-count dispersion (abs AND rel) over healthy
    ///   workers. Always evaluated, so high-count / low-KV imbalance is still
    ///   caught when KV looks even.
    /// Whether to abandon cache affinity for shortest-queue because the pool is
    /// imbalanced — by backend KV usage (overload ceiling or hot-vs-cool spread)
    /// or by request-count spread. `min_load`/`max_load` are the request-count
    /// bounds over the healthy workers, which `select_worker` gathers in its
    /// single worker pass (tests use the `imbalanced` helper to fold them).
    fn is_imbalanced(
        &self,
        workers: &[Arc<dyn Worker>],
        healthy_indices: &[usize],
        min_load: usize,
        max_load: usize,
    ) -> bool {
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

        // Count spread (abs AND rel) over healthy workers.
        max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold)
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
                .or_insert_with(|| Arc::new(TokenTree::new()));

            for worker in model_workers {
                string_tree.insert_text("", worker.url());
                token_tree.insert_tokens(&[], worker.url());
            }
        }
    }

    /// Add a single worker to the trees (incremental update)
    pub fn add_worker(&self, worker: &dyn Worker) {
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
            .or_insert_with(|| Arc::new(TokenTree::new()));
        token_tree.insert_tokens(&[], worker.url());
    }

    /// Add a worker by URL and model (for backward compatibility)
    pub fn add_worker_by_url(&self, url: &str, model_id: &str) {
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
            .or_insert_with(|| Arc::new(TokenTree::new()));
        token_tree.insert_tokens(&[], url);
    }

    /// Remove a worker from the trees
    ///
    /// Note: Currently a no-op. Stale entries are cleaned up by LRU eviction.
    /// Worker registry removes workers first, so routing will skip them anyway.
    /// TODO: Implement efficient remove_tenant in kv_index with reverse index.
    #[expect(
        clippy::unused_self,
        reason = "no-op stub; will use self once remove_tenant is implemented"
    )]
    pub fn remove_worker(&self, _worker: &dyn Worker) {
        // No-op: rely on LRU eviction to clean up stale entries
    }

    /// Remove a worker by URL (removes from all model trees for backward compatibility)
    ///
    /// Note: Currently a no-op. Stale entries are cleaned up by LRU eviction.
    /// TODO: Implement efficient remove_tenant in kv_index with reverse index.
    #[expect(
        clippy::unused_self,
        reason = "no-op stub; will use self once remove_tenant is implemented"
    )]
    pub fn remove_worker_by_url(&self, _url: &str) {
        // No-op: rely on LRU eviction to clean up stale entries
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
        min_load_idx: Option<usize>,
        model_id: &str,
    ) -> Option<usize> {
        // Log load balancing trigger (only compute worker loads if debug enabled)
        if tracing::enabled!(tracing::Level::DEBUG) {
            let worker_loads: Vec<(&str, usize)> =
                workers.iter().map(|w| (w.url(), w.load())).collect();
            debug!("Load balancing triggered | workers: {:?}", worker_loads);
        }

        // Shortest queue when imbalanced. The min-load index is gathered upstream
        // in select_worker with the (load, processed_requests, idx) tie-break
        // from #1714 (spreads load when decode outpaces prefill).
        let min_load_idx = min_load_idx?;

        let worker_url = workers[min_load_idx].url();

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

        // Increment processed counter
        workers[min_load_idx].increment_processed();

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
                    .or_insert_with(|| Arc::new(TokenTree::new()))
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

impl LoadBalancingPolicy for CacheAwarePolicy {
    fn select_worker(&self, workers: &[Arc<dyn Worker>], info: &SelectWorkerInfo) -> Option<usize> {
        let request_text = info.request_text;
        let request_tokens = info.tokens;

        // Single O(workers) gather: read each worker once via routing_state()
        // (status + load + processed under one ArcSwap guard), replacing the
        // former separate passes whose per-worker guard traffic dominated routing
        // CPU at scale. Collects healthy indices, load min/max, and the min-load
        // index; cache-hit tenant lookup is a hash-free scan over healthy_indices.
        let mut healthy_indices: Vec<usize> = Vec::with_capacity(workers.len());
        let mut min_load = usize::MAX;
        let mut max_load = 0usize;
        // Min-load worker, (load, processed_requests, idx) tie-break (#1714);
        // `processed` rides the same guard as `load`, so it is free here.
        let mut min_key: Option<(usize, usize, usize)> = None;
        let mut min_load_idx: Option<usize> = None;
        for (idx, worker) in workers.iter().enumerate() {
            let state = worker.routing_state();
            if state.healthy && state.can_execute {
                healthy_indices.push(idx);
                min_load = min_load.min(state.load);
                max_load = max_load.max(state.load);
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
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        // Determine the model for this set of workers (router pre-filters by model)
        // All workers should be from the same model
        let model_id = normalize_model_key(workers[healthy_indices[0]].model_id());

        // Abandon cache affinity for shortest-queue when the pool is imbalanced —
        // by request count (using the loads already gathered above), or (for
        // long-context workloads) by backend KV usage.
        if self.is_imbalanced(workers, &healthy_indices, min_load, max_load) {
            return self.select_worker_min_load(workers, info, min_load_idx, model_id);
        }

        // Cache-aware routing when balanced — three types (mutually exclusive):
        //   1. Event-driven: PositionalIndexer overlap scoring (gRPC + KV events)
        //   2. Approximate token tree: TokenTree prefix matching (gRPC, no events)
        //   3. Approximate string tree: Tree prefix matching (HTTP)
        if let Some(tokens) = request_tokens {
            if self.has_event_indexer(model_id) {
                self.select_worker_event_driven(
                    workers,
                    tokens,
                    &healthy_indices,
                    min_load_idx,
                    model_id,
                )
            } else {
                self.select_worker_with_tokens(
                    workers,
                    tokens,
                    &healthy_indices,
                    min_load_idx,
                    model_id,
                )
            }
        } else {
            let text = request_text.unwrap_or("");
            self.select_worker_with_text(workers, text, &healthy_indices, min_load_idx, model_id)
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

    /// Event-driven routing: PositionalIndexer overlap scoring (Type 1).
    ///
    /// Self-contained — when overlap is found, selects the worker with the best
    /// cache match. When no overlap (cold start, novel tokens, short request),
    /// falls back to min-load. Does NOT fall back to approximate token tree.
    fn select_worker_event_driven(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
        model_id: &str,
    ) -> Option<usize> {
        let guard = self.kv_monitor.read();
        let monitor = guard.as_ref()?;
        let indexer = monitor.get_indexer(model_id)?;

        // Per-model block_size: learned from events > config default
        let block_size = monitor
            .block_size(model_id)
            .unwrap_or(self.config.block_size);

        if let Some(idx) =
            Self::score_overlap(workers, tokens, healthy_indices, &indexer, block_size)
        {
            return Some(idx);
        }

        // No cache overlap — min-load fallback (min-load index gathered upstream)
        let min_idx = min_load_idx?;
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
    fn score_overlap(
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        indexer: &PositionalIndexer,
        block_size: usize,
    ) -> Option<usize> {
        let content_hashes = compute_request_content_hashes(tokens, block_size);
        if content_hashes.is_empty() {
            return None;
        }

        let overlap = indexer.find_matches(&content_hashes, false);
        if overlap.scores.is_empty() {
            return None;
        }

        // Select worker with best overlap among those that actually match.
        // Tie-break: lower load, then UNIFORMLY AT RANDOM. The previous final
        // key (smaller total tree size, then slice order) was a spreading
        // proxy but deterministic: equal-overlap equal-load workers herd onto
        // one index until the global tree-size ordering flips. Tree size is a
        // per-worker aggregate unrelated to this request's placement, and it
        // tracks event-stream health — an event-lagged worker looks "small"
        // and attracts the whole tie. Uniform random gives the same spreading
        // goal memorylessly; the next request's overlap scores restore
        // affinity to whichever worker actually cached the prefix.
        let mut best: Option<(u32, usize)> = None;
        let mut tied: Vec<usize> = Vec::new();
        for &idx in healthy_indices {
            let Some(score) = indexer
                .worker_id(workers[idx].url())
                .and_then(|id| overlap.scores.get(&id))
                .copied()
                .filter(|&s| s > 0)
            else {
                continue;
            };
            let load = workers[idx].load();
            match best {
                Some((best_score, best_load)) => {
                    if score > best_score || (score == best_score && load < best_load) {
                        best = Some((score, load));
                        tied.clear();
                        tied.push(idx);
                    } else if score == best_score && load == best_load {
                        tied.push(idx);
                    }
                }
                None => {
                    best = Some((score, load));
                    tied.push(idx);
                }
            }
        }
        let best_idx = match tied.len() {
            0 => return None,
            1 => tied[0],
            n => tied[rand::rng().random_range(0..n)],
        };

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

    /// Select worker using token-based tree (gRPC path)
    fn select_worker_with_tokens(
        &self,
        workers: &[Arc<dyn Worker>],
        tokens: &[u32],
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
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
            let result = tree.match_and_insert_with(tokens, |result| {
                let match_rate = if result.input_token_count == 0 {
                    0.0
                } else {
                    result.matched_token_count as f32 / result.input_token_count as f32
                };

                selected_idx = if match_rate > self.config.cache_threshold {
                    // Cache hit: scan healthy_indices for the tenant (hash-free;
                    // url() is cheap). "Healthy" excludes circuit-broken workers, so
                    // a CB-tripped tenant falls through to min-load (intended).
                    let tenant_url: &str = &result.tenant;
                    healthy_indices
                        .iter()
                        .copied()
                        .find(|&idx| workers[idx].url() == tenant_url)
                } else {
                    min_load_idx
                };

                // Insert for the selected worker (None => no insert, exactly
                // like the old `if let Some(idx)` guard around insert_tokens).
                selected_idx.map(|idx| workers[idx].url())
            });

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
                    self.hash_index
                        .entry(model_id.to_string())
                        .or_default()
                        .token_tree
                        .insert(kv_index::hash_token_path(tokens), matched_prefix);
                }
                workers[idx].increment_processed();
                return Some(idx);
            }

            // Selected worker no longer exists or unhealthy - fall back to first healthy
            // Stale entries will be cleaned up by LRU eviction
            healthy_indices.first().copied()
        } else {
            debug!(
                "Warning: No token tree found for model '{}', using random worker selection",
                model_id
            );
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            Some(healthy_indices[random_idx])
        }
    }

    /// Select worker using string-based tree (HTTP path)
    fn select_worker_with_text(
        &self,
        workers: &[Arc<dyn Worker>],
        text: &str,
        healthy_indices: &[usize],
        min_load_idx: Option<usize>,
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
                    // Cache hit: scan healthy_indices for the tenant (hash-free;
                    // url() is cheap). "Healthy" excludes circuit-broken workers, so
                    // a CB-tripped tenant falls through to min-load (intended).
                    let tenant_url: &str = &result.tenant;
                    healthy_indices
                        .iter()
                        .copied()
                        .find(|&idx| workers[idx].url() == tenant_url)
                } else {
                    min_load_idx
                };

                // Insert for the selected worker (None => no insert, exactly
                // like the old `if let Some(idx)` guard around insert_text).
                selected_idx.map(|idx| workers[idx].url())
            });

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
                }

                workers[idx].increment_processed();
                return Some(idx);
            }

            // Selected worker no longer exists or unhealthy - fall back to first healthy
            // Stale entries will be cleaned up by LRU eviction
            healthy_indices.first().copied()
        } else {
            debug!(
                "Warning: No string tree found for model '{}', using random worker selection",
                model_id
            );
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..healthy_indices.len());
            Some(healthy_indices[random_idx])
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

    use super::*;
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

    // ---- is_imbalanced: 3-term trigger (overload ∨ KV-spread ∨ count) ----

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

    /// Run the imbalance check the way `select_worker` does: fold the request-count
    /// bounds over the healthy workers (production gathers them in one pass), then
    /// call `is_imbalanced`.
    fn imbalanced(policy: &CacheAwarePolicy, workers: &[Arc<dyn Worker>]) -> bool {
        let healthy = all_healthy(workers);
        let (min_load, max_load) = healthy
            .iter()
            .fold((usize::MAX, 0usize), |(min, max), &idx| {
                let load = workers[idx].load();
                (min.min(load), max.max(load))
            });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };
        policy.is_imbalanced(workers, &healthy, min_load, max_load)
    }

    #[test]
    fn is_imbalanced_uniform_high_kv_does_not_fire() {
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
    fn is_imbalanced_one_hot_rest_idle_fires_via_spread() {
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
    fn is_imbalanced_overload_ceiling_fires_below_spread() {
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
    fn is_imbalanced_high_count_low_kv_caught_by_count() {
        // KV is even, so both KV triggers stay quiet — count must still catch it.
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
        // KV spread 0.0, max 0.3 → KV quiet; count 20 vs 0 → fire.
        assert!(
            imbalanced(&policy, &workers),
            "count spread must still trigger when KV utilization looks even"
        );
    }

    #[test]
    fn is_imbalanced_kv_disabled_by_default_ignores_snapshot() {
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
        let result = CacheAwarePolicy::score_overlap(&workers, &[1, 2, 3, 4], &[0, 1], &indexer, 4);
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
            let idx =
                CacheAwarePolicy::score_overlap(&workers, &[1, 2, 3, 4], &[0, 1], &indexer, 4)
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
        let result = CacheAwarePolicy::score_overlap(&workers, &[1, 2, 3], &[0], &indexer, 4);
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

        // Tokens must be >= PAGE_SIZE (16) to populate the tree; shorter
        // sequences are uncacheable and fall through to min-load.
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
}
