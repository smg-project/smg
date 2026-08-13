//! Composition root for the gateway's mesh sync adapters.
//!
//! [`MeshAdapters::start`] is the single call server startup makes to bring
//! the CRDT bridge online: it registers each adapter's namespace with the
//! right merge strategy, constructs the adapters against the gateway's
//! registries, and starts their inbound sync loops. It must run before the
//! mesh server's gossip starts so every remote op merges through its
//! registered engine.

use std::{fmt, sync::Arc};

use smg_mesh::{
    gossip::NodeStatus, ClusterState, MergeStrategy, MeshKV, StreamConfig, StreamRouting,
};
use tracing::debug;

use super::adapters::{
    tree_sync::{
        PeerList, RepairEntry, TreeRepairPage, TreeSyncAdapter, REPAIR_PAGE_PREFIX,
        REPAIR_REQUEST_PREFIX, TENANT_DELTA_PREFIX,
    },
    RateLimitSyncAdapter, WorkerSyncAdapter,
};
use crate::{
    policies::{CacheAwarePolicy, PolicyRegistry, TreeHandle, TreeKind},
    worker::WorkerRegistry,
};

/// Buffer budget for the broadcast tenant-delta namespace (`td:`). Undersized
/// buffers FIFO-evict pending deltas before drain — the receiver then repairs
/// through the (much more expensive) `tree:req:`/`tree:page:` path. Sized to
/// cover a few gossip rounds of steady-state routing traffic.
const TD_BUFFER_BYTES: usize = 4 * 1024 * 1024;

/// Buffer budget for the targeted repair-request namespace. Requests
/// carry small per-session metadata (session id, requester + target
/// peer id, model id, tree kind, an opaque resume cursor, and a
/// reason enum) so this stays modest.
const REPAIR_REQ_BUFFER_BYTES: usize = 256 * 1024;

/// Buffer budget for the targeted repair-page namespace. Cold-start replay
/// pages can span megabytes per session — keep this generous so a slow peer
/// draining pages doesn't drop earlier ones.
const REPAIR_PAGE_BUFFER_BYTES: usize = 32 * 1024 * 1024;

/// Owns the started mesh sync adapters. Mesh on means every adapter here is
/// constructed, its namespace registered, and its inbound loop running —
/// mesh off is represented by the absence of the whole struct.
#[derive(Debug)]
pub struct MeshAdapters {
    worker: Arc<WorkerSyncAdapter>,
    rate_limit: Arc<RateLimitSyncAdapter>,
    tree: Arc<TreeSyncAdapter>,
}

impl MeshAdapters {
    /// Register the `worker:` (last-writer-wins) and `rl:` (epoch-max-wins)
    /// CRDT namespaces plus the `td:` / `tree:req:` / `tree:page:` stream
    /// namespaces, construct the adapters, and start their inbound sync
    /// loops. One call because the adapters' `start` methods are not
    /// idempotent (each call spawns another subscription task).
    ///
    /// MUST be called before `MeshServer::start` spawns gossip: a remote op
    /// arriving for an unregistered prefix would merge through the default
    /// last-writer-wins engine with the wrong semantics, and a remote tenant
    /// delta arriving before the tree adapter subscribes would be dropped.
    ///
    /// After construction the tree adapter is attached to `policy_registry`,
    /// which propagates it (and flips `populate_hash_index` on) to every
    /// existing and future [`CacheAwarePolicy`] instance.
    ///
    /// # Panics
    ///
    /// Panics if any adapter's `start` invariants are violated — the
    /// concrete sources are a double `start` call on the same [`MeshKV`]
    /// (each adapter registers exactly one drain per prefix) and an
    /// empty `node_name` (asserted by every adapter); the rate-limit
    /// adapter additionally rejects a `node_name` containing `':'`
    /// because it is the shard-key separator.
    pub fn start(
        mesh_kv: &MeshKV,
        node_name: String,
        worker_registry: Arc<WorkerRegistry>,
        cluster_state: ClusterState,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Arc<Self> {
        let worker_ns = mesh_kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
        let rl_ns = mesh_kv.configure_crdt_prefix("rl:", MergeStrategy::EpochMaxWins);
        let td_ns = mesh_kv.configure_stream_prefix(
            TENANT_DELTA_PREFIX,
            StreamConfig {
                max_buffer_bytes: TD_BUFFER_BYTES,
                routing: StreamRouting::Broadcast,
            },
        );
        let repair_req_ns = mesh_kv.configure_stream_prefix(
            REPAIR_REQUEST_PREFIX,
            StreamConfig {
                max_buffer_bytes: REPAIR_REQ_BUFFER_BYTES,
                routing: StreamRouting::Targeted,
            },
        );
        let repair_page_ns = mesh_kv.configure_stream_prefix(
            REPAIR_PAGE_PREFIX,
            StreamConfig {
                max_buffer_bytes: REPAIR_PAGE_BUFFER_BYTES,
                routing: StreamRouting::Targeted,
            },
        );
        let worker = WorkerSyncAdapter::new(worker_ns, worker_registry);
        let rate_limit = RateLimitSyncAdapter::new(rl_ns, node_name.clone());
        let peers: Arc<dyn PeerList> =
            Arc::new(ClusterStatePeerList::new(cluster_state, node_name.clone()));
        let tree_handle: Arc<dyn TreeHandle> =
            Arc::new(PolicyRegistryTreeHandle::new(policy_registry.clone()));
        let tree = TreeSyncAdapter::new(
            td_ns,
            repair_req_ns,
            repair_page_ns,
            tree_handle,
            peers,
            node_name,
        );
        worker.start();
        rate_limit.start();
        tree.start();
        // Only after the adapter is subscribed to the `td:` stream do we
        // let policies start writing to it — otherwise the very first
        // hot-path delta races the subscription setup and is dropped.
        policy_registry.set_mesh_tree_sync(Some(tree.clone()));
        Arc::new(Self {
            worker,
            rate_limit,
            tree,
        })
    }

    /// Worker sync adapter.
    pub fn worker(&self) -> &Arc<WorkerSyncAdapter> {
        &self.worker
    }

    /// Rate-limit sync adapter.
    pub fn rate_limit(&self) -> &Arc<RateLimitSyncAdapter> {
        &self.rate_limit
    }

    /// Tree sync adapter (cache-aware tenant deltas + repair).
    pub fn tree(&self) -> &Arc<TreeSyncAdapter> {
        &self.tree
    }
}

/// [`PeerList`] impl backed by the mesh [`ClusterState`]. Reports every
/// node whose `NodeStatus == Alive` other than this node. Read on the
/// gossip tick (repair scan), so a read-lock is fine.
struct ClusterStatePeerList {
    state: ClusterState,
    self_name: String,
}

impl ClusterStatePeerList {
    fn new(state: ClusterState, self_name: String) -> Self {
        Self { state, self_name }
    }
}

impl fmt::Debug for ClusterStatePeerList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClusterStatePeerList")
            .field("self_name", &self.self_name)
            .finish_non_exhaustive()
    }
}

impl PeerList for ClusterStatePeerList {
    fn alive_peers(&self) -> Vec<String> {
        let state = self.state.read();
        state
            .values()
            .filter(|node| node.status == NodeStatus::Alive as i32 && node.name != self.self_name)
            .map(|node| node.name.clone())
            .collect()
    }
}

/// [`TreeHandle`] that dispatches per-`model_id` to whichever
/// [`CacheAwarePolicy`] the registry has for that model. Non-cache-aware
/// policies (or absent models) return the "unknown hash / no local tree"
/// defaults so the adapter can request repair or skip the update.
struct PolicyRegistryTreeHandle {
    registry: Arc<PolicyRegistry>,
}

impl PolicyRegistryTreeHandle {
    fn new(registry: Arc<PolicyRegistry>) -> Self {
        Self { registry }
    }

    /// Visit every distinct [`CacheAwarePolicy`] the registry could
    /// dispatch requests through for `model_id`, in
    /// most-specific-first order (per-model → default → PD/EPD legs),
    /// invoking `f` on each. The `Arc<dyn ...>` is held for the
    /// duration of each `f` call, so downcast borrows are safe
    /// against concurrent registry churn.
    ///
    /// Emits a single `debug!` when the chain contains no
    /// cache-aware policy so operators can distinguish a legitimate
    /// empty result from "this node is not authoritative for that
    /// model" — the two look identical on the wire otherwise, and
    /// the second case can drive spurious repair traffic.
    fn for_each_cache_aware(&self, model_id: &str, mut f: impl FnMut(&CacheAwarePolicy)) {
        let policies = self.registry.policies_for_model(model_id);
        let mut hit = false;
        for policy in &policies {
            if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                hit = true;
                f(cache_aware);
            }
        }
        if !hit {
            debug!(
                model_id,
                candidate_policies = policies.len(),
                "tree-sync bridge: no cache-aware policy for model on this node",
            );
        }
    }
}

impl fmt::Debug for PolicyRegistryTreeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolicyRegistryTreeHandle")
            .finish_non_exhaustive()
    }
}

impl TreeHandle for PolicyRegistryTreeHandle {
    fn apply_known_remote_insert(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
        node_hash: u64,
        worker_url: &str,
    ) -> bool {
        // First cache-aware policy in the chain that knows this
        // hash wins; further policies are skipped because the delta
        // has already been applied where it belongs.
        let mut applied = false;
        self.for_each_cache_aware(model_id, |policy| {
            if applied {
                return;
            }
            if policy.apply_known_remote_insert(model_id, tree_kind, node_hash, worker_url) {
                applied = true;
            }
        });
        applied
    }

    fn open_repair_stream(
        &self,
        model_id: &str,
        tree_kind: TreeKind,
    ) -> Option<Box<dyn Iterator<Item = RepairEntry> + Send>> {
        // Return the first non-empty stream in the chain: the
        // returned iterator is self-contained (underlying tree is
        // Arc-shared) so the policy borrow can end when the closure
        // returns.
        let mut stream: Option<Box<dyn Iterator<Item = RepairEntry> + Send>> = None;
        self.for_each_cache_aware(model_id, |policy| {
            if stream.is_some() {
                return;
            }
            stream = policy.open_repair_stream(model_id, tree_kind);
        });
        stream
    }

    fn apply_repair_page(&self, page: &TreeRepairPage) -> usize {
        // Seed every cache-aware policy in the chain from the same
        // page: in PD/EPD deployments both the prefill leg and the
        // decode leg maintain their own routing tree and each needs
        // the peer's state; delivering the page once per leg is
        // cheap (per-entry `insert_text` / `insert_tokens`).
        let mut total: usize = 0;
        self.for_each_cache_aware(&page.model_id, |policy| {
            total = total.saturating_add(policy.apply_repair_page(page));
        });
        total
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, time::Duration};

    use openai_protocol::worker::HealthCheckConfig;
    use parking_lot::RwLock;
    use smg_mesh::{gossip::NodeState, WorkerState};
    use tokio::time::sleep;

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        policies::SelectWorkerInfo,
        worker::{BasicWorkerBuilder, Worker, WorkerType},
    };

    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        }
    }

    fn empty_cluster_state() -> ClusterState {
        Arc::new(RwLock::new(BTreeMap::new()))
    }

    fn started(mesh: &MeshKV) -> Arc<MeshAdapters> {
        MeshAdapters::start(
            mesh,
            "node-a".into(),
            Arc::new(WorkerRegistry::new()),
            empty_cluster_state(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        )
    }

    #[tokio::test]
    async fn start_wires_worker_inbound_end_to_end() {
        let mesh = MeshKV::new("node-a".into());
        let registry = Arc::new(WorkerRegistry::new());
        let adapters = MeshAdapters::start(
            &mesh,
            "node-a".into(),
            registry.clone(),
            empty_cluster_state(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        );

        // A put through the adapter echoes back through the namespace
        // subscription, exercising the registered prefix and the live
        // inbound loop end to end.
        let state = WorkerState {
            worker_id: "w1".into(),
            model_id: "llama-3".into(),
            url: "http://remote:8080".into(),
            health: true,
            load: 0.0,
            version: 1,
            spec: vec![],
        };
        adapters.worker().on_worker_changed("w1", &state);

        for _ in 0..100 {
            if registry.get_by_url("http://remote:8080").is_some() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("inbound worker sync loop is not running");
    }

    #[tokio::test]
    async fn rl_namespace_uses_epoch_max_wins() {
        let mesh = MeshKV::new("node-a".into());
        let adapters = started(&mesh);

        // Under EpochMaxWins a lower-epoch write cannot rewind the shard;
        // under a mis-registered LWW engine the later write would win and
        // the aggregate would read 100.
        adapters.rate_limit().sync_counter("global", 2, 5);
        adapters.rate_limit().sync_counter("global", 1, 100);
        assert_eq!(adapters.rate_limit().get_aggregate("global"), 5);
    }

    #[tokio::test]
    async fn tree_adapter_registers_and_flips_populate_flag() {
        let mesh = MeshKV::new("node-a".into());
        let policy_registry = Arc::new(PolicyRegistry::new(cache_aware_policy_config()));
        let _adapters = MeshAdapters::start(
            &mesh,
            "node-a".into(),
            Arc::new(WorkerRegistry::new()),
            empty_cluster_state(),
            policy_registry.clone(),
        );

        // A fresh model → policy created on demand should inherit the
        // adapter + populate flag from the registry.
        let policy = policy_registry.get_policy_or_default("llama-3");
        let cache_aware = policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("cache_aware default policy");
        assert!(
            cache_aware.should_populate_hash_index_for_test(),
            "wiring must flip populate_hash_index when the adapter is attached",
        );
    }

    #[tokio::test]
    async fn peer_list_reports_alive_only() {
        let state: ClusterState = Arc::new(RwLock::new(BTreeMap::from([
            (
                "node-a".to_string(),
                NodeState {
                    name: "node-a".into(),
                    address: "127.0.0.1:9000".into(),
                    status: NodeStatus::Alive as i32,
                    version: 1,
                    metadata: Default::default(),
                },
            ),
            (
                "node-b".to_string(),
                NodeState {
                    name: "node-b".into(),
                    address: "127.0.0.1:9001".into(),
                    status: NodeStatus::Alive as i32,
                    version: 1,
                    metadata: Default::default(),
                },
            ),
            (
                "node-c".to_string(),
                NodeState {
                    name: "node-c".into(),
                    address: "127.0.0.1:9002".into(),
                    status: NodeStatus::Down as i32,
                    version: 1,
                    metadata: Default::default(),
                },
            ),
        ])));
        let peers = ClusterStatePeerList::new(state, "node-a".into());
        assert_eq!(peers.alive_peers(), vec!["node-b".to_string()]);
    }

    #[tokio::test]
    #[should_panic(expected = "already configured")]
    async fn start_panics_on_second_call() {
        let mesh = MeshKV::new("node-a".into());
        let _adapters = started(&mesh);
        let _again = started(&mesh);
    }

    #[tokio::test]
    #[should_panic(expected = "must not contain ':'")]
    async fn start_panics_on_colon_node_name() {
        let mesh = MeshKV::new("node-a".into());
        let _ = MeshAdapters::start(
            &mesh,
            "node:a".into(),
            Arc::new(WorkerRegistry::new()),
            empty_cluster_state(),
            Arc::new(PolicyRegistry::new(PolicyConfig::Random)),
        );
    }

    fn cache_aware_policy_config() -> PolicyConfig {
        PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 60,
            max_tree_size: 128,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
        }
    }

    /// The atomic setter must reach model policies that already exist
    /// before wiring runs — not just future-created ones. Regression
    /// guard for the propagation path in `set_mesh_tree_sync`.
    #[tokio::test]
    async fn start_propagates_to_preexisting_model_policies() {
        let policy_registry = Arc::new(PolicyRegistry::new(cache_aware_policy_config()));
        // Create a per-model cache-aware policy BEFORE wiring runs, so
        // it lives in `model_policies` at attach time.
        let preexisting = policy_registry.get_policy_or_default("qwen2");
        let cache_aware = preexisting
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("cache_aware model policy");
        assert!(
            !cache_aware.should_populate_hash_index_for_test(),
            "populate flag defaults off before wiring",
        );

        let mesh = MeshKV::new("node-a".into());
        let _adapters = MeshAdapters::start(
            &mesh,
            "node-a".into(),
            Arc::new(WorkerRegistry::new()),
            empty_cluster_state(),
            policy_registry.clone(),
        );

        assert!(
            cache_aware.should_populate_hash_index_for_test(),
            "wiring must propagate the adapter to already-existing model policies",
        );
    }

    /// `set_mesh_tree_sync(None)` must undo both flips atomically so
    /// the hot path stops emitting deltas and stops populating the
    /// index — otherwise we'd keep growing an index nothing reads.
    #[tokio::test]
    async fn detach_clears_populate_flag() {
        let policy_registry = Arc::new(PolicyRegistry::new(cache_aware_policy_config()));
        let mesh = MeshKV::new("node-a".into());
        let _adapters = MeshAdapters::start(
            &mesh,
            "node-a".into(),
            Arc::new(WorkerRegistry::new()),
            empty_cluster_state(),
            policy_registry.clone(),
        );
        let policy = policy_registry.get_policy_or_default("llama-3");
        let cache_aware = policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("cache_aware default policy");
        assert!(cache_aware.should_populate_hash_index_for_test());

        policy_registry.set_mesh_tree_sync(None);
        assert!(
            !cache_aware.should_populate_hash_index_for_test(),
            "detach must clear the populate flag",
        );
    }

    /// Every producer-side hash-index write must publish a matching
    /// `TreeDelta` — this is what closes #1578. Deleting either
    /// `sync_local_insert` call at the two `select_worker_*` sites
    /// makes this test fail (the pending buffer stays empty on the
    /// exercised branch), which is the regression guard we lacked.
    #[tokio::test]
    async fn producer_hook_publishes_delta_on_select_worker() {
        use crate::worker::UNKNOWN_MODEL_ID;

        let policy_registry = Arc::new(PolicyRegistry::new(cache_aware_policy_config()));
        let mesh = MeshKV::new("node-a".into());
        let adapters = MeshAdapters::start(
            &mesh,
            "node-a".into(),
            Arc::new(WorkerRegistry::new()),
            empty_cluster_state(),
            policy_registry.clone(),
        );

        // Workers with no model_id → select_worker derives
        // UNKNOWN_MODEL_ID; assert on the same bucket the delta
        // lands under.
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

        let policy = policy_registry.get_policy_or_default(UNKNOWN_MODEL_ID);
        let cache_aware = policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("cache_aware policy");
        cache_aware.init_workers(&workers);
        assert_eq!(
            adapters
                .tree()
                .pending_delta_count_for_test(UNKNOWN_MODEL_ID),
            0,
            "no delta should be pending before any request",
        );

        // Drive a string request — populates the string tree and
        // must call `sync_local_insert`.
        let _ = policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                request_text: Some("the quick brown fox jumps over the lazy dog"),
                ..Default::default()
            },
        );
        let after_string = adapters
            .tree()
            .pending_delta_count_for_test(UNKNOWN_MODEL_ID);
        assert!(
            after_string >= 1,
            "string select_worker must publish at least one TreeDelta (got {after_string})",
        );

        // Drive a token request — populates the token tree and
        // must call `sync_local_insert` on the other branch.
        let tokens: Vec<u32> = (0..32).collect();
        let _ = policy.select_worker(
            &workers,
            &SelectWorkerInfo {
                tokens: Some(&tokens),
                ..Default::default()
            },
        );
        let after_token = adapters
            .tree()
            .pending_delta_count_for_test(UNKNOWN_MODEL_ID);
        assert!(
            after_token > after_string,
            "token select_worker must publish an additional TreeDelta \
             (before {after_string}, after {after_token})",
        );
    }

    /// The `TreeHandle` bridge must return the documented fallback
    /// values (`false` / `None` / `0`) when the model has no
    /// cache-aware policy in the chain — otherwise the tree adapter
    /// cannot distinguish "delta legitimately unknown" from "no
    /// policy on this node" and drives spurious repair traffic.
    #[tokio::test]
    async fn bridge_returns_fallbacks_when_no_cache_aware_policy() {
        use crate::mesh::adapters::tree_sync::TreeRepairPage;

        let policy_registry = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
        let handle = PolicyRegistryTreeHandle::new(policy_registry);

        assert!(
            !handle.apply_known_remote_insert("no-such-model", TreeKind::String, 42, "http://w"),
            "apply_known_remote_insert must return false when no cache-aware policy is reachable",
        );
        assert!(
            handle
                .open_repair_stream("no-such-model", TreeKind::String)
                .is_none(),
            "open_repair_stream must return None when no cache-aware policy is reachable",
        );
        let page = TreeRepairPage {
            session_id: uuid::Uuid::nil(),
            model_id: "no-such-model".into(),
            tree_kind: TreeKind::String,
            page_index: 0,
            entries: Vec::new(),
            next_cursor: None,
            is_last: true,
        };
        assert_eq!(
            handle.apply_repair_page(&page),
            0,
            "apply_repair_page must return 0 when no cache-aware policy is reachable",
        );
    }

    /// A cache-aware default policy (no per-model entry, no PD legs)
    /// must be reachable through the bridge — that's the shape a
    /// single-model non-PD deployment has, and the fallback CodeRabbit
    /// flagged as missing.
    #[tokio::test]
    async fn bridge_dispatches_through_default_policy() {
        use kv_index::hash_node_path;

        use crate::mesh::adapters::tree_sync::TreeRepairPage;

        let policy_registry = Arc::new(PolicyRegistry::new(cache_aware_policy_config()));
        // Do NOT touch `get_policy_or_default("m")` — we want the
        // default policy path, not a per-model entry. Manually seed
        // the default policy's hash_index so `apply_known_remote_insert`
        // has something to resolve.
        let default_policy = policy_registry.get_default_policy();
        let cache_aware = default_policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("default policy is cache-aware");
        cache_aware.set_populate_hash_index_for_test_true();
        let text = "hello world";
        let path_hash = hash_node_path(text);
        cache_aware.seed_hash_index_for_test(
            crate::worker::UNKNOWN_MODEL_ID,
            TreeKind::String,
            path_hash,
            text,
        );

        let handle = PolicyRegistryTreeHandle::new(policy_registry);
        assert!(
            handle.apply_known_remote_insert(
                crate::worker::UNKNOWN_MODEL_ID,
                TreeKind::String,
                path_hash,
                "http://w1:8000",
            ),
            "bridge must reach default_policy for models without a per-model entry",
        );

        // apply_repair_page over the same model should reach the
        // default policy too — empty page trivially returns 0
        // entries applied, but the fact that it does not panic
        // proves the chain walked.
        let page = TreeRepairPage {
            session_id: uuid::Uuid::nil(),
            model_id: crate::worker::UNKNOWN_MODEL_ID.into(),
            tree_kind: TreeKind::String,
            page_index: 0,
            entries: Vec::new(),
            next_cursor: None,
            is_last: true,
        };
        assert_eq!(handle.apply_repair_page(&page), 0);
    }
}
