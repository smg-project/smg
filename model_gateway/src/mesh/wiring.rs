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
    WorkerSyncAdapter,
};
use crate::{
    policies::{as_cache_aware, CacheAwarePolicy, PolicyRegistry, TreeHandle, TreeKind},
    routers::common::snapshot::WorkerRegistry,
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
    tree: Arc<TreeSyncAdapter>,
}

impl MeshAdapters {
    /// Register the `worker:` (last-writer-wins) CRDT namespace plus the
    /// `td:` / `tree:req:` / `tree:page:` stream namespaces, construct the
    /// adapters, and start their inbound sync loops. One call because the
    /// adapters' `start` methods are not idempotent (each call spawns another
    /// subscription task).
    ///
    /// MUST be called before `MeshServer::start` spawns gossip: a remote op
    /// arriving for an unregistered prefix would merge through the default
    /// last-writer-wins engine with the wrong semantics, and a remote tenant
    /// delta arriving before the tree adapter subscribes would be dropped.
    ///
    /// After construction the tree adapter is attached to `policy_registry`,
    /// which propagates it (and raises the mesh flag) to every
    /// existing and future [`CacheAwarePolicy`] instance.
    ///
    /// # Panics
    ///
    /// Panics if any adapter's `start` invariants are violated — the
    /// concrete sources are a double `start` call on the same [`MeshKV`]
    /// (each adapter registers exactly one drain per prefix) and an
    /// empty `node_name` (asserted by every adapter).
    pub fn start(
        mesh_kv: &MeshKV,
        node_name: String,
        worker_registry: Arc<WorkerRegistry>,
        cluster_state: ClusterState,
        policy_registry: Arc<PolicyRegistry>,
    ) -> Arc<Self> {
        let worker_ns = mesh_kv.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
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
        let worker = WorkerSyncAdapter::new(
            worker_ns,
            worker_registry,
            node_name.clone(),
            cluster_state.clone(),
        );
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
        tree.start();
        // Only after the adapter is subscribed to the `td:` stream do we
        // let policies start writing to it — otherwise the very first
        // hot-path delta races the subscription setup and is dropped.
        policy_registry.set_mesh_tree_sync(Some(tree.clone()));
        Arc::new(Self { worker, tree })
    }

    /// Worker sync adapter.
    pub fn worker(&self) -> &Arc<WorkerSyncAdapter> {
        &self.worker
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
    /// most-specific-first order (per-model → default → PD/EPD roles),
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
            if let Some(cache_aware) = as_cache_aware(policy.as_ref()) {
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
        // page: in PD/EPD deployments both the prefill role and the
        // decode role maintain their own routing tree and each needs
        // the peer's state; delivering the page once per role is
        // cheap (per-entry `insert_text` / `insert_tokens`).
        let mut total: usize = 0;
        self.for_each_cache_aware(&page.model_id, |policy| {
            total = total.saturating_add(policy.apply_repair_page(page));
        });
        total
    }
}

#[cfg(test)]
#[path = "../../tests/unit/mesh/wiring/tests.rs"]
mod tests;
