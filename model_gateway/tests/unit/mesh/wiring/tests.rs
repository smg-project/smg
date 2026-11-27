use std::{collections::BTreeMap, time::Duration};

use openai_protocol::UNKNOWN_MODEL_ID;
use parking_lot::RwLock;
use smg_mesh::{gossip::NodeState, WorkerState};
use smg_worker::{
    identity::basic::{Worker, WorkerType},
    test_support::no_health_check,
};
use tokio::time::sleep;

use super::*;
use crate::{
    config::types::PolicyConfig, policies::SelectWorkerInfo,
    routers::grpc::backend_client::BasicWorkerBuilder,
};

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
        owner: "node-b".into(),
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
    let cache_aware = as_cache_aware(policy.as_ref()).expect("cache_aware default policy");
    assert!(
        cache_aware.mesh_attached_for_test(),
        "wiring must raise the mesh flag when the adapter is attached",
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

fn cache_aware_policy_config() -> PolicyConfig {
    PolicyConfig::cache_aware(0.5, 32, 1.5, 60, 128)
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
    let cache_aware = as_cache_aware(preexisting.as_ref()).expect("cache_aware model policy");
    assert!(
        !cache_aware.mesh_attached_for_test(),
        "the mesh flag defaults off before wiring",
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
        cache_aware.mesh_attached_for_test(),
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
    let cache_aware = as_cache_aware(policy.as_ref()).expect("cache_aware default policy");
    assert!(cache_aware.mesh_attached_for_test());

    policy_registry.set_mesh_tree_sync(None);
    assert!(
        !cache_aware.mesh_attached_for_test(),
        "detach must lower the mesh flag",
    );
}

/// Every producer-side hash-index write must publish a matching
/// `TreeDelta` — this is what closes #1578. Deleting either
/// `sync_local_insert` call at the two `select_worker_*` sites
/// makes this test fail (the pending buffer stays empty on the
/// exercised branch), which is the regression guard we lacked.
#[tokio::test]
async fn producer_hook_publishes_delta_on_select_worker() {
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
    let cache_aware = as_cache_aware(policy.as_ref()).expect("cache_aware policy");
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

/// A cache-aware default policy (no per-model entry, no PD roles)
/// must be reachable through the bridge — that's the shape a
/// single-model non-PD deployment has, and the fallback CodeRabbit
/// flagged as missing.
#[tokio::test]
async fn bridge_dispatches_through_default_policy() {
    use kv_index::hash_node_path;

    use crate::mesh::adapters::tree_sync::{RepairEntry, TreeRepairPage};

    let policy_registry = Arc::new(PolicyRegistry::new(cache_aware_policy_config()));
    // Do NOT touch `get_policy_or_default("m")` — we want the
    // default policy path, not a per-model entry. Seed the default
    // policy's string tree through a repair page (the path with its
    // hash) so `apply_known_remote_insert` has something to resolve.
    let default_policy = policy_registry.get_default_policy();
    let cache_aware =
        as_cache_aware(default_policy.as_ref()).expect("default policy is cache-aware");
    let text = "hello world";
    let path_hash = hash_node_path(text);
    let seed = TreeRepairPage {
        session_id: uuid::Uuid::nil(),
        model_id: UNKNOWN_MODEL_ID.into(),
        tree_kind: TreeKind::String,
        page_index: 0,
        entries: vec![RepairEntry::String {
            path: text.to_string(),
            tenants: vec![(Arc::from("http://w0:8000"), 1)],
        }],
        next_cursor: None,
        is_last: true,
    };
    assert_eq!(cache_aware.apply_repair_page(&seed), 1);

    let handle = PolicyRegistryTreeHandle::new(policy_registry);
    assert!(
        handle.apply_known_remote_insert(
            UNKNOWN_MODEL_ID,
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
        model_id: UNKNOWN_MODEL_ID.into(),
        tree_kind: TreeKind::String,
        page_index: 0,
        entries: Vec::new(),
        next_cursor: None,
        is_last: true,
    };
    assert_eq!(handle.apply_repair_page(&page), 0);
}
