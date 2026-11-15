use std::{collections::BTreeMap, time::Duration};

use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg_mesh::{gossip::NodeState, MergeStrategy, MeshKV};
use tokio::{sync::mpsc::error::TryRecvError, time::sleep};
use tracing_test::traced_test;

use super::*;
use crate::worker::{BasicWorkerBuilder, WorkerType};

fn worker_namespace(mesh: &MeshKV) -> Arc<CrdtNamespace> {
    mesh.configure_crdt_prefix(PREFIX, MergeStrategy::LastWriterWins)
}

fn sample_state(worker_id: &str, url: &str) -> WorkerState {
    WorkerState {
        worker_id: worker_id.into(),
        model_id: "llama-3".into(),
        url: url.into(),
        health: true,
        load: 0.25,
        version: 1,
        spec: vec![],
        owner: "node-b".into(),
    }
}

fn owned_state(worker_id: &str, url: &str, owner: &str) -> WorkerState {
    WorkerState {
        owner: owner.into(),
        ..sample_state(worker_id, url)
    }
}

fn empty_cluster() -> ClusterState {
    Arc::new(parking_lot::RwLock::new(BTreeMap::new()))
}

fn cluster_with(nodes: &[(&str, NodeStatus)]) -> ClusterState {
    let state = nodes
        .iter()
        .map(|(name, status)| {
            (
                (*name).to_string(),
                NodeState {
                    name: (*name).to_string(),
                    address: "127.0.0.1:9000".into(),
                    status: *status as i32,
                    version: 1,
                    metadata: Default::default(),
                },
            )
        })
        .collect();
    Arc::new(parking_lot::RwLock::new(state))
}

fn set_status(cluster: &ClusterState, name: &str, status: NodeStatus) {
    cluster.write().get_mut(name).expect("node present").status = status as i32;
}

/// An adapter for `node-a` over an empty membership view (every owner
/// unknown, hence live).
fn adapter(ns: Arc<CrdtNamespace>, registry: Arc<WorkerRegistry>) -> Arc<WorkerSyncAdapter> {
    adapter_as(ns, registry, "node-a")
}

/// An adapter under another node's name over an empty membership view:
/// the two-life tests run an owner as `node-p` beside its peer as `node-s`.
fn adapter_as(
    ns: Arc<CrdtNamespace>,
    registry: Arc<WorkerRegistry>,
    node: &str,
) -> Arc<WorkerSyncAdapter> {
    WorkerSyncAdapter::new(ns, registry, node.into(), empty_cluster())
}

fn adapter_in(
    ns: Arc<CrdtNamespace>,
    registry: Arc<WorkerRegistry>,
    cluster: ClusterState,
) -> Arc<WorkerSyncAdapter> {
    WorkerSyncAdapter::new(ns, registry, "node-a".into(), cluster)
}

/// A fresh `node-a` bed: the worker namespace on a new mesh, an empty
/// registry and the adapter over them with an empty membership view. The
/// mesh handle is dropped here: the namespace holds the store and the
/// subscriber registry by `Arc`, and no test read the handle again.
fn setup() -> (
    Arc<CrdtNamespace>,
    Arc<WorkerRegistry>,
    Arc<WorkerSyncAdapter>,
) {
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);
    let registry = Arc::new(WorkerRegistry::new());
    let adapter = adapter(ns.clone(), registry.clone());
    (ns, registry, adapter)
}

fn local_worker(url: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("llama-3"))
            .build(),
    )
}

/// The `version` the store holds for `key`, when the key is live and
/// decodes.
fn version_of(ns: &CrdtNamespace, key: &str) -> Option<u64> {
    ns.get(key)
        .and_then(|bytes| decode_worker_state(&bytes).ok())
        .map(|state| state.version)
}

/// Poll until `cond` is true or ~1s elapses.
async fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
    for _ in 0..100 {
        if cond() {
            return true;
        }
        sleep(Duration::from_millis(10)).await;
    }
    false
}

#[tokio::test]
async fn on_worker_changed_writes_decodable_state() {
    let (ns, _registry, adapter) = setup();

    let state = sample_state("w1", "http://worker-a:8080");
    adapter.on_worker_changed("w1", &state);

    let raw = ns.get("worker:w1").expect("adapter wrote through to CRDT");
    let decoded: WorkerState = bincode::deserialize(&raw).unwrap();
    assert_eq!(decoded, state);
}

#[tokio::test]
async fn on_worker_removed_tombstones_the_entry() {
    let (ns, _registry, adapter) = setup();

    let state = sample_state("w1", "http://worker-a:8080");
    adapter.on_worker_changed("w1", &state);
    assert!(ns.get("worker:w1").is_some());

    adapter.on_worker_removed("w1");
    assert!(
        ns.get("worker:w1").is_none(),
        "tombstone must hide the prior value from readers"
    );
}

#[tokio::test]
async fn start_routes_remote_state_into_registry() {
    // Two adapters over one store mimic a remote node's write
    // (publisher) arriving at a local subscriber. The underlying
    // store is shared so the publisher's put fires the
    // subscriber adapter's subscription.
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);

    let registry = Arc::new(WorkerRegistry::new());
    let subscriber = adapter(ns.clone(), registry.clone());
    subscriber.start();

    let publisher_registry = Arc::new(WorkerRegistry::new());
    let publisher = adapter(ns, publisher_registry);
    publisher.on_worker_changed("w1", &sample_state("w1", "http://remote:8080"));

    // Subscription fanout is async; poll briefly.
    for _ in 0..20 {
        if registry.get_by_url("http://remote:8080").is_some() {
            return;
        }
        sleep(Duration::from_millis(10)).await;
    }
    panic!("registry did not see the remote worker");
}

#[tokio::test]
async fn start_ignores_malformed_payload() {
    // A bad payload should not propagate into the registry and
    // must not kill the spawned task — a subsequent valid write
    // still lands.
    let (ns, registry, adapter) = setup();
    adapter.start();

    ns.put("worker:bogus", b"not-bincode".to_vec());
    sleep(Duration::from_millis(20)).await;
    assert!(registry.get_by_url("http://remote:8080").is_none());

    let good = sample_state("w1", "http://remote:8080");
    ns.put(
        "worker:w1",
        bincode::serialize(&good).expect("state serializes"),
    );
    for _ in 0..20 {
        if registry.get_by_url("http://remote:8080").is_some() {
            return;
        }
        sleep(Duration::from_millis(10)).await;
    }
    panic!("subscription task aborted after a bad payload");
}

#[tokio::test]
async fn start_backfills_preexisting_entries() {
    // Rolling-restart scenario: the `worker:` namespace already
    // contains gossiped state before `start` runs. The adapter
    // must backfill the registry on spawn, not wait for the next
    // live event.
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);

    let seeded = sample_state("w-seeded", "http://seeded:8080");
    ns.put(
        "worker:w-seeded",
        bincode::serialize(&seeded).expect("state serializes"),
    );

    let registry = Arc::new(WorkerRegistry::new());
    let adapter = adapter(ns, registry.clone());
    adapter.start();

    for _ in 0..20 {
        if registry.get_by_url("http://seeded:8080").is_some() {
            return;
        }
        sleep(Duration::from_millis(10)).await;
    }
    panic!("registry did not see the pre-existing worker");
}

#[tokio::test]
#[should_panic(expected = "WorkerSyncAdapter requires a namespace scoped to `worker:`")]
async fn new_rejects_wrong_prefix() {
    let mesh = MeshKV::new("node-a".into());
    let ns = mesh.configure_crdt_prefix("policy:", MergeStrategy::LastWriterWins);
    let registry = Arc::new(WorkerRegistry::new());
    let _ = WorkerSyncAdapter::new(ns, registry, "node-a".into(), empty_cluster());
}

#[tokio::test]
#[should_panic(expected = "non-empty node name")]
async fn new_rejects_empty_node_name() {
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);
    let registry = Arc::new(WorkerRegistry::new());
    let _ = WorkerSyncAdapter::new(ns, registry, String::new(), empty_cluster());
}

#[tokio::test]
async fn remote_tombstone_removes_mesh_imported_worker() {
    let (ns, registry, adapter) = setup();
    adapter.start();

    // Import a peer's worker, then deliver its tombstone.
    ns.put(
        "worker:peer-w1",
        bincode::serialize(&sample_state("peer-w1", "http://remote:8080")).unwrap(),
    );
    assert!(
        wait_for(|| registry.get_by_url("http://remote:8080").is_some()).await,
        "import landed"
    );

    ns.delete("worker:peer-w1");
    assert!(
        wait_for(|| registry.get_by_url("http://remote:8080").is_none()).await,
        "remote tombstone must remove the mesh-imported worker"
    );
}

#[tokio::test]
async fn remote_tombstone_never_removes_local_worker() {
    let (ns, registry, adapter) = setup();
    adapter.start();

    let id = registry
        .register(local_worker("http://local:8080"))
        .unwrap();
    let key = format!("worker:{}", id.as_str());
    assert!(wait_for(|| ns.get(&key).is_some()).await, "publish landed");

    // A hostile/buggy peer tombstones OUR key: the registry must
    // refuse, the worker stays registered, and the key is re-asserted
    // so peers that dropped their imports re-learn it.
    ns.delete(&key);
    assert!(
        wait_for(|| ns.get(&key).is_some()).await,
        "owned key must be re-published after a foreign tombstone"
    );
    assert!(
        registry.get(&id).is_some(),
        "a remote tombstone must never remove a locally-owned worker"
    );
}

#[tokio::test]
async fn outbound_publishes_local_registration() {
    let (ns, registry, adapter) = setup();
    adapter.start();

    let id = registry
        .register(local_worker("http://local:8080"))
        .unwrap();

    let key = format!("worker:{}", id.as_str());
    assert!(
        wait_for(|| ns.get(&key).is_some()).await,
        "local registration was not published to mesh"
    );
    let decoded: WorkerState =
        bincode::deserialize(&ns.get(&key).unwrap()).expect("published state decodes");
    assert_eq!(decoded.worker_id, id.as_str());
    assert_eq!(decoded.url, "http://local:8080");
    assert_eq!(decoded.model_id, "llama-3");
    assert!(
        !decoded.spec.is_empty(),
        "spec rides along for faithful import"
    );
}

#[tokio::test]
async fn outbound_never_publishes_local_zmq_worker() {
    // ZMQ workers are host-local capacity: their `ipc://` endpoint is
    // meaningless off this machine, so they stay out of the mesh.
    let (ns, registry, adapter) = setup();
    adapter.start();

    let zmq: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("ipc:///tmp/smg-local.sock")
            .model(ModelCard::new("llama-3"))
            .connection_mode(ConnectionMode::Zmq)
            .build(),
    );
    let zmq_id = registry.register(zmq).unwrap();
    registry
        .get(&zmq_id)
        .unwrap()
        .set_status(WorkerStatus::Ready);

    // A published HTTP worker is the ordering fence: once its key is in
    // the store the outbound loop has drained past the ZMQ events.
    let http_id = registry
        .register(local_worker("http://local:8080"))
        .unwrap();
    assert!(
        wait_for(|| ns.get(&format!("worker:{}", http_id.as_str())).is_some()).await,
        "local HTTP registration was not published"
    );
    assert!(
        ns.get(&format!("worker:{}", zmq_id.as_str())).is_none(),
        "a ZMQ worker must never be published to the mesh"
    );

    // The resync/lag path takes the same gate.
    let mut published = HashMap::new();
    adapter.resync_local(&mut published);
    assert!(
        ns.get(&format!("worker:{}", zmq_id.as_str())).is_none(),
        "resync must not publish a ZMQ worker"
    );
    assert!(published.contains_key(&http_id) && !published.contains_key(&zmq_id));
}

#[tokio::test]
async fn outbound_never_republishes_mesh_imported_worker() {
    let (ns, registry, adapter) = setup();
    adapter.start();

    // A peer's state arrives (direct put models the gossiped write).
    let original = sample_state("peer-w1", "http://remote:8080");
    ns.put(
        "worker:peer-w1",
        bincode::serialize(&original).expect("state serializes"),
    );
    assert!(
        wait_for(|| registry.get_by_url("http://remote:8080").is_some()).await,
        "import did not land in the registry"
    );

    // The import fired a Registered event; give the outbound loop time
    // to (wrongly) act on it, then prove the stored state is untouched.
    sleep(Duration::from_millis(50)).await;
    let stored: WorkerState =
        bincode::deserialize(&ns.get("worker:peer-w1").unwrap()).expect("state decodes");
    assert_eq!(
        stored, original,
        "a mesh-imported worker must never be re-published"
    );
}

#[tokio::test]
async fn outbound_tombstones_local_removal() {
    let (ns, registry, adapter) = setup();
    adapter.start();

    let id = registry
        .register(local_worker("http://local:8080"))
        .unwrap();
    let key = format!("worker:{}", id.as_str());
    assert!(wait_for(|| ns.get(&key).is_some()).await, "publish landed");

    registry.remove(&id);
    assert!(
        wait_for(|| ns.get(&key).is_none()).await,
        "local removal was not tombstoned"
    );
}

#[tokio::test]
async fn reconcile_recovers_missed_put_and_missed_tombstone() {
    let (ns, registry, adapter) = setup();
    // No start(): models every notification being dropped.

    ns.put(
        "worker:peer-w1",
        bincode::serialize(&sample_state("peer-w1", "http://remote:8080")).unwrap(),
    );
    adapter.reconcile_once();
    assert!(
        registry.get_by_url("http://remote:8080").is_some(),
        "reconcile recovers a missed put"
    );

    ns.delete("worker:peer-w1");
    adapter.reconcile_once();
    assert!(
        registry.get_by_url("http://remote:8080").is_none(),
        "reconcile recovers a missed tombstone"
    );
}

#[tokio::test]
#[traced_test]
async fn the_reconcile_pass_leaves_a_local_zmq_worker_alone_which_it_never_published() {
    // A host-local ZMQ worker has no key by design (it is never
    // published); its absent key is not a missed remote tombstone, so a
    // pass neither asks the registry to remove it nor warns of a refused
    // tombstone no peer sent.
    let (ns, registry, adapter) = setup();
    let zmq: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("ipc:///tmp/smg-local.sock")
            .model(ModelCard::new("llama-3"))
            .connection_mode(ConnectionMode::Zmq)
            .build(),
    );
    let zmq_id = registry.register(zmq).unwrap();
    let http_id = registry
        .register(local_worker("http://local:8080"))
        .unwrap();

    adapter.reconcile_once();
    adapter.reconcile_once();

    assert!(
        registry.get(&zmq_id).is_some() && registry.get(&http_id).is_some(),
        "both local workers stay registered"
    );
    assert!(
        ns.get(&format!("worker:{}", zmq_id.as_str())).is_none(),
        "the pass does not publish the ZMQ worker either"
    );
    assert!(
        ns.get(&format!("worker:{}", http_id.as_str())).is_some(),
        "the pass re-asserts the published worker whose key is missing"
    );
    assert!(
        !logs_contain("Refusing remote tombstone"),
        "a pass must not take the absent key of a worker this node never publishes \
             for a remote tombstone"
    );
}

#[tokio::test]
async fn rapid_put_then_delete_converges_to_absent() {
    // Both queued events resolve against the store (which holds the
    // tombstone by the time they drain), so a stale put echo can never
    // resurrect a removed worker regardless of drain timing.
    let (ns, registry, adapter) = setup();
    adapter.start();

    ns.put(
        "worker:peer-w1",
        bincode::serialize(&sample_state("peer-w1", "http://remote:8080")).unwrap(),
    );
    ns.delete("worker:peer-w1");

    sleep(Duration::from_millis(100)).await;
    assert!(
        registry.get_by_url("http://remote:8080").is_none(),
        "store truth wins: the tombstoned worker must not survive"
    );
}

#[tokio::test]
async fn published_spec_round_trips_into_importing_registry() {
    // The faithful-import contract: a gRPC decode worker published by
    // one node must import as a gRPC decode worker, not fall back to
    // the minimal HTTP/Regular builder. (bincode could not round-trip
    // WorkerSpec's serde-skip attributes; JSON does.)
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);

    let pub_registry = Arc::new(WorkerRegistry::new());
    let publisher = adapter(ns.clone(), pub_registry.clone());
    publisher.start();

    // The importer is another member: under the publisher's own name it
    // would take the key for a leftover of an earlier life and tombstone it.
    let sub_registry = Arc::new(WorkerRegistry::new());
    let subscriber = adapter_as(ns, sub_registry.clone(), "node-b");
    subscriber.start();

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://remote:9000")
            .model(ModelCard::new("llama-3"))
            .connection_mode(ConnectionMode::Grpc)
            .worker_type(WorkerType::Decode)
            .build(),
    );
    pub_registry.register(worker).unwrap();

    assert!(
        wait_for(|| sub_registry.get_by_url("grpc://remote:9000").is_some()).await,
        "import did not land"
    );
    let imported = sub_registry.get_by_url("grpc://remote:9000").unwrap();
    assert_eq!(
        *imported.connection_mode(),
        ConnectionMode::Grpc,
        "connection mode must survive the spec round trip"
    );
    assert_eq!(
        *imported.worker_type(),
        WorkerType::Decode,
        "worker type must survive the spec round trip"
    );
}

#[tokio::test]
async fn an_owners_replacement_in_place_reaches_its_peers_import() {
    // One node publishes a worker and another imports it; the owner
    // then replaces the worker in place (`register_or_replace`: the
    // discovery reconciler's upsert, the start-up workflow, a `PUT`)
    // with a new model and role under the same address. The `Replaced`
    // event publishes the new state under the same key, and the
    // import must follow it instead of keeping the first specification
    // for as long as it lives.
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);

    let pub_registry = Arc::new(WorkerRegistry::new());
    let publisher = adapter(ns.clone(), pub_registry.clone());
    publisher.start();

    let sub_registry = Arc::new(WorkerRegistry::new());
    let subscriber = adapter_as(ns, sub_registry.clone(), "node-b");
    subscriber.start();

    let url = "http://remote:9000";
    pub_registry.register(local_worker(url)).unwrap();
    assert!(
        wait_for(|| sub_registry.get_by_url(url).is_some()).await,
        "import did not land"
    );
    let imported_as = sub_registry.get_id_by_url(url).unwrap();

    let replacement: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("llama-4"))
            .worker_type(WorkerType::Decode)
            .build(),
    );
    pub_registry.register_or_replace(replacement);

    assert!(
        wait_for(|| sub_registry
            .get_by_url(url)
            .is_some_and(|worker| worker.model_id() == "llama-4"))
        .await,
        "the import did not follow the owner's replacement; it serves {:?}",
        sub_registry
            .get_by_url(url)
            .map(|w| w.model_id().to_string())
    );
    let followed = sub_registry.get_by_url(url).unwrap();
    assert_eq!(
        *followed.worker_type(),
        WorkerType::Decode,
        "the role follows too"
    );
    assert_eq!(
        sub_registry.get_id_by_url(url),
        Some(imported_as),
        "under the id the import adopted"
    );
}

#[tokio::test]
async fn claimed_import_publishes_from_next_mutation() {
    // The restart race end-to-end: a mesh import wins the URL, the local
    // workflow claims it via register_or_replace, and the worker must be
    // published from its next mutation onward.
    let (ns, registry, adapter) = setup();
    adapter.start();

    // Ghost state from a previous incarnation arrives first.
    ns.put(
        "worker:old-id",
        bincode::serialize(&sample_state("old-id", "http://local:8080")).unwrap(),
    );
    assert!(
        wait_for(|| registry.get_by_url("http://local:8080").is_some()).await,
        "import landed"
    );

    // The local workflow claims the URL (register_or_replace path).
    let claimed = registry.register_or_replace(local_worker("http://local:8080"));

    // The Replaced event publishes the claimed worker (live origin gate).
    let key = format!("worker:{}", claimed.as_str());
    assert!(
        wait_for(|| {
            ns.get(&key)
                .and_then(|bytes| bincode::deserialize::<WorkerState>(&bytes).ok())
                .is_some_and(|s| !s.spec.is_empty())
        })
        .await,
        "claimed worker must be published with its local spec"
    );
}

#[tokio::test]
async fn resync_skips_republish_when_state_unchanged() {
    // Every put mints a fresh Lamport op (a watermark delta to every
    // peer), so a lag resync must not re-put unchanged state. A put
    // always notifies local subscribers, so an empty channel after the
    // second resync proves the skip.
    let (ns, registry, adapter) = setup();

    let id = registry
        .register(local_worker("http://local:8080"))
        .unwrap();
    let mut published = HashMap::new();
    adapter.resync_local(&mut published);

    let mut sub = ns.subscribe("");
    adapter.resync_local(&mut published);
    assert!(
        matches!(sub.receiver.try_recv(), Err(TryRecvError::Empty)),
        "unchanged state must not be re-put on resync"
    );

    // A real change (health flip) is still republished.
    registry.get(&id).unwrap().set_status(WorkerStatus::Ready);
    adapter.resync_local(&mut published);
    assert!(
        sub.receiver.try_recv().is_ok(),
        "changed state must be republished on resync"
    );
}

#[tokio::test]
async fn resync_publishes_local_skips_mesh_and_tombstones_stale() {
    let (ns, registry, adapter) = setup();

    // One local worker, one mesh import, one phantom previously
    // published id whose worker is gone.
    let local_id = registry
        .register(local_worker("http://local:8080"))
        .unwrap();
    registry.on_remote_worker_state(&sample_state("peer-w1", "http://remote:8080"));
    let phantom = WorkerId::from_string("gone-w9".to_string());
    ns.put(
        "worker:gone-w9",
        bincode::serialize(&sample_state("gone-w9", "http://gone:8080")).unwrap(),
    );

    let mut published: HashMap<WorkerId, Published> = [(
        phantom,
        Published {
            url: "http://gone:8080".into(),
            instance: None,
        },
    )]
    .into_iter()
    .collect();
    adapter.resync_local(&mut published);

    assert!(
        ns.get(&format!("worker:{}", local_id.as_str())).is_some(),
        "local worker is re-published"
    );
    assert!(
        ns.get("worker:gone-w9").is_none(),
        "stale published key is tombstoned"
    );
    assert_eq!(
        published.keys().cloned().collect::<HashSet<_>>(),
        HashSet::from([local_id]),
        "published set tracks exactly the local workers"
    );
    assert!(
        adapter.holds_back(&sample_state("x", "http://gone:8080")),
        "a worker removed during the lag retires its endpoint too"
    );
}

#[tokio::test]
async fn published_state_names_this_node_as_owner() {
    let (ns, registry, adapter) = setup();

    let id = registry
        .register(local_worker("http://local:8080"))
        .unwrap();
    let mut published = HashMap::new();
    adapter.resync_local(&mut published);

    let stored: WorkerState =
        bincode::deserialize(&ns.get(&format!("worker:{}", id.as_str())).unwrap())
            .expect("published state decodes");
    assert_eq!(stored.owner, "node-a", "a key names the node that owns it");
}

#[tokio::test]
async fn legacy_state_without_owner_still_imports() {
    // A record from a node that predates `owner` is shorter than the
    // current layout; it must import (with an unknown owner), not be
    // dropped as undecodable.
    #[derive(serde::Serialize)]
    struct Legacy {
        worker_id: String,
        model_id: String,
        url: String,
        health: bool,
        load: f64,
        version: u64,
        spec: Vec<u8>,
    }
    let (ns, registry, adapter) = setup();

    let legacy = Legacy {
        worker_id: "old-w1".into(),
        model_id: "llama-3".into(),
        url: "http://old-node:8080".into(),
        health: true,
        load: 0.0,
        version: 1,
        spec: vec![],
    };
    ns.put("worker:old-w1", bincode::serialize(&legacy).unwrap());
    adapter.reconcile_once();
    assert!(
        registry.get_by_url("http://old-node:8080").is_some(),
        "a record without an owner field still imports"
    );
}

#[tokio::test(start_paused = true)]
async fn dead_owner_key_is_not_imported() {
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);
    let registry = Arc::new(WorkerRegistry::new());
    let cluster = cluster_with(&[
        ("node-a", NodeStatus::Alive),
        ("node-b", NodeStatus::Alive),
        ("node-c", NodeStatus::Down),
        ("node-d", NodeStatus::Leaving),
    ]);
    let adapter = adapter_in(ns.clone(), registry.clone(), cluster);

    // The gone peers have been gone for longer than the grace.
    adapter.reconcile_once();
    tokio::time::advance(DEAD_OWNER_GRACE).await;

    for (key, state) in [
        (
            "worker:c-w1",
            owned_state("c-w1", "http://gone-c:8080", "node-c"),
        ),
        (
            "worker:d-w1",
            owned_state("d-w1", "http://gone-d:8080", "node-d"),
        ),
        (
            "worker:b-w1",
            owned_state("b-w1", "http://live-b:8080", "node-b"),
        ),
        (
            "worker:x-w1",
            owned_state("x-w1", "http://unknown-x:8080", "node-x"),
        ),
    ] {
        ns.put(key, bincode::serialize(&state).unwrap());
    }
    adapter.reconcile_once();

    assert!(
        registry.get_by_url("http://gone-c:8080").is_none(),
        "a Down owner's key must not import"
    );
    assert!(
        registry.get_by_url("http://gone-d:8080").is_none(),
        "a Leaving owner's key must not import"
    );
    assert!(
        registry.get_by_url("http://live-b:8080").is_some(),
        "an alive owner's key imports"
    );
    assert!(
        registry.get_by_url("http://unknown-x:8080").is_some(),
        "an owner the cluster state does not list is not evidence of death"
    );
}

#[tokio::test(start_paused = true)]
async fn import_of_an_owner_that_went_down_is_removed_after_the_grace_and_returns_with_it() {
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);
    let registry = Arc::new(WorkerRegistry::new());
    let cluster = cluster_with(&[("node-a", NodeStatus::Alive), ("node-c", NodeStatus::Alive)]);
    let adapter = adapter_in(ns.clone(), registry.clone(), cluster.clone());

    ns.put(
        "worker:c-w1",
        bincode::serialize(&owned_state("c-w1", "http://remote:8080", "node-c")).unwrap(),
    );
    adapter.reconcile_once();
    assert!(
        registry.get_by_url("http://remote:8080").is_some(),
        "import landed while the owner was alive"
    );

    // The owner drops out: a pass within the grace keeps the import.
    set_status(&cluster, "node-c", NodeStatus::Down);
    adapter.reconcile_once();
    assert!(
        registry.get_by_url("http://remote:8080").is_some(),
        "one Down sighting within the grace is not evidence yet"
    );

    tokio::time::advance(DEAD_OWNER_GRACE).await;
    adapter.reconcile_once();
    assert!(
        registry.get_by_url("http://remote:8080").is_none(),
        "the import of a dead owner is removed"
    );
    assert!(
        ns.get("worker:c-w1").is_some(),
        "the key is the owner's to tombstone; it stays"
    );

    // The owner comes back: its key imports again.
    set_status(&cluster, "node-c", NodeStatus::Alive);
    adapter.reconcile_once();
    assert!(
        registry.get_by_url("http://remote:8080").is_some(),
        "a returning owner's key is imported again"
    );
}

#[tokio::test(start_paused = true)]
async fn replaced_worker_is_not_resurrected_from_a_dead_owners_key() {
    // The rollout scenario: a peer that is gone for good published the
    // same URL this node also owns locally. While the local worker exists
    // the peer's key is inert; once the local worker is removed (its
    // instance was replaced) the dead key must not bring the URL back.
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);
    let registry = Arc::new(WorkerRegistry::new());
    let cluster = cluster_with(&[("node-a", NodeStatus::Alive), ("node-c", NodeStatus::Down)]);
    let adapter = adapter_in(ns.clone(), registry.clone(), cluster);
    adapter.start();
    tokio::time::advance(DEAD_OWNER_GRACE).await;

    let local_id = registry
        .register(local_worker("http://worker:8080"))
        .unwrap();
    let local_key = format!("worker:{}", local_id.as_str());
    assert!(
        wait_for(|| ns.get(&local_key).is_some()).await,
        "publish landed"
    );
    ns.put(
        "worker:c-w1",
        bincode::serialize(&owned_state("c-w1", "http://worker:8080", "node-c")).unwrap(),
    );
    adapter.reconcile_once();
    assert_eq!(
        registry.origin_of(&local_id),
        Some(WorkerOrigin::Local),
        "the local worker is untouched by the peer's key"
    );

    registry.remove(&local_id);
    assert!(
        wait_for(|| ns.get(&local_key).is_none()).await,
        "local removal was tombstoned"
    );
    adapter.reconcile_once();
    assert!(
        registry.get_by_url("http://worker:8080").is_none(),
        "the replaced URL must not come back from the dead owner's key"
    );
}

#[tokio::test(start_paused = true)]
async fn dead_owner_key_does_not_touch_an_import_from_a_live_owner() {
    // Two routers published the same URL; one of them is gone. The import
    // came from the live one's key (it carries that key's id), so the dead
    // key must leave it alone on every pass, whatever the pass's order.
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);
    let registry = Arc::new(WorkerRegistry::new());
    let cluster = cluster_with(&[
        ("node-a", NodeStatus::Alive),
        ("node-b", NodeStatus::Alive),
        ("node-c", NodeStatus::Down),
    ]);
    let adapter = adapter_in(ns.clone(), registry.clone(), cluster);
    adapter.reconcile_once();
    tokio::time::advance(DEAD_OWNER_GRACE).await;

    ns.put(
        "worker:b-w1",
        bincode::serialize(&owned_state("b-w1", "http://shared:8080", "node-b")).unwrap(),
    );
    adapter.reconcile_once();
    let imported = registry
        .get_id_by_url("http://shared:8080")
        .expect("the live owner's key imported");
    assert_eq!(imported.as_str(), "b-w1");

    ns.put(
        "worker:c-w1",
        bincode::serialize(&owned_state("c-w1", "http://shared:8080", "node-c")).unwrap(),
    );
    let mut events = registry.subscribe_events();
    for _ in 0..8 {
        adapter.reconcile_once();
        assert_eq!(
            registry.get_id_by_url("http://shared:8080").as_ref(),
            Some(&imported),
            "the import from the live owner survives a pass with the dead key"
        );
    }
    assert!(
        matches!(
            events.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ),
        "no removal or re-registration churn from the dead key"
    );
}

#[tokio::test(start_paused = true)]
async fn a_live_owners_key_refills_a_url_the_dead_key_vacated_in_the_same_pass() {
    // The import came from the key that later dies while a live owner
    // publishes the same URL: the pass handles the dead key first, so the
    // live key re-imports the URL in the same pass, whatever the set's
    // iteration order.
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);
    let registry = Arc::new(WorkerRegistry::new());
    let cluster = cluster_with(&[
        ("node-a", NodeStatus::Alive),
        ("node-b", NodeStatus::Alive),
        ("node-c", NodeStatus::Alive),
    ]);
    let adapter = adapter_in(ns.clone(), registry.clone(), cluster.clone());

    ns.put(
        "worker:c-w1",
        bincode::serialize(&owned_state("c-w1", "http://shared:8080", "node-c")).unwrap(),
    );
    adapter.reconcile_once();
    assert_eq!(
        registry
            .get_id_by_url("http://shared:8080")
            .map(|id| id.as_str().to_string()),
        Some("c-w1".to_string()),
        "the import came from the owner that is about to die"
    );
    ns.put(
        "worker:b-w1",
        bincode::serialize(&owned_state("b-w1", "http://shared:8080", "node-b")).unwrap(),
    );

    set_status(&cluster, "node-c", NodeStatus::Down);
    adapter.reconcile_once();
    tokio::time::advance(DEAD_OWNER_GRACE).await;
    // The live key first: the order that left the URL out when the keys
    // were handled in one loop (a pass's set order is random; the dead
    // key must be handled first whatever the order).
    adapter.backfill_keys(["worker:b-w1".to_string(), "worker:c-w1".to_string()]);
    assert_eq!(
        registry
            .get_id_by_url("http://shared:8080")
            .map(|id| id.as_str().to_string()),
        Some("b-w1".to_string()),
        "the live owner's key refills the URL in the pass that dropped the dead import"
    );
    for _ in 0..8 {
        adapter.reconcile_once();
        assert!(
            registry.get_by_url("http://shared:8080").is_some(),
            "the URL a live owner still publishes is present after every pass"
        );
    }
    assert_eq!(
        registry
            .get_id_by_url("http://shared:8080")
            .map(|id| id.as_str().to_string()),
        Some("b-w1".to_string()),
        "the import now belongs to the live owner's key"
    );
}

#[tokio::test(start_paused = true)]
async fn returning_owner_is_live_at_once_on_the_event_path() {
    // The first-seen time is refreshed once per pass; an owner that is
    // alive again must not stay dead until that pass.
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);
    let registry = Arc::new(WorkerRegistry::new());
    let cluster = cluster_with(&[("node-a", NodeStatus::Alive), ("node-c", NodeStatus::Down)]);
    let adapter = adapter_in(ns.clone(), registry.clone(), cluster.clone());
    adapter.reconcile_once();
    tokio::time::advance(DEAD_OWNER_GRACE).await;
    assert_eq!(adapter.owner_liveness("node-c"), OwnerLiveness::Dead);

    set_status(&cluster, "node-c", NodeStatus::Alive);
    ns.put(
        "worker:c-w1",
        bincode::serialize(&owned_state("c-w1", "http://remote:8080", "node-c")).unwrap(),
    );
    adapter.sync_key_from_store("worker:c-w1", "c-w1");
    assert!(
        registry.get_by_url("http://remote:8080").is_some(),
        "a returning owner's key imports before the next pass"
    );
}
#[tokio::test]
async fn an_owner_that_comes_back_tombstones_the_keys_of_its_earlier_life_and_its_peer_rebinds_its_import(
) {
    // One store stands in for the gossip between the owner and its peer.
    let mesh = MeshKV::new("node-p".into());
    let ns = worker_namespace(&mesh);
    let url = "http://engine:8080";

    // The owner's first life publishes its worker (no tasks: a crashed
    // process runs nothing afterwards); the peer imports it under the
    // id that life gave it.
    let first_registry = Arc::new(WorkerRegistry::new());
    let first_life = adapter_as(ns.clone(), first_registry.clone(), "node-p");
    let first_id = first_registry.register(local_worker(url)).unwrap();
    first_life.resync_local(&mut HashMap::new());
    let first_key = format!("worker:{}", first_id.as_str());
    assert!(
        ns.get(&first_key).is_some(),
        "the first life published its worker"
    );
    drop(first_life);
    drop(first_registry);

    let peer_registry = Arc::new(WorkerRegistry::new());
    let peer = adapter_as(ns.clone(), peer_registry.clone(), "node-s");
    peer.start();
    assert!(
        wait_for(|| peer_registry.get_id_by_url(url).as_ref() == Some(&first_id)).await,
        "the peer imported the worker under the owner's first id"
    );

    // The owner comes back under its name: a fresh registry that holds
    // the same endpoint under a fresh id by the time the key of its
    // earlier life reaches it (the backfill at start).
    let second_registry = Arc::new(WorkerRegistry::new());
    let second_id = second_registry.register(local_worker(url)).unwrap();
    assert_ne!(second_id, first_id, "a fresh registry mints a fresh id");
    let second_life = adapter_as(ns.clone(), second_registry.clone(), "node-p");
    second_life.start();

    assert!(
        wait_for(|| ns.get(&first_key).is_none()).await,
        "the owner tombstones the key of its earlier life"
    );
    assert!(
        wait_for(|| peer_registry.get_id_by_url(url).as_ref() == Some(&second_id)).await,
        "the peer's import follows the owner's new id, not the next pass: {:?}",
        peer_registry.get_id_by_url(url)
    );

    // The owner's later removal is a tombstone of a key the peer's
    // import now answers to.
    second_registry.remove(&second_id);
    assert!(
        wait_for(|| peer_registry.get_by_url(url).is_none()).await,
        "the owner's removal reaches the peer"
    );
}

#[tokio::test]
async fn a_key_back_before_the_new_life_registers_its_endpoint_is_claimed_under_its_id() {
    // Gossip serves before the start-up jobs register the configured
    // workers: the key of the earlier life can reach the new life while
    // its registry is still empty. Such a key is no leftover: it imports
    // as before, the start-up upsert claims it under the same id and the
    // peer's import never changes, where a tombstone would have left the
    // peer without the endpoint until the new registration published.
    // The test runs on a current-thread runtime, so the new life's
    // outbound loop runs only while the test awaits: each step that it
    // must see is awaited before the next.
    let mesh = MeshKV::new("node-p".into());
    let ns = worker_namespace(&mesh);
    let url = "http://engine:8080";
    let first_registry = Arc::new(WorkerRegistry::new());
    let first_life = adapter_as(ns.clone(), first_registry.clone(), "node-p");
    let first_id = first_registry.register(local_worker(url)).unwrap();
    first_life.resync_local(&mut HashMap::new());
    let first_key = format!("worker:{}", first_id.as_str());
    let first_version = version_of(&ns, &first_key).expect("the first life's key");
    drop(first_life);
    drop(first_registry);

    let peer_registry = Arc::new(WorkerRegistry::new());
    let peer = adapter_as(ns.clone(), peer_registry.clone(), "node-s");
    peer.start();
    assert!(
        wait_for(|| peer_registry.get_id_by_url(url).as_ref() == Some(&first_id)).await,
        "the peer imported the worker under the owner's first id"
    );

    // The owner comes back under its name with an empty registry: the
    // key of its earlier life reaches it before anything registers.
    let second_registry = Arc::new(WorkerRegistry::new());
    let second_life = adapter_as(ns.clone(), second_registry.clone(), "node-p");
    second_life.start();
    assert!(
        wait_for(|| second_registry.get_id_by_url(url).as_ref() == Some(&first_id)).await,
        "the new life imports the key of its earlier life instead of tombstoning it"
    );
    assert!(ns.get(&first_key).is_some(), "the key stands");

    // The start-up upsert claims the endpoint under the same id. The
    // replacement bumps the worker's revision, so the claim's publish
    // raises the key's version: from there the new life's outbound loop
    // tracks the key, and a pass sees this life's own live key.
    assert_eq!(
        second_registry.register_or_replace(local_worker(url)),
        first_id,
        "the claim keeps the id"
    );
    assert_eq!(
        second_registry.origin_of(&first_id),
        Some(WorkerOrigin::Local)
    );
    assert!(
        wait_for(|| version_of(&ns, &first_key).is_some_and(|v| v > first_version)).await,
        "the new life publishes the claimed worker under the same key"
    );
    second_life.reconcile_once();
    assert!(
        ns.get(&first_key).is_some(),
        "the claimed key is this life's own now"
    );
    assert_eq!(
        peer_registry.get_id_by_url(url).as_ref(),
        Some(&first_id),
        "the peer's import never moved"
    );

    // The owner's later removal still reaches the peer.
    second_registry.remove(&first_id);
    assert!(
        wait_for(|| peer_registry.get_by_url(url).is_none()).await,
        "the owner's removal reaches the peer"
    );
}

#[tokio::test]
async fn a_live_local_workers_own_key_is_never_taken_for_a_leftover() {
    // The echo of this node's own publish and every pass see a key of
    // this node's name; the worker it names is held here, so it stays.
    let (ns, registry, adapter) = setup();
    adapter.start();

    let id = registry
        .register(local_worker("http://local:8080"))
        .unwrap();
    let key = format!("worker:{}", id.as_str());
    assert!(wait_for(|| ns.get(&key).is_some()).await, "publish landed");

    adapter.reconcile_once();
    sleep(Duration::from_millis(50)).await;
    assert!(
        ns.get(&key).is_some(),
        "the key of a worker this node holds is not tombstoned"
    );
    assert!(
        registry.get(&id).is_some(),
        "and the worker stays registered"
    );
}

#[tokio::test]
async fn a_tombstone_that_removes_an_import_refills_its_endpoint_from_another_live_key_at_once() {
    // Two live publishers name one endpoint; the import came from the
    // first, the second's key refreshed its health only.
    let (ns, registry, adapter) = setup();
    adapter.start();
    let url = "http://shared:8080";
    ns.put(
        "worker:b-w1",
        bincode::serialize(&owned_state("b-w1", url, "node-b")).unwrap(),
    );
    assert!(
        wait_for(|| registry
            .get_id_by_url(url)
            .is_some_and(|id| id.as_str() == "b-w1"))
        .await,
        "imported from the first publisher's key"
    );
    ns.put(
        "worker:c-w1",
        bincode::serialize(&owned_state("c-w1", url, "node-c")).unwrap(),
    );
    sleep(Duration::from_millis(50)).await;
    assert!(
        registry
            .get_id_by_url(url)
            .is_some_and(|id| id.as_str() == "b-w1"),
        "a second key for a held endpoint refreshes health only"
    );

    // The first publisher removes its worker: the event path refills the
    // endpoint from the other key, as the pass would at its next run.
    ns.delete("worker:b-w1");
    assert!(
        wait_for(|| registry
            .get_id_by_url(url)
            .is_some_and(|id| id.as_str() == "c-w1"))
        .await,
        "the endpoint is refilled from the other live key in the same step: {:?}",
        registry.get_id_by_url(url)
    );
}

/// A worker of `url` serving `model`, as the start-up jobs register it.
fn local_worker_of(url: &str, model: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new(model))
            .build(),
    )
}

#[tokio::test]
async fn a_claim_after_a_restart_publishes_above_the_earlier_lifes_version() {
    // The published version is the owner's worker revision, which
    // restarts at zero with the process; the key outlives it. A life
    // that replaced its worker once published version 1 or more, and a
    // peer recorded that version with its import. The new life claims
    // the key its earlier life left (the start-up upsert over the
    // import), which bumps a fresh revision to 1: without the raise,
    // the claim publishes at or below what the peer holds, and the
    // peer keeps the earlier life's specification for good.
    let mesh = MeshKV::new("node-p".into());
    let ns = worker_namespace(&mesh);
    let url = "http://engine:8080";
    let first_registry = Arc::new(WorkerRegistry::new());
    let first_life = adapter_as(ns.clone(), first_registry.clone(), "node-p");
    let first_id = first_registry
        .register(local_worker_of(url, "model-a"))
        .unwrap();
    // A replacement in the first life (a `PUT`, a discovery refresh):
    // the key's version is 1 or more from here on.
    assert_eq!(
        first_registry.register_or_replace(local_worker_of(url, "model-a")),
        first_id
    );
    first_life.resync_local(&mut HashMap::new());
    let first_key = format!("worker:{}", first_id.as_str());
    let first_version = version_of(&ns, &first_key).expect("the first life's key");
    assert!(
        first_version >= 1,
        "the first life published at {first_version}"
    );
    drop(first_life);
    drop(first_registry);

    let peer_registry = Arc::new(WorkerRegistry::new());
    let peer = adapter_as(ns.clone(), peer_registry.clone(), "node-s");
    peer.start();
    assert!(
        wait_for(|| peer_registry.get_id_by_url(url).as_ref() == Some(&first_id)).await,
        "the peer imported the worker under the owner's first id"
    );
    assert_eq!(
        peer_registry
            .get_by_url(url)
            .map(|worker| worker.model_id().to_string()),
        Some("model-a".to_string())
    );

    // The owner comes back under its name, imports its earlier life's
    // key, and the start-up upsert claims it with the new configuration.
    let second_registry = Arc::new(WorkerRegistry::new());
    let second_life = adapter_as(ns.clone(), second_registry.clone(), "node-p");
    second_life.start();
    assert!(
        wait_for(|| second_registry.get_id_by_url(url).as_ref() == Some(&first_id)).await,
        "the new life imports the key of its earlier life"
    );
    assert_eq!(
        second_registry.register_or_replace(local_worker_of(url, "model-b")),
        first_id,
        "the claim keeps the id"
    );
    assert!(
        wait_for(|| version_of(&ns, &first_key).is_some_and(|v| v > first_version)).await,
        "the claim publishes above the earlier life's version {first_version}: {:?}",
        version_of(&ns, &first_key)
    );
    assert!(
        wait_for(|| {
            peer_registry
                .get_by_url(url)
                .is_some_and(|worker| worker.model_id() == "model-b")
        })
        .await,
        "the peer's import follows the claim's specification: {:?}",
        peer_registry
            .get_by_url(url)
            .map(|worker| worker.model_id().to_string())
    );
}

#[tokio::test]
async fn a_later_publication_of_the_earlier_life_is_outranked_after_the_claim() {
    // The claim raises the revision past the version this node saw for
    // the key. A peer may have seen a later publication of the earlier
    // life first (a replacement just before its death, gossiped there
    // and not here yet). When it arrives after the claim, the owner
    // raises the revision past it and publishes again, so that peer's
    // import follows the claim too.
    let mesh = MeshKV::new("node-p".into());
    let ns = worker_namespace(&mesh);
    let url = "http://engine:8080";
    let first_registry = Arc::new(WorkerRegistry::new());
    let first_life = adapter_as(ns.clone(), first_registry.clone(), "node-p");
    let first_id = first_registry
        .register(local_worker_of(url, "model-a"))
        .unwrap();
    assert_eq!(
        first_registry.register_or_replace(local_worker_of(url, "model-a")),
        first_id
    );
    first_life.resync_local(&mut HashMap::new());
    let first_key = format!("worker:{}", first_id.as_str());
    let earlier = ns
        .get(&first_key)
        .and_then(|bytes| decode_worker_state(&bytes).ok())
        .expect("the first life's key");
    drop(first_life);
    drop(first_registry);

    let peer_registry = Arc::new(WorkerRegistry::new());
    let peer = adapter_as(ns.clone(), peer_registry.clone(), "node-s");
    peer.start();
    assert!(
        wait_for(|| peer_registry.get_id_by_url(url).as_ref() == Some(&first_id)).await,
        "the peer imported the worker under the owner's first id"
    );

    let second_registry = Arc::new(WorkerRegistry::new());
    let second_life = adapter_as(ns.clone(), second_registry.clone(), "node-p");
    second_life.start();
    assert!(
        wait_for(|| second_registry.get_id_by_url(url).as_ref() == Some(&first_id)).await,
        "the new life imports the key of its earlier life"
    );
    assert_eq!(
        second_registry.register_or_replace(local_worker_of(url, "model-b")),
        first_id
    );
    assert!(
        wait_for(|| version_of(&ns, &first_key).is_some_and(|v| v > earlier.version)).await,
        "the claim publishes above the version this node saw"
    );
    let claimed = version_of(&ns, &first_key).expect("the claim's key");

    // The earlier life's last publication, seen by a peer first, arrives
    // after the claim: the earlier specification at a higher version.
    let late_version = claimed + 1;
    let late = WorkerState {
        version: late_version,
        ..earlier
    };
    ns.put(&first_key, bincode::serialize(&late).unwrap());

    let stored = || {
        ns.get(&first_key)
            .and_then(|bytes| decode_worker_state(&bytes).ok())
            .map(|state| (state.version, state.model_id))
    };
    assert!(
        wait_for(|| stored()
            .is_some_and(|(version, model)| version > late_version && model == "model-b"))
        .await,
        "the owner publishes the claimed worker above the late state: {:?}",
        stored()
    );
    assert!(
        wait_for(|| {
            peer_registry
                .get_by_url(url)
                .is_some_and(|worker| worker.model_id() == "model-b")
        })
        .await,
        "the peer's import follows the claim: {:?}",
        peer_registry
            .get_by_url(url)
            .map(|worker| worker.model_id().to_string())
    );
}

#[tokio::test]
async fn a_health_only_difference_at_the_same_version_leaves_the_revision_alone() {
    // Health is not versioned: a probe's demotion changes the worker's
    // state without a revision bump, and the outbound loop puts it a
    // moment later. A pass or an event that reads the key in that gap
    // sees this node's own earlier state, not a later state of its key.
    // The revision is a replacement token elsewhere, so it must not
    // move for that. Another specification at the same version is the
    // earlier life's later publication and does raise it.
    let (ns, registry, adapter) = setup();
    let worker = local_worker("http://local:8080");
    let id = registry.register(worker.clone()).unwrap();
    worker.set_status(WorkerStatus::Ready);
    adapter.resync_local(&mut HashMap::new());
    let key = format!("worker:{}", id.as_str());
    let published = worker_state_of(&id, &worker, "node-a");
    let revision = worker.revision();
    assert!(
        published.health,
        "the key is published with the worker healthy"
    );
    assert_eq!(version_of(&ns, &key), Some(revision));

    worker.set_status(WorkerStatus::NotReady);
    adapter.apply_state(id.as_str(), &published);
    assert_eq!(
        worker.revision(),
        revision,
        "the stale health at the same version is this node's own state"
    );
    assert_eq!(version_of(&ns, &key), Some(revision));

    let foreign = WorkerState {
        model_id: "other-model".to_string(),
        ..published
    };
    adapter.apply_state(id.as_str(), &foreign);
    assert_eq!(
        worker.revision(),
        revision + 1,
        "another specification at the same version is outranked"
    );
    assert_eq!(version_of(&ns, &key), Some(revision + 1));
}

/// A locally registered worker as discovery registers it: the provider's
/// instance id in its labels.
fn discovered_worker(url: &str, instance: &str) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .model(ModelCard::new("llama-3"))
            .label(DISCOVERY_ID_LABEL, instance)
            .build(),
    )
}

/// A peer's state for `url` whose spec names `instance`.
fn peer_state(worker_id: &str, url: &str, instance: &str) -> WorkerState {
    let mut spec = openai_protocol::worker::WorkerSpec::new(url);
    spec.labels
        .insert(DISCOVERY_ID_LABEL.to_string(), instance.to_string());
    WorkerState {
        spec: serde_json::to_vec(&spec).unwrap(),
        ..sample_state(worker_id, url)
    }
}

#[tokio::test]
async fn an_endpoint_this_gateway_removed_is_not_re_imported_from_a_peer() {
    // Discovery (or an operator) removed the worker here: a peer's key for
    // the same instance must not bring it back, a new instance at the
    // reused address imports, and a fresh local registration re-admits it.
    let (ns, registry, adapter) = setup();
    adapter.start();

    let url = "http://10.0.0.5:8000";
    let id = registry.register(discovered_worker(url, "u1")).unwrap();
    let key = format!("worker:{}", id.as_str());
    assert!(wait_for(|| ns.get(&key).is_some()).await, "publish landed");
    registry.remove(&id);
    assert!(
        wait_for(|| ns.get(&key).is_none()).await,
        "removal tombstoned"
    );
    assert!(
        wait_for(|| adapter.holds_back(&peer_state("peer-w1", url, "u1"))).await,
        "the local removal retired the endpoint for that instance"
    );

    ns.put(
        "worker:peer-w1",
        bincode::serialize(&peer_state("peer-w1", url, "u1")).unwrap(),
    );
    adapter.reconcile_once();
    sleep(Duration::from_millis(50)).await;
    assert!(
        registry.get_by_url(url).is_none(),
        "the removed instance is not re-imported from a peer"
    );

    ns.put(
        "worker:peer-w2",
        bincode::serialize(&peer_state("peer-w2", url, "u2")).unwrap(),
    );
    assert!(
        wait_for(|| registry.get_by_url(url).is_some()).await,
        "a new instance at the reused address imports"
    );
    ns.delete("worker:peer-w2");
    assert!(wait_for(|| registry.get_by_url(url).is_none()).await);

    // A fresh local registration re-admits the endpoint.
    let id = registry.register(discovered_worker(url, "u1")).unwrap();
    assert!(
        wait_for(|| !adapter.holds_back(&peer_state("peer-w1", url, "u1"))).await,
        "a local registration re-admits the endpoint"
    );
    registry.remove(&id);
}

#[tokio::test]
async fn a_peer_state_without_an_instance_is_held_back_at_a_removed_endpoint() {
    // No spec (an older publisher, an encoding failure) or a spec this
    // build cannot decode names no instance: nothing tells it from the
    // removed one, so it is held back; a decodable new instance imports.
    let (ns, registry, adapter) = setup();
    adapter.start();

    let url = "http://10.0.0.6:8000";
    let id = registry.register(discovered_worker(url, "u1")).unwrap();
    let key = format!("worker:{}", id.as_str());
    assert!(wait_for(|| ns.get(&key).is_some()).await, "publish landed");
    registry.remove(&id);
    assert!(
        wait_for(|| adapter.holds_back(&sample_state("peer-w1", url))).await,
        "a state without a spec is held back"
    );
    let undecodable = WorkerState {
        spec: b"not json".to_vec(),
        ..sample_state("peer-w1", url)
    };
    assert!(
        adapter.holds_back(&undecodable),
        "a state whose spec does not decode is held back"
    );
    assert!(
        !adapter.holds_back(&peer_state("peer-w2", url, "u2")),
        "a decodable spec naming another instance is not"
    );
}

#[tokio::test]
async fn a_removed_worker_without_an_instance_holds_back_every_peer_state() {
    // The removed worker carried no provider instance id (added by hand):
    // nothing can tell a peer's state apart, so every one is held back.
    let (ns, registry, adapter) = setup();
    adapter.start();

    let url = "http://10.0.0.7:8000";
    let id = registry.register(local_worker(url)).unwrap();
    let key = format!("worker:{}", id.as_str());
    assert!(wait_for(|| ns.get(&key).is_some()).await, "publish landed");
    registry.remove(&id);
    assert!(
        wait_for(|| adapter.holds_back(&peer_state("peer-w9", url, "u9"))).await,
        "every peer state for the endpoint is held back"
    );
}

#[tokio::test(start_paused = true)]
async fn a_router_that_starts_after_an_owner_died_never_imports_its_keys() {
    // The dead key is in the store before this router exists, so its
    // first sighting of the owner is `Down`: the grace must not turn that
    // into an import for a pass, on the backfill or on the event path.
    let mesh = MeshKV::new("node-a".into());
    let ns = worker_namespace(&mesh);
    let registry = Arc::new(WorkerRegistry::new());
    let cluster = cluster_with(&[("node-a", NodeStatus::Alive), ("node-c", NodeStatus::Down)]);
    ns.put(
        "worker:c-w1",
        bincode::serialize(&owned_state("c-w1", "http://gone-c:8080", "node-c")).unwrap(),
    );
    let adapter = adapter_in(ns.clone(), registry.clone(), cluster.clone());
    adapter.start();
    sleep(Duration::from_millis(50)).await;
    assert!(
        registry.get_by_url("http://gone-c:8080").is_none(),
        "a key whose owner is gone creates no import inside the grace"
    );

    ns.put(
        "worker:c-w2",
        bincode::serialize(&owned_state("c-w2", "http://gone-c:8081", "node-c")).unwrap(),
    );
    sleep(Duration::from_millis(50)).await;
    assert!(
        registry.get_by_url("http://gone-c:8081").is_none(),
        "nor does one that arrives inside the grace"
    );

    tokio::time::advance(DEAD_OWNER_GRACE).await;
    adapter.reconcile_once();
    assert!(
        registry.get_by_url("http://gone-c:8080").is_none()
            && registry.get_by_url("http://gone-c:8081").is_none(),
        "nor after it"
    );

    set_status(&cluster, "node-c", NodeStatus::Alive);
    adapter.reconcile_once();
    assert!(
        registry.get_by_url("http://gone-c:8080").is_some()
            && registry.get_by_url("http://gone-c:8081").is_some(),
        "the owner's return imports its keys"
    );
}

#[tokio::test]
async fn an_import_that_landed_before_the_retirement_was_recorded_is_removed() {
    // The outbound loop records a retirement after the registry has
    // dropped the worker; a pass in that gap imports the endpoint from a
    // peer's key. The hold removes that import on the key's next pass,
    // and leaves an import of another instance at the address alone.
    let (ns, registry, adapter) = setup();

    let url = "http://10.0.0.8:8000";
    ns.put(
        "worker:peer-w1",
        bincode::serialize(&peer_state("peer-w1", url, "u1")).unwrap(),
    );
    adapter.reconcile_once();
    assert!(
        registry.get_by_url(url).is_some(),
        "imported before the retirement was recorded"
    );

    adapter.retire(url, Some("u1".into()));
    adapter.reconcile_once();
    assert!(
        registry.get_by_url(url).is_none(),
        "the hold removes the import it could not prevent"
    );
    assert!(
        ns.get("worker:peer-w1").is_some(),
        "the peer's key is the peer's to tombstone"
    );

    ns.put(
        "worker:peer-w2",
        bincode::serialize(&peer_state("peer-w2", url, "u2")).unwrap(),
    );
    adapter.reconcile_once();
    assert_eq!(
        registry
            .get_id_by_url(url)
            .map(|id| id.as_str().to_string())
            .as_deref(),
        Some("peer-w2"),
        "another instance at the address imports"
    );
    adapter.reconcile_once();
    assert!(
        registry.get_by_url(url).is_some(),
        "and the held-back key does not remove it"
    );
}
