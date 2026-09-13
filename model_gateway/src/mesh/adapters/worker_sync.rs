//! `worker:` CRDT adapter: gateway ↔ mesh bridge for `WorkerState`.
//!
//! Outbound: `start` spawns a loop over the registry's `WorkerEvent`
//! stream that publishes every locally-owned worker's state under
//! `worker:{worker_id}` (and a tombstone on removal). Mesh-imported
//! workers are filtered out by their registration origin so a peer's
//! state is never re-published, and `Zmq` workers are filtered out as
//! host-local capacity (their `ipc://` endpoint is unreachable off this
//! machine). On broadcast lag the loop re-publishes all local workers
//! and tombstones any it published that no longer exist.
//!
//! Inbound: `start` also spawns a task that subscribes to the
//! namespace, routes each non-tombstone update through
//! `WorkerRegistry::on_remote_worker_state` (registry-side URL-dedupe,
//! health promotion, `Registered` event fan-out), and routes
//! tombstones through `WorkerRegistry::remove_remote`, which removes
//! only mesh-imported workers.
//!
//! The adapter writes through `CrdtNamespace::put`, which fires
//! local subscribers in addition to gossiping. A local write
//! therefore echoes back through the inbound loop and lands in
//! `on_remote_worker_state`, which ignores state for locally-owned
//! workers — so the echo is inert.
//!
//! Known gap: only the owner tombstones its keys, so a permanently-dead
//! node's `worker:` keys persist cluster-wide (imports stay registered,
//! demoted by local probes), and a crash-restart that registers locally
//! before its old state gossips back orphans up to one store key per
//! worker per restart. Tombstone metadata also accrues per removed
//! worker (time-based collection would resurrect deleted keys; it is
//! only sound at causal stability). Cleanup belongs to dead-node key GC.

use std::{collections::HashSet, sync::Arc, time::Duration};

use smg_mesh::{CrdtNamespace, WorkerState};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use crate::worker::{
    event::WorkerEvent, registry::WorkerId, ConnectionMode, Worker, WorkerOrigin, WorkerRegistry,
};

const PREFIX: &str = "worker:";

/// Cadence of the inbound store↔registry reconcile pass.
const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);

/// Bridge between the `worker:` CRDT namespace and the gateway's
/// in-process `WorkerRegistry`.
pub struct WorkerSyncAdapter {
    workers: Arc<CrdtNamespace>,
    worker_registry: Arc<WorkerRegistry>,
}

impl std::fmt::Debug for WorkerSyncAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerSyncAdapter")
            .field("prefix", &self.workers.prefix())
            .finish_non_exhaustive()
    }
}

impl WorkerSyncAdapter {
    /// Build an adapter wrapping a `worker:`-scoped namespace and the
    /// gateway's worker registry. Panics if the namespace is not
    /// scoped to `worker:` so a mis-wired caller fails fast at
    /// startup rather than silently routing updates to the wrong
    /// prefix.
    pub fn new(workers: Arc<CrdtNamespace>, worker_registry: Arc<WorkerRegistry>) -> Arc<Self> {
        assert_eq!(
            workers.prefix(),
            PREFIX,
            "WorkerSyncAdapter requires a namespace scoped to `{PREFIX}`",
        );
        Arc::new(Self {
            workers,
            worker_registry,
        })
    }

    /// Start both sync directions.
    ///
    /// Inbound: subscribes to the namespace first so no live event is
    /// lost, spawns the recv loop so it can start draining
    /// immediately, and then backfills from the calling thread — any
    /// entry already in the CRDT would otherwise have to wait for the
    /// next unrelated write before the registry saw it. Running the
    /// backfill outside the spawn keeps the live loop free to drain
    /// concurrently: `notify` uses `try_send` into a bounded mpsc, so
    /// a blocked recv while backfill is running could drop updates on
    /// a busy startup. `on_remote_worker_state` is idempotent on URL,
    /// so a key seen by both paths only refreshes health.
    ///
    /// Outbound: subscribes to registry events before anything else so
    /// no registration between the initial resync and the loop is
    /// missed, then spawns the publish loop.
    pub fn start(self: &Arc<Self>) {
        let events = self.worker_registry.subscribe_events();
        let outbound = Arc::clone(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "publish loop runs for the adapter's lifetime, which is the process lifetime; the task holds the registry alive, so the Closed arm is defensive only"
        )]
        tokio::spawn(async move {
            outbound.run_outbound(events).await;
        });

        let reconcile = Arc::clone(self);
        #[expect(
            clippy::disallowed_methods,
            reason = "reconcile task runs for the adapter's lifetime, which is the process lifetime"
        )]
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(RECONCILE_INTERVAL);
            // The first tick resolves immediately and start() already
            // backfills synchronously; skip the redundant pass.
            interval.tick().await;
            loop {
                interval.tick().await;
                reconcile.reconcile_once();
            }
        });

        let this = Arc::clone(self);
        let mut sub = self.workers.subscribe("");
        #[expect(
            clippy::disallowed_methods,
            reason = "subscription task ends automatically when the mesh KV drops and closes the channel; no handle needed"
        )]
        tokio::spawn(async move {
            while let Some((key, _snapshot)) = sub.receiver.recv().await {
                let Some(worker_id) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                    warn!(key, "worker: subscription yielded unexpected key shape");
                    continue;
                };
                // Act on store truth, not the event's value snapshot: a
                // queued event can be stale by the time it is drained (a put
                // echo dequeued after the key was tombstoned would resurrect
                // a removed worker; a stale state would regress health).
                // Re-reading the namespace makes delivery order irrelevant.
                this.sync_key_from_store(&key, worker_id);
            }
            debug!("WorkerSyncAdapter subscription closed");
        });
        self.backfill_existing();
    }

    /// Replay every entry currently in the `worker:` namespace into
    /// the registry. Safe to run alongside the live subscription loop
    /// — the sink is idempotent on URL (health refresh short-circuit),
    /// so overlap with a concurrent live event is fine.
    fn backfill_existing(&self) {
        self.backfill_keys(self.workers.keys(""));
    }

    fn backfill_keys(&self, keys: impl IntoIterator<Item = String>) {
        for key in keys {
            let Some(worker_id) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                warn!(key, "worker: backfill yielded unexpected key shape");
                continue;
            };
            self.sync_key_from_store(&key, worker_id);
        }
    }

    /// One reconcile pass aligning the registry with the store. Subscription
    /// notifications are delivered via a bounded `try_send` and a dropped one
    /// is never re-fired (re-merged ops are idempotent no-ops), so this
    /// periodic pass is the recovery path for missed puts — including a
    /// cold-start merge burst larger than the channel — and missed
    /// tombstones. Missing-key handling runs before backfill so a same-URL
    /// key from another publisher can re-import in the same pass.
    fn reconcile_once(&self) {
        // Key-presence set: `keys()` avoids cloning every live value just
        // to test membership, and feeds the backfill below so the
        // namespace is scanned once per pass.
        let live: HashSet<String> = self.workers.keys("").into_iter().collect();
        for (id, _) in self.worker_registry.get_all_with_ids() {
            let key = format!("{PREFIX}{}", id.as_str());
            if !live.contains(&key) {
                // Missing backing key: same handling as a live tombstone
                // event — imports are removed, locally-owned state is
                // re-asserted.
                self.sync_key_from_store(&key, id.as_str());
            }
        }
        self.backfill_keys(live);
    }

    /// Bring the registry in line with the store's current state for one
    /// key: a live value routes through `on_remote_worker_state`; a missing
    /// (tombstoned) one removes a mesh import, re-asserts a locally-owned
    /// worker's state, and ignores unknown ids. Tombstones resolve via the
    /// publisher's id, which the import adopted.
    fn sync_key_from_store(&self, key: &str, worker_id: &str) {
        match self.workers.get(key) {
            Some(bytes) => self.apply_incoming(worker_id, &bytes),
            None => {
                let id = WorkerId::from_string(worker_id.to_string());
                // A missing key for a locally-owned worker is anomalous
                // (foreign tombstone or a lost publish): the key is already
                // gone cluster-wide and peers dropped their imports, so
                // re-assert the authoritative state — otherwise the worker
                // stays delisted everywhere until its next status change.
                if let Some(worker) = self.worker_registry.get(&id) {
                    if self.is_publishable(&id, &worker) {
                        warn!(
                            worker_id,
                            "re-publishing locally-owned worker after foreign tombstone"
                        );
                        self.on_worker_changed(worker_id, &worker_state_of(&id, &worker));
                        return;
                    }
                }
                match self.worker_registry.remove_remote(&id) {
                    Some(_) => info!(worker_id, "removed worker on remote tombstone"),
                    None => debug!(worker_id, "ignored tombstone (unknown id)"),
                }
            }
        }
    }

    /// Whether this node should export the worker to the cluster. Only
    /// locally-owned workers are ours to publish, and a `Zmq` worker is
    /// host-local by construction — its `ipc://` endpoint names a socket on
    /// this machine only, so a peer that imported it would route to a path
    /// that resolves to nothing on its own host.
    fn is_publishable(&self, worker_id: &WorkerId, worker: &Arc<dyn Worker>) -> bool {
        self.worker_registry.origin_of(worker_id) == Some(WorkerOrigin::Local)
            && *worker.connection_mode() != ConnectionMode::Zmq
    }

    fn apply_incoming(&self, worker_id: &str, bytes: &[u8]) {
        match bincode::deserialize::<WorkerState>(bytes) {
            Ok(state) => self.worker_registry.on_remote_worker_state(&state),
            Err(err) => warn!(worker_id, %err, "failed to decode WorkerState"),
        }
    }

    /// Publish loop: forward every locally-owned worker mutation to the
    /// mesh. `published` tracks the keys this loop has written so removals
    /// can be tombstoned even after the registry has dropped the worker's
    /// origin entry, and so lag recovery can tombstone removals missed in
    /// the lag window.
    async fn run_outbound(self: Arc<Self>, mut events: broadcast::Receiver<WorkerEvent>) {
        let mut published: HashSet<WorkerId> = HashSet::new();
        self.resync_local(&mut published);
        loop {
            match events.recv().await {
                Ok(WorkerEvent::Registered { worker_id, worker }) => {
                    if self.is_publishable(&worker_id, &worker) {
                        self.on_worker_changed(
                            worker_id.as_str(),
                            &worker_state_of(&worker_id, &worker),
                        );
                        published.insert(worker_id);
                    }
                }
                // Gated on live origin (not the published set) so a worker
                // promoted to Local ownership mid-life — e.g. a mesh import
                // claimed by register_or_replace — starts publishing from its
                // next mutation.
                Ok(WorkerEvent::Replaced { worker_id, new, .. }) => {
                    if self.is_publishable(&worker_id, &new) {
                        self.on_worker_changed(
                            worker_id.as_str(),
                            &worker_state_of(&worker_id, &new),
                        );
                        published.insert(worker_id);
                    } else if published.remove(&worker_id) {
                        // The replacement is no longer exportable (e.g. the
                        // endpoint moved to a host-local ZMQ transport):
                        // withdraw the key peers already imported.
                        self.on_worker_removed(worker_id.as_str());
                    }
                }
                Ok(WorkerEvent::StatusChanged {
                    worker_id, worker, ..
                }) => {
                    if self.is_publishable(&worker_id, &worker) {
                        self.on_worker_changed(
                            worker_id.as_str(),
                            &worker_state_of(&worker_id, &worker),
                        );
                        published.insert(worker_id);
                    }
                }
                Ok(WorkerEvent::Removed { worker_id, .. }) => {
                    if published.remove(&worker_id) {
                        self.on_worker_removed(worker_id.as_str());
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!(lagged = n, "outbound worker sync lagged; resyncing");
                    self.resync_local(&mut published);
                }
                Err(broadcast::error::RecvError::Closed) => {
                    debug!("WorkerSyncAdapter outbound loop closed");
                    return;
                }
            }
        }
    }

    /// Re-publish every locally-owned worker and tombstone any key this
    /// loop published whose worker no longer exists. Doubles as the
    /// bootstrap (empty prior set) and the lag-recovery path. Workers whose
    /// stored state is already equivalent are skipped: every re-put mints a
    /// fresh Lamport op, invalidating each peer's send watermark for the
    /// key, so a blanket resync of W workers would burst W ops to every
    /// peer even when nothing changed.
    fn resync_local(&self, published: &mut HashSet<WorkerId>) {
        let mut current = HashSet::new();
        for (id, worker) in self.worker_registry.get_all_with_ids() {
            if self.is_publishable(&id, &worker) {
                let state = worker_state_of(&id, &worker);
                if !self.store_matches(&id, &state) {
                    self.on_worker_changed(id.as_str(), &state);
                }
                current.insert(id);
            }
        }
        for stale in published.difference(&current) {
            info!(worker_id = %stale.as_str(), "tombstoning worker removed during lag");
            self.on_worker_removed(stale.as_str());
        }
        *published = current;
    }

    /// True when the store already holds an equivalent state for the
    /// worker. `load` is not compared (volatile and unread by importers);
    /// specs compare semantically as JSON values — `WorkerSpec` holds maps,
    /// so two encodings of identical specs can differ byte-wise. Cheap
    /// scalar fields and a byte-equality fast path gate the JSON parses.
    fn store_matches(&self, id: &WorkerId, state: &WorkerState) -> bool {
        let Some(bytes) = self.workers.get(&format!("{PREFIX}{}", id.as_str())) else {
            return false;
        };
        let Ok(stored) = bincode::deserialize::<WorkerState>(&bytes) else {
            return false;
        };
        if stored.worker_id != state.worker_id
            || stored.model_id != state.model_id
            || stored.url != state.url
            || stored.health != state.health
            || stored.version != state.version
        {
            return false;
        }
        // Byte equality is sufficient (not necessary): re-serialising the
        // same spec instance is byte-stable in-process, so this skips the
        // JSON parses in the steady state; key-order drift (e.g. across
        // restarts) falls through to the semantic comparison.
        if stored.spec == state.spec {
            return true;
        }
        match (
            serde_json::from_slice::<serde_json::Value>(&stored.spec),
            serde_json::from_slice::<serde_json::Value>(&state.spec),
        ) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }

    /// Publish a worker update to the cluster. Callers pass the
    /// registry's current state; the adapter owns (de)serialisation
    /// and key formatting.
    pub fn on_worker_changed(&self, worker_id: &str, state: &WorkerState) {
        match bincode::serialize(state) {
            Ok(bytes) => self.workers.put(&format!("{PREFIX}{worker_id}"), bytes),
            Err(err) => warn!(worker_id, %err, "failed to serialize WorkerState"),
        }
    }

    /// Publish a tombstone for a worker, removing it from the CRDT.
    pub fn on_worker_removed(&self, worker_id: &str) {
        self.workers.delete(&format!("{PREFIX}{worker_id}"));
    }
}

/// Snapshot a live worker into the mesh wire shape. `spec` carries the
/// `WorkerSpec` (minus `api_key`, which never leaves the node) so the
/// importing side can rebuild the worker; `version` is the registry
/// revision (informational — CRDT ordering is owned by the Lamport clock).
///
/// The spec is JSON, not bincode: `WorkerSpec` uses `skip_serializing_*`
/// serde attributes, which a positional format cannot round-trip — bincode
/// deserialization fails for every spec, silently downgrading imports to
/// the minimal builder. JSON is self-describing and round-trips them.
fn worker_state_of(worker_id: &WorkerId, worker: &Arc<dyn Worker>) -> WorkerState {
    let spec = serde_json::to_vec(&worker.metadata().spec).unwrap_or_else(|err| {
        warn!(url = %worker.url(), %err, "failed to encode WorkerSpec; publishing without spec");
        Vec::new()
    });
    WorkerState {
        worker_id: worker_id.as_str().to_string(),
        model_id: worker.model_id().to_string(),
        url: worker.url().to_string(),
        health: worker.is_healthy(),
        load: worker.load() as f64,
        version: worker.revision(),
        spec,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
    use smg_mesh::{MergeStrategy, MeshKV};
    use tokio::{sync::mpsc::error::TryRecvError, time::sleep};

    use super::*;
    use crate::worker::BasicWorkerBuilder;

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
        }
    }

    fn local_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("llama-3"))
                .build(),
        )
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry);

        let state = sample_state("w1", "http://worker-a:8080");
        adapter.on_worker_changed("w1", &state);

        let raw = ns.get("worker:w1").expect("adapter wrote through to CRDT");
        let decoded: WorkerState = bincode::deserialize(&raw).unwrap();
        assert_eq!(decoded, state);
    }

    #[tokio::test]
    async fn on_worker_removed_tombstones_the_entry() {
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry);

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
        let subscriber = WorkerSyncAdapter::new(ns.clone(), registry.clone());
        subscriber.start();

        let publisher_registry = Arc::new(WorkerRegistry::new());
        let publisher = WorkerSyncAdapter::new(ns, publisher_registry);
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
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
        let adapter = WorkerSyncAdapter::new(ns, registry.clone());
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
        let _ = WorkerSyncAdapter::new(ns, registry);
    }

    #[tokio::test]
    async fn remote_tombstone_removes_mesh_imported_worker() {
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
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
        let mut published = HashSet::new();
        adapter.resync_local(&mut published);
        assert!(
            ns.get(&format!("worker:{}", zmq_id.as_str())).is_none(),
            "resync must not publish a ZMQ worker"
        );
        assert!(published.contains(&http_id) && !published.contains(&zmq_id));
    }

    #[tokio::test]
    async fn outbound_never_republishes_mesh_imported_worker() {
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
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
    async fn rapid_put_then_delete_converges_to_absent() {
        // Both queued events resolve against the store (which holds the
        // tombstone by the time they drain), so a stale put echo can never
        // resurrect a removed worker regardless of drain timing.
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
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
        let publisher = WorkerSyncAdapter::new(ns.clone(), pub_registry.clone());
        publisher.start();

        let sub_registry = Arc::new(WorkerRegistry::new());
        let subscriber = WorkerSyncAdapter::new(ns, sub_registry.clone());
        subscriber.start();

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://remote:9000")
                .model(ModelCard::new("llama-3"))
                .connection_mode(ConnectionMode::Grpc)
                .worker_type(crate::worker::WorkerType::Decode)
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
            crate::worker::WorkerType::Decode,
            "worker type must survive the spec round trip"
        );
    }

    #[tokio::test]
    async fn claimed_import_publishes_from_next_mutation() {
        // The restart race end-to-end: a mesh import wins the URL, the local
        // workflow claims it via register_or_replace, and the worker must be
        // published from its next mutation onward.
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());

        let id = registry
            .register(local_worker("http://local:8080"))
            .unwrap();
        let mut published = HashSet::new();
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = WorkerSyncAdapter::new(ns.clone(), registry.clone());

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

        let mut published: HashSet<WorkerId> = [phantom].into_iter().collect();
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
            published,
            [local_id].into_iter().collect(),
            "published set tracks exactly the local workers"
        );
    }
}
