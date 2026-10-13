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
//! Dead owners: only the owner tombstones its keys, so a node that leaves
//! for good (a rollout, a crash) leaves them behind, and a URL the live
//! nodes later drop would be re-imported from such a key with the stale
//! `health=true`. Every published state therefore names its owner, and the
//! inbound side ignores (and removes a registered import of) any key whose
//! owner the cluster state has shown as `Down` or `Leaving` for
//! [`DEAD_OWNER_GRACE`]. The key itself stays until its owner returns and
//! re-publishes or tombstones it; a returning owner's keys import again.
//!
//! Removed endpoints: the adapter also remembers every endpoint this gateway
//! removed locally (it is where this owner's keys are tombstoned), with the
//! instance that held it, and holds back a peer's state for the same
//! instance, or one naming no instance, until a fresh local registration
//! re-admits the endpoint. That covers the keys the owner rule cannot see
//! (an owner this node cannot tell is gone, a peer that is behind) without
//! the registry or service discovery knowing anything about the mesh.
//!
//! Remaining gap: a crash-restart that registers locally before its old
//! state gossips back orphans up to one store key per worker per restart
//! (the owner is alive, so the key is not dead). Tombstone metadata also
//! accrues per removed worker (time-based collection would resurrect
//! deleted keys; it is only sound at causal stability).

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Duration,
};

use parking_lot::Mutex;
use smg_mesh::{gossip::NodeStatus, ClusterState, CrdtNamespace, WorkerState};
use tokio::{sync::broadcast, time::Instant};
use tracing::{debug, info, warn};

use crate::{
    service_discovery::DISCOVERY_ID_LABEL,
    worker::{
        endpoint_key, event::WorkerEvent, registry::WorkerId, ConnectionMode, EndpointKey, Worker,
        WorkerOrigin, WorkerRegistry,
    },
};

const PREFIX: &str = "worker:";

/// Cadence of the inbound store↔registry reconcile pass.
const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);

/// How long the cluster state must show a peer as `Down` or `Leaving` before
/// its `worker:` keys count as dead. The grace protects an existing import
/// from a one-round blip: inside it a gone owner's key removes nothing, and
/// creates nothing either. Longer than the gossip layer's own suspicion
/// step; shorter than [`RECONCILE_INTERVAL`], so a peer seen gone on two
/// consecutive passes is dead by the second.
const DEAD_OWNER_GRACE: Duration = Duration::from_secs(20);

/// How a key's owner stands in the membership view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OwnerLiveness {
    /// Alive, or not a node this one can judge: an empty name from an older
    /// publisher, a node this one has not learnt yet, or this node itself.
    Live,
    /// Shown `Down` or `Leaving`, for less than [`DEAD_OWNER_GRACE`] so far.
    GoneWithinGrace,
    /// Shown `Down` or `Leaving` for at least [`DEAD_OWNER_GRACE`].
    Dead,
}

/// Upper bound on the removed endpoints the adapter remembers.
const RETIRED_ENDPOINTS_CAP: usize = 4096;

/// Bridge between the `worker:` CRDT namespace and the gateway's
/// in-process `WorkerRegistry`.
pub struct WorkerSyncAdapter {
    workers: Arc<CrdtNamespace>,
    worker_registry: Arc<WorkerRegistry>,
    /// This node's mesh server name: the owner stamped on every key it
    /// publishes, and its own entry in the cluster state.
    node_name: String,
    /// The gossip membership view, read to tell a live owner from a dead one.
    cluster_state: ClusterState,
    /// When each peer was first seen `Down` or `Leaving`, refreshed once per
    /// reconcile pass; a peer that is alive again drops out.
    down_since: Mutex<HashMap<String, Instant>>,
    /// Endpoints this gateway removed locally, with the instance that held
    /// each; a peer's state for them is held back until a fresh local
    /// registration re-admits the endpoint.
    retired: Mutex<RetiredEndpoints>,
}

/// A locally published worker as the outbound loop remembers it, so a removal
/// (seen as an event, or found in a lag resync) can retire its endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Published {
    url: String,
    /// The provider's instance id the worker carried, if any.
    instance: Option<String>,
}

/// Endpoints this gateway removed locally. Bounded: past
/// [`RETIRED_ENDPOINTS_CAP`] the oldest endpoints are dropped.
#[derive(Debug, Default)]
struct RetiredEndpoints {
    by_endpoint: HashMap<EndpointKey, RetiredEndpoint>,
    /// Insertion order of the endpoints, for the cap.
    order: VecDeque<EndpointKey>,
}

#[derive(Debug)]
struct RetiredEndpoint {
    /// The instance that held the endpoint when it was removed, if the worker
    /// carried a provider instance id.
    instance: Option<String>,
    /// Whether the first held-back state has been logged (once per endpoint).
    logged: bool,
}

impl std::fmt::Debug for WorkerSyncAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerSyncAdapter")
            .field("prefix", &self.workers.prefix())
            .field("node_name", &self.node_name)
            .finish_non_exhaustive()
    }
}

impl WorkerSyncAdapter {
    /// Build an adapter wrapping a `worker:`-scoped namespace and the
    /// gateway's worker registry. `node_name` is this node's mesh server
    /// name (the owner it stamps on its keys) and `cluster_state` the
    /// gossip membership view it reads owners' liveness from. Panics if
    /// the namespace is not scoped to `worker:` or the node name is empty,
    /// so a mis-wired caller fails fast at startup rather than silently
    /// routing updates to the wrong prefix or publishing ownerless keys.
    pub fn new(
        workers: Arc<CrdtNamespace>,
        worker_registry: Arc<WorkerRegistry>,
        node_name: String,
        cluster_state: ClusterState,
    ) -> Arc<Self> {
        assert_eq!(
            workers.prefix(),
            PREFIX,
            "WorkerSyncAdapter requires a namespace scoped to `{PREFIX}`",
        );
        assert!(
            !node_name.is_empty(),
            "WorkerSyncAdapter requires a non-empty node name"
        );
        Arc::new(Self {
            workers,
            worker_registry,
            node_name,
            cluster_state,
            down_since: Mutex::new(HashMap::new()),
            retired: Mutex::new(RetiredEndpoints::default()),
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
        self.refresh_membership();
        self.backfill_keys(self.workers.keys(""));
    }

    /// Dead owners' keys go first: a dead key may remove the import at a URL
    /// that a live owner's key then re-imports in the same pass, where the
    /// other order would refresh an import that is removed right after and
    /// leave the URL out until the next pass. The first loop only classifies;
    /// every live key is read again from the store when it is applied, so a
    /// value from the scan that a tombstone overtook meanwhile is never
    /// applied (the subscription task acts on store truth the same way).
    fn backfill_keys(&self, keys: impl IntoIterator<Item = String>) {
        let mut live = Vec::new();
        for key in keys {
            let Some(worker_id) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                warn!(key, "worker: backfill yielded unexpected key shape");
                continue;
            };
            let worker_id = worker_id.to_string();
            match self
                .workers
                .get(&key)
                .map(|bytes| decode_worker_state(&bytes))
            {
                Some(Ok(state)) if self.owner_liveness(&state.owner) == OwnerLiveness::Dead => {
                    self.drop_dead_owner_import(&state);
                }
                // Live or inside the grace, undecodable (the apply logs it),
                // or tombstoned between the key scan and the read: all
                // resolve against the store.
                _ => live.push((key, worker_id)),
            }
        }
        for (key, worker_id) in live {
            self.sync_key_from_store(&key, &worker_id);
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
        self.refresh_membership();
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
                        self.on_worker_changed(
                            worker_id,
                            &worker_state_of(&id, &worker, &self.node_name),
                        );
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

    /// Re-read the cluster state: a peer shown `Down` or `Leaving` keeps the
    /// time it was first seen so; any other status (or no entry) clears it.
    fn refresh_membership(&self) {
        let now = Instant::now();
        let state = self.cluster_state.read();
        let mut down = self.down_since.lock();
        down.retain(|name, _| state.get(name).is_some_and(|node| is_gone(node.status)));
        for (name, node) in state.iter() {
            if is_gone(node.status) {
                down.entry(name.clone()).or_insert(now);
            }
        }
    }

    /// How `owner` stands in the membership view. A key is only held or
    /// dropped on positive evidence: an owner this node cannot judge is live.
    fn owner_liveness(&self, owner: &str) -> OwnerLiveness {
        if owner.is_empty() || owner == self.node_name {
            return OwnerLiveness::Live;
        }
        let gone_now = self
            .cluster_state
            .read()
            .get(owner)
            .is_some_and(|node| is_gone(node.status));
        if !gone_now {
            return OwnerLiveness::Live;
        }
        // Dead once first seen gone at least a grace ago: the first-seen time
        // is refreshed once per pass, so on the live path it can only delay a
        // "dead" verdict, never extend one past the owner's return.
        match self.down_since.lock().get(owner) {
            Some(since) if since.elapsed() >= DEAD_OWNER_GRACE => OwnerLiveness::Dead,
            _ => OwnerLiveness::GoneWithinGrace,
        }
    }

    /// The import `state`'s key created, if it exists. An import adopts the
    /// publisher's id, so an import of the same URL that came from another
    /// key holds a different id and is left to that key: several live
    /// routers publishing one URL is the normal case.
    fn import_of_key(&self, state: &WorkerState) -> Option<WorkerId> {
        self.worker_registry
            .get_id_by_url(&state.url)
            .filter(|id| id.as_str() == state.worker_id)
            .filter(|id| self.worker_registry.origin_of(id) == Some(WorkerOrigin::Mesh))
    }

    /// A dead owner's key is neither imported nor replayed into an import's
    /// health, and the import this key created goes the way the owner's own
    /// tombstone would have sent it. The key stays in the store: only the
    /// owner tombstones it, and once the owner is alive again its keys
    /// import again on their next event or pass.
    fn drop_dead_owner_import(&self, state: &WorkerState) {
        match self.import_of_key(state) {
            Some(id) => {
                if self.worker_registry.remove_remote(&id).is_some() {
                    info!(
                        owner = %state.owner,
                        url = %state.url,
                        worker_id = %id.as_str(),
                        "removed mesh-imported worker: its owner node is down"
                    );
                }
            }
            None => debug!(
                owner = %state.owner,
                url = %state.url,
                "ignoring worker state published by a node that is down"
            ),
        }
    }

    /// A held-back key's import is removed as well. The retirement is
    /// recorded by the outbound loop, after the registry has dropped the
    /// worker: a pass or event in that gap imports the endpoint, and the hold
    /// then takes it out on the key's next event or pass instead of keeping
    /// it from ever being cleaned up. This cannot churn: a key for another
    /// instance at the address holds a different id and is not held back.
    fn drop_held_back_import(&self, state: &WorkerState) {
        if let Some(id) = self.import_of_key(state) {
            if self.worker_registry.remove_remote(&id).is_some() {
                info!(
                    url = %state.url,
                    worker_id = %id.as_str(),
                    "removed mesh-imported worker: this gateway removed its endpoint"
                );
            }
        }
    }

    fn apply_incoming(&self, worker_id: &str, bytes: &[u8]) {
        let state = match decode_worker_state(bytes) {
            Ok(state) => state,
            Err(err) => {
                warn!(worker_id, %err, "failed to decode WorkerState");
                return;
            }
        };
        match self.owner_liveness(&state.owner) {
            OwnerLiveness::Dead => self.drop_dead_owner_import(&state),
            // Inside the grace a gone owner's key creates no import and
            // refreshes none: a router that starts after an owner died never
            // imports its keys, while an owner that blips keeps its imports
            // and has new ones delayed by a pass at most.
            OwnerLiveness::GoneWithinGrace => debug!(
                owner = %state.owner,
                url = %state.url,
                "leaving worker state alone: its owner node is gone, inside the grace"
            ),
            OwnerLiveness::Live => {
                if self.holds_back(&state) {
                    self.drop_held_back_import(&state);
                } else {
                    self.worker_registry.on_remote_worker_state(&state);
                }
            }
        }
    }

    /// Remember an endpoint this gateway removed locally, with the instance
    /// that held it, so a peer's state does not bring it back: the live peers
    /// remove it too and a dead one's key falls to the owner rule; this guards
    /// the keys neither covers (an owner this node cannot tell is gone, a peer
    /// that is behind). A fresh local registration re-admits the endpoint.
    fn retire(&self, url: &str, instance: Option<String>) {
        let Ok(key) = endpoint_key(url) else {
            return;
        };
        let mut retired = self.retired.lock();
        let RetiredEndpoints { by_endpoint, order } = &mut *retired;
        if !by_endpoint.contains_key(&key) {
            order.push_back(key.clone());
        }
        by_endpoint.insert(
            key,
            RetiredEndpoint {
                instance,
                logged: false,
            },
        );
        while by_endpoint.len() > RETIRED_ENDPOINTS_CAP {
            match order.pop_front() {
                Some(oldest) => {
                    by_endpoint.remove(&oldest);
                }
                None => break,
            }
        }
    }

    /// A local registration at `url` re-admits the endpoint for import.
    fn readmit(&self, url: &str) {
        let Ok(key) = endpoint_key(url) else {
            return;
        };
        let mut retired = self.retired.lock();
        if retired.by_endpoint.remove(&key).is_some() {
            retired.order.retain(|k| k != &key);
        }
    }

    /// Whether a peer's `state` is for an endpoint this gateway removed and is
    /// held back: it is unless it names a different instance than the one
    /// removed (a reused address with a new instance imports as usual). A
    /// state without a decodable spec names no instance and is held back, as
    /// is every state when the removed worker carried no instance id. The
    /// first held-back state per endpoint is logged at info. The caller also
    /// removes the import the key may already have created.
    fn holds_back(&self, state: &WorkerState) -> bool {
        let Ok(key) = endpoint_key(&state.url) else {
            return false;
        };
        let mut retired = self.retired.lock();
        let Some(entry) = retired.by_endpoint.get_mut(&key) else {
            return false;
        };
        let incoming = instance_of(&state.spec);
        if let (Some(removed), Some(incoming)) = (&entry.instance, &incoming) {
            if removed != incoming {
                return false;
            }
        }
        if !entry.logged {
            entry.logged = true;
            info!(
                url = %state.url,
                removed_instance = ?entry.instance,
                incoming_instance = ?incoming,
                "Ignoring a peer's state for an endpoint this gateway removed (logged once per endpoint)"
            );
        }
        true
    }

    /// Publish loop: forward every locally-owned worker mutation to the
    /// mesh. `published` tracks the keys this loop has written so removals
    /// can be tombstoned even after the registry has dropped the worker's
    /// origin entry, and so lag recovery can tombstone removals missed in
    /// the lag window.
    async fn run_outbound(self: Arc<Self>, mut events: broadcast::Receiver<WorkerEvent>) {
        let mut published: HashMap<WorkerId, Published> = HashMap::new();
        self.resync_local(&mut published);
        loop {
            match events.recv().await {
                Ok(WorkerEvent::Registered { worker_id, worker }) => {
                    if self.is_publishable(&worker_id, &worker) {
                        self.on_worker_changed(
                            worker_id.as_str(),
                            &worker_state_of(&worker_id, &worker, &self.node_name),
                        );
                        // A fresh local registration re-admits its endpoint.
                        self.readmit(worker.url());
                        published.insert(worker_id, published_of(&worker));
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
                            &worker_state_of(&worker_id, &new, &self.node_name),
                        );
                        self.readmit(new.url());
                        published.insert(worker_id, published_of(&new));
                    } else if published.remove(&worker_id).is_some() {
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
                            &worker_state_of(&worker_id, &worker, &self.node_name),
                        );
                        published.insert(worker_id, published_of(&worker));
                    }
                }
                Ok(WorkerEvent::Removed { worker_id, .. }) => {
                    if let Some(removed) = published.remove(&worker_id) {
                        self.on_worker_removed(worker_id.as_str());
                        // This gateway removed the worker: its endpoint is not
                        // re-imported from a peer until registered here again.
                        self.retire(&removed.url, removed.instance);
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
    fn resync_local(&self, published: &mut HashMap<WorkerId, Published>) {
        let mut current = HashMap::new();
        for (id, worker) in self.worker_registry.get_all_with_ids() {
            if self.is_publishable(&id, &worker) {
                let state = worker_state_of(&id, &worker, &self.node_name);
                if !self.store_matches(&id, &state) {
                    self.on_worker_changed(id.as_str(), &state);
                }
                current.insert(id, published_of(&worker));
            }
        }
        for (stale, removed) in published
            .iter()
            .filter(|(id, _)| !current.contains_key(*id))
        {
            info!(worker_id = %stale.as_str(), "tombstoning worker removed during lag");
            self.on_worker_removed(stale.as_str());
            self.retire(&removed.url, removed.instance.clone());
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
            || stored.owner != state.owner
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
fn worker_state_of(worker_id: &WorkerId, worker: &Arc<dyn Worker>, owner: &str) -> WorkerState {
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
        owner: owner.to_string(),
    }
}

fn is_gone(status: i32) -> bool {
    status == NodeStatus::Down as i32 || status == NodeStatus::Leaving as i32
}

/// What the outbound loop remembers about a worker it published.
fn published_of(worker: &Arc<dyn Worker>) -> Published {
    Published {
        url: worker.url().to_string(),
        instance: worker
            .metadata()
            .spec
            .labels
            .get(DISCOVERY_ID_LABEL)
            .cloned(),
    }
}

/// The provider's instance id a published spec carries, if it decodes and
/// has one.
fn instance_of(spec: &[u8]) -> Option<String> {
    if spec.is_empty() {
        return None;
    }
    serde_json::from_slice::<openai_protocol::worker::WorkerSpec>(spec)
        .ok()
        .and_then(|spec| spec.labels.get(DISCOVERY_ID_LABEL).cloned())
}

/// Decode a published state, accepting the layout that predates `owner`.
/// bincode is positional, so a record from an older node is shorter than the
/// current struct and fails its decode; it is read with the old layout and
/// imported with an unknown owner instead of being dropped.
fn decode_worker_state(bytes: &[u8]) -> bincode::Result<WorkerState> {
    bincode::deserialize::<WorkerState>(bytes).or_else(|err| {
        bincode::deserialize::<LegacyWorkerState>(bytes)
            .map(WorkerState::from)
            .map_err(|_| err)
    })
}

/// `WorkerState` as published before it named its owner.
#[derive(serde::Deserialize)]
struct LegacyWorkerState {
    worker_id: String,
    model_id: String,
    url: String,
    health: bool,
    load: f64,
    version: u64,
    spec: Vec<u8>,
}

impl From<LegacyWorkerState> for WorkerState {
    fn from(legacy: LegacyWorkerState) -> Self {
        Self {
            worker_id: legacy.worker_id,
            model_id: legacy.model_id,
            url: legacy.url,
            health: legacy.health,
            load: legacy.load,
            version: legacy.version,
            spec: legacy.spec,
            owner: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, time::Duration};

    use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
    use smg_mesh::{gossip::NodeState, MergeStrategy, MeshKV};
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
        adapter_in(ns, registry, empty_cluster())
    }

    fn adapter_in(
        ns: Arc<CrdtNamespace>,
        registry: Arc<WorkerRegistry>,
        cluster: ClusterState,
    ) -> Arc<WorkerSyncAdapter> {
        WorkerSyncAdapter::new(ns, registry, "node-a".into(), cluster)
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
        let adapter = adapter(ns.clone(), registry);

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
        let adapter = adapter(ns.clone(), registry);

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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = adapter(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = adapter(ns.clone(), registry.clone());
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
        let adapter = adapter(ns.clone(), registry.clone());
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
        let adapter = adapter(ns.clone(), registry.clone());
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
        let adapter = adapter(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = adapter(ns.clone(), registry.clone());
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
        let adapter = adapter(ns.clone(), registry.clone());
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
        let adapter = adapter(ns.clone(), registry.clone());
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
        let adapter = adapter(ns.clone(), registry.clone());
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

        let sub_registry = Arc::new(WorkerRegistry::new());
        let subscriber = adapter(ns, sub_registry.clone());
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
        let adapter = adapter(ns.clone(), registry.clone());
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
        let adapter = adapter(ns.clone(), registry.clone());

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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = adapter(ns.clone(), registry.clone());

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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = adapter(ns.clone(), registry.clone());

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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = adapter(ns.clone(), registry.clone());

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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = adapter(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = adapter(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = adapter(ns.clone(), registry.clone());
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
        let mesh = MeshKV::new("node-a".into());
        let ns = worker_namespace(&mesh);
        let registry = Arc::new(WorkerRegistry::new());
        let adapter = adapter(ns.clone(), registry.clone());

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
}
