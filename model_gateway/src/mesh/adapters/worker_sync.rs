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
//! `DEAD_OWNER_GRACE`. The key itself stays until its owner returns and
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
//! Restarts: a node that comes back under its name after a crash registers
//! its endpoints under fresh ids, and the keys of its earlier life gossip
//! back to it from the peers (the owner is alive, so no peer treats them as
//! dead). A key that names this node as its owner whose endpoint a `Local`
//! worker of this process holds under another id is such a leftover: the
//! node tombstones it, being its owner. A key whose endpoint this process
//! does not hold yet imports as any other (gossip serves before the start-up
//! jobs register the configured workers), and the start-up upsert claims it
//! under its id. A peer whose import a tombstone removes re-imports the
//! endpoint from the live key that names it in the same step, so the peer
//! rebinds to the new id without waiting for a pass, and the owner's later
//! tombstone of that id reaches the import.
//!
//! Remaining gaps: a key of this node's name whose endpoint the new life
//! never registers again (it left the configuration across the crash) is
//! imported as a `Mesh` worker on this node and on every peer and tombstoned
//! by nobody (it is in no outbound `published` set and it is no leftover by
//! the rule above), so it stays in each registry for as long as its owner is
//! live: routable while its probes pass, demoted to `Failed` once they fail
//! (a `Mesh` worker is never removed for failed probes), and imported again
//! from the store after a restart; a pass that tombstones a `Mesh` import
//! whose key names this node, once the start-up jobs are done, is its rule.
//! Tombstone metadata accrues per removed worker (time-based collection would
//! resurrect deleted keys; it is only sound at causal stability).

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
        endpoint_key,
        identity::endpoint::Endpoint,
        registry::{event::WorkerEvent, WorkerId},
        ConnectionMode, EndpointKey, Worker, WorkerOrigin, WorkerRegistry,
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
                if let Some(url) = this.sync_key_from_store(&key, worker_id) {
                    this.reimport_endpoint(&url);
                }
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
        for (id, worker) in self.worker_registry.get_all_with_ids() {
            let key = format!("{PREFIX}{}", id.as_str());
            if live.contains(&key) {
                continue;
            }
            // A local worker this node never publishes (`is_publishable`
            // keeps a host-local ZMQ worker off the mesh) has no key by
            // design: nothing is missing for it, where a published worker's
            // missing key is a foreign tombstone to re-assert and an
            // import's a tombstone to apply.
            let never_published = self.worker_registry.origin_of(&id) == Some(WorkerOrigin::Local)
                && !self.is_publishable(&id, &worker);
            if never_published {
                continue;
            }
            // Missing backing key: same handling as a live tombstone
            // event — imports are removed, locally-owned state is
            // re-asserted. What a tombstone vacates, the backfill below
            // refills from the live keys.
            self.sync_key_from_store(&key, id.as_str());
        }
        self.backfill_keys(live);
    }

    /// Bring the registry in line with the store's current state for one
    /// key: a live value routes through `on_remote_worker_state`; a missing
    /// (tombstoned) one removes a mesh import, re-asserts a locally-owned
    /// worker's state, and ignores unknown ids. Tombstones resolve via the
    /// publisher's id, which the import adopted. Returns the endpoint a
    /// removed import held, for the caller to refill from another live key
    /// when no backfill of its own follows.
    fn sync_key_from_store(&self, key: &str, worker_id: &str) -> Option<String> {
        match self.workers.get(key) {
            Some(bytes) => {
                self.apply_incoming(worker_id, &bytes);
                None
            }
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
                        return None;
                    }
                }
                match self.worker_registry.remove_remote(&id) {
                    Some(worker) => {
                        info!(worker_id, "removed worker on remote tombstone");
                        Some(worker.url().to_owned())
                    }
                    None => {
                        debug!(worker_id, "ignored tombstone (unknown id)");
                        None
                    }
                }
            }
        }
    }

    /// The endpoint a tombstone just vacated, imported again from another
    /// live key that names it, as the reconcile pass does within one pass
    /// (missing keys first, live keys after). On the event path the live key
    /// was applied before the tombstone and only refreshed the import's
    /// health, so without this step the peer has no worker at the endpoint
    /// until its next pass: a reconcile interval at most, on every owner
    /// restart (the owner's new key for the endpoint arrives before its
    /// tombstone of the old one) and whenever one of several publishers of
    /// an endpoint removes it. Each key's value is decoded once, and the
    /// walk ends at the first key whose import lands.
    fn reimport_endpoint(&self, url: &str) {
        for key in self.workers.keys("") {
            let Some(worker_id) = key.strip_prefix(PREFIX).filter(|s| !s.is_empty()) else {
                continue;
            };
            let Some(state) = self
                .workers
                .get(&key)
                .and_then(|bytes| decode_worker_state(&bytes).ok())
                .filter(|state| state.url == url)
            else {
                continue;
            };
            self.apply_state(worker_id, &state);
            if self.worker_registry.get_by_url(url).is_some() {
                return;
            }
        }
    }

    /// Whether a key is the leftover of this node's earlier life under the
    /// same name: it names this node as its owner, no worker of this process
    /// holds its id, and a `Local` worker of this process holds its URL under
    /// another id (the restart registered the endpoint again before the key
    /// gossiped back). The earlier life could not tombstone what it had
    /// published (a crash); only the owner tombstones its keys, and this
    /// process is that owner now. A key of this node's name whose endpoint
    /// this process does not hold yet is no leftover: gossip serves before
    /// the start-up jobs register the configured workers, so the key imports
    /// as before and the start-up upsert claims it under the same id, and the
    /// peers see no change. This node's own live keys echo back through the
    /// same path and are held, so they are never taken for leftovers. Two
    /// nodes under one name would tombstone each other's keys: names are
    /// unique by contract, the membership table is keyed by them. One id
    /// under two names is possible (a discovery registers an endpoint another
    /// member published, and the registration claims the import with its
    /// id): the two write the key in turns under their own names, and a
    /// member back from a crash whose name the key last carried tombstones it
    /// although the other holds the id. The other re-publishes it on the
    /// tombstone (the foreign-tombstone branch of `sync_key_from_store`), so
    /// the imports of that id are gone until its re-put arrives, and no
    /// longer. A key of this name that the new life never claims again (its
    /// endpoint left the configuration) stays, as before this rule.
    fn is_own_orphan(&self, worker_id: &str, state: &WorkerState) -> bool {
        state.owner == self.node_name
            && self
                .worker_registry
                .get(&WorkerId::from_string(worker_id.to_string()))
                .is_none()
            && self
                .worker_registry
                .get_id_by_url(&state.url)
                .is_some_and(|held| {
                    held.as_str() != worker_id
                        && self.worker_registry.origin_of(&held) == Some(WorkerOrigin::Local)
                })
    }

    /// A key of this node's name whose id a `Local` worker of this process
    /// holds (the start-up claim of an earlier life's key), carrying a
    /// specification this node did not publish at or above the worker's
    /// revision: the earlier life's later publication, seen by a peer first
    /// and gossiped here after the claim. The claim raised the revision past
    /// the version this node had seen, not past this one, so the peers that
    /// hold it keep the earlier life's specification. The revision is raised
    /// past it and the worker published again at once. Only what a revision
    /// covers counts (`specification_matches`): health is not versioned, so
    /// this node's own state with another health at the same version (a
    /// status change the outbound loop has not put yet) is left to that loop,
    /// an echo of this node's own publication is left alone, and a lower
    /// version is outranked already.
    fn republish_above_own_key(&self, worker_id: &str, state: &WorkerState) -> bool {
        if state.owner != self.node_name {
            return false;
        }
        let id = WorkerId::from_string(worker_id.to_string());
        let Some(worker) = self.worker_registry.get(&id) else {
            return false;
        };
        if !self.is_publishable(&id, &worker) {
            return false;
        }
        let current = worker_state_of(&id, &worker, &self.node_name);
        let stored_outranks = state.version > current.version
            || (state.version == current.version && !specification_matches(state, &current));
        if !stored_outranks {
            return false;
        }
        worker.raise_revision_to(state.version + 1);
        let raised = worker_state_of(&id, &worker, &self.node_name);
        info!(
            worker_id,
            url = %Endpoint::redacted_str(&state.url),
            stored = state.version,
            published = raised.version,
            "publishing a claimed worker above a later state of its key: a peer saw the earlier \
             life's last publication first"
        );
        self.on_worker_changed(worker_id, &raised);
        true
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
                        url = %Endpoint::redacted_str(&state.url),
                        worker_id = %id.as_str(),
                        "removed mesh-imported worker: its owner node is down"
                    );
                }
            }
            None => debug!(
                owner = %state.owner,
                url = %Endpoint::redacted_str(&state.url),
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
                    url = %Endpoint::redacted_str(&state.url),
                    worker_id = %id.as_str(),
                    "removed mesh-imported worker: this gateway removed its endpoint"
                );
            }
        }
    }

    fn apply_incoming(&self, worker_id: &str, bytes: &[u8]) {
        match decode_worker_state(bytes) {
            Ok(state) => self.apply_state(worker_id, &state),
            Err(err) => warn!(worker_id, %err, "failed to decode WorkerState"),
        }
    }

    /// A live key's state into the registry: a leftover of this node's
    /// earlier life is tombstoned, a later state of a key this node claimed
    /// is published over, a dead owner's import dropped, a gone owner's left
    /// alone inside the grace, a live owner's imported or refreshed unless
    /// this node retired the endpoint.
    fn apply_state(&self, worker_id: &str, state: &WorkerState) {
        if self.is_own_orphan(worker_id, state) {
            info!(
                worker_id,
                url = %Endpoint::redacted_str(&state.url),
                "tombstoning a key of this node's name whose endpoint this process holds under \
                 another id: left by an earlier life of the node"
            );
            self.on_worker_removed(worker_id);
            return;
        }
        if self.republish_above_own_key(worker_id, state) {
            return;
        }
        match self.owner_liveness(&state.owner) {
            OwnerLiveness::Dead => self.drop_dead_owner_import(state),
            // Inside the grace a gone owner's key creates no import and
            // refreshes none: a router that starts after an owner died never
            // imports its keys, while an owner that blips keeps its imports
            // and has new ones delayed by a pass at most.
            OwnerLiveness::GoneWithinGrace => debug!(
                owner = %state.owner,
                url = %Endpoint::redacted_str(&state.url),
                "leaving worker state alone: its owner node is gone, inside the grace"
            ),
            OwnerLiveness::Live => {
                if self.holds_back(state) {
                    self.drop_held_back_import(state);
                } else {
                    self.worker_registry.on_remote_worker_state(state);
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
                url = %Endpoint::redacted_str(&state.url),
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
    /// worker (`states_match`).
    fn store_matches(&self, id: &WorkerId, state: &WorkerState) -> bool {
        let Some(bytes) = self.workers.get(&format!("{PREFIX}{}", id.as_str())) else {
            return false;
        };
        let Ok(stored) = bincode::deserialize::<WorkerState>(&bytes) else {
            return false;
        };
        states_match(&stored, state)
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
        warn!(url = %worker.redacted_url(), %err, "failed to encode WorkerSpec; publishing without spec");
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

/// Whether two states of one key are equivalent: the same specification at
/// the same version and health. `load` is not compared (volatile and unread
/// by importers).
fn states_match(stored: &WorkerState, state: &WorkerState) -> bool {
    stored.health == state.health
        && stored.version == state.version
        && specification_matches(stored, state)
}

/// Whether two states of one key carry the same specification: the fields a
/// revision covers (id, model, url, owner and the spec), not health or load.
/// Specs compare semantically as JSON values — `WorkerSpec` holds maps, so
/// two encodings of identical specs can differ byte-wise. Cheap scalar fields
/// and a byte-equality fast path gate the JSON parses.
fn specification_matches(stored: &WorkerState, state: &WorkerState) -> bool {
    if stored.worker_id != state.worker_id
        || stored.model_id != state.model_id
        || stored.url != state.url
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
mod tests;
