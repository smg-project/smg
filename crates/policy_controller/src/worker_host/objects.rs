//! The holder of a replica's worker objects: it follows the store's
//! membership and writes the registry.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
};

use parking_lot::{Mutex, RwLock};
use smg_worker::{
    connection::engine::EngineConnector,
    identity::basic::Worker,
    registry::{Generation, WorkerId as ObjectId, WorkerRegistry},
};

use super::object::{builder, object};
use crate::{
    store::{MembershipChange, MembershipTail, WorkerRow, WorkersStore},
    view::WorkerId,
};

/// What a host keeps of a registry's generation when it derives nothing
/// from one: the members.
#[derive(Debug)]
pub struct Members(Arc<[Arc<dyn Worker>]>);

impl Generation for Members {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        Self(members)
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.0
    }
}

/// The row an object stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Standing {
    /// The row's id in the store: where a write about the object's worker
    /// lands.
    pub row: WorkerId,
    /// The row's revision the object was built from: the guard a write of
    /// the row's health names.
    pub revision: u64,
}

/// Where the last read of the store ended.
#[derive(Clone, Copy, Default)]
struct Place {
    /// The store's epoch read to, `None` before the first read.
    epoch: Option<u64>,
    /// The history of the store that epoch stands in.
    history: u64,
}

/// What stands, both ways.
#[derive(Default)]
struct Held {
    /// A row's object, by the row's id.
    objects: HashMap<WorkerId, ObjectId>,
    /// An object's row, by the object's id in the registry.
    rows: HashMap<ObjectId, Standing>,
}

/// A replica's worker objects: for every row of the membership store that
/// stands one object in the registry, and none beside.
///
/// [`Self::follow`] is the one path, on a leader and on a standby: it reads
/// what the store's membership did since its last read and does one of four
/// things to a row's object. It builds the object of a row that arrived. It
/// drops the object of a row that went. It replaces, under the same id in
/// the registry, the object of a row whose facts changed. It does nothing at
/// a write that moved no fact: a status change, a health write. A new
/// incarnation of a key is another row: the old object is dropped and a new
/// one is built under a new id.
///
/// A row that leaves ends its object, even where a row stands under its id
/// again by the time the change is read: a row registered again under the
/// same key and incarnation is another life under the same id, and starts
/// at the same revision. Its object is dropped and a new one is built.
///
/// A store that took another store's rows whole holds another history: a
/// row may stand there under the id and the revision an object was built
/// from and state other facts. A read the store's tail no longer reaches
/// cannot tell a row that stood through from one that left and came back.
/// The holder trusts no revision it held in either, and lays every row
/// again. A row laid again keeps its place: its object is replaced under
/// its id in the registry, which carries the standing object's running
/// state over.
///
/// The holder is the registry's one writer: an object any other writer left
/// under a row's URL is replaced by the row's.
///
/// The registry numbers its own objects, so an object's id is not its row's.
/// The holder keeps the two together ([`Self::standing`], [`Self::object`]).
/// An object's entry is written before the registry tells of the object, at
/// a build and at a replacement, so a reader the registry's event woke finds
/// it. A reader on a hot path reads it there once and keeps it.
pub struct Objects<G> {
    registry: Arc<WorkerRegistry<G>>,
    /// What makes an object's engine connection, where the host handed one.
    connector: Option<Arc<dyn EngineConnector>>,
    /// Where the last read ended. Held through a read: one follows at a
    /// time.
    read_to: Mutex<Place>,
    held: RwLock<Held>,
}

impl<G> fmt::Debug for Objects<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Objects")
            .field("held", &self.held.read().objects.len())
            .finish_non_exhaustive()
    }
}

impl<G: Generation> Objects<G> {
    /// A holder that writes `registry` and holds nothing yet: the first
    /// [`Self::follow`] builds an object for every row that stands.
    pub fn new(registry: Arc<WorkerRegistry<G>>) -> Self {
        Self {
            registry,
            connector: None,
            read_to: Mutex::new(Place::default()),
            held: RwLock::new(Held::default()),
        }
    }

    /// Hand every object built from here on `connector`, which makes its
    /// engine connection. No row holds one: it is the host's. Without it an
    /// object of a gRPC or a host-local worker refuses its connect, which
    /// suits a host that reads no engine.
    #[must_use]
    pub fn with_connector(mut self, connector: Arc<dyn EngineConnector>) -> Self {
        self.connector = Some(connector);
        self
    }

    /// The registry the objects stand in.
    pub fn registry(&self) -> &Arc<WorkerRegistry<G>> {
        &self.registry
    }

    /// The row the object under `object` stands for, with the revision it
    /// was built from; `None` for an id the holder laid no object under, or
    /// one whose row went.
    pub fn standing(&self, object: &ObjectId) -> Option<Standing> {
        self.held.read().rows.get(object).copied()
    }

    /// The registry's id of the object that stands for `row`.
    pub fn object(&self, row: WorkerId) -> Option<ObjectId> {
        self.held.read().objects.get(&row).cloned()
    }

    /// How many objects stand.
    pub fn len(&self) -> usize {
        self.held.read().objects.len()
    }

    /// Whether no object stands.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The store's epoch the last read ended at, for a case.
    #[cfg(test)]
    pub(crate) fn read_to(&self) -> Option<u64> {
        self.read_to.lock().epoch
    }

    /// Read `store`'s membership to its end and leave the registry holding
    /// the object of every row that stands.
    ///
    /// The first read is the store's snapshot. So is a read the store's
    /// tail no longer reaches back to, and one behind a move of the store's
    /// history ([`WorkersStore::history`]). A snapshot lays every row
    /// again, whatever its object was built from.
    ///
    /// Every other read is the changes since the last, again and again
    /// until none is left: a write that lands while this works is read
    /// before it returns or by the next call. A change names a row, and the
    /// row is read as it stands now, so a row that changed twice is laid
    /// once more and one that came and went builds nothing. A change that
    /// says a row left ends its object first, so a row that left and came
    /// back under its id is built anew.
    ///
    /// Callers follow one at a time: a second waits for the first. The call
    /// waits on nothing else and reads no clock.
    pub fn follow<S: WorkersStore + ?Sized>(&self, store: &S) {
        let mut place = self.read_to.lock();
        loop {
            // The history ahead of the rows: a restore that lands behind
            // this read of it is seen by the next.
            let history = store.history();
            let turned = history != place.history;
            let tail = match place.epoch {
                Some(epoch) if !turned => store.changes_since(epoch),
                _ => MembershipTail::SnapshotRequired,
            };
            match tail {
                MembershipTail::Changes(changes) => {
                    let Some(&(last, _)) = changes.last() else {
                        return;
                    };
                    for (_, change) in changes {
                        self.apply(store, change);
                    }
                    place.epoch = Some(last);
                }
                MembershipTail::SnapshotRequired => {
                    let (epoch, rows) = store.snapshot();
                    self.reconcile(&rows);
                    *place = Place {
                        epoch: Some(epoch),
                        history,
                    };
                }
            }
        }
    }

    /// One change of the tail: a row it says left ends its object, then the
    /// rows it names are settled as they stand now. A status change names no
    /// fact and reads nothing.
    fn apply<S: WorkersStore + ?Sized>(&self, store: &S, change: MembershipChange) {
        match change {
            MembershipChange::StatusChanged { .. } => {}
            // The old incarnation first: its object leaves the URL the new
            // one's is built under. It leaves whatever stands under its id
            // now: an incarnation a key takes again is another life.
            MembershipChange::Replaced { old, new } => {
                self.end(old);
                self.settle(old, store.get(old).as_ref());
                self.settle(new, store.get(new).as_ref());
            }
            // The row left, so its object does, even where its id stands
            // again: that row is another life, at the first revision too.
            MembershipChange::Removed { id } => {
                self.end(id);
                self.settle(id, store.get(id).as_ref());
            }
            MembershipChange::Registered { id }
            | MembershipChange::Relabelled { id }
            | MembershipChange::Resized { id } => self.settle(id, store.get(id).as_ref()),
        }
    }

    /// The store's rows, whole: the objects of rows that went are dropped
    /// first, each leaving its URL to a later incarnation's, then every row
    /// is laid again. No revision held is trusted here: behind a gap of the
    /// tail, as in another history, a row under a held id and revision may
    /// be another life with other facts.
    fn reconcile(&self, rows: &[WorkerRow]) {
        let stand: HashSet<WorkerId> = rows.iter().map(|row| row.id).collect();
        let gone: Vec<WorkerId> = self
            .held
            .read()
            .objects
            .keys()
            .filter(|id| !stand.contains(id))
            .copied()
            .collect();
        for id in gone {
            self.end(id);
        }
        for row in rows {
            self.lay(row);
        }
    }

    /// Bring the object of the row `id` to what `row` says: the row as the
    /// store holds it, `None` for a row that went. A row whose object was
    /// built from the revision it stands at is left alone: read change by
    /// change, within one life of a row, the revision tells its facts.
    fn settle(&self, id: WorkerId, row: Option<&WorkerRow>) {
        let Some(row) = row else {
            self.end(id);
            return;
        };
        let built_from = self
            .object(id)
            .and_then(|held| self.standing(&held))
            .map(|standing| standing.revision);
        if built_from != Some(row.revision) {
            self.lay(row);
        }
    }

    /// Drop the object held for the row `id`, where one is.
    fn end(&self, id: WorkerId) {
        let Some(held) = self.object(id) else {
            return;
        };
        // The registry first: the entry stands while its event is sent.
        self.registry.remove(&held);
        let mut entries = self.held.write();
        entries.objects.remove(&id);
        entries.rows.remove(&held);
    }

    /// Build `row`'s object and put it in the registry: registered under a
    /// new id where the row's URL holds no object, else in the place of the
    /// one that stands there, under its id and with its running state.
    fn lay(&self, row: &WorkerRow) {
        let built: Arc<dyn Worker> = match &self.connector {
            Some(connector) => {
                Arc::new(builder(row).engine_connector(Arc::clone(connector)).build())
            }
            None => object(row),
        };
        // The id first, then the entry, then the registry's write: its
        // event finds the entry.
        let id = self.registry.reserve_id_for_url(&row.facts.key);
        {
            let mut entries = self.held.write();
            entries.objects.insert(row.id, id.clone());
            entries.rows.insert(
                id,
                Standing {
                    row: row.id,
                    revision: row.revision,
                },
            );
        }
        self.registry.register_or_replace(built);
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeMap};

    use smg_worker::{
        identity::builder::BasicWorkerBuilder, registry::event::WorkerEvent,
        worker::WorkerStatus as ObjectStatus,
    };
    use tokio::sync::broadcast::Receiver;

    use super::*;
    use crate::store::{
        HealthSink, Kinds, Maker, MemoryStore, WorkerFacts, WorkerPatch, WorkerStatus,
    };

    const A: &str = "http://a:8000";
    const B: &str = "http://b:8000";
    const C: &str = "http://c:8000";
    const D: &str = "http://d:8000";
    const E: &str = "http://e:8000";
    const F: &str = "http://f:8000";

    fn facts(key: &str, incarnation: u64) -> WorkerFacts {
        WorkerFacts {
            key: key.to_string(),
            engine: key.to_string(),
            incarnation,
            models: vec!["m".to_string()],
            dp_size: 1,
            ..WorkerFacts::default()
        }
    }

    /// [`facts`] with one label.
    fn labelled(key: &str, incarnation: u64, tier: &str) -> WorkerFacts {
        WorkerFacts {
            labels: BTreeMap::from([("tier".to_string(), tier.to_string())]),
            ..facts(key, incarnation)
        }
    }

    fn labels(tier: &str) -> WorkerPatch {
        WorkerPatch {
            labels: Some(BTreeMap::from([("tier".to_string(), tier.to_string())])),
            ..WorkerPatch::default()
        }
    }

    /// One write of a replica's store, under the name a failure says.
    type Write<'a> = (&'a str, &'a dyn Fn(&Replica));

    /// One replica: its store, its holder, and what its registry tells.
    struct Replica {
        store: MemoryStore,
        objects: Objects<Members>,
        events: Receiver<WorkerEvent>,
    }

    impl Replica {
        fn new() -> Self {
            let registry = Arc::new(WorkerRegistry::new());
            let events = registry.subscribe_events();
            Self {
                store: MemoryStore::new(),
                objects: Objects::new(registry),
                events,
            }
        }

        fn follow(&self) {
            self.objects.follow(&self.store);
        }

        /// What the registry told since the last call, in its order.
        fn told(&mut self) -> Vec<(&'static str, ObjectId)> {
            let mut told = Vec::new();
            while let Ok(event) = self.events.try_recv() {
                told.push(match event {
                    WorkerEvent::Registered { worker_id, .. } => ("registered", worker_id),
                    WorkerEvent::Replaced { worker_id, .. } => ("replaced", worker_id),
                    WorkerEvent::Removed { worker_id, .. } => ("removed", worker_id),
                    WorkerEvent::StatusChanged { worker_id, .. } => ("status", worker_id),
                });
            }
            told
        }

        fn at(&self, url: &str) -> Arc<dyn Worker> {
            self.objects.registry().get_by_url(url).expect("an object")
        }

        fn id(&self, url: &str) -> ObjectId {
            self.objects.registry().get_id_by_url(url).expect("an id")
        }

        fn row(&self, url: &str) -> WorkerRow {
            self.store.by_key(url).expect("a row")
        }

        fn patch(&self, url: &str, patch: WorkerPatch) {
            let row = self.row(url);
            let change = self.store.patch(row.id, patch, row.revision);
            assert_eq!(change, Some(MembershipChange::Relabelled { id: row.id }));
        }

        fn remove(&self, url: &str) {
            let row = self.row(url);
            assert!(self.store.remove(row.id, row.revision).is_some());
        }

        fn tier(&self, url: &str) -> Option<String> {
            self.at(url).metadata().spec.labels.get("tier").cloned()
        }
    }

    /// The first act: by the snapshot for a row that stood before the first
    /// read, by the tail for one that arrives after.
    #[test]
    fn a_row_that_arrives_has_its_object_built() {
        let mut replica = Replica::new();
        assert!(replica.objects.is_empty());
        let first = replica.store.upsert(facts(A, 1)).id();
        replica.follow();
        let second = replica.store.upsert(facts(B, 1)).id();
        replica.follow();
        assert_eq!(replica.objects.read_to(), Some(replica.store.epoch()));
        let registry = Arc::clone(replica.objects.registry());
        assert_eq!((replica.objects.len(), registry.len()), (2, 2));
        assert_eq!(registry.get_by_model("m").len(), 2);
        for (url, row) in [(A, first), (B, second)] {
            let id = replica.id(url);
            assert_eq!(replica.at(url).url(), url);
            assert_eq!(replica.objects.object(row), Some(id.clone()));
            let revision = replica.row(url).revision;
            assert_eq!(
                replica.objects.standing(&id),
                Some(Standing { row, revision })
            );
        }
        let told = [("registered", replica.id(A)), ("registered", replica.id(B))];
        assert_eq!(replica.told(), told);
    }

    /// The second act.
    #[test]
    fn a_row_that_goes_has_its_object_dropped() {
        let mut replica = Replica::new();
        let row = replica.store.upsert(facts(A, 1)).id();
        replica.store.upsert(facts(B, 1));
        replica.follow();
        let id = replica.id(A);
        replica.told();
        replica.remove(A);
        replica.follow();
        let registry = replica.objects.registry();
        assert!(registry.get_by_url(A).is_none());
        assert_eq!((replica.objects.len(), registry.len()), (1, 1));
        assert!(!replica.objects.is_empty());
        assert_eq!(replica.objects.object(row), None);
        assert_eq!(replica.objects.standing(&id), None);
        assert_eq!(replica.told(), [("removed", id)]);
    }

    /// The third act, at each write of a row's facts: a patch, a refresh of
    /// the registration, a new width. The id stands, the object is another,
    /// and what ran on the old one runs on the new: never a removal and an
    /// add.
    #[test]
    fn a_row_whose_facts_change_has_its_object_replaced_under_the_same_id() {
        let mut replica = Replica::new();
        let row = replica.store.upsert(facts(A, 1)).id();
        replica.follow();
        let id = replica.id(A);
        replica.told();
        let wider = WorkerFacts {
            dp_size: 2,
            ..facts(A, 1)
        };
        let writes: [Write<'_>; 3] = [
            ("a patch", &|replica| replica.patch(A, labels("gold"))),
            ("a refresh", &|replica| {
                let change = replica.store.upsert(facts(A, 1));
                assert_eq!(change, MembershipChange::Relabelled { id: row });
            }),
            ("a new width", &|replica| {
                let change = replica.store.upsert(wider.clone());
                assert_eq!(change, MembershipChange::Resized { id: row });
            }),
        ];
        let shown: [(Option<&str>, Option<usize>); 3] =
            [(Some("gold"), None), (None, None), (None, Some(2))];
        for (turn, ((name, write), shown)) in writes.into_iter().zip(shown).enumerate() {
            let before = replica.at(A);
            before.increment_load();
            write(&replica);
            replica.follow();
            let after = replica.at(A);
            assert!(!Arc::ptr_eq(&before, &after), "{name}: another object");
            assert_eq!(replica.id(A), id, "{name}: the same id");
            assert_eq!(after.load(), turn + 1, "{name}: the running state");
            assert_eq!(
                (replica.tier(A).as_deref(), after.dp_size()),
                shown,
                "{name}"
            );
            let revision = replica.row(A).revision;
            assert_eq!(
                replica.objects.standing(&id),
                Some(Standing { row, revision }),
                "{name}"
            );
            assert_eq!(replica.told(), [("replaced", id.clone())], "{name}");
        }
    }

    /// The fourth act: a health write and a status write move no fact. The
    /// store's status is the truth, and the object keeps its first.
    #[test]
    fn a_write_that_moves_no_fact_leaves_the_same_object() {
        let mut replica = Replica::new();
        let row = replica.store.upsert(facts(A, 1)).id();
        replica.follow();
        let (id, before) = (replica.id(A), replica.at(A));
        let standing = replica.objects.standing(&id);
        replica.told();
        let revision = replica.row(A).revision;
        replica.store.probe_outcome(row, true, revision);
        let moved = replica.store.set_status(row, WorkerStatus::Ready, revision);
        assert!(moved.is_some());
        assert_eq!(replica.row(A).revision, revision);
        replica.follow();
        assert!(Arc::ptr_eq(&before, &replica.at(A)));
        assert_eq!(replica.objects.standing(&id), standing);
        assert!(replica.told().is_empty());
        assert_eq!(replica.at(A).status(), ObjectStatus::Pending);
    }

    #[test]
    fn a_new_incarnation_of_a_key_drops_the_old_object_and_builds_one_under_a_new_id() {
        let mut replica = Replica::new();
        let old = replica.store.upsert(facts(A, 1)).id();
        replica.follow();
        let (old_id, before) = (replica.id(A), replica.at(A));
        before.increment_load();
        replica.told();
        let new = replica.store.upsert(facts(A, 2)).id();
        assert_ne!(old, new);
        replica.follow();
        let new_id = replica.id(A);
        assert_ne!(old_id, new_id);
        assert_eq!(replica.at(A).load(), 0, "nothing of the old object runs on");
        assert_eq!(replica.objects.len(), 1);
        assert_eq!(replica.objects.object(old), None);
        assert_eq!(replica.objects.standing(&old_id), None);
        assert_eq!(replica.objects.object(new), Some(new_id.clone()));
        let row = replica
            .objects
            .standing(&new_id)
            .map(|standing| standing.row);
        assert_eq!(row, Some(new));
        assert_eq!(
            replica.told(),
            [("removed", old_id), ("registered", new_id)]
        );
    }

    /// A row that leaves and is registered again under the same key and
    /// incarnation is another life under the same id, at the first revision
    /// again. Read in one pass, the change that says it left ends its
    /// object and the row that stands is built anew: nothing of the first
    /// life's object runs on. So it is where a key takes another incarnation
    /// and then the first one again.
    #[test]
    fn a_row_that_left_and_came_back_under_its_id_has_a_new_object_built() {
        let mut replica = Replica::new();
        let first = replica.store.upsert(facts(A, 1)).id();
        replica.follow();
        let (old_id, before) = (replica.id(A), replica.at(A));
        before.increment_load();
        replica.told();

        replica.remove(A);
        assert_eq!(replica.store.upsert(labelled(A, 1, "gold")).id(), first);
        assert_eq!(replica.row(A).revision, 1, "the first revision again");
        replica.follow();
        let new_id = replica.id(A);
        assert_ne!(old_id, new_id);
        assert_eq!(replica.tier(A).as_deref(), Some("gold"));
        assert_eq!(replica.at(A).load(), 0, "nothing of the first life runs on");
        assert_eq!(replica.objects.object(first), Some(new_id.clone()));
        assert_eq!(replica.objects.standing(&old_id), None);
        assert_eq!(
            replica.told(),
            [("removed", old_id), ("registered", new_id.clone())]
        );

        replica.store.upsert(facts(A, 2));
        assert_eq!(replica.store.upsert(labelled(A, 1, "silver")).id(), first);
        replica.follow();
        let last_id = replica.id(A);
        assert_ne!(new_id, last_id);
        assert_eq!(replica.tier(A).as_deref(), Some("silver"));
        let registry = Arc::clone(replica.objects.registry());
        assert_eq!((replica.objects.len(), registry.len()), (1, 1));
        assert_eq!(replica.objects.object(first), Some(last_id.clone()));
        assert_eq!(
            replica.told(),
            [("removed", new_id), ("registered", last_id)]
        );
    }

    /// A row is read as it stands now, not as each change left it.
    #[test]
    fn two_changes_of_a_row_lay_it_once_and_a_row_come_and_gone_builds_nothing() {
        let mut replica = Replica::new();
        replica.store.upsert(facts(A, 1));
        replica.follow();
        let id = replica.id(A);
        replica.told();
        replica.patch(A, labels("gold"));
        replica.patch(A, labels("silver"));
        let passing = replica.store.upsert(facts(B, 1)).id();
        replica.remove(B);
        replica.follow();
        assert_eq!(replica.told(), [("replaced", id)]);
        assert_eq!(replica.tier(A).as_deref(), Some("silver"));
        assert!(replica.objects.registry().get_by_url(B).is_none());
        assert_eq!(replica.objects.object(passing), None);
        // And a read with nothing to read does nothing.
        replica.follow();
        assert!(replica.told().is_empty());
    }

    /// Past the store's tail the read is the snapshot: one write of each
    /// kind and then more status writes than the tail keeps. The objects of
    /// the rows that went leave first, then every row is laid again, and the
    /// next read is the tail again. A row no write touched is laid again
    /// too: behind the gap it cannot be told from one that left and came
    /// back under its id and its revision, as one here did with a new label.
    #[test]
    fn a_read_the_tail_no_longer_reaches_takes_the_snapshot_and_lays_every_row_again() {
        let mut replica = Replica::new();
        let gone = replica.store.upsert(facts(A, 1)).id();
        replica.store.upsert(facts(B, 1));
        let replaced = replica.store.upsert(facts(C, 1)).id();
        replica.store.upsert(facts(E, 1));
        let back = replica.store.upsert(facts(F, 1)).id();
        replica.follow();
        let (a, b, c) = (replica.id(A), replica.id(B), replica.id(C));
        let (e, f) = (replica.id(E), replica.id(F));
        let read_to = replica.store.epoch();
        replica.told();

        replica.remove(A);
        replica.patch(B, labels("gold"));
        let next = replica.store.upsert(facts(C, 2)).id();
        let arrived = replica.store.upsert(facts(D, 1)).id();
        replica.remove(F);
        assert_eq!(replica.store.upsert(labelled(F, 1, "gold")).id(), back);
        assert_eq!(replica.row(F).revision, 1, "the first revision again");
        let row = replica.row(D);
        for turn in 0..1_100 {
            let status = [WorkerStatus::Ready, WorkerStatus::NotReady][turn % 2];
            assert!(replica
                .store
                .set_status(row.id, status, row.revision)
                .is_some());
        }
        assert_eq!(
            replica.store.changes_since(read_to),
            MembershipTail::SnapshotRequired
        );
        replica.follow();

        let registry = Arc::clone(replica.objects.registry());
        assert_eq!((replica.objects.len(), registry.len()), (5, 5));
        assert!(registry.get_by_url(A).is_none());
        assert_eq!(replica.objects.object(gone), None);
        assert_eq!(replica.objects.object(replaced), None);
        assert_eq!(replica.id(B), b);
        assert_eq!(replica.tier(B).as_deref(), Some("gold"));
        assert_eq!((replica.id(E), replica.id(F)), (e.clone(), f.clone()));
        assert_eq!(replica.tier(F).as_deref(), Some("gold"));
        let (c_next, d) = (replica.id(C), replica.id(D));
        assert_ne!(c_next, c);
        assert_eq!(replica.objects.object(next), Some(c_next.clone()));
        assert_eq!(replica.objects.object(arrived), Some(d.clone()));
        let told = replica.told();
        let left: HashSet<_> = told.iter().take(2).cloned().collect();
        assert_eq!(left, HashSet::from([("removed", a), ("removed", c)]));
        let laid: HashSet<_> = told.iter().skip(2).cloned().collect();
        let each = [
            ("replaced", b),
            ("registered", c_next),
            ("registered", d),
            ("replaced", e),
            ("replaced", f),
        ];
        assert_eq!((told.len(), laid), (7, HashSet::from(each)));

        replica.remove(D);
        replica.follow();
        assert!(registry.get_by_url(D).is_none());
        assert_eq!(replica.told().len(), 1);
    }

    /// A store that takes one more registration the first time a row is
    /// read of it: a write that lands while a read works.
    struct Late {
        store: MemoryStore,
        late: Mutex<Option<WorkerFacts>>,
    }

    impl WorkersStore for Late {
        fn upsert_by(&self, facts: WorkerFacts, maker: Maker) -> MembershipChange {
            self.store.upsert_by(facts, maker)
        }

        fn patch(
            &self,
            id: WorkerId,
            patch: WorkerPatch,
            revision: u64,
        ) -> Option<MembershipChange> {
            self.store.patch(id, patch, revision)
        }

        fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
            self.store.remove(id, revision)
        }

        fn set_status(
            &self,
            id: WorkerId,
            status: WorkerStatus,
            revision: u64,
        ) -> Option<MembershipChange> {
            self.store.set_status(id, status, revision)
        }

        fn get(&self, id: WorkerId) -> Option<WorkerRow> {
            let row = self.store.get(id);
            if let Some(facts) = self.late.lock().take() {
                self.store.upsert(facts);
            }
            row
        }

        fn by_key(&self, key: &str) -> Option<WorkerRow> {
            self.store.by_key(key)
        }

        fn epoch(&self) -> u64 {
            self.store.epoch()
        }

        fn snapshot(&self) -> (u64, Vec<WorkerRow>) {
            self.store.snapshot()
        }

        fn changes_since(&self, epoch: u64) -> MembershipTail {
            self.store.changes_since(epoch)
        }
    }

    #[test]
    fn a_write_that_lands_while_a_read_works_is_read_before_the_read_returns() {
        let registry = Arc::new(WorkerRegistry::<Members>::new());
        let objects = Objects::new(Arc::clone(&registry));
        let store = Late {
            store: MemoryStore::new(),
            late: Mutex::new(None),
        };
        objects.follow(&store);
        store.upsert(facts(A, 1));
        *store.late.lock() = Some(facts(B, 1));
        objects.follow(&store);
        assert!(registry.get_by_url(A).is_some());
        assert!(registry.get_by_url(B).is_some(), "the late row");
        assert_eq!(objects.len(), 2);
    }

    /// The holder is the registry's one writer.
    #[test]
    fn an_object_another_writer_left_under_a_rows_url_gives_way_to_the_rows() {
        let mut replica = Replica::new();
        replica.follow();
        let registry = Arc::clone(replica.objects.registry());
        let foreign: Arc<dyn Worker> = Arc::new(<BasicWorkerBuilder>::new(A).build());
        let id = registry.register(Arc::clone(&foreign)).expect("a free URL");
        let row = replica.store.upsert(facts(A, 1)).id();
        replica.follow();
        assert!(!Arc::ptr_eq(&foreign, &replica.at(A)));
        assert_eq!(replica.id(A), id);
        assert_eq!(replica.objects.object(row), Some(id.clone()));
        let standing = replica.objects.standing(&id).map(|standing| standing.row);
        assert_eq!(standing, Some(row));
        assert_eq!(
            replica.told(),
            [("registered", id.clone()), ("replaced", id)]
        );
        assert_eq!(format!("{:?}", replica.objects), "Objects { held: 1, .. }");
    }

    /// A store restored from another store's cut can stand at the epoch its
    /// holder read to, and hold under one id and one revision a row of
    /// other facts: the tail reads empty and the row reads as the one held.
    #[test]
    fn a_store_that_took_another_historys_rows_has_every_row_laid_again() {
        let mut replica = Replica::new();
        let row = replica.store.upsert(facts(A, 1)).id();
        replica.store.upsert(facts(B, 1));
        replica.follow();
        let (a, b) = (replica.id(A), replica.id(B));
        replica.told();

        let other = MemoryStore::new();
        other.upsert(WorkerFacts {
            labels: BTreeMap::from([("tier".to_string(), "gold".to_string())]),
            ..facts(A, 1)
        });
        other.upsert(facts(C, 1));
        let cut = other.cut(Kinds::Watched);
        let (epoch, theirs) = (cut.epoch, other.by_key(A).expect("a row"));
        assert_eq!(epoch, replica.store.epoch());
        assert_eq!((theirs.id, theirs.revision), (row, replica.row(A).revision));
        assert!(replica.store.restore_from(cut));
        assert_eq!(
            replica.store.changes_since(epoch),
            MembershipTail::Changes(Vec::new())
        );
        replica.follow();

        assert_eq!(replica.id(A), a);
        assert_eq!(replica.tier(A).as_deref(), Some("gold"));
        assert!(replica.objects.registry().get_by_url(B).is_none());
        assert_eq!(replica.objects.len(), 2);
        let c = replica.id(C);
        assert_eq!(
            replica.told(),
            [("removed", b), ("replaced", a), ("registered", c)]
        );
        // The next read is the restored store's tail, and it is empty.
        replica.follow();
        assert!(replica.told().is_empty());
    }

    thread_local! {
        /// The holder and the row a case watches.
        static WATCHED: RefCell<Option<(Arc<Objects<Seen>>, WorkerId)>> =
            const { RefCell::new(None) };
        /// What each publish read of them.
        static READ: RefCell<Vec<Option<u64>>> = const { RefCell::new(Vec::new()) };
    }

    /// A generation that reads, where the registry publishes it inside its
    /// own write, the revision the holder says the watched row's object was
    /// built from.
    #[derive(Debug)]
    struct Seen(Arc<[Arc<dyn Worker>]>);

    impl Generation for Seen {
        fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
            WATCHED.with_borrow(|watched| {
                if let Some((objects, row)) = watched {
                    let built = objects
                        .object(*row)
                        .and_then(|id| objects.standing(&id))
                        .map(|standing| standing.revision);
                    READ.with_borrow_mut(|read| read.push(built));
                }
            });
            Self(members)
        }

        fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
            &self.0
        }
    }

    /// An object's entry is written before the registry tells of the
    /// object, at a build and at a replacement: what the registry publishes
    /// inside its write already reads the row's revision of that write.
    #[test]
    fn an_objects_entry_stands_before_the_registry_tells_of_it() {
        let store = MemoryStore::new();
        let objects = Arc::new(Objects::<Seen>::new(Arc::new(WorkerRegistry::new())));
        let row = store.upsert(facts(A, 1)).id();
        let built = store.get(row).expect("a row").revision;
        WATCHED.set(Some((Arc::clone(&objects), row)));
        objects.follow(&store);
        let at_build = READ.take();
        assert!(!at_build.is_empty(), "a registration publishes");
        assert!(
            at_build.iter().all(|read| *read == Some(built)),
            "{at_build:?}"
        );

        assert!(store.patch(row, labels("gold"), built).is_some());
        let replaced = store.get(row).expect("a row").revision;
        assert_ne!(replaced, built);
        objects.follow(&store);
        let at_replacement = READ.take();
        assert!(!at_replacement.is_empty(), "a replacement publishes");
        assert!(
            at_replacement.iter().all(|read| *read == Some(replaced)),
            "{at_replacement:?}"
        );
        WATCHED.set(None);
    }
}
