//! The holder of a replica's worker objects: it follows the store's
//! membership and writes the registry. A host can also ask it to follow
//! the rows' status.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
};

use parking_lot::{Mutex, RwLock};
use smg_worker::{
    connection::engine::EngineConnector,
    identity::basic::Worker,
    registry::{Generation, WorkerId as ObjectId, WorkerRegistry},
};

use super::{
    credentials::Credentials,
    object::{builder, dial_address, status},
};
use crate::{
    store::{MembershipChange, MembershipTail, WorkerFacts, WorkerRow, WorkersStore},
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

/// A row and the object the holder built from it.
struct Laid {
    standing: Standing,
    object: Arc<dyn Worker>,
}

/// What stands, both ways.
#[derive(Default)]
struct Held {
    /// A row's object, by the row's id.
    objects: HashMap<WorkerId, ObjectId>,
    /// An object's row, by the object's id in the registry.
    rows: HashMap<ObjectId, Laid>,
    /// The facts a row's object was built from, by the row's id: what a
    /// snapshot read holds the row against.
    facts: HashMap<WorkerId, WorkerFacts>,
}

/// A replica's worker objects: for every row of the membership store that
/// stands one object in the registry, and none beside.
///
/// [`Self::follow`] is the one path, on a leader and on a standby: it reads
/// what the store's membership did since its last read. It builds the
/// object of a row that arrived, drops the object of a row that went, and
/// replaces the object of a row whose facts changed under the same id in
/// the registry. A new incarnation of a key gets a new object and id.
///
/// By default a status change reads no row and leaves the object alone.
/// With [`Self::with_status_followed`], the holder copies the row's status
/// on a status change, on a build or replacement, and on a snapshot read
/// that keeps the object. The registry makes the write, as it makes a
/// health driver's: its subscribers hear the change once, and a promoted
/// worker comes back as any promotion brings it, an open breaker on
/// trial. The host's own vetoes stand.
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
/// The holder trusts no revision it held alone in either. It holds each
/// row against the facts its object was built from, beside the revision.
/// A row whose facts and revision both stand keeps its object: one built
/// again from them would state the same row, and its replacement would
/// start every read of its worker again. A credential the host kept since
/// is no fact of a row: the object takes it at its next build. Any other
/// row is laid again and
/// keeps its place: its object is replaced under its id in the registry,
/// which carries the standing object's running state over. An entry's
/// revision never moves under an object that is kept.
///
/// With a keeper ([`Self::with_credentials`]), each read has the keeper
/// forget the rows that left, whoever wrote the removal, and an entry goes
/// with its last row. The keeper's clock is read ahead of the store: a row
/// bound behind that read stays, and the next read forgets it whatever it
/// finds.
///
/// The holder is the registry's one writer: an object any other writer left
/// under a row's URL is replaced by the row's.
///
/// A row takes the place of its own object alone: an object another row
/// holds under its URL is dropped first.
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
    status_followed: bool,
    /// What the host keeps for its workers, by the address they dial and
    /// the row a registration wrote.
    credentials: Option<Arc<Credentials>>,
    /// The keeper's clock as the last forgetting read it. Once the clock
    /// is past it, a registration answered since bound a row that
    /// forgetting did not ask about, and the next read forgets again.
    /// Written and read under the follow's turn.
    forgotten_at: AtomicU64,
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

impl<G> Objects<G> {
    /// For a host that drives no health of its own: an object's status
    /// follows its row's.
    #[must_use]
    pub fn with_status_followed(mut self) -> Self {
        self.status_followed = true;
        self
    }
}

impl<G: Generation> Objects<G> {
    /// A holder that writes `registry` and holds nothing yet: the first
    /// [`Self::follow`] builds an object for every row that stands.
    pub fn new(registry: Arc<WorkerRegistry<G>>) -> Self {
        Self {
            registry,
            connector: None,
            status_followed: false,
            credentials: None,
            forgotten_at: AtomicU64::new(0),
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

    /// Hand each object the credential the keeper holds for its row, at
    /// the address it dials, when it is built. The host supplies the
    /// keeper. An object is built without a credential if there is no
    /// keeper, or no registration of this host bound its row.
    #[must_use]
    pub fn with_credentials(mut self, credentials: Arc<Credentials>) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Lay again the object of each row that dials the address of `facts`
    /// and holds another credential than its row takes now. Each is built
    /// anew from its row and takes its standing object's place, under its
    /// id and with its running state. A registration calls it where its
    /// token's answer says so
    /// ([`KeptCredential::accepted`](super::KeptCredential::accepted),
    /// [`KeptCredential::put_back`](super::KeptCredential::put_back)),
    /// with the facts it kept. The call follows `store` first and holds
    /// the follow's turn to its end. It reads the keeper as a build does:
    /// where a registration is in flight at the address, an object that
    /// holds what it stated marks the flight, and its put-back says so.
    /// Return how many objects were laid again: none where no keeper was
    /// handed in, where no row dials the address, or where each object
    /// there holds what its row takes.
    pub fn lay_again<S: WorkersStore + ?Sized>(&self, store: &S, facts: &WorkerFacts) -> usize {
        let mut place = self.read_to.lock();
        self.follow_from(store, &mut place);
        let Some(credentials) = &self.credentials else {
            return 0;
        };
        let address = dial_address(facts);
        let stale: Vec<WorkerId> = {
            let held = self.held.read();
            held.rows
                .values()
                .filter(|laid| {
                    held.facts
                        .get(&laid.standing.row)
                        .is_some_and(|dialing| dial_address(dialing) == address)
                        && credentials.of(address, laid.standing.row).as_ref()
                            != laid.object.api_key()
                })
                .map(|laid| laid.standing.row)
                .collect()
        };
        let mut laid = 0;
        // A row that left behind the read above is the next follow's.
        for row in stale.into_iter().filter_map(|id| store.get(id)) {
            self.lay(store, &row);
            laid += 1;
        }
        laid
    }

    /// The registry the objects stand in.
    pub fn registry(&self) -> &Arc<WorkerRegistry<G>> {
        &self.registry
    }

    /// The row the object under `object` stands for, with the revision it
    /// was built from; `None` for an id the holder laid no object under, or
    /// one whose row went.
    pub fn standing(&self, object: &ObjectId) -> Option<Standing> {
        self.held.read().rows.get(object).map(|laid| laid.standing)
    }

    /// The row and revision the holder built this object from. A writer
    /// holding the object's registry lock uses this to name that revision.
    pub fn standing_of(&self, id: &ObjectId, object: &Arc<dyn Worker>) -> Option<Standing> {
        self.held
            .read()
            .rows
            .get(id)
            .filter(|laid| Arc::ptr_eq(&laid.object, object))
            .map(|laid| laid.standing)
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
    /// history ([`WorkersStore::history`]). A snapshot lays again every
    /// row whose facts or revision are not the ones its object was built
    /// from.
    ///
    /// Every other read is the changes since the last, again and again
    /// until none is left: a write that lands while this works is read
    /// before it returns or by the next call. A change names a row, and the
    /// row is read as it stands now, so a row that changed twice is laid
    /// once more and one that came and went builds nothing. A change that
    /// says a row left ends its object first, so a row that left and came
    /// back under its id is built anew.
    ///
    /// Calls to `follow` run one at a time. A status mirror reads the row
    /// and writes the object under the object's mutation lock.
    /// A store notification inside a health write can still hold that lock.
    /// Use the notification to wake the caller of `follow`.
    pub fn follow<S: WorkersStore + ?Sized>(&self, store: &S) {
        self.follow_from(store, &mut self.read_to.lock());
    }

    /// [`Self::follow`] for the caller that holds the turn.
    fn follow_from<S: WorkersStore + ?Sized>(&self, store: &S, place: &mut Place) {
        loop {
            // The keeper's clock ahead of the read of the store: a row a
            // registration bound behind it is kept by the forgetting.
            let read = self
                .credentials
                .as_ref()
                .map(|credentials| credentials.taken());
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
                        // A read that found nothing forgets all the same
                        // once the keeper's clock is past the last
                        // forgetting: an answer since bound a row that
                        // may have left before it.
                        if read.is_some_and(|read| read > self.forgotten_at.load(Relaxed)) {
                            self.forget_left(read);
                        }
                        return;
                    };
                    for (_, change) in changes {
                        self.apply(store, change);
                    }
                    place.epoch = Some(last);
                }
                MembershipTail::SnapshotRequired => {
                    let (epoch, rows) = store.snapshot();
                    self.reconcile(store, &rows);
                    *place = Place {
                        epoch: Some(epoch),
                        history,
                    };
                }
            }
            self.forget_left(read);
        }
    }

    /// Have the keeper forget the rows that left: each row a registration
    /// bound that no longer stands here, by the keeper's clock as it read
    /// ahead of the store. One pass over the keeper's rows, behind a read
    /// that found a change, and behind any read once the clock is past
    /// the last pass.
    fn forget_left(&self, read: Option<u64>) {
        let (Some(credentials), Some(read)) = (&self.credentials, read) else {
            return;
        };
        let held = self.held.read();
        credentials.forget_left(read, |row| held.facts.contains_key(&row));
        self.forgotten_at.store(read, Relaxed);
    }

    /// One change of the tail: a row it says left ends its object, then the
    /// rows it names are settled as they stand now.
    fn apply<S: WorkersStore + ?Sized>(&self, store: &S, change: MembershipChange) {
        match change {
            MembershipChange::StatusChanged { id, .. } => self.follow_status(store, id),
            // The old incarnation first: its object leaves the URL the new
            // one's is built under. It leaves whatever stands under its id
            // now: an incarnation a key takes again is another life.
            MembershipChange::Replaced { old, new } => {
                self.end(old);
                self.settle(store, old);
                self.settle(store, new);
            }
            // The row left, so its object does, even where its id stands
            // again: that row is another life, at the first revision too.
            MembershipChange::Removed { id } => {
                self.end(id);
                self.settle(store, id);
            }
            MembershipChange::Registered { id }
            | MembershipChange::Relabelled { id }
            | MembershipChange::Resized { id } => self.settle(store, id),
        }
    }

    /// The store's rows, whole: the objects of rows that went are dropped
    /// first, each leaving its URL to a later incarnation's, then every row
    /// that does not stand as its object was built is laid again. No
    /// revision held is trusted alone here: behind a gap of the tail, as in
    /// another history, a row under a held id and revision may be another
    /// life with other facts. The facts are compared beside it.
    fn reconcile<S: WorkersStore + ?Sized>(&self, store: &S, rows: &[WorkerRow]) {
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
            if self.stands(row) {
                self.follow_status(store, row.id);
            } else {
                self.lay(store, row);
            }
        }
    }

    /// Whether `row`'s object was built from the row as it stands now: the
    /// facts the row states and the revision it stands at. One that was
    /// keeps its object and its place.
    fn stands(&self, row: &WorkerRow) -> bool {
        let held = self.held.read();
        held.facts.get(&row.id) == Some(&row.facts)
            && held
                .objects
                .get(&row.id)
                .and_then(|object| held.rows.get(object))
                .is_some_and(|laid| laid.standing.revision == row.revision)
    }

    /// Settle the object of `id` against the row as the store holds it. A
    /// row whose object was built at its current revision keeps it. Within
    /// one life of a row, the revision tells its facts.
    fn settle<S: WorkersStore + ?Sized>(&self, store: &S, id: WorkerId) {
        let Some(row) = store.get(id) else {
            self.end(id);
            return;
        };
        let built_from = self
            .object(id)
            .and_then(|held| self.standing(&held))
            .map(|standing| standing.revision);
        if built_from != Some(row.revision) {
            self.lay(store, &row);
        }
    }

    #[inline]
    fn follow_status<S: WorkersStore + ?Sized>(&self, store: &S, row: WorkerId) {
        if self.status_followed {
            self.copy_status(store, row);
        }
    }

    /// Read the row under the lock the health adapter writes under. The
    /// object's entry must still name that row and its current revision.
    /// The registry writes the status it is handed and tells it.
    #[inline(never)]
    fn copy_status<S: WorkersStore + ?Sized>(&self, store: &S, row: WorkerId) {
        let Some(id) = self.object(row) else {
            return;
        };
        let Some(object) = self.registry.get(&id) else {
            return;
        };
        let write = |object: &Arc<dyn Worker>| {
            let next = self
                .standing_of(&id, object)
                .and_then(|standing| {
                    store
                        .get(standing.row)
                        .filter(|row| row.revision == standing.revision)
                })
                .map(|row| status(row.status));
            // The registry writes it, as it writes the adapter's: one
            // status event to its subscribers, and what goes with a
            // promotion. A status that stands writes nothing.
            ((), next)
        };
        self.registry
            .apply_if_revision(&id, object.revision(), write);
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
        entries.facts.remove(&id);
    }

    /// Build `row`'s object and put it in the registry: registered under a
    /// new id where the row's URL holds no object, else in the place of the
    /// one that stands there, under its id and with its running state. An
    /// object another row holds under the URL is dropped first: a row takes
    /// the place of its own object alone.
    fn lay<S: WorkersStore + ?Sized>(&self, store: &S, row: &WorkerRow) {
        let mut worker = builder(row);
        if let Some(connector) = &self.connector {
            worker = worker.engine_connector(Arc::clone(connector));
        }
        if let Some(credential) = self
            .credentials
            .as_ref()
            .and_then(|credentials| credentials.of(dial_address(&row.facts), row.id))
        {
            worker = worker.api_key(credential);
        }
        let built: Arc<dyn Worker> = Arc::new(worker.build());
        // The id first, then the entry, then the registry's write: its
        // event finds the entry.
        let mut id = self.registry.reserve_id_for_url(&row.facts.key);
        // A store restored inside a read can show a key's new incarnation
        // before any change says the old one left.
        if let Some(other) = self.standing(&id).filter(|other| other.row != row.id) {
            self.end(other.row);
            id = self.registry.reserve_id_for_url(&row.facts.key);
        }
        {
            let mut entries = self.held.write();
            entries.objects.insert(row.id, id.clone());
            entries.facts.insert(row.id, row.facts.clone());
            entries.rows.insert(
                id,
                Laid {
                    standing: Standing {
                        row: row.id,
                        revision: row.revision,
                    },
                    object: Arc::clone(&built),
                },
            );
        }
        self.registry.register_or_replace(built);
        self.follow_status(store, row.id);
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeMap, sync::mpsc, thread, time::Duration};

    use smg_worker::{
        health::{
            circuit_breaker::{CircuitBreakerConfig, CircuitState},
            port::HealthSink as _,
        },
        identity::builder::BasicWorkerBuilder,
        registry::event::WorkerEvent,
        runtime::StallReason,
        worker::WorkerStatus as ObjectStatus,
    };
    use tokio::sync::broadcast::Receiver;

    use super::*;
    use crate::{
        store::{HealthSink, Kinds, Maker, MemoryStore, WorkerFacts, WorkerPatch, WorkerStatus},
        worker_host::{object, HealthAdapter, KeptCredential, SoleLead},
    };

    const A: &str = "http://a:8000";
    const B: &str = "http://b:8000";
    const C: &str = "http://c:8000";
    const D: &str = "http://d:8000";
    const E: &str = "http://e:8000";
    const F: &str = "http://f:8000";
    const G: &str = "http://g:8000";
    const KEY_A: &str = "key-a";
    const KEY_B: &str = "key-b";
    const KEY_C: &str = "key-c";

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

    fn rank_facts(key: &str, engine: &str, rank: u16) -> WorkerFacts {
        WorkerFacts {
            engine: engine.to_string(),
            dp_rank: rank,
            dp_size: 2,
            ..facts(key, 1)
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

    /// What a registry told since the last read of `events`, in its order.
    fn told(events: &mut Receiver<WorkerEvent>) -> Vec<(&'static str, ObjectId)> {
        let mut told = Vec::new();
        while let Ok(event) = events.try_recv() {
            told.push(match event {
                WorkerEvent::Registered { worker_id, .. } => ("registered", worker_id),
                WorkerEvent::Replaced { worker_id, .. } => ("replaced", worker_id),
                WorkerEvent::Removed { worker_id, .. } => ("removed", worker_id),
                WorkerEvent::StatusChanged { worker_id, .. } => ("status", worker_id),
            });
        }
        told
    }

    /// One replica with its store, holder and keeper. Its registry tells
    /// what happened to the objects.
    struct Replica {
        store: Arc<MemoryStore>,
        objects: Objects<Members>,
        credentials: Arc<Credentials>,
        events: Receiver<WorkerEvent>,
    }

    impl Replica {
        fn new() -> Self {
            let registry = Arc::new(WorkerRegistry::new());
            let events = registry.subscribe_events();
            let credentials = Arc::new(Credentials::new());
            Self {
                store: Arc::new(MemoryStore::new()),
                objects: Objects::new(registry).with_credentials(Arc::clone(&credentials)),
                credentials,
                events,
            }
        }

        fn follow(&self) {
            self.objects.follow(self.store.as_ref());
        }

        /// What the registry told since the last call, in its order.
        fn told(&mut self) -> Vec<(&'static str, ObjectId)> {
            told(&mut self.events)
        }

        fn at(&self, url: &str) -> Arc<dyn Worker> {
            self.objects.registry().get_by_url(url).expect("an object")
        }

        /// A registration taken: its keep, its write, the token accepted
        /// with the row the door answered, and the address laid again
        /// where the acceptance says so. Return the row.
        fn register(&self, facts: WorkerFacts, credential: Option<&str>) -> WorkerId {
            let token = self.kept(&facts, credential);
            let row = self.store.upsert(facts.clone()).id();
            if token.accepted(row) {
                self.lay_again(&facts);
            }
            row
        }

        /// A registration's keep, its answer left to the case.
        fn kept(&self, facts: &WorkerFacts, credential: Option<&str>) -> KeptCredential {
            self.credentials
                .keep(facts, credential.map(str::to_owned))
                .expect("no registration in flight at the address")
        }

        /// The holder lays again the objects at the address `facts` dial.
        fn lay_again(&self, facts: &WorkerFacts) -> usize {
            self.objects.lay_again(self.store.as_ref(), facts)
        }

        fn holds_credential(&self, url: &str, credential: Option<&str>) -> bool {
            self.at(url).api_key().map(String::as_str) == credential
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

    #[test]
    fn a_row_laid_behind_a_kept_credential_builds_an_object_that_holds_it() {
        let replica = Replica::new();
        replica.register(facts(A, 1), Some(KEY_A));
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_A)));

        // A row that names no engine dials its key: the keeper works the
        // address out as the holder does.
        let keyed = WorkerFacts {
            engine: String::new(),
            ..facts(B, 1)
        };
        replica.register(keyed, Some(KEY_B));
        // A row no registration here stated takes none.
        replica.store.upsert(facts(C, 1));
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_A)));
        assert!(replica.holds_credential(B, Some(KEY_B)));
        assert!(replica.holds_credential(C, None));
    }

    #[test]
    fn a_holder_handed_no_keeper_builds_objects_without_a_credential() {
        let registry = Arc::new(WorkerRegistry::<Members>::new());
        let objects = Objects::new(Arc::clone(&registry));
        let store = MemoryStore::new();
        store.upsert(facts(A, 1));
        objects.follow(&store);
        let laid = registry.get_by_url(A).expect("the row's object");
        assert!(laid.api_key().is_none());
        assert_eq!(objects.lay_again(&store, &facts(A, 1)), 0);
    }

    #[test]
    fn each_rank_row_of_an_engine_holds_the_engines_credential() {
        let replica = Replica::new();
        replica.register(rank_facts(A, C, 0), Some(KEY_C));
        replica.register(rank_facts(B, C, 1), Some(KEY_C));
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_C)));
        assert!(replica.holds_credential(B, Some(KEY_C)));
    }

    #[test]
    fn a_rank_row_another_host_registered_takes_no_credential_until_it_is_stated_here() {
        let replica = Replica::new();
        replica.register(rank_facts(A, C, 0), Some(KEY_C));
        // The engine's other rank came through another host: no keep here.
        let other = replica.store.upsert(rank_facts(B, C, 1)).id();
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_C)));
        assert!(replica.holds_credential(B, None));

        // It is stated here too. A door writes nothing over equal facts,
        // and the acceptance says the address is to be laid again.
        let stated = replica.kept(&rank_facts(B, C, 1), Some(KEY_C));
        assert!(stated.accepted(other));
        assert_eq!(replica.lay_again(&rank_facts(B, C, 1)), 1);
        assert!(replica.holds_credential(B, Some(KEY_C)));
    }

    #[test]
    fn a_registration_refreshed_behind_a_new_credential_is_laid_with_it() {
        let mut replica = Replica::new();
        let registered = facts(A, 1);
        replica.register(registered.clone(), Some(KEY_A));
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_A)));
        let id = replica.id(A);
        replica.told();

        // The row is written again behind another credential, and the
        // follow builds its object while the registration is in flight.
        let refreshed = replica.kept(&registered, Some(KEY_B));
        let row = replica.store.upsert(registered.clone()).id();
        replica.follow();
        assert_eq!(replica.told(), [("replaced", id)]);
        assert!(replica.holds_credential(A, Some(KEY_B)));

        // The object already holds what its row takes.
        assert!(refreshed.accepted(row));
        assert_eq!(replica.lay_again(&registered), 0);
    }

    #[test]
    fn a_follow_forgets_the_credential_of_an_address_its_last_row_left() {
        let replica = Replica::new();
        replica.register(facts(A, 1), Some(KEY_A));
        replica.follow();

        // The removal is another writer's: this host calls nothing.
        replica.remove(A);
        replica.follow();
        assert!(replica.credentials.is_empty());

        // A later worker at the address is built without the old key.
        replica.store.upsert(facts(A, 2));
        replica.follow();
        assert!(replica.holds_credential(A, None));
    }

    #[test]
    fn a_follow_forgets_a_worker_removed_before_its_first_object() {
        let replica = Replica::new();
        replica.follow();
        let token = replica.kept(&facts(A, 1), Some(KEY_A));
        let row = replica.store.upsert(facts(A, 1)).id();
        assert!(token.accepted(row));
        // Another writer's removal, before this holder's next follow.
        replica.remove(A);
        replica.follow();
        assert!(replica.credentials.is_empty());
        replica.store.upsert(facts(A, 2));
        replica.follow();
        assert!(replica.holds_credential(A, None));
    }

    /// A removal followed before its registration's answer leaves that
    /// read's forgetting no bound row. The answer then binds the row that
    /// left: the next read finds nothing in the store, and forgets it by
    /// the keeper's clock.
    #[test]
    fn a_follow_forgets_a_worker_removed_before_its_registration_was_answered() {
        let replica = Replica::new();
        replica.follow();
        let token = replica.kept(&facts(A, 1), Some(KEY_A));
        let row = replica.store.upsert(facts(A, 1)).id();
        // Another writer's removal, followed before the door's answer
        // reaches the token.
        replica.remove(A);
        replica.follow();
        assert!(token.accepted(row));
        assert_eq!(replica.lay_again(&facts(A, 1)), 0);

        // A read that finds nothing forgets the row all the same.
        replica.follow();
        assert!(replica.credentials.is_empty());
        replica.store.upsert(facts(A, 2));
        replica.follow();
        assert!(replica.holds_credential(A, None));
    }

    #[test]
    fn a_worker_that_takes_a_removed_workers_address_between_two_follows_is_built_without_its_key()
    {
        let replica = Replica::new();
        replica.register(facts(A, 1), Some(KEY_A));
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_A)));

        // Another host takes the removal and registers the next worker at
        // the address, before this holder follows. The read that builds
        // the new object has not forgotten the old row yet.
        replica.remove(A);
        replica.store.upsert(facts(A, 2));
        replica.follow();
        assert!(replica.holds_credential(A, None));
        assert!(replica.credentials.is_empty());
    }

    #[test]
    fn a_follow_forgets_nothing_while_another_row_dials_the_address() {
        let replica = Replica::new();
        replica.register(rank_facts(A, C, 0), Some(KEY_C));
        replica.register(rank_facts(B, C, 1), Some(KEY_C));
        replica.follow();

        replica.remove(A);
        replica.follow();
        assert_eq!(replica.credentials.len(), 1);
        // The other rank's row is written again: it is bound still.
        replica.store.upsert(rank_facts(B, C, 1));
        replica.follow();
        assert!(replica.holds_credential(B, Some(KEY_C)));
        replica.remove(B);
        replica.follow();
        assert!(replica.credentials.is_empty());
    }

    #[test]
    fn a_follow_keeps_what_a_registration_in_flight_stated() {
        let replica = Replica::new();
        replica.register(facts(A, 1), Some(KEY_A));
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_A)));

        replica.remove(A);
        let next = replica.kept(&facts(A, 2), Some(KEY_B));
        replica.follow();
        // The removed row is forgotten. What the flight stated stays.
        assert_eq!(replica.credentials.len(), 1);
        let row = replica.store.upsert(facts(A, 2)).id();
        assert!(next.accepted(row));
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_B)));
    }

    #[test]
    fn a_removal_after_a_registration_that_moved_nothing_forgets_the_credential() {
        // On either answer of the registration that moved nothing.
        for put_back in [true, false] {
            let replica = Replica::new();
            let row = replica.register(facts(A, 1), Some(KEY_A));
            replica.follow();

            // The same registration again: a door writes nothing over
            // equal facts, and nothing is to be laid again.
            let again = replica.kept(&facts(A, 1), Some(KEY_A));
            if put_back {
                assert!(!again.put_back());
            } else {
                assert!(!again.accepted(row));
            }
            replica.follow();

            replica.remove(A);
            replica.follow();
            assert!(replica.credentials.is_empty());
        }
    }

    #[test]
    fn a_registration_refused_at_an_address_that_stands_leaves_the_standing_workers_credential() {
        let replica = Replica::new();
        replica.register(facts(A, 1), Some(KEY_A));
        replica.follow();

        let before = replica.at(A);
        assert!(!replica.kept(&facts(A, 1), Some(KEY_B)).put_back());
        replica.follow();
        assert!(Arc::ptr_eq(&before, &replica.at(A)));

        replica.store.upsert(facts(A, 1));
        replica.follow();
        assert!(!Arc::ptr_eq(&before, &replica.at(A)));
        assert!(replica.holds_credential(A, Some(KEY_A)));
    }

    #[test]
    fn a_registration_that_moved_no_row_behind_a_new_credential_is_laid_again_with_it() {
        let mut replica = Replica::new();
        let row = replica.register(facts(A, 1), Some(KEY_A));
        replica.follow();
        let id = replica.id(A);
        let standing = replica.objects.standing(&id);
        replica.told();

        // The same worker registered with another credential: a door
        // writes nothing over equal facts, so no row moves and no object
        // is built. The acceptance says the address is to be laid again.
        let again = replica.kept(&facts(A, 1), Some(KEY_B));
        assert!(again.accepted(row));
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_A)));

        // The holder lays the object again: one replacement under its id,
        // at the revision its row stands at.
        assert_eq!(replica.lay_again(&facts(A, 1)), 1);
        assert!(replica.holds_credential(A, Some(KEY_B)));
        assert_eq!(replica.told(), [("replaced", id.clone())]);
        assert_eq!(replica.objects.standing(&id), standing);

        // An object that holds what its row takes is not laid again.
        assert_eq!(replica.lay_again(&facts(A, 1)), 0);
        assert!(replica.told().is_empty());

        // A registration that states no credential changes it too, and
        // the object is laid again without one.
        assert!(replica.kept(&facts(A, 1), None).accepted(row));
        assert_eq!(replica.lay_again(&facts(A, 1)), 1);
        assert!(replica.holds_credential(A, None));
    }

    #[test]
    fn an_object_built_with_a_refused_credential_is_laid_again_with_what_stood() {
        let replica = Replica::new();
        replica.register(facts(A, 1), Some(KEY_A));
        replica.follow();

        // The standing row is written again while a registration is in
        // flight: its object is built with the stated credential.
        let refused = replica.kept(&facts(A, 1), Some(KEY_B));
        replica.store.upsert(facts(A, 1));
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_B)));

        // The put-back says so, and the holder lays the object again.
        assert!(refused.put_back());
        assert_eq!(replica.lay_again(&facts(A, 1)), 1);
        assert!(replica.holds_credential(A, Some(KEY_A)));
    }

    #[test]
    fn a_row_built_while_a_first_registration_is_in_flight_loses_its_credential_at_a_put_back() {
        let replica = Replica::new();
        // The row came through another host while a registration here
        // stated a credential for its address and was then refused.
        let refused = replica.kept(&facts(A, 1), Some(KEY_A));
        replica.store.upsert(facts(A, 1));
        replica.follow();
        assert!(replica.holds_credential(A, Some(KEY_A)));

        assert!(refused.put_back());
        assert_eq!(replica.lay_again(&facts(A, 1)), 1);
        assert!(replica.holds_credential(A, None));
        assert!(replica.credentials.is_empty());
    }

    #[test]
    fn a_registration_refused_behind_a_rotation_not_yet_laid_leaves_the_rotated_credential() {
        let replica = Replica::new();
        let row = replica.register(facts(A, 1), Some(KEY_A));
        replica.follow();

        // The credential alone changes. It is accepted, and the holder
        // has not laid the address again yet.
        assert!(replica.kept(&facts(A, 1), Some(KEY_B)).accepted(row));
        // The next registration states the first credential again. The
        // holder lays the address meanwhile: the object already holds
        // what is stated, and the read marks the flight.
        let refused = replica.kept(&facts(A, 1), Some(KEY_A));
        assert_eq!(replica.lay_again(&facts(A, 1)), 0);
        // Refused: the rotated credential stands again, and the put-back
        // says an object holds the refused one.
        if refused.put_back() {
            replica.lay_again(&facts(A, 1));
        }
        assert!(replica.holds_credential(A, Some(KEY_B)));
    }

    #[test]
    fn laying_again_reaches_each_rank_row_that_dials_the_address_and_no_other() {
        let replica = Replica::new();
        let first = replica.register(rank_facts(A, C, 0), Some(KEY_C));
        replica.register(rank_facts(B, C, 1), Some(KEY_C));
        replica.register(facts(D, 1), Some(KEY_A));
        replica.follow();
        let other = replica.at(D);

        // One rank's registration states another credential: it is the
        // engine's, for each rank row bound there.
        let rotated = replica.kept(&rank_facts(A, C, 0), Some(KEY_B));
        assert!(rotated.accepted(first));
        assert_eq!(replica.lay_again(&rank_facts(A, C, 0)), 2);
        assert!(replica.holds_credential(A, Some(KEY_B)));
        assert!(replica.holds_credential(B, Some(KEY_B)));
        assert!(Arc::ptr_eq(&other, &replica.at(D)));

        // An address no row dials lays nothing.
        assert_eq!(replica.lay_again(&facts(E, 1)), 0);
    }

    #[test]
    fn laying_again_follows_the_store_first() {
        let replica = Replica::new();
        let token = replica.kept(&facts(A, 1), Some(KEY_A));
        let row = replica.store.upsert(facts(A, 1)).id();
        assert!(token.accepted(row));
        // No follow ran: the call reads the row and builds its object with
        // its credential, so nothing is left to lay again.
        assert_eq!(replica.lay_again(&facts(A, 1)), 0);
        assert!(replica.holds_credential(A, Some(KEY_A)));

        // A row that left behind the last follow is ended, not laid.
        assert!(replica.kept(&facts(A, 1), Some(KEY_B)).accepted(row));
        replica.remove(A);
        assert_eq!(replica.lay_again(&facts(A, 1)), 0);
        assert!(replica.objects.registry().get_by_url(A).is_none());
    }

    #[test]
    fn a_registration_with_no_credential_clears_one_a_removed_worker_left() {
        let replica = Replica::new();
        replica.register(facts(A, 1), Some(KEY_A));
        replica.follow();
        replica.remove(A);

        replica.register(facts(A, 1), None);
        replica.follow();
        assert!(replica.holds_credential(A, None));
    }

    /// A registration answered behind the follow's read of the keeper's
    /// clock binds a row the read may not hold yet, so the forgetting
    /// behind that read keeps it. Its worker left inside the same read:
    /// the next read finds nothing, and forgets the row all the same.
    #[test]
    fn a_forgetting_the_clock_refused_is_tried_again_behind_the_next_read() {
        let registry = Arc::new(WorkerRegistry::<Members>::new());
        let credentials = Arc::new(Credentials::new());
        let objects =
            Objects::new(Arc::clone(&registry)).with_credentials(Arc::clone(&credentials));
        let store = Late::default();
        let token = credentials
            .keep(&facts(A, 1), Some(KEY_A.to_owned()))
            .expect("no registration in flight at the address");
        let first = store.upsert(facts(A, 1)).id();
        assert!(token.accepted(first));
        objects.follow(&store);

        // Between the follow's read of the clock and its read of the
        // tail: the worker leaves, a second one registers at the address
        // with another key, and leaves too.
        let keeper = Arc::clone(&credentials);
        *store.before_tail.lock() = Some(Box::new(move |store| {
            let row = store.by_key(A).expect("the first worker's row");
            assert!(store.remove(row.id, row.revision).is_some());
            let token = keeper
                .keep(&facts(A, 2), Some(KEY_B.to_owned()))
                .expect("no registration in flight at the address");
            let second = store.upsert(facts(A, 2)).id();
            assert!(token.accepted(second));
            let row = store.by_key(A).expect("the second worker's row");
            assert!(store.remove(row.id, row.revision).is_some());
        }));
        objects.follow(&store);
        assert!(credentials.is_empty(), "the second worker's too");

        // A later worker at the address takes no key.
        store.upsert(facts(A, 3));
        objects.follow(&store);
        let later = registry.get_by_url(A).expect("the later worker's object");
        assert!(later.api_key().is_none());
    }

    #[test]
    fn the_row_is_answered_only_for_the_object_the_entry_was_written_for() {
        let replica = Replica::new();
        let row = replica.store.upsert(facts(A, 1)).id();
        replica.store.upsert(facts(B, 1));
        replica.follow();
        let id = replica.id(A);
        let laid = replica.at(A);
        let standing = Standing {
            row,
            revision: replica.row(A).revision,
        };
        assert_eq!(replica.objects.standing_of(&id, &laid), Some(standing));
        assert_eq!(replica.objects.standing_of(&replica.id(B), &laid), None);
        assert_eq!(
            replica.objects.standing_of(&id, &object(&replica.row(A))),
            None
        );

        replica.patch(A, labels("gold"));
        replica.follow();
        assert_eq!(replica.objects.standing_of(&id, &laid), None);
        assert_eq!(
            replica.objects.standing_of(&id, &replica.at(A)),
            Some(Standing {
                row,
                revision: replica.row(A).revision,
            })
        );
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

    /// By default a health or status write leaves the object alone.
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
        let store = Late {
            store: Arc::clone(&replica.store),
            after_get: Mutex::new(Some(Box::new(|_| panic!("a status change reads no row")))),
            ..Late::default()
        };
        replica.objects.follow(&store);
        assert!(Arc::ptr_eq(&before, &replica.at(A)));
        assert_eq!(replica.objects.standing(&id), standing);
        assert!(replica.told().is_empty());
        assert_eq!(replica.at(A).status(), ObjectStatus::Pending);
    }

    #[test]
    fn a_followed_status_makes_a_worker_available_only_while_ready() {
        let mut replica = Replica::new();
        replica.objects = replica.objects.with_status_followed();
        replica.follow();
        replica.store.upsert(facts(A, 1));
        replica.follow();
        let row = replica.row(A);
        let (id, worker) = (replica.id(A), replica.at(A));
        assert_eq!(worker.status(), ObjectStatus::Pending);
        assert!(!worker.is_available());
        assert_eq!(replica.told(), [("registered", id.clone())]);

        for (row_status, object_status, available) in [
            (WorkerStatus::Ready, ObjectStatus::Ready, true),
            (WorkerStatus::NotReady, ObjectStatus::NotReady, false),
        ] {
            assert!(replica
                .store
                .set_status(row.id, row_status, row.revision)
                .is_some());
            replica.follow();
            assert!(Arc::ptr_eq(&worker, &replica.at(A)));
            assert_eq!(worker.status(), object_status);
            assert_eq!(worker.is_available(), available);
            // The registry's subscribers hear each change once.
            assert_eq!(replica.told(), [("status", id.clone())]);
        }

        for _ in 0..CircuitBreakerConfig::default().failure_threshold {
            worker.record_circuit_breaker_outcome(false);
        }
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
        assert!(replica
            .objects
            .registry()
            .set_worker_overloaded(&worker, true));
        assert!(worker.set_stall(Some(StallReason::Unreachable)));
        assert!(replica
            .store
            .set_status(row.id, WorkerStatus::Ready, row.revision)
            .is_some());
        replica.told();
        replica.follow();
        assert_eq!(worker.status(), ObjectStatus::Ready);
        assert_eq!(replica.told(), [("status", id)]);
        // A promotion comes back as any promotion does: the open breaker
        // goes on trial. The host's own vetoes stand.
        assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);
        assert!(worker.is_overloaded());
        assert_eq!(worker.stall_reason(), Some(StallReason::Unreachable));
        assert!(!worker.is_available());
    }

    #[test]
    fn a_status_at_a_new_row_revision_waits_for_the_new_object() {
        let mut replica = Replica::new();
        replica.objects = replica.objects.with_status_followed();
        replica.store.upsert(facts(A, 1));
        replica.follow();
        let row = replica.row(A);
        let before = replica.at(A);
        let change = replica
            .store
            .set_status(row.id, WorkerStatus::Ready, row.revision)
            .expect("the status moved");
        replica.patch(A, labels("gold"));

        replica.objects.apply(replica.store.as_ref(), change);
        assert_eq!(before.status(), ObjectStatus::Pending);
        replica.follow();
        let after = replica.at(A);
        assert!(!Arc::ptr_eq(&before, &after));
        assert_eq!(after.status(), ObjectStatus::Ready);
    }

    #[test]
    fn a_status_change_leaves_an_object_the_holder_did_not_lay_alone() {
        let mut replica = Replica::new();
        replica.objects = replica.objects.with_status_followed();
        replica.store.upsert(facts(A, 1));
        replica.follow();
        let row = replica.row(A);
        assert!(replica
            .store
            .set_status(row.id, WorkerStatus::Ready, row.revision)
            .is_some());
        replica.follow();
        let registry = replica.objects.registry();
        let foreign = object(&row);
        assert!(registry.replace(&replica.id(A), Arc::clone(&foreign)));
        assert_eq!(foreign.status(), ObjectStatus::Ready);

        assert!(replica
            .store
            .set_status(row.id, WorkerStatus::NotReady, row.revision)
            .is_some());
        replica.follow();
        assert_eq!(foreign.status(), ObjectStatus::Ready);
    }

    #[test]
    fn a_snapshot_follows_status_for_both_kept_and_replaced_objects() {
        let mut replica = Replica::new();
        replica.objects = replica.objects.with_status_followed();
        replica.store.upsert(facts(A, 1));
        replica.store.upsert(facts(B, 1));
        replica.follow();
        let (a, b) = (replica.at(A), replica.at(B));
        let (id_a, id_b) = (replica.id(A), replica.id(B));
        assert!(!a.is_available() && !b.is_available());
        replica.told();

        let other = MemoryStore::new();
        other.upsert(facts(A, 1));
        other.upsert(labelled(B, 1, "gold"));
        for url in [A, B] {
            let row = other.by_key(url).expect("a row in the snapshot");
            assert!(other
                .set_status(row.id, WorkerStatus::Ready, row.revision)
                .is_some());
        }
        assert!(replica.store.restore_from(other.cut(Kinds::Watched)));
        replica.follow();

        assert!(Arc::ptr_eq(&a, &replica.at(A)));
        assert!(a.is_available());
        assert!(!Arc::ptr_eq(&b, &replica.at(B)));
        assert!(replica.at(B).is_available());
        // Each object's change is told, kept or laid again.
        let told = replica.told();
        for id in [&id_a, &id_b] {
            assert!(told.contains(&("status", id.clone())), "{told:?}");
        }

        let row = other.by_key(A).expect("a row in the snapshot");
        assert!(other
            .set_status(row.id, WorkerStatus::NotReady, row.revision)
            .is_some());
        assert!(replica.store.restore_from(other.cut(Kinds::Watched)));
        replica.follow();
        assert!(Arc::ptr_eq(&a, &replica.at(A)));
        assert!(!a.is_available());
        assert_eq!(replica.told(), [("status", id_a)]);
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
    /// the rows that went leave first, then each row that does not stand as
    /// its object was built is laid again, and the next read is the tail
    /// again. A row only a status write touched keeps the very object. A
    /// registration refreshed with equal facts moved its revision and is
    /// laid again, as the tail lays it. So is the row that left and came
    /// back under its id and its revision with a new label: its facts moved.
    #[test]
    fn a_read_the_tail_no_longer_reaches_takes_the_snapshot_and_lays_the_moved_rows_again() {
        let mut replica = Replica::new();
        let gone = replica.store.upsert(facts(A, 1)).id();
        replica.store.upsert(facts(B, 1));
        let replaced = replica.store.upsert(facts(C, 1)).id();
        replica.store.upsert(facts(E, 1));
        let back = replica.store.upsert(facts(F, 1)).id();
        replica.store.upsert(facts(G, 1));
        replica.follow();
        let (a, b, c) = (replica.id(A), replica.id(B), replica.id(C));
        let (e, f, g) = (replica.id(E), replica.id(F), replica.id(G));
        let stood = replica.at(E);
        let read_to = replica.store.epoch();
        replica.told();

        replica.remove(A);
        replica.patch(B, labels("gold"));
        let next = replica.store.upsert(facts(C, 2)).id();
        let arrived = replica.store.upsert(facts(D, 1)).id();
        replica.remove(F);
        assert_eq!(replica.store.upsert(labelled(F, 1, "gold")).id(), back);
        assert_eq!(replica.row(F).revision, 1, "the first revision again");
        // A status write moves neither E's facts nor its revision.
        let unmoved = replica.row(E);
        assert!(replica
            .store
            .set_status(unmoved.id, WorkerStatus::NotReady, unmoved.revision)
            .is_some());
        assert_eq!(replica.row(E).revision, unmoved.revision);
        // A registration refreshed with equal facts moves G's revision alone.
        let refreshed = replica.row(G);
        replica.store.upsert(facts(G, 1));
        assert_eq!(replica.row(G).facts, refreshed.facts);
        assert_ne!(replica.row(G).revision, refreshed.revision);
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
        assert_eq!((replica.objects.len(), registry.len()), (6, 6));
        assert!(registry.get_by_url(A).is_none());
        assert_eq!(replica.objects.object(gone), None);
        assert_eq!(replica.objects.object(replaced), None);
        assert_eq!(replica.id(B), b);
        assert_eq!(replica.tier(B).as_deref(), Some("gold"));
        assert_eq!((replica.id(E), replica.id(F)), (e.clone(), f.clone()));
        assert_eq!(replica.tier(F).as_deref(), Some("gold"));
        // E stands as its object was built: the very object, and its entry
        // names the row's revision still.
        assert!(Arc::ptr_eq(&stood, &replica.at(E)));
        let standing = replica.objects.standing(&e).expect("E's row");
        assert_eq!(standing.revision, replica.row(E).revision);
        // G's revision moved: an object in its place, built at that revision.
        assert_eq!(replica.id(G), g);
        let standing = replica.objects.standing(&g).expect("G's row");
        assert_eq!(standing.revision, replica.row(G).revision);
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
            ("replaced", f),
            ("replaced", g),
        ];
        assert_eq!((told.len(), laid), (7, HashSet::from(each)));

        replica.remove(D);
        replica.follow();
        assert!(registry.get_by_url(D).is_none());
        assert_eq!(replica.told().len(), 1);
    }

    /// A store that runs a callback after one row read, another just before
    /// a tail read, and can take another store's rows whole there.
    #[derive(Default)]
    struct Late {
        store: Arc<MemoryStore>,
        after_get: Mutex<Option<AfterGet>>,
        before_tail: Mutex<Option<AfterGet>>,
        turn: Mutex<Option<(MemoryStore, WorkerId)>>,
    }

    type AfterGet = Box<dyn FnOnce(&MemoryStore) + Send>;

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
            if let Some(after_get) = self.after_get.lock().take() {
                after_get(&self.store);
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

        fn history(&self) -> u64 {
            self.store.history()
        }

        fn changes_since(&self, epoch: u64) -> MembershipTail {
            if let Some(before_tail) = self.before_tail.lock().take() {
                before_tail(&self.store);
            }
            if let Some((other, row)) = self.turn.lock().take() {
                assert!(self.store.restore_from(other.cut(Kinds::Watched)));
                let revision = self.store.get(row).expect("the row").revision;
                assert!(self.store.patch(row, labels("gold"), revision).is_some());
            }
            self.store.changes_since(epoch)
        }
    }

    #[test]
    fn a_write_that_lands_while_a_read_works_is_read_before_the_read_returns() {
        let registry = Arc::new(WorkerRegistry::<Members>::new());
        let objects = Objects::new(Arc::clone(&registry));
        let store = Late::default();
        objects.follow(&store);
        store.upsert(facts(A, 1));
        *store.after_get.lock() = Some(Box::new(|store| {
            store.upsert(facts(B, 1));
        }));
        objects.follow(&store);
        assert!(registry.get_by_url(A).is_some());
        assert!(registry.get_by_url(B).is_some(), "the late row");
        assert_eq!(objects.len(), 2);
    }

    /// A restore that lands between a read of the store's history and the
    /// read of its tail: the tail names a key's new incarnation, and no
    /// change said the old one left.
    #[test]
    fn a_restore_inside_a_read_never_puts_a_new_incarnation_in_the_old_ones_place() {
        let registry = Arc::new(WorkerRegistry::<Members>::new());
        let mut events = registry.subscribe_events();
        let objects = Objects::new(Arc::clone(&registry));
        let store = Late::default();
        let old = store.upsert(facts(A, 1)).id();
        objects.follow(&store);
        let before = objects.object(old).expect("the old one's object");
        registry.get(&before).expect("a worker").increment_load();
        told(&mut events);

        let other = MemoryStore::new();
        let new = other.upsert(facts(A, 2)).id();
        assert_eq!(other.epoch(), store.epoch());
        *store.turn.lock() = Some((other, new));
        objects.follow(&store);

        assert_eq!(objects.object(old), None);
        let after = objects.object(new).expect("the new one's object");
        assert_ne!(before, after);
        assert_eq!(registry.get(&after).expect("a worker").load(), 0);
        let seen = told(&mut events);
        let first = seen.first().expect("an event");
        assert_eq!(
            *first,
            ("removed", before),
            "the old one's object leaves first"
        );
        let rest = seen.iter().skip(1);
        assert!(rest.clone().count() > 0 && rest.clone().all(|(_, object)| *object == after));
        assert_eq!((objects.len(), registry.len()), (1, 1));
    }

    #[test]
    fn an_adapter_write_waits_for_the_holders_status_read() {
        let registry = Arc::new(WorkerRegistry::<Members>::new());
        let objects = Arc::new(Objects::new(Arc::clone(&registry)).with_status_followed());
        let store = Late::default();
        let row = store.upsert(facts(A, 1)).id();
        objects.follow(&store);
        let id = objects.object(row).expect("the row's object");
        let worker = registry.get(&id).expect("a registered worker");
        let revision = store.get(row).expect("the worker's row").revision;
        assert!(store
            .set_status(row, WorkerStatus::Ready, revision)
            .is_some());
        objects.follow(&store);
        assert_eq!(worker.status(), ObjectStatus::Ready);

        let adapter = HealthAdapter::new(
            Arc::clone(&objects),
            Arc::clone(&store.store),
            Arc::new(SoleLead),
            SoleLead::TERM,
        );
        assert!(store
            .set_status(row, WorkerStatus::NotReady, revision)
            .is_some());
        let (reading, read) = mpsc::channel();
        let (attempting, attempt) = mpsc::channel();
        let (written, write) = mpsc::channel();
        *store.after_get.lock() = Some(Box::new(move |_| {
            reading.send(()).expect("the adapter waits for the read");
            attempt
                .recv_timeout(Duration::from_secs(5))
                .expect("the adapter starts its write");
            assert_eq!(
                write.recv_timeout(Duration::from_secs(1)),
                Err(mpsc::RecvTimeoutError::Timeout)
            );
        }));
        thread::scope(|scope| {
            let connected = scope.spawn(move || {
                read.recv_timeout(Duration::from_secs(5))
                    .expect("the holder reads the row");
                attempting.send(()).expect("the holder waits for the write");
                let answer = adapter.connected(&id, worker.revision());
                let _ = written.send(());
                assert_eq!(answer, Some((ObjectStatus::NotReady, ObjectStatus::Ready)));
                assert_eq!(worker.status(), ObjectStatus::Ready);
            });
            objects.follow(&store);
            connected.join().expect("the adapter finished");
        });
        assert_eq!(
            store.get(row).map(|row| row.status),
            Some(WorkerStatus::Ready)
        );
        assert_eq!(
            registry.get_by_url(A).expect("the worker").status(),
            ObjectStatus::Ready
        );
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
    fn a_store_that_took_another_historys_rows_has_its_moved_rows_laid_again() {
        let mut replica = Replica::new();
        let row = replica.store.upsert(facts(A, 1)).id();
        replica.store.upsert(facts(B, 1));
        replica.store.upsert(facts(D, 1));
        replica.follow();
        let (a, b) = (replica.id(A), replica.id(B));
        let stood = replica.at(D);
        replica.told();

        let other = MemoryStore::new();
        other.upsert(WorkerFacts {
            labels: BTreeMap::from([("tier".to_string(), "gold".to_string())]),
            ..facts(A, 1)
        });
        other.upsert(facts(C, 1));
        other.upsert(facts(D, 1));
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
        assert_eq!(replica.objects.len(), 3);
        // D states the same facts in the other history: the very object.
        assert!(Arc::ptr_eq(&stood, &replica.at(D)));
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
