//! The holder of a replica's worker objects: it follows the store's
//! membership and writes the registry. A leader's door writes a row through
//! it, and the store follows that write ([`mod@write`]). A standby's applier
//! hands it each event of its leader's stream ([`mod@standby`]). A host can
//! also ask it to follow the rows' status, or to build each object with the
//! thresholds its own load monitor judges overload by. A door reads the row
//! that stands off it ([`mod@read`]).

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
    health::overload::OverloadThresholds,
    identity::basic::Worker,
    registry::{Generation, WorkerId as ObjectId, WorkerRegistry},
};

#[cfg(feature = "in-process")]
use super::members::{EncodeColumns, Members, Stated};
#[cfg(feature = "in-process")]
use super::reach::{ListReach, Reach};
use super::{
    credentials::Credentials,
    membership_writer::MembershipWriteRefused,
    object::{builder, dial_address, external, status, thresholds, unchecked},
    remainder::RowRemainder,
};
#[cfg(feature = "in-process")]
use crate::{
    document::spec::Pairing,
    engine::{FixedFacts, LentList, PartnerClasses, Role},
    store::{
        encoding::{EncodeClasses, EncodeMember, EncodeOffer},
        pairing::{Member, PairClasses},
    },
};
use crate::{
    engine::view::WorkerId,
    store::{MembershipChange, MembershipTail, Reachability, WorkerFacts, WorkerRow, WorkersStore},
};

mod host_count;
mod read;
mod standby;
mod write;

/// The row an object stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowRevision {
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

/// A row and the object that stands for it.
struct Laid {
    /// What of the row the object does not hold as stated, with the row's
    /// id and the revision the object was built from.
    remainder: Arc<RowRemainder>,
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
    /// The rows of each engine's ranks, by the address they dial: what
    /// routes a rank's batch on the engine's one events stream.
    ranks: HashMap<String, Vec<(u16, WorkerId)>>,
}

impl Held {
    /// State `row`'s facts in the entry, where every lay states them: at a
    /// build, and where a row another maker made again is stated anew. The
    /// row's place among its engine's ranks moves with them: a row laid
    /// again leaves the rank its old facts stated.
    fn state(&mut self, row: &WorkerRow) {
        self.unrank(row.id);
        self.rank(row.id, &row.facts);
        self.facts.insert(row.id, row.facts.clone());
    }

    /// `row` stands among its engine's ranks, by the facts it is laid from.
    fn rank(&mut self, row: WorkerId, facts: &WorkerFacts) {
        self.ranks
            .entry(dial_address(facts).to_owned())
            .or_default()
            .push((facts.dp_rank, row));
    }

    /// Take `row` out of its engine's ranks, by the facts it was laid from.
    fn unrank(&mut self, row: WorkerId) {
        let Some(address) = self.facts.get(&row).map(dial_address) else {
            return;
        };
        let Some(rows) = self.ranks.get_mut(address) else {
            return;
        };
        rows.retain(|&(_, ranked)| ranked != row);
        if rows.is_empty() {
            self.ranks.remove(address);
        }
    }

    /// Take the row `id` out of every table of the entry: its place among
    /// its engine's ranks, its facts, its object's id, and that object's
    /// own entry. A row that leaves goes through here, at its object's end
    /// and where its next life takes its object's place.
    fn leave(&mut self, id: WorkerId) {
        // The rank first: it is found by the row's facts.
        self.unrank(id);
        self.facts.remove(&id);
        if let Some(object) = self.objects.remove(&id) {
            self.rows.remove(&object);
        }
    }
}

/// A replica's worker objects: for every row of the membership store that
/// stands one object in the registry, and none beside.
///
/// [`Self::follow`] reads what the store's membership did since its last
/// read: each change that neither a door nor a standby's applier wrote
/// through the holder, a status change among them. It builds the
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
/// revision never moves under an object that stands.
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
    /// Whether the object of a host-local row is built with its checks off.
    host_local_unchecked: bool,
    /// The overload thresholds an object built from an engine's row states
    /// for each signal its row names none for, where the host asked for
    /// objects with thresholds ([`Self::with_overload_thresholds`]). An
    /// external row's object states its row's own alone.
    overload_defaults: Option<OverloadThresholds>,
    /// One registration that keeps its credential at a time, from its keep
    /// to the lay-again its keeper's answer asks for ([`Self::keeping`]).
    keeping: Mutex<()>,
    /// Where the last read ended. Held through a read and through a write
    /// of the holder's own ([`mod@write`]): one follows or writes at a time.
    ///
    /// The order of the locks, said here once: a door's turn, then this
    /// one, then the store's membership lock. Nothing that holds this one
    /// takes a door's turn, and no lock of the store is held over a write
    /// of the registry.
    read_to: Mutex<Place>,
    held: RwLock<Held>,
    /// How many members of a generation the holder held no row's object
    /// for when it cut the generation's lists ([`Self::unstated`]).
    #[cfg(feature = "in-process")]
    unstated: AtomicU64,
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

    /// For a host that is no gateway: it holds the socket of no host-local
    /// wire, so the object of a row that one gateway alone reaches is built
    /// with its checks off, whatever the row's policy says. A health loop
    /// over the registry then dials no worker its host cannot reach.
    #[must_use]
    pub fn with_host_local_unchecked(mut self) -> Self {
        self.host_local_unchecked = true;
        self
    }

    /// Build each object from a row with overload thresholds: the row's
    /// own, and `defaults` for each signal the row names none for. A row
    /// whose thresholds change is laid again, so its object states the new
    /// ones.
    ///
    /// A row that states the external runtime takes its own alone, at its
    /// first build and at every later one. A gateway registers an engine's
    /// worker with its default thresholds and an external worker with none,
    /// so its monitor judges an external worker by nothing where its
    /// registration names nothing. A gateway's update of a worker's
    /// properties builds the worker again with the defaults, an external
    /// worker too: this holder does not.
    ///
    /// For a host whose own load monitor judges overload: the worker
    /// crate's monitor writes its verdict on an object by the thresholds
    /// the object states, at each load report of its worker.
    ///
    /// Without it an object built from a row states no threshold, and a
    /// load monitor writes no verdict on it: the store alone judges such a
    /// worker, for its own decisions, and the holder does not copy its
    /// verdict onto the object's flag.
    ///
    /// The monitor and the store do not read the same thresholds. The
    /// crate's two defaults, which the store reads under a binding whose
    /// block is on for each signal a row names none for, are not read here:
    /// `defaults` stands in their place, and with no signal in `defaults`
    /// the monitor does not judge a row that names none. No document names
    /// a threshold. The store judges an engine's ranks together, and the
    /// monitor judges each object by its own report.
    #[must_use]
    pub fn with_overload_thresholds(mut self, defaults: OverloadThresholds) -> Self {
        self.overload_defaults = Some(defaults);
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
            host_local_unchecked: false,
            overload_defaults: None,
            keeping: Mutex::new(()),
            read_to: Mutex::new(Place::default()),
            held: RwLock::new(Held::default()),
            #[cfg(feature = "in-process")]
            unstated: AtomicU64::new(0),
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

    /// The keeper the holder reads at an object's build, where the host
    /// handed one ([`Self::with_credentials`]). A registration line that
    /// keeps reads it here, so the line's write and the build read one
    /// keeper.
    pub fn credentials(&self) -> Option<&Arc<Credentials>> {
        self.credentials.as_ref()
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
                        .get(&laid.remainder.standing.row)
                        .is_some_and(|dialing| dial_address(dialing) == address)
                        && credentials
                            .of(address, laid.remainder.standing.row)
                            .as_ref()
                            != laid.object.api_key()
                })
                .map(|laid| laid.remainder.standing.row)
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

    /// Run `write`, the write of a registration that keeps its credential,
    /// in a turn of its own, and return its answer. `write` is the whole of
    /// it: the keep, the door's write, the keeper's answer and the
    /// lay-again that answer asks for. One runs at a time, so a credential
    /// is settled and its object laid again before the next registration
    /// reads either.
    ///
    /// `over` is the object a guarded write replaces, the one its asker
    /// read and built from. A registration that states another credential
    /// alone moves no row and lays another object, so an object changes
    /// while its row's revision stands. The call refuses the write as
    /// moved, ahead of it, where another object stands under `url` by then.
    pub(super) fn keeping<T>(
        &self,
        url: &str,
        over: Option<&Arc<dyn Worker>>,
        write: impl FnOnce() -> T,
    ) -> Result<T, MembershipWriteRefused> {
        let _turn = self.keeping.lock();
        if let Some(seen) = over {
            let stands = self.registry.get_by_url(url);
            if !stands.is_some_and(|stands| Arc::ptr_eq(&stands, seen)) {
                return Err(MembershipWriteRefused::Moved);
            }
        }
        Ok(write())
    }

    /// The registry the objects stand in.
    pub fn registry(&self) -> &Arc<WorkerRegistry<G>> {
        &self.registry
    }

    /// Whether this host dials `worker`. No for the object of a row that
    /// one gateway alone reaches, where the host asked for such an object
    /// with its checks off ([`Self::with_host_local_unchecked`]). Yes for
    /// every other, for a worker the holder laid no object for, and for
    /// every worker of a host that asked for no such object. What a
    /// host's load monitor asks its workers by
    /// ([`load_monitor`](super::load_monitor)).
    ///
    /// The answer is the row's reach as the object's entry holds it. A
    /// row that states another reach is laid again and its object built
    /// anew, so the answer stands for an object's life.
    pub(super) fn dials(&self, worker: &dyn Worker) -> bool {
        if !self.host_local_unchecked {
            return true;
        }
        let Some(object) = self.registry.get_id_by_url(worker.url()) else {
            return true;
        };
        let held = self.held.read();
        held.rows
            .get(&object)
            .is_none_or(|laid| laid.remainder.reachable_from == Reachability::Any)
    }

    /// The row the object under `object` stands for, with the revision it
    /// was built from; `None` for an id the holder laid no object under, or
    /// one whose row went.
    pub fn standing(&self, object: &ObjectId) -> Option<RowRevision> {
        let held = self.held.read();
        held.rows.get(object).map(|laid| laid.remainder.standing)
    }

    /// What the row of the object under `object` states and the object
    /// does not hold as stated: the facts a decision reads of the row
    /// alone among them. `None` as [`Self::standing`] answers it. The
    /// remainder is shared with the entry: a row laid again gets another.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(
            dead_code,
            reason = "a controller's standby reads the instant a row was admitted at"
        )
    )]
    pub(crate) fn remainder(&self, object: &ObjectId) -> Option<Arc<RowRemainder>> {
        let held = self.held.read();
        held.rows
            .get(object)
            .map(|laid| Arc::clone(&laid.remainder))
    }

    /// The row the object under `object` stands for, read back from the
    /// object and what the holder keeps beside it, the two as one turn
    /// of the entry holds them. It is the row the object was laid from,
    /// with the object's own status: no table of rows is read. `None`
    /// as [`Self::standing`] answers it.
    pub(crate) fn row_of(&self, object: &ObjectId) -> Option<WorkerRow> {
        let held = self.held.read();
        held.rows
            .get(object)
            .map(|laid| laid.remainder.row(laid.object.as_ref()))
    }

    /// The row and revision the holder built this object from. A writer
    /// holding the object's registry lock uses this to name that revision.
    pub fn standing_of(&self, id: &ObjectId, object: &Arc<dyn Worker>) -> Option<RowRevision> {
        self.held
            .read()
            .rows
            .get(id)
            .filter(|laid| Arc::ptr_eq(&laid.object, object))
            .map(|laid| laid.remainder.standing)
    }

    /// The registry's id of the object that stands for `row`.
    pub fn object(&self, row: WorkerId) -> Option<ObjectId> {
        self.held.read().objects.get(&row).cloned()
    }

    /// The row of rank `rank` of the engine whose rank 0 row is `row`:
    /// `row` itself for rank 0, else the row that dials the same address
    /// under that rank. `None` where the holder laid no such row, holds no
    /// object for `row`, or `row` stands above rank 0 by the facts it is
    /// laid from. Only an engine's rank 0 row holds the engine's one events
    /// stream, and the stream's writer routes a rank's batch by this
    /// answer: a row laid again above rank 0 routes none.
    pub fn rank_row(&self, row: WorkerId, rank: u16) -> Option<WorkerId> {
        let held = self.held.read();
        let facts = held.facts.get(&row)?;
        if facts.dp_rank != 0 {
            return None;
        }
        if rank == 0 {
            return Some(row);
        }
        held.ranks
            .get(dial_address(facts))?
            .iter()
            .find(|&&(ranked, _)| ranked == rank)
            .map(|&(_, row)| row)
    }

    /// Whether the row `row` stands at rank `rank`, by the facts its
    /// object was laid from. `false` for a row the holder holds no object
    /// for. Asked of the worker a dispatch report names, which may have
    /// left or changed its rank since its decision. One read of the
    /// entry under its read lock, a leaf: the caller may hold a turn of
    /// the leases' files.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "a controller's word to its engine asks it")
    )]
    pub(crate) fn lists_rank(&self, row: WorkerId, rank: u16) -> bool {
        let held = self.held.read();
        held.facts
            .get(&row)
            .is_some_and(|facts| facts.dp_rank == rank)
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
                    let (epoch, rows) = store.all_rows();
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
    /// life with other facts. The facts are compared beside it. A row that
    /// stands so may still be another life, made by another or admitted at
    /// another time: its object stays, and its entry is stated again.
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
                if self.remade(row) {
                    self.restate(row);
                }
                self.follow_status(store, row.id);
                // The store may have given the row a new place in its
                // counts, which holds no count yet: a lay hands its own.
                self.counted(store, row.id);
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
                .is_some_and(|laid| laid.remainder.standing.revision == row.revision)
    }

    /// Whether the entry kept for `row` names another maker of the row, or
    /// another time it was admitted at, than the row states.
    fn remade(&self, row: &WorkerRow) -> bool {
        let held = self.held.read();
        held.objects
            .get(&row.id)
            .and_then(|object| held.rows.get(object))
            .is_some_and(|laid| {
                laid.remainder.maker != row.maker || laid.remainder.admitted_at != row.admitted_at
            })
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
        self.held.write().leave(id);
    }

    /// Lay `row`'s object: one built from the row.
    fn lay<S: WorkersStore + ?Sized>(&self, store: &S, row: &WorkerRow) {
        self.put(store, row, self.build(row));
    }

    /// State the entry of the object that stands for `row` again, from the
    /// row: its maker and the time it was admitted at. The caller found the
    /// row's facts and revision to be the ones the object was built from
    /// ([`Self::stands`]), so nothing a generation's lists, pair classes or
    /// encode columns are derived from moves here, and the registry is not
    /// written.
    fn restate(&self, row: &WorkerRow) {
        let held = &mut *self.held.write();
        let entry = held.objects.get(&row.id);
        let Some(laid) = entry.and_then(|id| held.rows.get_mut(id)) else {
            return;
        };
        laid.remainder = Arc::new(RowRemainder::of(row));
        held.state(row);
    }

    /// Build `row`'s object from its facts, with what the host handed the
    /// holder for it.
    fn build(&self, row: &WorkerRow) -> Arc<dyn Worker> {
        // A row one gateway alone reaches is built unchecked where the
        // host asked for it: nothing here can dial its worker.
        let reached = !self.host_local_unchecked || row.facts.reachable_from == Reachability::Any;
        let mut worker = if reached {
            builder(row)
        } else {
            unchecked(row)
        };
        // The thresholds the host's own monitor judges the object by: the
        // row's own, each over the host's default for its signal. An
        // external row takes no default, as a gateway's external worker
        // takes none.
        if let Some(defaults) = self.overload_defaults {
            worker = worker.overload(thresholds(&row.facts.overload));
            if !external(&row.facts) {
                worker = worker.overload_defaults(defaults);
            }
        }
        // The registry's signal sender: the object says its connect signal
        // and each move of its liveness veto to the collector's loop.
        let mut worker = worker.connect_signal_tx(self.registry.connect_signal_sender());
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
        Arc::new(worker.build())
    }

    /// Put `object` in the registry as `row`'s: registered under a new id
    /// where the row's URL holds no object, else in the place of the one
    /// that stands there, under its id and with its running state. An
    /// object another row holds under the URL is dropped first: a row takes
    /// the place of its own object alone. Where the registry holds nothing
    /// under the URL by then, the object is registered under a new id.
    fn put<S: WorkersStore + ?Sized>(&self, store: &S, row: &WorkerRow, object: Arc<dyn Worker>) {
        // The id first, then the entry, then the registry's write: its
        // event finds the entry.
        let mut id = self.registry.reserve_id_for_url(&row.facts.key);
        // A store restored inside a read can show a key's new incarnation
        // before any change says the old one left.
        let others = |other: &RowRevision| other.row != row.id;
        if let Some(other) = self.standing(&id).filter(others) {
            self.end(other.row);
            id = self.registry.reserve_id_for_url(&row.facts.key);
        }
        {
            let mut entries = self.held.write();
            entries.objects.insert(row.id, id.clone());
            entries.state(row);
            entries.rows.insert(
                id,
                Laid {
                    remainder: Arc::new(RowRemainder::of(row)),
                    object: Arc::clone(&object),
                },
            );
        }
        self.registry.register_or_replace(object);
        // Behind the registry's write: an object laid in another's place
        // takes the standing object's running state there, and with it the
        // leases the store counted on that object.
        self.counted(store, row.id);
        self.follow_status(store, row.id);
    }
}

#[cfg(feature = "in-process")]
impl Objects<Members> {
    /// `generation`'s workers of `role`, in the registry's order, lent with
    /// the cell beside them and with what the holder's rows say of each
    /// ([`FixedFacts`]): for a host that decides over the registry's worker
    /// lists. A single request's list is the workers of [`Role::Single`]
    /// alone. A host takes a model's generation from the registry
    /// ([`WorkerRegistry::model_generation`]).
    ///
    /// A generation's lists are cut at its first ask, once, by one pass
    /// over its members and one read of the holder's rows, whoever
    /// published it: the holder's own write, or another writer of the
    /// registry. Asks that meet there wait for that one pass. Every later
    /// ask is one read. A generation keeps what its first ask read for as
    /// long as it stands, and a membership change publishes another.
    ///
    /// A fact these lists carry does not move within a row's life: every
    /// registration path starts another life where the image bit or a
    /// demand fact differs, which publishes. A raw store write that moves
    /// either in place publishes too: the holder's next read lays the row
    /// again, and its new object takes the standing one's place in the
    /// registry.
    ///
    /// A member the holder holds no row's object for is left out of the
    /// lists and counted ([`Self::unstated`]). So every worker lent states
    /// its row's id, and a decision's target names a worker by that id and
    /// by its place in the list lent, not among the generation's members.
    ///
    /// The first ask of a generation reads the registry's id of each
    /// member, and then the holder's entry once, under its read lock and
    /// with no other lock taken under it. The holder asks no list and
    /// calls nothing of the engine while it writes its entry. So an ask
    /// and a membership change can wait on each other for the length of
    /// the entry's lock, and never in a cycle: neither holds a lock while
    /// it takes the other's.
    #[must_use]
    pub fn lent<'a>(&self, generation: &'a Members, role: Role) -> LentList<'a> {
        generation.lent(role, |members| self.fixed_facts(members))
    }

    /// Who can dial each worker of the list [`Self::lent`] lends for
    /// `role`: the wires the worker's row states and the gateways that
    /// dial it, at the worker's place in that list. A host that decides
    /// for one gateway reads it for the places that gateway may take.
    ///
    /// The reach is cut with the generation's lists, by the same pass and
    /// the same read of the holder's rows, at the generation's first ask
    /// of either. So it holds as many workers as the list lent, in the
    /// list's order, and a decision that takes both of one generation
    /// reads the two as one pass saw them.
    ///
    /// Who dials a worker moves within a row's life. The holder lays the
    /// row again there, and its new object publishes a generation whose
    /// reach is cut anew.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "a controller's deciding thread asks it")
    )]
    pub(crate) fn reach<'a>(&self, generation: &'a Members, role: Role) -> &'a ListReach {
        generation.reach(role, |members| self.fixed_facts(members))
    }

    /// How many members were left out of the lists so far: a worker
    /// another writer registered, or one whose row's object was replaced
    /// or dropped between its generation's publish and that generation's
    /// first ask.
    pub fn unstated(&self) -> u64 {
        self.unstated.load(Relaxed)
    }

    /// What the holder's rows say of each of `members`, in their order:
    /// whether the worker takes images, the row's id, what the worker
    /// states for a request's hard demands, and who can dial the worker.
    /// `None` for a member the holder holds no row's object for, which is
    /// counted.
    fn fixed_facts(&self, members: &[Arc<dyn Worker>]) -> Box<[Option<Stated>]> {
        // The registry's ids first: the entry's lock is not held across a
        // read of the registry.
        let id_of = |member: &Arc<dyn Worker>| self.registry.get_id_by_url(member.url());
        let ids: Vec<Option<ObjectId>> = members.iter().map(id_of).collect();
        let held = self.held.read();
        let stated = members.iter().zip(ids).map(|(member, id)| {
            let laid = id.and_then(|id| held.rows.get(&id));
            let laid = laid.filter(|laid| Arc::ptr_eq(&laid.object, member));
            if laid.is_none() {
                self.unstated.fetch_add(1, Relaxed);
            }
            laid.map(|laid| Stated {
                facts: FixedFacts {
                    vision: laid.remainder.vision,
                    id: Some(laid.remainder.standing.row),
                    // The row's statement whole.
                    demands: laid.remainder.demand_facts,
                },
                reach: Reach::of(&laid.remainder),
            })
        });
        stated.collect()
    }
}

#[cfg(feature = "in-process")]
impl Objects<Members> {
    /// The pair classes of `generation`'s prefill and decode lists under
    /// the pairing mode `mode`: each worker's class by its place in the
    /// list AS LENT ([`Self::lent`]), and the table of the classes that
    /// pair. A member the lists leave out has no place and no class: a
    /// caller numbers nothing by the generation's own members.
    ///
    /// The verdicts are the store's own pairing rule and no other
    /// ([`PairClasses::derive`]), over one member for each lent worker,
    /// made of what the worker's row states: its wires, its engine kind and
    /// its pairing descriptor. A lent worker whose row's object was
    /// replaced or ended before a set is derived pairs with nobody in that
    /// set: class `0`. A set that stands is not derived again for a worker
    /// that goes later. The generation is the registry's old one by then,
    /// and its classes stand as its lists do.
    ///
    /// A generation keeps one set of classes a pairing mode, derived at
    /// the first ask under the mode. What a set is derived from does not
    /// move under a generation: a write that moves a row's wires, its
    /// engine kind or its descriptor gets the row a new object, in place
    /// or as another life, and a new object publishes another generation.
    /// The first ask derives the set behind the cut of the lists: it reads
    /// the holder's entry once, under its read lock, and runs the rule
    /// with that lock let go. Its work grows with the two lists and with
    /// the pairs of offers that differ among them. Every later ask under
    /// the mode is one read and takes no lock, and a decision under way
    /// keeps the set it read.
    ///
    /// The holder calls nothing of the engine while it writes its entry,
    /// and an ask holds no lock while it takes the entry's.
    #[must_use]
    pub fn pair_classes<'a>(&self, generation: &'a Members, mode: Pairing) -> &'a PartnerClasses {
        generation.pair_classes(mode, || self.derive_pairs(generation, mode))
    }

    /// The classes of `generation`'s two lists as lent, derived now.
    fn derive_pairs(&self, generation: &Members, mode: Pairing) -> PartnerClasses {
        /// A row as the rule reads it.
        fn member(row: &Arc<RowRemainder>) -> Member<'_> {
            let descriptor = row.pairing.as_deref();
            Member::new(
                row.standing.row,
                row.wire_modes,
                &row.engine_kind,
                descriptor,
            )
        }
        let lists = [Role::Prefill, Role::Decode].map(|role| self.lent(generation, role));
        let stated = self.remainders(lists);
        let members = [&stated[0], &stated[1]]
            .map(|side| side.iter().flatten().map(member).collect::<Vec<_>>());
        let derived = PairClasses::derive(&members[0], &members[1], mode);
        let stands = |side: usize| stated[side].iter().map(Option::is_some);
        derived.by_place(stands(0), stands(1))
    }

    /// What an encode request's decision reads beside `generation`'s
    /// encoder list and beside its prefill list: what the encode side
    /// reads of each encoder and the encode class each prefill names, by a
    /// worker's place in its list AS LENT ([`Self::lent`]). Each column
    /// holds one entry a worker of its list, whatever the workers state,
    /// so a host hands both with the lists as they are
    /// ([`EncodeLists::facts`](crate::engine::EncodeLists::facts),
    /// [`EncodeLists::prefill_classes`](crate::engine::EncodeLists::prefill_classes)).
    ///
    /// The classes are the store's own encode rule and no other
    /// ([`EncodeClasses::derive`]), over one member for each lent worker
    /// in its list's order, made of what the worker's row states: its
    /// engine kind and its end of the encode handoff. So a place of the
    /// rule is a place of the list lent. A lent worker whose row's object
    /// was replaced or ended before the columns are derived states no end
    /// there: an encoder of class `0`, which takes no item, and a prefill
    /// no encoder hands to.
    ///
    /// The rule runs over whatever is lent. Where no prefill is lent, an
    /// encoder keeps the class of its end, and the pair's own decision
    /// answers for the side that is missing. The store's pools read
    /// otherwise there: the store derives no class for a model with no
    /// prefill, and each of its encoders reads class `0`.
    ///
    /// A generation keeps one pair of columns, derived at the first ask
    /// as a set of its pair classes is ([`Self::pair_classes`]): behind
    /// the cut of the lists, with one read of the holder's entry under its
    /// read lock, and the rule run with that lock let go. What they are
    /// derived from does not move under a generation: a write that moves a
    /// row's engine kind or its encode offer gets the row a new object,
    /// and a new object publishes another generation. Every later ask is
    /// one read and takes no lock.
    #[must_use]
    pub fn encode_columns<'a>(&self, generation: &'a Members) -> &'a EncodeColumns {
        generation.encode_columns(|| self.derive_encode(generation))
    }

    /// The encode columns of `generation`'s two lists as lent, derived now.
    fn derive_encode(&self, generation: &Members) -> EncodeColumns {
        /// A lent worker as the rule reads it: its row's engine kind and
        /// offer, and for a worker no row stands for an unknown runtime
        /// with `none`, the offer that states nothing.
        fn member<'a>(
            row: Option<&'a Arc<RowRemainder>>,
            none: &'a EncodeOffer,
        ) -> EncodeMember<'a> {
            row.map_or(
                EncodeMember {
                    runtime: "",
                    offer: none,
                },
                |row| EncodeMember {
                    runtime: &row.engine_kind,
                    offer: &row.encode,
                },
            )
        }
        let lists = [Role::Encode, Role::Prefill].map(|role| self.lent(generation, role));
        let [encoders, prefills] = self.remainders(lists);
        let none = EncodeOffer::default();
        let classes = EncodeClasses::derive(
            encoders.iter().map(|row| member(row.as_ref(), &none)),
            prefills.iter().map(|row| member(row.as_ref(), &none)),
        );
        let offers = encoders.iter().map(|row| member(row.as_ref(), &none).offer);
        let prefill_classes = (0..prefills.len()).map(|at| classes.prefill_class(at));
        EncodeColumns::new(classes.encoder_facts(offers), prefill_classes.collect())
    }

    /// What the row of each worker of `lists` states, at the worker's
    /// place in its list, where the row's object still stands for the
    /// worker: one read of the holder's entry, under its read lock, which
    /// is let go before a rule runs over what was read.
    fn remainders<const N: usize>(
        &self,
        lists: [LentList<'_>; N],
    ) -> [Vec<Option<Arc<RowRemainder>>>; N] {
        let held = self.held.read();
        lists.map(|list| {
            let facts = list.facts.unwrap_or_default();
            let row = |(worker, fact): (&Arc<dyn Worker>, &FixedFacts)| {
                let laid = held.rows.get(held.objects.get(&fact.id?)?)?;
                Arc::ptr_eq(&laid.object, worker).then(|| Arc::clone(&laid.remainder))
            };
            list.workers.iter().zip(facts).map(row).collect()
        })
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
        store::{
            GatewayId, HealthSink, MemoryStore, Overload, Registrant, WorkerFacts, WorkerPatch,
            WorkerStatus,
        },
        worker_host::{object, HealthAdapter, KeptCredential, Members, SoleLead},
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

    /// The decision of a retry that demands `runtime`, made by `engine`
    /// over the list a holder lends.
    #[cfg(feature = "in-process")]
    fn retry_demanding(
        engine: &crate::engine::PolicyEngine,
        lent: LentList<'_>,
        id: &'static [u8],
        runtime: u32,
    ) -> crate::engine::Decision {
        use std::time::Instant;

        use crate::{
            document::spec::AppliedSpec,
            engine::{
                Constraints, Decided, Demands, Request, RequestFacts, RequestMaker, WorkerLists,
            },
        };

        /// A request stated ahead, with the list its host hands for it.
        struct Asks<'a>(Request<'a>, WorkerLists<'a>);

        impl RequestMaker for Asks<'_> {
            fn model(&self) -> &str {
                self.0.facts.model
            }

            fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
                self.0
            }

            fn workers(&self) -> WorkerLists<'_> {
                self.1
            }
        }

        let asked = Request {
            id,
            facts: RequestFacts {
                model: "m",
                seed: 7,
                ..RequestFacts::default()
            },
            constraints: Constraints {
                demands: Demands {
                    runtime,
                    ..Demands::default()
                },
                ..Constraints::default()
            },
        };
        let asks = Asks(asked, WorkerLists::Single(lent));
        match engine.decide_at(&asks, Instant::now).expect("an answer") {
            Decided::Whole(decided, _) => decided.decision,
            Decided::Alone(_) => panic!("a place for a request that states a demand"),
        }
    }

    /// The holder lends what each stored worker states for a request's hard
    /// demands, and a decision over its list reads it. A retry that demands
    /// a runtime lands on the worker that states that runtime, whichever
    /// request it is, and one that demands a runtime no worker states is
    /// shed under the stage `wires`. The classic router holds a retry to
    /// its first plan's runtime and transport by one test of each worker
    /// (`model_gateway/src/routers/common/placement.rs:84-93`).
    #[cfg(feature = "in-process")]
    #[test]
    fn a_stored_workers_demand_facts_are_lent_and_hold_a_retry_to_its_runtime() {
        use crate::{
            document::spec::{AppliedSpec, PolicySpec},
            engine::{Decision, DemandFacts, PolicyEngine, Role, ShedReason},
        };

        let mut replica = Replica::new();
        replica.objects = replica.objects.with_status_followed();
        // A stored worker that states `runtime`, ready by a check's word.
        let ready = |key: &str, runtime: u32| {
            let stated = WorkerFacts {
                demand_facts: DemandFacts {
                    runtime,
                    ..DemandFacts::default()
                },
                ..facts(key, 1)
            };
            let id = replica.register(stated, None);
            let row = replica.store.get(id).expect("the registered row");
            let ready = replica
                .store
                .set_status(id, WorkerStatus::Ready, row.revision);
            ready.expect("ready");
            id
        };
        let (seven, nine) = (ready(A, 7), ready(B, 9));
        replica.follow();

        let registry = replica.objects.registry();
        let generation = registry
            .model_generation("m")
            .expect("the model's generation");
        let lent = replica.objects.lent(&generation, Role::Single);
        let mut stated: Vec<(Option<WorkerId>, DemandFacts)> = lent
            .facts
            .expect("the holder lends facts")
            .iter()
            .map(|facts| (facts.id, facts.demands))
            .collect();
        stated.sort_by_key(|(_, demands)| demands.runtime);
        let states = |runtime| DemandFacts {
            runtime,
            ..DemandFacts::default()
        };
        assert_eq!(
            stated,
            [(Some(seven), states(7)), (Some(nine), states(9))],
            "each worker's statement, with its row's id"
        );

        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = 1;
        let applied = AppliedSpec::validate(&document).expect("the document validates");
        let engine = PolicyEngine::new(applied);
        // A retry that demands `runtime`, decided over the list lent.
        let decide = |id: &'static [u8], runtime: u32| retry_demanding(&engine, lent, id, runtime);
        let landed = |id: &'static [u8], runtime: u32| match decide(id, runtime) {
            Decision::Pick(pick) => pick.targets.first().map(|target| target.id),
            other => panic!("a worker states runtime {runtime}: {other:?}"),
        };
        for id in [&b"first"[..], b"second", b"third", b"fourth"] {
            assert_eq!(landed(id, 9), Some(nine), "the runtime's one worker");
            assert_eq!(landed(id, 7), Some(seven), "the runtime's one worker");
        }
        match decide(b"none", 5) {
            Decision::Shed(shed) => {
                assert_eq!(
                    (shed.reason, shed.emptied_by),
                    (ShedReason::NoEndpoints, "wires")
                );
            }
            other => panic!("no worker states runtime 5: {other:?}"),
        }
    }

    /// A registration that changes the runtime a stored worker states is
    /// read by the decisions behind it. The door starts another life for
    /// the worker, and one follow later the holder lends the new life's
    /// statement under its id: a retry that demands the new runtime lands
    /// on the new life, and one that demands the old runtime is shed under
    /// the stage `wires`, as no worker states it any more.
    #[cfg(feature = "in-process")]
    #[test]
    fn a_registration_that_changes_a_runtime_moves_the_retries_that_demand_it() {
        use crate::{
            document::spec::{AppliedSpec, PolicySpec},
            engine::{Decision, DemandFacts, PolicyEngine, Role, ShedReason},
            worker_host::{
                membership_writer::tests::StoreDoor, MembershipWriteOutcome, MembershipWriter,
                Precondition,
            },
        };

        let mut replica = Replica::new();
        replica.objects = replica.objects.with_status_followed();
        let door = StoreDoor::new(Arc::clone(&replica.store));
        // The registration of the worker at `A` as one that states `runtime`.
        let registers = |runtime: u32| {
            let stating = |_: Option<&WorkerFacts>| WorkerFacts {
                demand_facts: DemandFacts {
                    runtime,
                    ..DemandFacts::default()
                },
                ..facts(A, 1)
            };
            door.found(&Registrant::Control, Precondition::Any, A, &stating)
        };
        // What the holder lends of the model's workers as they stand: each
        // worker's id and the runtime it states.
        let lends = |replica: &Replica| {
            let registry = replica.objects.registry();
            let generation = registry
                .model_generation("m")
                .expect("the model's generation");
            let lent = replica.objects.lent(&generation, Role::Single);
            let stated: Vec<(Option<WorkerId>, u32)> = lent
                .facts
                .expect("the holder lends facts")
                .iter()
                .map(|facts| (facts.id, facts.demands.runtime))
                .collect();
            (generation, stated)
        };
        let mut document = PolicySpec::preset("random").expect("a preset of this build");
        document.generation = 1;
        let applied = AppliedSpec::validate(&document).expect("the document validates");
        let engine = PolicyEngine::new(applied);
        // Where a retry that demands `runtime` lands over the list the
        // holder lends now, or the stage it is shed under.
        let retry = |replica: &Replica, runtime: u32| {
            let (generation, _) = lends(replica);
            let lent = replica.objects.lent(&generation, Role::Single);
            match retry_demanding(&engine, lent, b"again", runtime) {
                Decision::Pick(pick) => Ok(pick.targets.first().map(|target| target.id)),
                Decision::Shed(shed) => Err((shed.reason, shed.emptied_by)),
                other @ Decision::Queue(_) => panic!("neither a pick nor a shed: {other:?}"),
            }
        };
        let unstated = Err((ShedReason::NoEndpoints, "wires"));

        let wrote = registers(7);
        let Ok(MembershipWriteOutcome::Registered(first)) = wrote else {
            panic!("the first registration is a join: {wrote:?}");
        };
        // Its checks are on: a check's word makes the first life ready.
        let row = replica.store.get(first).expect("the registered row");
        let ready = replica
            .store
            .set_status(first, WorkerStatus::Ready, row.revision);
        ready.expect("ready");
        replica.follow();
        assert_eq!(lends(&replica).1, [(Some(first), 7)]);
        assert_eq!(retry(&replica, 7), Ok(Some(first)));
        assert_eq!(retry(&replica, 9), unstated);

        // The same worker registers again and states another runtime: the
        // store hands the first life's status on to the next.
        let wrote = registers(9);
        let Ok(MembershipWriteOutcome::Another { old, new }) = wrote else {
            panic!("a changed runtime is another life: {wrote:?}");
        };
        assert_eq!(old, first);
        replica.follow();
        assert_eq!(lends(&replica).1, [(Some(new), 9)], "under the new id");
        assert_eq!(retry(&replica, 9), Ok(Some(new)));
        assert_eq!(retry(&replica, 7), unstated);
    }

    /// A host that is no gateway asks for a host-local row's object with
    /// its checks off, whatever the row's policy says. The object of a row
    /// any gateway dials keeps its checks, and so does the row's own once
    /// its reach turns. A holder that was not asked builds each as its
    /// policy says.
    #[test]
    fn a_host_local_rows_object_is_unchecked_where_the_host_asked() {
        let checked = |replica: &Replica, url| {
            !replica
                .at(url)
                .metadata()
                .health_config
                .disable_health_check
        };
        let local = WorkerFacts {
            reachable_from: Reachability::Gateway(GatewayId(1)),
            ..facts(A, 1)
        };
        let mut outside = Replica::new();
        let registry = Arc::clone(outside.objects.registry());
        outside.objects = Objects::new(registry).with_host_local_unchecked();
        for (replica, asked) in [(Replica::new(), false), (outside, true)] {
            replica.store.upsert(local.clone());
            replica.store.upsert(facts(B, 1));
            replica.follow();
            assert_eq!(checked(&replica, A), !asked, "the host-local row");
            assert!(checked(&replica, B), "a row any gateway dials");

            replica.store.upsert(facts(A, 1));
            replica.follow();
            assert!(checked(&replica, A), "the row once any gateway dials it");
        }
    }

    /// A host whose own load monitor judges overload asks for each object
    /// built with its row's thresholds, each over the host's default for
    /// its signal. A row whose thresholds move is laid again, and its
    /// object states the new ones. A holder that was not asked builds an
    /// object that states none, so its host's monitor writes no verdict.
    #[test]
    fn an_object_states_its_rows_thresholds_where_the_host_judges_overload() {
        let defaults = OverloadThresholds {
            waiting_requests: Some(8),
            token_usage: Some(0.8),
        };
        let none = OverloadThresholds::default();
        let stated = |replica: &Replica, url| replica.at(url).metadata().overload;
        let mut judging = Replica::new();
        let registry = Arc::clone(judging.objects.registry());
        judging.objects = Objects::new(registry).with_overload_thresholds(defaults);
        for (replica, asked) in [(Replica::new(), false), (judging, true)] {
            replica.store.upsert(facts(A, 1));
            replica.store.upsert(WorkerFacts {
                overload: Overload {
                    kv_usage_permille: Some(650),
                    waiting_requests: None,
                },
                ..facts(B, 1)
            });
            replica.follow();
            let own = OverloadThresholds {
                waiting_requests: Some(8),
                token_usage: Some(0.65),
            };
            assert_eq!(stated(&replica, A), if asked { defaults } else { none });
            assert_eq!(stated(&replica, B), if asked { own } else { none });

            // The row's thresholds move: its object is laid again.
            let moved = Overload {
                kv_usage_permille: None,
                waiting_requests: Some(3),
            };
            replica.patch(
                B,
                WorkerPatch {
                    overload: Some(moved),
                    ..WorkerPatch::default()
                },
            );
            replica.follow();
            let own = OverloadThresholds {
                waiting_requests: Some(3),
                token_usage: Some(0.8),
            };
            assert_eq!(stated(&replica, B), if asked { own } else { none });
        }
    }

    /// A row that states the external runtime takes no default: its object
    /// states the thresholds its row names and no other, a queue alone or a
    /// usage alone, and in any spelling of the runtime's name. An engine's
    /// row takes the default, one that names its runtime too. A gateway
    /// registers an engine's worker with its default thresholds (its
    /// registration workflow's `overload_defaults`) and an external worker
    /// with none (`build_external_worker`), so until that worker's first
    /// update its monitor judges it by nothing.
    #[test]
    fn an_external_rows_object_states_no_default_threshold() {
        let defaults = OverloadThresholds {
            waiting_requests: Some(8),
            token_usage: Some(0.8),
        };
        let mut replica = Replica::new();
        let registry = Arc::clone(replica.objects.registry());
        replica.objects = Objects::new(registry).with_overload_thresholds(defaults);
        let of_kind = |url, kind: &str| WorkerFacts {
            engine_kind: kind.to_string(),
            ..facts(url, 1)
        };
        replica.store.upsert(of_kind(A, "external"));
        replica.store.upsert(WorkerFacts {
            overload: Overload {
                kv_usage_permille: None,
                waiting_requests: Some(3),
            },
            ..of_kind(B, "external")
        });
        replica.store.upsert(facts(C, 1));
        replica.store.upsert(WorkerFacts {
            overload: Overload {
                kv_usage_permille: Some(500),
                waiting_requests: None,
            },
            ..of_kind(D, "external")
        });
        replica.store.upsert(of_kind(E, "vllm"));
        replica.store.upsert(of_kind(F, "EXTERNAL"));
        replica.follow();

        let stated = |url| replica.at(url).metadata().overload;
        let queue = OverloadThresholds {
            waiting_requests: Some(3),
            token_usage: None,
        };
        let usage = OverloadThresholds {
            waiting_requests: None,
            token_usage: Some(0.5),
        };
        let none = OverloadThresholds::default();
        assert_eq!(stated(A), none);
        assert_eq!(stated(B), queue, "an external row states its own alone");
        assert_eq!(stated(D), usage, "and no default queue beside its usage");
        assert_eq!(stated(F), none, "in any spelling of the name");
        assert_eq!(stated(C), defaults, "an engine's row takes the default");
        assert_eq!(stated(E), defaults, "one that names its runtime too");
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
                Some(RowRevision { row, revision })
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
        // The read `lay_again` makes finds nothing, and forgets the row
        // all the same.
        assert_eq!(replica.lay_again(&facts(A, 1)), 0);
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

    /// One registration that keeps its credential at a time: its write
    /// runs inside the holder's keeping turn, and the turn is let go
    /// behind it.
    #[test]
    fn a_keeping_write_runs_in_a_turn_of_its_own() {
        let replica = Replica::new();
        let objects = &replica.objects;
        let held = objects.keeping(A, None, || objects.keeping.is_locked());
        assert_eq!(held, Ok(true), "the write runs inside the turn");
        assert!(!objects.keeping.is_locked(), "let go behind the write");
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
        let standing = RowRevision {
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
            Some(RowRevision {
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
                Some(RowRevision { row, revision }),
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
        replica.store.probe_outcome(row, true, revision, &|| true);
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
        assert!(replica.store.restore_from(other.checkpoint()));
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
        assert!(replica.store.restore_from(other.checkpoint()));
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
        fn upsert_by(&self, facts: WorkerFacts, maker: Registrant) -> MembershipChange {
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

        fn all_rows(&self) -> (u64, Vec<WorkerRow>) {
            self.store.all_rows()
        }

        fn history(&self) -> u64 {
            self.store.history()
        }

        fn changes_since(&self, epoch: u64) -> MembershipTail {
            if let Some(before_tail) = self.before_tail.lock().take() {
                before_tail(&self.store);
            }
            if let Some((other, row)) = self.turn.lock().take() {
                assert!(self.store.restore_from(other.checkpoint()));
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
        let cut = other.checkpoint();
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

    /// An engine's ranks are answered by their rows, asked of its rank 0
    /// row. A rank no row holds is answered by none, a row above rank 0 is
    /// answered none, and a row that is its own engine has its one rank.
    #[test]
    fn an_engines_ranks_are_answered_by_their_rows() {
        let replica = Replica::new();
        let lower = replica.store.upsert(rank_facts(A, C, 0)).id();
        let upper = replica.store.upsert(rank_facts(B, C, 1)).id();
        let alone = replica.store.upsert(facts(D, 1)).id();
        replica.follow();
        let objects = &replica.objects;
        assert_eq!(objects.rank_row(lower, 0), Some(lower));
        assert_eq!(objects.rank_row(lower, 1), Some(upper));
        assert_eq!(objects.rank_row(lower, 2), None);
        for rank in 0..3 {
            assert_eq!(objects.rank_row(upper, rank), None);
        }
        assert_eq!(objects.rank_row(alone, 0), Some(alone));
        assert_eq!(objects.rank_row(alone, 1), None);
    }

    #[test]
    fn a_rank_row_that_leaves_is_answered_no_longer() {
        let replica = Replica::new();
        let lower = replica.store.upsert(rank_facts(A, C, 0)).id();
        let upper = replica.store.upsert(rank_facts(B, C, 1)).id();
        replica.follow();
        assert_eq!(replica.objects.rank_row(lower, 1), Some(upper));

        replica.remove(B);
        replica.follow();
        assert_eq!(replica.objects.rank_row(lower, 1), None);
        assert_eq!(replica.objects.rank_row(upper, 1), None);
        assert_eq!(replica.objects.rank_row(lower, 0), Some(lower));
    }

    /// A row stands at the rank its facts state and at no other, and at
    /// none once it left: what a late dispatch report is held against.
    #[test]
    fn a_row_stands_at_its_own_rank_until_it_leaves() {
        let replica = Replica::new();
        let lower = replica.store.upsert(rank_facts(A, C, 0)).id();
        let upper = replica.store.upsert(rank_facts(B, C, 1)).id();
        replica.follow();
        let objects = &replica.objects;
        assert!(objects.lists_rank(lower, 0));
        assert!(objects.lists_rank(upper, 1));
        assert!(!objects.lists_rank(lower, 1), "another row's rank");
        assert!(!objects.lists_rank(upper, 0), "another row's rank");

        replica.remove(B);
        replica.follow();
        assert!(!objects.lists_rank(upper, 1), "a row that left");
        assert!(objects.lists_rank(lower, 0));
    }

    /// A rank's key that comes back as a new incarnation is another row, and
    /// the rank is answered by it. A row whose facts move to another rank is
    /// answered under its new rank alone.
    #[test]
    fn a_rank_row_laid_again_is_answered_by_its_new_row() {
        let replica = Replica::new();
        let lower = replica.store.upsert(rank_facts(A, C, 0)).id();
        let upper = replica.store.upsert(rank_facts(B, C, 1)).id();
        replica.follow();

        let again = replica
            .store
            .upsert(WorkerFacts {
                incarnation: 2,
                ..rank_facts(B, C, 1)
            })
            .id();
        replica.follow();
        assert_ne!(again, upper);
        assert_eq!(replica.objects.rank_row(lower, 1), Some(again));

        let moved = replica
            .store
            .upsert(WorkerFacts {
                incarnation: 2,
                dp_rank: 2,
                dp_size: 3,
                ..rank_facts(B, C, 1)
            })
            .id();
        replica.follow();
        assert_eq!(moved, again);
        assert_eq!(replica.objects.rank_row(lower, 1), None);
        assert_eq!(replica.objects.rank_row(lower, 2), Some(again));
    }
}
