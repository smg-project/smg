//! The holder of a replica's worker objects: it follows the store's
//! membership and writes the registry. A host can also ask it to follow
//! the rows' status, and to write the store's overload verdicts on the
//! objects.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::{
        atomic::{
            AtomicBool, AtomicU64,
            Ordering::{Acquire, Relaxed, Release},
        },
        Arc,
    },
};

use parking_lot::{Mutex, RwLock};
use smg_worker::{
    connection::engine::EngineConnector,
    health::overload::{log_overload_edge, OverloadReading, OverloadThresholds},
    identity::{basic::Worker, endpoint::Endpoint},
    registry::{Generation, WorkerId as ObjectId, WorkerRegistry},
};
use tracing::warn;

use super::{
    credentials::Credentials,
    door::{Refused, Wrote},
    found::facts,
    object::{builder, dial_address, status, unchecked},
};
use crate::{
    store::{
        MembershipChange, MembershipTail, MemoryStore, Overload, Reachability, WorkerFacts,
        WorkerRow, WorkersStore,
    },
    view::{LoadRow, WorkerId},
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

/// A row and the object that stands for it.
struct Laid {
    standing: Standing,
    object: Arc<dyn Worker>,
    /// Whether a registration handed the object in. The holder built it
    /// from the row otherwise.
    handed: bool,
}

/// The objects on offer for one write, each for its own row. Dropping it
/// ends the offer whole: an object no row took is never left for a later
/// lay.
struct Offer<'a>(&'a Mutex<Vec<Arc<dyn Worker>>>);

impl<'a> Offer<'a> {
    fn put(slot: &'a Mutex<Vec<Arc<dyn Worker>>>, built: &[Arc<dyn Worker>]) -> Self {
        *slot.lock() = built.to_vec();
        Self(slot)
    }
}

impl Drop for Offer<'_> {
    fn drop(&mut self) {
        self.0.lock().clear();
    }
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
    /// build, at a hand-in, and where a kept object's entry moves. The
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
/// [`Self::follow`] is the one path, on a leader and on a standby: it reads
/// what the store's membership did since its last read. It builds the
/// object of a row that arrived, drops the object of a row that went, and
/// replaces the object of a row whose facts changed under the same id in
/// the registry. A new incarnation of a key gets a new object and id,
/// unless a registration hands its object in.
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
/// revision never moves under an object the holder built and kept.
///
/// A host whose registrations build their workers in its own process has
/// the holder keep those objects ([`Self::with_built_kept`]). A
/// registration hands its object in through its write ([`Self::handing`]),
/// and the object is the row's while it states the row's facts. A handed
/// object goes for the next one a writer hands in, and for nothing else: a
/// row that moves with no hand-in keeps it, and its entry alone moves to
/// the row. Such an object holds its own credential, so that holder needs
/// no keeper.
///
/// A registration whose write makes the row another life keeps the
/// worker's id in the registry: the object it hands in takes the place of
/// the old life's object. The id is the worker's handle for its operator,
/// and the row's id is the life.
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
    /// Whether the host's registrations hand in the objects they built.
    built_kept: bool,
    /// The objects on offer through one registration's write, one a row.
    /// Empty outside one.
    offered: Mutex<Vec<Arc<dyn Worker>>>,
    /// One hand-in at a time, from the offer to its end.
    handing: Mutex<()>,
    /// How many lays found no object handed in where the host hands them.
    unhanded: AtomicU64,
    /// The epoch of the store's overload verdicts the objects were last
    /// held against ([`Self::mirror_overload`]). Held through a walk: one
    /// mirrors at a time.
    mirrored: Mutex<Option<u64>>,
    /// Whether the registry took an object since the last walk began. Its
    /// replacement releases a verdict and a new object starts with none,
    /// so the next mirror walks whatever the epoch says.
    mirror_owed: AtomicBool,
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

    /// For a host that is no gateway: it holds the socket of no host-local
    /// wire, so the object of a row that one gateway alone reaches is built
    /// with its checks off, whatever the row's policy says. A health loop
    /// over the registry then dials no worker its host cannot reach.
    #[must_use]
    pub fn with_host_local_unchecked(mut self) -> Self {
        self.host_local_unchecked = true;
        self
    }

    /// For a host whose registrations build their workers in its own
    /// process: the holder keeps the object a registration hands it
    /// ([`Self::handing`]) and builds none from that row. A built object
    /// holds what no row states: its credential, its connection settings
    /// and its model cards.
    ///
    /// Such a holder expects an object with every row it lays. A lay that
    /// finds none keeps a handed object that stands for the row, and
    /// builds from the row where none does. It warns and counts either
    /// way ([`Self::unhanded`]).
    ///
    /// A holder takes kept objects or a keeper
    /// ([`Self::with_credentials`]), never both: a built object holds its
    /// own credential. The later of the two calls stands, so this one
    /// drops a keeper handed before it.
    #[must_use]
    pub fn with_built_kept(mut self) -> Self {
        self.built_kept = true;
        self.credentials = None;
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
            built_kept: false,
            offered: Mutex::new(Vec::new()),
            handing: Mutex::new(()),
            unhanded: AtomicU64::new(0),
            mirrored: Mutex::new(None),
            mirror_owed: AtomicBool::new(false),
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
    ///
    /// A holder takes a keeper or kept objects
    /// ([`Self::with_built_kept`]), never both. The later of the two calls
    /// stands, so this one ends the keeping of built objects.
    #[must_use]
    pub fn with_credentials(mut self, credentials: Arc<Credentials>) -> Self {
        self.credentials = Some(credentials);
        self.built_kept = false;
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

    /// Put `built` on offer through `write`, the write of their rows at the
    /// host's door, and return the write's answer. The offer is the
    /// registration's set: one object a row, for that one turn.
    ///
    /// The row the write leaves under an object's URL is laid with that
    /// object, where it states the row's facts ([`facts`]): it takes the
    /// place of the object that stands there, under its id and with its
    /// running state. A write that moved no fact of a row leaves no change
    /// for a follow to read, so that row is laid here. An object that
    /// stands for its row already is left as it is.
    ///
    /// The call follows `store` behind an answered write and holds the
    /// follow's turn to its end, so the registry holds the objects when it
    /// returns. The offer ends whole with the call: an object whose row the
    /// write refused or did not state is taken back, never left for a later
    /// lay. One call runs at a time. A host keeps its follows in the
    /// write's turn, as a door's listener does: a follow on another thread
    /// may lay a row with the offer of a write that is still open.
    ///
    /// `over` is the object a guarded write replaces, the one its asker
    /// read and built from. A guarded write states one row. An object can
    /// change while its row's revision stands: a write that moved no fact
    /// still laid another. So the call refuses the write as moved, ahead of
    /// it, where another object stands under that row's URL by then.
    ///
    /// A write that makes the row another life keeps the worker's id:
    /// the read lays `built` in the place of the old life's object, by the
    /// store's tail or by its snapshot. With no object on offer, it drops
    /// that object and lays the new life's under another id.
    ///
    /// A holder built without [`Self::with_built_kept`] runs `write` alone.
    pub fn handing<S: WorkersStore + ?Sized>(
        &self,
        store: &S,
        built: &[Arc<dyn Worker>],
        over: Option<&Arc<dyn Worker>>,
        write: impl FnOnce() -> Result<Vec<Wrote>, Refused>,
    ) -> Result<Vec<Wrote>, Refused> {
        if !self.built_kept {
            return write();
        }
        let _turn = self.handing.lock();
        if let Some(seen) = over {
            let stands = match built {
                [one] => self.registry.get_by_url(one.url()),
                _ => None,
            };
            if !stands.is_some_and(|stands| Arc::ptr_eq(&stands, seen)) {
                return Err(Refused::Moved);
            }
        }
        let offer = Offer::put(&self.offered, built);
        let answer = write();
        if let Ok(wrote) = &answer {
            let mut place = self.read_to.lock();
            self.follow_from(store, &mut place);
            for wrote in wrote {
                // Equal facts left no change to read. The row is laid
                // here unless the read above laid it. A row that left
                // behind the write is the next follow's.
                let Wrote::Unmoved(id) = *wrote else {
                    continue;
                };
                let waits = !self.offered.lock().is_empty();
                if let Some(row) = store.get(id).filter(|_| waits) {
                    self.lay_unmoved(store, &row);
                }
            }
        }
        drop(offer);
        answer
    }

    /// Run `write`, the write of a registration that keeps its credential,
    /// in the hand-in's turn, and return its answer. `write`
    /// is the whole of it: the keep, the door's write, the keeper's answer
    /// and the lay-again that answer asks for. One runs at a time, so a
    /// credential is settled and its object laid again before the next
    /// registration reads either. A holder takes a keeper or kept objects,
    /// never both, so the one turn serves the writes of either.
    ///
    /// `over` is the object a guarded write replaces, as in
    /// [`Self::handing`]. A registration that states another credential
    /// alone moves no row and lays another object, so here too an object
    /// changes while its row's revision stands. The call refuses the write
    /// as moved, ahead of it, where another object stands under `url` by
    /// then.
    pub(super) fn keeping<T>(
        &self,
        url: &str,
        over: Option<&Arc<dyn Worker>>,
        write: impl FnOnce() -> T,
    ) -> Result<T, Refused> {
        let _turn = self.handing.lock();
        if let Some(seen) = over {
            let stands = self.registry.get_by_url(url);
            if !stands.is_some_and(|stands| Arc::ptr_eq(&stands, seen)) {
                return Err(Refused::Moved);
            }
        }
        Ok(write())
    }

    /// [`Self::handing`] for the write of one row: a refresh's rebuilt
    /// worker, or a registration of one. The answer is that row's.
    pub fn handing_one<S: WorkersStore + ?Sized>(
        &self,
        store: &S,
        built: &Arc<dyn Worker>,
        over: Option<&Arc<dyn Worker>>,
        write: impl FnOnce() -> Result<Wrote, Refused>,
    ) -> Result<Wrote, Refused> {
        let wrote = self.handing(store, std::slice::from_ref(built), over, || {
            write().map(|wrote| vec![wrote])
        })?;
        wrote.first().copied().ok_or(Refused::Moved)
    }

    /// How many lays found no object handed in, under
    /// [`Self::with_built_kept`]: rows a writer moved or made with no
    /// hand-in, and offers that did not state the row. Each one warned.
    pub fn unhanded(&self) -> u64 {
        self.unhanded.load(Relaxed)
    }

    /// Write the overload verdicts `store` publishes on the objects. An
    /// object reads overloaded where any row of its worker is published
    /// under a verdict, and clear where none is. The holder judges
    /// nothing. The registry makes each write
    /// ([`WorkerRegistry::set_worker_overloaded`]), so its counters and its
    /// gauge move once an edge. An edge is said in the worker crate's one
    /// line ([`log_overload_edge`]), under the source `policy store`.
    ///
    /// A host calls it behind each publication it makes. It walks the
    /// objects only where the world's verdicts moved since the last walk
    /// ([`World::overload_epoch`](crate::store::World::overload_epoch)), or
    /// the registry took an object since. Return how many flags moved.
    ///
    /// Calls run one at a time, and each reads the world of its instant
    /// inside that turn. A late caller so cannot write an older world's
    /// verdicts over a newer one's, and a host needs no turn of its own
    /// around the call. What the walk needs of the world is taken inside
    /// its load, which is let go ahead of the registry's writes and the
    /// edges' lines.
    ///
    /// The flag follows the world that shows a verdict by one call: a
    /// world that a decision's own build published is mirrored at the
    /// host's next.
    ///
    /// The line states the worker's published figures and the thresholds
    /// its row names for itself. They need not explain the verdict. The
    /// store judges an engine's ranks together, and under the pool's
    /// thresholds where the row names none.
    pub fn mirror_overload(&self, store: &MemoryStore) -> usize {
        let mut mirrored = self.mirrored.lock();
        // The objects whose flag is not their rows' verdict, each with the
        // verdict and what its edge's line states.
        let stale = store.with_world(|world| {
            let owed = self.mirror_owed.swap(false, Acquire);
            let epoch = world.overload_epoch();
            if !owed && *mirrored == Some(epoch) {
                return Vec::new();
            }
            *mirrored = Some(epoch);
            let mut over: Vec<WorkerId> = world.overloaded().map(|row| row.id).collect();
            over.sort_unstable();
            let held = self.held.read();
            let stale = held.rows.values().filter_map(|laid| {
                let row = laid.standing.row;
                let verdict = over.binary_search(&row).is_ok();
                if laid.object.is_overloaded() == verdict {
                    return None;
                }
                let own = held.facts.get(&row).map(|facts| facts.overload);
                let load = world.first_row_of(row).and_then(|row| row.load);
                let figures = reading(load, own.unwrap_or_default());
                Some((Arc::clone(&laid.object), verdict, figures))
            });
            stale.collect::<Vec<_>>()
        });
        let mut edges = 0;
        for (object, verdict, figures) in stale {
            if self.registry.set_worker_overloaded(&object, verdict) {
                edges += 1;
                log_overload_edge(object.as_ref(), verdict, &figures, MIRRORED_FROM);
            }
        }
        edges
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
            // now: an incarnation a key takes again is another life. An
            // object on offer for the row under its URL now takes the old
            // one's place instead, and the worker keeps its id.
            MembershipChange::Replaced { old, new } => {
                if !self.succeed(store, old, |url| store.by_key(url)) {
                    self.end(old);
                }
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
    ///
    /// A row that went leaves its object's place to the life that took its
    /// URL, where a registration has that life's object on offer: the
    /// worker keeps its id, as on the tail's read.
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
            let under = |url: &str| rows.iter().find(|row| row.facts.key == url).cloned();
            if !self.succeed(store, id, under) {
                self.end(id);
            }
        }
        for row in rows {
            if self.stands(row) {
                self.follow_status(store, row.id);
            } else {
                self.lay(store, row);
            }
        }
    }

    /// The URL the object of the row `old` was laid under, where an object
    /// is on offer under it.
    fn offered_at(&self, old: WorkerId) -> Option<String> {
        let offered = self.offered.lock();
        let urls: Vec<String> = offered.iter().map(|built| built.url().to_owned()).collect();
        drop(offered);
        urls.into_iter().find(|url| self.stood_at(old, url))
    }

    /// Whether the object of the row `id` was laid under `url`.
    fn stood_at(&self, id: WorkerId, url: &str) -> bool {
        let held = self.held.read();
        held.facts.get(&id).is_some_and(|facts| facts.key == url)
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
        self.held.write().leave(id);
    }

    /// Lay `row`'s object: the one on offer for it, else one built from
    /// the row. A handed object that stands for the row goes for another
    /// its writer hands in, and for nothing else: the row states less than
    /// the object holds. With no offer it is kept, and its entry moves to
    /// the row's revision and facts alone.
    fn lay<S: WorkersStore + ?Sized>(&self, store: &S, row: &WorkerRow) {
        if let Some(built) = self.take_offered(row) {
            return self.lay_handed(store, row, built, None);
        }
        if self.kept(row.id).is_some() {
            self.missed(row, "the object that stands is kept");
            self.restate(row);
            self.follow_status(store, row.id);
            return;
        }
        if self.built_kept {
            self.missed(row, "its object is built from the row");
        }
        self.put(store, row, self.build(row), false, None);
    }

    /// Take the object on offer that is `row`'s: under the row's URL, and
    /// stating the row's facts over the row's own ([`facts`]). The facts a
    /// door states itself are the row's by that reading.
    fn take_offered(&self, row: &WorkerRow) -> Option<Arc<dyn Worker>> {
        let mut offered = self.offered.lock();
        let at = offered.iter().position(|built| {
            built.url() == row.facts.key && facts(built.as_ref(), Some(&row.facts)) == row.facts
        })?;
        Some(offered.swap_remove(at))
    }

    /// The object that stands for `row`, where a registration handed it in.
    fn kept(&self, row: WorkerId) -> Option<Arc<dyn Worker>> {
        let held = self.held.read();
        held.objects
            .get(&row)
            .and_then(|id| held.rows.get(id))
            .filter(|laid| laid.handed)
            .map(|laid| Arc::clone(&laid.object))
    }

    /// Move the entry of the object kept for `row` to the row as it
    /// stands: its revision and its facts. The registry is not written.
    fn restate(&self, row: &WorkerRow) {
        let held = &mut *self.held.write();
        let entry = held.objects.get(&row.id);
        let Some(laid) = entry.and_then(|id| held.rows.get_mut(id)) else {
            return;
        };
        laid.standing.revision = row.revision;
        held.state(row);
    }

    /// A lay that found no object handed in for `row`: warn and count.
    fn missed(&self, row: &WorkerRow, did: &'static str) {
        self.unhanded.fetch_add(1, Relaxed);
        warn!(
            worker_url = %Endpoint::redacted_str(&row.facts.key),
            "No built object that states the worker's row was handed in: {did}"
        );
    }

    /// Lay the object a registration handed in for `row`. It takes the
    /// row's status ahead of the registry's write and again behind it: a
    /// replacement takes the standing object's running state, its status
    /// with it. A row's worker neither leads its row nor trails it.
    /// `succeeds` is the life `row` took the place of: the object takes the
    /// place of that life's object.
    fn lay_handed<S: WorkersStore + ?Sized>(
        &self,
        store: &S,
        row: &WorkerRow,
        built: Arc<dyn Worker>,
        succeeds: Option<WorkerId>,
    ) {
        // A write's second attempt hands in the object that stands: a
        // replacement by itself would end its own background tasks.
        let stands = self
            .kept(row.id)
            .is_some_and(|standing| Arc::ptr_eq(&standing, &built));
        if stands {
            self.restate(row);
            self.copy_status(store, row.id);
            return;
        }
        let at = status(row.status);
        if built.status() != at {
            built.set_status(at);
        }
        self.put(store, row, built, true, succeeds);
    }

    /// Lay the life that stands under the URL of the row `old`, with the
    /// object on offer for it, in the place of the old row's object: under
    /// its id in the registry and with its running state. `standing`
    /// answers the row under a URL as the read holds it, by the store or
    /// by its snapshot. The old row leaves every table of the entry.
    ///
    /// Return whether an object was laid. With none on offer under the old
    /// row's URL that states the row there, the read drops the old row's
    /// object, and a new life is laid under a new id.
    fn succeed<S: WorkersStore + ?Sized>(
        &self,
        store: &S,
        old: WorkerId,
        standing: impl FnOnce(&str) -> Option<WorkerRow>,
    ) -> bool {
        let Some(url) = self.offered_at(old) else {
            return false;
        };
        let offered = standing(&url).and_then(|row| Some((self.take_offered(&row)?, row)));
        let Some((built, row)) = offered else {
            return false;
        };
        self.lay_handed(store, &row, built, Some(old));
        true
    }

    /// Lay the row a write left unmoved with the object still on offer.
    /// An offer that does not state the row is left out: the object that
    /// stands states it.
    fn lay_unmoved<S: WorkersStore + ?Sized>(&self, store: &S, row: &WorkerRow) {
        match self.take_offered(row) {
            Some(built) => self.lay_handed(store, row, built, None),
            None => self.missed(row, "the object that stands is kept"),
        }
    }

    /// Build `row`'s object from its facts, with what the host handed the
    /// holder for it.
    fn build(&self, row: &WorkerRow) -> Arc<dyn Worker> {
        // A row one gateway alone reaches is built unchecked where the
        // host asked for it: nothing here can dial its worker.
        let reached = !self.host_local_unchecked || row.facts.reachable_from == Reachability::Any;
        let worker = if reached {
            builder(row)
        } else {
            unchecked(row)
        };
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
    /// the place of its own object, or of the object of `succeeds`, the
    /// life it took the place of. Where the registry holds nothing under
    /// the URL by then, the object is registered under a new id.
    fn put<S: WorkersStore + ?Sized>(
        &self,
        store: &S,
        row: &WorkerRow,
        object: Arc<dyn Worker>,
        handed: bool,
        succeeds: Option<WorkerId>,
    ) {
        // The id first, then the entry, then the registry's write: its
        // event finds the entry.
        let mut id = self.registry.reserve_id_for_url(&row.facts.key);
        // A store restored inside a read can show a key's new incarnation
        // before any change says the old one left.
        let others = |other: &Standing| other.row != row.id && Some(other.row) != succeeds;
        if let Some(other) = self.standing(&id).filter(others) {
            self.end(other.row);
            id = self.registry.reserve_id_for_url(&row.facts.key);
        }
        {
            let mut entries = self.held.write();
            // The old life leaves the entry in the turn the new one enters
            // it: a reader finds the object under one of the two rows.
            if let Some(old) = succeeds {
                entries.leave(old);
            }
            entries.objects.insert(row.id, id.clone());
            entries.state(row);
            entries.rows.insert(
                id,
                Laid {
                    standing: Standing {
                        row: row.id,
                        revision: row.revision,
                    },
                    object: Arc::clone(&object),
                    handed,
                },
            );
        }
        self.registry.register_or_replace(object);
        // Behind the registry's write, and released to the mirror that
        // takes it: a mirror that reads the mark finds the object in its
        // place.
        self.mirror_owed.store(true, Release);
        if handed {
            self.copy_status(store, row.id);
        } else {
            self.follow_status(store, row.id);
        }
    }
}

/// Who judged a verdict the holder mirrors, as the edge's line says it.
const MIRRORED_FROM: &str = "policy store";

/// What the line of a mirrored edge states: the figures the worker's
/// published row carries, zero where it has none, and the thresholds its
/// row names for itself.
fn reading(load: Option<LoadRow>, own: Overload) -> OverloadReading {
    OverloadReading {
        waiting_requests: load.map_or(0, |load| i64::from(load.queued)),
        kv_pressure: load.map_or(0.0, |load| f64::from(load.active_kv_permille) / 1_000.0),
        thresholds: OverloadThresholds {
            waiting_requests: own
                .waiting_requests
                .map(|count| usize::try_from(count).unwrap_or(usize::MAX)),
            token_usage: own
                .kv_usage_permille
                .map(|permille| f64::from(permille) / 1_000.0),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        collections::BTreeMap,
        sync::{mpsc, Barrier},
        thread,
        time::{Duration, Instant},
    };

    use smg_worker::{
        health::{
            circuit_breaker::{CircuitBreakerConfig, CircuitState},
            port::HealthSink as _,
        },
        identity::builder::BasicWorkerBuilder,
        model_card::ModelCard,
        registry::event::WorkerEvent,
        runtime::StallReason,
        worker::{HealthCheckConfig, WorkerStatus as ObjectStatus, WorkerType},
    };
    use tokio::sync::broadcast::Receiver;

    use super::*;
    use crate::{
        spec::{BindingSpec, Match, OverloadSpec},
        store::{
            GatewayId, HealthSink, Kinds, LoadSink as _, LoadSource, Maker, MemoryStore,
            WorkerFacts, WorkerPatch, WorkerStatus,
        },
        worker_host::{
            facts as stated, object, Expect, HealthAdapter, KeptCredential, Membership, Over,
            SoleLead, StoreDoor,
        },
        AppliedSpec, PolicySpec,
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
        /// One object handed in through `write`, the write of its one row.
        fn handing(
            &self,
            built: &Arc<dyn Worker>,
            over: Option<&Arc<dyn Worker>>,
            write: impl FnOnce() -> Result<Wrote, Refused>,
        ) -> Result<Wrote, Refused> {
            let store = self.store.as_ref();
            self.objects.handing_one(store, built, over, write)
        }

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

    /// A replica whose registrations hand in the workers they built, and
    /// the door they write at. The door has no listener: the hand-in
    /// follows the store itself.
    fn handing() -> (Replica, StoreDoor) {
        let mut replica = Replica::new();
        let registry = Arc::clone(replica.objects.registry());
        replica.objects = Objects::new(registry).with_built_kept();
        let door = StoreDoor::new(Arc::clone(&replica.store));
        (replica, door)
    }

    /// A worker a registration built at `url`. Its credential and its
    /// card's window are what no row states.
    fn built(url: &str, credential: &str) -> Arc<dyn Worker> {
        let card = ModelCard::new("m").with_context_length(4096);
        let worker = <BasicWorkerBuilder>::new(url).model(card);
        Arc::new(worker.api_key(credential).build())
    }

    /// `built` with one label more, which a row does state.
    fn built_gold(url: &str, credential: &str) -> Arc<dyn Worker> {
        let worker = <BasicWorkerBuilder>::new(url).model(ModelCard::new("m"));
        Arc::new(worker.api_key(credential).label("tier", "gold").build())
    }

    /// One registration of `built`, handed to the holder through its write.
    fn registered(
        replica: &Replica,
        door: &StoreDoor,
        built: &Arc<dyn Worker>,
        expect: Expect,
    ) -> Result<Wrote, Refused> {
        replica.handing(built, None, || {
            door.found_built(&Maker::Control, expect, built, None)
        })
    }

    /// One refresh of `rebuilt`, a worker rebuilt from `over`, handed to
    /// the holder through its write.
    fn refreshed(
        replica: &Replica,
        door: &StoreDoor,
        seen: Standing,
        rebuilt: &Arc<dyn Worker>,
        over: &Arc<dyn Worker>,
    ) -> Result<Wrote, Refused> {
        replica.handing(rebuilt, Some(over), || door.refreshed(seen, rebuilt, over))
    }

    /// A refresh lays the object it rebuilt whatever its write moved: equal
    /// labels move no row and write no revision, and a label moves the row
    /// in place. Each object takes the standing one's place, id and running
    /// state, and no lay finds its hand empty. A refresh rebuilt from the
    /// object that one replaced is an older rebuild: it is refused as
    /// moved and lays nothing, though the row's revision may still stand.
    #[test]
    fn a_refresh_lays_the_object_it_rebuilt_whether_or_not_a_label_moved() {
        type Build = fn(&str, &str) -> Arc<dyn Worker>;
        let rows: [(&str, Build, bool); 2] = [
            ("equal labels", built, false),
            ("a label", built_gold, true),
        ];
        for (name, build, moved) in rows {
            let (mut replica, door) = handing();
            let first = built(A, KEY_A);
            registered(&replica, &door, &first, Expect::Anything).expect("the first write");
            first.increment_load();
            let id = replica.id(A);
            let seen = replica.objects.standing(&id).expect("the object's row");
            replica.told();

            let rebuilt = build(A, KEY_A);
            let wrote = refreshed(&replica, &door, seen, &rebuilt, &first);
            let row = replica.row(A);
            let expected = if moved {
                Wrote::InPlace(row.id)
            } else {
                Wrote::Unmoved(row.id)
            };
            assert_eq!(wrote, Ok(expected), "{name}");
            assert_eq!(row.revision != seen.revision, moved, "{name}: the revision");
            assert!(Arc::ptr_eq(&replica.at(A), &rebuilt), "{name}");
            assert_eq!(replica.at(A).load(), 1, "{name}: the running state");
            assert!(entry_stands(&replica, A), "{name}");
            let seen = replica.objects.standing(&id).expect("the object's row");
            assert_eq!(replica.told(), [("replaced", id)], "{name}");

            let late = refreshed(&replica, &door, seen, &build(A, KEY_A), &first);
            assert_eq!(late, Err(Refused::Moved), "{name}");
            assert!(Arc::ptr_eq(&replica.at(A), &rebuilt), "{name}");
            assert_eq!(replica.told(), [], "{name}");
            assert_eq!(replica.objects.unhanded(), 0, "{name}");
        }
    }

    /// The entry of the object under `url` names the row as it stands.
    fn entry_stands(replica: &Replica, url: &str) -> bool {
        let row = replica.row(url);
        replica.objects.standing(&replica.id(url))
            == Some(Standing {
                row: row.id,
                revision: row.revision,
            })
    }

    #[test]
    fn a_handed_object_is_the_rows_object_and_holds_what_no_row_states() {
        let (mut replica, door) = handing();
        let first = built(A, KEY_A);
        let wrote = registered(&replica, &door, &first, Expect::Anything);
        assert!(matches!(wrote, Ok(Wrote::Registered(_))), "{wrote:?}");
        assert!(Arc::ptr_eq(&replica.at(A), &first));
        assert!(replica.holds_credential(A, Some(KEY_A)));
        assert_eq!(replica.at(A).models()[0].context_length, Some(4096));
        assert!(entry_stands(&replica, A));
        assert_eq!(replica.told(), [("registered", replica.id(A))]);
        assert_eq!(replica.objects.unhanded(), 0);
    }

    /// A registration lays the object it built whatever its write moved:
    /// equal facts move no row, and a label moves it in place. Each object
    /// takes the standing one's place, id and running state.
    #[test]
    fn a_registration_lays_the_object_it_built_whether_or_not_its_write_moved_the_row() {
        type Build = fn(&str, &str) -> Arc<dyn Worker>;
        type Row = (&'static str, Build, fn(&Wrote) -> bool);
        let rows: [Row; 2] = [
            ("equal facts", built, |wrote| {
                matches!(wrote, Wrote::Unmoved(_))
            }),
            ("a label", built_gold, |wrote| {
                matches!(wrote, Wrote::InPlace(_))
            }),
        ];
        for (name, build, expected) in rows {
            let (mut replica, door) = handing();
            let first = built(A, KEY_A);
            registered(&replica, &door, &first, Expect::Anything).expect("the first write");
            first.increment_load();
            let id = replica.id(A);
            replica.told();

            let second = build(A, KEY_B);
            let wrote = registered(&replica, &door, &second, Expect::Anything);
            assert!(wrote.as_ref().is_ok_and(expected), "{name}: {wrote:?}");
            assert!(Arc::ptr_eq(&replica.at(A), &second), "{name}");
            assert!(replica.holds_credential(A, Some(KEY_B)), "{name}");
            assert_eq!(replica.at(A).load(), 1, "{name}: the running state");
            assert!(entry_stands(&replica, A), "{name}");
            assert_eq!(replica.told(), [("replaced", id)], "{name}");
            assert_eq!(replica.objects.unhanded(), 0, "{name}");
        }
    }

    /// A row moved by a writer that hands nothing in keeps its handed
    /// object. Its entry moves to the row, the registry is not written,
    /// and the holder counts the lay. An offer a refused write left is
    /// gone by then, though it states the row.
    #[test]
    fn a_row_moved_with_no_hand_in_keeps_its_handed_object_and_moves_its_entry() {
        let (mut replica, door) = handing();
        let first = built(A, KEY_A);
        registered(&replica, &door, &first, Expect::Anything).expect("the first write");
        replica.told();
        let late = built(A, KEY_B);
        let refused = registered(&replica, &door, &late, Expect::Nothing);
        assert_eq!(refused, Err(Refused::Stands));

        let facts = replica.row(A).facts;
        let change = replica.store.upsert(facts);
        assert!(matches!(change, MembershipChange::Relabelled { .. }));
        replica.follow();
        assert!(Arc::ptr_eq(&replica.at(A), &first));
        assert!(entry_stands(&replica, A));
        assert!(replica.told().is_empty());
        assert_eq!(replica.objects.unhanded(), 1);

        replica.patch(A, labels("gold"));
        replica.follow();
        assert!(Arc::ptr_eq(&replica.at(A), &first));
        assert!(entry_stands(&replica, A));
        assert_eq!(replica.objects.unhanded(), 2);
    }

    /// An object is taken only where it states the row its write left. One
    /// whose write stated another label, and one at another URL, are left
    /// out: the row's object is built from the row, and the lay is counted.
    #[test]
    fn an_object_that_does_not_state_the_row_is_left_out() {
        let (replica, door) = handing();
        let stray = built(A, KEY_A);
        let wrote = replica.handing(&stray, None, || {
            door.found(&Maker::Control, Expect::Anything, A, &|standing| {
                labelled_over(stated(stray.as_ref(), standing))
            })
        });
        assert!(matches!(wrote, Ok(Wrote::Registered(_))), "{wrote:?}");
        assert!(!Arc::ptr_eq(&replica.at(A), &stray));
        assert!(replica.holds_credential(A, None));
        assert_eq!(replica.tier(A).as_deref(), Some("gold"));
        assert_eq!(replica.objects.unhanded(), 1);

        let elsewhere = built(B, KEY_B);
        let wrote = replica.handing(&elsewhere, None, || {
            door.found_built(&Maker::Control, Expect::Anything, &built(C, KEY_C), None)
        });
        assert!(matches!(wrote, Ok(Wrote::Registered(_))), "{wrote:?}");
        assert!(replica.holds_credential(C, None));
        assert!(replica.objects.registry().get_by_url(B).is_none());
        assert_eq!(replica.objects.unhanded(), 2);
    }

    /// A registration's set on offer lays each row of the write with that
    /// row's own object. An object on offer that no row of the write took
    /// is taken back with the turn, not left for a later lay.
    #[test]
    fn a_set_on_offer_lays_each_row_with_its_own_object_and_ends_whole() {
        let (replica, door) = handing();
        let rank = |at: usize| -> Arc<dyn Worker> {
            let worker = <BasicWorkerBuilder>::new(A).model(ModelCard::new("m"));
            Arc::new(worker.dp_config(at, 2).api_key(KEY_A).build())
        };
        let ranks = [rank(0), rank(1)];
        let stray = built(B, KEY_B);
        let offered = [
            Arc::clone(&ranks[1]),
            Arc::clone(&stray),
            Arc::clone(&ranks[0]),
        ];
        let store = replica.store.as_ref();
        let wrote = replica.objects.handing(store, &offered, None, || {
            door.found_rows(&Maker::Control, Expect::Nothing, &ranks, None)
        });
        assert!(
            matches!(
                wrote.as_deref(),
                Ok([Wrote::Registered(_), Wrote::Registered(_)])
            ),
            "{wrote:?}"
        );
        for rank in &ranks {
            assert!(Arc::ptr_eq(&replica.at(rank.url()), rank));
        }
        assert_eq!(replica.objects.unhanded(), 0);

        // The stray object's offer ended with that turn: its row, written
        // later with no hand-in, gets an object built from the row.
        door.found_built(&Maker::Control, Expect::Nothing, &stray, None)
            .expect("the stray worker's row");
        replica.follow();
        assert!(!Arc::ptr_eq(&replica.at(B), &stray));
        assert_eq!(replica.objects.unhanded(), 1);
    }

    fn labelled_over(facts: WorkerFacts) -> WorkerFacts {
        WorkerFacts {
            labels: BTreeMap::from([("tier".to_string(), "gold".to_string())]),
            ..facts
        }
    }

    /// A holder that expects a hand-in builds from a row laid with none
    /// and counts it. A holder that expects none runs the write alone:
    /// it builds from the row and counts nothing.
    #[test]
    fn a_row_laid_with_no_hand_in_is_built_from_the_row_and_counted_where_one_was_expected() {
        let (replica, _) = handing();
        replica.store.upsert(facts(A, 1));
        replica.follow();
        assert!(replica.holds_credential(A, None));
        assert_eq!(replica.objects.unhanded(), 1);

        let plain = Replica::new();
        let door = StoreDoor::new(Arc::clone(&plain.store));
        let first = built(A, KEY_A);
        let wrote = registered(&plain, &door, &first, Expect::Anything);
        assert!(matches!(wrote, Ok(Wrote::Registered(_))), "{wrote:?}");
        assert!(plain.objects.registry().get_by_url(A).is_none());
        plain.follow();
        assert!(!Arc::ptr_eq(&plain.at(A), &first));
        assert_eq!(plain.objects.unhanded(), 0);
    }

    /// A write's second attempt hands in the object that stands. The
    /// registry is not written: a replacement by itself would end the
    /// object's own background tasks.
    #[test]
    fn a_second_attempt_that_hands_in_the_standing_object_writes_no_registry() {
        let (mut replica, door) = handing();
        let first = built(A, KEY_A);
        registered(&replica, &door, &first, Expect::Anything).expect("the first write");
        let revision = first.revision();
        replica.told();
        let wrote = registered(&replica, &door, &first, Expect::Anything);
        assert!(matches!(wrote, Ok(Wrote::Unmoved(_))), "{wrote:?}");
        assert!(Arc::ptr_eq(&replica.at(A), &first));
        assert_eq!(first.revision(), revision);
        assert!(replica.told().is_empty());
        assert_eq!(replica.objects.unhanded(), 0);
    }

    /// A guarded write names the object it replaces. Once another object
    /// stands under the URL, the write is refused as moved ahead of the
    /// door, though the row's revision never moved. The object that stands
    /// stays, and a write that names it passes.
    #[test]
    fn a_guarded_write_is_refused_once_another_object_stands_under_its_url() {
        let (mut replica, door) = handing();
        let first = built(A, KEY_A);
        registered(&replica, &door, &first, Expect::Anything).expect("the first write");
        let second = built(A, KEY_B);
        let wrote = registered(&replica, &door, &second, Expect::Anything);
        assert!(matches!(wrote, Ok(Wrote::Unmoved(_))), "{wrote:?}");
        replica.told();

        let guarded = |built: &Arc<dyn Worker>, over: &Arc<dyn Worker>| {
            replica.handing(built, Some(over), || {
                door.found_built(
                    &Maker::Control,
                    Expect::Anything,
                    built,
                    Some(Over::Updated(over)),
                )
            })
        };
        let stale = built(A, KEY_C);
        assert_eq!(guarded(&stale, &first), Err(Refused::Moved));
        assert!(Arc::ptr_eq(&replica.at(A), &second));
        let fresh = built(A, KEY_C);
        let wrote = guarded(&fresh, &second);
        assert!(matches!(wrote, Ok(Wrote::Unmoved(_))), "{wrote:?}");
        assert!(Arc::ptr_eq(&replica.at(A), &fresh));
        assert_eq!(replica.told(), [("replaced", replica.id(A))]);
    }

    /// A registration whose write makes the row another life keeps the
    /// worker's id: the built object takes the old life's place in the
    /// registry, which tells a replacement and no removal. Nothing of the
    /// old row is left in the holder, the one lay took the object handed
    /// in, and the object reads its new row's status: Pending where the
    /// row is probed, Ready where its checks are off. A guarded write and
    /// an upsert are laid alike.
    #[test]
    fn a_registration_that_starts_another_life_keeps_its_workers_id() {
        let (mut replica, door) = handing();
        let first = built(A, KEY_A);
        registered(&replica, &door, &first, Expect::Anything).expect("the first write");
        let id = replica.id(A);
        replica.told();

        // Another model is another life. A guarded write names its object.
        let old = replica.row(A);
        let renamed = <BasicWorkerBuilder>::new(A).model(ModelCard::new("n"));
        let second: Arc<dyn Worker> = Arc::new(renamed.api_key(KEY_B).build());
        let seen = replica.objects.standing_of(&id, &first).expect("its row");
        let wrote =
            replica
                .objects
                .handing_one(replica.store.as_ref(), &second, Some(&first), || {
                    door.found_built(
                        &Maker::Control,
                        Expect::Seen(seen),
                        &second,
                        Some(Over::Replaced(&first)),
                    )
                });
        let new = replica.row(A);
        let another = Wrote::Another {
            old: old.id,
            new: new.id,
        };
        assert_eq!(wrote, Ok(another));
        assert!(replica.store.get(old.id).is_none());
        assert_eq!(replica.id(A), id);
        assert!(Arc::ptr_eq(&replica.at(A), &second));
        assert!(entry_stands(&replica, A));
        assert_eq!(replica.objects.object(old.id), None);
        assert_eq!(replica.objects.len(), 1);
        assert_eq!(new.status, WorkerStatus::Pending);
        assert_eq!(second.status(), ObjectStatus::Pending);
        assert_eq!(replica.told(), [("replaced", id.clone())]);

        // Another role is another life, here by an upsert with checks off.
        let old = new;
        let unchecked = HealthCheckConfig {
            disable_health_check: true,
            ..HealthCheckConfig::default()
        };
        let prefill = <BasicWorkerBuilder>::new(A)
            .model(ModelCard::new("n"))
            .worker_type(WorkerType::Prefill)
            .health_config(unchecked);
        let third: Arc<dyn Worker> = Arc::new(prefill.build());
        let wrote = registered(&replica, &door, &third, Expect::Anything);
        let new = replica.row(A);
        let another = Wrote::Another {
            old: old.id,
            new: new.id,
        };
        assert_eq!(wrote, Ok(another));
        assert_eq!(replica.id(A), id);
        assert!(Arc::ptr_eq(&replica.at(A), &third));
        assert!(entry_stands(&replica, A));
        assert_eq!(replica.objects.object(old.id), None);
        assert_eq!(replica.objects.len(), 1);
        assert_eq!(new.status, WorkerStatus::Ready);
        assert_eq!(third.status(), ObjectStatus::Ready);
        let told = replica.told();
        assert!(told.contains(&("replaced", id.clone())), "{told:?}");
        assert!(
            told.iter()
                .all(|(what, _)| *what != "removed" && *what != "registered"),
            "{told:?}"
        );
        assert_eq!(replica.objects.unhanded(), 0);
    }

    /// The way out of that replacement: where the registry holds nothing
    /// under the URL by then, the new life's object is registered anew and
    /// the entry holds the id it got. Nothing of the old row is left.
    #[test]
    fn a_new_life_whose_old_object_left_the_registry_is_registered_anew() {
        let (mut replica, door) = handing();
        let first = built(A, KEY_A);
        registered(&replica, &door, &first, Expect::Anything).expect("the first write");
        let id = replica.id(A);
        let old = replica.row(A);
        replica.objects.registry().remove(&id);
        replica.told();

        let renamed = <BasicWorkerBuilder>::new(A).model(ModelCard::new("n"));
        let second: Arc<dyn Worker> = Arc::new(renamed.build());
        let wrote = registered(&replica, &door, &second, Expect::Anything);
        assert!(matches!(wrote, Ok(Wrote::Another { .. })), "{wrote:?}");
        let anew = replica.id(A);
        assert_ne!(anew, id);
        assert!(Arc::ptr_eq(&replica.at(A), &second));
        assert!(entry_stands(&replica, A));
        assert_eq!(replica.objects.standing(&id), None);
        assert_eq!(replica.objects.object(old.id), None);
        assert_eq!(replica.objects.len(), 1);
        assert_eq!(replica.objects.unhanded(), 0);
        assert_eq!(replica.told(), [("registered", anew)]);
    }

    /// The same replacement where the URL took two lives inside one read:
    /// the row under the URL at the read takes the old object's place, and
    /// the life between the two is laid nowhere.
    #[test]
    fn a_new_life_keeps_its_workers_id_through_two_lives_in_one_read() {
        let (mut replica, door) = handing();
        let first = built(A, KEY_A);
        registered(&replica, &door, &first, Expect::Anything).expect("the first write");
        let id = replica.id(A);
        let old = replica.row(A);
        replica.told();

        // A life between, written with no read behind it.
        let between = WorkerFacts {
            models: vec!["x".to_string()],
            incarnation: old.facts.incarnation + 1_000,
            ..old.facts.clone()
        };
        let between = replica.store.upsert(between).id();
        assert_ne!(between, old.id);

        let renamed = <BasicWorkerBuilder>::new(A).model(ModelCard::new("n"));
        let second: Arc<dyn Worker> = Arc::new(renamed.api_key(KEY_B).build());
        let wrote =
            replica
                .objects
                .handing_one(replica.store.as_ref(), &second, Some(&first), || {
                    door.found_built(
                        &Maker::Control,
                        Expect::Anything,
                        &second,
                        Some(Over::Replaced(&first)),
                    )
                });
        let new = replica.row(A);
        let another = Wrote::Another {
            old: between,
            new: new.id,
        };
        assert_eq!(wrote, Ok(another));
        assert_eq!(replica.id(A), id);
        assert!(Arc::ptr_eq(&replica.at(A), &second));
        assert!(entry_stands(&replica, A));
        assert_eq!(replica.objects.object(old.id), None);
        assert_eq!(replica.objects.object(between), None);
        assert_eq!(replica.objects.len(), 1);
        assert_eq!(replica.objects.unhanded(), 0);
        assert_eq!(replica.told(), [("replaced", id.clone())]);
    }

    /// The same replacement through a read that takes the store's
    /// snapshot. The tail no longer reaches, the row that went leaves its
    /// object's place to the life on offer under its URL, and the worker
    /// keeps its id. A row beside it keeps its object.
    #[test]
    fn a_new_life_keeps_its_workers_id_through_a_snapshot_read() {
        let (mut replica, door) = handing();
        let first = built(A, KEY_A);
        registered(&replica, &door, &first, Expect::Anything).expect("the first write");
        let beside = built(B, KEY_B);
        registered(&replica, &door, &beside, Expect::Anything).expect("the second write");
        let (id, other) = (replica.id(A), replica.id(B));
        let old = replica.row(A);
        let read_to = replica.store.epoch();
        replica.told();

        // More status writes than the tail keeps, and no read between.
        let flipped = replica.row(B);
        for turn in 0..1_100 {
            let status = [WorkerStatus::Ready, WorkerStatus::NotReady][turn % 2];
            assert!(replica
                .store
                .set_status(flipped.id, status, flipped.revision)
                .is_some());
        }
        assert_eq!(
            replica.store.changes_since(read_to),
            MembershipTail::SnapshotRequired
        );

        let renamed = <BasicWorkerBuilder>::new(A).model(ModelCard::new("n"));
        let second: Arc<dyn Worker> = Arc::new(renamed.api_key(KEY_C).build());
        let wrote =
            replica
                .objects
                .handing_one(replica.store.as_ref(), &second, Some(&first), || {
                    door.found_built(
                        &Maker::Control,
                        Expect::Anything,
                        &second,
                        Some(Over::Replaced(&first)),
                    )
                });
        assert!(matches!(wrote, Ok(Wrote::Another { .. })), "{wrote:?}");
        assert_eq!(replica.id(A), id);
        assert!(Arc::ptr_eq(&replica.at(A), &second));
        assert!(entry_stands(&replica, A));
        assert_eq!(replica.objects.object(old.id), None);
        assert_eq!(replica.id(B), other);
        assert!(Arc::ptr_eq(&replica.at(B), &beside));
        assert_eq!(replica.objects.len(), 2);
        assert_eq!(replica.objects.unhanded(), 0);
        let told = replica.told();
        assert!(told.contains(&("replaced", id.clone())), "{told:?}");
        assert!(
            told.iter()
                .all(|(what, _)| *what != "removed" && *what != "registered"),
            "{told:?}"
        );
    }

    /// A replacement takes the standing object's status, so a handed
    /// object is set to its row's status behind the registry's write. Two
    /// rows: a second hand-in that turns the checks off ends Ready with
    /// its row, and a hand-in over a Ready worker that had no row ends
    /// Pending with its row.
    #[test]
    fn a_handed_object_reads_its_rows_status_behind_a_replacement() {
        let (replica, door) = handing();
        registered(&replica, &door, &built(A, KEY_A), Expect::Anything).expect("the first write");
        assert_eq!(replica.at(A).status(), ObjectStatus::Pending);
        let unchecked = HealthCheckConfig {
            disable_health_check: true,
            ..HealthCheckConfig::default()
        };
        let worker = <BasicWorkerBuilder>::new(A).model(ModelCard::new("m"));
        let second: Arc<dyn Worker> = Arc::new(worker.health_config(unchecked).build());
        let wrote = registered(&replica, &door, &second, Expect::Anything);
        assert!(matches!(wrote, Ok(Wrote::InPlace(_))), "{wrote:?}");
        assert_eq!(replica.row(A).status, WorkerStatus::Ready);
        assert!(Arc::ptr_eq(&replica.at(A), &second));
        assert_eq!(second.status(), ObjectStatus::Ready);

        let stray = <BasicWorkerBuilder>::new(B).status(ObjectStatus::Ready);
        let registry = replica.objects.registry();
        assert!(registry.register(Arc::new(stray.build())).is_some());
        let handed = built(B, KEY_B);
        registered(&replica, &door, &handed, Expect::Anything).expect("the write");
        assert_eq!(replica.row(B).status, WorkerStatus::Pending);
        assert!(Arc::ptr_eq(&replica.at(B), &handed));
        assert_eq!(handed.status(), ObjectStatus::Pending);
    }

    /// One hand-in at a time. A second one waits at the turn while the
    /// first one's write is open, and runs behind its lay.
    #[test]
    fn a_second_hand_in_waits_for_the_first_ones_turn() {
        let (replica, door) = handing();
        let (first, second) = (built(A, KEY_A), built(B, KEY_B));
        let (entered, seen) = mpsc::channel();
        thread::scope(|scope| {
            let wrote = replica.handing(&first, None, || {
                scope.spawn(|| {
                    replica.handing(&second, None, || {
                        entered.send(()).expect("the case listens");
                        door.found_built(&Maker::Control, Expect::Anything, &second, None)
                    })
                });
                assert!(seen.recv_timeout(Duration::from_millis(200)).is_err());
                door.found_built(&Maker::Control, Expect::Anything, &first, None)
            });
            assert!(matches!(wrote, Ok(Wrote::Registered(_))), "{wrote:?}");
        });
        assert!(seen.try_recv().is_ok());
        assert!(Arc::ptr_eq(&replica.at(A), &first));
        assert!(Arc::ptr_eq(&replica.at(B), &second));
        assert_eq!(replica.objects.unhanded(), 0);
    }

    /// A handed object starts at its row's status, as one built from the
    /// row does: here a row that stood Ready before the holder's first read.
    /// The registry is handed the object at that status, so it tells one
    /// registration and no status change behind it.
    #[test]
    fn a_handed_object_starts_at_its_rows_status() {
        let (mut replica, door) = handing();
        let first = built(A, KEY_A);
        door.found_built(&Maker::Control, Expect::Anything, &first, None)
            .expect("a write the holder has not read");
        let row = replica.row(A);
        replica
            .store
            .set_status(row.id, WorkerStatus::Ready, row.revision);
        assert_eq!(replica.row(A).status, WorkerStatus::Ready);

        let second = built(A, KEY_B);
        assert_eq!(second.status(), ObjectStatus::Pending);
        let wrote = registered(&replica, &door, &second, Expect::Anything);
        assert!(matches!(wrote, Ok(Wrote::Unmoved(_))), "{wrote:?}");
        assert!(Arc::ptr_eq(&replica.at(A), &second));
        assert_eq!(second.status(), ObjectStatus::Ready);
        let registry = replica.objects.registry();
        let id = registry.get_id_by_url(A).expect("the object's id");
        assert_eq!(replica.told(), [("registered", id)]);
        assert_eq!(replica.objects.unhanded(), 0);
    }

    /// Two registrations of one URL at once: each hands its own object
    /// in, one turn after the other. The later one stands, and no lay
    /// missed its object.
    #[test]
    fn two_registrations_of_one_url_at_once_each_hand_their_object_in() {
        for _ in 0..64 {
            let (replica, door) = handing();
            let pair = [built(A, KEY_A), built(A, KEY_B)];
            let start = Barrier::new(2);
            thread::scope(|scope| {
                for built in &pair {
                    scope.spawn(|| {
                        start.wait();
                        registered(&replica, &door, built, Expect::Anything).expect("a write");
                    });
                }
            });
            let stands = replica.at(A);
            assert!(pair.iter().any(|built| Arc::ptr_eq(built, &stands)));
            assert!(entry_stands(&replica, A));
            assert_eq!(replica.objects.unhanded(), 0);
        }
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

    /// A rank's row whose next life keeps its object leaves its engine's
    /// ranks with the rest of its entry: the rank is answered by its new
    /// row alone.
    #[test]
    fn a_new_life_that_keeps_its_object_takes_its_rows_rank() {
        let (replica, door) = handing();
        let ranked = |rank: usize, model: &str| -> Arc<dyn Worker> {
            let worker = <BasicWorkerBuilder>::new(A).model(ModelCard::new(model));
            Arc::new(worker.dp_config(rank, 2).build())
        };
        // A worker's ranks are one registration: each write states both.
        let written = |built: &[Arc<dyn Worker>], expect: Expect| {
            let store = replica.store.as_ref();
            replica.objects.handing(store, built, None, || {
                door.found_rows(&Maker::Control, expect, built, None)
            })
        };
        let (lower, first) = (ranked(0, "m"), ranked(1, "m"));
        let wrote = written(&[Arc::clone(&lower), Arc::clone(&first)], Expect::Nothing);
        assert!(
            matches!(
                wrote.as_deref(),
                Ok([Wrote::Registered(_), Wrote::Registered(_)])
            ),
            "{wrote:?}"
        );
        let engine = replica.row(lower.url()).id;
        let old = replica.row(first.url()).id;
        assert_eq!(replica.objects.rank_row(engine, 1), Some(old));

        let second = ranked(1, "n");
        let wrote = written(&[Arc::clone(&lower), Arc::clone(&second)], Expect::Anything);
        assert!(
            matches!(
                wrote.as_deref(),
                Ok([Wrote::Unmoved(_), Wrote::Another { .. }])
            ),
            "{wrote:?}"
        );
        let new = replica.row(second.url()).id;
        assert_ne!(new, old);
        assert!(Arc::ptr_eq(&replica.at(second.url()), &second));
        assert_eq!(replica.objects.rank_row(engine, 1), Some(new));
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

    /// A kept object's row that moves with no hand-in keeps its object, and
    /// its entry moves to the row's facts: the engine's rank is answered by
    /// them.
    #[test]
    fn a_kept_row_that_moves_with_no_hand_in_is_answered_under_its_new_rank() {
        let (replica, door) = handing();
        let lower = replica.store.upsert(rank_facts(A, C, 0)).id();
        let kept = built(B, KEY_B);
        registered(&replica, &door, &kept, Expect::Anything).expect("the write");
        let row = replica.row(B);
        assert_eq!(replica.objects.rank_row(lower, 1), None);

        // The row becomes rank 1 of the first row's engine.
        let moved = WorkerFacts {
            engine: C.to_string(),
            dp_rank: 1,
            dp_size: 2,
            ..row.facts.clone()
        };
        assert_eq!(replica.store.upsert(moved).id(), row.id);
        replica.follow();
        assert!(Arc::ptr_eq(&replica.at(B), &kept));
        assert_eq!(replica.objects.rank_row(lower, 1), Some(row.id));
    }

    // ---------------------------------------------- the overload mirror

    /// A replica whose store shows each load write at once.
    fn judging() -> Replica {
        let mut replica = Replica::new();
        replica.store = Arc::new(MemoryStore::with_publish_window(Duration::ZERO));
        replica
    }

    /// [`facts`] of a worker that names a queue of two for itself.
    fn bounded(key: &str) -> WorkerFacts {
        WorkerFacts {
            overload: Overload {
                waiting_requests: Some(2),
                kv_usage_permille: None,
            },
            ..facts(key, 1)
        }
    }

    /// A generation that hands the pools of `model` a queue of two, and
    /// every other pool no threshold.
    fn bound(model: &str) -> AppliedSpec {
        let mut document = PolicySpec::preset("round_robin").expect("a preset of this build");
        document.generation = 1;
        let profiles = &document.profiles;
        let profile = profiles.keys().next().expect("the preset's profile");
        let binding = |model: &str, overload| BindingSpec {
            matcher: Match {
                model: model.to_string(),
                ..Match::default()
            },
            profile: Some(profile.clone()),
            overload: Some(overload),
            ..BindingSpec::default()
        };
        let two = OverloadSpec {
            waiting_requests: Some(2),
            ..OverloadSpec::default()
        };
        let off = OverloadSpec {
            enabled: false,
            ..OverloadSpec::default()
        };
        document.bindings = vec![binding(model, two), binding("*", off)];
        AppliedSpec::validate(&document).expect("the document validates")
    }

    impl Replica {
        /// A registered row with its object, and its build paid.
        fn stand(&self, facts: WorkerFacts) -> WorkerId {
            let row = self.register(facts, None);
            self.follow();
            self.store.publish_now();
            row
        }

        /// A pushed record of `row`'s rank zero with `queued` requests
        /// waiting.
        fn push(&self, row: WorkerId, queued: u32) {
            let load = LoadRow {
                queued,
                ..LoadRow::default()
            };
            self.store
                .record(row, 0, load, Instant::now(), LoadSource::Stream);
        }

        fn mirror(&self) -> usize {
            self.objects.mirror_overload(self.store.as_ref())
        }

        /// How many workers of the model the registry counts overloaded.
        fn counted(&self) -> usize {
            self.objects.registry().overloaded_worker_count("m")
        }
    }

    #[test]
    fn a_published_overload_verdict_is_written_on_the_object_once_an_edge() {
        let replica = judging();
        let row = replica.stand(bounded(A));
        let worker = replica.at(A);
        // Over the row's own count. The store judges and shows the verdict,
        // and no flag moves until the host mirrors.
        replica.push(row, 3);
        assert!(!worker.is_overloaded());
        assert_eq!(replica.mirror(), 1);
        assert!(worker.is_overloaded());
        assert_eq!(replica.counted(), 1);
        // The verdict stands: nothing moves, and the count is moved once.
        replica.push(row, 4);
        assert_eq!(replica.mirror(), 0);
        assert_eq!(replica.counted(), 1);
        // The queue drains, and the next mirror clears the flag.
        replica.push(row, 1);
        assert_eq!(replica.mirror(), 1);
        assert!(!worker.is_overloaded());
        assert_eq!(replica.counted(), 0);
    }

    #[test]
    fn a_world_of_the_same_verdicts_is_not_walked_again() {
        let replica = judging();
        let row = replica.stand(bounded(A));
        let worker = replica.at(A);
        replica.push(row, 3);
        assert_eq!(replica.mirror(), 1);
        // A flag moved behind the mirror shows whether a call walked.
        let registry = replica.objects.registry();
        assert!(registry.set_worker_overloaded(&worker, false));
        // Another publication of the same verdicts is not walked.
        replica.push(row, 4);
        assert_eq!(replica.mirror(), 0);
        assert!(!worker.is_overloaded());
        // The verdict moves and moves back, and the walk writes the flag.
        replica.push(row, 1);
        assert_eq!(replica.mirror(), 0);
        replica.push(row, 3);
        assert_eq!(replica.mirror(), 1);
        assert!(worker.is_overloaded());
    }

    #[test]
    fn an_object_put_in_the_registry_owes_a_mirror_whatever_the_epoch() {
        let replica = judging();
        let row = replica.stand(bounded(A));
        replica.push(row, 3);
        assert_eq!(replica.mirror(), 1);
        let first = replica.at(A);
        // The row's labels move, and the holder replaces its object. The
        // registry releases the verdict the first object carried. The
        // build the write owes is not paid, so the world stands.
        replica.patch(A, labels("gold"));
        replica.follow();
        let second = replica.at(A);
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(!second.is_overloaded());
        assert_eq!(replica.counted(), 0);
        // The mirror walks for the object, though no verdict moved.
        assert_eq!(replica.mirror(), 1);
        assert!(second.is_overloaded());
        assert_eq!(replica.counted(), 1);
    }

    #[test]
    fn a_worker_of_two_pools_reads_overloaded_where_one_of_its_rows_is() {
        let replica = judging();
        replica.store.apply(bound("m")).expect("applied");
        let row = replica.stand(WorkerFacts {
            models: vec!["m".to_string(), "n".to_string()],
            ..facts(A, 1)
        });
        replica.push(row, 3);
        let over = replica.store.with_world(|world| world.overloaded().count());
        assert_eq!(over, 1, "one of the worker's two rows is under a verdict");
        assert_eq!(replica.mirror(), 1);
        assert!(replica.at(A).is_overloaded());
    }

    #[test]
    fn a_mirrored_edge_is_said_in_the_worker_crates_line_under_the_stores_name() {
        use tracing::{subscriber, Level};

        let log = tempfile::NamedTempFile::new().expect("a file for this case");
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
            .with_ansi(false)
            .without_time()
            .with_writer(log.reopen().expect("the log writer"))
            .finish();
        let replica = judging();
        let row = replica.stand(bounded(A));
        let _kept = subscriber::set_default(subscriber);
        replica.push(row, 3);
        assert_eq!(replica.mirror(), 1);
        replica.push(row, 1);
        assert_eq!(replica.mirror(), 1);
        let log = std::fs::read_to_string(log.path()).expect("the log text");
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 2, "{log}");
        for (line, verdict, queued) in [(lines[0], "set", 3), (lines[1], "cleared", 1)] {
            assert!(line.contains("smg_worker::load::monitor"), "{line}");
            assert!(
                line.contains(&format!("Worker overload veto {verdict}")),
                "{line}"
            );
            assert!(
                line.contains(&format!("waiting_requests={queued} ")),
                "{line}"
            );
            assert!(line.contains("waiting_threshold=Some(2)"), "{line}");
            assert!(line.contains(r#"source="policy store""#), "{line}");
        }
    }
}
