//! The holder of a replica's worker objects: every writer of a row writes
//! through it, and it writes the registry. A leader's door writes a row through
//! it, and the store follows that write ([`mod@write`]). A standby's applier
//! hands it each event of its leader's stream ([`mod@standby`]). On a
//! replica that leads an object's status leads its row's: a status is
//! written on the object, and the holder tells the store ([`mod@status`],
//! which says the lock order once). A host can
//! also ask it to build each object with the
//! thresholds its own load monitor judges overload by. A door reads the row
//! that stands off it ([`mod@read`]).

#[cfg(feature = "in-process")]
use std::sync::atomic::AtomicU64;
use std::{
    collections::{HashMap, HashSet},
    fmt,
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, Ordering::Relaxed},
        Arc,
    },
};

use parking_lot::{Mutex, RwLock};
use smg_worker::{
    connection::engine::EngineConnector,
    health::overload::OverloadThresholds,
    identity::basic::Worker,
    registry::{Generation, WorkerId as ObjectId, WorkerRegistry},
    worker::WorkerStatus as ObjectStatus,
};

#[cfg(feature = "in-process")]
use super::found::pairing_descriptor;
#[cfg(feature = "in-process")]
use super::members::{EncodeColumns, Members, Stated};
#[cfg(feature = "in-process")]
use super::reach::{ListReach, Reach};
use super::{
    credentials::{Credentials, RegistrationInFlight},
    found::engine_of,
    kv_tables::KvTables,
    membership_writer::{MembershipWriteOutcome, MembershipWriteRefused},
    object::{builder, dial_address, external, status, takes_default, thresholds, unchecked},
    remainder::RowRemainder,
};
#[cfg(feature = "in-process")]
use crate::{
    document::spec::Pairing,
    engine::{EncodeClasses, EncodeMember, FixedFacts, LentList, PartnerClasses, Role},
    store::{
        encoding::EncodeOffer,
        pairing::{Member, PairClasses, PairingDescriptor},
    },
};
use crate::{
    engine::{framework::contained, view::WorkerId},
    store::{MemoryStore, Reachability, WorkerFacts, WorkerRow, WorkersStore},
};

mod host_count;
mod read;
mod standby;
mod status;
mod write;

/// The row an object stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RowRevision {
    /// The row's id in the store: where a write about the object's worker
    /// lands.
    pub row: WorkerId,
    /// The row's revision the object was built from: the guard a write of
    /// the row's health names.
    pub revision: u64,
}

/// A row and the object that stands for it: two pointers, so a copy
/// names the same remainder and the same object.
#[derive(Clone)]
struct Laid {
    /// What of the row the object does not hold as stated, with the row's
    /// id and the revision the object was built from.
    remainder: Arc<RowRemainder>,
    object: Arc<dyn Worker>,
}

/// What stands, both ways.
#[derive(Default)]
struct Held {
    /// A row's object, by the row's id: the object's id in the registry,
    /// and what that object's entry holds. The two entries name one
    /// remainder and one object ([`Self::replace`] writes both). So a
    /// reader that asks this entry for the remainder or the object takes
    /// one lookup: the ranks' readers, a cut's compare, the lists' reader
    /// and the host count. A read of a whole row (`Objects::row`), a
    /// status copy and a standby's kept instant still go by the object's
    /// id, in two lookups.
    objects: HashMap<WorkerId, (ObjectId, Laid)>,
    /// An object's row, by the object's id in the registry.
    rows: HashMap<ObjectId, Laid>,
    /// The rows of each engine's ranks, by the address they dial: what
    /// routes a rank's batch on the engine's one events stream.
    ranks: HashMap<String, Vec<(u16, WorkerId)>>,
    /// The KV tables the host handed the holder, where it handed any. A
    /// row's lane there is made where its facts are stated and retired
    /// where the row leaves, inside this entry's lock: a lane stands
    /// exactly while a rank's batch can be routed to its row. The one lock
    /// order is this entry's, then the tables' own. Nothing of the tables
    /// calls the holder.
    tables: Option<Arc<KvTables>>,
}

impl Held {
    /// State `row`'s place in the entry, where every lay states it: at a
    /// build, and where a row another maker made again is stated anew. The
    /// row's place among its engine's ranks moves with its facts: a row laid
    /// again leaves the rank its standing object dials. Its lane in the KV
    /// tables is made, kept or made again as the facts ask, in this turn.
    fn state(&mut self, row: &WorkerRow) {
        self.unrank(row.id);
        self.rank(row.id, &row.facts);
        if let Some(tables) = &self.tables {
            tables.member(row);
        }
    }

    /// `row` stands among its engine's ranks, by the facts it is laid from.
    fn rank(&mut self, row: WorkerId, facts: &WorkerFacts) {
        self.ranks
            .entry(dial_address(facts).to_owned())
            .or_default()
            .push((facts.dp_rank, row));
    }

    /// Take `row` out of its engine's ranks, by the address its standing
    /// object dials.
    fn unrank(&mut self, row: WorkerId) {
        let standing = self.objects.get(&row);
        let Some(address) = standing.map(|(_, laid)| engine_of(laid.object.as_ref())) else {
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
    /// its engine's ranks, its object's id, and that object's
    /// own entry. Its lane in the KV tables goes with it, with the blocks
    /// it wrote. A row that leaves goes through here, at its object's end
    /// and where its next life takes its object's place.
    fn leave(&mut self, id: WorkerId) {
        // The rank first: it is found by the row's object.
        self.unrank(id);
        self.replace(id, None);
        if let Some(tables) = &self.tables {
            tables.forget(id);
        }
    }

    /// Replace what stands for `row`, both ways: `laid` under its object's
    /// id, or nothing where the row leaves. The one writer of the two
    /// maps. The row's own entry takes the two pointers its object's
    /// entry holds. So a read by the row's id and a read by the object's
    /// id find one remainder and one object. Answers the entry the write
    /// took out.
    fn replace(&mut self, row: WorkerId, laid: Option<(ObjectId, Laid)>) -> Option<Laid> {
        let Some((id, laid)) = laid else {
            let (id, _) = self.objects.remove(&row)?;
            return self.rows.remove(&id);
        };
        self.objects.insert(row, (id.clone(), laid.clone()));
        self.rows.insert(id, laid)
    }
}

/// A replica's worker objects: for every row of the membership store that
/// stands one object in the registry, and none beside.
///
/// Every writer of a row writes through the holder: a door's write call,
/// a standby's applier, a watcher's, and a restore. Each builds the
/// object of a row that arrived, drops the object of a row that went, and
/// replaces the object of a row whose facts changed under the same id in
/// the registry. A new incarnation of a key gets a new object and id.
///
/// A status change builds no object: an object's status leads its row's.
///
/// A row that leaves ends its object behind the store's write of the
/// leave: the object stands in the registry until then. A row registered
/// again under the same key and incarnation is another life under the
/// same id, and starts at the same revision: its object is built anew.
///
/// A store that took another store's rows whole holds another history: a
/// row may stand there under the id and the revision an object was built
/// from and state other facts. The holder trusts no revision it held
/// alone there. It holds each
/// row against the facts read back off its object, beside the revision.
/// Models another writer of the registry set on the object read back as
/// set: a row that states others is laid again.
/// A row whose facts and revision both stand keeps its object: one built
/// again from them would state the same row, and its replacement would
/// start every read of its worker again. A credential the host kept since
/// is no fact of a row: the object takes it at its next build. Any other
/// row is laid again and
/// keeps its place: its object is replaced under its id in the registry,
/// which carries the standing object's running state over. An entry's
/// revision never moves under an object that stands.
///
/// With a keeper ([`Self::with_credentials`]), a cut laid whole has the
/// keeper forget the rows that left, whoever wrote the removal. An entry
/// goes with its last row. A cut
/// laid whole lays the cut's own rows, then reads the status the table
/// holds for each. A write
/// the store refuses reads no row. The
/// keeper's clock is read ahead of the pass: a row bound behind that read
/// stays, and a later pass forgets it where it left.
/// Two turns forget with no read. A write call that takes a row
/// out has the keeper forget that row. [`Self::lay_again`] has it forget
/// each bound row that holds no object: an answer may bind a row that
/// left before it.
///
/// The holder is the registry's one writer: an object any other writer left
/// under a row's URL is replaced by the row's.
///
/// Every writer of a row's facts holds the holder's turn from its decision
/// to the journal's telling: a write call, a standby's applier, a
/// watcher's, and a restore. So a first copy cut in that turn states each
/// row the journal told at the copy's place, and none it did not. Three
/// writes move the journal's place with no turn of the holder, and a copy
/// stays whole beside each. A status tell changes no row's facts, and a
/// copy reads the told statuses with the place in one hold of the store's
/// membership lock. A generation's apply and a term's start change no row.
/// A change handed to the store around the holder
/// ([`MemoryStore::follow`]) takes no turn either: no product code hands
/// one so. A row written so has no object,
/// and no copy holds it, until a cut is laid whole. A standby's applier
/// and a watcher also lay it at the next event that tells the row and
/// that the store takes.
///
/// A row takes the place of its own object alone: an object another row
/// holds under its URL is dropped first.
///
/// # A host's write calls
///
/// [`Self::upsert`], [`Self::patch`], [`Self::remove`], [`Self::connect_row`]
/// and [`Self::drain_row`] are the write side of a host that keeps worker
/// objects, and each names no term. A controller's registration calls write
/// through the entries the five stand on, which are private to the crate
/// and take the call's term. Each
/// call decides over the object that stands and tells `store` what it
/// decided. The object the call leaves stands when it returns, and a
/// removed row's object is gone by then. The three calls that write a row
/// ([`Self::upsert`], [`Self::patch`], [`Self::remove`]) run in the
/// holder's turn, one at a time, and take the store's membership lock
/// inside the turn for the store's own write. One that lays an object in
/// another's place then tells `store` that object's status as a status
/// call does: inside the turn and behind the store's write, it takes the
/// object's lock and the store's lock inside it. The two that write a
/// status ([`Self::connect_row`], [`Self::drain_row`]) take no turn: each
/// runs under its object's mutation lock, the registry's lock of one
/// worker, and takes the store's membership lock inside it. So a status
/// call can run at one time with a row call, or with a status call on
/// another object. No lock of the store is held over a
/// write of the registry. A call whose caller read a row that no longer
/// stands so writes nothing. Every call but [`Self::connect_row`] says so
/// in its answer: the caller reads the row again and decides again. A host
/// numbers each life of a key itself. A call takes the incarnation its
/// caller states, and a key registered again under an incarnation it had
/// stands under its old id at the first revision: a patch or a removal
/// guarded by the earlier life's id and revision lands on it. No call asks
/// whether its replica leads. A host asks that before it calls. A
/// controller's door, whose term can end while a status waits for a lock,
/// writes one through the crate's `connect_row_under` and
/// `drain_row_under`, which ask its word of its term under the object's
/// lock and the store's, and hand the term to the store's journal with the
/// status. A leader's write through these calls is told to
/// its log, and its standbys and watchers are sent it as they are sent a
/// registration call's. A host that does not lead must not write: such a
/// write lands in its own store alone, its leader's next event no longer
/// fits that store, and the leader's snapshot takes the write back.
///
/// # A host's read calls
///
/// [`Self::read_row`], [`Self::read_row_under`] and [`Self::checkpoint`]
/// are what a host with a holder reads of its rows. Each takes the
/// holder's turn: a writer of a row waits for the read, and no decision
/// does. A row read alone carries the status its object states, and a
/// cut's row the status the store holds. Either carries the models its
/// object serves now. Each builds the rows it hands back again from
/// their objects: a copy of their facts a call.
///
/// The turn is taken once at a time, and the thread that holds it does
/// not take it again. So code that the holder runs inside a turn must
/// call nothing of the holder that takes the turn: these reads,
/// [`Self::upsert`], [`Self::patch`] and [`Self::remove`] among them.
/// Such a call never returns: it waits for the turn its own thread holds.
///
/// A host's own code runs there too. The store tells its journal of a
/// row's write inside that write's turn. The journal's last call of a
/// write, [`crate::store::Journal::released`], arrives under no lock of
/// the store, and still inside that turn. The overload switch asks the
/// host's `leads` inside its turn ([`Self::state_overload_switch`]).
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
    /// What the host keeps for its workers, by the address they dial and
    /// the row a registration wrote.
    credentials: Option<Arc<Credentials>>,
    /// Whether the object of a host-local row is built with its checks off.
    host_local_unchecked: bool,
    /// The overload thresholds an object built from an engine's row states
    /// for each signal its row names none for, where the host asked for
    /// objects with thresholds ([`Self::with_overload_thresholds`]). An
    /// external row's object states its row's own alone.
    overload_defaults: Option<OverloadThresholds>,
    /// Whether those defaults are stated at all: the overload switch of
    /// the document in force, as the host stated it last
    /// ([`Self::state_overload_switch`]). On until a host says otherwise.
    /// Written under the holder's turn, so every build reads one word.
    defaults_on: AtomicBool,
    /// One registration that keeps its credential at a time, from its keep
    /// to the lay-again its keeper's answer asks for ([`Self::keeping`]).
    keeping: Mutex<()>,
    /// The holder's turn: one runs at a time. It is held through a write
    /// of the holder's own ([`mod@write`]), a standby's event, a cut laid
    /// whole, a door's read, a first copy, a lay-again and the overload
    /// switch.
    ///
    /// The order of the locks, said here once: a door's turn or a document
    /// door's lock, then this one, then an object's mutation lock, then
    /// the store's membership lock. Nothing that holds this one takes a
    /// door's turn or a document door's lock, and no lock of the store is
    /// held over a write of the registry.
    turn: Mutex<()>,
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
    /// load monitor writes no verdict on it: nothing judges such a worker.
    ///
    /// The crate's two defaults are not read here: `defaults` stands in
    /// their place, and with no signal in `defaults` the monitor does not
    /// judge a row that names none. No document names a threshold. The
    /// monitor judges each object by its own report.
    ///
    /// The host states its document's overload switch beside them
    /// ([`Self::state_overload_switch`]). Switched off, an engine's row
    /// takes its own alone too, as an external row does.
    #[must_use]
    pub fn with_overload_thresholds(mut self, defaults: OverloadThresholds) -> Self {
        self.overload_defaults = Some(defaults);
        self
    }
}

impl<G: Generation> Objects<G> {
    /// A holder that writes `registry` and holds nothing yet. Each write
    /// through it lays its row's object, and a cut laid whole lays the
    /// object of every row of the cut.
    pub fn new(registry: Arc<WorkerRegistry<G>>) -> Self {
        Self {
            registry,
            connector: None,
            credentials: None,
            host_local_unchecked: false,
            overload_defaults: None,
            defaults_on: AtomicBool::new(true),
            keeping: Mutex::new(()),
            turn: Mutex::new(()),
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
    /// handed one ([`Self::with_credentials`]). The registration line that
    /// keeps is handed no keeper: the holder keeps for it in this one, so
    /// the line's write and the build read one keeper.
    pub fn credentials(&self) -> Option<&Arc<Credentials>> {
        self.credentials.as_ref()
    }

    /// Whether the host handed the holder a keeper
    /// ([`Self::with_credentials`]). A registration line that keeps is
    /// built over such a holder alone.
    pub(super) fn keeps_credentials(&self) -> bool {
        self.credentials.is_some()
    }

    /// Hand the holder the KV tables its rows' lanes are made in. The
    /// holder states each row to them where it states the row's facts, and
    /// retires the row's lane where the row leaves: a lane stands exactly
    /// while a rank's batch can be routed to its row. The host's cache-event
    /// feed reads the tables here ([`Self::kv_tables`]), so they are stated
    /// once. A holder handed none makes no lane, which suits a host that
    /// indexes no cache event.
    #[must_use]
    pub fn with_kv_tables(mut self, tables: Arc<KvTables>) -> Self {
        self.held.get_mut().tables = Some(tables);
        self
    }

    /// The KV tables the host handed the holder ([`Self::with_kv_tables`]),
    /// where it handed any.
    pub fn kv_tables(&self) -> Option<Arc<KvTables>> {
        self.held.read().tables.clone()
    }

    /// Lay again the object of each row that dials the address of `facts`
    /// and holds another credential than its row takes now. Each is built
    /// anew from its row and takes its standing object's place, under its
    /// id and with its running state. A registration calls it where its
    /// token's answer says so
    /// ([`KeptCredential::accepted`](super::KeptCredential::accepted),
    /// [`KeptCredential::put_back`](super::KeptCredential::put_back)),
    /// with the facts it kept. The call takes the holder's turn and reads
    /// no store: each object is built anew from the row read back from it,
    /// and `store` is handed the row's count. The keeper forgets first, by
    /// the objects that stand: an answer may have bound a row that left
    /// before it. The call reads the keeper as a build does:
    /// where a registration is in flight at the address, an object that
    /// holds what it stated marks the flight, and its put-back says so.
    /// Return how many objects were laid again: none where no keeper was
    /// handed in, where no row dials the address, or where each object
    /// there holds what its row takes.
    pub fn lay_again<S: WorkersStore + ?Sized>(&self, store: &S, facts: &WorkerFacts) -> usize {
        let _turn = self.turn.lock();
        let Some(credentials) = &self.credentials else {
            return 0;
        };
        self.forget_left(self.keeper_clock());
        let address = dial_address(facts);
        let stale: Vec<WorkerId> = {
            let held = self.held.read();
            held.rows
                .values()
                .filter(|laid| {
                    engine_of(laid.object.as_ref()) == address
                        && credentials
                            .of(address, laid.remainder.standing.row)
                            .as_ref()
                            != laid.object.api_key()
                })
                .map(|laid| laid.remainder.standing.row)
                .collect()
        };
        let mut laid = 0;
        for row in stale.into_iter().filter_map(|id| self.row(id)) {
            self.lay(store, &row);
            laid += 1;
        }
        laid
    }

    /// State the overload switch of the host's document in force: whether
    /// the defaults of [`Self::with_overload_thresholds`] are stated at
    /// all. Off, the object of an engine's row states its row's own
    /// thresholds alone, as an external row's does, and the monitor judges
    /// no worker that names none. A holder that was handed no defaults
    /// keeps the word and builds as before.
    ///
    /// Where the word moved, lay again the object of each row it changes:
    /// an engine's row that names none for a signal the defaults state.
    /// Each is built anew from the row as the holder holds it, read back
    /// from its standing object, and takes that object's place, under its
    /// id and with its running state, as a credential's lay-again does
    /// ([`Self::lay_again`]). A row that names both of its own, and an
    /// external row, keep their objects. The same word again lays nothing.
    /// The call holds the holder's turn from the word to the last lay, so
    /// every build reads one word. It reads no row of `store`: a row the
    /// holder has not read yet, and another life of a row under its id,
    /// are built under the word where a cut laid whole lays them. So a host
    /// that states the word ahead of its first write builds each object
    /// once.
    ///
    /// A host calls it where its document takes force. A controller's
    /// document door does, behind each generation's entry and inside the
    /// door's lock, which comes ahead of the holder's turn.
    ///
    /// An object laid again loses its overload verdict until its worker's
    /// next load report, the verdict of its row's own threshold too: the
    /// registry's replacement releases it, as at a credential's lay-again.
    /// One laid with no threshold, which no report judges, keeps none that
    /// a report set inside that replacement: the holder releases it again
    /// behind the registry's swap.
    ///
    /// An object laid again took its standing object's status. Behind each
    /// lay the call settles that status with the table's, as a write call
    /// does behind a put that replaces ([`Self::upsert`]): a move a health
    /// driver made on the standing object ahead of its telling finds that
    /// object replaced, and tells nothing. `leads` is the host's own word
    /// of its lead, asked inside the holder's turn, under the object's
    /// lock and again under the store's. It must call nothing that takes
    /// the turn, that object's lock or the store's lock: none of the
    /// holder's read, write or status calls. Where it holds, the table is
    /// told the object's status.
    /// Where it does not, the table is not written, and the object takes
    /// the status its row holds.
    /// Return how many objects were laid again.
    pub fn state_overload_switch(
        &self,
        store: &MemoryStore,
        on: bool,
        leads: &dyn Fn() -> bool,
    ) -> usize {
        let _turn = self.turn.lock();
        if self.defaults_on.swap(on, Relaxed) == on {
            return 0;
        }
        let Some(defaults) = self.overload_defaults else {
            return 0;
        };
        let changed: Vec<WorkerId> = {
            let held = self.held.read();
            let rows = held.rows.values().map(|laid| &laid.remainder);
            rows.filter(|row| takes_default(row.overload, &row.engine_kind, defaults))
                .map(|row| row.standing.row)
                .collect()
        };
        let mut laid = 0;
        // Each row as the holder holds it: a row the holder has not read
        // is built under the word where a cut laid whole lays it.
        for row in changed.into_iter().filter_map(|id| self.row(id)) {
            self.lay(store, &row);
            // What lays an object in another's place settles the two.
            self.settle_row(store, row.id, row.revision, leads);
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
    fn keeping<T>(
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

    /// One registration that keeps its credential, whole, in a turn of its
    /// own ([`Self::keeping`]). The holder keeps what `stated` holds, the
    /// facts and the credential of the registration's first worker, ahead
    /// of `write`: a row's object may be built inside the door's turn.
    /// `write` writes the rows at the door, as a guarded write over `over`
    /// under `url` where it names one. The holder answers its keeper with
    /// the door's answer, with no await between the two: the rows the door
    /// wrote are accepted, and a refusal puts back what stood. Where the
    /// keeper says an object may hold another credential than its row
    /// takes, the holder lays the address again and hands `store` each
    /// row's count ([`Self::lay_again`]).
    ///
    /// One registration is in flight at an address: a second is refused
    /// ahead of `write` ([`RegistrationInFlight`]), and its caller tries
    /// again. A registration that states no worker keeps nothing and
    /// writes all the same, and so does one over a holder with no keeper.
    pub(super) fn register_keeping<S: WorkersStore + ?Sized>(
        &self,
        store: &S,
        stated: Option<(&WorkerFacts, Option<String>)>,
        (url, over): (&str, Option<&Arc<dyn Worker>>),
        write: impl FnOnce() -> Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused>,
    ) -> Result<Result<Vec<MembershipWriteOutcome>, MembershipWriteRefused>, RegistrationInFlight>
    {
        let whole = || {
            let kept = match (&self.credentials, stated) {
                (Some(credentials), Some((facts, credential))) => {
                    Some((credentials.keep(facts, credential)?, facts))
                }
                _ => None,
            };
            let answer = write();
            if let Some((kept, facts)) = kept {
                let rows: Vec<WorkerId> = answer
                    .iter()
                    .flatten()
                    .filter_map(MembershipWriteOutcome::written_row)
                    .collect();
                let again = if rows.is_empty() {
                    kept.put_back()
                } else {
                    kept.accepted_rows(&rows)
                };
                if again {
                    self.lay_again(store, facts);
                }
            }
            Ok(answer)
        };
        match self.keeping(url, over, whole) {
            Ok(answer) => answer,
            Err(moved) => Ok(Err(moved)),
        }
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
    /// one whose object ended.
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
        let held = self.held.read();
        held.objects.get(&row).map(|(id, _)| id.clone())
    }

    /// The row of rank `rank` of the engine whose rank 0 row is `row`:
    /// `row` itself for rank 0, else the row that dials the same address
    /// under that rank. `None` where the holder laid no such row, holds no
    /// object for `row`, or `row` stands above rank 0 by the facts it is
    /// laid from. Only an engine's rank 0 row holds the engine's one events
    /// stream, and the stream's writer routes a rank's batch by this
    /// answer: a row laid again above rank 0 routes none. The row's
    /// entry is found by one lookup, by the row's id; a rank above 0
    /// reads the engine's ranks too, by the address the row's object
    /// dials.
    pub fn rank_row(&self, row: WorkerId, rank: u16) -> Option<WorkerId> {
        let held = self.held.read();
        let (_, laid) = held.objects.get(&row)?;
        if laid.remainder.dp_rank != 0 {
            return None;
        }
        if rank == 0 {
            return Some(row);
        }
        held.ranks
            .get(engine_of(laid.object.as_ref()))?
            .iter()
            .find(|&&(ranked, _)| ranked == rank)
            .map(|&(_, row)| row)
    }

    /// The rows of every rank of the engine whose rank 0 row is `row`,
    /// its own among them: the rows that dial the address `row`'s object
    /// dials. None where the holder holds no object for `row`, or `row`
    /// stands above rank 0 by the facts it is laid from: such a row's
    /// stream is routed no rank ([`Self::rank_row`]). One read of the
    /// entry under its read lock.
    pub(crate) fn rank_rows(&self, row: WorkerId) -> Vec<WorkerId> {
        let held = self.held.read();
        let lowest = held.objects.get(&row);
        let lowest = lowest.filter(|(_, laid)| laid.remainder.dp_rank == 0);
        let ranks = lowest.and_then(|(_, laid)| held.ranks.get(engine_of(laid.object.as_ref())));
        let ranks = ranks.into_iter().flatten();
        ranks.map(|&(_, ranked)| ranked).collect()
    }

    /// Whether the row `row` stands at rank `rank`, by the facts its
    /// object was laid from. `false` for a row the holder holds no object
    /// for. Asked of the worker a dispatch report names, which may have
    /// left or changed its rank since its decision. One read of the
    /// entry under its read lock, one lookup by the row's id, a leaf: the
    /// caller may hold a turn of the leases' files.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "a controller's word to its engine asks it")
    )]
    pub(crate) fn lists_rank(&self, row: WorkerId, rank: u16) -> bool {
        let held = self.held.read();
        held.objects
            .get(&row)
            .is_some_and(|(_, laid)| laid.remainder.dp_rank == rank)
    }

    /// How many objects stand.
    pub fn len(&self) -> usize {
        self.held.read().objects.len()
    }

    /// Whether no object stands.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The keeper's clock, read ahead of a pass of the forgetting: a row a
    /// registration bound behind it is kept by that pass.
    fn keeper_clock(&self) -> Option<u64> {
        self.credentials
            .as_ref()
            .map(|credentials| credentials.taken())
    }

    /// Have the keeper forget the rows that left: each row a registration
    /// bound that no longer stands here, by the keeper's clock as it read
    /// ahead of the pass. One pass over the keeper's rows, behind a cut
    /// laid whole and at a lay-again.
    fn forget_left(&self, read: Option<u64>) {
        let (Some(credentials), Some(read)) = (&self.credentials, read) else {
            return;
        };
        let held = self.held.read();
        credentials.forget_left(read, |row| held.objects.contains_key(&row));
    }

    /// Have the keeper forget the one row `left`, where no object stands
    /// for it any more: a write took it out. No other bound row is asked
    /// about.
    /// A row bound for a write the holder has not read yet so stays.
    fn forget_one(&self, read: Option<u64>, left: WorkerId) {
        let (Some(credentials), Some(read)) = (&self.credentials, read) else {
            return;
        };
        let held = self.held.read();
        credentials.forget_left(read, |row| row != left || held.objects.contains_key(&row));
    }

    /// A cut's rows, whole: the objects of rows that went are dropped
    /// first, each leaving its URL to a later incarnation's, then every row
    /// its object does not state as it stands is laid again. No
    /// revision held is trusted alone here: in another history, a row
    /// under a held id and revision may be another life with other
    /// facts. The facts are compared beside it, read back off the object. A row that
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
                // The store may have given the row a new place in its
                // counts, which holds no count yet: a lay hands its own.
                self.counted(store, row.id);
            } else {
                self.lay(store, row);
            }
        }
    }

    /// Whether `row`'s object states the row as it stands now: the
    /// revision it stands at, and the facts the row states, read back off
    /// the object. One that does keeps its object and its place. Models
    /// set on the object since its build read back as set: a row that
    /// states others does not stand.
    fn stands(&self, row: &WorkerRow) -> bool {
        let held = self.held.read();
        held.objects.get(&row.id).is_some_and(|(_, laid)| {
            laid.remainder.standing.revision == row.revision
                && laid.remainder.row(laid.object.as_ref()).facts == row.facts
        })
    }

    /// Whether the entry kept for `row` names another maker of the row, or
    /// another time it was admitted at, than the row states.
    fn remade(&self, row: &WorkerRow) -> bool {
        let held = self.held.read();
        held.objects.get(&row.id).is_some_and(|(_, laid)| {
            laid.remainder.maker != row.maker || laid.remainder.admitted_at != row.admitted_at
        })
    }

    /// The status the table holds for the row an object was built from,
    /// where the row stands at that revision still: what an object takes
    /// from the table. It is the store's read of one told status
    /// ([`MemoryStore::told_status`]): no row is copied. Read under the
    /// object's mutation lock, so of two such reads the later one's
    /// status stands on the object.
    fn table_status(store: &MemoryStore, standing: RowRevision) -> Option<ObjectStatus> {
        store
            .told_status(standing.row, standing.revision)
            .map(status)
    }

    /// Read the row's told status under the lock the health adapter
    /// writes under. The object's entry must still name that row and its
    /// current revision. The registry writes the status it is handed and
    /// tells it.
    #[inline(never)]
    fn copy_status(&self, store: &MemoryStore, row: WorkerId) {
        let Some(id) = self.object(row) else {
            return;
        };
        let Some(object) = self.registry.get(&id) else {
            return;
        };
        let write = |object: &Arc<dyn Worker>| {
            let next = self
                .standing_of(&id, object)
                .and_then(|standing| Self::table_status(store, standing));
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
    /// row's facts and revision to be the ones its object states
    /// ([`Self::stands`]), so nothing a generation's lists, pair classes or
    /// encode columns are derived from moves here, and the registry is not
    /// written.
    fn restate(&self, row: &WorkerRow) {
        let held = &mut *self.held.write();
        let standing = held.objects.get(&row.id);
        let standing = standing.map(|(id, laid)| (id.clone(), Arc::clone(&laid.object)));
        let Some((id, object)) = standing else {
            return;
        };
        let remainder = Arc::new(RowRemainder::of(row));
        held.replace(row.id, Some((id, Laid { remainder, object })));
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
        // takes none, and no row takes one while the host's document
        // switches the default off.
        if let Some(defaults) = self.overload_defaults {
            worker = worker.overload(thresholds(&row.facts.overload));
            if !external(&row.facts.engine_kind) && self.defaults_on.load(Relaxed) {
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
    ///
    /// A row holds one object. Where the row's own object stands under
    /// another URL, that one is dropped first too. A cut that names an id
    /// again under another address lays its rows in order: the first's
    /// object ends, and the last stands until a later row of the cut takes
    /// its address.
    fn put<S: WorkersStore + ?Sized>(&self, store: &S, row: &WorkerRow, object: Arc<dyn Worker>) {
        // The id first, then the entry, then the registry's write: its
        // event finds the entry.
        let mut id = self.registry.reserve_id_for_url(&row.facts.key);
        // A key's next life finds its old life's object under the URL:
        // that one leaves first.
        let others = |other: &RowRevision| other.row != row.id;
        if let Some(other) = self.standing(&id).filter(others) {
            self.end(other.row);
            id = self.registry.reserve_id_for_url(&row.facts.key);
        }
        // The row's own object under another URL leaves first too.
        if self.object(row.id).is_some_and(|own| own != id) {
            self.end(row.id);
        }
        let standing = {
            let mut entries = self.held.write();
            // The ranks first: a row laid again leaves the rank its
            // standing object dials.
            entries.state(row);
            let laid = Laid {
                remainder: Arc::new(RowRemainder::of(row)),
                object: Arc::clone(&object),
            };
            let standing = entries.replace(row.id, Some((id, laid)));
            standing.map(|standing| standing.object)
        };
        self.registry.register_or_replace(Arc::clone(&object));
        if let Some(standing) = standing {
            self.release_verdict(&standing, &object);
        }
        // Behind the registry's write: an object laid in another's place
        // takes the standing object's running state there, and with it the
        // leases the store counted on that object.
        self.counted(store, row.id);
    }

    /// Release the overload verdict `laid` took over from `standing`, the
    /// object it was laid in the place of, where a load report judges
    /// `standing` and none judges `laid`: `standing` states a threshold and
    /// `laid` states none.
    ///
    /// The registry releases a verdict ahead of its swap, and outside the
    /// lock its overload writers hold. A report that lands between the two
    /// sets the flag again, on the running state both objects share. An
    /// object that states a threshold is judged again at its worker's next
    /// report. One that states none is judged by no report, so the flag
    /// would stand on it until its worker left.
    ///
    /// What follows leans on the registry's one writer of the flag
    /// ([`WorkerRegistry::set_worker_overloaded`]): each write is made
    /// under that lock and behind its test that the registry holds the
    /// handle still, and a call returns ahead of the lock where the flag
    /// holds the word asked already.
    ///
    /// Behind the swap that test drops a write through `standing`. One
    /// writer can hold the lock from ahead of the swap still, its write not
    /// made yet. So a write through `standing` goes first, of the word the
    /// flag does not hold: the registry drops it, behind that writer. Where
    /// the flag moved ahead of it, that writer has written. Either way the
    /// release that follows is the last word on the flag.
    ///
    /// The routine leaves when the registry releases and swaps under one
    /// hold of that lock.
    fn release_verdict(&self, standing: &Arc<dyn Worker>, laid: &Arc<dyn Worker>) {
        let judged = |object: &Arc<dyn Worker>| object.metadata().overload.is_enabled();
        if !judged(standing) || judged(laid) {
            return;
        }
        // The swap went through: the registry holds `laid`, and no longer
        // `standing`.
        let held = self.registry.get_by_url(laid.url());
        if !held.is_some_and(|held| Arc::ptr_eq(&held, laid)) {
            return;
        }
        self.registry
            .set_worker_overloaded(standing, !standing.is_overloaded());
        self.registry.set_worker_overloaded(laid, false);
    }
}

/// A removal of the row `id` in its two steps: `told`, the store's write
/// of it, then `behind`, what the removal does behind the store's yes.
/// Answers what `told` answers. Where `told` answers yes, `behind` runs.
/// Where a step unwinds, the store is asked for the row. Where it holds
/// the row no more, `behind` runs, a second time where it had begun. A
/// fault of that read or that run stops there ([`contained`]): its payload
/// is released in the holder's turn, and what the run had not done stays
/// undone. Once that release returns, the caught unwinding is resumed.
///
/// The read takes the store's membership lock and lets it go. `behind`
/// runs there as it runs behind a yes: in the holder's turn, and under no
/// lock of the two steps, whose frames have unwound.
fn leave_whole(
    store: &MemoryStore,
    id: WorkerId,
    told: impl FnOnce() -> bool,
    behind: impl Fn(),
) -> bool {
    let steps = || {
        let taken = told();
        if taken {
            behind();
        }
        taken
    };
    match catch_unwind(AssertUnwindSafe(steps)) {
        Ok(taken) => taken,
        Err(unwinding) => {
            contained(|| {
                if store.get(id).is_none() {
                    behind();
                }
            });
            resume_unwind(unwinding)
        }
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
    /// either in place publishes where a cut laid whole lays the row
    /// again: its new object takes the standing one's place in the
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
    /// its pairing descriptor. A row that states no descriptor is read by
    /// what its worker object offers a partner ([`pairing_descriptor`]):
    /// the object was built of the row, so those are the pairing facts of
    /// the row's labels and engine kind, as a gateway reads them of its
    /// own worker. A controller's door lays no descriptor in a row, so its
    /// workers pair by their labels. A lent worker whose row's object was
    /// replaced or ended before a set is derived pairs with nobody in that
    /// set: class `0`. A set that stands is not derived again for a worker
    /// that goes later. The generation is the registry's old one by then,
    /// and its classes stand as its lists do.
    ///
    /// A generation keeps one set of classes a pairing mode, derived at
    /// the first ask under the mode. What a set is derived from does not
    /// move under a generation: a write that moves a row's wires, its
    /// engine kind, its labels or its descriptor gets the row a new object,
    /// in place or as another life, and a new object publishes another
    /// generation.
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
    ///
    /// One side alone is no pair: where a list lends no worker whose row's
    /// object stands, no member is made and the rule does not run, and
    /// every worker of both lists takes class `0`.
    fn derive_pairs(&self, generation: &Members, mode: Pairing) -> PartnerClasses {
        /// A row as the rule reads it: by the descriptor it states, else by
        /// `own`, what its worker object offers a partner.
        fn member<'a>(
            row: &'a Arc<RowRemainder>,
            own: Option<&'a PairingDescriptor>,
        ) -> Member<'a> {
            let descriptor = row.pairing.as_deref().or(own);
            Member::new(
                row.standing.row,
                row.wire_modes,
                &row.engine_kind,
                descriptor,
            )
        }
        let lists = [Role::Prefill, Role::Decode].map(|role| self.lent(generation, role));
        let stated = self.remainders(lists);
        if stated.iter().any(|side| side.iter().all(Option::is_none)) {
            let nobody = |side: usize| vec![0; stated[side].len()].into();
            return PartnerClasses::derived(
                nobody(0),
                nobody(1),
                Default::default(),
                Default::default(),
            );
        }
        // What each lent worker offers by itself, where its row states no
        // descriptor. The object was built of the row, so these are the
        // row's labels and engine kind under the worker's own rule.
        let own = [0, 1].map(|side| {
            let offered = |(worker, row): (&Arc<dyn Worker>, &Option<Arc<RowRemainder>>)| {
                let silent = row.as_ref().is_some_and(|row| row.pairing.is_none());
                silent
                    .then(|| pairing_descriptor(worker.as_ref()))
                    .flatten()
            };
            let side = lists[side].workers.iter().zip(&stated[side]);
            side.map(offered).collect::<Vec<_>>()
        });
        let members = [0, 1].map(|side| {
            let rows = stated[side].iter().zip(&own[side]);
            rows.filter_map(|(row, own)| Some(member(row.as_ref()?, own.as_deref())))
                .collect::<Vec<_>>()
        });
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
    /// The classes are the engine's encode rule and no other
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
    /// answers for the side that is missing.
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
                let (_, laid) = held.objects.get(&fact.id?)?;
                Arc::ptr_eq(&laid.object, worker).then(|| Arc::clone(&laid.remainder))
            };
            list.workers.iter().zip(facts).map(row).collect()
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/objects/tests.rs"]
pub(super) mod tests;
