//! The one-process store: every ingress trait over plain tables behind
//! mutexes. The controller's own store once its drivers move in (soft state
//! its sources republish, no log), and the double the contract tests and the
//! controller's tests run over. Nothing here is on a decision path: a view is
//! built from the world the store publishes ([`MemoryStore::world`]), not
//! through these locks.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    mem,
    ops::{Deref, DerefMut},
    sync::Arc,
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::FxHashMap;

use crate::{
    lease::{LeaseId, Lifecycle, Membership, Outcome, ReleaseReason, Side, TransportClass},
    spec::AppliedSpec,
    store::{
        health::{HealthSink, HealthWrite, VetoWrite},
        index::{indexed_under, KvEvent, KvTables, LaneCounts},
        journal::{Journal, Kinds, Position},
        kv_events::{BatchFacts, BatchOutcome, KvCursor, KvEventSink, ResyncReason},
        leases::{
            Counts, Dispatch, LeaseRecord, LeaseRefused, LeaseState, LeasesStore, Said, Slot,
        },
        loads::{LoadSample, LoadSink, LoadSource, PollMode, Wake},
        reservations::{ReservationKind, ReservationStore},
        sessions::{Pin, SessionsStore},
        workers::{
            identity, MembershipChange, MembershipTail, WorkerFacts, WorkerPatch, WorkerRow,
            WorkerStatus, WorkersStore,
        },
        world::{
            left_models, micros, same_offer, Ask, Built, Classes, EncodePairBuffers, OutsideLists,
            PairBuffers, StaleGeneration, ViewBuffers, World, Writer, PUBLISH_WINDOW,
        },
        GatewayId,
    },
    view::{LoadRow, Role, WorkerId},
};

mod blocks;
mod turn;

use blocks::Blocks;
#[cfg(any(test, feature = "controller"))]
use blocks::Minted;
pub use turn::{AtWork, Hand};

/// Membership changes kept for watchers; a watcher further behind snapshots
/// again (the registry's broadcast is sized the same today).
const CHANGE_LOG: usize = 1_024;

/// The store's counts one scrape reports: the first two of one state of
/// the store.
#[derive(Clone, Copy, Debug, Default)]
pub struct StoreCounts {
    /// The members outside the standing generation's lists.
    pub outside: OutsideLists,
    /// The load writes that moved an overload verdict.
    pub overload_flips: u64,
    /// The lifecycle calls of a lease that panicked and were held
    /// ([`MemoryStore::lifecycle_faults`]): a count of its own, read
    /// beside the two.
    pub lifecycle_faults: u64,
}

/// The store. `Default` is empty with term `0`.
pub struct MemoryStore {
    /// The rows, behind the membership lock. A write that records a change
    /// holds it as a [`Writing`], which tells the journal when the write's
    /// locks are let go.
    workers: Mutex<Workers>,
    health: Mutex<HealthTables>,
    /// The load rows, one entry per member: an entry is created with its
    /// worker's row and deleted with it, both under the workers lock (taken
    /// first, as the health writers take it). A load write goes into its
    /// worker's entry or is dropped, under this lock alone: the entry is the
    /// membership test, so a read that lands after a removal re-creates
    /// nothing and a replaced incarnation starts empty. One load write takes
    /// the workers lock first: a poll's answer for a rank that waits for its
    /// build, which it pays ([`LoadSink::report`]). The world's write side
    /// sits behind the same lock: a record is on file and folded into the
    /// working rows in one step, and every world is made and shown under
    /// it, so the deciders see the worlds in the order they were made. It
    /// is the last lock a writer takes: the workers lock, then the health
    /// lock, then this one.
    loads: Mutex<Loads>,
    /// The KV lanes and the indexes they write, behind locks of their own
    /// ([`KvTables`]): a membership write takes the table's after the
    /// workers lock, a batch takes it alone. The writer holds the same
    /// tables and reads each pool's standing in them.
    kv: Arc<KvTables>,
    reservations: Mutex<Reservations>,
    /// The pins of record, in the shards of the engine's own table
    /// ([`crate::sessions::shard_of`]): a dispatch's write and a lookup
    /// hold one shard, and a sweep or a worker's removal walks them one
    /// hold after another.
    sessions: Vec<Mutex<FxHashMap<u64, Pin>>>,
    /// When the pins were last swept and how often a pass runs
    /// ([`Self::sweep_pins`]).
    #[cfg(feature = "controller")]
    pins_swept: Mutex<PinsSwept>,
    /// The files of leases behind their turns, one a decider ([`turn`]).
    files: turn::Files,
    /// The blocks of the leading term's counter, each block's file and the
    /// tables kept for terms that led ([`blocks`]): how a lease's id names
    /// its file.
    blocks: Blocks,
    /// What a decision reads, through one load: swapped whole at each
    /// publication.
    world: ArcSwap<World>,
    /// Every member that an owed build may list and no published world
    /// lists yet, each with its slot in the counts. The write that files
    /// a member puts it here. So does a refresh that changes an offer: a
    /// member that offers no model stands in no pool and in no world. A
    /// membership build empties the table behind its publication, and no
    /// other publication touches it. A leaf lock: nothing is taken under
    /// it.
    joined: Mutex<Vec<(WorkerId, Slot)>>,
    /// The store's counts of the leases on file, for the write that gives
    /// a joining member its slot.
    counts: Arc<Counts>,
    /// Where the rows' clock starts.
    origin: Instant,
    /// Where the term is told, and that a write's locks are let go; `None`
    /// for a store nobody follows. The membership writes tell through
    /// their table's own handle.
    journal: Option<Arc<dyn Journal>>,
    /// The same journal where it carries load rows: where the load writes
    /// are told.
    load_journal: Option<Arc<dyn Journal>>,
    /// The host's wake, once its loop handed one ([`LoadSink::wake_with`]):
    /// called where a membership write leaves a build owed and none was,
    /// the one instant the host's loop cannot have read. A lock of its own,
    /// the last taken and held for the call alone.
    wake: Mutex<Option<Wake>>,
}

/// The store's replicated tables at one place of its history
/// ([`MemoryStore::cut`]).
#[derive(Clone, Debug)]
pub struct Cut {
    /// The last journaled write of the kinds asked that the tables hold,
    /// with its term; the default for a store with no journal.
    pub at: Position,
    /// The membership epoch.
    pub epoch: u64,
    /// Every row, by registration key.
    pub rows: Vec<WorkerRow>,
    /// Every load row on file, by worker and rank; none in a cut of the
    /// watched kinds.
    pub loads: Vec<(WorkerId, u16, LoadSample)>,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::with_publish_window(PUBLISH_WINDOW)
    }
}

impl std::fmt::Debug for MemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // One guard per table, each dropped before the next is taken: the
        // mutex is not reentrant and a chained statement keeps every guard
        // to its end.
        let (workers, epoch) = {
            let workers = self.workers.lock();
            (workers.rows.len(), workers.epoch)
        };
        // Each turn is tried, never taken: a line that prints the store
        // from under a turn (a tell's, a panic's) reads no count there.
        let leases = self.files.iter().try_fold(0, |leases, file| {
            Some(leases + file.try_take()?.records.len())
        });
        let mut form = f.debug_struct("MemoryStore");
        form.field("workers", &workers).field("epoch", &epoch);
        match leases {
            Some(leases) => form.field("leases", &leases),
            None => form.field("leases", &format_args!("under the turn")),
        };
        form.finish_non_exhaustive()
    }
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// A store whose folded load records wait at most `window` for their
    /// publication, where [`MemoryStore::new`] takes [`PUBLISH_WINDOW`]:
    /// for a test of the window itself. The build a membership write owes
    /// waits for the same window. A window the clock cannot add to an
    /// instant never ends: a waiting record is then shown by the next
    /// status write or the next build, and a build is paid by the ask
    /// alone ([`MemoryStore::publish_now`]).
    pub fn with_publish_window(window: Duration) -> Self {
        let now = Instant::now();
        // An hour before the store was made, where the host's clock reaches
        // that far: a sample older than the store keeps an instant of its
        // own.
        let origin = now.checked_sub(Duration::from_secs(3_600)).unwrap_or(now);
        let kv = Arc::new(KvTables::default());
        let writer = Writer::new(window, origin).over(Arc::clone(&kv));
        let world = ArcSwap::new(writer.world());
        // The leases' file writes the counts the writer's shapes name, and
        // a membership write gives a joining member its slot in them.
        let counts = writer.counts();
        let leases = Leases {
            counts: Arc::clone(&counts),
            ..Leases::default()
        };
        Self {
            workers: Mutex::default(),
            health: Mutex::default(),
            loads: Mutex::new(Loads {
                ranks: FxHashMap::default(),
                writer,
            }),
            kv,
            reservations: Mutex::default(),
            sessions: (0..crate::sessions::SHARDS)
                .map(|_| Mutex::default())
                .collect(),
            #[cfg(feature = "controller")]
            pins_swept: Mutex::default(),
            files: turn::Files::one(leases),
            blocks: Blocks::default(),
            world,
            joined: Mutex::default(),
            counts,
            origin,
            journal: None,
            load_journal: None,
            wake: Mutex::new(None),
        }
    }

    /// A store that tells `journal` its replicated writes (membership and
    /// status, the term, and the load rows where the journal carries
    /// them), each inside the write's own critical section: a store a
    /// standby or a watcher of its rows can follow.
    pub fn journaled(journal: Arc<dyn Journal>) -> Self {
        let mut store = Self::default();
        store.workers.get_mut().journal = Some(Arc::clone(&journal));
        store.load_journal = journal.carries_loads().then(|| Arc::clone(&journal));
        store.journal = Some(journal);
        store
    }

    /// The term the journal followed where the store's last take started
    /// no term in it: the journal had seen that term or a later one, so
    /// the store's writes stand under no term and no standby follows them
    /// until a later take. `None` while the store leads the journal's
    /// term, and for a store with no journal.
    pub fn followed(&self) -> Option<u64> {
        self.journal.as_ref().and_then(|journal| journal.followed())
    }

    /// The tables of `kinds` as one journaled write of those kinds left
    /// them, with that write's position: the rows, and with [`Kinds::All`]
    /// the load rows. What a watcher of the rows starts from, and what a
    /// standby is restored from. Taken under the locks of the kinds asked,
    /// in the store's order (the membership lock; the load lock for the
    /// load rows alone): every journaled write holds its table's lock from
    /// its apply to its telling, so none stands half told here. A writer
    /// of those tables waits for the copy; a decision reads the published
    /// world and does not, and a cut of the rows leaves the load writers
    /// be. Where the journal carries no load row a cut of every kind has
    /// no tail: that journal is lost to a reader of every kind.
    pub fn cut(&self, kinds: Kinds) -> Cut {
        // Copied under the locks, sorted after them: a writer waits for the
        // copy alone.
        let (at, epoch, mut rows, mut samples) = {
            let workers = self.workers.lock();
            let loads = (kinds == Kinds::All).then(|| self.loads.lock());
            let at = self
                .journal
                .as_ref()
                .map_or_else(Position::default, |journal| journal.head(kinds));
            let rows: Vec<WorkerRow> = workers.rows.values().map(|row| row.row.clone()).collect();
            let samples: Vec<(WorkerId, u16, LoadSample)> = loads
                .iter()
                .flat_map(|loads| loads.ranks.iter())
                .flat_map(|(id, ranks)| {
                    ranks
                        .samples
                        .iter()
                        .map(|(rank, sample)| (*id, *rank, *sample))
                })
                .collect();
            (at, workers.epoch, rows, samples)
        };
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        samples.sort_by_key(|(id, rank, _)| (*id, *rank));
        Cut {
            at,
            epoch,
            rows,
            loads: samples,
        }
    }

    /// The tables replaced by `cut`, a leader's snapshot whole: what a
    /// standby starts from, and starts from again when the leader's log no
    /// longer holds its place. Every row and load row the store held leaves
    /// with what lives as long as a row (the prober's counts, the gateways'
    /// completions, their vetoes in the table and in the world), the epoch
    /// is the cut's, and the journal starts again at the cut's place inside
    /// the same locks: a reader of the tables sees them before the restore
    /// or after it, never between. The KV lanes follow the rows: one that
    /// left gives up its lane, one that stays keeps what this replica heard
    /// for it. The store's term rises to the cut's, as
    /// following that term's start leaves it. The engine is told the
    /// members once, by id, as an apply tells a generation's
    /// ([`Membership::Snapshot`]), and the journal's listener once the
    /// restore's locks are let go. `false`, with nothing changed, where the
    /// journal refuses the place ([`Journal::restart`]).
    ///
    /// The pools are built from the cut's rows at once, whatever the
    /// publication window, and no build is left owed: a decision after the
    /// restore reads its rows.
    ///
    /// A replica that restores holds no lease. What stands on file was
    /// filed while this replica led and is another history's by now: its
    /// end was told to another leader, and no frame of the stream ends it
    /// here. Behind the restore's locks each lease leaves its file on the
    /// count it booked: its end is handed to the file as a removed
    /// worker's release ([`ReleaseReason::WorkerRemoved`]) and told as a
    /// frame's is, by the file's decider at work or by the restore where
    /// that decider rests. The engine of the lease's own generation hears
    /// it once, where that generation stands. The restore returns when
    /// every end is handed: a lease of a file whose decider works stands,
    /// its count booked, until that decider's next take, and a full ring
    /// holds the restore's thread until then. A gateway's
    /// `Hello` files what it lists again once this replica leads
    /// ([`LeasesStore::restore`]).
    ///
    /// A take beside the restore stands whole on one side of it. The
    /// take's write waits for the files' turns ([`LeasesStore::set_term`]),
    /// which the restore holds from its start until it has read the ids
    /// on file: the leases it ends are those, and none of a term taken
    /// behind it. Ahead of the restore, the journal weighs the cut
    /// against the take. An issue reads its slot before it takes its
    /// file's turn ([`LeasesStore::issue`]): one that read it ahead of the
    /// swap and waited here books a slot the restored world does not
    /// read, until its lease ends.
    pub fn restore_from(&self, cut: Cut) -> bool {
        // Taken before the first lock, so dropped after the last: the journal
        // restarts inside the locks, and its listener is told behind them.
        let _released = self.released();
        // Every file's turn first, as their order has it, and held until
        // the ids on file are read: a take's write waits for them, so a
        // take lands whole ahead of the restore or whole behind it.
        let mut turns = self.every_turn();
        let mut workers = self.workers.lock();
        let held: Vec<WorkerId>;
        {
            let mut health = self.health.lock();
            let mut loads = self.loads.lock();
            if let Some(journal) = &self.journal {
                if !journal.restart(cut.at) {
                    return false;
                }
            }
            health.vetoes.clear();
            health.ok_at.clear();
            held = workers.rows.keys().copied().collect();
            workers.rows.clear();
            workers.by_key.clear();
            workers.log.clear();
            workers.epoch = cut.epoch;
            // A world's vetoes leave with their rows at a membership write:
            // one over no row takes them all, and its world is shown to
            // no one. The fields each member was read at under the
            // generation's lists leave with them: a cut can bring a row at
            // an id and a revision the world read under other facts.
            drop(
                loads
                    .writer
                    .rebuild(cut.epoch, std::iter::empty(), |_, _| None, Classes::Derive),
            );
            loads.ranks.clear();
            for row in cut.rows {
                workers.by_key.insert(row.facts.key.clone(), row.id);
                loads.ranks.insert(row.id, Ranks::default());
                let slot = self.join(row.id);
                workers.rows.insert(
                    row.id,
                    Row {
                        row,
                        probes: Probes::default(),
                        slot,
                    },
                );
            }
            for (id, rank, sample) in cut.loads {
                if let Some(ranks) = loads.ranks.get_mut(&id) {
                    ranks.samples.insert(rank, sample);
                }
            }
        }
        // The index is this replica's own: a row that left gives up its
        // lane, a row that stays keeps what was heard for it, and a row the
        // cut brings takes the lane its facts ask for.
        for id in held {
            if !workers.rows.contains_key(&id) {
                self.kv.forget(id);
            }
        }
        for row in workers.rows.values() {
            self.kv.member(&row.row);
        }
        self.swap_world(&workers, Classes::Derive);
        // Every row was replaced: the engine is told the members once, by
        // id, as an apply tells a generation's.
        self.tell_members(&workers);
        drop(workers);
        // Under the turns taken at the start the term rises and the ids on
        // file are read: each of those leases then leaves its file as a
        // frame's release does, handed with the turns let go
        // ([`Self::empty`]). A later generation's stages never saw the
        // issue and hear nothing of it.
        self.rise(&mut turns, cut.at.term);
        self.empty(turns);
        true
    }

    /// One of the leader's membership or status writes, applied as the
    /// replication stream tells it: a standby's side of a journaled write
    /// of [`WorkersStore`] or [`HealthSink`]. `row` is the row as the
    /// leader's write left it, and none for a removal or a status change;
    /// a resize is told as a relabel and recorded as the leader recorded
    /// it, by the widths. Nothing is minted and no revision is guarded: the
    /// leader's log is the order. A row that leaves takes its tables with
    /// it as the leader's own write does, but for the sessions' pins of
    /// record that name it, which the leader's own write alone drops, and
    /// a row that stays keeps its admission instant. `at` is the event's
    /// place in that log, which the applier holds ([`Journal::member`]):
    /// the journal takes the write there, and no write of the store's own
    /// does.
    ///
    /// The pools follow as at the leader's own write: a join, a refresh and
    /// a patch leave their build owed, which the replica's loop pays at the
    /// window's end, and a leave pays its own before it is told: a removal,
    /// a replaced incarnation, a registration that takes its worker off a
    /// model.
    ///
    /// The change is applied only where it fits the tables: the epoch is
    /// the next one, a new row's id is free and a registration's key too,
    /// a row that stays or is replaced keeps its key, and any other change
    /// names a row the store holds (a status change, in the status it
    /// leaves). A change that does not fit writes nothing and tells the
    /// journal nothing, so the guard of its position does not land and the
    /// caller starts again from a snapshot.
    ///
    /// The world's engine is told the change as at the leader's own write
    /// ([`MembershipChange::as_lifecycle`]), and with it the models a
    /// registration took its worker off ([`Membership::Left`]): ahead of
    /// the change where the registration keeps its id, and behind it, for
    /// the new id, where a new incarnation replaced the row. A standby's
    /// stages hold what its leader's hold, with no step at a take.
    pub fn follow(
        &self,
        at: Position,
        epoch: u64,
        change: MembershipChange,
        row: Option<WorkerRow>,
    ) {
        let mut workers = self.writing_at(Some(at));
        if epoch != workers.epoch + 1 {
            return;
        }
        // A row told again with the offer it had moves no pair class, here
        // as in the leader's own write: the classes are carried.
        let mut carried = false;
        // The models a registration that stays under its id takes its
        // worker off, for the stages: here as in the leader's own write.
        let mut off = Vec::new();
        let (change, left) = match (change, row) {
            (MembershipChange::Registered { id }, Some(row))
                if row.id == id
                    && !workers.rows.contains_key(&id)
                    && !workers.by_key.contains_key(&row.facts.key) =>
            {
                workers.by_key.insert(row.facts.key.clone(), id);
                let probes = Probes::default();
                let slot = self.join(id);
                workers.rows.insert(id, Row { row, probes, slot });
                (change, None)
            }
            (
                MembershipChange::Relabelled { id } | MembershipChange::Resized { id },
                Some(mut row),
            ) if row.id == id => {
                let held = workers.rows.get_mut(&id);
                let Some(held) = held.filter(|held| held.row.facts.key == row.facts.key) else {
                    return;
                };
                let resized = held.row.facts.dp_size != row.facts.dp_size;
                carried = same_offer(&held.row.facts, &row.facts);
                row.admitted_at = held.row.admitted_at;
                let old = mem::replace(&mut held.row, row);
                // A row that carries its offer left no model.
                if !carried {
                    off = left_models(old.facts, &held.row.facts);
                    // As at the leader's own refresh: the build this
                    // change owes may be the first to list the member,
                    // so its slot is filed again for the leases' doors.
                    self.joined.lock().push((id, held.slot));
                }
                if resized {
                    (MembershipChange::Resized { id }, None)
                } else {
                    (MembershipChange::Relabelled { id }, None)
                }
            }
            (MembershipChange::Replaced { old, new }, Some(row))
                if row.id == new
                    && !workers.rows.contains_key(&new)
                    && workers
                        .rows
                        .get(&old)
                        .is_some_and(|held| held.row.facts.key == row.facts.key) =>
            {
                let left = workers.rows.remove(&old).map(|held| held.row);
                workers.by_key.insert(row.facts.key.clone(), new);
                let probes = Probes::default();
                let slot = self.join(new);
                workers.rows.insert(new, Row { row, probes, slot });
                self.forget_worker(old);
                (change, left)
            }
            (MembershipChange::Removed { id }, None) => {
                let Some(held) = workers.rows.remove(&id) else {
                    return;
                };
                workers.by_key.remove(&held.row.facts.key);
                self.forget_worker(id);
                (change, Some(held.row))
            }
            (MembershipChange::StatusChanged { id, from, to }, None) => {
                let held = workers.rows.get_mut(&id);
                let Some(held) = held.filter(|held| held.row.status == from) else {
                    return;
                };
                // The revision counts a row's stated facts: a status
                // change leaves it, here as in the leader's own write.
                held.row.status = to;
                (change, None)
            }
            _ => return,
        };
        workers.record_over(change, left.as_ref());
        // The build each write leaves, as the leader's own: a join, a
        // refresh and a patch owe it, and a leave pays its own before it is
        // told, a registration that takes its worker off a model among
        // them. What is owed, the replica's loop pays at the window's end.
        let now = Instant::now();
        match change {
            MembershipChange::Registered { id } => {
                // A row's load entry is made after its telling, as a
                // registration's is.
                self.loads.lock().ranks.insert(id, Ranks::default());
                self.index_member(&workers, id);
                let classes = Classes::Written {
                    left: None,
                    joined: Some(id),
                };
                self.owe(classes, now);
                self.tell_membership(change);
            }
            MembershipChange::Replaced { old, new } => {
                self.loads.lock().ranks.insert(new, Ranks::default());
                self.index_member(&workers, new);
                let classes = Classes::Written {
                    left: Some(old),
                    joined: Some(new),
                };
                // A replacement that takes its worker off a model says so
                // for the new id, behind the build it pays and behind its
                // own word, as at the leader's own write.
                let models = Self::replaced_models(left, workers.rows.get(&new));
                self.pay_leave(&workers, classes, now);
                self.tell_membership(change);
                self.tell_left(new, &models);
            }
            MembershipChange::Relabelled { id } | MembershipChange::Resized { id } => {
                self.index_member(&workers, id);
                let classes = if carried {
                    Classes::Carry
                } else {
                    Classes::Written {
                        left: Some(id),
                        joined: Some(id),
                    }
                };
                self.tell_leaving(&workers, classes, change, &off, now);
            }
            MembershipChange::StatusChanged { id, to, .. } => self.show_status(id, to),
            MembershipChange::Removed { id } => {
                let classes = Classes::Written {
                    left: Some(id),
                    joined: None,
                };
                self.pay_leave(&workers, classes, now);
                self.tell_membership(change);
            }
        }
    }

    /// One of the leader's load rows, applied as the replication stream
    /// tells it: [`LoadSink::record`] on a standby, journaled at `at`, the
    /// event's place, which the applier holds ([`Journal::load`]).
    pub fn follow_load(
        &self,
        at: Position,
        id: WorkerId,
        rank: u16,
        row: LoadRow,
        received_at: Instant,
        source: LoadSource,
    ) {
        self.record_at(Some(at), id, rank, row, received_at, source);
    }

    /// The start of a leader's term, applied as the replication stream
    /// tells it: it writes no row. The journal keeps the event's place,
    /// `at`, where the applier holds it ([`Journal::term_followed`]), and
    /// the store's term rises to the term's. The store took nothing: it
    /// leads no term by it, as it does by a take
    /// ([`LeasesStore::set_term`]).
    pub fn follow_term(&self, at: Position) {
        // Taken before the turns, so dropped after them, as at a take.
        let _released = self.released();
        let mut turns = self.every_turn();
        if let Some(journal) = &self.journal {
            journal.term_followed(at);
        }
        self.rise(&mut turns, at.term);
    }

    /// Apply a generation: every world from here on is published under
    /// `spec`, the first at once, with the standing membership, each row's
    /// engine family, zone and labels read under the generation's own
    /// lists. The store's one entry for a generation; a store starts under
    /// [`AppliedSpec::unbound`].
    ///
    /// The number only rises. A generation whose number is not above the
    /// standing one is refused by naming the standing one, and nothing is
    /// published: a reader that saw a generation never sees a lower one,
    /// and a document that names no generation (number 0) is never applied.
    ///
    /// The generation applied before the standing one is let go here, by
    /// the applying thread and outside the store's lock, and so is a
    /// generation the store refuses.
    ///
    /// The generation is the one writer of the mode the pair classes are
    /// derived under ([`AppliedSpec::pairing`]). A generation of another
    /// mode derives the standing membership's pools again and publishes
    /// them once, under the new generation: no member moved, so the
    /// store's epoch stays and no watcher is told. An apply takes the
    /// workers lock and then the load lock, as a membership write does: a
    /// writer's wait, never a decision's.
    ///
    /// The generation's engine is then told every member once, by id
    /// ([`Membership::Snapshot`]), under the workers lock and with the load
    /// lock let go: its stages start empty, and each membership write after
    /// it is told behind the same lock, in its order.
    pub fn apply(&self, spec: AppliedSpec) -> Result<(), StaleGeneration> {
        // The writer takes a second reference: when it refuses, the last
        // one is this one, let go after the lock.
        let spec = Arc::new(spec);
        let before = {
            let workers = self.workers.lock();
            let mut loads = self.loads.lock();
            let Loads { ranks, writer } = &mut *loads;
            let ranks = &*ranks;
            let (world, before) = writer.apply(
                Arc::clone(&spec),
                workers.epoch,
                workers.rows.values().map(|row| (&row.row, row.slot)),
                |id, rank| ranks.get(&id)?.samples.get(&rank).copied(),
            )?;
            writer.replaced(self.world.swap(world));
            // A generation of another pairing mode builds the membership
            // and lists every member. One of the standing mode publishes
            // the pools as they stood: the owed build is left to its
            // window, and the table with it.
            if !writer.owes() {
                self.listed();
            }
            drop(loads);
            // The pins' sweep learns a generation's interval here, not at
            // its first pass: a generation replaced between two ticks of
            // the sweep still leaves its interval for the pins it ended.
            // It is saved under the workers lock, so two applies save
            // theirs in the order of their publications.
            #[cfg(feature = "controller")]
            if let Some(every) = spec.engine().pins_shortest_sweep() {
                self.pins_swept.lock().every = Some(every);
            }
            self.tell_members(&workers);
            before
        };
        drop(before);
        Ok(())
    }

    /// The world as the last publication left it, taken through no lock.
    /// Its identity moves with every publication: [`World::epoch`] says
    /// whether the membership moved.
    ///
    /// This door hands an owned pointer, for a reader that keeps a world
    /// across calls. Taking it and dropping it each write the count every
    /// holder of that world shares, so deciding threads that took it would
    /// write one cache line twice a decision. A decision builds through
    /// [`build_view`](Self::build_view), a pair decision through
    /// [`build_pair`](Self::build_pair), and a decision that runs the
    /// generation's engine over its rows through
    /// [`deciding`](Self::deciding): all three borrow the world and write
    /// no shared count.
    ///
    /// A world kept across two publications of a pool keeps the rows it
    /// lists as they are: the second publication writes over other rows,
    /// and allocates that pool's rows when the writer holds none alone
    /// ([`publication_allocations`](Self::publication_allocations)).
    pub fn world(&self) -> Arc<World> {
        self.world.load_full()
    }

    /// Build one decision's view of the pool serving `model` in `role`
    /// ([`World::build`]) from the world of this instant.
    pub fn build_view(
        &self,
        model: &str,
        role: Role,
        ask: &Ask,
        out: &mut ViewBuffers,
    ) -> Option<Built> {
        self.world.load().build(model, role, ask, out)
    }

    /// Read the world of this instant under one load, held until `read`
    /// returns: the rows it takes ([`World::view`]), the generation and its
    /// engine ([`World::spec`]) are of one publication, and a decision made
    /// inside is made over that publication's rows where they stand. Like
    /// [`build_view`](Self::build_view) it borrows the world and writes no
    /// shared count. The load is held for as long as `read` runs: a reader
    /// that waits inside keeps the publication's rows from their next use.
    pub fn with_world<R>(&self, read: impl FnOnce(&World) -> R) -> R {
        let world = self.world.load();
        read(&world)
    }

    /// Build both sides of one pair decision from the world of this instant
    /// under one borrow ([`World::build_pair`]): the model's prefill pool,
    /// its decode pool and the table of their classes, all of one
    /// publication. `None`, with every buffer cleared, when either pool has
    /// no member: one side alone is no pair.
    ///
    /// This is a pair decision's door. It borrows the world, so the two
    /// builds and the table read one publication and no deciding thread
    /// writes the world's shared count.
    pub fn build_pair(
        &self,
        model: &str,
        ask: &Ask,
        out: &mut PairBuffers,
    ) -> Option<(Built, Built)> {
        self.world.load().build_pair(model, ask, out)
    }

    /// Build one decision's pair and its encode pool from the world as it
    /// is now, under one borrow ([`World::build_encode_pair`]): the pair's
    /// two pools and table, the model's encode pool and the two columns the
    /// encode side reads, all of one publication. `None`, with every buffer
    /// cleared, when either side of the pair has no member.
    ///
    /// This is the door of a request that carries encode items. Like the
    /// pair's door it borrows the world for the builds and writes no shared
    /// count.
    pub fn build_encode_pair(
        &self,
        model: &str,
        ask: &Ask,
        out: &mut EncodePairBuffers,
    ) -> Option<(Built, Built)> {
        self.world.load().build_encode_pair(model, ask, out)
    }

    /// One decision's door: `decide` is handed the world of this instant
    /// under one borrow, for the build of its view ([`World::build`]) and
    /// for the engine of the generation those rows were published under
    /// ([`World::spec`]). The rows, the engine and the generation's number
    /// are then of one load, and the borrow writes no shared count, so no
    /// two deciding threads write one line through it. A publication that
    /// replaces the world meanwhile is the next decision's: the world
    /// handed here stays whole until `decide` returns.
    pub fn deciding<R>(&self, decide: impl FnOnce(&World) -> R) -> R {
        decide(&self.world.load())
    }

    /// `at` in the rows' clock: microseconds since the store's origin.
    pub fn micros(&self, at: Instant) -> u64 {
        micros(self.origin, at)
    }

    /// Build the pools of the membership as it stands and show them in one
    /// world, now, whatever the window: nothing is left owed. What a write
    /// that replaces every row does, which tells the engine in other words
    /// than one change's ([`MemoryStore::restore_from`]). Called under the
    /// workers lock; it takes the load lock and lets it go at its return.
    fn swap_world(&self, workers: &Workers, classes: Classes) {
        let mut loads = self.loads.lock();
        let Loads { ranks, writer } = &mut *loads;
        let ranks = &*ranks;
        let world = writer.rebuild(
            workers.epoch,
            workers.rows.values().map(|row| (&row.row, row.slot)),
            |id, rank| ranks.get(&id)?.samples.get(&rank).copied(),
            classes,
        );
        writer.replaced(self.world.swap(world));
        // The build listed every member: none is left for the table.
        self.listed();
    }

    /// Tell the standing generation's engine every member once, by id
    /// ([`Membership::Snapshot`]): what an apply tells the generation it
    /// applied and a restore the one that stands. Called under the workers
    /// lock, after the world of those members is shown and with the load
    /// lock let go.
    fn tell_members(&self, workers: &Workers) {
        let mut members: Vec<WorkerId> = workers.rows.keys().copied().collect();
        members.sort_unstable();
        let told = Membership::Snapshot(&members);
        self.tell_engine(&told);
    }

    /// Tell the engine of the world the store shows one membership word:
    /// under the workers lock, with the load lock let go.
    fn tell_engine(&self, word: &Membership<'_>) {
        self.world.load().spec().engine().on_membership(word);
    }

    /// Tell the standing generation's engine what `change` is to it
    /// ([`MembershipChange::as_lifecycle`]). Called by every membership
    /// write under the workers lock, after the write left its build owed or
    /// paid its leave's, and with the load lock let go: the engine hears the
    /// writes in their order, a load write under the load lock alone waits
    /// for no stage, and an apply, which takes the workers lock, moves no
    /// generation in between. A leave is told after the build that takes its
    /// rows out of their pools: the worker is in no pool when its write
    /// returns, whoever wrote. A join is told when the store takes it, and
    /// the world that lists its row follows at its window's build. A
    /// decision takes neither lock and meets the call at a stage's own lock
    /// alone. One that built its view before a leave's swap may still name
    /// the worker: a stage answers that from its own state, as the ring
    /// fences a pruned id ([`crate::ring::FENCE`]).
    fn tell_membership(&self, change: MembershipChange) {
        if let Some(change) = change.as_lifecycle() {
            self.tell_engine(&change);
        }
    }

    /// [`MemoryStore::tell_membership`] for a registration that stays under
    /// its id, with `left`, the models it takes its worker off
    /// ([`left_models`]). Where it takes the worker off none, the build is
    /// owed to the window as a join's is. Where it does, the write is a
    /// leave from those models' pools and pays its own build first, as a
    /// removal does ([`MemoryStore::pay_leave`]): the worker is in none of
    /// them when the engine hears [`Membership::Left`], said ahead of the
    /// write's own word.
    fn tell_leaving(
        &self,
        workers: &Workers,
        classes: Classes,
        change: MembershipChange,
        left: &[String],
        now: Instant,
    ) {
        if left.is_empty() {
            self.owe(classes, now);
        } else {
            self.pay_leave(workers, classes, now);
            let id = change.id();
            self.tell_engine(&Membership::Left { id, models: left });
        }
        self.tell_membership(change);
    }

    /// Tell the engine the models a replaced incarnation left, for the id
    /// that replaced it, and no word where it left none. Said behind the
    /// build the replacement pays and behind its own word: a stage has
    /// retired the old id and carried its places to the new one by then.
    fn tell_left(&self, id: WorkerId, models: &[String]) {
        if !models.is_empty() {
            self.tell_engine(&Membership::Left { id, models });
        }
    }

    /// The models a replaced incarnation stood in a pool of and its
    /// successor does not ([`left_models`]), from the row a replacement
    /// took out and the row it put in: none where either is not there.
    fn replaced_models(old: Option<WorkerRow>, new: Option<&Row>) -> Vec<String> {
        match (old, new) {
            (Some(old), Some(new)) => left_models(old.facts, &new.row.facts),
            _ => Vec::new(),
        }
    }

    /// A membership write left the pools to be built again. Called under
    /// the workers lock by every membership write, with what the write asks
    /// of the pair classes. The build is owed until the store's window ends
    /// and takes in every write since the last: a storm of registrations
    /// builds once a window, never once a write. A poll's answer that
    /// needs a worker's row asks for the build sooner
    /// ([`Self::publish_now`]): registrations each followed by one inside
    /// their window build once a registration. A lease asks for none: it
    /// books the slot the member's write gave it ([`Self::join`]).
    fn owe(&self, classes: Classes, at: Instant) {
        let first = self.loads.lock().writer.owe(classes, at);
        self.wake_host(first);
    }

    /// Have the host read the due instant again where a write left the
    /// `first` build owed: its loop sleeps to an instant it read before.
    /// Called with the load lock released, so the host's read does not wait
    /// for this writer.
    fn wake_host(&self, first: bool) {
        if first {
            if let Some(wake) = &*self.wake.lock() {
                wake();
            }
        }
    }

    /// Pay the build the membership writes left owed: the pools of the
    /// membership as it stands, each row with its load record on file, in
    /// one world. Called under the workers lock; it takes the load lock, as
    /// a membership write does, so no record is lost across the swap. The
    /// host's call names its instant, and a build is paid when it is due at
    /// it; where none is, what waits for the window is published by the
    /// window's own rule. The ask names no instant and is paid at once.
    fn pay(&self, workers: &Workers, at: Option<Instant>) {
        let mut loads = self.loads.lock();
        let Loads { ranks, writer } = &mut *loads;
        let ranks = &*ranks;
        let (world, built) = match at {
            Some(now) if !writer.build_due(now) => (writer.publish_if_due(now), false),
            _ => {
                let world = writer.build_owed(
                    at,
                    workers.epoch,
                    workers.rows.values().map(|row| (&row.row, row.slot)),
                    |id, rank| ranks.get(&id)?.samples.get(&rank).copied(),
                );
                let built = world.is_some();
                (world, built)
            }
        };
        self.show(writer, world);
        // A membership build lists every member: the table of the ones no
        // world listed is emptied behind its publication.
        if built {
            self.listed();
        }
    }

    /// A leave pays its own build. A removal, a replaced incarnation and a
    /// registration that takes its worker off a model owe the build as
    /// every membership write does and pay it in the same write, as the ask
    /// pays it, with every write the window held. Called under the workers
    /// lock, before the leave is told: the worker is in no pool it left
    /// when its write returns, under every profile and stage and whoever
    /// wrote, a served call or a followed change. A storm of removals so
    /// builds once a removal. The window stands as it was, and the host is
    /// not woken: nothing is left owed.
    fn pay_leave(&self, workers: &Workers, classes: Classes, at: Instant) {
        self.loads.lock().writer.owe(classes, at);
        self.pay(workers, None);
    }

    /// Build what the membership writes left owed, now, whatever the
    /// window. Where nothing is owed, nothing is done.
    ///
    /// The build of a registration, a refresh and a patch starts within one
    /// publication window of their answer, when the store's host pays the
    /// due instant ([`LoadSink::publish_due`]), and they are in force for
    /// decisions when it ends; a registration that takes its worker off a
    /// model has paid its own. A caller that needs its own write in force
    /// at once asks here: a test, a bench's fill, a host with no loop. The
    /// store asks for itself where a write is of a worker's row: a poll's
    /// answer for a rank of it. The ask builds the whole membership under
    /// the workers lock and the load lock: a writer's wait, never a
    /// decision's and never a lease's.
    pub fn publish_now(&self) {
        let workers = self.workers.lock();
        self.pay(&workers, None);
    }

    /// Show the deciders the world a write made, when it made one. Called
    /// with the load lock held: the worlds reach the cell in the order the
    /// writer made them, and the writer takes back the one each replaces.
    fn show(&self, writer: &mut Writer, world: Option<Arc<World>>) {
        if let Some(world) = world {
            writer.replaced(self.world.swap(world));
        }
    }

    /// Fold a status change into the world and publish it at once. Called
    /// under the workers lock, as the status was written.
    fn show_status(&self, id: WorkerId, status: WorkerStatus) {
        let mut loads = self.loads.lock();
        let world = loads.writer.fold_status(id, status);
        self.show(&mut loads.writer, world);
    }

    /// Publish `gateway`'s veto on `id`, or its clearing, at once. Called
    /// under the health lock, which serializes the vetoes' writers, so the
    /// published vetoes follow the health table's own order.
    fn show_veto(&self, gateway: GatewayId, id: WorkerId, until: Option<Instant>) {
        let mut loads = self.loads.lock();
        let world = loads.writer.veto(gateway, id, until);
        self.show(&mut loads.writer, world);
    }

    /// Publish the lifting of every veto of the gateways in `gone` at once,
    /// in one world. Called under the health lock, as a veto's own write is.
    fn show_gateways_gone(&self, gone: &[GatewayId]) {
        let mut loads = self.loads.lock();
        let world = loads.writer.drop_gateways(gone);
        self.show(&mut loads.writer, world);
    }

    /// How many entries the health tables hold, over every gateway. A
    /// gateway's table that holds none counts as one: a table left behind
    /// shows.
    #[cfg(test)]
    pub(crate) fn health_entries(&self) -> usize {
        let health = self.health.lock();
        let completions = health.ok_at.values().map(|workers| workers.len().max(1));
        health.vetoes.len() + completions.sum::<usize>()
    }

    /// The counts one scrape reports. The load table's two are read under
    /// one hold of the load lock, so they are of one state of the store.
    pub fn counts(&self) -> StoreCounts {
        let loads = self.loads.lock();
        StoreCounts {
            outside: loads.writer.outside(),
            overload_flips: loads.writer.flips(),
            lifecycle_faults: self.lifecycle_faults(),
        }
    }

    /// How many writes to the load table moved an overload verdict: a
    /// record, a report, the word that no report came, an eviction. A
    /// membership build and an applied generation judge again and count
    /// nothing.
    pub fn overload_flips(&self) -> u64 {
        self.counts().overload_flips
    }

    /// How many members of the standing membership name an engine family
    /// or a zone that the standing generation's list does not hold, by the
    /// list. Such a worker is admitted and reads position 0 in that field;
    /// a generation that declares no list counts none.
    pub fn outside_lists(&self) -> OutsideLists {
        self.counts().outside
    }

    /// How many workers the load table holds an entry for.
    #[cfg(test)]
    pub(super) fn load_entries(&self) -> usize {
        self.loads.lock().ranks.len()
    }

    /// How many ranks stand unnamed by their engine's last answer.
    #[cfg(test)]
    pub(super) fn unnamed_ranks(&self) -> usize {
        self.loads.lock().writer.unnamed_ranks()
    }

    /// Whether a load write is judged: an engine has a threshold, or a rank
    /// stands unnamed.
    #[cfg(test)]
    pub(super) fn judges_writes(&self) -> bool {
        self.loads.lock().writer.judges_writes()
    }

    /// How many rows that left their pool the writer still keeps.
    #[cfg(test)]
    pub(super) fn held_rows(&self) -> usize {
        self.loads.lock().writer.held_rows()
    }

    /// How many members the table of the ones no world lists holds.
    #[cfg(test)]
    pub(super) fn filed_members(&self) -> usize {
        self.joined.lock().len()
    }

    /// How many whole derivations made what the writer keeps of `model`'s
    /// pair classes; `None` where it keeps nothing of them.
    #[cfg(test)]
    pub(super) fn whole_derivations(&self, model: &str) -> Option<usize> {
        self.loads.lock().writer.whole_derivations(model)
    }

    /// Whether no lock a journaled write takes is held at this instant:
    /// the membership lock, the turn of the leases' file and the load lock,
    /// each tried.
    #[cfg(test)]
    pub(crate) fn unlocked(&self) -> bool {
        self.workers.try_lock().is_some()
            && self.files.iter().all(turn::File::free)
            && self.loads.try_lock().is_some()
    }

    /// Whether the membership lock is free at this instant, tried: a stage
    /// told a membership write finds it held.
    #[cfg(all(test, feature = "controller"))]
    pub(crate) fn workers_lock_free(&self) -> bool {
        self.workers.try_lock().is_some()
    }

    /// Whether the load lock is free at this instant, tried: a stage told a
    /// membership write finds it free.
    #[cfg(all(test, feature = "controller"))]
    pub(crate) fn load_lock_free(&self) -> bool {
        self.loads.try_lock().is_some()
    }

    /// The KV tables, for the index's cases.
    #[cfg(test)]
    pub(super) fn kv(&self) -> &KvTables {
        &self.kv
    }

    /// Sets a row's probe counters: no case reaches a counter's end by
    /// probes.
    #[cfg(test)]
    pub(super) fn probed(&self, id: WorkerId, failures: u32, successes: u32, pending_total: u32) {
        if let Some(row) = self.workers.lock().rows.get_mut(&id) {
            row.probes = Probes {
                failures,
                successes,
                pending_total,
            };
        }
    }

    /// A worker's row is gone (removed, or replaced by a new incarnation):
    /// the store's tables that live exactly as long as the row leave with
    /// it, here and nowhere else: the health tables, the load rows and the
    /// worker's KV lane with its blocks in the index. The vetoes a world
    /// carries are not among them: they leave with the row in the
    /// publication of the membership write that calls this. Nor are the
    /// sessions' pins of record that name the worker: the leader's own
    /// removal or replacement drops them once it let its lock go
    /// ([`Self::forget_pins`]), and a row that leaves by a followed change
    /// or a restore keeps them. A standby is sent no pin, so those are of
    /// a term this store led itself: no decision takes one while its
    /// worker is gone, and a worker that returns under the same key and
    /// incarnation is pinned by them again. Called under the workers
    /// lock:
    /// the health writers take it first and a load write goes into its
    /// worker's entry or nowhere, so a write racing the removal finds the
    /// row gone or lands before the tables are cleared.
    fn forget_worker(&self, id: WorkerId) {
        self.kv.forget(id);
        let mut health = self.health.lock();
        health.forget(id);
        let mut loads = self.loads.lock();
        loads.ranks.remove(&id);
    }

    /// The sessions' pins of record that name a worker whose row is gone
    /// leave here, once the membership write let the workers lock go: the
    /// walk is as long as the pins of record, and no writer of the
    /// membership or of a worker's health waits on it. A pin a dispatch
    /// writes past the walk names a row that is gone and is taken by no
    /// decision; where its rule has a TTL, the pins' sweep ends it. A
    /// worker's id is its key and incarnation: where the same incarnation
    /// registers again before the walk ends, a pin a dispatch writes for
    /// it in a shard the walk has not reached leaves though its row
    /// stands, until the key's next dispatch writes it again.
    fn forget_pins(&self, id: WorkerId) {
        SessionsStore::drop_worker(self, id);
    }

    /// How many times a publication allocated a pool's rows since the store
    /// was made. A publication writes a pool's rows over rows no world
    /// lists any more: the ones the pool published before its last, else
    /// any of its length that left a pool. It allocates when it finds
    /// none: at a pool's first publication after a membership build,
    /// unless rows that build retired fit it, and past that only while a
    /// reader holds a world from before the pool's last publication. A
    /// membership build's own rows are not counted.
    pub fn publication_allocations(&self) -> u64 {
        self.loads.lock().writer.allocations()
    }

    /// How many worlds the store published since it was made. A write that
    /// changes nothing publishes none, and the gateways one sweep lets go
    /// leave in one.
    pub fn publications(&self) -> u64 {
        self.loads.lock().writer.publications()
    }

    /// How many times the pools were built from the membership since the
    /// store was made. A storm of membership writes builds once a window,
    /// never once a write, and once more where a poll's answer asks for
    /// the build: the count says so by run.
    pub fn membership_builds(&self) -> u64 {
        self.loads.lock().writer.builds()
    }

    /// Give the row of `id` its KV lane as its facts ask
    /// ([`KvTables::member`]). Called under the workers lock by every write
    /// that makes a row or changes its facts.
    fn index_member(&self, workers: &Workers, id: WorkerId) {
        if let Some(row) = workers.rows.get(&id) {
            self.kv.member(&row.row);
        }
    }

    /// How many resyncs were recorded for `reason`.
    pub fn resyncs(&self, reason: ResyncReason) -> u64 {
        self.kv.resyncs(reason)
    }

    /// Blocks the index holds for the row of `id` at `rank`; `0` for a row
    /// the store does not index.
    pub fn indexed_blocks(&self, id: WorkerId, rank: u16) -> usize {
        self.kv.indexed_blocks(id, rank)
    }

    /// What the lane of `id` at `rank` counted beside its blocks; `None`
    /// for a row the store does not index.
    pub fn lane_counts(&self, id: WorkerId, rank: u16) -> Option<LaneCounts> {
        self.kv.counts(id, rank)
    }

    /// Rows that ask for events and hold no lane. Each was written while
    /// its model's index stood at its ceiling
    /// ([`INDEX_WORKERS`](crate::store::INDEX_WORKERS)): it holds nothing
    /// in the index until one of its own later writes finds a place.
    pub fn unindexed_rows(&self) -> usize {
        let workers = self.workers.lock();
        let asking = workers
            .rows
            .values()
            .filter(|row| indexed_under(&row.row).is_some())
            .count();
        asking.saturating_sub(self.kv.lanes())
    }
}

// ---------------------------------------------------------------- workers

#[derive(Default)]
struct Workers {
    rows: FxHashMap<WorkerId, Row>,
    by_key: HashMap<String, WorkerId>,
    epoch: u64,
    log: VecDeque<(u64, MembershipChange)>,
    /// The store's journal: every membership and status write passes
    /// [`Writing::record`], which tells it.
    journal: Option<Arc<dyn Journal>>,
}

struct Row {
    row: WorkerRow,
    probes: Probes,
    /// The slot of the worker's count of the leases on file, given at the
    /// write that first files it ([`MemoryStore::join`]). A member keeps
    /// it across builds, and the slot of a worker that left is given to no
    /// one: the leases still on file for it end on the count they booked.
    slot: Slot,
}

/// The prober's counters for one row.
#[derive(Default)]
struct Probes {
    failures: u32,
    successes: u32,
    pending_total: u32,
}

impl Workers {
    /// The row under `id` when `revision` is still its revision.
    fn guarded(&mut self, id: WorkerId, revision: u64) -> Option<&mut Row> {
        self.rows
            .get_mut(&id)
            .filter(|row| row.row.revision == revision)
    }

    /// What a health write says where [`Self::guarded`] gave no row: stale
    /// where a row stands under another revision, no row where none stands.
    /// Either way the write was judged against nothing.
    fn unjudged(&self, id: WorkerId) -> HealthWrite {
        if self.rows.contains_key(&id) {
            HealthWrite::Stale
        } else {
            HealthWrite::NoRow
        }
    }
}

/// Tells the journal, at its drop, that a write's locks are let go
/// ([`Journal::released`]). A write takes it before its first lock: what a
/// scope takes first it drops last.
struct Released<'a>(Option<&'a dyn Journal>);

impl Drop for Released<'_> {
    fn drop(&mut self) {
        if let Some(journal) = self.0 {
            journal.released();
        }
    }
}

/// The membership lock as a journaled write holds it. The fields drop in
/// their order: the lock, then the word to the journal that the write's
/// locks are let go. What the journal wakes for a write, it wakes behind
/// no writer and no reader of the rows.
struct Writing<'a> {
    workers: MutexGuard<'a, Workers>,
    /// The place of the leader's event the write applies, for the journal:
    /// none for a write of the store's own.
    at: Option<Position>,
    _released: Released<'a>,
}

impl Deref for Writing<'_> {
    type Target = Workers;

    fn deref(&self) -> &Workers {
        &self.workers
    }
}

impl DerefMut for Writing<'_> {
    fn deref_mut(&mut self) -> &mut Workers {
        &mut self.workers
    }
}

/// A change is recorded through the write's guard alone: a path that takes
/// the membership lock bare has no way to journal a write, so none is
/// journaled whose release the journal does not hear.
impl Writing<'_> {
    fn record(&mut self, change: MembershipChange) -> MembershipChange {
        self.record_over(change, None)
    }

    /// [`Writing::record`] for a write that took a row out of the table (a
    /// removal, a replacement): `left` is that row, for the journal.
    fn record_over(
        &mut self,
        change: MembershipChange,
        left: Option<&WorkerRow>,
    ) -> MembershipChange {
        let workers = &mut *self.workers;
        workers.epoch += 1;
        if workers.log.len() == CHANGE_LOG {
            workers.log.pop_front();
        }
        workers.log.push_back((workers.epoch, change));
        if let Some(journal) = &workers.journal {
            let row = workers.rows.get(&change.id()).map(|row| &row.row);
            journal.member(self.at, workers.epoch, change, row, left);
        }
        change
    }

    fn set_status(&mut self, id: WorkerId, status: WorkerStatus) -> Option<MembershipChange> {
        let row = self.workers.rows.get_mut(&id)?;
        let from = row.row.status;
        if from == status {
            return None;
        }
        // The revision counts the row's stated facts: a status change
        // leaves it, so the holder of a row's revision stays current while
        // the worker's health moves.
        row.row.status = status;
        Some(self.record(MembershipChange::StatusChanged {
            id,
            from,
            to: status,
        }))
    }
}

impl MemoryStore {
    /// The word to the journal, at its drop, that a write's locks are let
    /// go.
    fn released(&self) -> Released<'_> {
        Released(self.journal.as_deref())
    }

    /// The membership lock for a write the journal is told: every path
    /// that records a change takes the lock here.
    fn writing(&self) -> Writing<'_> {
        self.writing_at(None)
    }

    /// [`MemoryStore::writing`] for a write the journal is told with `at`:
    /// the place of the leader's event a standby's applier applies by it.
    fn writing_at(&self, at: Option<Position>) -> Writing<'_> {
        let released = self.released();
        Writing {
            workers: self.workers.lock(),
            at,
            _released: released,
        }
    }
}

impl WorkersStore for MemoryStore {
    fn upsert(&self, mut facts: WorkerFacts) -> MembershipChange {
        // A blank value of a descriptor or of an encode offer is no report:
        // the row of record holds what the two rules read. A descriptor that
        // stands so is kept where it is, shared or not: a row written again
        // copies none.
        if let Some(descriptor) = facts
            .pairing
            .as_mut()
            .filter(|descriptor| !descriptor.is_normalized())
        {
            Arc::make_mut(descriptor).normalize();
        }
        facts.encode = facts.encode.normalized();
        let mut workers = self.writing();
        let now = Instant::now();
        if let Some(old_id) = workers.by_key.get(&facts.key).copied() {
            let same_incarnation = workers
                .rows
                .get(&old_id)
                .is_some_and(|row| row.row.facts.incarnation == facts.incarnation);
            if same_incarnation {
                let Some(row) = workers.rows.get_mut(&old_id) else {
                    return MembershipChange::Removed { id: old_id };
                };
                let resized = row.row.facts.dp_size != facts.dp_size;
                // A registration refreshed with the offer it had moves no
                // pair class and takes its worker off no model: the role and
                // the models are the offer's. Another names the models it
                // took the worker off, for the stages to hear ahead of the
                // write's own word. The facts it replaces end here either
                // way, ahead of the write's world.
                let (classes, left) = if same_offer(&row.row.facts, &facts) {
                    row.row.facts = facts;
                    (Classes::Carry, Vec::new())
                } else {
                    let old = mem::replace(&mut row.row.facts, facts);
                    // The build this write owes may be the first to list
                    // the member: one in no pool stands in no world, and
                    // the build behind its first write emptied its entry.
                    // Its slot is filed again for the leases' doors; a
                    // twin of an entry on file reads the same slot.
                    self.joined.lock().push((old_id, row.slot));
                    let classes = Classes::Written {
                        left: Some(old_id),
                        joined: Some(old_id),
                    };
                    (classes, left_models(old, &row.row.facts))
                };
                row.row.revision += 1;
                let change = workers.record(if resized {
                    MembershipChange::Resized { id: old_id }
                } else {
                    MembershipChange::Relabelled { id: old_id }
                });
                self.index_member(&workers, old_id);
                self.tell_leaving(&workers, classes, change, &left, now);
                return change;
            }
            // A new incarnation under the key: the old id leaves with its
            // counters; its status carries over when the registration
            // matches (role, models and wires), as the registry inherits a
            // replaced worker's runtime state today. A drain is of the
            // registration it named and ends with it: a draining
            // incarnation hands no status on.
            let old = workers.rows.remove(&old_id);
            let new_id = identity(&facts.key, facts.incarnation);
            let status = old
                .as_ref()
                .filter(|old| {
                    old.row.status != WorkerStatus::Draining
                        && old.row.facts.role == facts.role
                        && old.row.facts.same_models(&facts)
                        && old.row.facts.wire_modes == facts.wire_modes
                })
                .map_or(WorkerStatus::Pending, |old| old.row.status);
            workers.by_key.insert(facts.key.clone(), new_id);
            workers.rows.insert(
                new_id,
                Row {
                    row: WorkerRow {
                        id: new_id,
                        facts,
                        status,
                        revision: 1,
                        admitted_at: now,
                    },
                    probes: Probes::default(),
                    slot: self.join(new_id),
                },
            );
            // The old incarnation's tables leave with its row, as on a
            // removal, under the same lock and before the replacement is
            // told: a veto, a completed dispatch or a load row keyed by the
            // old id would outlive it for good, and no load of the old
            // incarnation stands after the replacement in the journal.
            self.forget_worker(old_id);
            let change = workers.record_over(
                MembershipChange::Replaced {
                    old: old_id,
                    new: new_id,
                },
                old.as_ref().map(|old| &old.row),
            );
            // The new incarnation starts with an empty load entry, made
            // after the telling: nothing the old one reported is the new
            // one's, and no load of it stands before the replacement.
            self.loads.lock().ranks.insert(new_id, Ranks::default());
            self.index_member(&workers, new_id);
            let classes = Classes::Written {
                left: Some(old_id),
                joined: Some(new_id),
            };
            // The models the new incarnation no longer stands in a pool of:
            // the stages hear them for the new id, behind the build the
            // replacement pays and behind its own word, which retires the
            // old id and carries its places first.
            let left = Self::replaced_models(old.map(|old| old.row), workers.rows.get(&new_id));
            self.pay_leave(&workers, classes, now);
            self.tell_membership(change);
            self.tell_left(new_id, &left);
            drop(workers);
            self.forget_pins(old_id);
            return change;
        }
        let id = identity(&facts.key, facts.incarnation);
        workers.by_key.insert(facts.key.clone(), id);
        workers.rows.insert(
            id,
            Row {
                row: WorkerRow {
                    id,
                    facts,
                    status: WorkerStatus::Pending,
                    revision: 1,
                    admitted_at: now,
                },
                probes: Probes::default(),
                slot: self.join(id),
            },
        );
        // Recorded before the row's load entry exists: no load of a worker
        // stands before its registration in the journal.
        let change = workers.record(MembershipChange::Registered { id });
        // The row's load entry, created with it: what a member's load
        // writes go into.
        let first = {
            let mut loads = self.loads.lock();
            loads.ranks.insert(id, Ranks::default());
            let classes = Classes::Written {
                left: None,
                joined: Some(id),
            };
            loads.writer.owe(classes, now)
        };
        self.wake_host(first);
        self.index_member(&workers, id);
        self.tell_membership(change);
        change
    }

    fn patch(&self, id: WorkerId, patch: WorkerPatch, revision: u64) -> Option<MembershipChange> {
        let mut workers = self.writing();
        let row = workers.guarded(id, revision)?;
        // Every field is named: one more stops the build here until the
        // carry below is asked of it.
        let WorkerPatch {
            labels,
            priority,
            overload,
        } = patch;
        if let Some(labels) = labels {
            row.row.facts.labels = labels;
        }
        if let Some(priority) = priority {
            row.row.facts.priority = priority;
        }
        if let Some(overload) = overload {
            row.row.facts.overload = overload;
        }
        row.row.revision += 1;
        let change = workers.record(MembershipChange::Relabelled { id });
        // Labels, a priority and an overload are no part of an offer.
        self.owe(Classes::Carry, Instant::now());
        self.tell_membership(change);
        Some(change)
    }

    fn remove(&self, id: WorkerId, revision: u64) -> Option<MembershipChange> {
        let mut workers = self.writing();
        workers.guarded(id, revision)?;
        let row = workers.rows.remove(&id)?;
        workers.by_key.remove(&row.row.facts.key);
        // The worker's tables leave with its row, under the workers lock
        // and before the removal is told: a veto, a completed dispatch or a
        // load row keyed by an id that is gone would stay for good, the next
        // incarnation having a new id, and no load of the worker stands
        // after its removal in the journal. A poll or a record that lands
        // later finds no load entry and writes nothing.
        self.forget_worker(id);
        let change = workers.record_over(MembershipChange::Removed { id }, Some(&row.row));
        let classes = Classes::Written {
            left: Some(id),
            joined: None,
        };
        self.pay_leave(&workers, classes, Instant::now());
        self.tell_membership(change);
        drop(workers);
        self.forget_pins(id);
        Some(change)
    }

    fn set_status(
        &self,
        id: WorkerId,
        status: WorkerStatus,
        revision: u64,
    ) -> Option<MembershipChange> {
        let mut workers = self.writing();
        workers.guarded(id, revision)?;
        let change = workers.set_status(id, status)?;
        self.show_status(id, status);
        Some(change)
    }

    fn get(&self, id: WorkerId) -> Option<WorkerRow> {
        self.workers.lock().rows.get(&id).map(|row| row.row.clone())
    }

    fn by_key(&self, key: &str) -> Option<WorkerRow> {
        let workers = self.workers.lock();
        let id = workers.by_key.get(key)?;
        workers.rows.get(id).map(|row| row.row.clone())
    }

    fn epoch(&self) -> u64 {
        self.workers.lock().epoch
    }

    fn snapshot(&self) -> (u64, Vec<WorkerRow>) {
        let workers = self.workers.lock();
        let mut rows: Vec<WorkerRow> = workers.rows.values().map(|row| row.row.clone()).collect();
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        (workers.epoch, rows)
    }

    fn changes_since(&self, epoch: u64) -> MembershipTail {
        let workers = self.workers.lock();
        if epoch > workers.epoch {
            return MembershipTail::SnapshotRequired;
        }
        if epoch == workers.epoch {
            return MembershipTail::Changes(Vec::new());
        }
        match workers.log.front() {
            // The change after `epoch` is still in the log.
            Some(&(first, _)) if first <= epoch + 1 => MembershipTail::Changes(
                workers
                    .log
                    .iter()
                    .filter(|(at, _)| *at > epoch)
                    .copied()
                    .collect(),
            ),
            _ => MembershipTail::SnapshotRequired,
        }
    }
}

// ----------------------------------------------------------------- health

#[derive(Default)]
struct HealthTables {
    /// A gateway's veto on a worker, until the instant.
    vetoes: FxHashMap<(GatewayId, WorkerId), Instant>,
    /// The latest completed dispatch of class `Ok` on a worker by each
    /// gateway: the precedence rule is per gateway. One table a gateway:
    /// a gateway's end lets its own go whole and walks no other's.
    ok_at: FxHashMap<GatewayId, FxHashMap<WorkerId, Instant>>,
}

impl HealthTables {
    /// Drop every gateway's entries on the worker; a gateway's table that
    /// holds none after it goes too.
    fn forget(&mut self, id: WorkerId) {
        self.vetoes.retain(|(_, worker), _| *worker != id);
        self.ok_at.retain(|_, workers| {
            workers.remove(&id);
            !workers.is_empty()
        });
    }

    /// Drop the entries of every gateway in `gone`, a sorted list, on every
    /// worker: one pass over the vetoes for them all.
    fn forget_gateways(&mut self, gone: &[GatewayId]) {
        self.vetoes
            .retain(|(owner, _), _| gone.binary_search(owner).is_err());
        for gateway in gone {
            self.ok_at.remove(gateway);
        }
    }
}

/// The status machine of one probe over the row's counters: the registry's
/// `compute_next_status`, with the counters here instead of on the handle.
fn next_status(row: &mut Row, ok: bool) -> Option<WorkerStatus> {
    let policy = row.row.facts.health;
    let status = row.row.status;
    let failure = policy.failure_threshold.max(1);
    let success = policy.success_threshold.max(1);
    // A threshold is any number its writer took: the products and the
    // counters stop at their range's end, so a threshold whose product
    // passes the range is never met before the counter's end.
    let liveness = failure.saturating_mul(3);
    let pending_cap = failure.saturating_mul(10);
    let probes = &mut row.probes;
    if ok {
        probes.failures = 0;
        probes.successes = probes.successes.saturating_add(1);
        probes.pending_total = probes.pending_total.saturating_add(1);
        if matches!(
            status,
            WorkerStatus::Pending | WorkerStatus::NotReady | WorkerStatus::Failed
        ) && probes.successes >= success
        {
            probes.successes = 0;
            probes.pending_total = 0;
            return Some(WorkerStatus::Ready);
        }
        if status == WorkerStatus::Pending && probes.pending_total >= pending_cap {
            probes.successes = 0;
            return Some(WorkerStatus::Failed);
        }
        return None;
    }
    probes.successes = 0;
    probes.failures = probes.failures.saturating_add(1);
    probes.pending_total = probes.pending_total.saturating_add(1);
    match status {
        WorkerStatus::Ready if probes.failures >= failure => {
            probes.failures = 0;
            Some(WorkerStatus::NotReady)
        }
        WorkerStatus::NotReady if probes.failures >= liveness => {
            probes.failures = 0;
            Some(WorkerStatus::Failed)
        }
        WorkerStatus::Pending if probes.pending_total >= pending_cap => {
            probes.failures = 0;
            Some(WorkerStatus::Failed)
        }
        _ => None,
    }
}

impl HealthSink for MemoryStore {
    fn probe_outcome(&self, id: WorkerId, ok: bool, revision: u64) -> HealthWrite {
        let mut workers = self.writing();
        let Some(row) = workers.guarded(id, revision) else {
            return workers.unjudged(id);
        };
        let from = row.row.status;
        let Some(to) = next_status(row, ok) else {
            return HealthWrite::Stood(from);
        };
        // The table refuses a status write for an absent row or for the
        // status that stands. The row stands under this lock and the
        // machine never names the status it stands in, so no write reaches
        // this arm; it answers what stands.
        if workers.set_status(id, to).is_none() {
            return HealthWrite::Stood(from);
        }
        self.show_status(id, to);
        HealthWrite::Moved { from, to }
    }

    fn connected(&self, id: WorkerId, revision: u64) -> HealthWrite {
        let mut workers = self.writing();
        let Some(row) = workers.guarded(id, revision) else {
            return workers.unjudged(id);
        };
        let from = row.row.status;
        if matches!(from, WorkerStatus::Draining | WorkerStatus::Ready) {
            return HealthWrite::Stood(from);
        }
        row.probes = Probes::default();
        // No write reaches this arm either, as at a probe's outcome: the
        // row stands under this lock, and it is not ready.
        if workers.set_status(id, WorkerStatus::Ready).is_none() {
            return HealthWrite::Stood(from);
        }
        self.show_status(id, WorkerStatus::Ready);
        HealthWrite::Moved {
            from,
            to: WorkerStatus::Ready,
        }
    }

    fn gateway_veto(&self, gateway: GatewayId, id: WorkerId, until: Option<Instant>) {
        match until {
            Some(until) => {
                // Under the workers lock, taken before the health lock as
                // every writer takes it: a veto landing after the worker's
                // removal (a breaker opening on a torn-down worker's
                // failures) would re-create its entry for good.
                let workers = self.workers.lock();
                if !workers.rows.contains_key(&id) {
                    return;
                }
                let mut health = self.health.lock();
                health.vetoes.insert((gateway, id), until);
                self.show_veto(gateway, id, Some(until));
            }
            None => {
                let mut health = self.health.lock();
                health.vetoes.remove(&(gateway, id));
                self.show_veto(gateway, id, None);
            }
        }
    }

    fn gateway_veto_noted(
        &self,
        gateway: GatewayId,
        id: WorkerId,
        until: Option<Instant>,
        stands: bool,
        now: Instant,
    ) -> VetoWrite {
        // The workers lock first, as every writer takes it, and held to the
        // end: a row leaves and comes back under it, so the note's check
        // and the write read one membership.
        let workers = self.workers.lock();
        let mut health = self.health.lock();
        // Asked only behind a note: a write with none reads no entry.
        let left = stands
            && !health
                .vetoes
                .get(&(gateway, id))
                .is_some_and(|until| *until > now);
        if left {
            return VetoWrite::Left;
        }
        match until {
            Some(until) => {
                if !workers.rows.contains_key(&id) {
                    return VetoWrite::NoRow;
                }
                health.vetoes.insert((gateway, id), until);
                self.show_veto(gateway, id, Some(until));
            }
            None => {
                health.vetoes.remove(&(gateway, id));
                self.show_veto(gateway, id, None);
            }
        }
        VetoWrite::Written
    }

    fn vetoed(&self, gateway: GatewayId, id: WorkerId, now: Instant) -> bool {
        self.health
            .lock()
            .vetoes
            .get(&(gateway, id))
            .is_some_and(|until| *until > now)
    }

    fn completed(&self, gateway: GatewayId, id: WorkerId, class: TransportClass, at: Instant) {
        if class != TransportClass::Ok {
            return;
        }
        // Under the workers lock, taken before the health lock: a dispatch
        // completing after its worker's removal (the usual order, a gateway's
        // in-flight work ends after the row is gone) would re-create the
        // entry for good.
        let workers = self.workers.lock();
        if !workers.rows.contains_key(&id) {
            return;
        }
        let mut health = self.health.lock();
        let latest = health
            .ok_at
            .entry(gateway)
            .or_default()
            .entry(id)
            .or_insert(at);
        if at > *latest {
            *latest = at;
        }
    }

    fn completed_ok_since(&self, gateway: GatewayId, id: WorkerId, since: Instant) -> bool {
        self.health
            .lock()
            .ok_at
            .get(&gateway)
            .and_then(|workers| workers.get(&id))
            .is_some_and(|at| *at >= since)
    }
}

// ------------------------------------------------------------------ loads

impl LoadSource {
    /// A source whose records arrive on their own (a stream or its relay):
    /// what the poll decision counts as "the stream working".
    const fn pushes(self) -> bool {
        matches!(self, Self::Stream | Self::Relay(_))
    }
}

/// One worker's ranks: the latest sample per rank and, apart from it, the
/// instant each rank's own record last arrived (a poll never moves it).
#[derive(Default)]
struct Ranks {
    samples: BTreeMap<u16, LoadSample>,
    pushed_at: BTreeMap<u16, Instant>,
}

impl Ranks {
    /// Whether the rank's own record arrived at or after `since`.
    fn pushed_since(&self, rank: u16, since: Instant) -> bool {
        self.pushed_at.get(&rank).is_some_and(|&at| at >= since)
    }
}

/// What a poll's answer brought of one engine, the least first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Brought {
    /// No row for a rank of it.
    Nothing,
    /// A row for a rank that pushed since the poll began: the row is passed
    /// over, and the engine is named.
    Named,
    /// A row that was written.
    Heard,
}

/// Note what an answer brought of `engine`: the most it brought stands.
/// Inlined: the pass over an answer's rows runs it a row.
#[inline]
fn note(
    first: &mut Option<(u32, Brought)>,
    others: &mut Vec<(u32, Brought)>,
    engine: u32,
    brought: Brought,
) {
    match first {
        None => *first = Some((engine, brought)),
        Some(seen) if seen.0 == engine => seen.1 = seen.1.max(brought),
        Some(_) => match others.iter_mut().find(|seen| seen.0 == engine) {
            Some(seen) => seen.1 = seen.1.max(brought),
            None => others.push((engine, brought)),
        },
    }
}

/// The load table and the world's write side, behind one lock.
struct Loads {
    ranks: FxHashMap<WorkerId, Ranks>,
    writer: Writer,
}

impl Loads {
    /// Whether a rank `answer` names waits for a build ([`Writer::waits`]):
    /// a member's rank with no row for the answer's words to be written
    /// on. A worker with no entry here is no member and waits for none.
    fn waits(&mut self, answer: &[(WorkerId, u16, Option<LoadRow>)]) -> bool {
        let Self { ranks, writer } = self;
        answer
            .iter()
            .any(|&(id, rank, _)| writer.waits(id, rank) && ranks.contains_key(&id))
    }
}

impl MemoryStore {
    /// [`LoadSink::record`], told to the journal with `at`: the place of
    /// the leader's event the row applies, none for a row of the store's
    /// own.
    fn record_at(
        &self,
        at: Option<Position>,
        id: WorkerId,
        rank: u16,
        row: LoadRow,
        received_at: Instant,
        source: LoadSource,
    ) {
        let age = Duration::from_millis(u64::from(row.report_age_ms));
        let sample = LoadSample {
            row,
            received_at,
            sampled_at: received_at.checked_sub(age).unwrap_or(received_at),
            source,
        };
        let mut loads = self.loads.lock();
        let Loads {
            ranks: table,
            writer,
        } = &mut *loads;
        // No entry, no member: the worker left (or never registered) and
        // the record is dropped, never an entry of its own.
        let Some(ranks) = table.get_mut(&id) else {
            return;
        };
        if source.pushes() {
            ranks.pushed_at.insert(rank, received_at);
        }
        ranks.samples.insert(rank, sample);
        if let Some(journal) = &self.load_journal {
            journal.load(at, id, rank, &sample);
        }
        if let Some(place) = writer.fold_load(id, rank, Some(&sample)) {
            writer.recorded(place, id, rank, received_at);
            let world = writer.folded_at(received_at);
            self.show(writer, world);
        }
    }
}

impl LoadSink for MemoryStore {
    fn record(
        &self,
        id: WorkerId,
        rank: u16,
        row: LoadRow,
        received_at: Instant,
        source: LoadSource,
    ) {
        self.record_at(None, id, rank, row, received_at, source);
    }

    fn report(
        &self,
        answer: &[(WorkerId, u16, Option<LoadRow>)],
        polled_at: Instant,
        source: LoadSource,
    ) -> Vec<usize> {
        let mut loads = self.loads.lock();
        // A rank the answer names may have no row yet, inside the window of
        // its registration or of a refresh that moved it. Written now, its
        // row would be heard for no engine, the word that no report came
        // would be said of none, and a rank left out beside it would take
        // no mark. The build the window owes is paid first, as an ask pays
        // it, with the load lock let go for it: one build an answer at the
        // most, none where every rank the answer names has its row, and
        // none for a record, whose rank the build reads by what it
        // reported. The workers lock stays held to the answer's own write:
        // no membership write stands between the build and the answer.
        let _membership = loads.waits(answer).then(|| {
            MutexGuard::unlocked(&mut loads, || {
                let workers = self.workers.lock();
                self.pay(&workers, None);
                workers
            })
        });
        let Loads {
            ranks: table,
            writer,
        } = &mut *loads;
        // The answer is one engine's: its rows are written whole before
        // its judgment and the one publication, so its ranks are read
        // as one answer left them. A caller's mix of engines judges each,
        // and each engine is heard by its own ranks' rows.
        let mut first: Option<(u32, Brought)> = None;
        let mut others: Vec<(u32, Brought)> = Vec::new();
        let mut omits = false;
        // The door's answer: the positions of the rows it passes over.
        let mut passed: Vec<usize> = Vec::new();
        for (position, &(id, rank, row)) in answer.iter().enumerate() {
            // A report that lands after its worker left (the poll was out
            // when the row went) finds no entry and is dropped.
            let Some(ranks) = table.get_mut(&id) else {
                continue;
            };
            // A rank that pushed since the poll began keeps its record,
            // and its stream is heard: no word of the poll stands over it.
            // A row the answer brings for it is passed over. It names its
            // engine all the same, which is read below where it matters.
            if ranks.pushed_since(rank, polled_at) {
                if row.is_some() {
                    passed.push(position);
                }
                continue;
            }
            let (engine, brought) = match row {
                Some(row) => {
                    let age = Duration::from_millis(u64::from(row.report_age_ms));
                    let sample = LoadSample {
                        row,
                        received_at: polled_at,
                        sampled_at: polled_at.checked_sub(age).unwrap_or(polled_at),
                        source,
                    };
                    ranks.samples.insert(rank, sample);
                    if let Some(journal) = &self.load_journal {
                        journal.load(None, id, rank, &sample);
                    }
                    let place = writer.fold_load(id, rank, Some(&sample));
                    let engine = place.and_then(|place| {
                        let engine = writer.engine_at(place)?;
                        writer.hear(place, polled_at);
                        Some(engine)
                    });
                    (engine, Brought::Heard)
                }
                // A rank the answer omits keeps what is on file.
                None => {
                    omits = true;
                    (writer.engine_of(id), Brought::Nothing)
                }
            };
            if let Some(engine) = engine {
                note(&mut first, &mut others, engine, brought);
            }
        }
        // A row passed over names its engine as a written row does: the
        // naming is the answer's, whichever sample stands. It moves a mark
        // only beside a rank the answer left out, so the passed-over rows
        // are read for their engines there and nowhere else.
        if omits && !passed.is_empty() {
            for &(id, rank, row) in answer {
                let pushed = table
                    .get(&id)
                    .is_some_and(|ranks| ranks.pushed_since(rank, polled_at));
                if !pushed || row.is_none() {
                    continue;
                }
                if let Some(engine) = writer.engine_of(id) {
                    note(&mut first, &mut others, engine, Brought::Named);
                }
            }
        }
        let Some(first) = first else {
            return passed;
        };
        // A rank an answer leaves unnamed beside a rank it names is marked,
        // and a rank an answer names is named again: read once the
        // answer's engines are known and before any is judged. An answer
        // with no row for an engine is the word that none came, which
        // moves no mark, and a rank that pushed since the poll began
        // named itself.
        let marked = writer.any_unnamed();
        if omits || marked {
            for &(id, rank, row) in answer {
                let pushed = table
                    .get(&id)
                    .is_none_or(|ranks| ranks.pushed_since(rank, polled_at));
                if pushed {
                    continue;
                }
                let Some(engine) = writer.engine_of(id) else {
                    continue;
                };
                let named = if first.0 == engine {
                    first.1 != Brought::Nothing
                } else {
                    others
                        .iter()
                        .any(|seen| seen.0 == engine && seen.1 != Brought::Nothing)
                };
                if named {
                    writer.name(id, rank, row.is_none());
                }
            }
        }
        let mut changed = false;
        for (engine, brought) in std::iter::once(first).chain(others) {
            changed |= match brought {
                // A row was written: the engine is judged once, and the
                // write is shown whether or not a verdict moved.
                Brought::Heard => {
                    writer.judged(engine, true);
                    true
                }
                // No row was written, and the answer named a rank that
                // pushed since the poll began beside a rank it left out:
                // no word that none came. The mark the answer moved is
                // judged and shown with it.
                Brought::Named => writer.judged(engine, false),
                // No row came for the engine: the word that none came,
                // unless a rank of it pushed since the poll began. The
                // word is shown where it moves a verdict. A rank the answer
                // does not name may stand inside the window of its
                // registration, or of a refresh that moved it, with no row
                // at its rank here: the build reads what it reported
                // against the word and the instant this poll began.
                Brought::Nothing => {
                    let pushed = writer.ranks_of(engine).any(|(id, rank)| {
                        table
                            .get(&id)
                            .is_some_and(|ranks| ranks.pushed_since(rank, polled_at))
                    });
                    !pushed && writer.silence(engine, polled_at)
                }
            };
        }
        // The last mark went with this answer: every engine is judged once
        // more, behind the answer's own judgments, which read each of its
        // engines as the whole answer left it. A verdict that moves with
        // the mark's end is counted once, with whatever else of its engine
        // the answer moved.
        if marked && !writer.any_unnamed() {
            writer.judge_every_behind_a_door();
        }
        if changed {
            let world = writer.folded_at(polled_at);
            self.show(writer, world);
        }
        passed
    }

    fn heard_every(&self, period: Duration) -> bool {
        let mut loads = self.loads.lock();
        let (stands, moved) = loads.writer.heard_every(period);
        if moved {
            // Every engine was judged again under the period: shown as
            // a load write with no instant of its own is.
            let world = loads.writer.folded();
            self.show(&mut loads.writer, world);
        }
        stands
    }

    fn poll_mode(&self, id: WorkerId, now: Instant, interval: Duration) -> PollMode {
        let loads = self.loads.lock();
        let Some(ranks) = loads.ranks.get(&id) else {
            return PollMode::Poll;
        };
        // The receipts, not the samples: a poll's row replaces a silent
        // rank's sample but never its receipt, so one pushing rank cannot
        // hide another that fell silent.
        let mut pushing = ranks.pushed_at.values();
        let Some(first) = pushing.next() else {
            return PollMode::Poll;
        };
        let fresh = |at: &Instant| now.saturating_duration_since(*at) < interval;
        if fresh(first) && pushing.all(fresh) {
            PollMode::Suppressed
        } else {
            PollMode::Fallback
        }
    }

    fn load(&self, id: WorkerId, rank: u16) -> Option<LoadSample> {
        self.loads
            .lock()
            .ranks
            .get(&id)?
            .samples
            .get(&rank)
            .copied()
    }

    fn ranks(&self, id: WorkerId) -> usize {
        self.loads
            .lock()
            .ranks
            .get(&id)
            .map_or(0, |ranks| ranks.samples.len())
    }

    /// A member's rows and receipts go and its entry stays, empty: the entry
    /// is what its next report goes into, and it leaves with the worker's
    /// row alone (`remove`, a replacing `upsert`). A worker that left has no
    /// entry and nothing on file already.
    fn evict(&self, id: WorkerId) {
        let mut loads = self.loads.lock();
        let Loads {
            ranks: table,
            writer,
        } = &mut *loads;
        if let Some(ranks) = table.get_mut(&id) {
            let mut folded = None;
            for &rank in ranks.samples.keys() {
                folded = writer.fold_load(id, rank, None).or(folded);
            }
            *ranks = Ranks::default();
            // The member's marks go with its rows: a rank with nothing on
            // file is a rank again, and its engine is judged for it.
            let forgot = writer.forget(id);
            let engine = match folded {
                Some(place) => writer.engine_at(place),
                None if forgot => writer.engine_of(id),
                None => None,
            };
            if folded.is_some() || forgot {
                if let Some(engine) = engine {
                    writer.judged(engine, false);
                }
                // The last mark went with the member: every engine is
                // judged once more, behind the eviction's own judgment.
                if forgot && !writer.any_unnamed() {
                    writer.judge_every_behind_a_door();
                }
                let world = writer.folded();
                self.show(writer, world);
            }
        }
    }

    fn publish_due(&self) -> Option<Instant> {
        self.loads.lock().writer.due()
    }

    fn publish(&self, now: Instant) {
        {
            let mut loads = self.loads.lock();
            if !loads.writer.build_due(now) {
                let world = loads.writer.publish_if_due(now);
                self.show(&mut loads.writer, world);
                return;
            }
        }
        // A build is owed and due. The workers lock comes first, as every
        // membership write takes it, and the build is asked again under
        // both: another call may have paid it meanwhile.
        let workers = self.workers.lock();
        self.pay(&workers, Some(now));
    }

    fn wake_with(&self, wake: Wake) {
        *self.wake.lock() = Some(wake);
    }
}

// -------------------------------------------------------------- KV events

impl KvEventSink for MemoryStore {
    /// The batch's events as its collector decoded them: the store applies
    /// them to the index of the worker's model
    /// ([`index`](crate::store::index)).
    type Batch = [KvEvent];

    fn admit(
        &self,
        id: WorkerId,
        rank: u16,
        facts: BatchFacts,
        batch: &Self::Batch,
    ) -> BatchOutcome {
        self.kv.admit(id, rank, facts, batch)
    }

    fn reconnected(&self, id: WorkerId) {
        self.kv.reconnected(id);
    }

    fn resync(&self, id: WorkerId, rank: Option<u16>, reason: ResyncReason) {
        self.kv.resync(id, rank, reason);
    }

    fn cursor(&self, id: WorkerId, rank: u16) -> Option<KvCursor> {
        self.kv.cursor(id, rank)
    }

    fn removed(&self, id: WorkerId) {
        self.kv.removed(id);
    }

    fn block_size(&self, model: &str) -> Option<u32> {
        self.kv.block_size(model)
    }

    fn set_block_size(&self, model: &str, size: u32) {
        self.kv.set_block_size(model, size);
    }
}

// ----------------------------------------------------------- reservations

/// A gateway's lease: the key a claim is held under.
type LeaseKey = (GatewayId, LeaseId);

#[derive(Default)]
struct Reservations {
    held: FxHashMap<(WorkerId, ReservationKind), u32>,
    by_lease: FxHashMap<LeaseKey, Vec<(WorkerId, ReservationKind, u32)>>,
}

impl ReservationStore for MemoryStore {
    fn window(&self, id: WorkerId, _kind: ReservationKind) -> Option<u32> {
        // Both kinds read the engine's admission window today; a configured
        // per-side cap is the host's until the window moves into the store.
        self.workers.lock().rows.get(&id)?.row.facts.max_running
    }

    fn try_reserve(
        &self,
        gateway: GatewayId,
        id: WorkerId,
        kind: ReservationKind,
        count: u32,
        lease: LeaseId,
    ) -> bool {
        // A worker the store does not know (never registered, or removed) is
        // refused before its window is read: a missing row is no live worker
        // without a bound, and a commit racing a removal gets nothing on it.
        let window = {
            let workers = self.workers.lock();
            let Some(row) = workers.rows.get(&id) else {
                return false;
            };
            // Both kinds read the engine's admission window today, as
            // `window` does.
            row.row.facts.max_running
        };
        let mut reservations = self.reservations.lock();
        let holds_kind = reservations
            .by_lease
            .get(&(gateway, lease))
            .is_some_and(|held| held.iter().any(|(_, k, _)| *k == kind));
        if holds_kind {
            return false;
        }
        let held = reservations.held.entry((id, kind)).or_insert(0);
        let total = held.saturating_add(count);
        if window.is_some_and(|window| total > window) {
            return false;
        }
        *held = total;
        reservations
            .by_lease
            .entry((gateway, lease))
            .or_default()
            .push((id, kind, count));
        true
    }

    fn release(&self, gateway: GatewayId, lease: LeaseId) {
        let mut reservations = self.reservations.lock();
        let Some(held) = reservations.by_lease.remove(&(gateway, lease)) else {
            return;
        };
        for (id, kind, count) in held {
            if let Some(total) = reservations.held.get_mut(&(id, kind)) {
                *total = total.saturating_sub(count);
            }
        }
    }

    fn held(&self, id: WorkerId, kind: ReservationKind) -> u32 {
        self.reservations
            .lock()
            .held
            .get(&(id, kind))
            .copied()
            .unwrap_or(0)
    }
}

// --------------------------------------------------------------- sessions

impl MemoryStore {
    /// The shard that holds `key`'s pin of record.
    fn pins_of(&self, key: u64) -> &Mutex<FxHashMap<u64, Pin>> {
        &self.sessions[crate::sessions::shard_of(key)]
    }
}

impl SessionsStore for MemoryStore {
    fn pin(&self, key: u64, worker: WorkerId, at: Instant, ends: Option<u64>) {
        if key == 0 {
            return;
        }
        self.pins_of(key).lock().insert(
            key,
            Pin {
                worker,
                last_used: at,
                ends,
            },
        );
    }

    fn pin_no_earlier(&self, key: u64, worker: WorkerId, at: Instant, ends: Option<u64>) {
        if key == 0 {
            return;
        }
        let pin = Pin {
            worker,
            last_used: at,
            ends,
        };
        let mut pins = self.pins_of(key).lock();
        let stood = pins.entry(key).or_insert(pin);
        // The later end stays, and no end is later than any instant.
        let later = stood.ends.zip(ends).map(|(stood, asked)| stood.max(asked));
        *stood = Pin { ends: later, ..pin };
    }

    fn lookup(&self, key: u64) -> Option<Pin> {
        self.pins_of(key).lock().get(&key).copied()
    }

    fn unpin(&self, key: u64) {
        self.pins_of(key).lock().remove(&key);
    }

    fn drop_worker(&self, worker: WorkerId) -> usize {
        self.sessions
            .iter()
            .map(|shard| {
                let mut pins = shard.lock();
                let before = pins.len();
                pins.retain(|_, pin| pin.worker != worker);
                before - pins.len()
            })
            .sum()
    }

    fn expire(&self, now: u64) -> usize {
        self.sessions
            .iter()
            .map(|shard| {
                let mut pins = shard.lock();
                let count = pins.len();
                pins.retain(|_, pin| pin.ends.is_none_or(|ends| ends > now));
                count - pins.len()
            })
            .sum()
    }

    fn merge(&self, key: u64, pin: Pin) {
        if key == 0 {
            return;
        }
        let mut pins = self.pins_of(key).lock();
        let entry = pins.entry(key).or_insert(pin);
        if pin.last_used > entry.last_used {
            *entry = pin;
        }
    }

    fn len(&self) -> usize {
        self.sessions.iter().map(|shard| shard.lock().len()).sum()
    }
}

/// When the pins were last swept, and at what interval.
#[cfg(feature = "controller")]
#[derive(Default)]
struct PinsSwept {
    /// The instant of the last pass; `None` before the first.
    last: Option<Instant>,
    /// The interval of the last generation one of whose rules swept, in
    /// microseconds, learned at its apply; `None` before one was applied.
    every: Option<u64>,
}

#[cfg(feature = "controller")]
impl MemoryStore {
    /// One pass of the pins' sweep at `now`, where one is due: each pin of
    /// record past its own end leaves the table, whatever the other rules
    /// keep. A pass is due once the interval of the standing rule that
    /// sweeps most often has passed since the last one, so a clock that
    /// ticks faster walks nothing between two passes. The interval is the
    /// standing generation's at each tick: the pins outlive generations,
    /// and an apply that shortens it waits out no earlier one. Where no
    /// standing rule sweeps, the interval of the last generation one of
    /// whose rules swept stands, as its apply left it, so the pins it
    /// ended still leave; before any such generation no pass runs. Returns
    /// how many pins left.
    pub(crate) fn sweep_pins(&self, now: Instant) -> usize {
        let standing = self.with_world(|world| world.spec().engine().pins_shortest_sweep());
        {
            let mut swept = self.pins_swept.lock();
            let Some(every) = standing.or(swept.every) else {
                return 0;
            };
            let interval = Duration::from_micros(every);
            if swept
                .last
                .is_some_and(|last| now.saturating_duration_since(last) < interval)
            {
                return 0;
            }
            swept.last = Some(now);
        }
        // The pins' ends stand on the decision clock: the pass's instant
        // is turned into it once.
        SessionsStore::expire(self, self.micros(now))
    }
}

// ----------------------------------------------------------------- leases

/// The leases on file, the ids the store mints for a decision's, and the
/// order their dispatch deadlines fall due in.
#[derive(Default)]
struct Leases {
    term: u64,
    records: FxHashMap<Key, Held>,
    /// The store's counts, a cell a worker: what the leases on file book.
    counts: Arc<Counts>,
    /// The block of the term's counter the file mints from, as far as it
    /// minted into it ([`MemoryStore::issue_decided`]).
    #[cfg(any(test, feature = "controller"))]
    minted: Minted,
    /// Every lease filed as issued, by its dispatch deadline, the earliest
    /// first. A lease that left the file or was dispatched keeps its place
    /// until the issues that follow step over it, so no door searches the
    /// queue: it holds the leases that wait for a dispatch, and behind the
    /// first of those the places of what one dispatch deadline issued.
    due: VecDeque<(Instant, Key)>,
}

/// A lease's key on file: its gateway and its id.
type Key = (GatewayId, LeaseId);

/// The leases past their dispatch deadline that one issue ends at most.
#[cfg(any(test, feature = "controller"))]
pub(crate) const ABANDONED: usize = 2;

/// The places of the deadline queue one issue steps over at most, the
/// places of leases that were dispatched or left the file among them: four
/// for the one it adds, so the queue shortens under any run of issues.
#[cfg(any(test, feature = "controller"))]
const DUE_STEPS: usize = 4;

/// A lease on file with the slot of the count it books on its worker.
struct Held {
    record: LeaseRecord,
    /// `None` for a lease on a worker that is no member, or on a member in
    /// no pool, when the lease was written or moved ([`Slots::of`]): it
    /// books no row a decision reads.
    booked: Option<Slot>,
}

impl Leases {
    /// The next id under the store's term: the term in bits 32 to 62 over a
    /// counter that starts at 1 in each term, as the contract bounds a
    /// controller's ids. The counter comes from the block `file` holds of
    /// `blocks`, or from the next one it claims ([`Minted::mint`]). `None`
    /// for a term past 31 bits and for a term whose counter is spent: an
    /// id is never 0, never carries the bit of a gateway's own ids and
    /// never repeats.
    #[cfg(any(test, feature = "controller"))]
    fn mint(&mut self, blocks: &Blocks, file: u8) -> Option<LeaseId> {
        let term = self.term;
        if term >= 1 << 31 {
            return None;
        }
        let counter = self.minted.mint(blocks, term, file)?;
        Some(LeaseId(term << 32 | u64::from(counter)))
    }

    /// Files `record` under `key`: a record on file under that key gives
    /// its counts back first.
    fn file(&mut self, key: Key, record: LeaseRecord, booked: Option<Slot>) {
        self.end(&key);
        self.records.insert(key, Held { record, booked });
        self.laid(key, &record, booked);
    }

    /// What a record just laid on file takes: it books `booked`, its
    /// worker's count, and its dispatch deadline is queued where it is
    /// filed as issued.
    fn laid(&mut self, key: Key, record: &LeaseRecord, booked: Option<Slot>) {
        if let Some(booked) = booked {
            self.counts.book(booked);
        }
        if record.state == LeaseState::Issued {
            self.queue(record.dispatch_deadline, key);
        }
    }

    /// The keys of `gateway`'s leases on file.
    fn of(&self, gateway: GatewayId) -> Vec<Key> {
        let keys = self.records.keys();
        keys.filter(|(owner, _)| *owner == gateway)
            .copied()
            .collect()
    }

    /// Takes the lease under `key` off the file and gives its counts back:
    /// the one way a record leaves the file, whichever door ends it.
    fn end(&mut self, key: &Key) -> Option<LeaseRecord> {
        let held = self.records.remove(key)?;
        if let Some(booked) = held.booked {
            self.counts.free(booked);
        }
        Some(held.record)
    }

    /// A dispatch of the lease under `key` at `at`; `false` where it is
    /// not on file. A first dispatch that names another target of the pick
    /// moves the lease to it: the record's target and index, and the count
    /// it books, read as every door of the leases reads it
    /// ([`Slots::of`]).
    fn dispatched(&mut self, slots: Slots<'_>, key: &Key, dispatch: Dispatch, at: Instant) -> bool {
        let Some(held) = self.records.get_mut(key) else {
            return false;
        };
        if held.record.state != LeaseState::Issued {
            return true;
        }
        held.record.state = LeaseState::Dispatched { at };
        // A dispatch on another target of the pick: the record names that
        // target's index from here on. Where the target is another worker
        // the lease books it and the count of the first one is free, the
        // new count added before the first is taken; another rank of the
        // lease's own worker moves no count.
        let Some((target, _)) = held.record.target_at(dispatch.target_index) else {
            return true;
        };
        if target != held.record.target {
            let first = held.booked;
            held.booked = slots.of(target);
            self.counts.moved(first, held.booked);
            held.record.target = target;
        }
        held.record.lease.target_index = dispatch.target_index;
        true
    }

    /// A side of the lease under `key` completed; `false` where it is not
    /// on file.
    fn completed(&mut self, key: &Key, outcome: &Outcome) -> bool {
        let Some(held) = self.records.get(key) else {
            return false;
        };
        // A pair's room lives until its decode side ends: the prefill side's
        // completion leaves the record standing.
        if held.record.decode.is_some() && outcome.side == Side::Prefill {
            return true;
        }
        self.end(key);
        true
    }

    /// Queues `deadline` in its order. The deadlines come in rising order
    /// but for two deciders whose clock reads lie a moment apart and for a
    /// record written with a deadline of its own: those are placed by a
    /// search.
    fn queue(&mut self, deadline: Instant, key: Key) {
        if self.due.back().is_none_or(|&(last, _)| last <= deadline) {
            self.due.push_back((deadline, key));
        } else {
            let at = self.due.partition_point(|&(due, _)| due <= deadline);
            self.due.insert(at, (deadline, key));
        }
    }

    /// Ends the first lease of the queue that stands issued past its
    /// dispatch deadline at `now`: released as abandoned. A place whose
    /// lease left the file, was dispatched or was written again with
    /// another deadline is stepped over whatever its own deadline, `steps`
    /// of them at most over the calls that share the count: while its
    /// first place moves, the queue holds the leases that wait for a
    /// dispatch, not every issue of the deadline's span. Behind a first
    /// lease that waits out its deadline it keeps a place for every issue
    /// since, the deadline's span of them at most. `None` at a first place
    /// whose lease waits inside its deadline, at an empty queue and once
    /// the steps are spent.
    fn end_due(&mut self, now: Instant, steps: &mut usize) -> Option<LeaseRecord> {
        while *steps > 0 {
            let &(due, key) = self.due.front()?;
            let waits = self.records.get(&key).is_some_and(|held| {
                held.record.state == LeaseState::Issued && held.record.dispatch_deadline == due
            });
            if waits && due > now {
                return None;
            }
            self.due.pop_front();
            *steps -= 1;
            if waits {
                let mut record = self.end(&key)?;
                record.state = LeaseState::Released(ReleaseReason::Abandoned);
                return Some(record);
            }
        }
        None
    }
}

impl MemoryStore {
    /// Spends the ids of the store's term but for the last `left`: no door
    /// reaches a term's last ids in under 2^32 issues, and their cases
    /// start here.
    #[cfg(test)]
    pub(crate) fn spend_ids(&self, left: u32) {
        let mut leases = self.turn_of(0);
        leases.minted = self.blocks.spend(leases.term, left, 0);
    }

    /// The terms whose tables the blocks keep, the newest first: no door
    /// tells them.
    #[cfg(test)]
    pub(crate) fn kept_terms(&self) -> Vec<u64> {
        self.blocks.kept_terms()
    }

    /// The places of the deadline queue: no door tells its length.
    #[cfg(test)]
    pub(crate) fn queued(&self) -> usize {
        let files = 0..self.files.len();
        files.map(|at| self.turn_of(at).due.len()).sum()
    }

    /// Issues the lease of one decision made under `record.term`, the
    /// store's own term: the store mints its id ([`Leases::mint`]), files
    /// the record under it and books `booked`, the slot of the record's
    /// target's count as the decision's own world names it
    /// ([`World::booked`]). `Ok` is the lease as filed, with its id.
    ///
    /// # Errors
    ///
    /// The record's term is above the store's, or the store mints no id
    /// under it: the term is not the store's own any more, is past the
    /// width an id carries, or its counter is spent.
    ///
    /// Everything handed to the file is applied first ([`turn`]), and
    /// `issued` is told the lease as filed under the same hold of the turn
    /// as its write: a lease is told issued before any holder can end it.
    /// The same call ends the leases at the head of the deadline queue that no
    /// gateway named a dispatch for by `record.issued_at`, [`ABANDONED`]
    /// of them at most, released as abandoned, and tells their engine
    /// under the turn, as a frame's end is told. So a lease
    /// past its deadline stands no longer than the next issues take, and no
    /// call passes over the file: an issue adds one place to the queue and
    /// steps over [`DUE_STEPS`] at most.
    #[cfg(any(test, feature = "controller"))]
    pub(crate) fn issue_decided(
        &self,
        mut record: LeaseRecord,
        booked: Option<Slot>,
        issued: impl FnOnce(&crate::Lease),
    ) -> Result<crate::Lease, LeaseRefused> {
        use std::collections::hash_map::Entry;

        // The file of the decider that issues, under its turn: what is
        // handed is applied, and the due leases ended and told, where the
        // caller may act at that file.
        let at = self.acting();
        let file = u8::try_from(at).unwrap_or(u8::MAX);
        let mut leases = self.files[at].take();
        let mut told = turn::Told::default();
        let acts = self.applied(at, &mut leases, &mut told);
        let (lease, store) = (record.term, leases.term);
        if lease > store {
            return Err(LeaseRefused::TermAhead { lease, store });
        }
        let unminted = LeaseRefused::Unminted { lease, store };
        if lease < store {
            return Err(unminted);
        }
        let key = loop {
            record.lease.id = leases.mint(&self.blocks, file).ok_or(unminted)?;
            let key = (record.gateway, record.lease.id);
            // An id on file already is a restored lease's: the next one.
            if let Entry::Vacant(place) = leases.records.entry(key) {
                place.insert(Held { record, booked });
                break key;
            }
        };
        leases.laid(key, &record, booked);
        // The file's write and its tell under one hold of the turn: the
        // lease is told issued before any holder can end it.
        issued(&record.lease);
        // A second lease is looked for where a first one was ended: a
        // call that ends none stopped at a lease that waits, at an empty
        // queue or at its last step, and so would the next.
        let (now, mut steps) = (record.issued_at, DUE_STEPS);
        for _ in 0..if acts { ABANDONED } else { 0 } {
            let Some(ended) = leases.end_due(now, &mut steps) else {
                break;
            };
            let said = Said::Released(None, ReleaseReason::Abandoned);
            self.tell(at, told.world(self), &ended, said);
        }
        Ok(record.lease)
    }
}

impl MemoryStore {
    /// The store with a file of leases a decider, for a host that decides
    /// on `deciders` threads and names them
    /// ([`crate::decider::Decider`]): each decider issues into its own
    /// file under its own turn, and a lease's events are told under the
    /// name of the decider whose file it is on. The files start empty,
    /// under the store's term, and keep the word to the engine and the
    /// fault count the store had.
    ///
    /// # Panics
    ///
    /// Where a lease is on file: its file would go with it, and the count
    /// it books would stand for good.
    #[must_use]
    pub fn with_deciders(mut self, deciders: usize) -> Self {
        // Every file is read, each under its own turn: a lease on any of
        // them would go with its file.
        let mut files = self.files.iter();
        let empty = files.all(|file| file.take().records.is_empty());
        assert!(empty, "a store's files are laid before a lease is filed");
        let (term, counts) = {
            let leases = self.files[0].take();
            (leases.term, Arc::clone(&leases.counts))
        };
        let files = turn::Files::of(deciders, &self.files, || Leases {
            term,
            counts: Arc::clone(&counts),
            ..Leases::default()
        });
        self.files = files;
        self
    }

    /// How many files of leases the store holds: one a decider.
    #[must_use]
    pub fn deciders(&self) -> usize {
        self.files.len()
    }

    /// Raises the files' term to `term` where it is above theirs, and
    /// calls `rose` under their turns, every one held
    /// ([`Self::every_turn`]). Each file reads the term under its own
    /// turn.
    fn raise(&self, term: u64, rose: impl FnOnce()) {
        if term <= self.files[0].take().term {
            return;
        }
        let mut turns = self.every_turn();
        if self.rise(&mut turns, term) {
            rose();
        }
    }

    /// Every file's turn, taken in the files' order with nothing applied.
    /// A term's rise holds them, the start of a term the store follows
    /// ([`Self::follow_term`]) and a standby's restore
    /// ([`Self::restore_from`]): no other path holds two turns.
    fn every_turn(&self) -> Vec<turn::Turn<'_>> {
        self.files.iter().map(turn::File::take).collect()
    }

    /// Writes `term` on every file where it is above theirs, under their
    /// turns, and says whether it rose.
    fn rise(&self, turns: &mut [turn::Turn<'_>], term: u64) -> bool {
        if !turns.first().is_some_and(|leases| term > leases.term) {
            return false;
        }
        for leases in &mut *turns {
            leases.term = term;
        }
        // The blocks start again under the new term, and the table the
        // rise leaves is kept while a lease of its term is on some file.
        let on_file = turns.iter().flat_map(|leases| leases.records.keys());
        self.blocks.rise(term, on_file.map(|&(_, lease)| lease));
        true
    }

    /// Takes every lease off its file: what a restore leaves of the
    /// leases this replica filed. `turns` are the files', every one held
    /// by the restore since its start, and nothing is told under them:
    /// the ids on file are read and each file's deadline queue is
    /// cleared, since every end is a removed worker's release and never a
    /// deadline's. The turns are let go there, and each lease's end is
    /// handed to its file with no turn held ([`Self::hand`]), in the
    /// files' order: applied and told as a frame's is, by the file's
    /// decider at work or by this caller where none is. A lease filed
    /// behind the turns is none of those read, and stays.
    fn empty(&self, mut turns: Vec<turn::Turn<'_>>) {
        let removed = Said::Released(None, ReleaseReason::WorkerRemoved);
        let mut on_file: Vec<Key> = Vec::new();
        for leases in &mut turns {
            leases.due.clear();
            on_file.extend(leases.records.keys().copied());
        }
        drop(turns);
        for (gateway, lease) in on_file {
            self.hand(&Hand::new(gateway), lease, removed);
        }
    }
}

/// Where a door of the leases reads the slot of a worker's count: the
/// store's standing world, and its table of the members no world lists
/// yet. One word, the store's own: the dispatch arm carries it under
/// every take of a turn.
#[derive(Clone, Copy)]
struct Slots<'a>(&'a MemoryStore);

impl Slots<'_> {
    /// The slot of `target`'s count: `None` for a worker that is no member
    /// and for a member in no pool behind its first build: inside its
    /// window such a member is read from the table, as every joiner is.
    /// The standing world lists a member from its first build on, and a
    /// listed target takes no lock. Inlined into each door: a listed
    /// target's read costs the door no call, one read for each lease it
    /// files.
    #[inline(always)]
    fn of(self, target: WorkerId) -> Option<Slot> {
        if let Some(slot) = self.0.world.load().booked_of(target) {
            return Some(slot);
        }
        self.filed(target)
    }

    /// The slot of a target the standing world does not list. A member
    /// inside its registration window is read from the table its write
    /// filed it in, and so is one in no pool behind the refresh that may
    /// list it. A build publishes before it empties the table, so a
    /// target the table no longer holds is read off the world again. The
    /// table's lock is a leaf, taken under the turn or with no lock held.
    #[cold]
    fn filed(self, target: WorkerId) -> Option<Slot> {
        let filed = {
            let joined = self.0.joined.lock();
            joined
                .iter()
                .find(|&&(id, _)| id == target)
                .map(|&(_, slot)| slot)
        };
        filed.or_else(|| self.0.world.load().booked_of(target))
    }
}

impl MemoryStore {
    /// The reader of the leases' doors ([`Slots::of`]).
    fn slots(&self) -> Slots<'_> {
        Slots(self)
    }

    /// Gives the member a membership write files its slot in the counts,
    /// and files it for the leases' doors until a world lists it. Called
    /// under the workers lock, by the write that makes the member's row.
    fn join(&self, id: WorkerId) -> Slot {
        let slot = self.counts.claim();
        self.joined.lock().push((id, slot));
        slot
    }

    /// Empties the table of the members no world listed. Called behind
    /// the publication of a membership build, under the workers lock:
    /// every member the table held is in the published world, stands in
    /// no pool or has left. A refresh that may list a member in no pool
    /// files its slot again.
    fn listed(&self) {
        self.joined.lock().clear();
    }
}

impl LeasesStore for MemoryStore {
    fn term(&self) -> u64 {
        self.files[0].take().term
    }

    fn set_term(&self, term: u64) {
        // Taken before the turn, so dropped after it: a rise is told to the
        // journal under the turn, and released once that is let go.
        let _released = self.released();
        self.raise(term, || {
            if let Some(journal) = &self.journal {
                journal.term_rose(term);
            }
        });
    }

    fn issue(&self, record: LeaseRecord) -> Result<(), LeaseRefused> {
        // The slot of the record's target's count: read before the turn is
        // taken.
        let booked = self.slots().of(record.target);
        // The file the id names for good, asked under a file's turn: a
        // rise of the term holds every turn, so the answer stands while
        // one is held. The first turn is the file's that the id reads with
        // no turn, and an answer that names another file is asked again
        // under that file's turn: the lease is filed under the turn of the
        // file its id names. An id whose file is not settled is filed in
        // none.
        let id = record.lease.id;
        let files = self.files.len();
        let mut at = self.file_of(id);
        let (mut leases, settled) = loop {
            let leases = self.turn_of(at);
            match self.blocks.settled(id, files) {
                Some(named) if named != at => at = named,
                settled => break (leases, settled),
            }
        };
        if record.term > leases.term {
            return Err(LeaseRefused::TermAhead {
                lease: record.term,
                store: leases.term,
            });
        }
        if settled.is_none() {
            return Err(LeaseRefused::Unclaimed);
        }
        let key = (record.gateway, record.lease.id);
        if leases.records.contains_key(&key) {
            return Err(LeaseRefused::Duplicate);
        }
        leases.file(key, record, booked);
        Ok(())
    }

    fn dispatched(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        dispatch: Dispatch,
        at: Instant,
    ) -> bool {
        self.turn_of(self.file_of(lease))
            .dispatched(self.slots(), &(gateway, lease), dispatch, at)
    }

    fn completed(&self, gateway: GatewayId, lease: LeaseId, outcome: &Outcome) -> bool {
        self.turn_of(self.file_of(lease))
            .completed(&(gateway, lease), outcome)
    }

    fn released(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        _side: Option<Side>,
        _reason: ReleaseReason,
    ) -> bool {
        // No side is read yet: a release of one side of a pair ends the
        // whole lease. The step that issues a pair's lease reads it.
        self.turn_of(self.file_of(lease))
            .end(&(gateway, lease))
            .is_some()
    }

    fn get(&self, gateway: GatewayId, lease: LeaseId) -> Option<LeaseRecord> {
        let leases = self.turn_of(self.file_of(lease));
        leases
            .records
            .get(&(gateway, lease))
            .map(|held| held.record)
    }

    /// Each file's turn in the files' order, one at a time, where the
    /// caller may act at the file: a decider at work ends its own file's
    /// due leases at its issues. Each end is told under the turn, as a
    /// frame's end is told.
    fn expire(&self, now: Instant) -> Vec<LeaseRecord> {
        let mut ended = Vec::new();
        for at in 0..self.files.len() {
            // A file whose decider works is passed with no wait for its
            // turn; the same is read again under the turn.
            if !self.may_act(at) {
                continue;
            }
            let mut leases = self.files[at].take();
            let mut told = turn::Told::default();
            if !self.applied(at, &mut leases, &mut told) {
                continue;
            }
            let mut steps = usize::MAX;
            while let Some(record) = leases.end_due(now, &mut steps) {
                let said = Said::Released(None, ReleaseReason::Abandoned);
                self.tell(at, told.world(self), &record, said);
                ended.push(record);
            }
        }
        ended
    }

    fn restore(&self, gateway: GatewayId, records: &[LeaseRecord]) -> usize {
        // A page may list a lease on a member whose first build is owed:
        // its slot is read where its write filed it, so each lease the
        // pages list books its worker's count and no build is asked for.
        let slots = self.slots();
        // Each file's turn in the files' order, one at a time: the
        // gateway's leases there go, and the records its ids name are
        // filed. A record whose id names no file for good is filed in
        // none ([`LeaseRefused::Unclaimed`]).
        let files = self.files.len();
        let mut held = 0;
        for at in 0..self.files.len() {
            let mut leases = self.turn_of(at);
            let term = leases.term;
            for key in leases.of(gateway) {
                leases.end(&key);
            }
            for record in records {
                let id = record.lease.id;
                let here = self.blocks.settled(id, files) == Some(at);
                if record.gateway != gateway || record.term > term || !here {
                    continue;
                }
                let booked = slots.of(record.target);
                leases.file((gateway, record.lease.id), *record, booked);
            }
            held += leases.of(gateway).len();
        }
        held
    }

    fn drop_gateways(&self, gone: &[GatewayId]) -> Vec<usize> {
        if gone.is_empty() {
            return Vec::new();
        }
        // Sorted once: each table's one pass asks the list by search. A
        // gateway named twice is found at one place, so its one count is
        // said twice.
        let mut sorted = gone.to_vec();
        sorted.sort_unstable();
        let mut released = vec![0_usize; sorted.len()];
        // Each file's turn in the files' order, one at a time.
        for file in 0..self.files.len() {
            let mut leases = self.turn_of(file);
            // One pass names the leases of every gateway that goes. Each
            // then leaves by the file's one end, which gives back the count
            // it books.
            let ended: Vec<(usize, Key)> = leases
                .records
                .keys()
                .filter_map(|key| sorted.binary_search(&key.0).ok().map(|at| (at, *key)))
                .collect();
            for (at, key) in &ended {
                released[*at] += 1;
                leases.end(key);
            }
        }
        // The turn is taken first or not at all, so it is let go here. The
        // gateways' vetoes and completed dispatches leave under the health
        // lock and the published vetoes under the load lock inside it: the
        // order a veto's own write takes.
        let mut health = self.health.lock();
        health.forget_gateways(&sorted);
        self.show_gateways_gone(&sorted);
        gone.iter()
            .map(|gateway| sorted.binary_search(gateway).map_or(0, |at| released[at]))
            .collect()
    }

    fn live(&self, gateway: GatewayId) -> usize {
        let live = |held: &Held| {
            matches!(
                held.record.state,
                LeaseState::Issued | LeaseState::Dispatched { .. }
            )
        };
        let files = 0..self.files.len();
        files
            .map(|at| {
                let leases = self.turn_of(at);
                let records = leases.records.iter();
                records
                    .filter(|((owner, _), held)| *owner == gateway && live(held))
                    .count()
            })
            .sum()
    }
}
