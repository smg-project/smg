//! The one-process store: every ingress trait over plain tables behind
//! mutexes. The controller's own store once its drivers move in (soft state
//! its sources republish, no log), and the double the contract tests and the
//! controller's tests run over. Nothing here is on a decision path: a decision
//! reads the world the store publishes ([`MemoryStore::snapshot`]), not
//! through these locks.

// A case pauses a snapshot behind its copy of the rows: the store holds
// that pause in a field, so the two ends of its channel are named here.
#[cfg(test)]
use std::sync::mpsc::{Receiver, SyncSender};
use std::{
    collections::{HashMap, VecDeque},
    mem,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU64, Ordering::Relaxed},
        Arc,
    },
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::FxHashMap;

#[cfg(feature = "controller")]
use crate::engine::CacheIndexes;
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision_engine::DecodeRefused,
        view::{Session, WorkerId, LOAD_REPORT_CLOCK_HISTORY_MS},
    },
    store::{
        applied::EngineFaults,
        health::{HealthSink, VetoWrite},
        journal::{Journal, Position},
        leases::{
            DecodeMark, DecodeSide, HostCount, InFlightCounts, LeaseRecord, LeaseRefused,
            LeaseState, LeasesStore, Sides, WorkerSlot,
        },
        membership::same_worker,
        snapshot::{left_models, micros, OutsideLists, StoreSnapshot, Writer, PUBLISH_WINDOW},
        workers::{MembershipChange, WorkerFacts, WorkerRow, WorkerStatus, WorkersStore},
        GatewayId, StaleGeneration,
    },
    tracking::lease::{
        Dispatch, LeaseId, Membership, Outcome, ReleaseReason, Said, Side, DISPATCH_TTL,
    },
};

mod blocks;
mod turn;

use blocks::{Blocks, Minted};
pub use turn::{AtWork, Hand, LookOutcome};

/// The host's wake: what the store calls to have its host read
/// [`MemoryStore::publish_due`] again. It is called under the writer's
/// locks: it sets a flag or a notification and returns.
pub type Wake = Box<dyn Fn() + Send + Sync>;

/// The store's counts one scrape reports.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct StoreStats {
    /// The members outside the standing generation's lists.
    pub outside: OutsideLists,
    /// The lifecycle calls of a lease that panicked and were held
    /// ([`MemoryStore::lifecycle_faults`]): a count of its own, read
    /// beside the first.
    pub lifecycle_faults: u64,
    /// The panics the store's engine held under its own catches, over
    /// every document the store applied: counts of their own too.
    pub engine_faults: EngineFaults,
}

/// The store. `Default` is empty with term `0`.
pub struct MemoryStore {
    /// The rows, behind the membership lock. A write that records a change
    /// holds it as a [`Writing`], which tells the journal when the write's
    /// locks are let go.
    workers: Mutex<Workers>,
    #[cfg(test)]
    snapshot_pause: Mutex<Option<(SyncSender<()>, Receiver<()>)>>,
    health: Mutex<HealthTables>,
    /// The world's write side: every world is made and shown under this
    /// lock, so the deciders see the worlds in the order they were made. It
    /// is the last lock a writer takes: the workers lock, then the health
    /// lock, then this one.
    loads: Mutex<Loads>,
    /// The files of leases behind their turns, one a decider ([`turn`]).
    files: turn::Files,
    /// The blocks of the leading term's counter, each block's file and the
    /// tables kept for terms that led ([`blocks`]): how a lease's id names
    /// its file.
    blocks: Blocks,
    /// What a decision reads, through one load: swapped whole at each
    /// publication.
    world: ArcSwap<StoreSnapshot>,
    /// Every member on file that no published world names yet, each with
    /// its slot in the counts. The write that files a member puts it
    /// here, and no other write does: a world names every member its
    /// build read, in a pool or in none. A membership build empties the
    /// table behind its publication, and no other publication touches
    /// it. A leaf lock: nothing is taken under it.
    joined: Mutex<Vec<(WorkerId, WorkerSlot)>>,
    /// The store's counts of the leases on file, for the write that gives
    /// a joining member its slot.
    counts: Arc<InFlightCounts>,
    /// Where the store's clock starts.
    origin: Instant,
    /// Where the term is told, and that a write's locks are let go; `None`
    /// for a store nobody follows. The membership writes tell through
    /// their table's own handle.
    journal: Option<Arc<dyn Journal>>,
    /// The host's wake, once its loop handed one ([`MemoryStore::wake_with`]):
    /// called where a membership write leaves a build owed and none was,
    /// the one instant the host's loop cannot have read. A lock of its own,
    /// the last taken and held for the call alone.
    wake: Mutex<Option<Wake>>,
}

/// The replicated rows at one place of the log's history. A controller's is
/// cut from its worker objects by their holder, and the store's own entry
/// cuts one from its table ([`MemoryStore::checkpoint`]). The contract's
/// three snapshot messages are built from it: the operator's, the watch's
/// and the replication's.
#[derive(Clone, Debug)]
#[expect(clippy::exhaustive_structs, reason = "its mark comes with its call")]
pub struct StoreCheckpoint {
    /// The last journaled write that the tables hold, with its term; the
    /// default for a store with no journal.
    pub at: Position,
    /// The membership epoch.
    pub epoch: u64,
    /// Every row, by registration key.
    pub rows: Vec<WorkerRow>,
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

    /// A store whose owed builds wait at most `window` for their
    /// publication, where [`MemoryStore::new`] takes [`PUBLISH_WINDOW`]:
    /// for a test of the window itself. A window the clock cannot add to an
    /// instant never ends: a build is then paid by the ask
    /// alone ([`MemoryStore::publish_now`]).
    pub fn with_publish_window(window: Duration) -> Self {
        let now = Instant::now();
        // An hour before the store was made, where the host's clock reaches
        // that far: a sample older than the store keeps an instant of its
        // own.
        let history = Duration::from_millis(u64::from(LOAD_REPORT_CLOCK_HISTORY_MS));
        let origin = now.checked_sub(history).unwrap_or(now);
        let writer = Writer::new(window, origin);
        let world = ArcSwap::new(writer.world());
        // The leases' file writes the counts the writer's shapes name, and
        // a membership write gives a joining member its slot in them.
        let counts = writer.counts();
        let leases = Leases::empty(0, Arc::clone(&counts), origin);
        Self {
            workers: Mutex::default(),
            #[cfg(test)]
            snapshot_pause: Mutex::default(),
            health: Mutex::default(),
            loads: Mutex::new(Loads { writer }),
            files: turn::Files::one(leases),
            blocks: Blocks::default(),
            world,
            joined: Mutex::default(),
            counts,
            origin,
            journal: None,
            wake: Mutex::new(None),
        }
    }

    /// A store that tells `journal` its replicated writes (membership and
    /// status, and the term), each inside the write's own critical
    /// section: a store a standby or a watcher of its rows can follow.
    pub fn journaled(journal: Arc<dyn Journal>) -> Self {
        let mut store = Self::default();
        store.workers.get_mut().journal = Some(Arc::clone(&journal));
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

    /// The table's rows as one journaled write left them, with that
    /// write's position. Taken under the membership lock: every journaled
    /// write holds its table's lock through its apply and its telling, so
    /// none stands half told here. A writer of the rows waits for the copy; a
    /// decision reads the published world and does not.
    ///
    /// No product code calls it. What a watcher of the rows starts from,
    /// what a standby is restored from and what an operator reads are cut
    /// from the worker objects by their holder, with the status the
    /// journal told (`MemoryStore::told`, under the `worker-host`
    /// feature). This cut goes with the table.
    pub fn checkpoint(&self) -> StoreCheckpoint {
        // Copied under the lock, sorted after it: a writer waits for the
        // copy alone.
        let (at, epoch, mut rows) = {
            let workers = self.workers.lock();
            let at = self
                .journal
                .as_ref()
                .map_or_else(Position::default, |journal| journal.head());
            let rows: Vec<WorkerRow> = workers.rows.values().map(|row| row.row.clone()).collect();
            (at, workers.epoch, rows)
        };
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        StoreCheckpoint { at, epoch, rows }
    }

    /// The tables replaced by `cut`, a leader's snapshot whole: what a
    /// standby starts from, and starts from again when the leader's log no
    /// longer holds its place. Every row the store held leaves
    /// with what lives as long as a row (the prober's counts, the gateways'
    /// vetoes in the table and in the world), the epoch
    /// is the cut's, and the journal starts again at the cut's place inside
    /// the same locks: a reader of the tables sees them before the restore
    /// or after it, never between. The session pins take the steps of the
    /// writes the cut stands for, as at a watch ([`Self::watch_from`]). The
    /// store's term rises to
    /// the cut's, as following that term's start leaves it. The engine is told the
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
    /// that decider rests. The stages that stood at the lease's issue hear
    /// it once, where a generation that holds them stands. The restore
    /// returns when
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
    pub fn restore_from(&self, cut: StoreCheckpoint) -> bool {
        // Taken before the first lock, so dropped after the last: the journal
        // restarts inside the locks, and its listener is told behind them.
        let _released = self.released();
        // Every file's turn first, as their order has it, and held until
        // the ids on file are read: a take's write waits for them, so a
        // take lands whole ahead of the restore or whole behind it.
        let mut turns = self.every_turn();
        let term = cut.at.term;
        let Some(ended) = self.lay_snapshot(cut, false) else {
            return false;
        };
        // The term rises and the leases on file are read under the turns.
        // Their releases are handed to the files once the turns are let go.
        self.rise(&mut turns, term);
        self.empty(turns);
        self.forget_gone(&ended);
        true
    }

    /// Lays the worker rows and epoch of a `WatchWorkers` snapshot and
    /// builds the world at once. The store keeps its lease files and their
    /// term. Its session pins take the steps of the writes the cut stands
    /// for. A key the cut holds under a new id takes its pins where it is
    /// the worker it was, and another worker's pins leave. A key the cut
    /// drops is filed as gone by this store's own clock, as a followed
    /// removal files it. A worker still present keeps its count
    /// slot and gateway vetoes. Its row is the cut's, the admission instant
    /// too: a holder of worker objects states the instant it keeps on the
    /// cut's row before it hands the cut here (`Objects::watch_from`).
    ///
    /// Returns `false` with the store as it was if the journal refuses the
    /// snapshot's place.
    ///
    /// A follower's store has no journal. On a journaled store the journal
    /// takes the cut's place: it follows the cut's term and leads no
    /// longer, as at a restore, and the lease files keep their own term.
    /// The two terms are in step after a restore and not after this door.
    pub fn watch_from(&self, cut: StoreCheckpoint) -> bool {
        let _released = self.released();
        let Some(ended) = self.lay_snapshot(cut, true) else {
            return false;
        };
        self.forget_gone(&ended);
        true
    }

    /// Lays the rows and publishes their world. A watch carries local
    /// state for the ids it sees again. A standby gives every row a
    /// fresh slot. The session pins take the steps
    /// of the writes the cut stands for ([`Workers::lay_gone`]). The ids
    /// whose pins leave are returned, for the caller to forget behind its
    /// locks ([`Self::forget_gone`]). `None`, with the store as it was,
    /// where the journal refuses the cut's place.
    fn lay_snapshot(&self, cut: StoreCheckpoint, keep_local: bool) -> Option<Vec<WorkerId>> {
        let mut workers = self.workers.lock();
        // The previous membership. A watch carries its local state for the
        // ids it sees again. A restore carries none of it. Either door reads
        // the rows the cut drops: their session pins take a leave's steps.
        let mut previous;
        {
            let mut health = self.health.lock();
            let mut loads = self.loads.lock();
            if let Some(journal) = &self.journal {
                if !journal.restart(cut.at) {
                    return None;
                }
            }
            previous = mem::take(&mut workers.rows);
            if !keep_local {
                health.vetoes.clear();
                // An unpublished build over no row drops the restore's old
                // vetoes and the state derived from its membership.
                drop(loads.writer.rebuild(std::iter::empty()));
            }
            workers.by_key.clear();
            workers.epoch = cut.epoch;
            workers.rows.reserve(cut.rows.len());
            for row in cut.rows {
                let id = row.id;
                workers.by_key.insert(row.facts.key.clone(), id);
                let kept = if keep_local {
                    previous.remove(&id).or_else(|| workers.rows.remove(&id))
                } else {
                    None
                };
                let held = if let Some(kept) = kept {
                    Row { row, ..kept }
                } else {
                    Row {
                        row,
                        slot: self.join(id),
                    }
                };
                workers.rows.insert(id, held);
            }
            if keep_local {
                // Prune each table once for the whole membership.
                let stays = |id: &WorkerId| workers.rows.contains_key(id);
                health.vetoes.retain(|(_, id), _| stays(id));
            }
        }
        // The rows the cut drops stay in hand. Each one's slot is given to
        // no row again, so the count its host handed for it goes.
        for held in previous.values() {
            self.counts.left(held.slot);
        }
        previous.retain(|id, _| !workers.rows.contains_key(id));
        // The session pins take the steps of the writes the cut stands
        // for, ahead of the world that lists its members: no decision reads
        // a new row before its pins stand under its id. Until that world
        // stands, a member the last one listed under its old id reads as in
        // no pool: a request of its key is placed by the rule, as inside a
        // replacement ([`Self::pins_follow`]).
        let left = previous.into_iter().map(|(id, held)| (id, held.row.facts));
        let (follows, ended) = workers.lay_gone(left, Instant::now(), self.pins_absence());
        for (old, new) in follows {
            self.pins_follow(old, new);
        }
        self.swap_world(&workers);
        Self::tell_members(&workers, |list| self.tell_engine(list));
        Some(ended)
    }

    /// One of the leader's membership or status writes, applied as the
    /// replication stream tells it: a standby's side of a journaled write
    /// of a row or of [`HealthSink`]. `row` is the row as the
    /// leader's write left it, and none for a removal or a status change;
    /// a resize is told as a relabel and recorded as the leader recorded
    /// it, by the widths. Nothing is minted and no revision is guarded: the
    /// leader's log is the order. A row that leaves takes its tables with
    /// it as the leader's own write does, and its key's session pins take
    /// the leader's steps: they stand through the key's absence, follow
    /// the worker it was to its next id and leave with another worker. A
    /// row that stays keeps its admission instant. `at` is the event's
    /// place in that log, which the applier holds ([`Journal::member`],
    /// [`Journal::status`]): the journal takes the write there, and no
    /// write of the store's own does.
    ///
    /// It is the call of a store with no holder of worker objects, and of a
    /// fixture. A holder is told nothing of a change written here. The row
    /// such a change leaves has no object until a cut is laid whole, and the
    /// holder's registration of its key is refused until then. A standby's
    /// applier and a watcher also lay the row at the next event that tells
    /// it and that the store takes.
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
    ///
    /// A status change has a door of its own, and this call hands it on
    /// with no list of models (`MemoryStore::status_followed`): the journal
    /// is told the ones the table's row serves.
    pub fn follow(
        &self,
        at: Position,
        epoch: u64,
        change: MembershipChange,
        row: Option<WorkerRow>,
    ) {
        if let (MembershipChange::StatusChanged { id, from, to }, None) = (change, &row) {
            self.status_followed(at, epoch, id, from, to, None);
        } else {
            self.follow_landed(at, epoch, change, row, None);
        }
    }

    /// [`Self::follow`] for a membership change, and whether the change
    /// landed. A holder of worker objects asks it: the object of the row a
    /// change leaves is put in its registry only where the store took the
    /// change. `left` is the row the change takes out, as the holder read
    /// it back from its object: a removal's row, a replacement's old life.
    /// The journal is told it, as at a leader's write
    /// (`MemoryStore::follow_objects`). A status change fits no table
    /// here: it has its own door ([`Self::status_followed`]).
    pub(crate) fn follow_landed(
        &self,
        at: Position,
        epoch: u64,
        change: MembershipChange,
        row: Option<WorkerRow>,
        left: Option<&mut WorkerRow>,
    ) -> bool {
        let workers = self.writing_at(Some(at));
        epoch == workers.epoch + 1 && self.decided(workers, Instant::now(), change, row, left)
    }

    /// One of the leader's status writes, applied as the replication
    /// stream tells it: a standby's side of a told status, as
    /// `MemoryStore::status_told` is a leader's. The row `id` left `from`
    /// and took `to`. The write lands only where it fits the table: the
    /// epoch is the next one, and the row stands in the status it leaves.
    /// One that does not fit writes nothing and tells the journal nothing,
    /// as [`Self::follow`] says of every change. `at` is the event's place
    /// in the leader's log, which the applier holds: the journal takes the
    /// write there ([`Journal::status`]). No world holds a status, so none is
    /// published. The row's revision stands: it counts the row's stated
    /// facts.
    ///
    /// `models` are the models the row's worker object serves, read by the
    /// holder of the objects in its turn and handed ahead of this lock.
    /// The journal is told them with the status, and no row. A holder
    /// hands a list only where the store tells a journal
    /// (`MemoryStore::journals`), so a host that tells none builds none. A
    /// writer that hands none to a store that tells one has the table's
    /// row's told: a host that hands the store a status itself
    /// ([`Self::follow`]), and a holder with no object under the row's id.
    pub(crate) fn status_followed(
        &self,
        at: Position,
        epoch: u64,
        id: WorkerId,
        from: WorkerStatus,
        to: WorkerStatus,
        models: Option<Arc<[String]>>,
    ) -> bool {
        let mut workers = self.writing_at(Some(at));
        epoch == workers.epoch + 1 && workers.follow_status(id, from, to, models)
    }

    /// A decided membership change, written under the write's hold of the
    /// membership lock: the tables, the epoch and the
    /// journal's telling, the member's count slot, the build
    /// the write owes or pays, the word to the stages and the session
    /// pins. One body for every writer that hands a change with the
    /// row it leaves: [`Self::follow`] hands a leader's, and where the
    /// replica keeps worker objects their holder hands the ones it decided
    /// (`MemoryStore::follow_objects`). `now` is the write's one read of the
    /// clock, behind the lock. Whether the change fit the tables, as
    /// [`Self::follow`] says it: one that does not writes nothing and tells
    /// the journal nothing.
    ///
    /// The journal is told the row its writer handed, with the admission
    /// instant of the row it is laid over, and the table takes that row
    /// behind the telling: the event of such a write tells no row read
    /// from the table. The row a write takes out is told as its writer
    /// handed it too, where one did: `left`, which a holder of worker
    /// objects read back from the object, on a leader and on a standby.
    /// The status the table held is laid over it here: the status the log
    /// last told of a row is the log's own word, which the table holds for
    /// it. A row changed in place is laid no status here: a standby's told
    /// row comes in its leader's status, and a leader's holder lays the
    /// table's in its own entry (`MemoryStore::follow_objects`). A writer
    /// that hands none tells the row the table held: [`Self::follow`], and
    /// a holder with no object under the row's id. A status is no change
    /// of this body: it comes with no row, its event holds none, and it
    /// is written by a door of its own on a standby
    /// ([`Self::status_followed`]) and on a leader
    /// (`MemoryStore::status_told`).
    fn decided(
        &self,
        mut workers: Writing<'_>,
        now: Instant,
        change: MembershipChange,
        row: Option<WorkerRow>,
        left: Option<&mut WorkerRow>,
    ) -> bool {
        use std::collections::hash_map::Entry;

        // The models a registration that stays under its id takes its
        // worker off, for the stages: here as in the leader's own write.
        let mut off = Vec::new();
        // A key's session pins, here as in the leader's own write: the ids
        // they follow their worker between, and the ids they leave with.
        let mut follows = None;
        let mut ended = Vec::new();
        // The row the change leaves, as its writer handed it: the journal
        // is told from it, and the table takes it behind the telling. With
        // it, the row the table held where the write takes one out.
        let (change, laid, held) = match (change, row) {
            (MembershipChange::Registered { id }, Some(row))
                if row.id == id
                    && !workers.rows.contains_key(&id)
                    && !workers.by_key.contains_key(&row.facts.key) =>
            {
                workers.by_key.insert(row.facts.key.clone(), id);
                let (old, past) = workers.take_gone(&row.facts, now, self.pins_absence());
                follows = old.map(|old| (old, id));
                ended = past;
                (change, Some(row), None)
            }
            (
                MembershipChange::Relabelled { id } | MembershipChange::Resized { id },
                Some(mut row),
            ) if row.id == id => {
                let held = workers.rows.get(&id);
                let Some(held) = held.filter(|held| held.row.facts.key == row.facts.key) else {
                    return false;
                };
                let resized = held.row.facts.dp_size != row.facts.dp_size;
                row.admitted_at = held.row.admitted_at;
                if resized {
                    (MembershipChange::Resized { id }, Some(row), None)
                } else {
                    (MembershipChange::Relabelled { id }, Some(row), None)
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
                let held = workers.rows.remove(&old).map(|held| self.row_left(held));
                let same = |held: &WorkerRow| same_worker(&held.facts, &row.facts);
                if held.as_ref().is_some_and(same) {
                    follows = Some((old, new));
                } else {
                    ended.push(old);
                }
                workers.by_key.insert(row.facts.key.clone(), new);
                self.forget_worker(old);
                (change, Some(row), held)
            }
            (MembershipChange::Removed { id }, None) => {
                let Some(held) = workers.rows.remove(&id) else {
                    return false;
                };
                workers.by_key.remove(&held.row.facts.key);
                self.forget_worker(id);
                (change, None, Some(self.row_left(held)))
            }
            // A status is no membership change: it has its own door
            // (`MemoryStore::status_followed`), and fits no table here.
            _ => return false,
        };
        // The row the write took out is told as its writer handed it, in
        // the status the table held, and as the table held it where the
        // writer handed none.
        let told = match (left, held.as_ref()) {
            (Some(left), Some(held)) => {
                left.status = held.status;
                // While the table stands, a debug build holds a handed row
                // to the one the table held, where an entry is told: the
                // two are one row. A host with no journal tells nothing,
                // and a gateway's refresh of a worker's models sets a fact
                // of its object behind the build.
                debug_assert!(
                    workers.journal.is_none() || *left == *held,
                    "a row its holder read back is the row the table held"
                );
                Some(&*left)
            }
            (left, held) => {
                debug_assert!(
                    left.is_none(),
                    "a row is handed as taken out by a write that takes one out"
                );
                held
            }
        };
        workers.record_over(change, laid.as_ref(), told);
        // The table follows the row told: over the row that stood under
        // its id, or under a new id with a count slot of its own.
        if let Some(row) = laid {
            match workers.rows.entry(row.id) {
                Entry::Occupied(mut held) => {
                    debug_assert!(
                        matches!(
                            change,
                            MembershipChange::Relabelled { .. } | MembershipChange::Resized { .. }
                        ),
                        "a row laid over the one that stands under its id is a change in place"
                    );
                    let old = mem::replace(&mut held.get_mut().row, row);
                    off = left_models(old.facts, &held.get().row.facts);
                }
                Entry::Vacant(free) => {
                    let slot = self.join(row.id);
                    free.insert(Row { row, slot });
                }
            }
        }
        // The build each write leaves, as the leader's own: a join, a
        // refresh and a patch owe it, and a leave pays its own before it is
        // told, a registration that takes its worker off a model among
        // them. What is owed, the replica's loop pays at the window's end.
        match change {
            MembershipChange::Registered { .. } => {
                self.owe(now);
                self.tell_membership(change);
            }
            MembershipChange::Replaced { new, .. } => {
                // A replacement that takes its worker off a model says so
                // for the new id, behind the build it pays and behind its
                // own word, as at the leader's own write.
                let models = Self::replaced_models(held, workers.rows.get(&new));
                self.pay_leave(&workers, now);
                self.tell_membership(change);
                self.tell_left(new, &models);
            }
            MembershipChange::Relabelled { .. } | MembershipChange::Resized { .. } => {
                self.tell_leaving(&workers, change, &off, now);
            }
            // A status never comes this far: its own door writes it.
            MembershipChange::StatusChanged { .. } => {}
            MembershipChange::Removed { id } => {
                self.pay_leave(&workers, now);
                self.tell_membership(change);
                // The key is filed as gone, by this store's own clock.
                if let Some(row) = held {
                    ended = workers.file_gone(row.facts, id, now, self.pins_absence());
                }
            }
        }
        if let Some((old, new)) = follows {
            self.pins_follow(old, new);
        }
        drop(workers);
        self.forget_gone(&ended);
        true
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
    /// No member moved, so the store's epoch stays and no watcher is told:
    /// the pools are published as they stood, under the new generation,
    /// whatever its pairing mode, and a build the membership writes left
    /// owed is left to its window. An apply takes the workers lock and then
    /// the load lock, as a membership write does: a writer's wait, never a
    /// decision's.
    ///
    /// In a build with the `in-process` feature the store keeps one
    /// engine for its life and the generation is put in force on it, by
    /// the call a host that decides in its own process makes
    /// (`InProcess::apply`). The session table and its pins, the records
    /// of sent prefixes and the stage objects of each profile that did not
    /// change stand across the apply, with what they hold. A build without
    /// that feature keeps nothing: every stage is built new. That call
    /// runs under both locks, behind the last word that may refuse the
    /// generation: a generation the store does not publish is never in
    /// force on the engine. A number that does not rise is refused behind
    /// that word too, by the number alone, and keeps nothing. The stages
    /// it built new are then told every member once, by id
    /// ([`Membership::Snapshot`]), under the workers lock and with the load
    /// lock let go: they start empty, and a stage kept heard each member as
    /// it came. Each membership write after it is told behind the same
    /// lock, in its order. A count of sends stands where a profile that
    /// reads it is kept. One thing built new does not start empty: a count
    /// of sends none of whose readers is kept. It takes over the standing
    /// generation's count under the same two settings, or else a copy of
    /// the sends that generation's counts hold, before the publication
    /// ([`Engine::succeed`](crate::engine::Engine::succeed)). The
    /// take-over is a step of the call that puts the generation in place
    /// of the standing one, under the workers lock and the load lock. A
    /// copy takes as long as the sends it copies, and a taker of the load
    /// lock alone waits for it too. A send told behind a copy and ahead of
    /// the publication enters the standing generation's counts: it is in
    /// no copy.
    ///
    /// A decision that read its world ahead of the apply ends under that
    /// world's generation: its list's value, its lease and the stages told
    /// its issue are that generation's
    /// ([`StoreSnapshot::spec`]).
    ///
    /// The journal is told the generation's number there too, behind the
    /// publication ([`Journal::generation`]).
    pub fn apply(&self, spec: AppliedSpec) -> Result<(), StaleGeneration> {
        self.apply_at(None, spec, || true).map(drop)
    }

    /// [`MemoryStore::apply`] where `admit` has the last word: it is asked
    /// under the two locks the generation is published under, behind the
    /// last one this entry waits for and ahead of the publication, and
    /// where it refuses nothing is published and the answer is `false`.
    /// The entry of a source that writes only while its replica leads
    /// reads the replica's fence here, so a lead that ended while the
    /// entry waited for either lock publishes nothing.
    #[cfg(feature = "controller")]
    pub(crate) fn apply_admitted(
        &self,
        spec: AppliedSpec,
        admit: impl FnOnce() -> bool,
    ) -> Result<bool, StaleGeneration> {
        self.apply_at(None, spec, admit)
    }

    /// A leader's generation, applied as the replication stream tells it:
    /// [`MemoryStore::apply`] with the place of the leader's event, `at`,
    /// which the journal keeps for this write where the applier holds it.
    /// A generation the store refuses tells the journal nothing, so the
    /// guard of its position does not land.
    pub fn follow_generation(
        &self,
        at: Position,
        spec: AppliedSpec,
    ) -> Result<(), StaleGeneration> {
        self.apply_at(Some(at), spec, || true).map(drop)
    }

    /// A leader's generation event that changes nothing here: the store
    /// runs that generation already, or the generation stood no longer when
    /// its leader sent the event and a later event tells what replaced it.
    /// It writes nothing, and the journal keeps the place under the
    /// membership lock, as an apply's is kept.
    pub fn follow_generation_place(&self, at: Position, generation: u64) {
        let _released = self.released();
        let _workers = self.workers.lock();
        if let Some(journal) = &self.journal {
            journal.generation(Some(at), generation);
        }
    }

    /// Lend this store's engine the indexes of cache events its host
    /// keeps, once: a later call is ignored, by the engine's own rule
    /// ([`Affinity::set_indexes`](crate::engine::Affinity::set_indexes)).
    /// The store's one engine is handed them here and keeps them under
    /// every policy spec applied from now on: an apply builds no engine
    /// and hands nothing on. The store keeps none of what it is handed and
    /// calls nothing of it. The call takes no lock: every world of the
    /// store holds the one engine, so an apply beside it loses nothing.
    #[cfg(feature = "controller")]
    pub(crate) fn lend_cache_indexes(&self, indexes: Arc<dyn CacheIndexes>) {
        self.world.load().front().affinity().set_indexes(indexes);
    }

    /// [`MemoryStore::apply`], its telling at `at` where a standby's
    /// applier holds that place, and `admit` asked under both its locks
    /// (`MemoryStore::apply_admitted`, a controller's): `false` where it
    /// refused.
    fn apply_at(
        &self,
        at: Option<Position>,
        spec: AppliedSpec,
        admit: impl FnOnce() -> bool,
    ) -> Result<bool, StaleGeneration> {
        let generation = spec.generation();
        let before = {
            // Taken before the lock, so dropped after it: the journal
            // wakes its readers with the membership lock let go.
            let _released = self.released();
            let workers = self.workers.lock();
            let mut loads = self.loads.lock();
            // The last word, behind every lock this entry waits for: what
            // follows is the publication, or the refusal of a number that
            // does not rise. The generation is put in force on the
            // store's engine behind it and never ahead: the engine takes
            // no generation back.
            if !admit() {
                return Ok(false);
            }
            // Behind the workers lock no apply moves the standing
            // generation. A number that does not rise is refused here, by
            // the number alone. A generation refused here or by `admit`
            // is let go after the locks: this call holds it still.
            let standing = self.world.load().spec().generation();
            if generation <= standing {
                return Err(StaleGeneration {
                    standing,
                    offered: generation,
                });
            }
            // The generation is put in place inside this call, and there a
            // count of sends built new takes over what the standing counts
            // hold. A copy of them is so made under both locks: a taker of
            // the load lock alone waits for it.
            let writer = &mut loads.writer;
            let (world, before) = writer.apply(spec, workers.members())?;
            // A case's hook runs here, under both locks, between the engine's
            // turn and the publication of the generation's world.
            #[cfg(test)]
            seam::turned();
            // The pools are published as they stood: a build the membership
            // writes left owed is left to its window, and with it the table
            // of the members no world named.
            writer.replaced(self.world.swap(world));
            drop(loads);
            // A stage the generation kept heard each member as it came.
            // The world that stands is the generation's: the workers lock
            // holds every other apply off.
            let shown = self.world.load();
            Self::tell_members(&workers, |list| shown.applied().tell_built_new(list));
            if let Some(journal) = &self.journal {
                journal.generation(at, generation);
            }
            before
        };
        drop(before);
        Ok(true)
    }

    /// The snapshot as the last publication left it, taken through no lock.
    /// Its identity moves with every publication.
    ///
    /// This door hands an owned pointer, for a reader that keeps a world
    /// across calls. Taking it and dropping it each write the count every
    /// holder of that world shares, so deciding threads that took it would
    /// write one cache line twice a decision. A decision reads the world
    /// through [`deciding`](Self::deciding), which borrows it and writes no
    /// shared count.
    pub fn snapshot(&self) -> Arc<StoreSnapshot> {
        self.world.load_full()
    }

    /// Read the snapshot of this instant under one load, held until `read`
    /// returns: the vetoes it reads, the generation and its engine
    /// ([`StoreSnapshot::spec`]) are of one publication. Like
    /// [`deciding`](Self::deciding) it borrows the world and writes no
    /// shared count. The load is held for as long as `read` runs: a reader
    /// that waits inside keeps that world from being let go.
    pub fn with_snapshot<R>(&self, read: impl FnOnce(&StoreSnapshot) -> R) -> R {
        let world = self.world.load();
        read(&world)
    }

    /// One decision's door: `decide` is handed the world of this instant
    /// under one borrow, with the engine of the generation it was published
    /// under ([`StoreSnapshot::spec`]). What the decision reads of the world,
    /// the engine and the generation's number are then of one load, and the
    /// borrow writes no shared count, so no two deciding threads write one
    /// line through it. A publication that replaces the world meanwhile is
    /// the next decision's: the world handed here stays whole until `decide`
    /// returns.
    pub fn deciding<R>(&self, decide: impl FnOnce(&StoreSnapshot) -> R) -> R {
        decide(&self.world.load())
    }

    /// `at` in the store's clock: microseconds since the store's origin.
    pub fn micros(&self, at: Instant) -> u64 {
        micros(self.origin, at)
    }

    /// Publish the pools of a snapshot at once, with every row read again
    /// under the generation's lists. The caller holds the workers lock.
    fn swap_world(&self, workers: &Workers) {
        let mut loads = self.loads.lock();
        let writer = &mut loads.writer;
        writer.clear_listed();
        let world = writer.rebuild(workers.members());
        writer.replaced(self.world.swap(world));
        // The build listed every member: none is left for the table.
        self.listed();
    }

    /// Tell every member once, by id ([`Membership::Snapshot`]), through
    /// `tell`: a snapshot tells the engine of the generation that stands,
    /// and an apply the stages its generation built new
    /// ([`Applied::tell_built_new`](super::applied::Applied::tell_built_new)).
    /// Called under the workers lock, after the world of those members is
    /// shown and with the load lock let go.
    fn tell_members(workers: &Workers, tell: impl FnOnce(&Membership<'_>)) {
        let mut members: Vec<WorkerId> = workers.rows.keys().copied().collect();
        members.sort_unstable();
        tell(&Membership::Snapshot(&members));
    }

    /// Tell the store's engine one membership word, through the world the
    /// store shows: under the workers lock, with the load lock let go. The
    /// engine tells the generation in force on it, which is the standing
    /// one: an apply holds the workers lock.
    fn tell_engine(&self, word: &Membership<'_>) {
        self.world.load().applied().on_membership(word);
    }

    /// Tell the standing generation's engine what `change` is to it
    /// ([`MembershipChange::as_lifecycle`]). Called by every membership
    /// write under the workers lock, after the write left its build owed or
    /// paid its leave's, and with the load lock let go: the engine hears the
    /// writes in their order, and an apply, which takes the workers lock, moves no
    /// generation in between. A leave is told after the build that takes its
    /// worker out of its pools: the worker is in no pool when its write
    /// returns, whoever wrote. A join is told when the store takes it, and
    /// the world that lists its worker follows at its window's build. A
    /// decision takes neither lock and meets the call at a stage's own lock
    /// or at a record of sent prefixes' own, and nowhere else. One that
    /// read its world before a leave's swap may still name the worker: a
    /// stage answers that from its own state, as the ring fences a pruned
    /// id ([`crate::stages::RETIRED_WORKER_TTL`]).
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
        change: MembershipChange,
        left: &[String],
        now: Instant,
    ) {
        if left.is_empty() {
            self.owe(now);
        } else {
            self.pay_leave(workers, now);
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
    /// the workers lock by every membership write. The build is owed until
    /// the store's window ends and takes in every write since the last: a
    /// storm of registrations builds once a window, never once a write. A
    /// lease asks for none: it books the slot the member's write gave it
    /// ([`Self::join`]).
    fn owe(&self, at: Instant) {
        let first = self.loads.lock().writer.owe(at);
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
    /// membership as it stands, in one world. Called under the workers
    /// lock; it takes the load lock, as a membership write does. The
    /// host's call names its instant, and a build is paid when it is due at
    /// it. The ask names no instant and is paid at once.
    fn pay(&self, workers: &Workers, at: Option<Instant>) {
        let mut loads = self.loads.lock();
        let writer = &mut loads.writer;
        let (world, built) = match at {
            Some(now) if !writer.build_due(now) => (None, false),
            _ => {
                let world = writer.build_owed(at, workers.members());
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
    fn pay_leave(&self, workers: &Workers, at: Instant) {
        self.loads.lock().writer.owe(at);
        self.pay(workers, None);
    }

    /// Build what the membership writes left owed, now, whatever the
    /// window. Where nothing is owed, nothing is done.
    ///
    /// The build of a registration, a refresh and a patch starts within one
    /// publication window of their answer, when the store's host pays the
    /// due instant ([`Self::publish_due`]), and they are in force for
    /// decisions when it ends; a registration that takes its worker off a
    /// model has paid its own. A caller that needs its own write in force
    /// at once asks here: a test, a bench's fill, a host with no loop. The
    /// ask builds the whole membership under the workers lock and the load
    /// lock: a writer's wait, never a decision's and never a lease's.
    pub fn publish_now(&self) {
        let workers = self.workers.lock();
        self.pay(&workers, None);
    }

    /// When the build the membership writes left owed is due to be shown;
    /// `None` when none is owed. The controller's serve keeps the instant;
    /// a host that writes from a task of its own reads it after a write and
    /// calls [`Self::publish`] at it. A host with no loop asks for the
    /// build ([`Self::publish_now`]).
    pub fn publish_due(&self) -> Option<Instant> {
        self.loads.lock().writer.due()
    }

    /// Show the readers the build that waits, when `now` is at or past the
    /// instant [`Self::publish_due`] names; a call before it shows nothing.
    pub fn publish(&self, now: Instant) {
        if !self.loads.lock().writer.build_due(now) {
            return;
        }
        // A build is owed and due. The workers lock comes first, as every
        // membership write takes it, and the build is asked again under
        // both: another call may have paid it meanwhile.
        let workers = self.workers.lock();
        self.pay(&workers, Some(now));
    }

    /// Hand the store the host's wake. The store calls it, on the writing
    /// thread, where a write leaves an instant for
    /// [`Self::publish_due`] to name and none stood: the build a
    /// membership write owes. The host's loop reads the instant again at
    /// each call. The last wake handed stands.
    pub fn wake_with(&self, wake: Wake) {
        *self.wake.lock() = Some(wake);
    }

    /// Show the deciders the world a write made, when it made one. Called
    /// with the load lock held: the worlds reach the cell in the order the
    /// writer made them, and the writer takes back the one each replaces.
    fn show(&self, writer: &mut Writer, world: Option<Arc<StoreSnapshot>>) {
        if let Some(world) = world {
            writer.replaced(self.world.swap(world));
        }
    }

    /// Publish `gateway`'s veto on `id` with its hold's end, or its
    /// clearing, at once. Called under the health lock, which serializes
    /// the vetoes' writers, so the published vetoes follow the health
    /// table's own order.
    fn show_veto(
        &self,
        gateway: GatewayId,
        id: WorkerId,
        until: Option<Instant>,
        hold: Option<Instant>,
    ) {
        let mut loads = self.loads.lock();
        let world = loads.writer.veto(gateway, id, until, hold);
        self.show(&mut loads.writer, world);
    }

    /// Publish the lifting of every veto of the gateways in `gone` at once,
    /// in one world. Called under the health lock, as a veto's own write is.
    fn show_gateways_gone(&self, gone: &[GatewayId]) {
        let mut loads = self.loads.lock();
        let world = loads.writer.drop_gateways(gone);
        self.show(&mut loads.writer, world);
    }

    /// The counts one scrape reports.
    pub fn stats(&self) -> StoreStats {
        let loads = self.loads.lock();
        StoreStats {
            outside: loads.writer.outside(),
            lifecycle_faults: self.lifecycle_faults(),
            engine_faults: self.world.load().applied().faults(),
        }
    }

    /// How many members of the standing membership name an engine family
    /// or a zone that the standing generation's list does not hold, by the
    /// list. Such a worker is admitted and reads position 0 in that field;
    /// a generation that declares no list counts none.
    pub fn outside_lists(&self) -> OutsideLists {
        self.stats().outside
    }

    /// A worker's row is gone (removed, or replaced by a new incarnation):
    /// the store's tables that live exactly as long as the row leave with
    /// it: the health tables. The vetoes a world
    /// carries are not among them: they leave with the row in the
    /// publication of the membership write that calls this. Nor are the
    /// engine's session pins that name the worker: they stand through
    /// the key's absence ([`Workers::gone`]) and leave where the key does
    /// not return as the worker it was ([`Self::forget_gone`]), at the
    /// store's own write and at a followed one. A row a snapshot drops
    /// keeps them too: its key is filed as gone by this store's own
    /// clock ([`Workers::lay_gone`]), and a worker that returns as the
    /// one it was is pinned by them again. A standby is sent no
    /// pin, so its own are of a term this store led itself. Called under
    /// the workers lock: the health writers take it first, so a write racing
    /// the removal finds the row gone or lands before the tables are
    /// cleared.
    fn forget_worker(&self, id: WorkerId) {
        let mut health = self.health.lock();
        health.forget(id);
    }

    /// The keys that stayed away past the absence's time leave the list
    /// at the store's tick, with their pins: a membership that stands
    /// still ends them too, under any rules.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "called by the served controller alone")
    )]
    pub(crate) fn end_gone(&self, now: Instant) {
        let ended = {
            let mut workers = self.workers.lock();
            if workers.gone.is_empty() {
                return;
            }
            workers.ended_gone(now, self.pins_absence())
        };
        self.forget_gone(&ended);
    }

    /// Keys that left do not return as the workers they were, or stayed
    /// away past the absence's time: their session pins leave the standing
    /// engine's table, in one walk for them all. With the workers lock let
    /// go: the walk is as long as the table, and no writer of the
    /// membership or of a worker's health waits on it.
    fn forget_gone(&self, ids: &[WorkerId]) {
        if ids.is_empty() {
            return;
        }
        self.world.load().spec().engine().pins_forget(ids);
    }

    /// A key is registered again as the worker it was: its session pins
    /// name `new` where they named `old`, in the standing engine's
    /// table. Under the workers lock and behind the
    /// registration's own word, at the store's own write and at a
    /// followed one. A registration leaves its build owed: no decision
    /// reads the new row before its pins stand under its id. A
    /// replacement has paid its build, and a decision between that build
    /// and this call reads a pin no pool holds: that one request is
    /// placed by the rule. Renamed ahead of that build, the pin would
    /// read as in no pool to the old world's decisions for the build's
    /// whole time.
    fn pins_follow(&self, old: WorkerId, new: WorkerId) {
        self.world.load().spec().engine().pins_follow(old, new);
    }

    /// How long a key that left keeps its session pins: the standing
    /// engine's time ([`Engine::pins_absence_micros`](crate::engine::Engine::pins_absence_micros)).
    fn pins_absence(&self) -> Duration {
        Duration::from_micros(self.world.load().spec().engine().pins_absence_micros())
    }

    /// How many worlds the store published since it was made. A write that
    /// changes nothing publishes none, and neither does a status change: a
    /// world holds no worker's status. The gateways one sweep lets go leave
    /// in one.
    pub fn publications(&self) -> u64 {
        self.loads.lock().writer.publications()
    }

    /// How many times the pools were built from the membership since the
    /// store was made. A storm of membership writes builds once a window,
    /// never once a write: the count says so by run.
    pub fn membership_builds(&self) -> u64 {
        self.loads.lock().writer.builds()
    }
}

// ---------------------------------------------------------------- workers

/// The most keys the store files as gone ([`Workers::gone`]). A leave
/// past it forgets the one that left first ([`Workers::file_gone`]). A
/// snapshot past it keeps the list and lets the pins of the keys it
/// would add leave ([`Workers::lay_gone`]): its rows left at one
/// instant, and none of them left before a key on file.
pub(super) const GONE_KEYS: usize = 4_096;

/// A key that left membership while its session pins stand
/// ([`Workers::gone`]).
struct Gone {
    /// The id the key had.
    id: WorkerId,
    /// The instant it left.
    left: Instant,
    /// The registration it stood under: the worker it was.
    facts: WorkerFacts,
}

#[derive(Default)]
struct Workers {
    rows: FxHashMap<WorkerId, Row>,
    by_key: HashMap<String, WorkerId>,
    /// The keys that left by a removal, the store's own or a followed
    /// one, while their session pins stand: one entry a key,
    /// [`GONE_KEYS`] at most. A key registered again as the worker it
    /// was, inside the absence's time ([`MemoryStore::pins_absence`]),
    /// takes its pins under its new id. One that is not, or that stays
    /// away longer, is forgotten. In no journal and no snapshot: a
    /// follower files its own by its own clock, at the removals it
    /// follows and at the rows a snapshot drops ([`Self::lay_gone`]).
    gone: HashMap<String, Gone>,
    epoch: u64,
    /// The store's journal: every membership write passes
    /// [`Writing::record_over`], and every status write
    /// [`Writing::follow_status`] or [`Writing::set_status`], which tell
    /// it.
    journal: Option<Arc<dyn Journal>>,
}

struct Row {
    row: WorkerRow,
    /// The slot of the worker's count of the leases on file, given at the
    /// write that first files it ([`MemoryStore::join`]). A member keeps
    /// it across builds, and the slot of a worker that left is given to no
    /// one: the leases still on file for it end on the count they booked.
    slot: WorkerSlot,
}

impl Workers {
    /// The ids of the keys gone longer than `bound` at `now`: they leave
    /// the list, and their pins are the caller's to forget.
    fn ended_gone(&mut self, now: Instant, bound: Duration) -> Vec<WorkerId> {
        let mut ended = Vec::new();
        self.gone.retain(|_, gone| {
            let stays = now.saturating_duration_since(gone.left) <= bound;
            if !stays {
                ended.push(gone.id);
            }
            stays
        });
        ended
    }

    /// The key of `facts` is registered again at `now` and leaves the
    /// list. The id its pins follow from is returned where the key left
    /// inside `bound` as the worker `facts` registers. The ids whose pins
    /// leave now come with it: the key's own where another worker
    /// registers or it stayed away longer, and those of the keys gone
    /// longer than `bound`.
    fn take_gone(
        &mut self,
        facts: &WorkerFacts,
        now: Instant,
        bound: Duration,
    ) -> (Option<WorkerId>, Vec<WorkerId>) {
        let returned = self.gone.remove(&facts.key);
        let mut ended = self.ended_gone(now, bound);
        let old = returned.and_then(|gone| {
            let inside = now.saturating_duration_since(gone.left) <= bound;
            if inside && same_worker(&gone.facts, facts) {
                return Some(gone.id);
            }
            ended.push(gone.id);
            None
        });
        (old, ended)
    }

    /// Files the key of `facts` as gone at `now`, under the id it had. The
    /// ids whose pins leave now are returned: the keys gone longer than
    /// `bound`, and the one that left first where the list is full.
    fn file_gone(
        &mut self,
        facts: WorkerFacts,
        id: WorkerId,
        now: Instant,
        bound: Duration,
    ) -> Vec<WorkerId> {
        let mut ended = self.ended_gone(now, bound);
        if self.gone.len() >= GONE_KEYS {
            let first = self.gone.iter().min_by_key(|(_, gone)| gone.left);
            let first = first.map(|(key, _)| key.clone());
            if let Some(gone) = first.and_then(|key| self.gone.remove(&key)) {
                ended.push(gone.id);
            }
        }
        self.gone.insert(
            facts.key.clone(),
            Gone {
                id,
                left: now,
                facts,
            },
        );
        ended
    }

    /// A snapshot's rows are laid: the session pins take the steps of the
    /// writes the snapshot stands for. `left` holds the rows it did not
    /// keep, each by the id it had. A key the rows hold again, of `left`
    /// or of the list, was registered again: its pins follow to its new
    /// id where it is the worker it was, the pairs returned first.
    /// Another worker's pins leave, with those of the keys gone longer
    /// than `bound`: the ids returned second. A row of `left` whose key
    /// the rows do not hold left: it is filed as gone at `now`, as a
    /// followed removal files it, and where the list is full its pins
    /// leave now.
    fn lay_gone(
        &mut self,
        left: impl Iterator<Item = (WorkerId, WorkerFacts)>,
        now: Instant,
        bound: Duration,
    ) -> (Vec<(WorkerId, WorkerId)>, Vec<WorkerId>) {
        let mut follows = Vec::new();
        let mut ended = self.ended_gone(now, bound);
        let (rows, by_key) = (&self.rows, &self.by_key);
        let same = |facts: &WorkerFacts, new: WorkerId| {
            let held = rows.get(&new);
            held.is_some_and(|held| same_worker(facts, &held.row.facts))
        };
        self.gone.retain(|key, gone| {
            let Some(&new) = by_key.get(key) else {
                return true;
            };
            if same(&gone.facts, new) {
                follows.push((gone.id, new));
            } else {
                ended.push(gone.id);
            }
            false
        });
        for (old, facts) in left {
            match by_key.get(&facts.key) {
                Some(&new) if same(&facts, new) => follows.push((old, new)),
                Some(_) => ended.push(old),
                None if self.gone.len() < GONE_KEYS => {
                    let gone = Gone {
                        id: old,
                        left: now,
                        facts,
                    };
                    self.gone.insert(gone.facts.key.clone(), gone);
                }
                None => ended.push(old),
            }
        }
        (follows, ended)
    }

    /// Every member as a build reads it: its row and the slot of its count.
    fn members(&self) -> impl Iterator<Item = (&WorkerRow, WorkerSlot)> {
        self.rows.values().map(|row| (&row.row, row.slot))
    }

    /// The row under `id` when `revision` is still its revision. A status
    /// a holder of worker objects tells is its one caller.
    #[cfg(any(test, feature = "worker-host"))]
    fn guarded(&mut self, id: WorkerId, revision: u64) -> Option<&mut Row> {
        self.rows
            .get_mut(&id)
            .filter(|row| row.row.revision == revision)
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
    /// The term the write is made under, for the journal: none for a
    /// writer that names no term.
    term: Option<u64>,
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

/// The models the table's `row` serves, as the one shared list a status
/// entry holds: what a status tells where its writer handed no list. Each
/// id is copied, under the membership lock. It is the one copy of the
/// table's models for a status entry, and it leaves with the table.
fn table_models(row: &WorkerRow) -> Arc<[String]> {
    row.facts.models.as_slice().into()
}

/// A change is recorded through the write's guard alone: a path that takes
/// the membership lock bare has no way to journal a write, so none is
/// journaled whose release the journal does not hear.
impl Writing<'_> {
    /// The epoch's step for a change the journal took ([`Self::tell`]).
    fn step(&mut self) {
        self.workers.epoch += 1;
    }

    /// Tell the journal `change` ahead of the epoch's step, with the epoch
    /// that step gives and the term the write is made under
    /// ([`Journal::member`]). Whether the write goes on: a journal says no
    /// to a write whose term its store does not lead, and nothing has
    /// moved then. A store with no journal has no term to lead.
    fn tell(
        &self,
        change: MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) -> bool {
        let epoch = self.workers.epoch + 1;
        let journal = self.workers.journal.as_ref();
        journal.is_none_or(|journal| journal.member(self.at, self.term, epoch, change, row, left))
    }

    /// Write a status the store follows on the row under `id`: the row
    /// left `from` and takes `to`. Whether it fits the table: the row
    /// stands in the status it leaves, and one that does not writes
    /// nothing and tells nothing. The journal is told first
    /// ([`Journal::status`]): the change, the models the row serves and
    /// the epoch the step gives, and no row. `models` is the list its
    /// writer handed. A writer that hands none tells the table's row's,
    /// copied here where a journal stands ([`table_models`]). Its writer
    /// names no term, so the journal takes it. Behind the telling stand
    /// the field's write and the epoch's step. The row is looked up once,
    /// and no row is copied: the twin of [`Self::set_status`] for a told
    /// status.
    fn follow_status(
        &mut self,
        id: WorkerId,
        from: WorkerStatus,
        to: WorkerStatus,
        models: Option<Arc<[String]>>,
    ) -> bool {
        let (at, term) = (self.at, self.term);
        let workers = &mut *self.workers;
        let held = workers.rows.get_mut(&id);
        let Some(held) = held.filter(|held| held.row.status == from) else {
            return false;
        };
        // While the table stands, a debug build holds the handed models to
        // the ones the table's row serves, as a leader's status write does:
        // the object was laid from that row.
        debug_assert!(
            models
                .as_deref()
                .is_none_or(|models| held.row.facts.models[..] == *models),
            "the models a holder read off its object are the ones the table's row serves"
        );
        if let Some(journal) = &workers.journal {
            let models = models.unwrap_or_else(|| table_models(&held.row));
            let epoch = workers.epoch + 1;
            let goes_on = journal.status(at, term, epoch, id, from, to, models);
            debug_assert!(goes_on, "a write that names no term goes on");
        }
        // The revision counts a row's stated facts: a status change leaves
        // it, here as in the leader's own write.
        held.row.status = to;
        workers.epoch += 1;
        true
    }

    /// Record a change whose writer handed the row it leaves. The journal
    /// is told `row`, that row (none for a removal), and `left`, the row
    /// the write took out (a removal's, a replacement's old incarnation):
    /// as its writer handed it, or as the table held it where the writer
    /// handed none. No row of the table is read here: the table takes
    /// `row` behind the telling, and the epoch steps behind it too. Its
    /// writers name no term, so the journal takes it.
    fn record_over(
        &mut self,
        change: MembershipChange,
        row: Option<&WorkerRow>,
        left: Option<&WorkerRow>,
    ) -> MembershipChange {
        let goes_on = self.tell(change, row, left);
        debug_assert!(goes_on, "a write that names no term goes on");
        self.step();
        change
    }

    /// Write `status` on the row under `id`, under the hold's term. The
    /// journal is told first ([`Journal::status`]): the change, the models
    /// the row serves and the epoch the step gives, and no row. `models`
    /// is the list its writer handed. A writer that hands none tells the
    /// table's row's, copied here where a journal stands
    /// ([`table_models`]). Where the
    /// journal refuses the hold's term, nothing has moved. Behind a yes
    /// stand two moves alone, and neither refuses: the field's write and
    /// the epoch's step. The row is looked up once, and no row is copied.
    /// A status a holder of worker objects tells is its one caller.
    #[cfg(any(test, feature = "worker-host"))]
    fn set_status(
        &mut self,
        id: WorkerId,
        status: WorkerStatus,
        models: Option<Arc<[String]>>,
    ) -> StatusSet {
        let (at, term) = (self.at, self.term);
        let workers = &mut *self.workers;
        let Some(held) = workers.rows.get_mut(&id) else {
            return StatusSet::Stood;
        };
        let from = held.row.status;
        if from == status {
            return StatusSet::Stood;
        }
        if let Some(journal) = &workers.journal {
            let models = models.unwrap_or_else(|| table_models(&held.row));
            let epoch = workers.epoch + 1;
            if !journal.status(at, term, epoch, id, from, status, models) {
                return StatusSet::Refused;
            }
        }
        // The revision counts the row's stated facts: a status change
        // leaves it, so the holder of a row's revision stays current while
        // the worker's health moves.
        held.row.status = status;
        workers.epoch += 1;
        StatusSet::Moved
    }
}

/// What a status write under a hold of the membership lock made
/// ([`Writing::set_status`]).
#[cfg(any(test, feature = "worker-host"))]
enum StatusSet {
    /// The status moved.
    Moved,
    /// Nothing to write: the row holds the status already, or no row
    /// stands.
    Stood,
    /// The journal refused the hold's term, and nothing moved.
    Refused,
}

impl MemoryStore {
    /// The word to the journal, at its drop, that a write's locks are let
    /// go.
    fn released(&self) -> Released<'_> {
        Released(self.journal.as_deref())
    }

    /// The membership lock for a write the journal is told: every path
    /// that records a change takes the lock here. A holder of worker
    /// objects is its one caller: a change it decided, and a status it
    /// tells.
    #[cfg(any(test, feature = "worker-host"))]
    fn writing(&self) -> Writing<'_> {
        self.writing_at(None)
    }

    /// [`MemoryStore::writing`] for a write the journal is told with `at`:
    /// the place of the leader's event a standby's applier applies by it.
    fn writing_at(&self, at: Option<Position>) -> Writing<'_> {
        self.hold(at, None)
    }

    /// [`MemoryStore::writing`] for a write made under `term`: the journal
    /// takes its event only while the store leads that term.
    #[cfg(feature = "worker-host")]
    fn writing_under(&self, term: Option<u64>) -> Writing<'_> {
        self.hold(None, term)
    }

    fn hold(&self, at: Option<Position>, term: Option<u64>) -> Writing<'_> {
        let released = self.released();
        Writing {
            workers: self.workers.lock(),
            at,
            term,
            _released: released,
        }
    }
}

#[cfg(feature = "worker-host")]
use super::health::{HealthWrite, LeadWord};

/// The status the journal last told of each row, at one place of the log
/// ([`MemoryStore::told`]).
#[cfg(feature = "worker-host")]
#[derive(Debug)]
pub(crate) struct Told {
    /// The last journaled write the statuses hold, with its term; the
    /// default for a store with no journal.
    pub(crate) at: Position,
    /// The membership epoch.
    pub(crate) epoch: u64,
    statuses: FxHashMap<WorkerId, WorkerStatus>,
}

#[cfg(feature = "worker-host")]
impl Told {
    /// The status the journal last told of the row `id`. `None` for a
    /// row that does not stand at that place.
    pub(crate) fn status(&self, id: WorkerId) -> Option<WorkerStatus> {
        self.statuses.get(&id).copied()
    }
}

#[cfg(feature = "worker-host")]
impl MemoryStore {
    /// Whether the store tells a journal its writes. It is fixed where the
    /// store is built, and read under no lock: a holder of worker objects
    /// builds what a status entry names only for a store that tells one
    /// ([`Self::status_told`]).
    pub(crate) fn journals(&self) -> bool {
        self.journal.is_some()
    }

    /// The told status of every row: what the journal last told of its
    /// id, with the journal's place and the membership epoch,
    /// all read in one hold of the membership lock.
    /// A status write holds that lock from its telling to its apply, so
    /// no status stands half told here. The hold copies a place and one
    /// status a row, and no row.
    ///
    /// The told status is the log's own word, kept beside the epoch:
    /// no copy of what a decision reads. A decision reads a
    /// worker's status on its object. The told status has six readers:
    ///
    /// - the next status event, which says the status its row moved from
    ///   ([`Self::status_told`]);
    /// - the event of a row changed in place, which carries it: a status
    ///   moves by its own event alone ([`Self::follow_objects`]);
    /// - the event of a row a write takes out, which is told in it where
    ///   its writer handed the row ([`Self::decided`]);
    /// - a first copy, which the holder of the worker objects cuts with
    ///   each row's told status (`Objects::checkpoint`,
    ///   `Objects::rows_serving`), by this read;
    /// - a tell the store refused, whose object takes the told status
    ///   back (`Objects::table_status`);
    /// - a holder that follows its leader's log, which lays on an object
    ///   the status the log told of its row (`Objects::copy_status`).
    ///
    /// What else reads a row's status reads the table's whole row and
    /// goes with the table: the store's own reads of a row.
    pub(crate) fn told(&self) -> Told {
        let workers = self.workers.lock();
        let at = self
            .journal
            .as_ref()
            .map_or_else(Position::default, |journal| journal.head());
        let statuses = workers
            .rows
            .iter()
            .map(|(id, held)| (*id, held.row.status))
            .collect();
        Told {
            at,
            epoch: workers.epoch,
            statuses,
        }
    }

    /// One membership write this replica's worker objects decided, applied
    /// as their holder tells it: a leader's side of [`Self::follow`], over
    /// the one body ([`Self::decided`]). The holder decided the change over
    /// its objects and built the object of the row the change leaves. The
    /// store steps its own epoch and journals the write as its own. `row`
    /// is that row, and none for a removal.
    ///
    /// `seen` is the revision of the row the holder decided over, and none
    /// where it read no row under the key. The table's row must stand at
    /// that revision still, under this hold of the membership lock.
    /// A writer outside the
    /// holder's turn can move the row between the holder's read and this
    /// call: a change handed to the store around the holder
    /// ([`Self::follow`]) takes no turn of the holder.
    ///
    /// `left` is the row the write takes out, as the holder read it back
    /// in its turn: a removal's row, a replacement's old life, and none
    /// for any other change. The journal is told it: of the rows such a
    /// write tells, the table gives a status alone.
    ///
    /// A worker object's status leads its row's. The holder read `row`'s
    /// off the object, and the rule every writer decides by handed it on
    /// to a key's next life ([`WorkerRow::hands_on`]): the store takes
    /// that. A row changed in place keeps the status the table holds: a
    /// status moves by its own event alone, and the holder tells the
    /// object's behind the write ([`Self::status_told`]). A row the write
    /// takes out is told in the status the table held too, which the one
    /// body lays over it for a leader and a standby alike
    /// ([`Self::decided`]).
    ///
    /// Whether the store took the change. One it does not take writes
    /// nothing: another writer moved the row under the holder's turn, or
    /// the change does not fit the tables, as [`Self::follow`] says it.
    pub(crate) fn follow_objects(
        &self,
        change: MembershipChange,
        mut row: Option<WorkerRow>,
        seen: Option<u64>,
        left: Option<&mut WorkerRow>,
    ) -> bool {
        let workers = self.writing();
        let table = |id: WorkerId| workers.rows.get(&id).map(|held| &held.row);
        // The row the change was decided over, as the table holds it now.
        let over = match change {
            MembershipChange::Registered { .. } => None,
            MembershipChange::Replaced { old, .. } => table(old),
            MembershipChange::Relabelled { id }
            | MembershipChange::Resized { id }
            | MembershipChange::Removed { id } => table(id),
            // A status is told by its own entry.
            MembershipChange::StatusChanged { .. } => return false,
        };
        if over.map(|over| over.revision) != seen {
            return false;
        }
        if let (Some(row), Some(over)) = (&mut row, over) {
            if !matches!(change, MembershipChange::Replaced { .. }) {
                row.status = over.status;
            }
        }
        self.decided(workers, Instant::now(), change, row, left)
    }

    /// The status this replica's worker object states for the row `id`,
    /// told by the holder of the objects: a leader's side of a told status
    /// ([`Self::follow`]). The object's status leads. Its holder tells each
    /// move here: a probe's verdict, a connect signal and a drain. The
    /// store steps its own epoch and journals the write as its own.
    ///
    /// `revision` is the row's revision the object was built from. The
    /// table's row must stand at it still, under this hold of the
    /// membership lock: a stale status is dropped, never written onto a
    /// successor.
    ///
    /// `word` is the writer's word of its lead. The store asks its read
    /// behind the lock the write waits for. Where it says no, the status
    /// is stale: a term that ended while the write waited there moves no
    /// status and tells the journal nothing. It is asked under the store's
    /// lock and takes none of the store's: a lock it takes is never held
    /// by its writer across a write of the store. A writer with no term
    /// to end hands a word that always holds.
    ///
    /// The word's term goes to the journal with the event, ahead of the
    /// table's move ([`Journal::status`]). The journal compares it in the
    /// take of its lock that gives the event its place. Where the store
    /// has left the term, the journal takes no event, the store moves
    /// nothing, and the status is stale.
    ///
    /// `models` are the models the object serves, read by its holder under
    /// the hold that read the status and handed ahead of this lock. The
    /// journal is told them with the status, and no row: of the table's
    /// row a leader's status entry holds the status it left alone. A
    /// holder hands a list only where the store tells a journal
    /// ([`Self::journals`]), so a host that tells none builds none. A
    /// writer that hands none to a store that tells one has the table's
    /// row's told.
    ///
    /// What the write found: the status moved; it stood, because the row
    /// holds it already; the row left `revision`; or no row stands.
    pub(crate) fn status_told(
        &self,
        id: WorkerId,
        status: WorkerStatus,
        revision: u64,
        word: LeadWord<'_>,
        models: Option<Arc<[String]>>,
    ) -> HealthWrite {
        let mut workers = self.writing_under(word.term);
        let Some(row) = workers.guarded(id, revision) else {
            // Judged against nothing: stale where a row stands under
            // another revision, no row where none stands.
            return if workers.rows.contains_key(&id) {
                HealthWrite::Stale
            } else {
                HealthWrite::NoRow
            };
        };
        if !(word.leads)() {
            return HealthWrite::Stale;
        }
        let from = row.row.status;
        // While the table stands, a debug build holds the handed models to
        // the ones the table's row serves: the object was laid from that
        // row. A host with no journal is handed none and held to nothing
        // here: a gateway's refresh of a worker's models sets them on its
        // object behind the build.
        debug_assert!(
            models
                .as_deref()
                .is_none_or(|models| row.row.facts.models[..] == *models),
            "the models a holder read off its object are the ones the table's row serves"
        );
        match workers.set_status(id, status, models) {
            StatusSet::Moved => HealthWrite::Moved { from, to: status },
            StatusSet::Stood => HealthWrite::Stood(from),
            StatusSet::Refused => HealthWrite::Stale,
        }
    }
}

impl WorkersStore for MemoryStore {
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

    /// Kept beside the cell of the row's slot in the counts, the one the
    /// leases' doors book. Under the workers lock, as the write that takes
    /// a row is: no count is kept for a row that left.
    fn host_count(&self, id: WorkerId, count: Box<dyn HostCount>) -> bool {
        let workers = self.workers.lock();
        let slot = workers.rows.get(&id).map(|held| held.slot);
        slot.is_some_and(|slot| self.counts.beside(slot, count))
    }

    fn all_rows(&self) -> (u64, Vec<WorkerRow>) {
        // Copied under the lock, sorted after it: a writer waits for the
        // copy alone.
        let (epoch, mut rows) = {
            let workers = self.workers.lock();
            let rows: Vec<WorkerRow> = workers.rows.values().map(|row| row.row.clone()).collect();
            (workers.epoch, rows)
        };
        // A case's pause, between the copy of the rows and their sort.
        #[cfg(test)]
        self.pause_snapshot();
        rows.sort_by(|a, b| a.facts.key.cmp(&b.facts.key));
        (epoch, rows)
    }
}

// ----------------------------------------------------------------- health

#[derive(Default)]
struct HealthTables {
    /// A gateway's veto on a worker, until the instant. The veto's hold is
    /// the world's alone: no answer of this table turns on it.
    vetoes: FxHashMap<(GatewayId, WorkerId), Instant>,
}

impl HealthTables {
    /// Drop every gateway's veto on the worker.
    fn forget(&mut self, id: WorkerId) {
        self.vetoes.retain(|(_, worker), _| *worker != id);
    }

    /// Drop the vetoes of every gateway in `gone`, a sorted list, on every
    /// worker: one pass over the vetoes for them all.
    fn forget_gateways(&mut self, gone: &[GatewayId]) {
        self.vetoes
            .retain(|(owner, _), _| gone.binary_search(owner).is_err());
    }
}

impl HealthSink for MemoryStore {
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
                // No kind is told through this door, so the veto holds to
                // its end: no rule lifts a veto it cannot name.
                self.show_veto(gateway, id, Some(until), Some(until));
            }
            None => {
                let mut health = self.health.lock();
                health.vetoes.remove(&(gateway, id));
                self.show_veto(gateway, id, None, None);
            }
        }
    }

    fn gateway_veto_noted(
        &self,
        gateway: GatewayId,
        id: WorkerId,
        until: Option<Instant>,
        hold: Option<Instant>,
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
                self.show_veto(gateway, id, Some(until), hold);
            }
            None => {
                health.vetoes.remove(&(gateway, id));
                self.show_veto(gateway, id, None, None);
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
}

// ------------------------------------------------------------------ loads

/// The world's write side, behind its lock.
struct Loads {
    writer: Writer,
}

// ----------------------------------------------------------------- leases

/// The leases on file, the ids the store mints for a decision's, and the
/// order their dispatch deadlines fall due in.
struct Leases {
    term: u64,
    #[cfg(not(test))]
    records: FxHashMap<Key, Held>,
    // A case counts a lease's lookups through this hasher.
    #[cfg(test)]
    records: HashMap<Key, Held, super::tests::LeaseHashes>,
    /// The decode side of each pair's lease on file: a place for each
    /// record that carries the mark of a pair's lease
    /// ([`LeaseRecord::decode`]) and for no other, named by the record's
    /// entry ([`Held::decode`]). A record and its place are laid under
    /// one hold of the turn ([`Leases::file`], and a decision's issue of
    /// a pair, `issue_decided_pair`) and leave under one
    /// ([`Leases::leave`]): no other door lays or takes either.
    decodes: Decodes,
    /// The store's counts, a cell a worker: what the leases on file book.
    counts: Arc<InFlightCounts>,
    /// The block of the term's counter the file mints from, as far as it
    /// minted into it ([`MemoryStore::issue_decided`]).
    minted: Minted,
    /// Every lease filed as issued, by its dispatch deadline, the earliest
    /// first. A lease that left the file or was dispatched keeps its place
    /// until the issues that follow step over it, so no door searches the
    /// queue to take a place out: it holds the leases that wait for a
    /// dispatch, and behind the first of those the places of what one
    /// dispatch deadline issued. One door reads it through, a request
    /// asked again's, for the lease its earlier ask was issued.
    due: VecDeque<(Instant, Key)>,
    /// When a sweep next asks the file, shared with the file so a sweep
    /// can pass it without taking its turn ([`NextDue`]).
    next_due: Arc<NextDue>,
}

/// When a sweep next asks a file, as microseconds from the store's origin
/// plus one. Zero while the sweep found its queue empty. One where the
/// queue's first place is new and no sweep has read it. Else the deadline
/// the sweep read at the first place on its last visit.
///
/// A deciding thread reads no clock for it: an issue marks a new first
/// place with one store, and the places its issues step over leave the
/// word as it is. The word is then the first place's deadline or earlier,
/// never later, and an early one costs the sweep one visit.
///
/// Relaxed: every store is made under the file's turn, and the sweep's
/// read may lag by a pass.
struct NextDue {
    origin: Instant,
    at: AtomicU64,
}

impl NextDue {
    fn new(origin: Instant) -> Self {
        Self {
            origin,
            at: AtomicU64::new(0),
        }
    }

    fn word(&self, at: Instant) -> u64 {
        micros(self.origin, at).saturating_add(1)
    }

    /// The queue's first place is new: the next sweep asks the file.
    fn ask(&self) {
        self.at.store(1, Relaxed);
    }

    /// The first place's deadline as the turn's holder reads it, or none
    /// for an empty queue.
    fn set(&self, at: Option<Instant>) {
        self.at.store(at.map_or(0, |at| self.word(at)), Relaxed);
    }

    fn reached(&self, now: Instant) -> bool {
        let at = self.at.load(Relaxed);
        at != 0 && at <= self.word(now)
    }
}

/// A lease's key on file: its gateway and its id.
type Key = (GatewayId, LeaseId);

/// The leases past their dispatch deadline that one issue ends at most.
pub(crate) const ABANDONED: usize = 2;

/// The places of the deadline queue one issue steps over at most, the
/// places of leases that were dispatched or left the file among them: four
/// for the one it adds, so the queue shortens under any run of issues.
const DUE_STEPS: usize = 4;

/// A lease on file with the slot of the count it books on its worker: a
/// single's lease whole, or the prefill part of a pair's.
struct Held {
    record: LeaseRecord,
    /// `None` for a lease on a worker that is no member when the lease was
    /// written or moved ([`Slots::of`]): it books no row a decision reads.
    /// A pair's prefill side that ended books none.
    booked: Option<WorkerSlot>,
    /// The place of a pair's decode side in the file's list
    /// ([`Decodes`]), read where the record carries the mark of a pair's
    /// lease. A single's entry names no place of its own.
    decode: u32,
    /// Whether the lease's sides book their keys' requests in flight, each
    /// on its worker's count ([`InFlightCounts::book`]): said once, where
    /// the lease is filed, so an end frees what the filing booked. A
    /// decision's lease counts where [`MemoryStore::issue_keyed`] files it,
    /// and a pair's decision always (`issue_decided_pair`). A
    /// record filed whole carries no such word and counts wherever it has
    /// a key: more than a cap reads, never less.
    keyed: bool,
}

/// The decode side of a pair's lease on file, with the slot of the count
/// it books until it ends.
#[derive(Clone, Copy)]
struct FiledDecode {
    side: DecodeSide,
    booked: Option<WorkerSlot>,
    /// Whether the side started as chosen, kept after issue or dispatch.
    filed_chosen: bool,
    /// How many calls issued the side. An undo names the count it issued
    /// under, so it undoes its own issue alone.
    issues: u8,
    /// The index of the pick's target the prefill's dispatch took: 0, the
    /// pick's first, until that dispatch names another target the side's
    /// pick holds.
    taken: u8,
    /// Whether the side's own dispatch deadline runs: the deadline's door
    /// started it at its first pass behind the end of the prefill the side
    /// was issued with ([`Leases::waits_on`]). The door's next pass ends a
    /// side that still waits.
    own_deadline: bool,
}

/// One issue of a chosen decode side, for a caller that may undo it
/// ([`MemoryStore::unissue_decode`]): the count of the side's issues with
/// this one. What the issue replaced stays with the side's place
/// ([`Decodes::keep`]).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    not(feature = "controller"),
    expect(dead_code, reason = "the served door alone undoes an issue")
)]
pub(crate) struct DecodeIssue(u8);

/// What an issue of a chosen decode side replaced: the side and the
/// lease's dispatch deadline as they stood.
#[cfg(feature = "controller")]
type Replaced = (DecodeSide, Instant);

/// What a decode call reads of the lease it names
/// ([`MemoryStore::held_pair`]). The held prefill's worker comes first,
/// then the worker and the rank of the decode its first decision chose.
/// Last comes the instant the lease was issued at.
#[cfg(feature = "controller")]
pub(crate) type HeldPair = (WorkerId, Option<(WorkerId, u16)>, u64);

impl FiledDecode {
    /// `side` as a door files it with `booked`, the slot of its worker's
    /// count. Only a live side books a count at filing.
    fn laid(side: DecodeSide, booked: Option<WorkerSlot>) -> Self {
        let booked = booked.filter(|_| side.state.is_live());
        let filed_chosen = side.state.is_chosen();
        Self {
            side,
            booked,
            filed_chosen,
            issues: 0,
            taken: 0,
            own_deadline: false,
        }
    }

    /// The decode, with its rank, of the target the prefill's dispatch
    /// took ([`Self::taken`]): the one a decode decided again behind that
    /// prefill starts from. Every target of a pair's pick names the one
    /// prefill, so the index its dispatch names says which decode its
    /// host took.
    #[cfg(feature = "controller")]
    fn taken_decode(&self) -> Option<(WorkerId, u16)> {
        self.side.target_at(self.taken)
    }
}

/// The decode sides of the pairs' leases on file, each in a place its
/// record's entry names: no door looks a decode side up by a key. A place
/// holds the key of its lease, so a place read under another key reads as
/// no side. A place let go is the next one laid: the list is as long as
/// the most pairs that stood at once, and it keeps its room.
#[derive(Default)]
struct Decodes {
    places: Vec<Option<(Key, FiledDecode)>>,
    free: Vec<u32>,
    /// What the last issue of a chosen side replaced, by the side's place
    /// and under its lease's key. A decode call's issue alone lays one: the
    /// list is empty where no pair asks its decode later.
    #[cfg(feature = "controller")]
    replaced: Vec<Option<(Key, Replaced)>>,
}

impl Decodes {
    /// Lays `decode` for the lease under `key` in a free place and names
    /// the place.
    fn lay(&mut self, key: Key, decode: FiledDecode) -> u32 {
        let laid = Some((key, decode));
        if let Some(at) = self.free.pop() {
            if let Some(place) = self.places.get_mut(at as usize) {
                *place = laid;
            }
            return at;
        }
        // A place past what four bytes name is named as the last they
        // name: read under this key it reads as no side. No memory holds
        // that many pairs at once.
        let at = u32::try_from(self.places.len()).unwrap_or(u32::MAX);
        self.places.push(laid);
        at
    }

    /// The decode side in the place `at`, where that place is the lease's
    /// under `key`.
    fn get(&self, at: u32, key: &Key) -> Option<&FiledDecode> {
        match self.places.get(at as usize)? {
            Some((owner, decode)) if owner == key => Some(decode),
            _ => None,
        }
    }

    /// [`Self::get`], to write.
    fn get_mut(&mut self, at: u32, key: &Key) -> Option<&mut FiledDecode> {
        match self.places.get_mut(at as usize)? {
            Some((owner, decode)) if owner == key => Some(decode),
            _ => None,
        }
    }

    /// Takes the decode side out of the place `at`, where that place is
    /// the lease's under `key`: the place is free.
    fn take(&mut self, at: u32, key: &Key) -> Option<FiledDecode> {
        let place = self.places.get_mut(at as usize)?;
        let (_, decode) = place.take_if(|(owner, _)| *owner == *key)?;
        self.free.push(at);
        Some(decode)
    }

    /// Keeps `replaced`, what an issue replaced of the side in the place
    /// `at` of the lease under `key`, until that side is put back. A place
    /// no side stands in keeps nothing.
    #[cfg(feature = "controller")]
    fn keep(&mut self, at: u32, key: Key, replaced: Replaced) {
        let at = at as usize;
        if at >= self.places.len() {
            return;
        }
        if self.replaced.len() <= at {
            self.replaced.resize(at + 1, None);
        }
        if let Some(place) = self.replaced.get_mut(at) {
            *place = Some((key, replaced));
        }
    }

    /// Whether an issue ever kept a side here. A file no decode call issued
    /// into keeps none.
    #[cfg(feature = "controller")]
    fn keeps_any(&self) -> bool {
        !self.replaced.is_empty()
    }

    /// The chosen side an issue replaced in the place `at` of the lease
    /// under `key`, while the side that issue laid is issued or dispatched.
    /// It is read and stays kept. `None` for a side no call issued, and for
    /// an issued side that ended: a place keeps what an issue replaced past
    /// its lease's end, so the side's own count of issues says whether a
    /// call laid it.
    #[cfg(feature = "controller")]
    fn replaced_side(&self, at: u32, key: &Key) -> Option<DecodeSide> {
        let issued = self.get(at, key)?;
        let kept = self.replaced.get(at as usize)?.as_ref();
        let (_, (first, _)) = kept.filter(|(owner, _)| owner == key)?;
        (issued.issues != 0 && issued.side.state.is_live()).then_some(*first)
    }

    /// Returns the issued, originally chosen decode at `at` and takes its saved side
    /// and deadline only if both belong to `key` and any named issue is still current.
    #[cfg(feature = "controller")]
    fn kept(
        &mut self,
        at: u32,
        key: &Key,
        issue: Option<DecodeIssue>,
    ) -> Option<(&mut FiledDecode, Replaced)> {
        let (owner, decode) = self.places.get_mut(at as usize)?.as_mut()?;
        if owner != key
            || !decode.filed_chosen
            || !decode.side.state.is_issued()
            || issue.is_some_and(|issue| issue.0 != decode.issues)
        {
            return None;
        }
        let place = self.replaced.get_mut(at as usize)?;
        let (_, replaced) = place.take_if(|(owner, _)| *owner == *key)?;
        Some((decode, replaced))
    }
}

/// A pair's decode side with the slot of its worker's count, as a door
/// that files a pair is handed them; `None` files a single's lease.
type Paired = Option<(DecodeSide, Option<WorkerSlot>)>;

/// Whether `side` of the lease `found` waits for its dispatch: a single's
/// lease is its one lease under any side.
fn waits(found: &Sides, side: Side) -> bool {
    let side = found.side(side);
    side.is_some_and(|(_, state)| state.is_issued())
}

/// Whether a dispatch moves `side` of the lease `found`
/// ([`Leases::dispatched_side`]): a side that waits for it, or a pair's
/// decode side still chosen, which a dispatch that names a target books.
/// A handed frame's dispatch is told by this, whatever index it names. A
/// record's own state is never chosen, so a single's dispatch and a
/// prefill's read one state: a pair's decode side alone is asked the
/// second.
fn takes_dispatch(found: &Sides, side: Side) -> bool {
    waits(found, side)
        || (side == Side::Decode && found.decode.is_some_and(|decode| decode.state.is_chosen()))
}

/// `record`, of a lease the dispatch deadline's end took, as that end
/// left it: released as abandoned where it waited. A pair's record is its
/// prefill part, which stands as it was where a dispatch had named it.
fn left(mut record: LeaseRecord) -> LeaseRecord {
    record.state = state_after_deadline(record.state);
    record
}

/// The state of a side after the deadline sweep. Only issued sides end.
fn state_after_deadline(state: LeaseState) -> LeaseState {
    if state.is_issued() {
        LeaseState::Released(ReleaseReason::Abandoned)
    } else {
        state
    }
}

/// Moves a side's booking to the slot `to`: the count it booked is free
/// and `to`'s is booked, and with them the count of `session` on each,
/// where the side counts for its key.
fn rebook(
    counts: &InFlightCounts,
    booked: &mut Option<WorkerSlot>,
    to: Option<WorkerSlot>,
    session: Session,
) {
    let first = *booked;
    *booked = to;
    counts.moved(first, to, session);
}

/// The session `record`'s own part books for on its worker's count
/// ([`InFlightCounts::book`]): its key's where its lease counts for its
/// key (`counts`, [`Held::keyed`]), and none where it does not. A
/// single's lease counts while it is on file, and a pair's prefill part
/// while it lives, as each books its worker's count ([`marked`]). What
/// the decision did with the key's pin says nothing here: a request
/// placed on a worker at its cap moves no pin and is one more of the
/// key's requests there. The part reads as it stands at the call, so an
/// end frees what the issue booked: a dispatch that moves it moves its
/// booking with the record's target.
fn keyed(counts: bool, record: &LeaseRecord) -> Session {
    let stands = counts && (record.decode.is_none() || record.state.is_live());
    if stands {
        record.lease.session
    } else {
        Session::NONE
    }
}

/// [`keyed`] for `decode`, a pair's decode side: it books while it lives.
fn keyed_decode(counts: bool, decode: &DecodeSide) -> Session {
    let stands = counts && decode.state.is_live();
    if stands {
        decode.session
    } else {
        Session::NONE
    }
}

impl Leases {
    /// An empty file under `term`, sharing its deadline hint with the
    /// file that owns its turn.
    fn empty(term: u64, counts: Arc<InFlightCounts>, origin: Instant) -> Self {
        Self {
            term,
            records: HashMap::default(),
            decodes: Decodes::default(),
            counts,
            minted: Minted::default(),
            due: VecDeque::new(),
            next_due: Arc::new(NextDue::new(origin)),
        }
    }

    /// The next id under the store's term: the term in bits 32 to 62 over a
    /// counter that starts at 1 in each term, as the contract bounds a
    /// controller's ids. The counter comes from the block `file` holds of
    /// `blocks`, or from the next one it claims ([`Minted::mint`]). `None`
    /// for a term past 31 bits and for a term whose counter is spent: an
    /// id is never 0, never carries the bit of a gateway's own ids and
    /// never repeats.
    fn mint(&mut self, blocks: &Blocks, file: u8) -> Option<LeaseId> {
        let term = self.term;
        if term >= 1 << 31 {
            return None;
        }
        let counter = self.minted.mint(blocks, term, file)?;
        Some(LeaseId(term << 32 | u64::from(counter)))
    }

    /// Files `record` under `key` with `booked`, the slot of the count it
    /// books: a single's lease, or with `pair` the prefill part of a
    /// pair's beside its decode side. A record on file under that key
    /// gives its counts back first.
    fn file(
        &mut self,
        key: Key,
        mut record: LeaseRecord,
        booked: Option<WorkerSlot>,
        pair: Paired,
    ) {
        self.end(&key);
        let booked = marked(&mut record, booked, &pair);
        let side = pair.map(|(side, booked)| FiledDecode::laid(side, booked));
        let decode = side.map_or(0, |side| self.decodes.lay(key, side));
        // A record filed whole says nothing of its decision: it counts
        // for its key wherever it has one.
        let held = Held {
            record,
            booked,
            decode,
            keyed: true,
        };
        self.records.insert(key, held);
        self.laid(key, &record, booked, true);
        if let Some(side) = side {
            self.decode_laid(key, &record, &side, true);
        }
    }

    /// What a record just laid on file takes: it books `booked`, its
    /// worker's count, and where its lease counts for its key (`counts`)
    /// the key's ([`keyed`]), and its dispatch deadline is queued where it
    /// is filed as issued.
    fn laid(&mut self, key: Key, record: &LeaseRecord, booked: Option<WorkerSlot>, counts: bool) {
        if let Some(booked) = booked {
            self.counts.book(booked, keyed(counts, record));
        }
        if record.state.is_issued() {
            self.queue(record.dispatch_deadline, key);
        }
    }

    /// What `decode`, the decode side just laid beside `record` under
    /// `key`, takes: it books its worker's count where it is filed live,
    /// and where the lease counts for its key (`counts`) the key's
    /// ([`keyed_decode`]), and the lease's dispatch deadline is queued for
    /// it where the record's own filing queued none: a lease takes one
    /// place of the queue.
    fn decode_laid(&mut self, key: Key, record: &LeaseRecord, decode: &FiledDecode, counts: bool) {
        if let Some(booked) = decode.booked {
            self.counts.book(booked, keyed_decode(counts, &decode.side));
        }
        if decode.side.state.is_issued() && !record.state.is_issued() {
            self.queue(record.dispatch_deadline, key);
        }
    }

    /// The place a decode call can issue into, while its prefill is held:
    /// a side still chosen. A side a repeat's ending put back is one. A
    /// side whose count of issues is full is none: it is refused as one
    /// that stands, so no count repeats, an undo names one issue for the
    /// lease's life, and the first pick's dispatch still books the side.
    fn chosen_decode(&mut self, key: &Key) -> Result<(&mut Held, &mut FiledDecode), DecodeRefused> {
        let held = self.records.get_mut(key).ok_or(DecodeRefused::NoLease)?;
        let found = held
            .record
            .decode
            .and_then(|_| self.decodes.get_mut(held.decode, key))
            .ok_or(DecodeRefused::NotChosen)?;
        if !found.side.state.is_chosen() {
            return Err(if found.filed_chosen {
                DecodeRefused::DecodeStands
            } else {
                DecodeRefused::NotChosen
            });
        }
        if !matches!(held.record.state, LeaseState::Dispatched { .. }) {
            return Err(DecodeRefused::PrefillNotHeld);
        }
        if found.issues == u8::MAX {
            return Err(DecodeRefused::DecodeStands);
        }
        Ok((held, found))
    }

    /// The chosen side a decode call's issue replaced on the lease under
    /// `key`, as the pair's first decision filed it
    /// ([`Decodes::replaced_side`]). A file no decode call issued into
    /// looks no record up.
    #[cfg(feature = "controller")]
    fn replaced_decode(&self, key: &Key) -> Option<DecodeSide> {
        if !self.decodes.keeps_any() {
            return None;
        }
        let at = self.records.get(key)?.decode;
        self.decodes.replaced_side(at, key)
    }

    /// Undoes the current decode issue if `issue` is `None` or matches, returning both prior sides.
    /// Returns `None` without changes if there is no matching issue with saved state.
    #[cfg(feature = "controller")]
    fn taken_back(&mut self, key: &Key, issue: Option<DecodeIssue>) -> Option<Sides> {
        use std::collections::hash_map::Entry;

        let Entry::Occupied(mut entry) = self.records.entry(*key) else {
            return None;
        };
        let held = entry.get_mut();
        let (decode, (before, deadline)) = self.decodes.kept(held.decode, key, issue)?;
        let found = Sides {
            record: held.record,
            decode: Some(decode.side),
        };
        let side_key = keyed_decode(held.keyed, &decode.side);
        decode.side = before;
        held.record.dispatch_deadline = deadline;
        let ended = !held.record.state.is_live() && !decode.side.state.is_live();
        if let Some(booked) = decode.booked.take() {
            self.counts.free(booked, side_key);
        }
        if ended {
            let held = entry.remove();
            self.gone(held, key, |_, _| ());
        }
        Some(found)
    }

    /// Replaces a chosen decode side in its place while the prefill is
    /// held. What the issue replaced is kept with the place, for its undo
    /// and for a repeat's ending. The side's lease comes with the issue's
    /// count.
    fn issue_decode(
        &mut self,
        key: Key,
        mut decode: DecodeSide,
        booked: Option<WorkerSlot>,
        now: Instant,
    ) -> Result<(crate::tracking::Lease, DecodeIssue), DecodeRefused> {
        let (held, found) = self.chosen_decode(&key)?;
        #[cfg(feature = "controller")]
        let kept = (held.decode, (found.side, held.record.dispatch_deadline));
        // The place's own test left room for one more: nothing wraps.
        let count = found.issues.wrapping_add(1);
        decode.state = LeaseState::ISSUED;
        decode.target_index = 0;
        let laid = FiledDecode {
            side: decode,
            booked,
            issues: count,
            ..*found
        };
        *found = laid;
        held.record.dispatch_deadline = now + DISPATCH_TTL;
        let (record, keyed) = (held.record, held.keyed);
        #[cfg(feature = "controller")]
        self.decodes.keep(kept.0, key, kept.1);
        self.decode_laid(key, &record, &laid, keyed);
        Ok((decode.lease(&record.lease), DecodeIssue(count)))
    }

    /// The keys of `gateway`'s leases on file.
    fn of(&self, gateway: GatewayId) -> Vec<Key> {
        let keys = self.records.keys();
        keys.filter(|(owner, _)| *owner == gateway)
            .copied()
            .collect()
    }

    /// Takes the lease under `key` off the file and gives its counts back
    /// ([`Self::gone`]): the way a record leaves the file by its key,
    /// whichever door ends it.
    fn end(&mut self, key: &Key) -> Option<LeaseRecord> {
        self.leave(key, |held, _| held.record)
    }

    /// [`Self::end`], with what a tell of the lease takes: both its parts
    /// as the file held them. Inline at each caller, so the answer is
    /// built where it is read.
    #[inline(always)]
    fn taken(&mut self, key: &Key) -> Option<Sides> {
        self.leave(key, |held, decode| {
            let record = held.record;
            Sides { record, decode }
        })
    }

    /// The one end behind [`Self::end`] and [`Self::taken`]: the record
    /// leaves the file and gives its count back, and a pair's decode side
    /// leaves with it under the same hold. `shape` makes the answer of
    /// what the file held. Inline in both, so an end whose caller reads
    /// no decode side builds none.
    #[inline(always)]
    fn leave<T>(
        &mut self,
        key: &Key,
        shape: impl FnOnce(Held, Option<DecodeSide>) -> T,
    ) -> Option<T> {
        let held = self.records.remove(key)?;
        Some(self.gone(held, key, shape))
    }

    /// What `held`, a lease that just left the file under `key`, gives
    /// back: its count and its key's booking, and a pair's decode side,
    /// which leaves with it under the same hold. `shape` makes the answer
    /// of what the file held. Every record that leaves the file passes
    /// here.
    #[inline(always)]
    fn gone<T>(
        &mut self,
        held: Held,
        key: &Key,
        shape: impl FnOnce(Held, Option<DecodeSide>) -> T,
    ) -> T {
        if let Some(booked) = held.booked {
            self.counts.free(booked, keyed(held.keyed, &held.record));
        }
        let decode = if held.record.decode.is_some() {
            self.unpaired(held.decode, key, held.keyed)
        } else {
            None
        };
        shape(held, decode)
    }

    /// Takes the decode side of the pair's lease under `key` out of its
    /// place `at` and gives its count back, and its key's booking where
    /// the lease counts for its key (`counts`). Out of line: the end of a
    /// single's lease carries none of it.
    #[inline(never)]
    fn unpaired(&mut self, at: u32, key: &Key, counts: bool) -> Option<DecodeSide> {
        let decode = self.decodes.take(at, key)?;
        if let Some(booked) = decode.booked {
            self.counts.free(booked, keyed_decode(counts, &decode.side));
        }
        Some(decode.side)
    }

    /// Both parts of the lease `held` under `key`: its record and, where
    /// the record carries the mark of a pair's lease, its decode side.
    fn sides(&self, key: &Key, held: &Held) -> Sides {
        let record = held.record;
        let decode = if record.decode.is_some() {
            (self.decodes.get(held.decode, key)).map(|decode| decode.side)
        } else {
            None
        };
        Sides { record, decode }
    }

    /// Whether a side of the lease `held` under `key` waits for its
    /// dispatch.
    fn waits(&self, key: &Key, held: &Held) -> bool {
        let issued = |state: LeaseState| state.is_issued();
        let decode = || self.decodes.get(held.decode, key);
        issued(held.record.state)
            || (held.record.decode.is_some()
                && decode().is_some_and(|decode| issued(decode.side.state)))
    }

    /// Whether a side of the lease `held` under `key` is issued or
    /// dispatched.
    fn is_live(&self, key: &Key, held: &Held) -> bool {
        let decode = || self.decodes.get(held.decode, key);
        held.record.state.is_live()
            || (held.record.decode.is_some()
                && decode().is_some_and(|decode| decode.side.state.is_live()))
    }

    /// Returns the sides before dispatch so the teller sees the original
    /// target and state even when this report moves the booking.
    #[inline(always)]
    fn dispatched(
        &mut self,
        slots: Slots<'_>,
        key: &Key,
        dispatch: Dispatch,
        at: Instant,
    ) -> Option<Sides> {
        let held = self.records.get_mut(key)?;
        let mut found = Sides {
            record: held.record,
            decode: None,
        };
        if held.record.decode.is_some() {
            if let Some(decode) = self.decodes.get_mut(held.decode, key) {
                found.decode = Some(decode.side);
                Self::dispatched_side(slots, &self.counts, held, decode, dispatch, at);
            }
            return Some(found);
        }
        if !held.record.state.is_issued() {
            return Some(found);
        }
        held.record.state = LeaseState::Dispatched { at };
        // A dispatch on another target of the pick: the record names that
        // target's index from here on. Where the target is another worker
        // the lease books it and the count of the first one is free, the
        // new count added before the first is taken; another rank of the
        // lease's own worker moves no count.
        let Some((target, _)) = held.record.target_at(dispatch.target_index) else {
            return Some(found);
        };
        if target != held.record.target {
            let first = held.booked;
            held.booked = slots.of(target);
            let session = if held.keyed {
                held.record.lease.session
            } else {
                Session::NONE
            };
            self.counts.moved(first, held.booked, session);
            held.record.target = target;
        }
        held.record.lease.target_index = dispatch.target_index;
        Some(found)
    }

    /// Dispatches the named side if it is issued or chosen. Every target
    /// of a pair's pick names the same prefill, so only the decode's
    /// booking can move. A chosen decode books its target at a dispatch
    /// that names one, and stays chosen at an index its pick holds no
    /// target at.
    fn dispatched_side(
        slots: Slots<'_>,
        counts: &InFlightCounts,
        held: &mut Held,
        decode: &mut FiledDecode,
        dispatch: Dispatch,
        at: Instant,
    ) {
        match dispatch.side {
            Side::Prefill if held.record.state.is_issued() => {
                held.record.state = LeaseState::Dispatched { at };
                // The target the prefill went to stays with the decode
                // side, for a decode decided again behind it. An index
                // the pick holds no target at moves nothing.
                if decode.side.target_at(dispatch.target_index).is_some() {
                    decode.taken = dispatch.target_index;
                }
            }
            Side::Decode if decode.side.state.is_issued() => {
                let side = &mut decode.side;
                let named = side.target_at(dispatch.target_index);
                side.state = LeaseState::Dispatched { at };
                if let Some((target, _)) = named {
                    if target != side.target {
                        let to = slots.of(target);
                        let session = if held.keyed {
                            side.session
                        } else {
                            Session::NONE
                        };
                        rebook(counts, &mut decode.booked, to, session);
                        side.target = target;
                    }
                    side.target_index = dispatch.target_index;
                }
            }
            Side::Decode if decode.side.state.is_chosen() => {
                let side = &mut decode.side;
                // The side has no issue outstanding, and a word that names
                // no target of its pick names no worker to book: the side
                // waits on for a call, or for a word that names one.
                let Some((target, _)) = side.target_at(dispatch.target_index) else {
                    return;
                };
                side.target = target;
                side.target_index = dispatch.target_index;
                side.state = LeaseState::Dispatched { at };
                decode.booked = slots.of(side.target);
                if let Some(booked) = decode.booked {
                    counts.book(booked, keyed_decode(held.keyed, side));
                }
            }
            _ => {}
        }
    }

    /// The lease under `key` given up on `side`, or with `None` whole;
    /// `None` where it is not on file, else what a tell of it takes, both
    /// its parts as the door found them. One side of a pair's lease ends
    /// alone. A release with no side, and any release of a single's
    /// lease, ends the lease. Inline at each caller: a single's release
    /// is its end.
    #[inline(always)]
    fn released(&mut self, key: &Key, side: Option<Side>, reason: ReleaseReason) -> Option<Sides> {
        match side {
            Some(side) => self.end_side(key, side, LeaseState::Released(reason)),
            None => self.taken(key),
        }
    }

    /// Returns the prior sides. Any side ends a single. A pair ignores absent or ended sides.
    /// Each side frees its bookings (none for a chosen decode). The last live side removes the pair.
    #[inline(always)]
    fn end_side(&mut self, key: &Key, side: Side, ended: LeaseState) -> Option<Sides> {
        use std::collections::hash_map::Entry;

        let Entry::Occupied(mut entry) = self.records.entry(*key) else {
            return None;
        };
        let held = entry.get_mut();
        let mut found = Sides {
            record: held.record,
            decode: None,
        };
        if held.record.decode.is_none() {
            let held = entry.remove();
            self.gone(held, key, |_, _| ());
            return Some(found);
        }
        let Some(decode) = self.decodes.get_mut(held.decode, key) else {
            return Some(found);
        };
        found.decode = Some(decode.side);
        // The session the side books for on its worker's count, read while
        // it lives.
        let side_key = match side {
            Side::Prefill => keyed(held.keyed, &held.record),
            Side::Decode => keyed_decode(held.keyed, &decode.side),
            Side::Single | Side::Encode => return Some(found),
        };
        let (state, booked) = match side {
            Side::Prefill => (&mut held.record.state, &mut held.booked),
            Side::Decode => (&mut decode.side.state, &mut decode.booked),
            Side::Single | Side::Encode => return Some(found),
        };
        if !(state.is_live() || state.is_chosen()) {
            return Some(found);
        }
        *state = ended;
        if let Some(booked) = booked.take() {
            self.counts.free(booked, side_key);
        }
        if !held.record.state.is_live() && !decode.side.state.is_live() {
            let held = entry.remove();
            self.gone(held, key, |_, _| ());
        }
        Some(found)
    }

    /// Queues `deadline` in its order. The deadlines come in rising order
    /// but for two deciders whose clock reads lie a moment apart and for a
    /// record written with a deadline of its own: those are placed by a
    /// search.
    fn queue(&mut self, deadline: Instant, key: Key) {
        let first = if self.due.back().is_none_or(|&(last, _)| last <= deadline) {
            self.due.push_back((deadline, key));
            self.due.len() == 1
        } else {
            let at = self.due.partition_point(|&(due, _)| due <= deadline);
            self.due.insert(at, (deadline, key));
            at == 0
        };
        // A new first place asks the sweep once; no clock is read here.
        if first {
            self.next_due.ask();
        }
    }

    /// Shows the sweep the deadline of the queue's first place, or that
    /// the queue is empty: the file is asked again once that instant is
    /// reached. The sweep's, at its take and behind its ends, under the
    /// file's turn.
    fn show_due(&self) {
        self.next_due.set(self.due.front().map(|&(due, _)| due));
    }

    /// Clears the deadline queue and its hint under the file's turn.
    fn clear_due(&mut self) {
        self.due.clear();
        self.next_due.set(None);
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
    /// the steps are spent. A lease waits while a side of it does
    /// ([`Leases::abandon`]). A pair's decode side that waits behind its
    /// prefill is not ended: its lease is queued again, a step as any
    /// place stepped over ([`Leases::waits_on`]).
    fn end_due(&mut self, now: Instant, steps: &mut usize) -> Option<Sides> {
        while *steps > 0 {
            let &(due, key) = self.due.front()?;
            let waits = self
                .records
                .get(&key)
                .is_some_and(|held| self.waits(&key, held) && held.record.dispatch_deadline == due);
            if waits && due > now {
                return None;
            }
            self.due.pop_front();
            *steps -= 1;
            if waits && !self.waits_on(&key, now) {
                return self.abandon(&key);
            }
        }
        None
    }

    /// Whether the pair's lease under `key`, whose deadline the door found
    /// due at `now`, waits on for its decode side alone: the lease's
    /// deadline is then written one span behind `now` and queued again,
    /// and nothing ends.
    ///
    /// A decode side issued with its prefill is sent when that prefill has
    /// answered, so the time to name its dispatch starts at the prefill's
    /// end. The side stands while its prefill is dispatched, as it stands
    /// in a gateway's own process, where no deadline runs. At the door's
    /// first pass behind that prefill's end it takes one whole deadline of
    /// its own ([`FiledDecode::own_deadline`]), and the pass behind that
    /// ends a side that still waits. An end hands the file no instant: the
    /// door's own pass dates the side's deadline, and no clock is read.
    ///
    /// `false` for every other lease, which the deadline ends as it did: a
    /// single's, a pair whose prefill waits too, and a side a decode call
    /// issued, whose deadline the call wrote ([`Leases::issue_decode`]).
    /// Cold, as [`Leases::abandon`] is: a due deadline alone reaches it.
    #[cold]
    fn waits_on(&mut self, key: &Key, now: Instant) -> bool {
        let Some(held) = self.records.get_mut(key) else {
            return false;
        };
        if held.record.decode.is_none() || held.record.state.is_issued() {
            return false;
        }
        let Some(decode) = self.decodes.get_mut(held.decode, key) else {
            return false;
        };
        if decode.filed_chosen || !decode.side.state.is_issued() {
            return false;
        }
        if !matches!(held.record.state, LeaseState::Dispatched { .. }) {
            if decode.own_deadline {
                return false;
            }
            decode.own_deadline = true;
        }
        let deadline = now + DISPATCH_TTL;
        held.record.dispatch_deadline = deadline;
        self.queue(deadline, *key);
        true
    }

    /// Ends what waits of the lease under `key` as abandoned: a single's
    /// lease whole, and of a pair's each side no dispatch named, while a
    /// side that was dispatched stands. The lease comes back as the call
    /// found it, both its parts: the sides it ended are those it holds as
    /// issued ([`waits`]). Cold: a deadline that ends a lease is the rare
    /// way off the file, and the issue that looks for one keeps the frame
    /// of its search.
    #[cold]
    fn abandon(&mut self, key: &Key) -> Option<Sides> {
        let abandoned = LeaseState::Released(ReleaseReason::Abandoned);
        let held = self.records.get(key)?;
        if held.record.decode.is_none() {
            return self.taken(key);
        }
        let found = self.sides(key, held);
        for side in [Side::Decode, Side::Prefill] {
            if waits(&found, side) {
                self.end_side(key, side, abandoned);
            }
        }
        Some(found)
    }

    /// Ends what waits of the lease `gateway`'s request was issued, the
    /// request whose id's fingerprint is `request`: the first such lease
    /// of the deadline queue, which holds every lease that waits for a
    /// dispatch. The lease comes back as [`Self::abandon`] found it, both
    /// its parts; `None` where no lease of the request waits. The walk
    /// reads a record for each place of the gateway's it passes: the
    /// leases that wait and, behind one that waits out its deadline, the
    /// issues since.
    #[cfg(feature = "controller")]
    fn end_asked_again(&mut self, gateway: GatewayId, request: u64) -> Option<Sides> {
        let waits = |key: &Key| {
            key.0 == gateway
                && self
                    .records
                    .get(key)
                    .is_some_and(|held| held.record.request == request && self.waits(key, held))
        };
        let key = self.due.iter().map(|&(_, key)| key).find(waits)?;
        self.abandon(&key)
    }
}

/// Marks `record` as a door files it, in place, and gives the slot it
/// books: with `pair` the record takes the mark of a pair's lease, and a
/// prefill side that ended books none; a
/// record alone is a single's and carries no mark. A pair's record states
/// no word of a request kept from its profile, whatever its caller wrote
/// ([`LeaseRecord::untold`]): both sides of a pair are told. In place: the
/// record is not copied on its way to the file.
fn marked(
    record: &mut LeaseRecord,
    booked: Option<WorkerSlot>,
    pair: &Paired,
) -> Option<WorkerSlot> {
    record.decode = pair.as_ref().map(|_| DecodeMark);
    record.untold &= pair.is_none();
    booked.filter(|_| pair.is_none() || record.state.is_live())
}

impl MemoryStore {
    /// Issues the lease of one decision made under `record.term`, the
    /// store's own term: the store mints its id ([`Leases::mint`]), files
    /// the record under it and books `booked`, the slot of the record's
    /// target's count as the decision's own world names it
    /// ([`StoreSnapshot::booked_of`]). `Ok` is the lease as filed, with its id.
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
    /// under the turn, as a frame's end is told: of a pair's lease each
    /// side that waited. So a lease
    /// past its deadline stands no longer than the next issues take, and no
    /// call passes over the file: an issue adds one place to the queue and
    /// steps over [`DUE_STEPS`] at most.
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "called by the served controller alone")
    )]
    pub(crate) fn issue_decided(
        &self,
        record: LeaseRecord,
        booked: Option<WorkerSlot>,
        issued: impl Fn(&crate::tracking::Lease),
    ) -> Result<crate::tracking::Lease, LeaseRefused> {
        self.issue_minted(record, booked, None, false, issued)
    }

    /// Issues the two leases of one pair's decision under the one id the
    /// store mints: [`Self::issue_keyed`] for `record`, the prefill
    /// part, with `side`, the decode side, laid beside it under the same
    /// hold of the turn. `booked` holds the slots of the counts the sides
    /// book, the prefill's first. `issued` hears the prefill and then the
    /// decode if it was issued. `Ok` is the prefill side's lease.
    ///
    /// The lease counts for each side's key, under any sessions rule and
    /// under none, as a keyed single's lease counts for its key
    /// ([`Self::issue_keyed`]): each side books one request of its own
    /// session on its worker while it lives, and a side with no session
    /// books none. A decode side filed chosen books none until its issue
    /// or its dispatch, and a decode call's side books the session its own
    /// frame states, whatever the pair's first frame stated.
    ///
    /// # Errors
    ///
    /// Those of [`Self::issue_decided`].
    #[cfg(feature = "controller")]
    pub(crate) fn issue_decided_pair(
        &self,
        record: LeaseRecord,
        side: DecodeSide,
        [booked, decode]: [Option<WorkerSlot>; 2],
        issued: impl Fn(&crate::tracking::Lease),
    ) -> Result<crate::tracking::Lease, LeaseRefused> {
        self.issue_minted(record, booked, Some((side, decode)), true, issued)
    }

    /// [`LeasesStore::issue_decode`] with the slot from the decision's world.
    /// The `issued` callback hears the decode lease under the file's turn.
    /// The lease comes with the issue's count, for a caller that may undo
    /// it ([`Self::unissue_decode`]).
    pub(crate) fn issue_decided_decode(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        (decode, booked): (DecodeSide, Option<WorkerSlot>),
        now: Instant,
        issued: impl FnOnce(&crate::tracking::Lease),
    ) -> Result<(crate::tracking::Lease, DecodeIssue), DecodeRefused> {
        let mut leases = self.turn_of(self.file_of(lease));
        let (lease, issue) = leases.issue_decode((gateway, lease), decode, booked, now)?;
        issued(&lease);
        Ok((lease, issue))
    }

    /// [`Self::issue_decided`] for a keyed single decision, under any
    /// sessions rule and under none: the lease books one request of its key
    /// on its worker until it leaves the file, whatever the decision did
    /// with the key's pin, and the worker's count of the key's requests
    /// steps with the lease's own ([`InFlightCounts::book`]). A pair's
    /// decision counts for each side's key (`issue_decided_pair`). The
    /// lease of any other decision books no key.
    ///
    /// # Errors
    ///
    /// Those of [`Self::issue_decided`].
    #[cfg_attr(
        not(any(test, feature = "controller")),
        expect(dead_code, reason = "called by the served controller alone")
    )]
    pub(crate) fn issue_keyed(
        &self,
        record: LeaseRecord,
        booked: Option<WorkerSlot>,
        issued: impl Fn(&crate::tracking::Lease),
    ) -> Result<crate::tracking::Lease, LeaseRefused> {
        self.issue_minted(record, booked, None, true, issued)
    }

    /// Takes back the issue that stands of the decode side of `gateway`'s
    /// lease `lease`, where a call issued the side and no dispatch named
    /// it: for a gateway that asks the call again and holds no answer to
    /// it, so no dispatch will name what that answer carried. The caller
    /// is the lease's decider, the thread at work at its file: a decode
    /// call is handed to the decider whose file holds its lease. An answer
    /// that does not leave its stream is given up on the stream's task,
    /// and hands the file a word instead ([`Self::hand_taken_back`]). The
    /// issue is released under `reason` with its engine told, and the side
    /// holds the first pick's targets again: that pick's dispatch books
    /// it, and a later call issues it anew. The lease's prefill keeps its
    /// booking. Everything handed to the file is applied first, as at an
    /// issue. `false` where no issue stands, a side still chosen, one a
    /// dispatch named or one that ended, and for a caller that may not act
    /// at the file.
    #[cfg(feature = "controller")]
    #[cold]
    pub(crate) fn take_back_decode(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        reason: ReleaseReason,
    ) -> bool {
        let at = self.file_of(lease);
        let mut leases = self.files[at].take();
        if !self.applied(at, &mut leases) {
            return false;
        }
        let Some(found) = leases.taken_back(&(gateway, lease), None) else {
            return false;
        };
        let mut one = turn::OneLoad::default();
        let released = Said::Released(Some(Side::Decode), reason);
        self.tell(at, one.world(self), &found, released);
        true
    }

    /// Applies the take-back of `issue` that was handed to the file at
    /// `at` ([`Self::hand_taken_back`]). Where that issue of the decode
    /// side under `key` still stands, the side returns to chosen and the
    /// engine hears the decode side's release under `reason`. Where the
    /// side moved since, nothing moves and nothing is told.
    #[cfg(feature = "controller")]
    #[cold]
    fn apply_taken_back(
        &self,
        at: usize,
        leases: &mut Leases,
        key: Key,
        issue: DecodeIssue,
        reason: ReleaseReason,
        one: &mut turn::OneLoad,
    ) {
        if let Some(found) = leases.taken_back(&key, Some(issue)) {
            let released = Said::Released(Some(Side::Decode), reason);
            self.tell(at, one.world(self), &found, released);
        }
    }

    /// Undoes `issue`, an issue of the decode side of `gateway`'s lease
    /// `lease` whose answer does not leave: the side stands as the issue
    /// found it and its booking is freed. `false` where the side moved
    /// since, and nothing moves: the caller tells its engine of a release
    /// only behind `true`.
    #[cfg(feature = "controller")]
    pub(crate) fn unissue_decode(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        issue: DecodeIssue,
    ) -> bool {
        self.turn_of(self.file_of(lease))
            .taken_back(&(gateway, lease), Some(issue))
            .is_some()
    }

    /// The pair of the lease a decode call names, where the lease still
    /// takes a decode side: its prefill is dispatched and held and its
    /// decode side is chosen. The prefill's worker comes first, then the
    /// worker and the rank of the decode the lease's first decision chose
    /// for the target the prefill's dispatch took: the pick's first
    /// target, or the one that dispatch named. The call reads this before
    /// it decides, and its issue tests the same again under its own turn.
    ///
    /// Last comes the instant the lease was issued at, as the record's own
    /// lease states it (`Lease::issued_micros`). Every word of the lease is
    /// told under that date, and a claim the call makes is dated by it.
    ///
    /// # Errors
    ///
    /// Why the lease takes no decode side: the test of an issue
    /// ([`Self::issue_decided_decode`]).
    #[cfg(feature = "controller")]
    pub(crate) fn held_pair(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
    ) -> Result<HeldPair, DecodeRefused> {
        self.turn_of(self.file_of(lease))
            .chosen_decode(&(gateway, lease))
            .map(|(held, decode)| {
                let issued_micros = held.record.lease.issued_micros;
                (held.record.target, decode.taken_decode(), issued_micros)
            })
    }

    /// The one issue behind the doors: a single's lease with no `pair`,
    /// which then reads no decode part and takes no branch of one, and
    /// with `keyed` a lease that books its key's requests in flight.
    /// Inline in each door, so the single's door is the issue it was and
    /// tests no key.
    #[inline(always)]
    fn issue_minted(
        &self,
        mut record: LeaseRecord,
        booked: Option<WorkerSlot>,
        pair: Paired,
        keyed: bool,
        issued: impl Fn(&crate::tracking::Lease),
    ) -> Result<crate::tracking::Lease, LeaseRefused> {
        use std::collections::hash_map::Entry;

        // The file of the decider that issues, under its turn: what is
        // handed is applied, and the due leases ended and told, where the
        // caller may act at that file.
        let at = self.acting();
        let file = u8::try_from(at).unwrap_or(u8::MAX);
        let mut leases = self.files[at].take();
        let acts = self.applied(at, &mut leases);
        let (lease, store) = (record.term, leases.term);
        if lease > store {
            return Err(LeaseRefused::TermAhead { lease, store });
        }
        let unminted = LeaseRefused::Unminted { lease, store };
        if lease < store {
            return Err(unminted);
        }
        let booked = marked(&mut record, booked, &pair);
        let side = pair.map(|(side, booked)| FiledDecode::laid(side, booked));
        let on_file = &mut *leases;
        let key = loop {
            record.lease.id = on_file.mint(&self.blocks, file).ok_or(unminted)?;
            let key = (record.gateway, record.lease.id);
            // An id on file already is a restored lease's: the next one.
            if let Entry::Vacant(entry) = on_file.records.entry(key) {
                // The decode side is laid under the lease's own key, and
                // the record's entry names its place.
                let decode = side.map_or(0, |side| on_file.decodes.lay(key, side));
                entry.insert(Held {
                    record,
                    booked,
                    decode,
                    keyed,
                });
                break key;
            }
        };
        leases.laid(key, &record, booked, keyed);
        if let Some(side) = side {
            leases.decode_laid(key, &record, &side, keyed);
        }
        // The file's write and its tell under one hold of the turn: the
        // lease is told issued before any holder can end it. The decode
        // is told after the prefill when both are issued together.
        issued(&record.lease);
        if let Some((side, _)) = pair.filter(|(side, _)| side.state.is_issued()) {
            issued(&side.lease(&record.lease));
        }
        // A second lease is looked for where a first one was ended: a
        // call that ends none stopped at a lease that waits, at an empty
        // queue or at its last step, and so would the next.
        let (now, mut steps) = (record.issued_at, DUE_STEPS);
        // The ends of one issue are told under one load of the world,
        // taken at the first: an issue that ends none loads none.
        let mut one = turn::OneLoad::default();
        for _ in 0..if acts { ABANDONED } else { 0 } {
            let Some(found) = leases.end_due(now, &mut steps) else {
                break;
            };
            self.tell_abandoned(at, one.world(self), &found);
        }
        Ok(record.lease)
    }

    /// Ends the lease `gateway`'s earlier ask of a request was issued,
    /// where that lease still waits for its dispatch: the door of a
    /// request its gateway asks again because the answer never arrived
    /// (`DecideRequest.again`). The gateway holds no answer, so no dispatch
    /// will name the lease. It ends here as its dispatch deadline would
    /// end it, released as abandoned with its engine told
    /// ([`Self::tell_abandoned`]), and not up to that deadline later,
    /// beside the lease the request is issued next. `request` is the
    /// fingerprint of the request's id, all a record keeps of it: two ids
    /// of one gateway that share it read as one request. Of a pair's lease
    /// each side that waits ends, and a side that was dispatched stands.
    ///
    /// The caller is the decider of the request's model, in its requests'
    /// order. Every request of a model goes to one decider, which issues
    /// into its own file: the earlier lease is on the caller's file, filed
    /// before this call. Everything handed to the file is applied first,
    /// as at an issue, so a lease whose dispatch was said in time stands.
    /// `false` where no lease of the request waits there, and for a caller
    /// that may not act at the file.
    #[cfg(feature = "controller")]
    #[cold]
    pub(crate) fn asked_again(&self, gateway: GatewayId, request: u64) -> bool {
        let at = self.acting();
        let mut leases = self.files[at].take();
        if !self.applied(at, &mut leases) {
            return false;
        }
        let Some(found) = leases.end_asked_again(gateway, request) else {
            return false;
        };
        let mut one = turn::OneLoad::default();
        self.tell_abandoned(at, one.world(self), &found);
        true
    }

    /// Tells the engine of each side the dispatch deadline's end took from
    /// `found`, the lease as that end found it on the file at `at`: a
    /// single's lease whole, with its record as the end left it, and of a
    /// pair's each side that waited, the prefill's first, with the lease
    /// as found: the word reads a side's lease where that side stands. An
    /// issue's ends and the store's own sweep tell by it, each under the
    /// one `world` it loaded for its ends ([`turn::OneLoad`]).
    fn tell_abandoned(&self, at: usize, world: &StoreSnapshot, found: &Sides) {
        let abandoned = |side| Said::Released(side, ReleaseReason::Abandoned);
        if found.decode.is_none() {
            let (record, decode) = (left(found.record), None);
            return self.tell(at, world, &Sides { record, decode }, abandoned(None));
        }
        for side in [Side::Prefill, Side::Decode] {
            if waits(found, side) {
                self.tell(at, world, found, abandoned(Some(side)));
            }
        }
    }
}

impl MemoryStore {
    /// The store with a file of leases for each decision slot, for a caller
    /// that decides on `deciders` threads and names them
    /// ([`crate::engine::DecisionSlot`]): each slot issues into its own
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
    pub fn with_decision_slots(mut self, deciders: usize) -> Self {
        // Every file is read, each under its own turn: a lease on any of
        // them would go with its file.
        let empty = self.files.iter().all(|file| file.take().records.is_empty());
        assert!(empty, "a store's files are laid before a lease is filed");
        let (term, counts) = {
            let leases = self.files[0].take();
            (leases.term, Arc::clone(&leases.counts))
        };
        let files = turn::Files::of(deciders, &self.files, || {
            Leases::empty(term, Arc::clone(&counts), self.origin)
        });
        self.files = files;
        self
    }

    /// How many files of leases the store holds: one for each decision slot.
    #[must_use]
    pub fn decision_slots(&self) -> usize {
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
            leases.clear_due();
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
    /// The slot of `target`'s count: `None` for a worker that is no
    /// member. The standing world names a member's slot from its first
    /// build on, in a pool or in none, and such a target takes no lock.
    /// Inlined into each door: a named target's read costs the door no
    /// call, one read for each lease it files.
    #[inline(always)]
    fn of(self, target: WorkerId) -> Option<WorkerSlot> {
        if let Some(slot) = self.0.world.load().booked_of(target) {
            return Some(slot);
        }
        self.filed(target)
    }

    /// The slot of a target the standing world does not name. A member
    /// inside its registration window is read from the table its write
    /// filed it in. A build publishes before it empties the table, so a
    /// target the table no longer holds is read off the world again.
    /// The table's lock is a leaf, taken under the turn or with no lock
    /// held.
    #[cold]
    fn filed(self, target: WorkerId) -> Option<WorkerSlot> {
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

    /// The slot of a target the world a decision read does not list,
    /// where its registration filed one ([`Slots::filed`]). A decision
    /// over worker objects can pick a member inside its registration
    /// window: the objects follow a registration as its write lands, a
    /// build ahead of the world that lists it.
    #[cfg(feature = "controller")]
    pub(crate) fn filed_slot(&self, target: WorkerId) -> Option<WorkerSlot> {
        self.slots().filed(target)
    }

    /// Gives the member a membership write files its slot in the counts,
    /// and files it for the leases' doors until a world lists it. Called
    /// under the workers lock, by the write that makes the member's row.
    fn join(&self, id: WorkerId) -> WorkerSlot {
        let slot = self.counts.claim();
        self.joined.lock().push((id, slot));
        slot
    }

    /// The row of a member that left the store, its slot told so: the
    /// count its host handed for it gives up the leases that stand and
    /// goes ([`InFlightCounts::left`]). Called under the workers lock, by
    /// the write that takes the member's row.
    fn row_left(&self, held: Row) -> WorkerRow {
        self.counts.left(held.slot);
        held.row
    }

    /// Empties the table of the members no world named. Called behind
    /// the publication of a membership build, under the workers lock:
    /// every member the table held is named by the published world or
    /// has left.
    fn listed(&self) {
        self.joined.lock().clear();
    }

    /// Ends the leases past their dispatch deadline at `now`
    /// ([`LeasesStore::expire`]). The files take their turns in their
    /// order, one at a time. A file takes its turn where it may hold work
    /// for a sweep and the caller may act there ([`Self::sweep_turn`]). A
    /// decider at work ends its own file's due leases at its issues. Each
    /// end is told under the turn, as a frame's end is told.
    ///
    /// A file is passed too where `settled` says no. That is asked under
    /// the file's turn, behind everything handed to it. A host may keep
    /// what was said of a file's leases beside its ring. It says no until
    /// it has handed that on, so no lease is ended ahead of a dispatch
    /// said of it. The file's deadline is shown ahead of that question
    /// too: a file passed at it is asked again at that instant.
    ///
    /// Returns both parts of each lease as the sweep left them.
    pub(crate) fn expire_where(&self, now: Instant, settled: impl Fn(usize) -> bool) -> Vec<Sides> {
        let mut ended = Vec::new();
        for at in 0..self.files.len() {
            let Some(mut leases) = self.sweep_turn(at, now) else {
                continue;
            };
            if !settled(at) {
                continue;
            }
            // The ends of one file's sweep are told under one load of
            // the world, taken at the first.
            let (mut steps, mut one) = (usize::MAX, turn::OneLoad::default());
            while let Some(found) = leases.end_due(now, &mut steps) {
                self.tell_abandoned(at, one.world(self), &found);
                let decode = found.decode.map(|decode| DecodeSide {
                    state: state_after_deadline(decode.state),
                    ..decode
                });
                ended.push(Sides {
                    record: left(found.record),
                    decode,
                });
            }
            leases.show_due();
        }
        ended
    }

    /// Files `record` under its own id, with `booked` and `pair` as
    /// [`Leases::file`] takes them: the door behind [`LeasesStore::issue`]
    /// and [`LeasesStore::issue_pair`]. Inline in both, so a single's
    /// issue builds no pair's part.
    #[inline(always)]
    fn issue_filed(
        &self,
        record: LeaseRecord,
        booked: Option<WorkerSlot>,
        pair: Paired,
    ) -> Result<(), LeaseRefused> {
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
        leases.file(key, record, booked, pair);
        Ok(())
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
                // A term's history starts with the generation that stands:
                // one applied under no term took no place, and a standby
                // joins at this start. The number is read behind the rise:
                // an apply that published ahead of it is read here, and
                // one that publishes behind it tells itself under the term.
                let standing = self.world.load().spec().generation();
                if standing != 0 {
                    journal.generation(None, standing);
                }
            }
        });
    }

    fn issue(&self, record: LeaseRecord) -> Result<(), LeaseRefused> {
        // The slot of the record's target's count: read before the turn is
        // taken.
        let booked = self.slots().of(record.target);
        self.issue_filed(record, booked, None)
    }

    fn issue_pair(&self, record: LeaseRecord, decode: DecodeSide) -> Result<(), LeaseRefused> {
        // The slot of each side's count, read as a single's.
        let slots = self.slots();
        let booked = slots.of(record.target);
        let pair = (decode, slots.of(decode.target));
        self.issue_filed(record, booked, Some(pair))
    }

    fn issue_decode(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        decode: DecodeSide,
        now: Instant,
    ) -> Result<(), DecodeRefused> {
        let booked = self.slots().of(decode.target);
        self.issue_decided_decode(gateway, lease, (decode, booked), now, |_| ())
            .map(|_| ())
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
            // A pair needs its decode place even to dispatch its prefill.
            .is_some_and(|found| found.record.decode.is_none() || found.decode.is_some())
    }

    fn completed(&self, gateway: GatewayId, lease: LeaseId, outcome: &Outcome) -> bool {
        self.turn_of(self.file_of(lease))
            .end_side(&(gateway, lease), outcome.side, LeaseState::Completed)
            .is_some()
    }

    fn released(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        side: Option<Side>,
        reason: ReleaseReason,
    ) -> bool {
        let found = self
            .turn_of(self.file_of(lease))
            .released(&(gateway, lease), side, reason);
        found.is_some()
    }

    fn get(&self, gateway: GatewayId, lease: LeaseId) -> Option<LeaseRecord> {
        let leases = self.turn_of(self.file_of(lease));
        leases
            .records
            .get(&(gateway, lease))
            .map(|held| held.record)
    }

    fn sides(&self, gateway: GatewayId, lease: LeaseId) -> Option<Sides> {
        let leases = self.turn_of(self.file_of(lease));
        let key = (gateway, lease);
        let held = leases.records.get(&key)?;
        Some(leases.sides(&key, held))
    }

    /// The store's pass over its files, none held back by a host.
    fn expire(&self, now: Instant) -> Vec<Sides> {
        self.expire_where(now, |_| true)
    }

    fn restore(&self, gateway: GatewayId, records: &[LeaseRecord]) -> usize {
        // A page may list a lease on a member whose first build is owed:
        // its slot is read where its write filed it, so each lease the
        // pages list books its worker's count and no build is asked for.
        let slots = self.slots();
        // Each file's turn in the files' order, one at a time: the
        // gateway's leases there go, and the records its ids name are
        // filed. A record whose id names no file yet is filed in none
        // ([`LeaseRefused::Unclaimed`]).
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
                leases.file((gateway, record.lease.id), *record, booked, None);
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
        // gateways' vetoes leave under the health
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
        let files = 0..self.files.len();
        files
            .map(|at| {
                let leases = self.turn_of(at);
                let live =
                    |(key, held): &(&Key, &Held)| key.0 == gateway && leases.is_live(key, held);
                leases.records.iter().filter(live).count()
            })
            .sum()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/store/memory/tests.rs"]
mod tests;

/// A seam for the cases: a hook the applying thread runs once inside an
/// apply of a generation, between the engine's turn and the publication of
/// the generation's world ([`MemoryStore::apply`]). No case reaches that
/// place from outside. In no build but the cases'.
#[cfg(test)]
#[path = "../../tests/unit/store/memory/seam.rs"]
pub(crate) mod seam;
