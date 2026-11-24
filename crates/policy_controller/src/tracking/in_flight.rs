//! The counts of requests in flight: one cell a worker, booked by the
//! leases the store holds on file and read by a decision beside its rows.
//!
//! The store keeps the table (`InFlightCounts`) and books it under its leases'
//! lock. A decision that is lent the table with a pool's slots
//! ([`PoolInFlightCounts`]) reads a row's count by the row's position:
//! nothing is written into a row and no lock is taken. A row's first read in
//! a decision is kept ([`InFlightBuffers`]) and every later read returns it,
//! so a decision reads one count a row.
//!
//! A host that counts a worker's requests on an object of its own hands
//! the table that count ([`HostCount`]), and the table keeps it equal to
//! the worker's cell: the engine over worker lists reads the object's count.

use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    OnceLock,
};

use parking_lot::{Mutex, RwLock, RwLockReadGuard};
use rustc_hash::FxHashMap;

use crate::engine::view::{Candidate, Session};

/// The place of one worker's count in the store's counts. A member is
/// given its slot at the membership write that first files it and keeps
/// it while its row stands, and no other member is given it after.
/// In-process only: the store gives a slot, and a host that keeps no
/// counts names [`WorkerSlot::NONE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkerSlot(u32);

impl WorkerSlot {
    /// The slot that names no cell: its count reads zero and books
    /// nothing. What a table that gave every slot gives.
    pub const NONE: Self = Self(u32::MAX);
}

/// The cells of the first piece of the counts; each piece after holds
/// twice the cells of the one before.
const FIRST: u64 = 1 << 10;

/// The pieces of the counts: together they hold every slot but the last
/// 1,024 a `u32` names, [`WorkerSlot::NONE`] among those.
const PIECES: usize = 22;

/// Cells by slot, grown in pieces that never move: piece `k` holds the
/// `FIRST << k` cells from slot `(FIRST << k) - FIRST` on, laid when a
/// slot of it is first asked for. The counts stand in one, and a stage
/// that keeps a number a member stands its cells in another, beside them.
#[derive(Debug)]
pub(crate) struct Pieces<T> {
    pieces: [OnceLock<Box<[T]>>; PIECES],
}

impl<T> Default for Pieces<T> {
    fn default() -> Self {
        Self {
            pieces: std::array::from_fn(|_| OnceLock::new()),
        }
    }
}

impl<T: Default> Pieces<T> {
    /// The cell of `slot`, its piece laid with every cell at its default
    /// where it was not: `None` for a slot no piece holds.
    pub(crate) fn lay(&self, slot: WorkerSlot) -> Option<&T> {
        let (piece, first) = Self::place(slot);
        let cells = self.pieces.get(piece)?;
        let cells = cells.get_or_init(|| (0..FIRST << piece).map(|_| T::default()).collect());
        cells.get(slot.0.wrapping_sub(first) as usize)
    }
}

impl<T> Pieces<T> {
    /// The first piece: the cells of the first 1,024 slots, and no cell
    /// before it is laid.
    pub(crate) fn first(&self) -> &[T] {
        let first = self.pieces.first().and_then(OnceLock::get);
        first.map_or(&[], |cells| cells)
    }

    /// The cell of `slot` where its piece is laid.
    pub(crate) fn cell(&self, slot: WorkerSlot) -> Option<&T> {
        let (cells, first) = self.piece(slot);
        cells.get(slot.0.wrapping_sub(first) as usize)
    }

    /// The cell of `slot` read through `first`, the first piece found once
    /// for a run of reads: no search for a slot the first piece holds.
    #[inline]
    pub(crate) fn cell_from<'a>(&'a self, first: &'a [T], slot: WorkerSlot) -> Option<&'a T> {
        match first.get(slot.0 as usize) {
            Some(cell) => Some(cell),
            None => self.cell(slot),
        }
    }

    /// The cells of the piece that holds `slot` and the piece's first
    /// slot: no cells where the piece is not laid yet or no piece holds
    /// the slot.
    fn piece(&self, slot: WorkerSlot) -> (&[T], u32) {
        let (piece, first) = Self::place(slot);
        let cells = self.pieces.get(piece).and_then(OnceLock::get);
        (cells.map_or(&[], |cells| cells), first)
    }

    /// The piece `slot` falls in and that piece's first slot.
    fn place(slot: WorkerSlot) -> (usize, u32) {
        let at = u64::from(slot.0) + FIRST;
        let piece = at.ilog2() - FIRST.trailing_zeros();
        let first = (FIRST << piece) - FIRST;
        (piece as usize, first as u32)
    }
}

/// What the leases on file book on each worker: the counts a decision
/// reads as its rows' requests in flight ([`RequestFacts::in_flight`]).
///
/// One table for the store's life, a cell a member that ever stood, grown
/// in pieces that never move. A membership write gives a joining member
/// the next cell and does nothing else for a count, a pool's shape names
/// each row's cell by its slot, and a lease on file keeps the slot it
/// booked. So a count outlives a membership write, a lease of a worker
/// that left ends on the count it booked, and a decision reads a count
/// with no lock.
///
/// [`RequestFacts::in_flight`]: crate::engine::view::RequestFacts::in_flight
#[derive(Debug, Default)]
pub(crate) struct InFlightCounts {
    /// A cell a slot, its piece laid when its first slot is given.
    cells: Pieces<AtomicU32>,
    /// The count each member's host keeps for it, by the member's slot
    /// ([`HostCount`]): a place a slot, its piece laid with the cell's.
    hosts: Pieces<HostPlace>,
    /// The slots given so far.
    given: AtomicU32,
}

/// The count a store's host keeps of one worker's requests, beside the
/// worker's cell in the store's counts. A controller's worker object
/// counts its requests in flight on itself, and the engine over worker
/// lists reads that count and no cell of the store's.
///
/// The store keeps it equal to the cell of the worker's row. It steps it
/// where it steps the cell: once for each lease that books the worker and
/// once for each that books it no longer, whatever door filed, moved or
/// ended the lease. A count handed for a row takes a step for each lease
/// that books the row by then, and the count it takes the place of gives
/// each of them up: a lease filed ahead of its worker's count is on it
/// all the same, and two counts that step one number leave it as it
/// stood. Where the row leaves the store, the count gives up the leases
/// that still book it and the store lets it go: a lease that outlives its
/// worker's row is on no host's count.
///
/// Each call is one step, from whichever thread books. Two files book one
/// worker at once and neither waits on the other. A hand-over and a row's
/// leaving wait for the steps under way, and a step waits for them: each
/// of the two makes one call a lease that stands.
///
/// A lease that counts for its session key steps the session's count on
/// the worker too, beside its plain step: a capped sessions rule reads how
/// many requests of a key stand on each worker. The store keeps the
/// sessions that stand on a row, so a count handed takes each of them and
/// the count before it gives each up, as the two do for the leases'
/// number.
///
/// The calls and the count's drop run under the store's locks: none of
/// them calls the store. A store that is dropped tells its counts
/// nothing.
pub trait HostCount: Send + Sync {
    /// One lease more books the worker.
    fn booked(&self);

    /// One lease that booked the worker books it no longer.
    fn freed(&self);

    /// One lease more of `session` books the worker, never
    /// [`Session::NONE`]. A count that keeps a session's requests on its
    /// worker keeps them under the text [`Session::text`] writes: the
    /// engine over worker lists reads a record's count under that text.
    /// The default keeps no count of a session.
    fn session_booked(&self, _session: Session) {}

    /// One lease of `session` that booked the worker books it no longer:
    /// the twin of [`Self::session_booked`].
    fn session_freed(&self, _session: Session) {}
}

/// One member's place for the count its host keeps.
///
/// A step of the member's cell holds the place shared around its two
/// writes, the cell's and the count's. A hand-over and the member's
/// leaving hold it alone, so neither finds a step half made: the cell
/// either reads is what the count holds.
#[derive(Default)]
struct HostPlace {
    count: RwLock<Option<Box<dyn HostCount>>>,
    /// The sessions whose leases book the member, each with how many of
    /// its leases do. A hand-over and the member's leaving replay them:
    /// the cell's number names no session. A step writes it with the
    /// place held shared, under this lock of its own. Empty for a member
    /// that no keyed lease books. A member that stands keeps the table's
    /// room when its last session leaves: its next keyed lease allocates
    /// nothing.
    sessions: Mutex<FxHashMap<Session, u32>>,
    /// The member left the store. No count is handed for it again, so
    /// nothing would replay a session: its place gave its table back and
    /// keeps none from then on. A place is never another member's, so a
    /// table kept here would stand for the store's life. Written with the
    /// place held alone and read with it held shared.
    left: AtomicBool,
}

impl std::fmt::Debug for HostPlace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stands = self.count.try_read().map(|count| count.is_some());
        f.debug_struct("HostPlace")
            .field("stands", &stands)
            .finish()
    }
}

/// A member's place held shared, for one step of the member's cell.
type Stepping<'a> = RwLockReadGuard<'a, Option<Box<dyn HostCount>>>;

impl InFlightCounts {
    /// The next slot of the table, its cell at zero. A table that gave
    /// every slot gives the one that names no cell ([`WorkerSlot::NONE`]).
    pub(crate) fn claim(&self) -> WorkerSlot {
        let next = |given: u32| given.checked_add(1);
        let given = self
            .given
            .try_update(Ordering::Relaxed, Ordering::Relaxed, next);
        let Ok(slot) = given.map(WorkerSlot) else {
            return WorkerSlot::NONE;
        };
        // The place of a host's count is laid with the cell: no booking
        // lays one.
        match (self.cells.lay(slot), self.hosts.lay(slot)) {
            (Some(_), Some(_)) => slot,
            _ => WorkerSlot::NONE,
        }
    }

    /// The counts of `slots`, in their order: one load a slot, the piece
    /// found once for each run of slots that share it.
    #[cfg(test)]
    pub(crate) fn of<'a>(&'a self, slots: &'a [WorkerSlot]) -> impl Iterator<Item = u32> + 'a {
        let (mut cells, mut first): (&[AtomicU32], u32) = (&[], 0);
        slots.iter().map(move |&slot| {
            let at = slot.0.wrapping_sub(first) as usize;
            if let Some(cell) = cells.get(at) {
                return cell.load(Ordering::Relaxed);
            }
            (cells, first) = self.cells.piece(slot);
            let at = slot.0.wrapping_sub(first) as usize;
            cells.get(at).map_or(0, |cell| cell.load(Ordering::Relaxed))
        })
    }

    /// One lease more books the worker of `slot`, in one atomic step. The
    /// turn of a leases' file orders that file's writers alone, and two
    /// files book one worker where it serves two models: neither write is
    /// lost under the other. No reader waits on the line.
    ///
    /// `session` is the lease's, where the lease counts for its key:
    /// under the same hold of the member's place, the place keeps the
    /// session and the host's count takes the session's step
    /// ([`HostCount::session_booked`]). A lease that counts for no key
    /// names [`Session::NONE`]: it reads no session and takes no lock for
    /// one.
    pub(crate) fn book(&self, slot: WorkerSlot, session: Session) {
        if let Some(cell) = self.cell(slot) {
            let host = self.stepping(slot);
            let one_more = |held: u32| held.saturating_add(1);
            let held = cell.update(Ordering::Relaxed, Ordering::Relaxed, one_more);
            // The host's count takes the step the cell took.
            let count = host.as_deref().and_then(Option::as_deref);
            if let Some(count) = count.filter(|_| held != u32::MAX) {
                count.booked();
            }
            if !session.is_none() {
                self.session_kept(slot, session, count);
            }
        }
    }

    /// One lease that booked the worker of `slot` left the file, in one
    /// atomic step as well: a count at zero stays there. `session` is the
    /// one its booking named ([`Self::book`]): its count leaves under the
    /// same hold of the member's place.
    pub(crate) fn free(&self, slot: WorkerSlot, session: Session) {
        if let Some(cell) = self.cell(slot) {
            let host = self.stepping(slot);
            let one_less = |held: u32| held.saturating_sub(1);
            let held = cell.update(Ordering::Relaxed, Ordering::Relaxed, one_less);
            let count = host.as_deref().and_then(Option::as_deref);
            if let Some(count) = count.filter(|_| held != 0) {
                count.freed();
            }
            if !session.is_none() {
                self.session_let_go(slot, session, count);
            }
        }
    }

    /// The place of `slot` keeps one lease more of `session`, and `count`,
    /// the host's count that stands, takes the session's step. Its caller
    /// holds the place shared. The place of a member that left keeps
    /// nothing: no count stands there, and none is handed again.
    #[inline(never)]
    fn session_kept(&self, slot: WorkerSlot, session: Session, count: Option<&dyn HostCount>) {
        let Some(place) = self.hosts.cell(slot) else {
            return;
        };
        if place.left.load(Ordering::Relaxed) {
            return;
        }
        *place.sessions.lock().entry(session).or_default() += 1;
        if let Some(count) = count {
            count.session_booked(session);
        }
    }

    /// The place of `slot` keeps one lease less of `session`, and its
    /// entry leaves with its last lease. `count` takes the step where the
    /// place held the session: a session it does not hold steps nothing.
    /// Its caller holds the place shared.
    #[inline(never)]
    fn session_let_go(&self, slot: WorkerSlot, session: Session, count: Option<&dyn HostCount>) {
        let Some(place) = self.hosts.cell(slot) else {
            return;
        };
        let stood = {
            let mut sessions = place.sessions.lock();
            let left = sessions.get_mut(&session).map(|leases| {
                *leases = leases.saturating_sub(1);
                *leases
            });
            if left == Some(0) {
                sessions.remove(&session);
            }
            left.is_some()
        };
        if let Some(count) = count.filter(|_| stood) {
            count.session_freed(session);
        }
    }

    /// How many sessions the place of `slot` keeps: zero once the last
    /// keyed lease that booked its member left.
    #[cfg(test)]
    pub(crate) fn sessions_kept(&self, slot: WorkerSlot) -> usize {
        let place = self.hosts.cell(slot);
        place.map_or(0, |place| place.sessions.lock().len())
    }

    /// How many sessions the table of `slot`'s place has room for: zero
    /// for a table that holds no memory.
    #[cfg(test)]
    pub(crate) fn sessions_room(&self, slot: WorkerSlot) -> usize {
        let place = self.hosts.cell(slot);
        place.map_or(0, |place| place.sessions.lock().capacity())
    }

    /// Takes `count`, the count the store's host keeps for the member of
    /// `slot`, in the place of the one handed before ([`HostCount`]). It
    /// takes a step for each lease that books the member and for each
    /// session those leases count for, and the one
    /// before it gives each of them up. `false` for a slot that names no
    /// cell: nothing is kept. A membership write's call, under the
    /// workers lock as the member's leaving is: no count is handed for a
    /// member that left.
    pub(crate) fn beside(&self, slot: WorkerSlot, count: Box<dyn HostCount>) -> bool {
        let (Some(cell), Some(place)) = (self.cell(slot), self.hosts.cell(slot)) else {
            return false;
        };
        let mut stood = place.count.write();
        // Added before they are taken: a number the two counts share
        // reads high between the two, never low.
        let standing = cell.load(Ordering::Relaxed);
        let sessions = place.sessions.lock();
        (0..standing).for_each(|_| count.booked());
        for (&session, &leases) in sessions.iter() {
            (0..leases).for_each(|_| count.session_booked(session));
        }
        if let Some(before) = stood.as_deref() {
            (0..standing).for_each(|_| before.freed());
            for (&session, &leases) in sessions.iter() {
                (0..leases).for_each(|_| before.session_freed(session));
            }
        }
        *stood = Some(count);
        true
    }

    /// The member of `slot` left the store: the count its host handed
    /// gives up the leases that still book the member, and is let go. A
    /// lease that outlives its worker's row steps the cell alone from
    /// here on. The place gives its table of sessions back with what the
    /// table held, and keeps no session behind the leaving: nothing is
    /// handed for the member again that would take one. A membership
    /// write's call, under the workers lock.
    pub(crate) fn left(&self, slot: WorkerSlot) {
        let (Some(cell), Some(place)) = (self.cell(slot), self.hosts.cell(slot)) else {
            return;
        };
        let mut stood = place.count.write();
        let mut sessions = place.sessions.lock();
        if let Some(count) = stood.take() {
            (0..cell.load(Ordering::Relaxed)).for_each(|_| count.freed());
            for (&session, &leases) in sessions.iter() {
                (0..leases).for_each(|_| count.session_freed(session));
            }
        }
        *sessions = FxHashMap::default();
        place.left.store(true, Ordering::Relaxed);
    }

    /// The place of the host's count for the member of `slot`, held shared
    /// for one step of the member's cell ([`HostPlace`]). Its piece was
    /// laid with the cell's ([`Self::claim`]).
    #[inline]
    fn stepping(&self, slot: WorkerSlot) -> Option<Stepping<'_>> {
        Some(self.hosts.cell(slot)?.count.read())
    }

    /// A lease's booking moves from the worker of `from` to the worker of
    /// `to`: added before it is taken. Where a thread's writes are seen in
    /// the order it made them (x86-64), a reader that reads the worker
    /// the booking leaves and then the worker it goes to counts the
    /// request on one of the two or on both. A reader that meets the two
    /// in the other order may count it on neither, as before the order.
    /// The writes and the reads are relaxed: a machine that orders them
    /// more loosely may show any reader the taking first. The count of
    /// `session`, the lease's where it counts for its key, moves with
    /// the lease's own: each worker's two under one hold of its place.
    pub(crate) fn moved(&self, from: Option<WorkerSlot>, to: Option<WorkerSlot>, session: Session) {
        if let Some(to) = to {
            self.book(to, session);
        }
        if let Some(from) = from {
            self.free(from, session);
        }
    }

    /// The count of `slot` at this call: zero for a slot that names no
    /// cell.
    pub(crate) fn in_flight(&self, slot: WorkerSlot) -> u32 {
        self.cell(slot)
            .map_or(0, |cell| cell.load(Ordering::Relaxed))
    }

    /// The table lent to one decision beside the rows whose slots are
    /// `slots`, in the rows' order: `kept` is opened for the decision and
    /// holds what it reads.
    #[cfg(test)]
    pub(crate) fn lend<'a>(
        &'a self,
        slots: &'a [WorkerSlot],
        kept: &'a mut InFlightBuffers,
    ) -> PoolInFlightCounts<'a> {
        kept.open(slots.len());
        self.lent(slots, kept)
    }

    /// The table beside `slots` with the cells a decision keeps its reads
    /// in, as they stand.
    #[cfg(test)]
    fn lent<'a>(
        &'a self,
        slots: &'a [WorkerSlot],
        kept: &'a InFlightBuffers,
    ) -> PoolInFlightCounts<'a> {
        PoolInFlightCounts {
            counts: self,
            slots,
            kept,
        }
    }

    /// The table's first piece: the cells of the first 1,024 members that
    /// ever stood, and no cell before the first slot is given.
    fn first(&self) -> &[AtomicU32] {
        self.cells.first()
    }

    fn cell(&self, slot: WorkerSlot) -> Option<&AtomicU32> {
        self.cells.cell(slot)
    }
}

/// What the leases on file book on each row of a pool, lent to a decision
/// beside rows that carry no count: the table and the pool's slots, a
/// row's count found by the row's position
/// ([`RequestFacts::in_flight`](crate::engine::RequestFacts::in_flight)). A
/// decision reads one count a row: the row's first read is the table's
/// count at that call, kept in the decision's own cells, and every later
/// read of that row returns it, whatever is booked meanwhile. A row no
/// stage reads costs nothing.
#[derive(Clone, Copy)]
pub struct PoolInFlightCounts<'a> {
    counts: &'a InFlightCounts,
    /// Each row's slot, by the row's position.
    slots: &'a [WorkerSlot],
    /// What the decision read of each row, by the row's position, under
    /// its stamp.
    kept: &'a InFlightBuffers,
}

impl<'a> PoolInFlightCounts<'a> {
    /// The count of the row at `position`, one number for the decision:
    /// zero past the rows.
    #[inline]
    pub(crate) fn at(&self, position: u32) -> u32 {
        self.run().at(position)
    }

    /// The table's first piece, the slots and the decision's cells, each
    /// found once for the reads of one run.
    #[inline]
    pub(crate) fn run(&self) -> LentRun<'a> {
        #[cfg(test)]
        tests::RUN_READERS.set(tests::RUN_READERS.get() + 1);
        LentRun {
            first: self.counts.first(),
            counts: self.counts,
            slots: self.slots,
            kept: &self.kept.cells,
            stamp: self.kept.stamp,
        }
    }
}

/// Runs a walk of rows with the reader of the rows' requests in flight, a
/// function of a row. The walk is written once and compiled for each kind
/// of rows: over rows that carry their counts the reader returns the row's
/// field, beside a lent table it asks the table's [`LentRun`]. The
/// request's facts are asked which once a walk, so no read tests it and
/// the lent table's parts are found once. A walk reads what the one reader
/// reads ([`RequestFacts::in_flight`](crate::engine::RequestFacts::in_flight)),
/// and a stage that reads a row or two asks that. The form
/// `|counts, written|` also binds `written`: true over rows that carry
/// their counts, false beside a lent table.
macro_rules! with_counts {
    ($facts:expr, |$counts:ident| $walk:expr) => {
        $crate::tracking::in_flight::with_counts!($facts, |$counts, _written| $walk)
    };
    ($facts:expr, |$counts:ident, $written:ident| $walk:expr) => {
        match $facts.lent_in_flight {
            Some(lent) => {
                let run = lent.run();
                let $counts = move |row: &$crate::engine::view::Candidate| run.of(row);
                let $written = false;
                $walk
            }
            None => {
                let $counts = |row: &$crate::engine::view::Candidate| row.in_flight;
                let $written = true;
                $walk
            }
        }
    };
}
pub(crate) use with_counts;

/// The reader of a run beside a lent table: each part of the table by
/// value, so a read loads none of them again.
#[derive(Clone, Copy)]
pub(crate) struct LentRun<'a> {
    /// The table's first piece, read with no search.
    first: &'a [AtomicU32],
    counts: &'a InFlightCounts,
    slots: &'a [WorkerSlot],
    /// The decision's cells and the stamp its reads are kept under.
    kept: &'a [AtomicU64],
    stamp: u32,
}

impl LentRun<'_> {
    /// The requests in flight of `row`, the decision's one number for it.
    #[inline]
    pub(crate) fn of(&self, row: &Candidate) -> u32 {
        self.at(row.position)
    }

    /// The slot of `row`: the one that names no cell past the rows.
    #[inline]
    pub(crate) fn slot_of(&self, row: &Candidate) -> WorkerSlot {
        let slot = self.slots.get(row.position as usize);
        slot.copied().unwrap_or(WorkerSlot::NONE)
    }

    /// The count of the row at `position`, one number for the decision:
    /// zero past the rows.
    #[inline]
    fn at(&self, position: u32) -> u32 {
        let Some(kept) = self.kept.get(position as usize) else {
            return 0;
        };
        let held = kept.load(Ordering::Relaxed);
        if (held >> 32) as u32 == self.stamp {
            return held as u32;
        }
        let count = self.read(position);
        let read = u64::from(self.stamp) << 32 | u64::from(count);
        kept.store(read, Ordering::Relaxed);
        count
    }

    /// The table's count of the row at `position` at this call.
    #[inline]
    fn read(&self, position: u32) -> u32 {
        let Some(&slot) = self.slots.get(position as usize) else {
            return 0;
        };
        match self.first.get(slot.0 as usize) {
            Some(cell) => cell.load(Ordering::Relaxed),
            None => self.counts.in_flight(slot),
        }
    }
}

/// What one decision read of its rows' counts: a cell a position, in the
/// buffers its host keeps across decisions. A cell holds a count under the
/// stamp of the decision that read it, so a decision finds every cell
/// unread and none is cleared between decisions. The cells are read and
/// written with no lock and no order: one decision runs on one thread.
#[derive(Debug, Default)]
pub struct InFlightBuffers {
    /// A stamp in the high half, the count it read in the low half.
    cells: Vec<AtomicU64>,
    /// The stamp of the decision the cells are open for; zero before the
    /// first, and the stamp of no decision.
    stamp: u32,
}

#[cfg(test)]
impl InFlightBuffers {
    /// Opens the cells for one decision over `rows` rows: each reads as
    /// unread. One add a decision. The stamps come round once in 2^32
    /// decisions, and every cell is cleared then: a count kept under the
    /// first of them would read as this decision's.
    fn open(&mut self, rows: usize) {
        if self.cells.len() < rows {
            self.cells.resize_with(rows, AtomicU64::default);
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            for cell in &mut self.cells {
                *cell.get_mut() = 0;
            }
            self.stamp = 1;
        }
    }
}

impl std::fmt::Debug for PoolInFlightCounts<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolInFlightCounts")
            .field("rows", &self.slots.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        cell::OnceCell,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Barrier,
        },
        thread,
    };

    use super::{InFlightBuffers, InFlightCounts, PoolInFlightCounts, WorkerSlot};
    use crate::engine::view::{Candidate, RequestFacts, Session};

    thread_local! {
        pub(crate) static RUN_READERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// A pool's counts taken out of its rows into a table of their own: the
    /// lent form of rows a case wrote its counts into.
    pub(crate) struct Lent<'a> {
        counts: InFlightCounts,
        slots: Vec<WorkerSlot>,
        /// The cells of the one decision a case's reads stand for.
        kept: InFlightBuffers,
        /// The table as a case's facts name it, lent at the first ask.
        lent: OnceCell<PoolInFlightCounts<'a>>,
    }

    impl<'a> Lent<'a> {
        /// Takes each row's count out of `rows` into a table, under the
        /// row's position: the rows read zero after it. A worker a case
        /// lists twice has one count.
        pub(crate) fn take(rows: &mut [Candidate]) -> Self {
            let counts = InFlightCounts::default();
            let places = rows.iter().map(|row| row.position as usize + 1).max();
            let mut slots = vec![WorkerSlot::NONE; places.unwrap_or(0)];
            for row in rows {
                let held = std::mem::take(&mut row.in_flight);
                let place = &mut slots[row.position as usize];
                if *place == WorkerSlot::NONE {
                    *place = counts.claim();
                    for _ in 0..held {
                        counts.book(*place, Session::NONE);
                    }
                }
                assert_eq!(counts.in_flight(*place), held, "one count a position");
            }
            let mut kept = InFlightBuffers::default();
            kept.open(slots.len());
            Self {
                counts,
                slots,
                kept,
                lent: OnceCell::new(),
            }
        }

        /// The slot the count of `row` stands in.
        pub(crate) fn slot_of(&self, row: &Candidate) -> WorkerSlot {
            self.slots[row.position as usize]
        }

        /// `facts` with the table lent beside the rows: every call lends
        /// the one decision a case's reads stand for.
        pub(crate) fn beside(&'a self, facts: &RequestFacts<'a>) -> RequestFacts<'a> {
            let lent = self
                .lent
                .get_or_init(|| self.counts.lent(&self.slots, &self.kept));
            RequestFacts {
                lent_in_flight: Some(lent),
                ..*facts
            }
        }
    }

    /// A decision reads one count a row: a row's first read is kept, and a
    /// booking after it moves no later read of that decision. A row read
    /// first after the booking reads it, and the next decision reads every
    /// row anew.
    #[test]
    fn a_rows_first_read_stands_for_its_decision_alone() {
        let counts = InFlightCounts::default();
        let slots = [counts.claim(), counts.claim()];
        counts.book(slots[0], Session::NONE);
        let mut kept = InFlightBuffers::default();
        let first = counts.lend(&slots, &mut kept);
        assert_eq!(first.at(0), 1);
        counts.book(slots[0], Session::NONE);
        counts.book(slots[1], Session::NONE);
        assert_eq!(
            (first.at(0), first.at(1)),
            (1, 1),
            "a kept read, a first read"
        );
        let next = counts.lend(&slots, &mut kept);
        assert_eq!((next.at(0), next.at(1)), (2, 1));
    }

    /// The stamps come round once in 2^32 decisions and the cells are
    /// cleared then: a count kept under the first stamp is not read as the
    /// count of the decision that opens under that stamp again.
    #[test]
    fn a_count_kept_under_a_stamp_is_not_read_when_the_stamps_come_round() {
        let counts = InFlightCounts::default();
        let slots = [counts.claim()];
        let mut kept = InFlightBuffers::default();
        assert_eq!(counts.lend(&slots, &mut kept).at(0), 0);
        counts.book(slots[0], Session::NONE);
        // The last stamp before the round.
        kept.stamp = u32::MAX;
        assert_eq!(counts.lend(&slots, &mut kept).at(0), 1);
        assert_eq!(kept.stamp, 1);
    }

    /// A walk is handed the reader of the rows its request was given: the
    /// row's field over rows that carry their counts, the lent count
    /// beside a lent table.
    #[test]
    fn a_walk_reads_by_the_kind_of_rows_its_request_was_given() {
        let row = |position, in_flight| Candidate {
            position,
            in_flight,
            ..Candidate::default()
        };
        let mut rows = [row(0, 4), row(1, 2)];
        let read = |facts: &RequestFacts<'_>, rows: &[Candidate]| -> Vec<u32> {
            with_counts!(facts, |counts| rows.iter().map(counts).collect())
        };
        let own = RequestFacts::default();
        assert_eq!(read(&own, &rows), [4, 2]);
        let lent = Lent::take(&mut rows);
        // The counts stand in the table: what the rows hold is not read.
        rows[0].in_flight = 9;
        assert_eq!(read(&lent.beside(&own), &rows), [4, 2]);
    }

    /// A run's reader and a single read share the decision's one number
    /// for a row: the first read by either is kept and the other returns
    /// it, whatever is booked between the two. A reader taken later in the
    /// decision reads the same cells.
    #[test]
    fn a_runs_reader_and_a_single_read_share_a_decisions_one_number() {
        let counts = InFlightCounts::default();
        let slots = [counts.claim(), counts.claim()];
        counts.book(slots[0], Session::NONE);
        let mut kept = InFlightBuffers::default();
        let lent = counts.lend(&slots, &mut kept);
        let run = lent.run();
        // A row beside a lent table: no reader returns its own field.
        let row = |position| Candidate {
            position,
            in_flight: 7,
            ..Candidate::default()
        };
        assert_eq!(run.of(&row(0)), 1, "the run reads the row first");
        counts.book(slots[0], Session::NONE);
        counts.book(slots[1], Session::NONE);
        assert_eq!(lent.at(0), 1, "a single read after the run's");
        assert_eq!(lent.at(1), 1, "a single read first");
        counts.book(slots[1], Session::NONE);
        assert_eq!((run.of(&row(0)), run.of(&row(1))), (1, 1));
        let later = lent.run();
        let read = [0, 1, 2].map(|position| later.of(&row(position)));
        assert_eq!(read, [1, 1, 0], "zero past the rows");
    }

    /// A lent count is read at its row's position, whichever piece of the
    /// table holds the row's slot: zero for a slot that names no cell and
    /// past the rows.
    #[test]
    fn a_lent_count_is_read_at_its_rows_position() {
        let counts = InFlightCounts::default();
        // Slots through the third piece: 1,024 in the first, 2,048 in the
        // second.
        let slots: Vec<WorkerSlot> = (0..3_073).map(|_| counts.claim()).collect();
        // A pool whose rows stand in each piece, one of them with no cell.
        let rows = [
            slots[3_072],
            slots[2],
            WorkerSlot::NONE,
            slots[1_024],
            slots[1_023],
        ];
        for (&slot, held) in rows.iter().zip([4, 1, 9, 2, 3]) {
            for _ in 0..held {
                counts.book(slot, Session::NONE);
            }
        }
        let mut kept = InFlightBuffers::default();
        let lent = counts.lend(&rows, &mut kept);
        let read: Vec<u32> = (0..6).map(|position| lent.at(position)).collect();
        assert_eq!(read, [4, 1, 0, 2, 3, 0]);
    }

    /// Two files book one worker where it serves two models: a write is one
    /// atomic step, so no booking of either writer is lost under the
    /// other's, and nothing they give back.
    #[test]
    fn two_writers_of_one_cell_lose_no_count() {
        const EACH: u32 = 400_000;
        let counts = InFlightCounts::default();
        let slot = counts.claim();
        let start = Barrier::new(2);
        let both = |write: fn(&InFlightCounts, WorkerSlot, Session)| {
            thread::scope(|scope| {
                for _ in 0..2 {
                    scope.spawn(|| {
                        start.wait();
                        for _ in 0..EACH {
                            write(&counts, slot, Session::NONE);
                        }
                    });
                }
            });
            counts.in_flight(slot)
        };
        assert_eq!(
            both(InFlightCounts::book),
            2 * EACH,
            "every booking of both"
        );
        assert_eq!(both(InFlightCounts::free), 0, "and each one given back");
    }

    /// A step the cell does not take is no step of the host's count: an
    /// end of a cell at zero, and a booking of a cell that holds every
    /// lease it can. A slot that names no cell keeps no count.
    #[test]
    fn a_step_the_cell_does_not_take_steps_no_hosts_count() {
        use std::sync::{atomic::AtomicU32, Arc};

        use super::HostCount;

        /// The bookings and the ends a count was told.
        #[derive(Default)]
        struct Told {
            bookings: AtomicU32,
            ends: AtomicU32,
        }

        struct Count(Arc<Told>);

        impl HostCount for Count {
            fn booked(&self) {
                self.0.bookings.fetch_add(1, Ordering::Relaxed);
            }

            fn freed(&self) {
                self.0.ends.fetch_add(1, Ordering::Relaxed);
            }
        }

        let counts = InFlightCounts::default();
        let slot = counts.claim();
        let told = Arc::new(Told::default());
        let count = || Box::new(Count(Arc::clone(&told)));
        assert!(!counts.beside(WorkerSlot::NONE, count()), "no cell");
        assert!(counts.beside(slot, count()));
        let read = || {
            let bookings = told.bookings.load(Ordering::Relaxed);
            (bookings, told.ends.load(Ordering::Relaxed))
        };
        counts.free(slot, Session::NONE);
        assert_eq!((counts.in_flight(slot), read()), (0, (0, 0)), "at zero");
        let cell = counts.cell(slot).expect("the slot's cell");
        cell.store(u32::MAX, Ordering::Relaxed);
        counts.book(slot, Session::NONE);
        assert_eq!(read(), (0, 0), "a cell that holds every lease it can");
        counts.free(slot, Session::NONE);
        assert_eq!((counts.in_flight(slot), read()), (u32::MAX - 1, (0, 1)));
        counts.book(slot, Session::NONE);
        assert_eq!(read(), (1, 1), "the last booking the cell takes");
    }

    /// Writers of one cell with no file's turn between them, while the
    /// cell's count is handed and then while its member leaves: the count
    /// takes what the cell holds at the hand-over and gives up what it
    /// holds at the leaving, and no step reaches it half made.
    #[test]
    fn a_count_handed_and_taken_among_writers_holds_what_the_cell_holds() {
        use std::sync::{atomic::AtomicI64, Arc};

        use super::HostCount;

        /// The leases a count holds: never an end with no booking under it.
        struct Held(Arc<AtomicI64>);

        impl HostCount for Held {
            fn booked(&self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }

            fn freed(&self) {
                let held = self.0.fetch_sub(1, Ordering::SeqCst);
                assert!(held > 0, "an end with no booking under it");
            }
        }

        /// Says a writer ended, however it ended.
        struct Ended<'a>(&'a AtomicUsize);

        impl Drop for Ended<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        const WRITERS: usize = 4;
        const EACH: usize = 50_000;
        let counts = InFlightCounts::default();
        let slot = counts.claim();
        let held = Arc::new(AtomicI64::new(0));
        // Four writers book twice and free once, `EACH` times: `among`
        // runs once they are under way, or once each ended.
        let among_writers = |among: &dyn Fn()| {
            let (wrote, ended) = (AtomicUsize::new(0), AtomicUsize::new(0));
            thread::scope(|scope| {
                for _ in 0..WRITERS {
                    scope.spawn(|| {
                        let _ended = Ended(&ended);
                        for _ in 0..EACH {
                            counts.book(slot, Session::NONE);
                            counts.book(slot, Session::NONE);
                            counts.free(slot, Session::NONE);
                            wrote.fetch_add(1, Ordering::SeqCst);
                        }
                    });
                }
                let under_way = || wrote.load(Ordering::SeqCst) >= 1_000;
                while !under_way() && ended.load(Ordering::SeqCst) < WRITERS {
                    thread::yield_now();
                }
                among();
            });
        };
        let wrote = i64::try_from(WRITERS * EACH).expect("a small count");

        among_writers(&|| {
            let count = Box::new(Held(Arc::clone(&held)));
            assert!(counts.beside(slot, count));
        });
        assert_eq!(i64::from(counts.in_flight(slot)), wrote);
        assert_eq!(held.load(Ordering::SeqCst), wrote, "what the cell holds");

        among_writers(&|| counts.left(slot));
        assert_eq!(i64::from(counts.in_flight(slot)), 2 * wrote);
        assert_eq!(held.load(Ordering::SeqCst), 0, "every lease given up");
    }

    /// A member that leaves gives its place's table back, with the
    /// sessions of the leases that outlive it, and its place keeps no
    /// session of a lease that books it behind the leaving. No count is
    /// handed for such a member again, so nothing would take them, and a
    /// place is never another member's: what it kept would stand for the
    /// store's life. A member that stands keeps its table's room when its
    /// last session leaves.
    #[test]
    fn a_member_that_left_keeps_no_session_and_no_table() {
        let counts = InFlightCounts::default();
        let slot = counts.claim();
        let kept = || (counts.sessions_kept(slot), counts.sessions_room(slot));
        for session in 1..=64 {
            counts.book(slot, Session::from_raw(session));
        }
        let (sessions, room) = kept();
        assert_eq!(sessions, 64);
        assert!(room >= 64, "room for the sessions that stand: {room}");
        for session in 1..=64 {
            counts.free(slot, Session::from_raw(session));
        }
        assert_eq!(kept(), (0, room), "a standing member keeps the room");
        for session in 1..=64 {
            counts.book(slot, Session::from_raw(session));
        }
        counts.left(slot);
        assert_eq!(kept(), (0, 0), "the table is given back at the leaving");
        // A lease that outlives the member ends, and a late one books it.
        counts.free(slot, Session::from_raw(1));
        counts.book(slot, Session::from_raw(65));
        assert_eq!(kept(), (0, 0), "no session is kept behind the leaving");
        assert_eq!(counts.in_flight(slot), 64, "the cell steps as it did");
    }

    /// A member's place keeps a session while a lease of it books the
    /// member, and the session's entry leaves with its last lease. A lease
    /// that counts for no key is kept under no session, and an end of a
    /// session the place does not keep takes nothing from it.
    #[test]
    fn a_place_keeps_a_session_until_its_last_lease_ends() {
        let counts = InFlightCounts::default();
        let slot = counts.claim();
        let kept = || (counts.in_flight(slot), counts.sessions_kept(slot));
        let (one, two) = (Session::from_raw(7), Session::from_raw(9));
        counts.book(slot, Session::NONE);
        assert_eq!(kept(), (1, 0), "a lease that counts for no key");
        counts.book(slot, one);
        counts.book(slot, one);
        counts.book(slot, two);
        assert_eq!(kept(), (4, 2));
        counts.free(slot, one);
        assert_eq!(kept(), (3, 2), "a lease of the first still books it");
        counts.free(slot, two);
        counts.free(slot, Session::NONE);
        assert_eq!(kept(), (1, 1));
        counts.free(slot, one);
        assert_eq!(kept(), (0, 0), "the last keyed lease ended");
        // A move takes the session with the lease.
        let next = counts.claim();
        counts.book(slot, two);
        counts.moved(Some(slot), Some(next), two);
        assert_eq!((kept(), counts.sessions_kept(next)), ((0, 0), 1));
        counts.free(next, one);
        assert_eq!(counts.sessions_kept(next), 1, "a session it does not keep");
        counts.free(next, two);
        assert_eq!(counts.sessions_kept(next), 0);
    }

    /// A count handed in the place of another takes every step before the
    /// one before it gives any up, the leases' and each session's alike: a
    /// number the two counts share reads high between the two and never
    /// low. A member that leaves gives up the same steps, and a session's
    /// end after it reaches no count.
    #[test]
    fn a_count_handed_takes_its_steps_before_the_one_before_gives_them_up() {
        use std::sync::Arc;

        use parking_lot::Mutex;

        use super::HostCount;

        /// What the counts of one member were told, in order, each line
        /// under its count's name.
        struct Says(&'static str, Arc<Mutex<Vec<String>>>);

        impl Says {
            fn say(&self, step: &str) {
                self.1.lock().push(format!("{} {step}", self.0));
            }
        }

        impl HostCount for Says {
            fn booked(&self) {
                self.say("booked");
            }

            fn freed(&self) {
                self.say("freed");
            }

            fn session_booked(&self, session: Session) {
                self.say(&format!("booked {}", session.get()));
            }

            fn session_freed(&self, session: Session) {
                self.say(&format!("freed {}", session.get()));
            }
        }

        let counts = InFlightCounts::default();
        let slot = counts.claim();
        let said = Arc::new(Mutex::new(Vec::new()));
        let count = |name| Box::new(Says(name, Arc::clone(&said)));
        let told = || std::mem::take(&mut *said.lock());
        // A lease filed ahead of the member's count, then the count.
        counts.book(slot, Session::from_raw(7));
        assert!(counts.beside(slot, count("first")));
        assert_eq!(told(), ["first booked", "first booked 7"]);
        assert!(counts.beside(slot, count("second")));
        let handed = [
            "second booked",
            "second booked 7",
            "first freed",
            "first freed 7",
        ];
        assert_eq!(told(), handed);
        // An end of a session the place does not keep reaches no count:
        // its lease's own step alone does.
        counts.book(slot, Session::NONE);
        counts.free(slot, Session::from_raw(9));
        assert_eq!(told(), ["second booked", "second freed"]);
        counts.left(slot);
        assert_eq!(told(), ["second freed", "second freed 7"]);
        // The lease outlives its member: its end steps the place alone.
        counts.free(slot, Session::from_raw(7));
        assert_eq!((told().len(), counts.sessions_kept(slot)), (0, 0));
    }

    /// A booking that moves is added before it is taken: a reader beside the
    /// mover that reads the worker it leaves and then the worker it goes to
    /// counts the request on one of the two or on both, never on neither.
    /// Read where another thread sees a thread's writes in their order: a
    /// machine that orders them more loosely may show the taking first. On
    /// one core a reader seldom lands between the two steps, and the case
    /// never fails a right order.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn a_moved_booking_is_never_read_on_neither_worker() {
        const MOVES: usize = 50_000;
        let counts = InFlightCounts::default();
        // A pair of workers a move, the booking on the first of them.
        let pairs: Vec<[WorkerSlot; 2]> = (0..MOVES)
            .map(|_| [counts.claim(), counts.claim()])
            .collect();
        for [from, _] in &pairs {
            counts.book(*from, Session::NONE);
        }
        let read = |[from, to]: &[WorkerSlot; 2]| (counts.in_flight(*from), counts.in_flight(*to));
        let at = AtomicUsize::new(0);
        thread::scope(|scope| {
            scope.spawn(|| {
                for (pair, [from, to]) in pairs.iter().enumerate() {
                    at.store(pair, Ordering::Relaxed);
                    counts.moved(Some(*from), Some(*to), Session::NONE);
                }
                at.store(MOVES, Ordering::Relaxed);
            });
            while let Some(pair) = pairs.get(at.load(Ordering::Relaxed)) {
                let (left, came) = read(pair);
                assert!(left + came >= 1, "a moved booking read on neither worker");
            }
        });
        assert!(
            pairs.iter().all(|pair| read(pair) == (0, 1)),
            "each moved once"
        );
    }
}
