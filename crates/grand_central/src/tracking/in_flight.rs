//! The counts of requests in flight: one cell a worker, booked by the
//! leases the store holds on file.
//!
//! The store keeps the table (`InFlightCounts`) and books it under its leases'
//! lock. No decision reads the table: a decision reads the count each of
//! its rows carries.
//!
//! A host that counts a worker's requests on an object of its own hands
//! the table that count ([`HostCount`]), and the table keeps it equal to
//! the worker's cell: the engine over worker lists reads the object's count.

use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    OnceLock,
};

use parking_lot::{Mutex, RwLock, RwLockReadGuard};
use rustc_hash::FxHashMap;

use crate::engine::view::Session;

/// The place of one worker's count in the store's counts. A member is
/// given its slot at the membership write that first files it and keeps
/// it while its row stands, and no other member is given it after.
/// In-process only: the store gives a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct WorkerSlot(u32);

impl WorkerSlot {
    /// The slot that names no cell: its count reads zero and books
    /// nothing. What a table that gave every slot gives.
    pub(crate) const NONE: Self = Self(u32::MAX);
}

/// The cells of the first piece of the counts; each piece after holds
/// twice the cells of the one before.
const FIRST: u64 = 1 << 10;

/// The pieces of the counts: together they hold every slot but the last
/// 1,024 a `u32` names, [`WorkerSlot::NONE`] among those.
const PIECES: usize = 22;

/// Cells by slot, grown in pieces that never move: piece `k` holds the
/// `FIRST << k` cells from slot `(FIRST << k) - FIRST` on, laid when a
/// slot of it is first asked for. The counts stand in one and the hosts'
/// counts in another, beside them.
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
    /// The cell of `slot` where its piece is laid.
    pub(crate) fn cell(&self, slot: WorkerSlot) -> Option<&T> {
        let (cells, first) = self.piece(slot);
        cells.get(slot.0.wrapping_sub(first) as usize)
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

/// What the leases on file book on each worker: the number a host's count
/// of the worker is kept equal to ([`HostCount`]).
///
/// One table for the store's life, a cell a member that ever stood, grown
/// in pieces that never move. A membership write gives a joining member
/// the next cell and does nothing else for a count, a pool's shape names
/// each row's cell by its slot, and a lease on file keeps the slot it
/// booked. So a count outlives a membership write, and a lease of a worker
/// that left ends on the count it booked.
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
    #[cfg(test)]
    pub(crate) fn in_flight(&self, slot: WorkerSlot) -> u32 {
        self.cell(slot)
            .map_or(0, |cell| cell.load(Ordering::Relaxed))
    }

    fn cell(&self, slot: WorkerSlot) -> Option<&AtomicU32> {
        self.cells.cell(slot)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/tracking/in_flight/tests.rs"]
mod tests;
