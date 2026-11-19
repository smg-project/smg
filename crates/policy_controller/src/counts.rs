//! The counts of requests in flight: one cell a worker, booked by the
//! leases the store holds on file and read by a decision beside its rows.
//!
//! The store keeps the table (`Counts`) and books it under its leases'
//! lock. A decision over a publication's own rows is lent the table with
//! the pool's slots ([`LentCounts`]) and reads a row's count by the row's
//! position: nothing is written into a row and no lock is taken. A row's
//! first read in a decision is kept ([`KeptCounts`]) and every later read
//! returns it, so a decision reads one count a row.

use std::sync::{
    atomic::{AtomicU32, AtomicU64, Ordering},
    OnceLock,
};

use crate::view::Candidate;

/// The place of one worker's count in the store's counts. A member is
/// given its slot at the membership write that first files it and keeps
/// it while its row stands, and no other member is given it after.
/// In-process only: the store gives a slot, and a host that keeps no
/// counts names [`Slot::NONE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot(u32);

impl Slot {
    /// The slot that names no cell: its count reads zero and books
    /// nothing. What a table that gave every slot gives.
    pub const NONE: Self = Self(u32::MAX);
}

/// The cells of the first piece of the counts; each piece after holds
/// twice the cells of the one before.
const FIRST: u64 = 1 << 10;

/// The pieces of the counts: together they hold every slot but the last
/// 1,024 a `u32` names, [`Slot::NONE`] among those.
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
    pub(crate) fn lay(&self, slot: Slot) -> Option<&T> {
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
    pub(crate) fn cell(&self, slot: Slot) -> Option<&T> {
        let (cells, first) = self.piece(slot);
        cells.get(slot.0.wrapping_sub(first) as usize)
    }

    /// The cell of `slot` read through `first`, the first piece found once
    /// for a run of reads: no search for a slot the first piece holds.
    #[inline]
    pub(crate) fn cell_from<'a>(&'a self, first: &'a [T], slot: Slot) -> Option<&'a T> {
        match first.get(slot.0 as usize) {
            Some(cell) => Some(cell),
            None => self.cell(slot),
        }
    }

    /// The cells of the piece that holds `slot` and the piece's first
    /// slot: no cells where the piece is not laid yet or no piece holds
    /// the slot.
    fn piece(&self, slot: Slot) -> (&[T], u32) {
        let (piece, first) = Self::place(slot);
        let cells = self.pieces.get(piece).and_then(OnceLock::get);
        (cells.map_or(&[], |cells| cells), first)
    }

    /// The piece `slot` falls in and that piece's first slot.
    fn place(slot: Slot) -> (usize, u32) {
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
/// [`RequestFacts::in_flight`]: crate::view::RequestFacts::in_flight
#[derive(Debug, Default)]
pub(crate) struct Counts {
    /// A cell a slot, its piece laid when its first slot is given.
    cells: Pieces<AtomicU32>,
    /// The slots given so far.
    given: AtomicU32,
}

impl Counts {
    /// The next slot of the table, its cell at zero. A table that gave
    /// every slot gives the one that names no cell ([`Slot::NONE`]).
    pub(crate) fn claim(&self) -> Slot {
        let next = |given: u32| given.checked_add(1);
        let given = self
            .given
            .try_update(Ordering::Relaxed, Ordering::Relaxed, next);
        let Ok(slot) = given.map(Slot) else {
            return Slot::NONE;
        };
        match self.cells.lay(slot) {
            Some(_) => slot,
            None => Slot::NONE,
        }
    }

    /// The counts of `slots`, in their order: one load a slot, the piece
    /// found once for each run of slots that share it.
    pub(crate) fn of<'a>(&'a self, slots: &'a [Slot]) -> impl Iterator<Item = u32> + 'a {
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
    pub(crate) fn book(&self, slot: Slot) {
        if let Some(cell) = self.cell(slot) {
            let one_more = |held: u32| held.saturating_add(1);
            cell.update(Ordering::Relaxed, Ordering::Relaxed, one_more);
        }
    }

    /// One lease that booked the worker of `slot` left the file, in one
    /// atomic step as well: a count at zero stays there.
    pub(crate) fn free(&self, slot: Slot) {
        if let Some(cell) = self.cell(slot) {
            let one_less = |held: u32| held.saturating_sub(1);
            cell.update(Ordering::Relaxed, Ordering::Relaxed, one_less);
        }
    }

    /// A lease's booking moves from the worker of `from` to the worker of
    /// `to`: added before it is taken. Where a thread's writes are seen in
    /// the order it made them (x86-64), a reader that reads the worker
    /// the booking leaves and then the worker it goes to counts the
    /// request on one of the two or on both. A reader that meets the two
    /// in the other order may count it on neither, as before the order.
    /// The writes and the reads are relaxed: a machine that orders them
    /// more loosely may show any reader the taking first.
    pub(crate) fn moved(&self, from: Option<Slot>, to: Option<Slot>) {
        if let Some(to) = to {
            self.book(to);
        }
        if let Some(from) = from {
            self.free(from);
        }
    }

    /// The count of `slot` at this call: zero for a slot that names no
    /// cell.
    pub(crate) fn in_flight(&self, slot: Slot) -> u32 {
        self.cell(slot)
            .map_or(0, |cell| cell.load(Ordering::Relaxed))
    }

    /// The table lent to one decision beside the rows whose slots are
    /// `slots`, in the rows' order: `kept` is opened for the decision and
    /// holds what it reads.
    pub(crate) fn lend<'a>(
        &'a self,
        slots: &'a [Slot],
        kept: &'a mut KeptCounts,
    ) -> LentCounts<'a> {
        kept.open(slots.len());
        self.lent(slots, kept)
    }

    /// The table beside `slots` with the cells a decision keeps its reads
    /// in, as they stand.
    fn lent<'a>(&'a self, slots: &'a [Slot], kept: &'a KeptCounts) -> LentCounts<'a> {
        LentCounts {
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

    fn cell(&self, slot: Slot) -> Option<&AtomicU32> {
        self.cells.cell(slot)
    }
}

/// What the leases on file book on each row of a pool, lent to a decision
/// beside rows that carry no count: the table and the pool's slots, a
/// row's count found by the row's position
/// ([`RequestFacts::in_flight`](crate::RequestFacts::in_flight)). A
/// decision reads one count a row: the row's first read is the table's
/// count at that call, kept in the decision's own cells, and every later
/// read of that row returns it, whatever is booked meanwhile. A row no
/// stage reads costs nothing.
#[derive(Clone, Copy)]
pub struct LentCounts<'a> {
    counts: &'a Counts,
    /// Each row's slot, by the row's position.
    slots: &'a [Slot],
    /// What the decision read of each row, by the row's position, under
    /// its stamp.
    kept: &'a KeptCounts,
}

impl<'a> LentCounts<'a> {
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
/// reads ([`RequestFacts::in_flight`](crate::RequestFacts::in_flight)),
/// and a stage that reads a row or two asks that.
macro_rules! with_counts {
    ($facts:expr, |$counts:ident| $walk:expr) => {
        match $facts.lent_in_flight {
            Some(lent) => {
                let run = lent.run();
                let $counts = move |row: &$crate::view::Candidate| run.of(row);
                $walk
            }
            None => {
                let $counts = |row: &$crate::view::Candidate| row.in_flight;
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
    counts: &'a Counts,
    slots: &'a [Slot],
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
    pub(crate) fn slot_of(&self, row: &Candidate) -> Slot {
        let slot = self.slots.get(row.position as usize);
        slot.copied().unwrap_or(Slot::NONE)
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
pub struct KeptCounts {
    /// A stamp in the high half, the count it read in the low half.
    cells: Vec<AtomicU64>,
    /// The stamp of the decision the cells are open for; zero before the
    /// first, and the stamp of no decision.
    stamp: u32,
}

impl KeptCounts {
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

impl std::fmt::Debug for LentCounts<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LentCounts")
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

    use super::{Counts, KeptCounts, LentCounts, Slot};
    use crate::view::{Candidate, RequestFacts};

    /// A pool's counts taken out of its rows into a table of their own: the
    /// lent form of rows a case wrote its counts into.
    pub(crate) struct Lent<'a> {
        counts: Counts,
        slots: Vec<Slot>,
        /// The cells of the one decision a case's reads stand for.
        kept: KeptCounts,
        /// The table as a case's facts name it, lent at the first ask.
        lent: OnceCell<LentCounts<'a>>,
    }

    impl<'a> Lent<'a> {
        /// Takes each row's count out of `rows` into a table, under the
        /// row's position: the rows read zero after it. A worker a case
        /// lists twice has one count.
        pub(crate) fn take(rows: &mut [Candidate]) -> Self {
            let counts = Counts::default();
            let places = rows.iter().map(|row| row.position as usize + 1).max();
            let mut slots = vec![Slot::NONE; places.unwrap_or(0)];
            for row in rows {
                let held = std::mem::take(&mut row.in_flight);
                let place = &mut slots[row.position as usize];
                if *place == Slot::NONE {
                    *place = counts.claim();
                    for _ in 0..held {
                        counts.book(*place);
                    }
                }
                assert_eq!(counts.in_flight(*place), held, "one count a position");
            }
            let mut kept = KeptCounts::default();
            kept.open(slots.len());
            Self {
                counts,
                slots,
                kept,
                lent: OnceCell::new(),
            }
        }

        /// The slot the count of `row` stands in.
        pub(crate) fn slot_of(&self, row: &Candidate) -> Slot {
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
        let counts = Counts::default();
        let slots = [counts.claim(), counts.claim()];
        counts.book(slots[0]);
        let mut kept = KeptCounts::default();
        let first = counts.lend(&slots, &mut kept);
        assert_eq!(first.at(0), 1);
        counts.book(slots[0]);
        counts.book(slots[1]);
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
        let counts = Counts::default();
        let slots = [counts.claim()];
        let mut kept = KeptCounts::default();
        assert_eq!(counts.lend(&slots, &mut kept).at(0), 0);
        counts.book(slots[0]);
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
        let counts = Counts::default();
        let slots = [counts.claim(), counts.claim()];
        counts.book(slots[0]);
        let mut kept = KeptCounts::default();
        let lent = counts.lend(&slots, &mut kept);
        let run = lent.run();
        // A row beside a lent table: no reader returns its own field.
        let row = |position| Candidate {
            position,
            in_flight: 7,
            ..Candidate::default()
        };
        assert_eq!(run.of(&row(0)), 1, "the run reads the row first");
        counts.book(slots[0]);
        counts.book(slots[1]);
        assert_eq!(lent.at(0), 1, "a single read after the run's");
        assert_eq!(lent.at(1), 1, "a single read first");
        counts.book(slots[1]);
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
        let counts = Counts::default();
        // Slots through the third piece: 1,024 in the first, 2,048 in the
        // second.
        let slots: Vec<Slot> = (0..3_073).map(|_| counts.claim()).collect();
        // A pool whose rows stand in each piece, one of them with no cell.
        let rows = [
            slots[3_072],
            slots[2],
            Slot::NONE,
            slots[1_024],
            slots[1_023],
        ];
        for (&slot, held) in rows.iter().zip([4, 1, 9, 2, 3]) {
            for _ in 0..held {
                counts.book(slot);
            }
        }
        let mut kept = KeptCounts::default();
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
        let counts = Counts::default();
        let slot = counts.claim();
        let start = Barrier::new(2);
        let both = |write: fn(&Counts, Slot)| {
            thread::scope(|scope| {
                for _ in 0..2 {
                    scope.spawn(|| {
                        start.wait();
                        for _ in 0..EACH {
                            write(&counts, slot);
                        }
                    });
                }
            });
            counts.in_flight(slot)
        };
        assert_eq!(both(Counts::book), 2 * EACH, "every booking of both");
        assert_eq!(both(Counts::free), 0, "and each one given back");
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
        let counts = Counts::default();
        // A pair of workers a move, the booking on the first of them.
        let pairs: Vec<[Slot; 2]> = (0..MOVES)
            .map(|_| [counts.claim(), counts.claim()])
            .collect();
        for [from, _] in &pairs {
            counts.book(*from);
        }
        let read = |[from, to]: &[Slot; 2]| (counts.in_flight(*from), counts.in_flight(*to));
        let at = AtomicUsize::new(0);
        thread::scope(|scope| {
            scope.spawn(|| {
                for (pair, [from, to]) in pairs.iter().enumerate() {
                    at.store(pair, Ordering::Relaxed);
                    counts.moved(Some(*from), Some(*to));
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
