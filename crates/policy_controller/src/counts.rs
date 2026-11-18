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

/// The place of one worker's count in the store's counts ([`Counts`]). A
/// member is given its slot at the membership write that first lists it
/// and keeps it while it stands, and no other member is given it after.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slot(u32);

impl Slot {
    /// The slot that names no cell: its count reads zero and books
    /// nothing. What a table that gave every slot gives.
    pub(crate) const NONE: Self = Self(u32::MAX);
}

/// The cells of the first piece of the counts; each piece after holds
/// twice the cells of the one before.
const FIRST: u64 = 1 << 10;

/// The pieces of the counts: together they hold every slot but the last
/// 1,024 a `u32` names, [`Slot::NONE`] among those.
const PIECES: usize = 22;

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
    /// Piece `k` holds the `FIRST << k` cells from slot
    /// `(FIRST << k) - FIRST` on: laid when its first slot is given.
    pieces: [OnceLock<Box<[AtomicU32]>>; PIECES],
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
        let (piece, _) = Self::place(slot);
        let Some(cells) = self.pieces.get(piece) else {
            return Slot::NONE;
        };
        cells.get_or_init(|| (0..FIRST << piece).map(|_| AtomicU32::new(0)).collect());
        slot
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
            (cells, first) = self.piece(slot);
            let at = slot.0.wrapping_sub(first) as usize;
            cells.get(at).map_or(0, |cell| cell.load(Ordering::Relaxed))
        })
    }

    /// One lease more books the worker of `slot`. The leases' lock orders
    /// the writers, so a write is a load and a store: no reader waits on
    /// the line and no writer locks it.
    pub(crate) fn book(&self, slot: Slot) {
        if let Some(cell) = self.cell(slot) {
            let held = cell.load(Ordering::Relaxed);
            cell.store(held.saturating_add(1), Ordering::Relaxed);
        }
    }

    /// One lease that booked the worker of `slot` left the file, under the
    /// same lock.
    pub(crate) fn free(&self, slot: Slot) {
        if let Some(cell) = self.cell(slot) {
            let held = cell.load(Ordering::Relaxed);
            cell.store(held.saturating_sub(1), Ordering::Relaxed);
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
        let (kept, stamp) = kept.open(slots.len());
        self.lent(slots, kept, stamp)
    }

    /// The table beside `slots` with the cells a decision of `stamp` keeps
    /// its reads in.
    fn lent<'a>(&'a self, slots: &'a [Slot], kept: &'a [AtomicU64], stamp: u32) -> LentCounts<'a> {
        let first = self.pieces.first().and_then(OnceLock::get);
        LentCounts {
            first: first.map_or(&[], |cells| cells),
            counts: self,
            slots,
            kept,
            stamp,
        }
    }

    fn cell(&self, slot: Slot) -> Option<&AtomicU32> {
        let (cells, first) = self.piece(slot);
        cells.get(slot.0.wrapping_sub(first) as usize)
    }

    /// The cells of the piece that holds `slot` and the piece's first
    /// slot: no cells where the piece is not laid yet or no piece holds
    /// the slot.
    fn piece(&self, slot: Slot) -> (&[AtomicU32], u32) {
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
    /// The table's first piece, read with no search: the cells of the
    /// first 1,024 members that ever stood.
    first: &'a [AtomicU32],
    counts: &'a Counts,
    /// Each row's slot, by the row's position.
    slots: &'a [Slot],
    /// What the decision read of each row, by the row's position, under
    /// its stamp ([`KeptCounts`]).
    kept: &'a [AtomicU64],
    stamp: u32,
}

impl LentCounts<'_> {
    /// The count of the row at `position`, one number for the decision:
    /// zero past the rows.
    #[inline]
    pub(crate) fn at(&self, position: u32) -> u32 {
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
    fn open(&mut self, rows: usize) -> (&[AtomicU64], u32) {
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
        (&self.cells, self.stamp)
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
    use std::sync::atomic::AtomicU64;

    use super::{Counts, KeptCounts, Slot};
    use crate::view::{Candidate, RequestFacts};

    /// A pool's counts taken out of its rows into a table of their own: the
    /// lent form of rows a case wrote its counts into.
    pub(crate) struct Lent {
        counts: Counts,
        slots: Vec<Slot>,
        /// The cells of the one decision a case's reads stand for.
        kept: Vec<AtomicU64>,
    }

    impl Lent {
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
            let kept = slots.iter().map(|_| AtomicU64::default()).collect();
            Self {
                counts,
                slots,
                kept,
            }
        }

        /// `facts` with the table lent beside the rows: every call lends
        /// the one decision a case's reads stand for.
        pub(crate) fn beside<'a>(&'a self, facts: &RequestFacts<'a>) -> RequestFacts<'a> {
            RequestFacts {
                lent_in_flight: Some(self.counts.lent(&self.slots, &self.kept, 1)),
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
}
