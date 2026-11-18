//! The leases store: every issued decision, per gateway, with the term of
//! the leader that issued it (design 4.3, 5.1 the leases row, 5.3 fencing).

use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        OnceLock,
    },
    time::Instant,
};

use crate::{
    decision::{Targets, MAX_TARGETS},
    lease::{Lease, LeaseId, Outcome, ReleaseReason, Side},
    store::GatewayId,
    view::WorkerId,
};

/// The time a gateway has to name a lease's dispatch: the design's default.
/// A lease still issued past it is released as abandoned.
#[cfg(feature = "controller")]
pub(crate) const DISPATCH_TTL: std::time::Duration = std::time::Duration::from_secs(5);

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

/// What the leases on file book on each worker: the counts a decision's
/// rows read as their requests in flight ([`Candidate::in_flight`]).
///
/// One table for the store's life, a cell a member that ever stood, grown
/// in pieces that never move. A membership write gives a joining member
/// the next cell and does nothing else for a count, a pool's shape names
/// each row's cell by its slot, and a lease on file keeps the slot it
/// booked. So a count outlives a membership write, a lease of a worker
/// that left ends on the count it booked, and a decision reads a count
/// with no lock.
///
/// [`Candidate::in_flight`]: crate::view::Candidate::in_flight
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

/// Where a lease stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseState {
    /// Decided, not yet dispatched; the dispatch deadline bounds this.
    Issued,
    /// Dispatched to its target at `at`; a pair's second side follows.
    Dispatched {
        at: Instant,
    },
    Completed,
    Released(ReleaseReason),
}

/// What a lease's record keeps of its pick beside the target it books: the
/// pick's other targets and every target's rank, so a `Dispatched` that
/// names a target by its index finds the worker. Kept inline: a record is
/// filed inside a decision's clock. The default is a pick of one target at
/// rank 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Picked {
    /// The pick's targets behind its first, in the pick's order;
    /// `WorkerId(0)` is no target.
    rest: [WorkerId; MAX_TARGETS - 1],
    /// Each target's rank, by its index in the pick.
    ranks: [u16; MAX_TARGETS],
}

impl Picked {
    /// What a record keeps of the pick `targets`.
    pub fn of(targets: &Targets) -> Self {
        let mut picked = Self::default();
        for (target, rank) in targets.iter().zip(picked.ranks.iter_mut()) {
            *rank = target.dp_rank;
        }
        for (target, id) in targets.iter().skip(1).zip(picked.rest.iter_mut()) {
            *id = target.id;
        }
        picked
    }
}

/// One lease as the store holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaseRecord {
    pub lease: Lease,
    /// The gateway the lease was issued to; the pair (gateway, lease id) is
    /// the key.
    pub gateway: GatewayId,
    /// The request id's fingerprint, for `Explain`.
    pub request: u64,
    /// The worker the lease books: the pick's first target, until a
    /// `Dispatched` names another one ([`LeaseRecord::target_at`]).
    pub target: WorkerId,
    /// The pick's other targets and the ranks, for that `Dispatched`.
    pub picked: Picked,
    /// The decode side of a pair.
    pub decode: Option<WorkerId>,
    /// The term of the leader that issued the lease.
    pub term: u64,
    pub issued_at: Instant,
    /// Past it, an undispatched lease is released as abandoned.
    pub dispatch_deadline: Instant,
    pub state: LeaseState,
}

impl LeaseRecord {
    /// The worker and the rank a `Dispatched` names by its target `index`:
    /// the target the lease books at the lease's own index, else the
    /// pick's target at that index. `None` for an index the pick holds no
    /// target at.
    pub fn target_at(&self, index: u8) -> Option<(WorkerId, u16)> {
        let place = usize::from(index);
        let rank = *self.picked.ranks.get(place)?;
        if index == self.lease.target_index {
            return Some((self.target, rank));
        }
        // A lease books the pick's first target at its issue: the others
        // stand behind it.
        let id = *self.picked.rest.get(place.checked_sub(1)?)?;
        (id != WorkerId(0)).then_some((id, rank))
    }
}

/// What a gateway's `Dispatched` report says beside its lease.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dispatch {
    /// Which target of the pick the gateway dispatched to.
    pub target_index: u8,
    pub side: Side,
    /// Decision to dispatch on the gateway's clock, in microseconds.
    pub since_decision_micros: u32,
}

/// Why a lease was not recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseRefused {
    /// The record's term is above the store's: the issuer has not noticed it
    /// lost, or this replica has not noticed it won.
    TermAhead { lease: u64, store: u64 },
    /// The (gateway, lease id) pair is on file already.
    Duplicate,
    /// The store mints no id under the record's term: the term is not the
    /// store's own any more, is past the width an id carries it in, or its
    /// counter is spent (the contract's bounds: a term below 2^31 over a
    /// counter of 1 to 2^32 - 1). Equal terms say the term itself mints no
    /// more.
    Unminted { lease: u64, store: u64 },
}

/// Where leases are recorded.
///
/// A lease carries the term of the leader that issued it; a record whose term
/// is above the store's own is refused, one below is kept (a lease the old
/// leader issued keeps its lifetime on the new one, learned from the gateway's
/// `Hello` pages, which restore exactly what they list and release what they
/// omit). A gateway gone past its grace period releases every lease of its
/// own. Reservations and pins are replicated; credits are rebuilt from
/// `Hello`.
pub trait LeasesStore: Send + Sync {
    /// The store's term: the leader's, from the elector; `0` before any.
    fn term(&self) -> u64;

    /// The elector's new term; never below the current one.
    fn set_term(&self, term: u64);

    fn issue(&self, record: LeaseRecord) -> Result<(), LeaseRefused>;

    /// The gateway dispatched the lease as `dispatch` says, at `at`;
    /// `false` when the lease is unknown. A first dispatch that names
    /// another target of the pick moves the lease to it: the record's
    /// target and index, and the count the lease books.
    fn dispatched(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        dispatch: Dispatch,
        at: Instant,
    ) -> bool;

    /// A side completed; `false` when the lease is unknown.
    fn completed(&self, gateway: GatewayId, lease: LeaseId, outcome: &Outcome) -> bool;

    /// The gateway gave the lease up, on `side` or, with `None`, on every
    /// side (a pair's leases share an id); `false` when the lease is
    /// unknown. A second release of one lease changes nothing.
    fn released(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        side: Option<Side>,
        reason: ReleaseReason,
    ) -> bool;

    fn get(&self, gateway: GatewayId, lease: LeaseId) -> Option<LeaseRecord>;

    /// Release the issued leases whose dispatch deadline passed at `now` as
    /// abandoned; returns them.
    fn expire(&self, now: Instant) -> Vec<LeaseRecord>;

    /// Restore what a gateway's `Hello` pages list, releasing what they
    /// omit; a record above the store's term is skipped. Returns how many
    /// stand after it.
    fn restore(&self, gateway: GatewayId, records: &[LeaseRecord]) -> usize;

    /// The gateways are gone past their grace: every lease of each released
    /// as client gone, and with them what the store holds of each gateway's
    /// own about a worker (its vetoes, its completed dispatches). One write
    /// a table for them all: one pass over the leases, one over the vetoes
    /// and one publication. Returns how many leases each held, in the order
    /// of `gone`. It is the store's one door for a gone gateway.
    fn drop_gateways(&self, gone: &[GatewayId]) -> Vec<usize>;

    /// The leases of a gateway that are issued or dispatched.
    fn live(&self, gateway: GatewayId) -> usize;
}
