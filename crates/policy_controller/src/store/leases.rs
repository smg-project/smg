//! The leases store: every issued decision, per gateway, with the term of
//! the leader that issued it (design 4.3, 5.1 the leases row, 5.3 fencing).

use std::time::Instant;

/// The table of counts the leases on file book and a member's place in
/// it, named here for the store's files that write it: their home is the
/// counts' own module.
pub(super) use crate::counts::{Counts, Slot};
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
    /// target at, and for the pick's first target once a dispatch moved
    /// the lease: the record keeps the target it books in that one's
    /// place. No caller asks then, since a lease moves once.
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

/// What is said of a lease, as a session hands a frame's word and a
/// `Decide` stream a release to the leases' file
/// ([`MemoryStore::hand`](super::MemoryStore::hand)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Said {
    /// The dispatch the frame names, and the instant the frame was read.
    Dispatched(Dispatch, Instant),
    /// The outcome of the side that ended.
    Completed(Outcome),
    /// The side given up, where the frame names one, and the reason.
    Released(Option<Side>, ReleaseReason),
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
    /// index, and where that target is another worker the record's target
    /// and the count the lease books.
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
