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
    view::{Session, WorkerId},
};

/// The time a gateway has to name a lease's dispatch: the design's default.
/// A lease still issued past it is released as abandoned.
pub(crate) const DISPATCH_TTL: std::time::Duration = std::time::Duration::from_secs(5);

/// Where a lease stands, or one side of a pair's lease: each side moves by
/// its own frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseState {
    /// Decided, not yet dispatched; the dispatch deadline bounds this.
    Issued,
    /// Dispatched to its target at `at`.
    Dispatched {
        at: Instant,
    },
    Completed,
    Released(ReleaseReason),
}

impl LeaseState {
    /// Whether a frame still moves a lease that stands so: issued or
    /// dispatched.
    pub const fn is_live(self) -> bool {
        matches!(self, Self::Issued | Self::Dispatched { .. })
    }
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

    /// What a pair's record keeps of the pick `targets` for its decode
    /// side: the decode of each target behind the first and every target's
    /// decode rank. A target that names no pair keeps none.
    pub fn of_decodes(targets: &Targets) -> Self {
        let mut picked = Self::default();
        let decodes = || targets.iter().map(|target| target.pair.unwrap_or_default());
        for (decode, rank) in decodes().zip(picked.ranks.iter_mut()) {
            *rank = decode.decode_dp_rank;
        }
        for (decode, id) in decodes().skip(1).zip(picked.rest.iter_mut()) {
            *id = decode.decode_id;
        }
        picked
    }

    /// The worker and the rank at `index` of the pick, for the side that
    /// books `booked` at the index `own`; `None` for an index the pick
    /// holds no target at.
    fn at(&self, booked: WorkerId, own: u8, index: u8) -> Option<(WorkerId, u16)> {
        let place = usize::from(index);
        let rank = *self.ranks.get(place)?;
        if index == own {
            return Some((booked, rank));
        }
        // A side books the pick's first target at its issue: the others
        // stand behind it.
        let id = *self.rest.get(place.checked_sub(1)?)?;
        (id != WorkerId(0)).then_some((id, rank))
    }
}

/// The decode side of a pair's lease as its record keeps it. The two sides
/// share the lease's id, its generation and its issue; each has its
/// worker, its booking and its state, and lives by its own frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeSide {
    /// The worker the side books: the chosen pair's decode, until the
    /// side's `Dispatched` names another target of the pick.
    pub target: WorkerId,
    /// The decodes of the pick's other targets and every target's decode
    /// rank, for that `Dispatched`.
    pub picked: Picked,
    /// The request's session under the decode phase.
    pub session: Session,
    /// Where the side stands.
    pub state: LeaseState,
    /// The token-work the side books on its worker.
    pub tokens: u32,
    /// Which target of the pick the side books.
    pub target_index: u8,
}

impl DecodeSide {
    /// The side's lease: `prefill`, the record's own, with what is the
    /// decode side's. The worker it books ([`Lease::booked`]) is the
    /// side's own, never the prefill's: a stage that books at the issue
    /// reads it at the side's dispatch, which is told of the side as its
    /// issue left it. Behind a dispatch that moved the side, the lease
    /// names the worker the side moved to.
    pub fn lease(&self, prefill: &Lease) -> Lease {
        Lease {
            side: Side::Decode,
            target_index: self.target_index,
            booked: self.target,
            session: self.session,
            tokens: self.tokens,
            ..*prefill
        }
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
    /// The decode side of a pair's lease: one record under one id with a
    /// part a side, where the prefill side is the record's own lease,
    /// target and state. `None` for a single's lease.
    pub decode: Option<DecodeSide>,
    /// The TTL of the sessions rule the lease's decision was made under,
    /// copied from its decision's scratch (`None`: a rule with no TTL, or
    /// no rule): a commit whose engine's table holds no such key ends the
    /// key's pin of record by it, no earlier than that pin's end stood.
    pub session_ttl_micros: Option<u64>,
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
        self.picked.at(self.target, self.lease.target_index, index)
    }

    /// The worker and the rank a `Dispatched` on `side` names by its
    /// target `index`: on a pair's decode side the decode of that target,
    /// else [`LeaseRecord::target_at`].
    pub fn target_of(&self, side: Side, index: u8) -> Option<(WorkerId, u16)> {
        match self.decode {
            Some(decode) if side == Side::Decode => {
                decode.picked.at(decode.target, decode.target_index, index)
            }
            _ => self.target_at(index),
        }
    }

    /// The lease `side` names on this record and where that side stands.
    /// A pair's record holds a prefill side, the record's own lease and
    /// state, and a decode side; `None` for another side. A single's
    /// record reads no side: it holds its one lease under any.
    pub fn side(&self, side: Side) -> Option<(Lease, LeaseState)> {
        let Some(decode) = self.decode else {
            return Some((self.lease, self.state));
        };
        match side {
            Side::Prefill => Some((self.lease, self.state)),
            Side::Decode => Some((decode.lease(&self.lease), decode.state)),
            Side::Single | Side::Encode => None,
        }
    }

    /// The lease of each side a frame on `side` ends on this record: the
    /// side named where it is issued or dispatched, and with `None` every
    /// such side. A single's record ends whole under any side.
    pub fn ending(&self, side: Option<Side>) -> [Option<Lease>; 2] {
        let live = |side| {
            self.side(side)
                .and_then(|(lease, state)| state.is_live().then_some(lease))
        };
        match (self.decode, side) {
            (None, _) => [Some(self.lease), None],
            (Some(_), Some(side)) => [live(side), None],
            (Some(_), None) => [live(Side::Prefill), live(Side::Decode)],
        }
    }

    /// Whether a side of the lease is issued or dispatched.
    pub fn is_live(&self) -> bool {
        self.state.is_live() || self.decode.is_some_and(|decode| decode.state.is_live())
    }

    /// Whether a side of the lease waits for its dispatch.
    pub fn waits(&self) -> bool {
        self.state == LeaseState::Issued
            || self
                .decode
                .is_some_and(|decode| decode.state == LeaseState::Issued)
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
    /// The file of the record's id is not settled yet: the id is of the
    /// store's term from a block of the counter that no file claimed, or
    /// of a term the store can still take. No decider minted it, and the
    /// block's claim would name its file anew with the lease on file. The
    /// same id is taken once a claim settles its file. A store of one
    /// file refuses none by it: there every id names that file.
    Unclaimed,
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
    /// and the count the lease books. On a pair's lease the dispatch moves
    /// the side it names and no other: a decode's on another target moves
    /// the decode's booking alone, and a side the lease does not hold
    /// changes nothing.
    fn dispatched(
        &self,
        gateway: GatewayId,
        lease: LeaseId,
        dispatch: Dispatch,
        at: Instant,
    ) -> bool;

    /// A side completed; `false` when the lease is unknown. A pair's side
    /// ends alone and gives its count back, and the lease leaves the file
    /// at its last side's end.
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
    /// abandoned; returns them. A pair's side that waits ends alone, and
    /// its record is returned as the call left it. A store that keeps a
    /// file of leases a decider ends them where its caller may act: it
    /// passes a file whose decider is at work, which ends its own at its
    /// issues, so what is returned can be a part of the leases that are
    /// due. Such a store tells its engine of each end itself: a caller
    /// tells none again.
    fn expire(&self, now: Instant) -> Vec<LeaseRecord>;

    /// Restore what a gateway's `Hello` pages list, releasing what they
    /// omit; a record above the store's term is skipped, and so is one
    /// whose id names no file yet ([`LeaseRefused::Unclaimed`]).
    /// Returns how many stand after it.
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
