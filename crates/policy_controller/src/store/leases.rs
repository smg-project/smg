//! The leases store: every issued decision, per gateway, with the term of
//! the leader that issued it (design 4.3, 5.1 the leases row, 5.3 fencing).

use std::time::Instant;

use crate::{
    lease::{Lease, LeaseId, Outcome, ReleaseReason, Side},
    store::GatewayId,
    view::WorkerId,
};

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

/// One lease as the store holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaseRecord {
    pub lease: Lease,
    /// The gateway the lease was issued to; the pair (gateway, lease id) is
    /// the key.
    pub gateway: GatewayId,
    /// The request id's fingerprint, for `Explain`.
    pub request: u64,
    pub target: WorkerId,
    /// The decode side of a pair.
    pub decode: Option<WorkerId>,
    /// The term of the leader that issued the lease.
    pub term: u64,
    pub issued_at: Instant,
    /// Past it, an undispatched lease is released as abandoned.
    pub dispatch_deadline: Instant,
    pub state: LeaseState,
}

/// Why a lease was not recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseRefused {
    /// The record's term is above the store's: the issuer has not noticed it
    /// lost, or this replica has not noticed it won.
    TermAhead { lease: u64, store: u64 },
    /// The (gateway, lease id) pair is on file already.
    Duplicate,
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

    /// The gateway dispatched `side` of the lease at `at`; `false` when the
    /// lease is unknown.
    fn dispatched(&self, gateway: GatewayId, lease: LeaseId, side: Side, at: Instant) -> bool;

    /// A side completed; `false` when the lease is unknown.
    fn completed(&self, gateway: GatewayId, lease: LeaseId, outcome: &Outcome) -> bool;

    /// The gateway gave the lease up; `false` when the lease is unknown. A
    /// second release of one lease changes nothing.
    fn released(&self, gateway: GatewayId, lease: LeaseId, reason: ReleaseReason) -> bool;

    fn get(&self, gateway: GatewayId, lease: LeaseId) -> Option<LeaseRecord>;

    /// Release the issued leases whose dispatch deadline passed at `now` as
    /// abandoned; returns them.
    fn expire(&self, now: Instant) -> Vec<LeaseRecord>;

    /// Restore what a gateway's `Hello` pages list, releasing what they
    /// omit; a record above the store's term is skipped. Returns how many
    /// stand after it.
    fn restore(&self, gateway: GatewayId, records: &[LeaseRecord]) -> usize;

    /// The gateway is gone past its grace: every lease of it released as
    /// client gone. Returns how many.
    fn drop_gateway(&self, gateway: GatewayId) -> usize;

    /// The leases of a gateway that are issued or dispatched.
    fn live(&self, gateway: GatewayId) -> usize;
}
