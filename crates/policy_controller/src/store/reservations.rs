//! The reservation store: prefill slots and decode rooms as leases, exclusive
//! under one fenced leader (design 4.3 and 5.1, the PD row).

use crate::{lease::LeaseId, store::GatewayId, view::WorkerId};

/// What a reservation holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReservationKind {
    /// A seat in a prefill worker's capacity window, held until its prefill
    /// phase ends.
    Slot,
    /// A room in a decode worker's running window, held until the decode
    /// side completes.
    Room,
}

/// Where reservations are claimed and released.
///
/// A claim is all or nothing within the worker's window; a lease holds at
/// most one reservation per kind, and a release is idempotent (a completion
/// and a late release of one lease free it once). The window is the engine's
/// own admission bound or the configured cap; a worker that reports none has
/// no window and the gate abstains. A worker the store does not know (never
/// registered, or removed) is refused: a missing row is not a live worker
/// without a bound. A claim is keyed by the gateway that drew the lease and
/// the lease itself: two gateways of one controller may draw the same lease
/// number, and neither sees or frees the other's claim.
pub trait ReservationStore: Send + Sync {
    /// The worker's window for `kind`; `None` when it has none.
    fn window(&self, id: WorkerId, kind: ReservationKind) -> Option<u32>;

    /// Claim `count` of `kind` on the worker under the gateway's `lease`:
    /// `true` and held when the claimed total stays within the window,
    /// `false` and nothing claimed otherwise (also when that lease already
    /// holds that kind).
    fn try_reserve(
        &self,
        gateway: GatewayId,
        id: WorkerId,
        kind: ReservationKind,
        count: u32,
        lease: LeaseId,
    ) -> bool;

    /// Release what the gateway's lease holds, of both kinds.
    fn release(&self, gateway: GatewayId, lease: LeaseId);

    /// What is held on the worker for `kind`, over every lease.
    fn held(&self, id: WorkerId, kind: ReservationKind) -> u32;
}
