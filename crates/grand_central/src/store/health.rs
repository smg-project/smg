//! The health sink: the gateways' standing masks (design 10.2, the health
//! row). A worker's status is its worker object's word: the holder of the
//! objects tells the store each one (`MemoryStore::status_told`).

#![expect(
    clippy::exhaustive_enums,
    reason = "the store's copy of this worker data leaves the crate, and its types with it"
)]

use std::time::Instant;

use crate::{
    engine::view::WorkerId,
    store::{workers::WorkerStatus, GatewayId},
};

/// What a noted write of a gateway's veto found
/// ([`HealthSink::gateway_veto_noted`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VetoWrite {
    /// The end is written, or the veto cleared.
    Written,
    /// The note says an end stands and the store shows none: the worker's
    /// row left since the note was written and took its vetoes with it.
    /// Nothing is written.
    Left,
    /// A veto for a worker without a row: nothing is written.
    NoRow,
}

/// What a told status found (`MemoryStore::status_told`): what happened to
/// the write, and the status that stands where the write was judged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HealthWrite {
    /// The status moved: `from` is the one the row left, `to` the one that
    /// stands.
    Moved {
        from: WorkerStatus,
        to: WorkerStatus,
    },
    /// The write was judged and the status stands.
    Stood(WorkerStatus),
    /// The row left the revision the write was read under: the write is
    /// dropped, never applied to a successor. It was judged against
    /// nothing, so it says no status; the row's revision is the membership
    /// door's to say.
    Stale,
    /// A write for a worker without a row: nothing is written.
    NoRow,
}

impl HealthWrite {
    /// The transition, where the status moved: the status the row left,
    /// then the one that stands.
    pub fn moved(self) -> Option<(WorkerStatus, WorkerStatus)> {
        match self {
            Self::Moved { from, to } => Some((from, to)),
            Self::Stood(_) | Self::Stale | Self::NoRow => None,
        }
    }

    /// The status that stands where the write was judged: none for a stale
    /// write and for a worker without a row.
    pub fn stands(self) -> Option<WorkerStatus> {
        match self {
            Self::Moved { to: status, .. } | Self::Stood(status) => Some(status),
            Self::Stale | Self::NoRow => None,
        }
    }
}

/// A status writer's word of its lead (`MemoryStore::status_told`): its own
/// read of whether it leads still, and the term that read speaks of where
/// the writer names one.
#[cfg(feature = "worker-host")]
#[derive(Clone, Copy)]
pub(crate) struct LeadWord<'a> {
    /// Whether the writer leads still, by its own read. The holder of the
    /// worker objects asks it under an object's lock, and the store asks
    /// it behind its membership lock.
    pub(crate) leads: &'a dyn Fn() -> bool,
    /// The term the write is made under. The journal compares it in the
    /// take of its lock that gives the event its place. A writer with no
    /// term names none, and its write is journaled as the store's own.
    pub(crate) term: Option<u64>,
}

#[cfg(feature = "worker-host")]
impl<'a> LeadWord<'a> {
    /// The word of a writer that names no term: `leads`, its own read.
    pub(crate) fn of(leads: &'a dyn Fn() -> bool) -> Self {
        Self { leads, term: None }
    }
}

#[cfg(feature = "worker-host")]
impl LeadWord<'static> {
    /// The word of a writer with no term to end: it always holds.
    pub(crate) const FREE: Self = Self {
        leads: &always,
        term: None,
    };
}

#[cfg(feature = "worker-host")]
fn always() -> bool {
    true
}

/// Where a gateway's own health words are written. The pool's view is the
/// prober's, on the worker objects; a gateway's breaker and liveness
/// vetoes are its own and mask the worker for that gateway alone.
pub trait HealthSink: Send + Sync {
    /// A gateway's standing veto on a worker until `until`; `None` clears
    /// it. A veto standing across a reconnect is re-asserted in the
    /// gateway's `Hello`. The store keeps no kind of a veto. It keeps two
    /// ends: the veto's, and the hold's where a part of the veto is one no
    /// rule may lift, as a breaker's open window is
    /// ([`gateway_veto_noted`](Self::gateway_veto_noted) writes both). A
    /// veto written here tells no kind, so the whole of it holds: a
    /// decision reads it as an open breaker's to its end, and no rule
    /// lifts it. A caller that means a doubt of the worker's liveness, the
    /// veto the all-vetoed rule lifts as its last resort, says so through
    /// the noted door, with no hold. A veto for a worker without a row
    /// (never registered, or
    /// removed) writes nothing; the clear needs no such guard. A gateway
    /// gone past its grace period takes its vetoes with it
    /// ([`LeasesStore::drop_gateways`](super::LeasesStore::drop_gateways)).
    /// The controller's presence table alone writes a gateway's vetoes,
    /// through the note it keeps of them. A veto set around the table never
    /// counts against the gateway's ceiling of vetoes. One lifted around it
    /// can still count until its own instant, the gateway's next frame for
    /// that worker, its next list, or its first frame at the ceiling after
    /// a membership write.
    fn gateway_veto(&self, gateway: GatewayId, id: WorkerId, until: Option<Instant>);

    /// [`gateway_veto`](Self::gateway_veto) for a caller that keeps a note
    /// of the gateway's vetoes on the worker, with the veto's hold. The
    /// note's check and the write run under the store's locks taken once,
    /// so no row leaves or comes back between them. `hold` is the end of
    /// the part of the veto no rule may lift, and `None` where the whole
    /// veto is a doubt of the worker's liveness. The store keeps it beside
    /// `until` and never past it: while it stands a decision of the
    /// gateway reads the worker as behind an open breaker, and from then
    /// to the veto's own end as doubted. The caller alone knows a veto's
    /// kinds and says which of them hold. `stands` says the note keeps an
    /// end that stands at `now` apart from this write. Where it does and
    /// the store shows no veto of the gateway on the worker at `now`, the
    /// worker's row left since the note was written, with its vetoes:
    /// nothing is written and the caller is told. A row registered again
    /// under the same id is never written an end of the row before it.
    fn gateway_veto_noted(
        &self,
        gateway: GatewayId,
        id: WorkerId,
        until: Option<Instant>,
        hold: Option<Instant>,
        stands: bool,
        now: Instant,
    ) -> VetoWrite;

    /// Whether `gateway` vetoes the worker at `now`.
    fn vetoed(&self, gateway: GatewayId, id: WorkerId, now: Instant) -> bool;
}
