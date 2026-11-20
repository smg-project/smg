//! The health sink: the prober's verdicts, the connect signal, the leader's
//! own doubt of a worker and the gateways' standing masks (design 5.1, the
//! health row and the precedence rule).

use std::time::Instant;

use crate::{
    lease::TransportClass,
    store::{workers::WorkerStatus, GatewayId},
    view::WorkerId,
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

/// What a health write found ([`HealthSink::probe_outcome`],
/// [`HealthSink::connected`]): what happened to the write, and the status
/// that stands where the write was judged.
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

/// What the leader's own worker object knows of a worker beside its status
/// ([`HealthSink::leader_bits`]): one projection of that object onto the
/// row's bits, so a later bit of the object rides the same write.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LeaderBits {
    /// The object's liveness veto stands: by the leader's own contact with
    /// the worker it is unreachable or wedged.
    pub stalled: bool,
}

/// Where health is written. The pool's view is the prober's and the reports';
/// a gateway's breaker and liveness vetoes are its own and mask the worker
/// for that gateway alone. The leader's own doubt of a worker masks it for
/// every gateway.
///
/// The precedence rule a view builder applies: a worker the prober cannot
/// reach stays routable for the gateways whose completed dispatches on it
/// succeeded within the last probe interval and is excluded for the others;
/// a worker every gateway vetoes is unhealthy whatever the prober says.
pub trait HealthSink: Send + Sync {
    /// The prober's verdict for one probe of the worker, under the row's
    /// [`HealthPolicy`](crate::store::HealthPolicy): the status machine moves
    /// after `failure_threshold` consecutive failures (ready to not ready),
    /// three times as many (to failed), `success_threshold` consecutive
    /// passes (back to ready), and a pending worker that flaps past ten times
    /// the failure threshold fails. The answer says what happened to the
    /// write: the status moved; it stood, and the probe is counted;
    /// `revision` moved (a stale outcome is dropped, never applied to a
    /// successor, and not counted); or the worker has no row.
    fn probe_outcome(&self, id: WorkerId, ok: bool, revision: u64) -> HealthWrite;

    /// The connect signal of a handshake-promoted worker (a worker whose
    /// checks are off, a host-local wire): ready at once, under the same
    /// revision guard. A ready or a draining row stands, and its counters
    /// with it.
    fn connected(&self, id: WorkerId, revision: u64) -> HealthWrite;

    /// What the leader's own worker object knows of the worker, written
    /// whole. Its liveness veto is the row's stalled bit, for every
    /// gateway and with no hold: a stage reads it as it reads a gateway's
    /// doubt of the worker's liveness
    /// ([`gateway_veto_noted`](Self::gateway_veto_noted)), and the
    /// all-vetoed rule lifts it as its last resort. A gateway's own vetoes
    /// stand beside it. The write moves no status and publishes at once
    /// where a bit moved. It is judged under the revision guard of a
    /// probe's outcome: the answer is the status that stands
    /// ([`HealthWrite::Stood`]), a revision the row left, or no row.
    ///
    /// The bits are the leader's own and no journal carries them: a
    /// standby holds none, and a new leader writes what its own objects
    /// know. They stand with the row across its revisions and leave with
    /// it: a key's new incarnation starts with none.
    fn leader_bits(&self, id: WorkerId, revision: u64, bits: LeaderBits) -> HealthWrite;

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

    /// A gateway's completed dispatch on the worker with its transport
    /// class: the precedence rule's second input beside the prober. A
    /// completion for a worker without a row (never registered, or removed)
    /// writes nothing.
    fn completed(&self, gateway: GatewayId, id: WorkerId, class: TransportClass, at: Instant);

    /// Whether `gateway` completed a dispatch of class `Ok` on the worker at
    /// or after `since`: the precedence rule reads one gateway's own
    /// dispatches, never another's.
    fn completed_ok_since(&self, gateway: GatewayId, id: WorkerId, since: Instant) -> bool;
}
