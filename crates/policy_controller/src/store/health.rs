//! The health sink: the prober's verdicts, the connect signal and the
//! gateways' standing masks (design 5.1, the health row and the precedence
//! rule).

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

/// Where health is written. The pool's view is the prober's and the reports';
/// a gateway's breaker and liveness vetoes are its own and mask the worker
/// for that gateway alone.
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
    /// the failure threshold fails. `Some` with the transition when the status
    /// moved; `None` when it stood, the row is gone or `revision` moved (a
    /// stale outcome is dropped, never applied to a successor).
    fn probe_outcome(
        &self,
        id: WorkerId,
        ok: bool,
        revision: u64,
    ) -> Option<(WorkerStatus, WorkerStatus)>;

    /// The connect signal of a handshake-promoted worker (a worker whose
    /// checks are off, a host-local wire): ready at once, under the same
    /// revision guard.
    fn connected(&self, id: WorkerId, revision: u64) -> Option<(WorkerStatus, WorkerStatus)>;

    /// A gateway's standing veto on a worker until `until`; `None` clears
    /// it. The breaker's open window and the liveness veto travel this way,
    /// and a veto standing across a reconnect is re-asserted in the gateway's
    /// `Hello`. A veto for a worker without a row (never registered, or
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
    /// of the gateway's vetoes on the worker: the note's check and the
    /// write under one hold of the store, so no row leaves or comes back
    /// between them. `stands` says the note holds an end that stands at
    /// `now` apart from this write. Where it does and the store shows no
    /// veto of the gateway on the worker at `now`, the worker's row left
    /// since the note was written, with its vetoes: nothing is written and
    /// the caller is told. A row registered again under the same id is
    /// never written an end of the row before it.
    fn gateway_veto_noted(
        &self,
        gateway: GatewayId,
        id: WorkerId,
        until: Option<Instant>,
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
