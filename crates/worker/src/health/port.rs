//! The port between the health collector and the sink of what its facts mean
//! for a worker: the three calls of a sink and the answer of the first. The
//! collector (the probe schedule of [`manager`](super::manager)) says what
//! it read: one probe's outcome, a worker's connect signal, and that a
//! worker's liveness veto moved. The sink keeps the status, the consecutive
//! counts, the thresholds they are held to and the revision guard, and
//! answers what a probe or a connect signal did to the status.
//!
//! The registry is this crate's sink: its impl runs the status state machine
//! over the worker objects it holds, and
//! [`WorkerManager::start`](super::manager::WorkerManager::start) hands it.
//! A host states the port over its own store and hands it to
//! [`WorkerManager::start_with_sink`](super::manager::WorkerManager::start_with_sink).
//! The collector's schedule goes on reading the status on the worker object,
//! so such a sink writes the status there before it answers: the places are
//! on [`HealthSink`].

use crate::{registry::WorkerId, worker::WorkerStatus};

/// What a probe's outcome did to the worker's status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeAnswer {
    /// The status moved from `old` to `new`.
    Moved {
        old: WorkerStatus,
        new: WorkerStatus,
    },
    /// The outcome was judged and the status stands at this value.
    Stood(WorkerStatus),
    /// The outcome was judged against nothing: the worker was replaced since
    /// the probe was launched (its revision moved), or no worker stands
    /// under the id. It is counted nowhere and applied to no successor.
    Stale,
}

/// Where the health collector's facts go. Every call is cold: a probe, a
/// connect signal, a move of the liveness veto.
///
/// The collector calls a sink from its loop and judges no outcome itself:
/// what a fact did to the status is the sink's answer. Its schedule still
/// reads the status on the worker object the registry holds, in three
/// places: where it lays a worker into the schedule (a registration, a
/// replacement, a rebuild from the registry's snapshot), where it launches
/// a probe (the probe's slot, the `launched` it hands to
/// [`Self::probe_outcome`], the engine id's read after a recovery), and at
/// a connect signal (the guard on a worker under removal, whether the probe
/// decides). The contact hook that raises a connect signal reads it there
/// too. So a sink that keeps the status elsewhere writes it on the worker
/// object before it answers, through the registry's guarded transition
/// ([`transition_status_if_revision`][guarded] or
/// [`apply_if_revision`][applied]) and never through the worker's own
/// setter: only the registry's write takes the worker's mutation lock, runs
/// what a status change carries and publishes it. The registry's sink
/// writes nowhere else.
///
/// A sink that holds no row for the worker answers [`ProbeAnswer::Stale`]
/// to a probe's outcome and `None` to a connect signal, and writes nothing
/// at a move of the liveness veto.
///
/// [guarded]: crate::registry::WorkerRegistry::transition_status_if_revision
/// [applied]: crate::registry::WorkerRegistry::apply_if_revision
pub trait HealthSink: Send + Sync {
    /// One probe of the worker passed or failed. `revision` is the worker's
    /// revision at the probe's launch: the outcome of a worker replaced
    /// since is [`ProbeAnswer::Stale`]. `launched` is the status the probe
    /// was launched at, for a sink that counts toward the cap on a pending
    /// worker's probes only the probes launched while it was pending.
    fn probe_outcome(
        &self,
        worker_id: &WorkerId,
        passed: bool,
        revision: u64,
        launched: WorkerStatus,
    ) -> ProbeAnswer;

    /// The connect signal of the worker at `revision`, where the signal
    /// itself brings the worker into service: the collector asks first
    /// whether the worker's probe decides instead, and then says nothing
    /// here. Answers the transition as `(old, new)`, or `None` for the
    /// three things that move nothing: the status stood (the worker serves
    /// already), the revision moved (the signal is of a worker replaced
    /// since), or no worker stands under the id.
    fn connected(
        &self,
        worker_id: &WorkerId,
        revision: u64,
    ) -> Option<(WorkerStatus, WorkerStatus)>;

    /// The liveness veto of the worker under `worker_id` may have moved
    /// ([`Worker::stall_reason`](crate::identity::basic::Worker::stall_reason)):
    /// it was set, changed its reason or was cleared, or the worker came
    /// under the collector with a veto no sink was told of. The collector
    /// says it in three places. At a signal of the worker
    /// ([`WorkerSignal::LivenessMoved`](crate::registry::event::WorkerSignal::LivenessMoved)).
    /// Where a worker is laid in another's place, whose running state it
    /// takes. And for every worker at the loop's start and behind a lagged
    /// subscription. `revision` is the worker's at the tell.
    ///
    /// The call carries no reading. A sink that keeps the veto elsewhere
    /// reads it on the worker object at its own write, so of two tells the
    /// later read stands. The default does nothing: the registry's sink
    /// keeps the veto on the worker object, where routing reads it.
    fn liveness_moved(&self, _worker_id: &WorkerId, _revision: u64) {}
}
