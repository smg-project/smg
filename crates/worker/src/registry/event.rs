//! Events emitted by [`WorkerRegistry`](crate::registry::WorkerRegistry) on state mutations.

use std::sync::Arc;

use super::WorkerId;
use crate::{identity::basic::Worker, worker::WorkerStatus};

/// Events broadcast when worker state changes.
///
/// Subscribers (WorkerManager, WorkerMonitor, WorkerSyncAdapter) use these for
/// incremental updates. Events carry `Arc<dyn Worker>` so subscribers can access
/// any worker data they need without re-querying the registry.
///
/// For `Removed`, the worker Arc is a pre-removal snapshot — the worker is
/// already gone from the registry when the event fires.
#[derive(Debug, Clone)]
pub enum WorkerEvent {
    /// A worker was added to the registry.
    Registered {
        worker_id: WorkerId,
        worker: Arc<dyn Worker>,
    },

    /// A worker was removed from the registry.
    /// The worker Arc is a pre-removal snapshot.
    Removed {
        worker_id: WorkerId,
        worker: Arc<dyn Worker>,
    },

    /// A worker was replaced (same URL, new worker object — e.g. property update).
    Replaced {
        worker_id: WorkerId,
        old: Arc<dyn Worker>,
        new: Arc<dyn Worker>,
    },

    /// A worker's lifecycle status changed (Pending→Ready, Ready→NotReady, etc.)
    StatusChanged {
        worker_id: WorkerId,
        worker: Arc<dyn Worker>,
        old_status: WorkerStatus,
        new_status: WorkerStatus,
    },
}

/// One-shot signal a worker fires the instant its backend connection
/// completes, so the manager can promote it to `Ready` immediately instead
/// of waiting for the next health poll.
///
/// Unlike [`WorkerEvent`] this is a point-to-point signal to the manager, not
/// a broadcast — the connect completes inside a detached handshake task that
/// has no registry handle, so it carries the worker's URL and the revision it
/// captured when the handshake began. The manager resolves the URL to a
/// worker id and applies the promotion only if that revision still matches,
/// discarding the signal if a same-URL replacement raced ahead.
#[derive(Debug, Clone)]
pub struct WorkerConnected {
    pub url: String,
    pub revision: u64,
}

/// What a worker object says to the manager's loop from a place that holds
/// no registry handle. Both ride one unbounded channel, whose sender the
/// registry hands each worker built for it
/// ([`WorkerRegistry::connect_signal_sender`][sender]).
///
/// [sender]: crate::registry::WorkerRegistry::connect_signal_sender
#[derive(Debug, Clone)]
pub enum WorkerSignal {
    /// The worker's backend connection completed, or a contact came from a
    /// worker demoted while its transport was gone.
    Connected(WorkerConnected),
    /// The liveness veto of the worker at `url` moved: it was set, changed
    /// its reason, or was cleared. The signal carries no reading and no
    /// revision. The loop resolves the URL to the worker that stands there
    /// now and tells its health sink, which reads the veto on the object
    /// ([`HealthSink::liveness_moved`][told]). At most one waits per
    /// worker, whoever reads the channel: the worker marks its running
    /// state at the send, and signals again only once the loop took the
    /// mark.
    ///
    /// [told]: crate::health::port::HealthSink::liveness_moved
    LivenessMoved { url: String },
}

impl From<WorkerConnected> for WorkerSignal {
    fn from(connected: WorkerConnected) -> Self {
        Self::Connected(connected)
    }
}
