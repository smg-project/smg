//! Events emitted by [`WorkerRegistry`](crate::worker::WorkerRegistry) on state mutations.

use std::sync::Arc;

use openai_protocol::worker::WorkerStatus;

use super::{basic::Worker, registry::WorkerId};

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
