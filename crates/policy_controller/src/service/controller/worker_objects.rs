//! The replica's worker objects under the controller: the wake of their
//! follower, their holder over the controller's store with the one task
//! that follows it, the lead of a replica that elects, and the health
//! probes and the load monitor it runs while it leads.

use std::sync::Arc;

use tokio::sync::Notify;

mod follower;
mod lead;
mod load_monitor;
mod prober;

pub(crate) use follower::WorkerObjects;
pub(crate) use lead::ElectedLead;
pub(crate) use load_monitor::LoadMonitor;
pub(crate) use prober::Prober;

/// The wake of the follower of the replica's worker objects: told once a
/// watched write of the store's log is released, and once the log
/// restarts. It is made before the log and handed to the place that builds
/// the log's one listener
/// ([`journaled_at`](super::replication::journaled_at)); its copy is the
/// follower's.
#[derive(Clone, Debug, Default)]
pub(crate) struct ObjectsWake {
    /// Kept once while the follower works or has not started.
    moved: Arc<Notify>,
}

impl ObjectsWake {
    /// The store's watched kinds moved. Called by the log's listener on the
    /// writer's thread: a wake and no work, under no lock.
    pub(crate) fn tell(&self) {
        self.moved.notify_one();
    }
}
