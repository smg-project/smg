//! The replica's worker objects under the controller: the wake of their
//! follower, and under the `worker-host` feature their holder over the
//! controller's store with the one task that follows it.

#[cfg(feature = "worker-host")]
use std::sync::Arc;

#[cfg(feature = "worker-host")]
use tokio::sync::Notify;

#[cfg(feature = "worker-host")]
mod follower;

#[cfg(feature = "worker-host")]
pub(crate) use follower::WorkerObjects;

/// The wake of the follower of the replica's worker objects: told once a
/// watched write of the store's log is released, and once the log
/// restarts. It is made before the log and handed to the place that builds
/// the log's one listener
/// ([`journaled_at`](super::replication::journaled_at)); its copy is the
/// follower's.
///
/// A build without `worker-host` has no follower: the wake is nothing
/// there, and a write pays nothing for it.
#[derive(Clone, Debug, Default)]
pub(crate) struct ObjectsWake {
    /// Kept once while the follower works or has not started.
    #[cfg(feature = "worker-host")]
    moved: Arc<Notify>,
}

impl ObjectsWake {
    /// The store's watched kinds moved. Called by the log's listener on the
    /// writer's thread: a wake and no work, under no lock.
    #[cfg_attr(
        not(feature = "worker-host"),
        expect(
            clippy::unused_self,
            reason = "one call for the log's listener in every build; with no follower it wakes nothing"
        )
    )]
    pub(crate) fn tell(&self) {
        #[cfg(feature = "worker-host")]
        self.moved.notify_one();
    }
}
