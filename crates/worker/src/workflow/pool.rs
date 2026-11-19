//! The pool-size gauge's groups, for the steps that change a pool.

use std::collections::HashSet;

use crate::{
    identity::basic::{ConnectionModeExt, Worker, WorkerTypeExt},
    metrics::writers::Metrics,
    registry::{Generation, WorkerRegistry},
    worker::{ConnectionMode, WorkerType},
};

/// The key the pool-size gauge is kept by: a worker's type, connection mode
/// and primary model id.
pub type PoolGroup = (WorkerType, ConnectionMode, String);

/// A set of unique pool groups.
pub type PoolGroups = HashSet<PoolGroup>;

/// The pool-size gauge's key for one worker. Registration and removal key the
/// gauge through this, so the two cannot drift apart.
pub fn pool_group(worker: &dyn Worker) -> PoolGroup {
    let meta = worker.metadata();
    (
        meta.spec.worker_type,
        meta.spec.connection_mode,
        worker.model_id().to_string(),
    )
}

/// Set the worker pool-size gauge of each group in `groups` to the registry's
/// current count of that group, after workers joined or left it.
pub fn refresh_worker_pool_sizes<G: Generation>(registry: &WorkerRegistry<G>, groups: &PoolGroups) {
    for (worker_type, connection_mode, model_id) in groups {
        let pool_size = registry
            .get_workers_filtered(
                Some(model_id),
                Some(*worker_type),
                Some(*connection_mode),
                None,
                false,
            )
            .len();
        Metrics::set_worker_pool_size(
            worker_type.as_metric_label(),
            connection_mode.as_metric_label(),
            model_id,
            pool_size,
        );
    }
}
