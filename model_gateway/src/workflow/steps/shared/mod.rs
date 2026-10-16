//! Shared worker registration steps used by both local and external workflows.
//!
//! These steps are designed to work with any worker type and can be composed
//! into different workflows using the DAG-based workflow engine.

mod activate;
mod register;
mod update_policies;

use std::{collections::HashSet, sync::Arc};

pub use activate::ActivateWorkersStep;
use openai_protocol::worker::{ConnectionMode, WorkerType};
pub use register::RegisterWorkersStep;
pub use update_policies::UpdatePoliciesStep;

use crate::{
    observability::metrics::Metrics,
    worker::{
        worker::{ConnectionModeExt, WorkerTypeExt},
        Worker, WorkerRegistry,
    },
};

/// The key the pool-size gauge is kept by: a worker's type, connection mode
/// and primary model id.
pub(crate) type PoolGroup = (WorkerType, ConnectionMode, String);

/// A set of unique pool groups.
pub(crate) type PoolGroups = HashSet<PoolGroup>;

/// The pool-size gauge's key for one worker. Registration and removal key the
/// gauge through this, so the two cannot drift apart.
pub(crate) fn pool_group(worker: &dyn Worker) -> PoolGroup {
    let meta = worker.metadata();
    (
        meta.spec.worker_type,
        meta.spec.connection_mode,
        worker.model_id().to_string(),
    )
}

/// Set the worker pool-size gauge of each group in `groups` to the registry's
/// current count of that group, after workers joined or left it.
pub(crate) fn refresh_worker_pool_sizes(registry: &WorkerRegistry, groups: &PoolGroups) {
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

/// Type alias for a collection of workers in workflow context.
/// Both local (single/DP-aware) and external (multi-model) workflows
/// use this unified type for consistency.
pub type WorkerList = Vec<Arc<dyn Worker>>;
