//! What the policy tests line their workers up with.

use std::sync::Arc;

use crate::worker::{test_support::no_health_check, BasicWorkerBuilder, Worker, WorkerType};

/// Healthy workers (health checks disabled) for the given URLs.
pub(crate) fn make_workers(urls: &[&str]) -> Vec<Arc<dyn Worker>> {
    urls.iter()
        .map(|u| {
            Arc::new(
                BasicWorkerBuilder::new(*u)
                    .worker_type(WorkerType::Regular)
                    .health_config(no_health_check())
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect()
}
