//! The gateway as the host of worker discovery: the providers reconcile
//! against its registry, and its job queue takes what a pass finds.

use std::sync::Arc;

use openai_protocol::worker::WorkerSpec;
use smg_service_discovery::DiscoveryHost;
use smg_worker::{
    identity::basic::Worker,
    jobs::WorkerJobs,
    registry::{WorkerId, WorkerOrigin},
};

use crate::{app_context::AppContext, config::RouterConfig};

impl DiscoveryHost for AppContext {
    fn workers(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.worker_registry.get_all_with_ids()
    }

    fn origin_of(&self, worker_id: &WorkerId) -> Option<WorkerOrigin> {
        self.worker_registry.origin_of(worker_id)
    }

    fn stamp(&self, spec: &mut WorkerSpec) {
        spec.api_key.clone_from(&self.router_config.api_key);
        spec.max_connection_attempts = self
            .router_config
            .connection_attempt_budget(RouterConfig::DISCOVERED_WORKER_ATTEMPTS_PER_SUCCESS);
    }

    fn jobs(&self) -> Option<Arc<dyn WorkerJobs>> {
        self.worker_job_queue
            .get()
            .map(|queue| Arc::clone(queue) as Arc<dyn WorkerJobs>)
    }
}

#[cfg(test)]
mod tests;
