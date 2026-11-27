//! The registry and job queue a discovery pass reads.

use std::sync::{Arc, OnceLock};

use smg_service_discovery::DiscoveryHost;
use smg_worker::{
    identity::basic::Worker,
    jobs::WorkerJobs,
    registry::{WorkerId, WorkerOrigin, WorkerRegistry},
    worker::WorkerSpec,
};

use super::flags::Credential;
use crate::worker_host::Members;

/// The controller as a host of worker discovery.
pub(crate) struct Host {
    registry: Arc<WorkerRegistry<Members>>,
    credential: Option<Credential>,
    jobs: OnceLock<Arc<dyn WorkerJobs>>,
}

impl Host {
    pub(crate) fn new(
        registry: Arc<WorkerRegistry<Members>>,
        credential: Option<Credential>,
    ) -> Self {
        Self {
            registry,
            credential,
            jobs: OnceLock::new(),
        }
    }

    /// Hands the host its job queue. A queue that stands is kept, and a
    /// second call returns `false`.
    pub(crate) fn set_jobs(&self, jobs: Arc<dyn WorkerJobs>) -> bool {
        self.jobs.set(jobs).is_ok()
    }
}

impl DiscoveryHost for Host {
    fn workers(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.registry.get_all_with_ids()
    }

    fn origin_of(&self, worker_id: &WorkerId) -> Option<WorkerOrigin> {
        self.registry.origin_of(worker_id)
    }

    /// The registration queue supplies the budget of connection attempts.
    fn stamp(&self, spec: &mut WorkerSpec) {
        spec.api_key = self.credential.as_ref().map(|key| key.value().to_owned());
    }

    fn jobs(&self) -> Option<Arc<dyn WorkerJobs>> {
        self.jobs.get().cloned()
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/service/controller/discovery/host/tests.rs"]
mod tests;
