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
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the queue is supplied by the host")
    )]
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
mod tests {
    use std::error::Error;

    use serde_json::to_value;
    use smg_service_discovery::testing::RecordedJobs;

    use super::*;
    use crate::{
        service::controller::worker_objects::{ObjectsWake, WorkerObjects},
        store::{MemoryStore, WorkerFacts, WorkersStore},
    };

    const URL: &str = "http://worker:8000";

    fn follow_a_row() -> (WorkerObjects, Option<WorkerId>) {
        let store = Arc::new(MemoryStore::new());
        let workers = WorkerObjects::new(Arc::clone(&store), ObjectsWake::default());
        store.upsert(WorkerFacts {
            key: URL.to_owned(),
            engine: URL.to_owned(),
            models: vec!["m".to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        });
        workers.objects().follow(&*store);
        let object = store
            .by_key(URL)
            .and_then(|row| workers.objects().object(row.id));
        (workers, object)
    }

    #[test]
    fn the_host_lists_the_objects_of_the_rows_that_stand() {
        let (workers, id) = follow_a_row();
        let host = Host::new(Arc::clone(workers.objects().registry()), None);
        let actual: Vec<_> = host
            .workers()
            .into_iter()
            .map(|(id, worker)| (id, worker.url().to_owned()))
            .collect();
        assert_eq!(actual, [(id.expect("followed row"), URL.to_owned())]);
    }

    #[test]
    fn a_laid_object_has_its_origin_and_an_unknown_id_has_none() {
        let (workers, id) = follow_a_row();
        let host = Host::new(Arc::clone(workers.objects().registry()), None);
        assert_eq!(
            host.origin_of(&id.expect("followed row")),
            Some(WorkerOrigin::Local)
        );
        assert_eq!(host.origin_of(&WorkerId::new()), None);
    }

    #[test]
    fn the_host_stamps_its_credential_and_nothing_else() -> Result<(), Box<dyn Error>> {
        for key in [Some("discovery-key"), None] {
            let host = Host::new(
                Arc::new(WorkerRegistry::new()),
                key.map(Credential::for_test),
            );
            let mut spec = WorkerSpec::new(URL);
            spec.api_key = Some("old-key".to_owned());
            spec.max_connection_attempts = 23;
            spec.labels.insert("role".to_owned(), "worker".to_owned());
            let before = to_value(&spec)?;
            host.stamp(&mut spec);
            assert_eq!(spec.api_key.as_deref(), key);
            assert_eq!(spec.max_connection_attempts, 23);
            assert_eq!(to_value(&spec)?, before);
        }
        Ok(())
    }

    #[test]
    fn the_host_has_no_queue_until_one_is_set_and_keeps_the_first() {
        let host = Host::new(Arc::new(WorkerRegistry::new()), None);
        let first: Arc<dyn WorkerJobs> = Arc::new(RecordedJobs::default());
        let second = Arc::new(RecordedJobs::default());
        assert!(host.jobs().is_none());
        assert!(host.set_jobs(Arc::clone(&first)));
        assert!(!host.set_jobs(second));
        assert!(Arc::ptr_eq(&host.jobs().expect("standing queue"), &first));
    }
}
