//! Shared test fixtures for the discovery modules.

#[cfg(feature = "kubernetes")]
use std::collections::BTreeMap;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
#[cfg(feature = "kubernetes")]
use k8s_openapi::{
    api::core::v1::{Pod, PodCondition, PodSpec, PodStatus},
    apimachinery::pkg::apis::meta::v1::ObjectMeta,
};
use openai_protocol::worker::WorkerSpec;
use parking_lot::Mutex;
use smg_worker::{
    identity::basic::Worker,
    jobs::{WorkerJobAsker, WorkerJobs, WorkerRegistrationMode},
    registry::{WorkerId, WorkerOrigin},
    test_support::{BasicWorkerBuilder, WorkerRegistry},
};

use super::host::DiscoveryHost;

/// The credential [`TestHost`] stamps on a spec discovery registers.
pub const STAMPED_API_KEY: &str = "host-key";
/// The budget of connection attempts [`TestHost`] stamps beside it.
pub const STAMPED_ATTEMPTS: u32 = 7;

/// A host for the crate's cases and for those of a crate that hosts
/// discovery: a registry of its own, a stamp a case can read back, and a job
/// queue that records what a pass asks of it.
pub struct TestHost {
    /// The registry the host answers from: a case registers workers in it and
    /// reads back what a pass left there.
    pub registry: Arc<WorkerRegistry>,
    jobs: Option<Arc<RecordedJobs>>,
}

impl TestHost {
    /// A host whose job queue does not stand yet: a pass defers what it
    /// finds.
    pub fn without_a_queue() -> Arc<Self> {
        Arc::new(Self {
            registry: Arc::new(WorkerRegistry::new()),
            jobs: None,
        })
    }

    /// A host whose queue records each job and runs none.
    pub fn recording() -> (Arc<Self>, Arc<RecordedJobs>) {
        Self::over(Arc::new(WorkerRegistry::new()), RecordedJobs::default())
    }

    /// A host whose queue records each job and applies it to the host's
    /// registry: an addition registers its worker and a removal takes its
    /// workers out.
    pub fn applying() -> (Arc<Self>, Arc<RecordedJobs>) {
        let registry = Arc::new(WorkerRegistry::new());
        let jobs = RecordedJobs {
            applies_to: Some(Arc::clone(&registry)),
            ..RecordedJobs::default()
        };
        Self::over(registry, jobs)
    }

    fn over(registry: Arc<WorkerRegistry>, jobs: RecordedJobs) -> (Arc<Self>, Arc<RecordedJobs>) {
        let jobs = Arc::new(jobs);
        let host = Arc::new(Self {
            registry,
            jobs: Some(Arc::clone(&jobs)),
        });
        (host, jobs)
    }
}

impl DiscoveryHost for TestHost {
    fn workers(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.registry.get_all_with_ids()
    }

    fn origin_of(&self, worker_id: &WorkerId) -> Option<WorkerOrigin> {
        self.registry.origin_of(worker_id)
    }

    fn stamp(&self, spec: &mut WorkerSpec) {
        spec.api_key = Some(STAMPED_API_KEY.to_string());
        spec.max_connection_attempts = STAMPED_ATTEMPTS;
    }

    fn jobs(&self) -> Option<Arc<dyn WorkerJobs>> {
        self.jobs
            .as_ref()
            .map(|jobs| Arc::clone(jobs) as Arc<dyn WorkerJobs>)
    }
}

/// One job the queue took from a pass.
#[derive(Debug, Clone)]
pub enum Queued {
    Add {
        spec: Box<WorkerSpec>,
        mode: WorkerRegistrationMode,
        asker: WorkerJobAsker,
    },
    Remove {
        url: String,
        guards: Vec<(WorkerId, u64)>,
        asker: WorkerJobAsker,
    },
}

/// What holds an address in the queue.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Hold {
    /// A registration: the address's addition is refused.
    Registration,
    /// A removal or a re-read of labels: the queue still takes the address's
    /// addition.
    AnotherJob,
}

#[derive(Default)]
struct Queue {
    queued: Vec<Queued>,
    refused: Vec<String>,
    held: HashMap<String, Hold>,
    held_after_a_read: HashSet<String>,
    slow: HashSet<String>,
    waiting: Vec<WorkerSpec>,
}

/// A job queue that records the jobs it took, in order, and the additions it
/// refused. An address a case holds reads as in flight. Its addition is
/// refused under a registration's hold and taken under another job's, as the
/// job port says of a host's queue. In one place it is not a host's queue:
/// its removal takes each worker a guard names whatever its revision, where
/// a host's queue takes it at the guard's revision alone.
#[derive(Default)]
pub struct RecordedJobs {
    queue: Mutex<Queue>,
    applies_to: Option<Arc<WorkerRegistry>>,
}

impl RecordedJobs {
    /// Every job the queue took, in order.
    pub fn queued(&self) -> Vec<Queued> {
        self.queue.lock().queued.clone()
    }

    /// The address of every addition the queue refused, in order.
    pub fn refused(&self) -> Vec<String> {
        self.queue.lock().refused.clone()
    }

    /// A registration holds `url` from now on, until [`Self::release`]: the
    /// address reads as in flight and its addition is refused.
    pub fn hold(&self, url: &str) {
        self.hold_as(url, Hold::Registration);
    }

    /// A removal or a re-read of labels holds `url` from now on, until
    /// [`Self::release`]: the address reads as in flight, and the queue still
    /// takes its addition.
    pub fn hold_for_another_job(&self, url: &str) {
        self.hold_as(url, Hold::AnotherJob);
    }

    fn hold_as(&self, url: &str, hold: Hold) {
        self.queue.lock().held.insert(url.to_string(), hold);
    }

    /// Another registration takes `url` right after the queue is next asked
    /// about it: that answer is "not in flight", and the address is held from
    /// then on.
    pub fn hold_after_the_next_read(&self, url: &str) {
        self.queue.lock().held_after_a_read.insert(url.to_string());
    }

    /// The next registration of `url` is slow: the queue takes it, the
    /// address is held, and the worker registers at [`Self::release`].
    pub fn register_at_release(&self, url: &str) {
        self.queue.lock().slow.insert(url.to_string());
    }

    /// End the hold on `url`. A slow registration of it lands now.
    pub fn release(&self, url: &str) {
        let landed: Vec<WorkerSpec> = {
            let mut queue = self.queue.lock();
            queue.held.remove(url);
            let (landed, waiting) = queue.waiting.drain(..).partition(|spec| spec.url == url);
            queue.waiting = waiting;
            landed
        };
        for spec in landed {
            self.register(spec);
        }
    }

    fn register(&self, spec: WorkerSpec) {
        if let Some(registry) = &self.applies_to {
            registry.register_or_replace(Arc::new(BasicWorkerBuilder::from_spec(spec).build()));
        }
    }
}

#[async_trait]
impl WorkerJobs for RecordedJobs {
    async fn remove_workers_at(
        &self,
        url: String,
        guards: Vec<(WorkerId, u64)>,
        asker: WorkerJobAsker,
    ) -> Result<(), String> {
        if let Some(registry) = &self.applies_to {
            for (worker_id, _) in &guards {
                registry.remove(worker_id);
            }
        }
        self.queue
            .lock()
            .queued
            .push(Queued::Remove { url, guards, asker });
        Ok(())
    }

    async fn refresh_worker_labels(
        &self,
        _url: String,
        _asker: WorkerJobAsker,
    ) -> Result<bool, String> {
        Ok(true)
    }

    async fn add_worker(
        &self,
        spec: WorkerSpec,
        mode: WorkerRegistrationMode,
        asker: WorkerJobAsker,
    ) -> Result<bool, String> {
        let lands_now = {
            let mut queue = self.queue.lock();
            if queue.held.get(&spec.url) == Some(&Hold::Registration) {
                queue.refused.push(spec.url);
                return Ok(false);
            }
            queue.queued.push(Queued::Add {
                spec: Box::new(spec.clone()),
                mode,
                asker,
            });
            if queue.slow.remove(&spec.url) {
                queue.held.insert(spec.url.clone(), Hold::Registration);
                queue.waiting.push(spec.clone());
                false
            } else {
                true
            }
        };
        if lands_now {
            self.register(spec);
        }
        Ok(true)
    }

    fn job_in_flight(&self, url: &str) -> bool {
        let mut queue = self.queue.lock();
        if queue.held.contains_key(url) {
            return true;
        }
        if queue.held_after_a_read.remove(url) {
            queue.held.insert(url.to_string(), Hold::Registration);
        }
        false
    }
}

/// A running, ready pod named `name` at `ip`, carrying `labels`: the shape
/// the Kubernetes watchers' tests feed to their label selectors.
#[cfg(feature = "kubernetes")]
pub fn make_labeled_pod(name: &str, ip: &str, labels: &[(&str, &str)]) -> Pod {
    let mut label_map = BTreeMap::new();
    for &(k, v) in labels {
        label_map.insert(k.to_string(), v.to_string());
    }
    Pod {
        metadata: ObjectMeta {
            name: Some(name.to_string()),
            uid: Some(format!("uid-{name}")),
            labels: Some(label_map),
            ..Default::default()
        },
        spec: Some(PodSpec::default()),
        status: Some(PodStatus {
            pod_ip: Some(ip.to_string()),
            phase: Some("Running".to_string()),
            conditions: Some(vec![PodCondition {
                type_: "Ready".to_string(),
                status: "True".to_string(),
                last_probe_time: None,
                last_transition_time: None,
                message: None,
                reason: None,
                observed_generation: None,
            }]),
            ..Default::default()
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The double refuses an addition as the job port says a host's queue
    /// does: under a registration of the address, and under no other job.
    #[tokio::test]
    async fn an_addition_is_refused_under_a_registration_alone() {
        let (_host, jobs) = TestHost::recording();
        let url = "http://10.0.0.1:8080";

        jobs.hold_for_another_job(url);
        assert!(jobs.job_in_flight(url));
        let taken = jobs
            .add_worker(
                WorkerSpec::new(url),
                WorkerRegistrationMode::Upsert,
                WorkerJobAsker::Discovery,
            )
            .await;
        assert_eq!(taken, Ok(true));
        assert!(jobs.refused().is_empty());

        jobs.hold(url);
        let refused = jobs
            .add_worker(
                WorkerSpec::new(url),
                WorkerRegistrationMode::Upsert,
                WorkerJobAsker::Discovery,
            )
            .await;
        assert_eq!(refused, Ok(false));
        assert_eq!(jobs.refused(), [url]);
        assert_eq!(jobs.queued().len(), 1, "a refused addition is not queued");
    }
}
