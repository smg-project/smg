//! Shared test fixtures for the discovery modules.

use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use k8s_openapi::{
    api::core::v1::{Pod, PodCondition, PodSpec, PodStatus},
    apimachinery::pkg::apis::meta::v1::ObjectMeta,
};
use openai_protocol::worker::WorkerSpec;
use parking_lot::Mutex;
use smg_worker::{
    identity::basic::Worker,
    jobs::{WorkerJobs, WorkerRegistrationMode},
    registry::{WorkerId, WorkerOrigin},
    test_support::{BasicWorkerBuilder, WorkerRegistry},
};

use super::host::DiscoveryHost;

/// The credential [`TestHost`] stamps on a spec discovery registers.
pub const STAMPED_API_KEY: &str = "host-key";
/// The budget of connection attempts [`TestHost`] stamps beside it.
pub const STAMPED_ATTEMPTS: u32 = 7;

/// A host for the module's cases: a registry of its own, a stamp a case can
/// read back, and a job queue that records what a pass asks of it.
pub struct TestHost {
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
    },
    Remove {
        url: String,
        guards: Vec<(WorkerId, u64)>,
    },
}

#[derive(Default)]
struct Queue {
    queued: Vec<Queued>,
    refused: Vec<String>,
    held: HashSet<String>,
    held_after_a_read: HashSet<String>,
    slow: HashSet<String>,
    waiting: Vec<WorkerSpec>,
}

/// A job queue that records the jobs it took, in order, and the additions it
/// refused. An address a case holds reads as in flight and its addition is
/// refused, as a host's queue refuses a registration of an address it holds.
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

    /// Another job holds `url` from now on, until [`Self::release`].
    pub fn hold(&self, url: &str) {
        self.queue.lock().held.insert(url.to_string());
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
    ) -> Result<(), String> {
        if let Some(registry) = &self.applies_to {
            for (worker_id, _) in &guards {
                registry.remove(worker_id);
            }
        }
        self.queue
            .lock()
            .queued
            .push(Queued::Remove { url, guards });
        Ok(())
    }

    async fn refresh_worker_labels(&self, _url: String) -> Result<bool, String> {
        Ok(true)
    }

    async fn add_worker(
        &self,
        spec: WorkerSpec,
        mode: WorkerRegistrationMode,
    ) -> Result<bool, String> {
        let lands_now = {
            let mut queue = self.queue.lock();
            if queue.held.contains(&spec.url) {
                queue.refused.push(spec.url);
                return Ok(false);
            }
            queue.queued.push(Queued::Add {
                spec: Box::new(spec.clone()),
                mode,
            });
            if queue.slow.remove(&spec.url) {
                queue.held.insert(spec.url.clone());
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
        if queue.held.contains(url) {
            return true;
        }
        if queue.held_after_a_read.remove(url) {
            queue.held.insert(url.to_string());
        }
        false
    }
}

/// A running, ready pod named `name` at `ip`, carrying `labels`: the shape
/// the Kubernetes watchers' tests feed to their label selectors.
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
