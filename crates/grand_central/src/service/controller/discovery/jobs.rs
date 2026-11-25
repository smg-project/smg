//! The job queue of a controller's own discovery.
//!
//! A discovery pass asks its host's queue to register the workers it found
//! and to remove the ones its source lists no more. The queue is the worker
//! crate's ([`JobQueue`]) and [`Jobs`] is its host: it runs each job as that
//! crate's workflow, with the store's write at the replica's own door
//! ([`registration_into_store`], [`removal_from_store`]). A registration
//! dials the worker as a gateway's does, and its last step writes the
//! worker's row. A removal marks the rows draining, waits for the workers to
//! settle and takes the rows out. The replica's worker objects follow the
//! rows: no step here writes the registry.
//!
//! The door reads the term this replica leads under at each write, so a job
//! that ends behind a lost lead writes nothing.
//!
//! A job's wait can end ahead of its run. Each engine sweeps the state such
//! a run leaves at its end ([`Jobs::swept`]).

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use smg_grpc_client::engine_port::GrpcConnector;
use smg_worker::{
    connection::{
        engine::EngineConnector,
        http_client::{WorkerHttpClientCache, WorkerHttpClientSettings},
    },
    health::overload::OverloadThresholds,
    identity::endpoint::Endpoint,
    jobs::{
        queue::{JobHost, JobQueue, JobQueueConfig, JobRun, QueueJob},
        WorkerJob, WorkerJobs,
    },
    registry::WorkerRegistry,
    settings::{
        RegistrationSettings, RetryConfig, UnhealthyRemovalScope, DEFAULT_ZMQ_STARTUP_TIMEOUT_SECS,
    },
    worker::HealthCheckConfig,
    workflow::{
        RegistrationGates, WorkerKind, WorkerRegistrationWorkflowData, WorkerRemovalRequest,
        WorkerRemovalWorkflowData, WorkflowHost, WORKER_REGISTRATION_WORKFLOW,
        WORKER_REMOVAL_WORKFLOW,
    },
};
use wfaas::{ValidationError, WorkflowEngine, WorkflowId};

use crate::worker_host::{
    registration_into_store, removal_from_store, Members, MembershipWriter, Objects,
};

/// How long a starting worker is polled before its registration fails, in
/// seconds: a gateway's default.
const STARTUP_TIMEOUT_SECS: u64 = 1800;
/// The pause between two probes of a starting worker, in seconds: a
/// gateway's default.
const STARTUP_CHECK_INTERVAL_SECS: u64 = 30;
/// The timeout of a request a registration sends its worker, in seconds.
const REQUEST_TIMEOUT_SECS: u64 = 30;
/// What a job's wait adds to its run's own bound.
const MARGIN: Duration = Duration::from_secs(30);
/// How long the state of an ended run is kept where its job's wait ended
/// first: a gateway's value.
const ENDED_RUN_KEPT: Duration = Duration::from_secs(300);
/// The pause between two sweeps of such states: a gateway's value.
const ENDED_RUN_SWEEP: Duration = Duration::from_secs(60);

type Registration = WorkerRegistrationWorkflowData<Jobs>;
type Removal = WorkerRemovalWorkflowData<Jobs>;

/// What a controller fixes for every registration it runs, beside
/// `overload`: the default overload thresholds its start states. The
/// object a registration builds states them, as the row's standing object
/// does while the document's overload switch is on. That object is
/// dropped, and no row reads them.
/// No flag sets another yet. A worker is polled for
/// [`STARTUP_TIMEOUT_SECS`] every [`STARTUP_CHECK_INTERVAL_SECS`] from the
/// job's start, over HTTP/1.1 where it speaks HTTP. Its row states the
/// worker crate's default health checks where its registration states none.
fn settings(overload: OverloadThresholds) -> RegistrationSettings {
    let health = HealthCheckConfig::default();
    RegistrationSettings {
        upstream_http2: false,
        health_check_timeout_secs: health.timeout_secs,
        startup_timeout_secs: STARTUP_TIMEOUT_SECS,
        startup_delay_secs: 0,
        startup_check_interval_secs: STARTUP_CHECK_INTERVAL_SECS,
        model_path: None,
        model_aliases: HashMap::new(),
        // The controller sends a worker no request of a client: the object a
        // registration builds states its row and is dropped.
        retry: RetryConfig::default(),
        retries_enabled: false,
        circuit_breaker: Default::default(),
        circuit_breaker_enabled: false,
        overload_defaults: overload,
        health,
        health_endpoint: "/health".to_owned(),
        zmq_startup_timeout_secs: DEFAULT_ZMQ_STARTUP_TIMEOUT_SECS,
        request_timeout_secs: REQUEST_TIMEOUT_SECS,
    }
}

/// The host of a controller's job queue: what runs a discovery pass's jobs.
pub(crate) struct Jobs {
    registry: Arc<WorkerRegistry<Members>>,
    clients: WorkerHttpClientCache,
    settings: RegistrationSettings,
    registration: WorkflowEngine<Registration>,
    removal: WorkflowEngine<Removal>,
    /// How long a removal's job waits for its run: the run's own bound. A
    /// job that gave up earlier would free the address while its run went
    /// on to take the worker out.
    removal_wait: Duration,
}

impl Jobs {
    /// The host of a queue whose jobs write at `door`, over the worker
    /// objects of `objects`, each registration under the start's default
    /// thresholds `overload`. A definition that does not validate is a fault
    /// of this file and is answered as it is.
    pub(crate) fn new(
        objects: &Arc<Objects<Members>>,
        door: Arc<dyn MembershipWriter>,
        overload: OverloadThresholds,
    ) -> Result<Arc<Self>, ValidationError> {
        Self::swept(objects, door, overload, ENDED_RUN_KEPT)
    }

    /// [`Self::new`] with `kept`, how long an ended run's state stands
    /// ahead of its sweep.
    ///
    /// A job's wait can end ahead of its run: a starting worker is probed
    /// for longer than the wait where each probe runs to its timeout.
    /// Nobody then takes the run's state at its end, and that state holds
    /// this host. So each engine sweeps the states of its ended runs, as a
    /// gateway's engines do. A sweep's task ends with its engine. Called
    /// in a runtime: the sweeps are tasks of it.
    fn swept(
        objects: &Arc<Objects<Members>>,
        door: Arc<dyn MembershipWriter>,
        overload: OverloadThresholds,
        kept: Duration,
    ) -> Result<Arc<Self>, ValidationError> {
        let settings = settings(overload);
        let registration = WorkflowEngine::new();
        registration.register_workflow(registration_into_store::<Self>(
            &settings,
            RegistrationGates::default(),
            Arc::clone(&door),
            Arc::clone(objects),
            &[],
        ))?;
        registration.start_cleanup_task(Some(kept), Some(ENDED_RUN_SWEEP));
        // A probe removes no row here, so no removal is the health loop's.
        let scope = UnhealthyRemovalScope::Disabled;
        let leave = removal_from_store::<Self>(door, Arc::clone(objects), scope);
        let removal_wait = leave
            .steps
            .iter()
            .map(|step| leave.get_timeout(step) * leave.get_retry_policy(step).max_attempts.max(1))
            .sum::<Duration>()
            + MARGIN;
        let removal = WorkflowEngine::new();
        removal.register_workflow(leave)?;
        removal.start_cleanup_task(Some(kept), Some(ENDED_RUN_SWEEP));
        Ok(Arc::new(Self {
            registry: Arc::clone(objects.registry()),
            clients: WorkerHttpClientCache::new(WorkerHttpClientSettings {
                client_identity: None,
                ca_certificates: Vec::new(),
                upstream_http2: settings.upstream_http2,
                request_timeout_secs: settings.request_timeout_secs,
                pool_idle_timeout_secs: 0,
            }),
            settings,
            registration,
            removal,
            removal_wait,
        }))
    }

    /// The worker crate's queue over this host. The queue holds its host
    /// weakly: whoever keeps the queue keeps the host beside it.
    pub(crate) fn queue(self: &Arc<Self>) -> Arc<dyn WorkerJobs> {
        JobQueue::new(JobQueueConfig::default(), Arc::downgrade(self))
    }

    /// How long a registration's job waits for its run: the worker's whole
    /// start, as a gateway's job waits.
    fn registration_wait(&self) -> Duration {
        Duration::from_secs(self.settings.startup_timeout_secs + self.settings.startup_delay_secs)
            + MARGIN
    }
}

impl WorkflowHost for Jobs {
    type Generation = Members;
    type RegistrationNotes = ();

    fn worker_registry(&self) -> &Arc<WorkerRegistry<Members>> {
        &self.registry
    }

    fn worker_client_cache(&self) -> &WorkerHttpClientCache {
        &self.clients
    }

    fn registration_settings(&self) -> &RegistrationSettings {
        &self.settings
    }

    // The connector each of the replica's worker objects dials through.
    fn engine_connector(&self) -> Arc<dyn EngineConnector> {
        Arc::new(GrpcConnector)
    }
}

#[async_trait]
impl JobHost for Jobs {
    type Job = WorkerJob;
    const NAME: &'static str = "controller discovery";

    async fn execute(self: &Arc<Self>, job: &WorkerJob, _: &JobRun<'_>) -> Result<String, String> {
        match job {
            WorkerJob::Add { spec, mode, asker } => {
                let mut data = Registration::new(
                    (**spec).clone(),
                    mode.clone(),
                    Arc::clone(self),
                    asker.clone(),
                );
                // The kind is its host's to state, and every step of the line
                // skips a registration of another kind. A controller's
                // discovery finds the engines its deployment runs.
                data.worker_kind = Some(WorkerKind::Local);
                let run = self
                    .registration
                    .start_workflow(WorkflowId::new(WORKER_REGISTRATION_WORKFLOW), data)
                    .await
                    .map_err(|error| format!("the registration did not start: {error:?}"))?;
                let label = Endpoint::redacted_str(&spec.url);
                self.registration
                    .wait_for_completion(run, &label, self.registration_wait())
                    .await
            }
            WorkerJob::Remove {
                url,
                expected_revisions,
                asker,
            } => {
                let data = Removal {
                    config: WorkerRemovalRequest {
                        url: url.clone(),
                        expected_revisions: expected_revisions.clone(),
                    },
                    asker: asker.clone(),
                    workers_to_remove: None,
                    worker_urls: Vec::new(),
                    affected_models: HashSet::new(),
                    host: Some(Arc::clone(self)),
                    actual_workers_to_remove: None,
                    drained_workers: Vec::new(),
                    removed_workers: Vec::new(),
                };
                let run = self
                    .removal
                    .start_workflow(WorkflowId::new(WORKER_REMOVAL_WORKFLOW), data)
                    .await
                    .map_err(|error| format!("the removal did not start: {error:?}"))?;
                let label = Endpoint::redacted_str(url);
                self.removal
                    .wait_for_completion(run, &label, self.removal_wait)
                    .await
            }
            // A discovery pass asks neither, and the health loop of a
            // controller is started with no queue.
            WorkerJob::Update { .. } | WorkerJob::RefreshLabels { .. } => Err(format!(
                "a controller's queue runs no {} job",
                job.job_type()
            )),
        }
    }
}

#[cfg(test)]
mod tests;
