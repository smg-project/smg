//! Jobs the gateway runs through its control plane. The application context
//! runs the workflows and handles their reservations and cancel marks.

use std::{
    collections::HashMap,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use async_trait::async_trait;
use dashmap::DashSet;
use openai_protocol::worker::{
    ConnectionMode, RuntimeType, WorkerSpec, WorkerType, WorkerUpdateRequest,
};
pub use queue::{JobOutcomeError, JobQueueConfig};
use smg_mcp::McpConfig;
use smg_worker::{
    identity::endpoint::Endpoint,
    jobs::{
        queue::{self, JobHost, JobRun, QueueJob},
        WorkerJob, WorkerJobAsker,
    },
    registry::WorkerId,
};
use tracing::{debug, info};
use wfaas::{
    StateStore, WorkflowData, WorkflowDefinition, WorkflowEngine, WorkflowError, WorkflowId,
    WorkflowStatus,
};

use crate::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    routers::common::snapshot::WorkerRegistry,
    workflow::{
        create_mcp_workflow_data, create_tokenizer_workflow_data,
        create_wasm_registration_workflow_data, create_wasm_removal_workflow_data,
        create_worker_removal_workflow, create_worker_removal_workflow_data,
        create_worker_update_workflow_data, create_worker_workflow_data,
        note_tokenizer_load_outcome, note_tokenizer_removed, registered_entries, registered_in,
        McpServerConfigRequest, TokenizerConfigRequest, TokenizerRemovalRequest,
        WasmModuleConfigRequest, WasmModuleRemovalRequest, WorkerRegistrationMode, WorkflowEngines,
    },
};

/// The control plane's queue, with the application context as its host.
pub type JobQueue = queue::JobQueue<AppContext>;

/// A job the gateway runs through the control plane.
#[derive(Debug, Clone)]
pub enum Job {
    AddWorker {
        config: Box<WorkerSpec>,
        registration_mode: WorkerRegistrationMode,
        asker: WorkerJobAsker,
    },
    UpdateWorker {
        url: String,
        update: Box<WorkerUpdateRequest>,
        asker: WorkerJobAsker,
    },
    /// Re-read a registered worker's labels from its engine and rebuild the
    /// worker if they changed: the worker came into service, as a restarted
    /// engine on the same address or one that finished loading after it was
    /// registered on provisional information. Runs through the update
    /// workflow.
    RefreshWorkerLabels {
        url: String,
        asker: WorkerJobAsker,
    },
    RemoveWorker {
        url: String,
        /// Per-worker revision guards keyed by registry worker id. A DP group
        /// shares one canonical URL but each rank holds its own revision, so
        /// one scalar cannot guard the group. An id mapped to `None` is
        /// removed at any revision (a `DELETE`'s pin: the id it names, not
        /// the build under it). `None` removes every match unguarded.
        expected_revisions: Option<HashMap<String, Option<u64>>>,
        asker: WorkerJobAsker,
    },
    InitializeWorkersFromConfig {
        router_config: Box<RouterConfig>,
    },
    InitializeMcpServers {
        mcp_config: Box<McpConfig>,
    },
    RegisterMcpServer {
        config: Box<McpServerConfigRequest>,
    },
    AddWasmModule {
        config: Box<WasmModuleConfigRequest>,
    },
    RemoveWasmModule {
        request: Box<WasmModuleRemovalRequest>,
    },
    AddTokenizer {
        config: Box<TokenizerConfigRequest>,
    },
    RemoveTokenizer {
        request: Box<TokenizerRemovalRequest>,
    },
}

impl Job {
    /// The type names of the tokenizer jobs: their status route reads them.
    pub(crate) const ADD_TOKENIZER_TYPE: &str = "AddTokenizer";
    pub(crate) const REMOVE_TOKENIZER_TYPE: &str = "RemoveTokenizer";

    /// Get job type as string for logging
    pub fn job_type(&self) -> &'static str {
        match self {
            Job::AddWorker { .. } => Self::REGISTRATION_TYPE,
            Job::UpdateWorker { .. } => WorkerJob::UPDATE_TYPE,
            Job::RefreshWorkerLabels { .. } => WorkerJob::REFRESH_LABELS_TYPE,
            Job::RemoveWorker { .. } => WorkerJob::REMOVAL_TYPE,
            Job::InitializeWorkersFromConfig { .. } => "InitializeWorkersFromConfig",
            Job::InitializeMcpServers { .. } => "InitializeMcpServers",
            Job::RegisterMcpServer { .. } => "RegisterMcpServer",
            Job::AddWasmModule { .. } => "AddWasmModule",
            Job::RemoveWasmModule { .. } => "RemoveWasmModule",
            Job::AddTokenizer { .. } => Self::ADD_TOKENIZER_TYPE,
            Job::RemoveTokenizer { .. } => Self::REMOVE_TOKENIZER_TYPE,
        }
    }

    /// `worker_url` as a log line writes it: the names that are not addresses
    /// (the start-up marker, an MCP server, a WASM module, a tokenizer id) as
    /// they are, since the address parser would respell or refuse them;
    /// anything else is a worker's address, written with a userinfo in it
    /// masked, so a worker job added later cannot write its address raw.
    pub fn shown_worker(job_type: &str, worker_url: &str) -> String {
        match job_type {
            "InitializeWorkersFromConfig"
            | "InitializeMcpServers"
            | "RegisterMcpServer"
            | "AddWasmModule"
            | "RemoveWasmModule"
            | Self::ADD_TOKENIZER_TYPE
            | Self::REMOVE_TOKENIZER_TYPE => worker_url.to_string(),
            _ => WorkerJob::shown_key(job_type, worker_url),
        }
    }

    /// The name a worker job's workflow goes by: the worker's address with a
    /// userinfo in it masked, as the job's own lines write it. The engine
    /// writes the name into the text a wait ends with (completed, cancelled,
    /// timed out), and that text is the job's result: its log line and its
    /// status. Nothing finds a workflow by its name, so two workers whose
    /// addresses differ in the userinfo alone go by one, each tracked under
    /// its own address.
    fn workflow_label(worker_url: &str) -> String {
        Endpoint::redacted_str(worker_url)
    }

    /// Get worker URL, MCP server name, WASM module, or tokenizer identifier for logging and status tracking
    pub fn worker_url(&self) -> &str {
        match self {
            Job::AddWorker { config, .. } => &config.url,
            Job::UpdateWorker { url, .. } => url,
            Job::RefreshWorkerLabels { url, .. } => url,
            Job::RemoveWorker { url, .. } => url,
            Job::InitializeWorkersFromConfig { .. } => "startup",
            Job::InitializeMcpServers { .. } => "startup",
            Job::RegisterMcpServer { config } => &config.name,
            Job::AddWasmModule { config } => &config.uuid_string,
            Job::RemoveWasmModule { request } => &request.uuid_string,
            Job::AddTokenizer { config } => &config.id,
            Job::RemoveTokenizer { request } => &request.id,
        }
    }
}

impl QueueJob for Job {
    const REGISTRATION_TYPE: &'static str = WorkerJob::REGISTRATION_TYPE;

    fn job_type(&self) -> &'static str {
        self.job_type()
    }

    fn key(&self) -> &str {
        self.worker_url()
    }

    fn shown_key(job_type: &str, key: &str) -> String {
        Self::shown_worker(job_type, key)
    }

    fn is_removal(&self) -> bool {
        matches!(self, Self::RemoveWorker { .. })
    }

    fn merge_removals(held: Self, newer: Self) -> Result<Self, (Self, Self)> {
        let jobs = (held, newer);
        let (
            Self::RemoveWorker {
                url,
                expected_revisions,
                asker,
            },
            Self::RemoveWorker {
                url: newer_url,
                expected_revisions: newer_revisions,
                asker: newer_asker,
            },
        ) = jobs
        else {
            return Err(jobs);
        };

        WorkerJob::merge_removals(
            WorkerJob::Remove {
                url,
                expected_revisions,
                asker,
            },
            WorkerJob::Remove {
                url: newer_url,
                expected_revisions: newer_revisions,
                asker: newer_asker,
            },
        )
        .map(Self::from)
        .map_err(|(held, newer)| (held.into(), newer.into()))
    }

    fn asker(&self) -> Option<&WorkerJobAsker> {
        match self {
            Job::AddWorker { asker, .. }
            | Job::UpdateWorker { asker, .. }
            | Job::RefreshWorkerLabels { asker, .. }
            | Job::RemoveWorker { asker, .. } => Some(asker),
            Job::InitializeWorkersFromConfig { .. }
            | Job::InitializeMcpServers { .. }
            | Job::RegisterMcpServer { .. }
            | Job::AddWasmModule { .. }
            | Job::RemoveWasmModule { .. }
            | Job::AddTokenizer { .. }
            | Job::RemoveTokenizer { .. } => None,
        }
    }
}

impl From<WorkerJob> for Job {
    fn from(job: WorkerJob) -> Self {
        match job {
            WorkerJob::Add { spec, mode, asker } => Self::AddWorker {
                config: spec,
                registration_mode: mode,
                asker,
            },
            WorkerJob::Update { url, update, asker } => Self::UpdateWorker { url, update, asker },
            WorkerJob::RefreshLabels { url, asker } => Self::RefreshWorkerLabels { url, asker },
            WorkerJob::Remove {
                url,
                expected_revisions,
                asker,
            } => Self::RemoveWorker {
                url,
                expected_revisions,
                asker,
            },
        }
    }
}

/// How a failed registration's cleanup removal went in: queued as the URL's
/// own job, or behind a removal in flight at the URL (kept to run on its
/// heels, merged with a removal kept there by the same asker).
#[derive(Debug, PartialEq, Eq)]
enum CleanupRemoval {
    Queued,
    Behind,
}

/// Cancels the `POST /workers` registration in flight for `worker_url`,
/// the one `worker_id` was reserved for, and says how many a `DELETE`
/// ends. A workflow the store holds is cancelled (one an earlier `DELETE`
/// cancelled, its job not ended yet, counts without a second cancel, so a
/// repeated `DELETE` queues no removal); a job that has not saved its
/// state yet is marked under `worker_id` and ends when it reads the mark.
/// The cancelled job's end releases the URL's reservation and records its
/// status as for any failed registration. A startup or discovery
/// registration of the same URL (`Upsert`) is not the API's
/// to end and runs on: none is cancelled, and no mark is set while one
/// runs (a `DELETE` then goes to the removal job as before).
pub(crate) async fn cancel_registration(
    queue: &JobQueue,
    worker_url: &str,
    worker_id: &WorkerId,
) -> Result<usize, String> {
    let Some(context) = queue.host() else {
        return Err(format!(
            "Job queue shutting down: {} dropped",
            AppContext::NAME
        ));
    };
    let (cancelled, upsert_running) = cancel_registration_states(&context, worker_url).await?;
    if cancelled > 0 || upsert_running || !add_worker_in_flight(queue, worker_url) {
        return Ok(cancelled);
    }
    // The job is queued, or started and has not saved its state yet. It
    // may have saved it since the listing above: a second listing catches
    // that, and the job's own read after its save catches the rest. A job
    // that ended meanwhile, or an id that no longer names the URL (its
    // registration over, a new one admitted), takes the mark back.
    queue.set_cancel_mark(worker_id);
    let (cancelled, upsert_running) = match cancel_registration_states(&context, worker_url).await {
        Ok(found) => found,
        Err(message) => {
            queue.clear_cancel_mark(worker_id);
            return Err(message);
        }
    };
    if cancelled > 0
        || upsert_running
        || !add_worker_in_flight(queue, worker_url)
        || context.worker_registry.get_url_by_id(worker_id).as_deref() != Some(worker_url)
    {
        queue.clear_cancel_mark(worker_id);
        return Ok(cancelled);
    }
    Ok(1)
}

/// Whether the status entry under `url` is a registration job that has
/// not ended: queued, or running (the type name of [`Job::job_type`]).
pub(crate) fn add_worker_in_flight(queue: &JobQueue, url: &str) -> bool {
    queue.get_status(url).is_some_and(|status| {
        status.job_type == Job::REGISTRATION_TYPE
            && matches!(status.status.as_str(), "pending" | "processing")
    })
}

/// The workflow engines, or the refusal a job answers while they are not
/// built yet.
fn workflow_engines(context: &AppContext) -> Result<&WorkflowEngines, String> {
    context
        .workflow_engines
        .get()
        .ok_or_else(|| "Workflow engines not initialized".to_string())
}

/// Cancels the `POST /workers` registration workflows the store holds for
/// `worker_url`: the active ones are cancelled, a cancelled one counts, one
/// that ended between the listing and the cancel is skipped. Says how many
/// it ended and whether a startup or discovery registration (`Upsert`) of
/// the URL is running, which is not the API's to end.
async fn cancel_registration_states(
    context: &AppContext,
    worker_url: &str,
) -> Result<(usize, bool), String> {
    let engine = &workflow_engines(context)?.worker_registration;
    let states = engine
        .state_store()
        .list_all()
        .await
        .map_err(|e| format!("Failed to list registration workflows: {e:?}"))?;
    let mut cancelled = 0;
    let mut upsert_running = false;
    for state in states {
        if state.context.data.config.url != worker_url {
            continue;
        }
        if !matches!(
            state.context.data.registration_mode,
            WorkerRegistrationMode::CreateOnly
        ) {
            // A `PUT`'s registration (`ReplaceById`) owns no id without a
            // worker behind it, so it blocks no mark.
            upsert_running |= matches!(
                state.context.data.registration_mode,
                WorkerRegistrationMode::Upsert
            ) && matches!(
                state.status,
                WorkflowStatus::Pending | WorkflowStatus::Running
            );
            continue;
        }
        match state.status {
            WorkflowStatus::Pending | WorkflowStatus::Running => {
                match engine.cancel_workflow(state.instance_id).await {
                    Ok(()) => cancelled += 1,
                    // Ended between the listing and the cancel: its job's
                    // end path runs as for any registration that ended.
                    Err(WorkflowError::NotFound(_)) => {}
                    Err(e) => {
                        return Err(format!(
                            "Failed to cancel registration workflow {}: {e:?}",
                            state.instance_id
                        ));
                    }
                }
            }
            // Cancelled by an earlier DELETE; its job has not ended yet.
            WorkflowStatus::Cancelled => cancelled += 1,
            WorkflowStatus::Paused | WorkflowStatus::Completed | WorkflowStatus::Failed => {}
        }
    }
    Ok((cancelled, upsert_running))
}

/// The id whose cancel mark a registration job owns: a `POST`'s job
/// (`CreateOnly`) owns the id its `POST` reserved, read from the URL's
/// mapping itself since the reservation has no worker behind it yet. A
/// `PUT`'s job and a startup or discovery registration own none: the mark
/// is a `DELETE` of a `POST`'s registration, and a job of another kind
/// started under that reservation (a `PUT` queued before the `POST`, a
/// discovery pass over the same URL) leaves it for the `POST`'s job.
fn own_mark_id(job: &Job, registry: &WorkerRegistry) -> Option<WorkerId> {
    match job {
        Job::AddWorker {
            config,
            registration_mode: WorkerRegistrationMode::CreateOnly,
            ..
        } => registry.get_id_by_url_or_reservation(&config.url),
        _ => None,
    }
}

/// Queues the removal of a registration's workers under `url`, pinned to
/// `entries` (their ids and revisions as the registry holds them now, so
/// a worker registered or replaced since is left alone): the job a
/// `DELETE` of a live worker queues, drain and bookkeeping included.
/// Behind a removal in flight at the URL (a `DELETE`'s, an earlier end's)
/// it runs on that one's heels, as a `DELETE`'s does: that one takes the
/// workers it was queued for and leaves these standing when they are
/// not among its pins, so a second is needed, and finds nothing left
/// when they were. A kept removal merges only with one from the same
/// asker.
async fn queue_removal(
    context: &Arc<AppContext>,
    url: &str,
    entries: HashMap<String, u64>,
    asker: WorkerJobAsker,
) -> Result<CleanupRemoval, String> {
    let queue = context
        .worker_job_queue
        .get()
        .ok_or_else(|| "Worker job queue not initialized".to_string())?;
    let job = Job::RemoveWorker {
        url: url.to_string(),
        expected_revisions: Some(
            entries
                .into_iter()
                .map(|(id, revision)| (id, Some(revision)))
                .collect(),
        ),
        asker,
    };
    if queue.claim_if_idle(&job) {
        return queue
            .submit_claimed(job)
            .await
            .map(|()| CleanupRemoval::Queued);
    }
    let submitted = queue.submit_or_defer(job).await?;
    Ok(if submitted {
        CleanupRemoval::Queued
    } else {
        CleanupRemoval::Behind
    })
}

/// Takes the mark under `own_id`, if there is one: true when a `DELETE`
/// ended this registration before its job had saved a workflow state.
fn take_mark(marks: &DashSet<WorkerId>, own_id: Option<&WorkerId>) -> bool {
    own_id.is_some_and(|id| marks.remove(id).is_some())
}

/// How long a removal job waits for its run. A job that gave up earlier
/// would report a failure and free the URL while its run went on to remove
/// the worker.
fn removal_wait() -> Duration {
    run_bound(&create_worker_removal_workflow())
}

/// The longest a run of `workflow` spends in its steps: each step's bound
/// times its attempts, added up. A step counts at least once: the engine
/// runs it once even under a policy of no attempts. The backoff between
/// attempts is not counted.
fn run_bound<D: WorkflowData>(workflow: &WorkflowDefinition<D>) -> Duration {
    workflow
        .steps
        .iter()
        .map(|step| {
            let attempts = workflow.get_retry_policy(step).max_attempts.max(1);
            workflow.get_timeout(step) * attempts
        })
        .sum()
}

/// Start the `workflow` definition on `engine` with `data`, log the
/// instance under `label`, and wait for it to end within `timeout`.
/// `what` names the workflow in the start failure and in the log line.
async fn run_workflow<D: WorkflowData, S: StateStore<D> + 'static>(
    engine: &WorkflowEngine<D, S>,
    workflow: &'static str,
    what: &str,
    label: &str,
    data: D,
    timeout: Duration,
) -> Result<String, String> {
    let instance_id = engine
        .start_workflow(WorkflowId::new(workflow), data)
        .await
        .map_err(|e| format!("Failed to start {what} workflow: {e:?}"))?;
    debug!("Started {what} workflow for {label} (instance: {instance_id})");
    engine
        .wait_for_completion(instance_id, label, timeout)
        .await
}

#[async_trait]
impl JobHost for AppContext {
    type Job = Job;
    const NAME: &'static str = "AppContext";

    fn mark_id(&self, job: &Job) -> Option<WorkerId> {
        own_mark_id(job, &self.worker_registry)
    }

    fn accepted(&self, job: &Job) {
        if let Job::AddTokenizer { config } = job {
            self.tokenizer_load_failures.load_started(&config.name);
        }
    }

    fn failed(&self, job: &Job) {
        if let Job::AddWorker {
            config,
            registration_mode: WorkerRegistrationMode::CreateOnly,
            ..
        } = job
        {
            // Release before the failed status lets a retry reserve the address.
            self.worker_registry.release_reservation(&config.url);
        }
    }

    async fn execute(self: &Arc<Self>, job: &Job, run: &JobRun<'_>) -> Result<String, String> {
        let context = self;
        match job {
            Job::AddWorker {
                config,
                registration_mode,
                asker,
            } => {
                // A DELETE that came before this job started ended the
                // registration (`cancel_registration`): the job ends as a
                // failed one, and its end path releases the URL. Only a POST's
                // job owns a mark (`own_mark_id`): a PUT's job and a startup
                // or discovery registration of the same URL read none, so no
                // other kind can consume a mark meant for the POST's job.
                if take_mark(run.cancelled_before_start, run.own_id) {
                    run.ended_by_delete.store(true, Ordering::Relaxed);
                    return Err("Registration cancelled before it started".to_string());
                }
                let engines = workflow_engines(context)?;

                // The startup delay stands ahead of the run's first probe:
                // a wait that left it out would end that much before the run.
                let timeout_duration = Duration::from_secs(
                    context.router_config.worker_startup_timeout_secs
                        + context.router_config.worker_startup_delay_secs
                        + 30,
                );

                let workflow_data = create_worker_workflow_data(
                    (**config).clone(),
                    registration_mode.clone(),
                    Arc::clone(context),
                    asker.clone(),
                );
                let instance_id = engines
                    .worker_registration
                    .start_workflow(WorkflowId::new("worker_registration"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start worker registration workflow: {e:?}"))?;
                // A DELETE between the read above and the state's save found
                // no state to cancel and left its mark: the workflow just
                // started is cancelled here, and the wait below ends with it.
                if take_mark(run.cancelled_before_start, run.own_id) {
                    // The `DELETE` is answered already: whatever the cancel
                    // and the wait below return, a `DELETE` ended this job.
                    run.ended_by_delete.store(true, Ordering::Relaxed);
                    engines
                        .worker_registration
                        .cancel_workflow(instance_id)
                        .await
                        .map_err(|e| {
                            format!("Failed to cancel worker registration workflow: {e:?}")
                        })?;
                }

                debug!(
                    "Started worker registration workflow for {} (instance: {})",
                    Endpoint::redacted_str(&config.url),
                    instance_id
                );

                let label = Job::workflow_label(&config.url);
                let engine = &engines.worker_registration;
                let create_only = matches!(registration_mode, WorkerRegistrationMode::CreateOnly);
                // A POST's wait ends at the job's own deadline: the engine's
                // timeout reclaims a state that has just ended, and the cancel
                // below needs it. The outer `Err` is that deadline alone.
                let waited = if create_only {
                    tokio::time::timeout(
                        timeout_duration,
                        engine.wait_for_terminal_state(instance_id, &label, Duration::MAX),
                    )
                    .await
                } else {
                    Ok(engine
                        .wait_for_terminal_state(instance_id, &label, timeout_duration)
                        .await)
                };
                let (state, outcome) = match waited {
                    Ok(Ok(state)) => {
                        let outcome = wfaas::completion_outcome(&state, &label);
                        (state, outcome)
                    }
                    // The wait's own failure. No job cleans up after a cancel
                    // of a registration that is not a POST's, so its run is
                    // left to finish.
                    Ok(Err(failure)) => return Err(failure),
                    // A POST's wait that ends first cancels its run: left
                    // going, the run would register the worker under a job
                    // that already failed. A cancelled run ends with the
                    // attempt it is in, whose bound is within the wait, and
                    // its end is read as any other. The job writes the
                    // engine's words for the wait that ended.
                    Err(_) => {
                        let waited_out = format!(
                            "Workflow timeout after {}s for {label}",
                            timeout_duration.as_secs()
                        );
                        engine.cancel_workflow(instance_id).await.map_err(|e| {
                            format!("{waited_out}; its run could not be cancelled: {e:?}")
                        })?;
                        let state = engine
                            .wait_for_terminal_state(instance_id, &label, timeout_duration)
                            .await
                            .map_err(|e| {
                                format!("{waited_out}; its cancelled run did not end: {e}")
                            })?;
                        (state, Err(format!("{waited_out}; its run was cancelled")))
                    }
                };
                // A POST's registration that put its worker in the registry
                // and did not finish (a DELETE's cancel landing after the
                // registry step, a later step's failure) leaves no worker
                // behind: it goes out through the removal job, as a DELETE of
                // a live worker sends it. A startup or discovery registration
                // is its own pass's to reconcile.
                match outcome {
                    Err(failure) if create_only => {
                        if state.status == WorkflowStatus::Cancelled {
                            run.ended_by_delete.store(true, Ordering::Relaxed);
                        }
                        // The registry says what this registration left in it
                        // (its workers still the entries under their URLs, by
                        // id and revision): a worker registered or replaced
                        // since is not this job's to remove, and the step
                        // record alone can be read before the step's end.
                        let entries = registered_entries(&context.worker_registry, &state);
                        Err(if entries.is_empty() {
                            if registered_in(&state) {
                                format!(
                                    "{failure}; its workers were registered and are no longer \
                                    the entries under their URLs (removed or replaced since)"
                                )
                            } else {
                                failure
                            }
                        } else {
                            // A registration is cancelled by a DELETE, or by
                            // its own job at its wait's end, and a POST's job
                            // is the API's too. If a step failed, the cleanup
                            // is part of the job that asked for the
                            // registration.
                            let asker = if state.status == WorkflowStatus::Cancelled {
                                WorkerJobAsker::Api
                            } else {
                                asker.clone()
                            };
                            match queue_removal(context, &config.url, entries, asker).await {
                                Ok(CleanupRemoval::Queued) => format!(
                                    "{failure}; its workers were registered and leave through \
                                    a queued removal"
                                ),
                                Ok(CleanupRemoval::Behind) => format!(
                                    "{failure}; its workers were registered and leave through \
                                    a removal queued behind the one in flight"
                                ),
                                Err(submit) => format!(
                                    "{failure}; its workers were registered and their removal \
                                    could not be queued: {submit}"
                                ),
                            }
                        })
                    }
                    outcome => outcome,
                }
            }
            Job::UpdateWorker { url, update, asker } => {
                let engines = workflow_engines(context)?;

                let workflow_data = create_worker_update_workflow_data(
                    url.to_string(),
                    (**update).clone(),
                    Arc::clone(context),
                    asker.clone(),
                );

                run_workflow(
                    &engines.worker_update,
                    "worker_update",
                    "worker update",
                    &Job::workflow_label(url),
                    workflow_data,
                    Duration::from_secs(30),
                )
                .await
            }
            Job::RefreshWorkerLabels { url, asker } => {
                let engines = workflow_engines(context)?;

                let mut workflow_data = create_worker_update_workflow_data(
                    url.to_string(),
                    WorkerUpdateRequest {
                        priority: None,
                        cost: None,
                        labels: None,
                        api_key: None,
                        health: None,
                    },
                    Arc::clone(context),
                    asker.clone(),
                );
                workflow_data.refresh_discovered_labels = true;

                run_workflow(
                    &engines.worker_update,
                    "worker_update",
                    "worker label refresh",
                    &Job::workflow_label(url),
                    workflow_data,
                    Duration::from_secs(30),
                )
                .await
            }
            Job::RemoveWorker {
                url,
                expected_revisions,
                asker,
            } => {
                let engines = workflow_engines(context)?;

                let workflow_data = create_worker_removal_workflow_data(
                    url.to_string(),
                    expected_revisions.clone(),
                    Arc::clone(context),
                    asker.clone(),
                );

                let instance_id = engines
                    .worker_removal
                    .start_workflow(WorkflowId::new("worker_removal"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start worker removal workflow: {e:?}"))?;

                debug!(
                    "Started worker removal workflow for {} (instance: {})",
                    Endpoint::redacted_str(url),
                    instance_id
                );

                // Completion releases the claim after the worker leaves.
                // Another kept job can take the URL as soon as it is released.
                engines
                    .worker_removal
                    .wait_for_completion(instance_id, &Job::workflow_label(url), removal_wait())
                    .await
            }
            Job::AddWasmModule { config } => {
                let engines = workflow_engines(context)?;

                let workflow_data =
                    create_wasm_registration_workflow_data(*config.clone(), Arc::clone(context));

                run_workflow(
                    &engines.wasm_registration,
                    "wasm_module_registration",
                    "WASM module registration",
                    &config.descriptor.name,
                    workflow_data,
                    Duration::from_secs(300), // 5 minutes
                )
                .await
            }
            Job::RemoveWasmModule { request } => {
                let engines = workflow_engines(context)?;

                let workflow_data =
                    create_wasm_removal_workflow_data(*request.clone(), Arc::clone(context));

                run_workflow(
                    &engines.wasm_removal,
                    "wasm_module_removal",
                    "WASM module removal",
                    &request.module_uuid.to_string(),
                    workflow_data,
                    Duration::from_secs(60), // 1 minute
                )
                .await
            }
            Job::InitializeWorkersFromConfig { router_config } => {
                let api_key = router_config.api_key.clone();
                let mut worker_count = 0;

                // Create iterator of (url, worker_type, bootstrap_port) tuples based on mode
                let workers: Vec<(String, &str, Option<u16>)> = match &router_config.mode {
                    RoutingMode::Regular { worker_urls } => worker_urls
                        .iter()
                        .map(|url| (url.clone(), "regular", None))
                        .collect(),
                    RoutingMode::PrefillDecode {
                        prefill_urls,
                        decode_urls,
                        ..
                    } => {
                        let prefill_workers = prefill_urls
                            .iter()
                            .map(|(url, port)| (url.clone(), "prefill", *port));

                        let decode_workers =
                            decode_urls.iter().map(|url| (url.clone(), "decode", None));

                        prefill_workers.chain(decode_workers).collect()
                    }
                    RoutingMode::EncodePrefillDecode {
                        encode_urls,
                        prefill_urls,
                        decode_urls,
                        ..
                    } => {
                        let encode_workers = encode_urls
                            .iter()
                            .map(|(url, port)| (url.clone(), "encode", *port));
                        let prefill_workers = prefill_urls
                            .iter()
                            .map(|(url, port)| (url.clone(), "prefill", *port));
                        let decode_workers =
                            decode_urls.iter().map(|url| (url.clone(), "decode", None));

                        encode_workers
                            .chain(prefill_workers)
                            .chain(decode_workers)
                            .collect()
                    }
                    RoutingMode::OpenAI { worker_urls }
                    | RoutingMode::Anthropic { worker_urls }
                    | RoutingMode::Gemini { worker_urls } => {
                        let provider_name = router_config.mode_type();
                        return submit_external_worker_jobs(
                            worker_urls,
                            provider_name,
                            router_config,
                            context,
                        )
                        .await;
                    }
                };

                debug!(
                    "Creating AddWorker jobs for {} workers from config",
                    workers.len()
                );

                // Process all workers with unified loop
                for (url, worker_type, bootstrap_port) in workers {
                    let url_for_error = url.clone(); // Clone for error message
                    let proto_worker_type = match worker_type {
                        "prefill" => WorkerType::Prefill,
                        "decode" => WorkerType::Decode,
                        "encode" => WorkerType::Encode,
                        _ => WorkerType::Regular,
                    };
                    let mut spec = WorkerSpec::new(url);
                    spec.worker_type = proto_worker_type;
                    spec.api_key.clone_from(&api_key);
                    spec.bootstrap_port = bootstrap_port;
                    apply_startup_worker_config(&mut spec, router_config);
                    let config = spec;

                    let job = Job::AddWorker {
                        config: Box::new(config),
                        registration_mode: WorkerRegistrationMode::Upsert,
                        asker: WorkerJobAsker::Startup,
                    };

                    if let Some(queue) = context.worker_job_queue.get() {
                        queue.submit(job).await.map_err(|e| {
                            format!(
                                "Failed to submit AddWorker job for {worker_type} worker {}: {e}",
                                Endpoint::redacted_str(&url_for_error)
                            )
                        })?;
                        worker_count += 1;
                    } else {
                        return Err("JobQueue not available".to_string());
                    }
                }

                Ok(format!("Submitted {worker_count} AddWorker jobs"))
            }
            Job::InitializeMcpServers { mcp_config } => {
                let mut server_count = 0;

                debug!(
                    "Creating RegisterMcpServer jobs for {} MCP servers from config",
                    mcp_config.servers.len()
                );

                // Submit RegisterMcpServer jobs for each server in the config
                for server_config in &mcp_config.servers {
                    let mcp_server_request = McpServerConfigRequest {
                        name: server_config.name.clone(),
                        config: server_config.clone(),
                    };

                    let job = Job::RegisterMcpServer {
                        config: Box::new(mcp_server_request),
                    };

                    if let Some(queue) = context.worker_job_queue.get() {
                        queue.submit(job).await.map_err(|e| {
                            format!(
                                "Failed to submit RegisterMcpServer job for '{}': {}",
                                server_config.name, e
                            )
                        })?;
                        server_count += 1;
                    } else {
                        return Err("JobQueue not available".to_string());
                    }
                }

                Ok(format!("Submitted {server_count} RegisterMcpServer jobs"))
            }
            Job::RegisterMcpServer { config } => {
                let engines = workflow_engines(context)?;

                let workflow_data =
                    create_mcp_workflow_data((**config).clone(), Arc::clone(context));

                run_workflow(
                    &engines.mcp,
                    "mcp_registration",
                    "MCP registration",
                    &config.name,
                    workflow_data,
                    Duration::from_secs(7200 + 30), // 2hr + margin
                )
                .await
            }
            Job::AddTokenizer { config } => {
                // The load was counted in flight at submission; its end is
                // noted for the model (a failure of the model's last load
                // answers its requests 503 tokenizer_load_failed).
                let result = async {
                    let engines = workflow_engines(context)?;

                    let workflow_data =
                        create_tokenizer_workflow_data(*config.clone(), Arc::clone(context));

                    let instance_id = engines
                        .tokenizer
                        .start_workflow(WorkflowId::new("tokenizer_registration"), workflow_data)
                        .await
                        .map_err(|e| {
                            format!("Failed to start tokenizer registration workflow: {e:?}")
                        })?;

                    debug!(
                        "Started tokenizer registration workflow for '{}' id={} (instance: {})",
                        config.name, config.id, instance_id
                    );

                    // Allow up to 10 minutes for HuggingFace downloads
                    let timeout_duration = Duration::from_secs(600);

                    engines
                        .tokenizer
                        .wait_for_completion(instance_id, &config.id, timeout_duration)
                        .await
                }
                .await;
                note_tokenizer_load_outcome(
                    &context.tokenizer_load_failures,
                    &context.tokenizer_registry,
                    &context.worker_registry,
                    &config.name,
                    &result,
                );
                result
            }
            Job::RemoveTokenizer { request } => {
                // Tokenizer removal is synchronous and fast
                if let Some(entry) = context.tokenizer_registry.remove_by_id(&request.id) {
                    context.multimodal_config_registry.remove(&entry.id);
                    context.worker_loaded_tokenizers.remove(&entry.id);
                    note_tokenizer_removed(
                        &context.tokenizer_load_failures,
                        &context.worker_registry,
                        &entry.name,
                    );
                    info!(
                        "Successfully removed tokenizer '{}' (id: {})",
                        entry.name, entry.id
                    );
                    Ok(format!("Tokenizer '{}' removed successfully", entry.name))
                } else {
                    Err(format!("Tokenizer with id '{}' not found", request.id))
                }
            }
        }
    }
}

/// Stamp the router-config-derived fields onto a startup worker's spec: the
/// pinned runtime, the grouped-ZMQ engine count, and the connection budget.
/// Identity fields (url, worker type, api key, bootstrap port) are the
/// caller's.
fn apply_startup_worker_config(spec: &mut WorkerSpec, router_config: &RouterConfig) {
    // ZMQ startup workers carry the runtime pinned by `--backend` (the shared
    // handshake cannot be probed for a wire protocol); `None` — HTTP/gRPC or no
    // `--backend` — keeps auto-detection in detect_backend.
    if let Some(runtime) = router_config.startup_worker_runtime_type {
        spec.runtime_type = runtime;
    }
    // A ZMQ worker given an engine count: the handshake awaits this many DP
    // engines on one socket set, and the worker registers as one worker per
    // engine. Only ZMQ URLs — dp_size on an HTTP/gRPC startup worker would
    // misread as a rank worker's.
    if let Some(count) = router_config.zmq_engine_count.filter(|&n| n > 1) {
        if ConnectionMode::from_url(&spec.url) == Some(ConnectionMode::Zmq) {
            spec.dp_size = Some(count);
        }
    }
    // Health config is resolved at worker build time from router defaults +
    // per-worker overrides (spec.health). No need to set spec.health here since
    // these workers have no overrides.
    spec.max_connection_attempts = router_config
        .connection_attempt_budget(RouterConfig::CONFIGURED_WORKER_ATTEMPTS_PER_SUCCESS);
}

/// Submit AddWorker jobs for external provider endpoints (OpenAI/Anthropic/Gemini).
async fn submit_external_worker_jobs(
    worker_urls: &[String],
    provider_name: &str,
    router_config: &RouterConfig,
    context: &Arc<AppContext>,
) -> Result<String, String> {
    let api_key = router_config.api_key.clone();
    let mut submitted_count = 0;

    for url in worker_urls {
        let url_for_error = url.clone();
        let config = build_external_worker_config(url, api_key.clone(), router_config);

        let job = Job::AddWorker {
            config: Box::new(config),
            registration_mode: WorkerRegistrationMode::Upsert,
            asker: WorkerJobAsker::Startup,
        };

        if let Some(queue) = context.worker_job_queue.get() {
            queue.submit(job).await.map_err(|e| {
                format!(
                    "Failed to submit AddWorker job for {provider_name} endpoint {}: {e}",
                    Endpoint::redacted_str(&url_for_error)
                )
            })?;
            submitted_count += 1;
        } else {
            return Err("JobQueue not available".to_string());
        }
    }

    if submitted_count == 0 {
        info!("{provider_name} mode: no worker URLs provided");
        return Ok(format!(
            "{provider_name} mode: no worker URLs to initialize"
        ));
    }

    Ok(format!(
        "Submitted {submitted_count} AddWorker jobs for {provider_name} endpoints"
    ))
}

/// Build a `WorkerSpec` for an external API endpoint (OpenAI/Anthropic mode).
fn build_external_worker_config(
    url: &str,
    api_key: Option<String>,
    router_config: &RouterConfig,
) -> WorkerSpec {
    let mut spec = WorkerSpec::new(url);
    spec.runtime_type = RuntimeType::External;
    spec.api_key = api_key;
    // Health config is resolved at worker build time from router
    // defaults + per-worker overrides (spec.health). No need to
    // set spec.health here since these workers have no overrides.
    spec.max_connection_attempts = router_config
        .connection_attempt_budget(RouterConfig::CONFIGURED_WORKER_ATTEMPTS_PER_SUCCESS);
    spec
}

#[cfg(test)]
pub(crate) fn recording_for_tests(
    context: &Arc<AppContext>,
) -> (Arc<JobQueue>, impl FnMut() -> Job) {
    use queue::testing::QueueTest;

    let (queue, mut next_job) = QueueTest::recording(context);
    (queue, move || next_job().expect("a queued job"))
}

#[cfg(test)]
#[path = "../../tests/unit/workflow/job_queue/tests.rs"]
mod tests;
