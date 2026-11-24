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
mod tests {
    use std::sync::{atomic::AtomicBool, Weak};

    use dashmap::DashMap;
    use openai_protocol::worker::JobStatus;
    use smg_worker::{
        identity::basic::Worker,
        jobs::{
            queue::testing::{Kept, Outcomes, Waiter, WaiterSlot},
            WorkerJobs,
        },
    };
    use tokio::sync::{mpsc, oneshot};

    use super::*;
    use crate::{
        app_context::testing::create_test_app_context,
        wasm::{WasmModuleDescriptor, WasmModuleType},
    };

    type QueueTest<'a> = queue::testing::QueueTest<'a, AppContext>;

    async fn execute_job(
        job: &Job,
        context: &Arc<AppContext>,
        cancelled_before_start: &DashSet<WorkerId>,
        own_id: Option<&WorkerId>,
    ) -> Result<String, String> {
        context
            .execute(
                job,
                &JobRun {
                    cancelled_before_start,
                    own_id,
                    ended_by_delete: &AtomicBool::new(false),
                },
            )
            .await
    }

    #[tokio::test]
    async fn a_tokenizer_job_notes_its_load_when_it_is_accepted() {
        use crate::workflow::TokenizerLoadFailure;

        let context = create_test_app_context();
        let (queue, mut next_job) = QueueTest::recording(&context);
        let failures = &context.tokenizer_load_failures;
        let name = "model";
        let job = Job::AddTokenizer {
            config: Box::new(TokenizerConfigRequest {
                id: "tokenizer-id".to_string(),
                name: name.to_string(),
                source: "tokenizer.json".to_string(),
                chat_template_path: None,
                cache_config: None,
                fail_on_duplicate: false,
                for_worker: false,
            }),
        };

        assert!(failures.record(name, TokenizerLoadFailure::NoSource));
        queue.submit(job.clone()).await.unwrap();
        assert!(matches!(next_job(), Some(Job::AddTokenizer { .. })));
        assert!(failures.get(name).is_none());
        assert!(!failures.record(name, TokenizerLoadFailure::NoSource));
        assert!(failures.load_ended(name), "one load was counted");

        drop(next_job);
        assert!(failures.record(name, TokenizerLoadFailure::NoSource));
        assert_eq!(
            queue.submit(job).await,
            Err("Job queue closed: its dispatcher stopped".to_string()),
            "the receiver was closed"
        );
        assert_eq!(failures.get(name), Some(TokenizerLoadFailure::NoSource));
    }

    #[tokio::test]
    async fn a_failed_create_releases_its_reservation_before_answering_its_waiter() {
        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        let url = "http://worker:8000";
        let id = context.worker_registry.reserve_id_for_url(url);
        let outcome = queue
            .submit_and_wait(
                Job::AddWorker {
                    config: Box::new(WorkerSpec::new(url)),
                    registration_mode: WorkerRegistrationMode::CreateOnly,
                    asker: WorkerJobAsker::Api,
                },
                Duration::from_secs(5),
            )
            .await;
        assert_eq!(
            outcome,
            Err(JobOutcomeError::Failed(
                "Workflow engines not initialized".to_string()
            ))
        );
        assert!(context.worker_registry.get_url_by_id(&id).is_none());
        assert_eq!(queue.get_status(url).unwrap().status, "failed");
    }

    #[tokio::test]
    async fn a_failed_registration_reads_failed_with_its_reason_at_its_id() {
        use axum::response::IntoResponse;

        use crate::{
            admin::worker_service::{GetWorkerResponse, WorkerService},
            endpoints::tokenize::get_tokenizer_status,
        };

        // A read as its client gets it: the address takes its shown form
        // where the answer is written.
        async fn written(read: GetWorkerResponse) -> serde_json::Value {
            let bytes = axum::body::to_bytes(read.into_response().into_body(), usize::MAX)
                .await
                .expect("a body");
            serde_json::from_slice(&bytes).expect("a JSON body")
        }

        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        context.worker_job_queue.set(Arc::clone(&queue)).unwrap();
        let service = WorkerService::new(
            Arc::clone(&context.worker_registry),
            Arc::clone(&context.worker_job_queue),
            context.router_config.clone(),
        );
        let url = "http://user:pass@worker:8000";
        let fail = || async {
            let id = context.worker_registry.reserve_id_for_url(url);
            let _ = queue
                .submit_and_wait(
                    Job::AddWorker {
                        config: Box::new(WorkerSpec::new(url)),
                        registration_mode: WorkerRegistrationMode::CreateOnly,
                        asker: WorkerJobAsker::Api,
                    },
                    Duration::from_secs(5),
                )
                .await;
            id
        };

        let id = fail().await;
        let read = service
            .get_worker(id.as_str())
            .expect("the id of a failed registration answers while its status is kept");
        // The result holds the address as registered until the answer is written.
        assert_eq!(read.0.spec.url, url);
        let read = written(read).await;
        assert_eq!(read["job_status"]["status"], "failed");
        assert_eq!(
            read["job_status"]["message"],
            "Workflow engines not initialized"
        );
        // No plain address: the answer writes it as the job's own lines write it.
        assert_eq!(read["url"], Endpoint::redacted_str(url));
        assert_eq!(read["job_status"]["worker_url"], read["url"]);
        // The tokenizer status route takes the status of a tokenizer job alone.
        let at_tokenizers = get_tokenizer_status(&context, id.as_str()).await;
        assert_eq!(at_tokenizers.status().as_u16(), 404);

        // A later failure of the same URL replaces the kept one.
        let later = fail().await;
        assert!(service.get_worker(later.as_str()).is_ok());
        assert!(service.get_worker(id.as_str()).is_err());

        // The pending view of a reserved id writes the address in that form too.
        let held = context.worker_registry.reserve_id_for_url(url);
        assert_eq!(
            written(service.get_worker(held.as_str()).unwrap()).await["url"],
            read["url"]
        );

        // A plain address reads as registered, byte for byte.
        let plain = "http://worker:8000/";
        let plain_id = context.worker_registry.reserve_id_for_url(plain);
        let _ = queue
            .submit_and_wait(
                Job::AddWorker {
                    config: Box::new(WorkerSpec::new(plain)),
                    registration_mode: WorkerRegistrationMode::CreateOnly,
                    asker: WorkerJobAsker::Api,
                },
                Duration::from_secs(5),
            )
            .await;
        assert_eq!(
            written(service.get_worker(plain_id.as_str()).unwrap()).await["url"],
            plain
        );
    }

    #[tokio::test]
    async fn the_port_carries_the_callers_asker_into_the_job() {
        let context = create_test_app_context();
        let (queue, mut next_job) = recording_for_tests(&context);
        let jobs: &dyn WorkerJobs = &*queue;
        let asker = WorkerJobAsker::Discovery;
        let id = WorkerId::from_string("worker-id".to_string());
        assert_eq!(
            jobs.add_worker(
                WorkerSpec::new("http://add:8000"),
                WorkerRegistrationMode::Upsert,
                asker.clone(),
            )
            .await,
            Ok(true)
        );
        assert_eq!(
            jobs.refresh_worker_labels("http://refresh:8000".to_string(), asker.clone())
                .await,
            Ok(true)
        );
        jobs.remove_worker("http://remove:8000".to_string(), &id, 7, asker.clone())
            .await
            .unwrap();
        jobs.remove_workers_at("http://remove-many:8000".to_string(), vec![(id, 7)], asker)
            .await
            .unwrap();

        assert!(matches!(
            next_job(),
            Job::AddWorker {
                asker: WorkerJobAsker::Discovery,
                ..
            }
        ));
        assert!(matches!(
            next_job(),
            Job::RefreshWorkerLabels {
                asker: WorkerJobAsker::Discovery,
                ..
            }
        ));
        for url in ["http://remove:8000", "http://remove-many:8000"] {
            assert!(
                matches!(
                    next_job(),
                    Job::RemoveWorker { url: queued_url, asker: WorkerJobAsker::Discovery, .. } if queued_url == url
                ),
                "{url}"
            );
        }
    }

    #[tokio::test]
    async fn the_startup_list_asks_as_startup() {
        let context = create_test_app_context();
        let (queue, mut next_job) = recording_for_tests(&context);
        context.worker_job_queue.set(queue).unwrap();
        let marks = DashSet::new();
        let url = "http://worker:8000";
        for mode in [
            RoutingMode::Regular {
                worker_urls: vec![url.to_string()],
            },
            RoutingMode::OpenAI {
                worker_urls: vec![url.to_string()],
            },
        ] {
            let job = Job::InitializeWorkersFromConfig {
                router_config: Box::new(RouterConfig {
                    mode,
                    ..RouterConfig::default()
                }),
            };
            execute_job(&job, &context, &marks, None).await.unwrap();
            assert!(matches!(
                next_job(),
                Job::AddWorker {
                    asker: WorkerJobAsker::Startup,
                    ..
                }
            ));
        }
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn kept_removals_of_one_asker_merge_and_of_two_stay_apart() {
        let context = create_test_app_context();
        let kept = DashMap::new();
        let status = Arc::new(DashMap::new());
        let url = "http://worker:8000";
        let removal = |asker, id, revision| Job::RemoveWorker {
            url: url.to_string(),
            expected_revisions: Some(pins(&[(id, Some(revision))])),
            asker,
        };
        QueueTest::keep(
            &kept,
            url,
            Job::RefreshWorkerLabels {
                url: url.to_string(),
                asker: WorkerJobAsker::Health,
            },
        );
        QueueTest::keep(&kept, url, removal(WorkerJobAsker::Health, "first", 1));
        QueueTest::keep(&kept, url, removal(WorkerJobAsker::Api, "second", 2));
        QueueTest::keep(&kept, url, removal(WorkerJobAsker::Health, "third", 3));
        // This one merges behind the health manager's removal and keeps its place.
        QueueTest::keep(&kept, url, removal(WorkerJobAsker::Api, "fourth", 4));

        assert!(QueueTest::take_next(&kept, url, |_| false).is_none());
        {
            let slots = kept.get(url).unwrap();
            assert_eq!(slots.removals.len(), 2);
            assert!(matches!(
                &slots.removals[0],
                Job::RemoveWorker { asker: WorkerJobAsker::Health, expected_revisions: Some(guards), .. }
                    if *guards == pins(&[("first", Some(1)), ("third", Some(3))])
            ));
            assert!(matches!(
                &slots.removals[1],
                Job::RemoveWorker { asker: WorkerJobAsker::Api, expected_revisions: Some(guards), .. }
                    if *guards == pins(&[("second", Some(2)), ("fourth", Some(4))])
            ));
        }

        QueueTest::run_kept(url, &context, &status, &kept, &DashSet::new()).await;
        assert!(kept.is_empty());
        logs_assert(|lines| {
            let starts: Vec<_> = lines
                .iter()
                .filter(|line| line.contains(" job kept for worker=http://worker:8000 "))
                .collect();
            let expected = [
                ("RemoveWorker", "Health"),
                ("RemoveWorker", "Api"),
                ("RefreshWorkerLabels", "Health"),
            ];
            if starts.len() == expected.len()
                && starts.iter().zip(expected).all(|(line, (job, asker))| {
                    line.contains(&format!("Running the {job} job kept"))
                        && line.contains(&format!("asker={asker}"))
                })
            {
                Ok(())
            } else {
                Err(format!("kept job starts in the wrong order: {starts:?}"))
            }
        });
    }

    #[tokio::test(start_paused = true)]
    async fn a_run_reads_its_jobs_asker() {
        use wfaas::{
            FunctionStep, StepDefinition, StepResult, WorkflowContext, WorkflowDefinition,
        };

        fn record_asker<D: WorkflowData>(
            engine: &WorkflowEngine<D>,
            asker_of: fn(&D) -> &WorkerJobAsker,
        ) -> mpsc::UnboundedReceiver<WorkerJobAsker> {
            let (tx, rx) = mpsc::unbounded_channel();
            let step = FunctionStep::new(move |context: &mut WorkflowContext<D>| {
                let _ = tx.send(asker_of(&context.data).clone());
                Box::pin(async { Ok(StepResult::Success) })
            });
            engine
                .register_workflow(
                    WorkflowDefinition::new(D::workflow_type(), "Read the asker").add_step(
                        StepDefinition::new("read_asker", "Read the asker", Arc::new(step)),
                    ),
                )
                .unwrap();
            rx
        }

        let context = create_test_app_context();
        let (queue, _next_job) = recording_for_tests(&context);
        context.worker_job_queue.set(Arc::clone(&queue)).unwrap();
        let engines = WorkflowEngines::new(&context.router_config);
        let mut registrations = record_asker(&engines.worker_registration, |data| &data.asker);
        let mut removals = record_asker(&engines.worker_removal, |data| &data.asker);
        let mut updates = record_asker(&engines.worker_update, |data| &data.asker);
        context.workflow_engines.set(engines).unwrap();
        let url = "http://worker:8000";
        let jobs = [
            (
                Job::AddWorker {
                    config: Box::new(WorkerSpec::new(url)),
                    registration_mode: WorkerRegistrationMode::Upsert,
                    asker: WorkerJobAsker::Discovery,
                },
                WorkerJobAsker::Discovery,
            ),
            (
                Job::RemoveWorker {
                    url: url.to_string(),
                    expected_revisions: None,
                    asker: WorkerJobAsker::Health,
                },
                WorkerJobAsker::Health,
            ),
            (
                Job::UpdateWorker {
                    url: url.to_string(),
                    update: Box::new(serde_json::from_str("{}").unwrap()),
                    asker: WorkerJobAsker::Startup,
                },
                WorkerJobAsker::Startup,
            ),
            (
                Job::RefreshWorkerLabels {
                    url: url.to_string(),
                    asker: WorkerJobAsker::Health,
                },
                WorkerJobAsker::Health,
            ),
        ];
        let marks = DashSet::new();
        for (job, expected) in jobs {
            if matches!(job, Job::RemoveWorker { .. }) {
                QueueTest::hold_url_for_tests(&queue, url, "RemoveWorker");
            }
            execute_job(&job, &context, &marks, None).await.unwrap();
            let read = match job {
                Job::AddWorker { .. } => registrations.try_recv(),
                Job::RemoveWorker { .. } => {
                    assert_eq!(queue.get_status(url).unwrap().status, "processing");
                    removals.try_recv()
                }
                _ => updates.try_recv(),
            };
            assert_eq!(read, Ok(expected), "{}", job.job_type());
        }
    }

    #[tokio::test]
    async fn a_cancelled_registration_cleans_up_for_the_delete_and_a_failed_one_for_its_caller() {
        use smg_worker::workflow::single_attempt;
        use tokio::sync::Notify;
        use wfaas::{
            FunctionStep, StepDefinition, StepResult, WorkflowContext, WorkflowDefinition,
        };

        use crate::{
            routers::grpc::backend_client::BasicWorkerBuilder, workflow::WorkerWorkflowData,
        };

        let context = create_test_app_context();
        let (queue, mut next_job) = recording_for_tests(&context);
        context.worker_job_queue.set(queue).unwrap();
        let engines = WorkflowEngines::new(&context.router_config);
        let engine = Arc::clone(&engines.worker_registration);
        context.workflow_engines.set(engines).unwrap();

        for (url, cancelled, expected) in [
            ("http://cancelled:8000", true, WorkerJobAsker::Api),
            ("http://failed:8000", false, WorkerJobAsker::Discovery),
        ] {
            let id = context.worker_registry.reserve_id_for_url(url);
            let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new(url).build());
            let reached = Arc::new(Notify::new());
            let release = Arc::new(Notify::new());
            let step = {
                let registry = Arc::clone(&context.worker_registry);
                let reached = Arc::clone(&reached);
                let release = Arc::clone(&release);
                FunctionStep::new(move |run: &mut WorkflowContext<WorkerWorkflowData>| {
                    run.data.actual_workers = Some(vec![Arc::clone(&worker)]);
                    let registry = Arc::clone(&registry);
                    let release = Arc::clone(&release);
                    let worker = Arc::clone(&worker);
                    reached.notify_one();
                    Box::pin(async move {
                        release.notified().await;
                        registry.register(worker).unwrap();
                        Ok(StepResult::Failure)
                    })
                })
            };
            engine
                .register_workflow(
                    WorkflowDefinition::new("worker_registration", "Write the worker").add_step(
                        StepDefinition::new("write_worker", "Write the worker", Arc::new(step))
                            .with_retry(single_attempt()),
                    ),
                )
                .unwrap();
            let job = Job::AddWorker {
                config: Box::new(WorkerSpec::new(url)),
                registration_mode: WorkerRegistrationMode::CreateOnly,
                asker: WorkerJobAsker::Discovery,
            };
            let marks = DashSet::new();
            let finish_write = async {
                tokio::time::timeout(Duration::from_secs(5), reached.notified())
                    .await
                    .unwrap();
                if cancelled {
                    context
                        .worker_service
                        .delete_worker(id.as_str())
                        .await
                        .unwrap();
                }
                release.notify_one();
            };
            let (result, ()) =
                tokio::join!(execute_job(&job, &context, &marks, Some(&id)), finish_write,);
            assert!(result
                .unwrap_err()
                .contains("leave through a queued removal"));
            assert!(
                matches!(
                    next_job(),
                    Job::RemoveWorker { asker, expected_revisions: Some(_), .. } if asker == expected
                ),
                "{url}"
            );
        }
    }

    /// A job line writes a worker's address masked and a non-address name as
    /// it is; a job type the list does not name counts as a worker's, so a
    /// type added later cannot write a credential into the six job lines.
    #[test]
    fn a_job_line_masks_every_address_and_keeps_the_names() {
        let credentialed = "http://user:secret@127.0.0.1:8000";
        for job_type in [
            "AddWorker",
            "UpdateWorker",
            "RefreshWorkerLabels",
            "RemoveWorker",
            "ReplaceWorker",
        ] {
            let shown = Job::shown_worker(job_type, credentialed);
            assert!(!shown.contains("secret"), "{job_type}: {shown}");
            assert!(shown.contains("***@127.0.0.1:8000"), "{job_type}: {shown}");
        }
        for (job_type, name) in [
            ("InitializeWorkersFromConfig", "startup"),
            ("InitializeMcpServers", "startup"),
            ("RegisterMcpServer", "search-tools"),
            ("AddWasmModule", "my_module"),
            ("RemoveWasmModule", "0f1e2d3c"),
            ("AddTokenizer", "org/tokenizer"),
            ("RemoveTokenizer", "org/tokenizer"),
        ] {
            assert_eq!(Job::shown_worker(job_type, name), name, "{job_type}");
        }
    }

    /// A start-up registration the queue refuses names its worker with the
    /// address masked: the refusal's text is the failed job's warning. A
    /// queue whose context is gone refuses every submission; both fleets are
    /// submitted, the local workers and an external provider's endpoints.
    #[tokio::test]
    #[tracing_test::traced_test]
    async fn a_refused_start_up_submission_names_its_worker_masked() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        let refusing = JobQueue::new(JobQueueConfig::default(), Weak::new());
        assert!(context.worker_job_queue.set(refusing).is_ok());
        let worker_urls = vec!["http://user:s3cret@10.0.0.7:8000/?token=s3cret".to_string()];
        let marks = DashSet::new();
        let statuses = Arc::new(DashMap::new());
        for mode in [
            RoutingMode::Regular {
                worker_urls: worker_urls.clone(),
            },
            RoutingMode::OpenAI { worker_urls },
        ] {
            let job = Job::InitializeWorkersFromConfig {
                router_config: Box::new(RouterConfig::builder().mode(mode).build_unchecked()),
            };
            let result = execute_job(&job, &context, &marks, None).await;
            assert!(result.is_err(), "the queue took the submission: {result:?}");
            QueueTest::record_job_completion(
                job.job_type(),
                job.worker_url(),
                Duration::ZERO,
                &result,
                &statuses,
            );
        }
        logs_assert(|lines: &[&str]| {
            let failed: Vec<_> = lines
                .iter()
                .filter(|line| line.contains("Failed job:"))
                .collect();
            let unmasked: Vec<_> = failed
                .iter()
                .filter(|line| {
                    line.contains("s3cret") || !line.contains("http://***@10.0.0.7:8000")
                })
                .collect();
            if failed.len() == 2 && unmasked.is_empty() {
                return Ok(());
            }
            Err(format!(
                "{} failed-job lines, wanted two; those that do not name their worker masked: \
                 {unmasked:#?}",
                failed.len()
            ))
        });
    }

    /// The engine the registrations of the next case reach: the health probe
    /// and the metadata a registration reads, a 404 elsewhere. Its
    /// `host:port`.
    async fn a_registering_engine() -> String {
        use http::StatusCode;
        use serde_json::json;

        use crate::workflow::steps::local::test_support::serve_json_routes;
        let base = serve_json_routes(vec![
            ("/health", StatusCode::OK, json!({"status": "ok"})),
            (
                "/model_info",
                StatusCode::OK,
                json!({"model_path": "m", "is_generation": true}),
            ),
            ("/server_info", StatusCode::OK, json!({"version": "test"})),
        ])
        .await;
        base.trim_start_matches("http://").to_string()
    }

    /// The text a worker job's wait ends with names the worker by its address
    /// masked: the engine builds it with the workflow's label, and it is the
    /// job's result, in its line and in its status. One job of each kind that
    /// waits under a worker's address. Two workers whose addresses differ in
    /// the userinfo alone register side by side and go by one label; each is
    /// found under its own address, so the update, the label refresh and the
    /// removal of the first leave the second registered.
    #[tokio::test]
    #[tracing_test::traced_test]
    async fn a_worker_jobs_result_names_its_worker_masked() {
        use crate::app_context::testing::live_app_context;
        let host = a_registering_engine().await;
        let first = format!("http://alice:s3cret-a@{host}");
        let second = format!("http://bob:s3cret-b@{host}");
        let context = live_app_context();
        let registry = &context.worker_registry;
        let add = |url: &str| Job::AddWorker {
            asker: WorkerJobAsker::Api,
            config: Box::new(WorkerSpec::new(url)),
            registration_mode: WorkerRegistrationMode::Upsert,
        };
        let update: WorkerUpdateRequest =
            serde_json::from_value(serde_json::json!({"labels": {"tier": "a"}}))
                .expect("an update");
        let jobs = [
            add(&first),
            add(&second),
            Job::UpdateWorker {
                asker: WorkerJobAsker::Api,
                url: first.clone(),
                update: Box::new(update),
            },
            Job::RefreshWorkerLabels {
                url: first.clone(),
                asker: WorkerJobAsker::Api,
            },
            Job::RemoveWorker {
                asker: WorkerJobAsker::Api,
                url: first.clone(),
                expected_revisions: None,
            },
        ];
        let done = format!("http://***@{host} completed successfully via workflow");
        let marks = DashSet::new();
        let statuses = Arc::new(DashMap::new());
        for job in &jobs {
            if matches!(job, Job::UpdateWorker { .. }) {
                assert!(
                    registry.get_by_url(&first).is_some() && registry.get_by_url(&second).is_some(),
                    "both registered, each under its own address"
                );
            }
            let result = execute_job(job, &context, &marks, None).await;
            assert_eq!(result.as_deref(), Ok(done.as_str()), "{}", job.job_type());
            QueueTest::record_job_completion(
                job.job_type(),
                job.worker_url(),
                Duration::ZERO,
                &result,
                &statuses,
            );
        }
        assert!(
            registry.get_by_url(&first).is_none(),
            "the removal found the first worker by its address"
        );
        assert!(
            registry.get_by_url(&second).is_some(),
            "the second worker, apart by its userinfo alone, stays registered"
        );
        logs_assert(|lines: &[&str]| {
            let completed = lines
                .iter()
                .filter(|line| line.contains("Completed job:"))
                .count();
            let masked = lines
                .iter()
                .filter(|line| line.contains(&format!("result={done}")))
                .count();
            let credentialed: Vec<_> = lines
                .iter()
                .filter(|line| line.contains("s3cret"))
                .collect();
            if completed == 5 && masked == 5 && credentialed.is_empty() {
                return Ok(());
            }
            Err(format!(
                "{completed} completed-job lines, {masked} with the masked result, wanted five \
                 of five; the lines with a credential: {credentialed:#?}"
            ))
        });
    }

    /// A gateway given no setting for it registers one worker per
    /// data-parallel rank an engine names. An HTTP engine whose server info
    /// names four ranks stands as four rank workers and none at its base
    /// address. An engine that names one rank, one that names no size and
    /// one whose server info fails to read each stand as one plain worker,
    /// and none of the four registrations fails.
    #[tokio::test]
    async fn a_registration_registers_one_worker_per_rank_the_engine_names() {
        use http::StatusCode;
        use serde_json::json;
        use smg_worker::workflow::find_workers_by_url;

        use crate::{
            app_context::testing::live_app_context,
            workflow::steps::local::test_support::serve_json_routes,
        };

        let context = live_app_context();
        let registry = &context.worker_registry;
        let marks = DashSet::new();
        for (status, server_info, ranks) in [
            (StatusCode::OK, json!({"dp_size": 2, "attn_dp_size": 2}), 4),
            (StatusCode::OK, json!({"dp_size": 1}), 1),
            (StatusCode::OK, json!({"version": "test"}), 1),
            (StatusCode::INTERNAL_SERVER_ERROR, json!({"dp_size": 4}), 1),
        ] {
            // The model list names the runtime whose server info reports the
            // size, so each of the four is read the same way.
            let base = serve_json_routes(vec![
                ("/health", StatusCode::OK, json!({"status": "ok"})),
                (
                    "/v1/models",
                    StatusCode::OK,
                    json!({"data": [{"id": "m", "owned_by": "sglang"}]}),
                ),
                (
                    "/model_info",
                    StatusCode::OK,
                    json!({"model_path": "m", "is_generation": true}),
                ),
                ("/server_info", status, server_info.clone()),
            ])
            .await;
            let job = Job::AddWorker {
                asker: WorkerJobAsker::Api,
                config: Box::new(WorkerSpec::new(base.as_str())),
                registration_mode: WorkerRegistrationMode::Upsert,
            };

            let result = execute_job(&job, &context, &marks, None).await;

            let case = format!("{status} {server_info}");
            assert!(result.is_ok(), "{case}: {result:?}");
            let mut workers = find_workers_by_url(registry, &base);
            workers.sort_by_key(|worker| worker.dp_rank());
            let registered: Vec<(String, Option<usize>, Option<usize>)> = workers
                .iter()
                .map(|worker| (worker.url().to_string(), worker.dp_rank(), worker.dp_size()))
                .collect();
            let expected: Vec<(String, Option<usize>, Option<usize>)> = if ranks == 1 {
                vec![(base.clone(), None, None)]
            } else {
                (0..ranks)
                    .map(|rank| (format!("{base}@{rank}"), Some(rank), Some(ranks)))
                    .collect()
            };
            assert_eq!(registered, expected, "{case}");
            assert_eq!(
                registry.get_by_url(&base).is_some(),
                ranks == 1,
                "{case}: a worker stands at the base address only where the engine is one worker"
            );
        }
    }

    /// A second `POST` of an engine that stands as rank workers is refused as
    /// a duplicate though its server info no longer reads. The failed read
    /// leaves the registration no shape, and the rank workers stand at
    /// `{base}@{rank}`, where the base address finds none. After it the two
    /// rank workers stand alone, with no plain worker beside them.
    #[tokio::test]
    async fn a_repeated_create_is_refused_for_rank_workers_whose_server_info_fails() {
        use std::sync::atomic::Ordering;

        use http::StatusCode;
        use serde_json::json;

        use crate::app_context::testing::live_app_context;

        let (base, failing) = engine_whose_server_info_turns(
            (StatusCode::OK, json!({"dp_size": 2})),
            (StatusCode::INTERNAL_SERVER_ERROR, json!({})),
        )
        .await;
        let context = live_app_context();
        let marks = DashSet::new();
        let create = add_worker(&base, WorkerRegistrationMode::CreateOnly);
        let ranks = vec![format!("{base}@0"), format!("{base}@1")];

        let first = execute_job(&create, &context, &marks, None).await;
        assert!(first.is_ok(), "{first:?}");
        assert_eq!(standing_at(&context, &base), ranks);

        failing.store(true, Ordering::Relaxed);
        let second = execute_job(&create, &context, &marks, None).await;
        assert!(
            matches!(&second, Err(said) if said.contains("already exists")),
            "{second:?}"
        );
        assert_eq!(
            standing_at(&context, &base),
            ranks,
            "no plain worker beside the ranks"
        );
    }

    /// An upsert that reads two ranks where the engine stands as one plain
    /// worker is refused by name: the first registration's server info
    /// failed to read. The register step replaces a worker under its own
    /// address alone, so the rank workers would stand beside the plain one.
    /// After the refusal the plain worker stands alone.
    #[tokio::test]
    async fn an_upsert_that_reads_ranks_is_refused_where_a_plain_worker_stands() {
        use std::sync::atomic::Ordering;

        use http::StatusCode;
        use serde_json::json;

        use crate::app_context::testing::live_app_context;

        let (base, reads) = engine_whose_server_info_turns(
            (StatusCode::INTERNAL_SERVER_ERROR, json!({})),
            (StatusCode::OK, json!({"dp_size": 2})),
        )
        .await;
        let context = live_app_context();
        let marks = DashSet::new();
        let upsert = add_worker(&base, WorkerRegistrationMode::Upsert);
        let plain = vec![base.clone()];

        let first = execute_job(&upsert, &context, &marks, None).await;
        assert!(first.is_ok(), "{first:?}");
        assert_eq!(standing_at(&context, &base), plain);

        reads.store(true, Ordering::Relaxed);
        let second = execute_job(&upsert, &context, &marks, None).await;
        assert!(
            matches!(&second, Err(said) if said.contains(
                "stands as one worker and this registration read 2 data-parallel ranks"
            )),
            "{second:?}"
        );
        assert_eq!(
            standing_at(&context, &base),
            plain,
            "no rank worker beside the plain one"
        );
    }

    /// An upsert takes a peer's rank workers that stand at its addresses
    /// though they state no rank: an import with no spec is built from the
    /// address alone. The registration reads the same two ranks and claims
    /// both under their addresses. After it the registry holds the engine
    /// once, each worker states its rank, and this node owns both.
    #[tokio::test]
    async fn an_upsert_takes_a_peers_rank_workers_that_state_no_rank() {
        use http::StatusCode;
        use serde_json::json;
        use smg_worker::{
            registry::{RemoteWorkerState, WorkerOrigin},
            workflow::find_workers_by_url,
        };

        use crate::app_context::testing::live_app_context;

        let two_ranks = (StatusCode::OK, json!({"dp_size": 2}));
        let (base, _) = engine_whose_server_info_turns(two_ranks.clone(), two_ranks).await;
        let context = live_app_context();
        let registry = &context.worker_registry;
        for rank in 0..2 {
            let url = format!("{base}@{rank}");
            registry.on_remote_worker_state(&RemoteWorkerState {
                worker_id: format!("peer-rank-{rank}"),
                model_id: "m".to_string(),
                url: url.clone(),
                health: true,
                version: 1,
                spec: Vec::new(),
            });
            let id = registry.get_id_by_url(&url).expect("the import");
            assert_eq!(registry.origin_of(&id), Some(WorkerOrigin::Mesh));
            let imported = registry.get(&id).expect("the import");
            assert_eq!(imported.dp_rank(), None, "built from the address alone");
        }
        let marks = DashSet::new();
        let upsert = add_worker(&base, WorkerRegistrationMode::Upsert);

        let result = execute_job(&upsert, &context, &marks, None).await;

        assert!(result.is_ok(), "{result:?}");
        let mut standing: Vec<(String, Option<usize>, Option<WorkerOrigin>)> =
            find_workers_by_url(registry, &base)
                .iter()
                .map(|worker| {
                    let id = registry.get_id_by_url(worker.url()).expect("its id");
                    (
                        worker.url().to_string(),
                        worker.dp_rank(),
                        registry.origin_of(&id),
                    )
                })
                .collect();
        standing.sort_by(|a, b| a.0.cmp(&b.0));
        let local = Some(WorkerOrigin::Local);
        assert_eq!(
            standing,
            [
                (format!("{base}@0"), Some(0), local),
                (format!("{base}@1"), Some(1), local)
            ]
        );
    }

    /// An engine stub whose server info answers `first` until the switch it
    /// returns is set, and `then` from then on: a read that fails, or one
    /// that comes back. Returns its base address and the switch.
    async fn engine_whose_server_info_turns(
        first: (http::StatusCode, serde_json::Value),
        then: (http::StatusCode, serde_json::Value),
    ) -> (String, Arc<AtomicBool>) {
        use std::sync::atomic::Ordering;

        use axum::{routing::get, Json};
        use http::StatusCode;
        use serde_json::json;

        use crate::workflow::steps::local::test_support::{json_router, serve_router};

        let turned = Arc::new(AtomicBool::new(false));
        let server_info = {
            let turned = Arc::clone(&turned);
            move || {
                let (status, body) = if turned.load(Ordering::Relaxed) {
                    then.clone()
                } else {
                    first.clone()
                };
                async move { (status, Json(body)) }
            }
        };
        let router = json_router(vec![
            ("/health", StatusCode::OK, json!({"status": "ok"})),
            (
                "/v1/models",
                StatusCode::OK,
                json!({"data": [{"id": "m", "owned_by": "sglang"}]}),
            ),
            (
                "/model_info",
                StatusCode::OK,
                json!({"model_path": "m", "is_generation": true}),
            ),
        ])
        .route("/server_info", get(server_info));
        (serve_router(router).await, turned)
    }

    /// A registration of the engine at `base` in `registration_mode`, as the
    /// worker API asks it.
    fn add_worker(base: &str, registration_mode: WorkerRegistrationMode) -> Job {
        Job::AddWorker {
            asker: WorkerJobAsker::Api,
            config: Box::new(WorkerSpec::new(base)),
            registration_mode,
        }
    }

    /// The addresses of the workers that stand for `base` in the registry,
    /// the plain one and the rank workers, in order.
    fn standing_at(context: &AppContext, base: &str) -> Vec<String> {
        let mut urls: Vec<String> =
            smg_worker::workflow::find_workers_by_url(&context.worker_registry, base)
                .iter()
                .map(|worker| worker.url().to_string())
                .collect();
        urls.sort();
        urls
    }

    /// The registration workflows the store holds that have not ended.
    async fn registrations_running(context: &AppContext) -> usize {
        context
            .workflow_engines
            .get()
            .expect("the engines")
            .worker_registration
            .state_store()
            .list_all()
            .await
            .expect("the registration states")
            .iter()
            .filter(|state| {
                matches!(
                    state.status,
                    WorkflowStatus::Pending | WorkflowStatus::Running
                )
            })
            .count()
    }

    /// A registration cancelled in its wait ends with a text that names its
    /// worker masked: the failed job's warning, at the default level, and the
    /// message of the status the worker API answers. The cancel finds a
    /// registration by its address, not by its label: of two whose addresses
    /// differ in the userinfo alone, on a port nothing answers, the first
    /// cancel ends one and leaves the other running, and each ends under its
    /// own status.
    #[tokio::test]
    #[tracing_test::traced_test]
    async fn a_cancelled_registrations_warning_and_status_name_its_worker_masked() {
        use smg_worker::test_support::RefusedAddress;

        use crate::app_context::testing::live_app_context;
        // Held for the whole case: no other case's listener can draw the
        // port and answer a registration's probe.
        let refused = RefusedAddress::reserve();
        let port = refused.port();
        let first = format!("http://alice:s3cret-a@127.0.0.1:{port}");
        let second = format!("http://bob:s3cret-b@127.0.0.1:{port}");
        let context = live_app_context();
        let add = |url: &str| Job::AddWorker {
            asker: WorkerJobAsker::Api,
            config: Box::new(WorkerSpec::new(url)),
            registration_mode: WorkerRegistrationMode::CreateOnly,
        };
        let jobs = [add(&first), add(&second)];
        let marks = DashSet::new();
        let cancel_one_then_the_other = async {
            // Bounded: a registration that ended before the cancel is a red
            // case, never a wait without end.
            tokio::time::timeout(Duration::from_secs(30), async {
                while registrations_running(&context).await < 2 {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("two registrations running within the bound");
            let ended = cancel_registration_states(&context, &first).await;
            assert_eq!(ended, Ok((1, false)), "the first address ends its own");
            assert_eq!(
                registrations_running(&context).await,
                1,
                "the second, apart by its userinfo alone, still runs"
            );
            let ended = cancel_registration_states(&context, &second).await;
            assert_eq!(ended, Ok((1, false)), "the second address ends its own");
        };
        let [job_a, job_b] = &jobs;
        let (result_a, result_b, ()) = tokio::join!(
            execute_job(job_a, &context, &marks, None),
            execute_job(job_b, &context, &marks, None),
            cancel_one_then_the_other,
        );

        let cancelled = format!("Workflow cancelled for http://***@127.0.0.1:{port}");
        let statuses = Arc::new(DashMap::new());
        for (job, result) in [(job_a, &result_a), (job_b, &result_b)] {
            assert_eq!(result.as_ref().err(), Some(&cancelled));
            QueueTest::record_job_completion(
                job.job_type(),
                job.worker_url(),
                Duration::ZERO,
                result,
                &statuses,
            );
            let status = statuses
                .get(job.worker_url())
                .expect("each registration ends under its own status");
            assert_eq!(status.status, "failed");
            assert_eq!(status.message.as_deref(), Some(cancelled.as_str()));
        }
        assert_eq!(statuses.len(), 2, "one status an address");
        logs_assert(|lines: &[&str]| {
            let failed = lines
                .iter()
                .filter(|line| line.contains("Failed job:"))
                .count();
            let masked = lines
                .iter()
                .filter(|line| line.contains(&format!("error={cancelled}")))
                .count();
            let credentialed: Vec<_> = lines
                .iter()
                .filter(|line| line.contains("s3cret"))
                .collect();
            if failed == 2 && masked == 2 && credentialed.is_empty() {
                return Ok(());
            }
            Err(format!(
                "{failed} failed-job lines, {masked} with the masked text, wanted two of two; \
                 the lines with a credential: {credentialed:#?}"
            ))
        });
    }

    /// A wasm module job's key is the uuid the submitter assigned (the
    /// registration) or named (the removal): the in-flight claim, the status
    /// entry and the outcome waiter hang on it, so a second submission for
    /// one uuid is refused while the first runs, and the answer names the
    /// module the job registered.
    #[test]
    fn a_wasm_module_jobs_key_is_the_module_uuid() {
        let module_uuid = uuid::Uuid::now_v7();
        let descriptor = WasmModuleDescriptor {
            name: "twin".to_string(),
            file_path: "/tmp/twin.wasm".to_string(),
            module_type: WasmModuleType::Middleware,
            attach_points: Vec::new(),
            add_result: None,
        };
        let registration = Job::AddWasmModule {
            config: Box::new(WasmModuleConfigRequest::new(descriptor, module_uuid)),
        };
        assert_eq!(registration.worker_url(), module_uuid.to_string());
        let removal = Job::RemoveWasmModule {
            request: Box::new(WasmModuleRemovalRequest::new(module_uuid)),
        };
        assert_eq!(removal.worker_url(), module_uuid.to_string());
    }

    /// A caller's waiter leaves with the caller: the slot removes the entry
    /// under its key while the entry is still this caller's, and leaves a
    /// later caller's waiter under the same key alone.
    #[test]
    fn a_waiter_slot_removes_its_own_waiter_alone() {
        let outcomes: Outcomes = Arc::new(DashMap::new());
        let (first, _first_rx) = oneshot::channel();
        outcomes.insert(
            "job-1".to_string(),
            Waiter {
                id: 1,
                sender: first,
            },
        );
        drop(WaiterSlot {
            outcomes: Arc::clone(&outcomes),
            key: "job-1".to_string(),
            id: 1,
        });
        assert!(outcomes.is_empty(), "the caller's own waiter goes with it");

        let (later, _later_rx) = oneshot::channel();
        outcomes.insert(
            "job-1".to_string(),
            Waiter {
                id: 2,
                sender: later,
            },
        );
        drop(WaiterSlot {
            outcomes: Arc::clone(&outcomes),
            key: "job-1".to_string(),
            id: 1,
        });
        assert_eq!(
            outcomes.get("job-1").map(|waiter| waiter.id),
            Some(2),
            "a later waiter under the key stays"
        );
    }

    /// A second caller for a job in flight is refused and the first keeps
    /// its waiter, so the outcome reaches the caller that submitted the job.
    /// The refusal names the key as a log line writes it, its userinfo masked.
    #[tokio::test]
    async fn a_second_waiter_under_a_held_key_is_refused_and_the_first_kept() {
        let queue = JobQueue::new(JobQueueConfig::default(), Weak::new());
        let key = "http://user:secret@127.0.0.1:8000".to_string();
        let (first, first_rx) = oneshot::channel();
        QueueTest::state(&queue).outcomes.insert(
            key.clone(),
            Waiter {
                id: 1,
                sender: first,
            },
        );
        let second = queue
            .submit_and_wait(
                Job::RefreshWorkerLabels {
                    url: key.clone(),
                    asker: WorkerJobAsker::Api,
                },
                Duration::from_secs(1),
            )
            .await;
        let shown = Job::shown_worker("RefreshWorkerLabels", &key);
        assert!(!shown.contains("secret"), "{shown}");
        assert_eq!(second, Err(JobOutcomeError::InFlight(shown)));
        assert_eq!(
            QueueTest::state(&queue)
                .outcomes
                .get(&key)
                .map(|waiter| waiter.id),
            Some(1)
        );
        QueueTest::deliver_outcome(
            QueueTest::state(&queue).outcomes,
            &key,
            Some(1),
            Ok("done".to_string()),
        );
        assert!(matches!(first_rx.await, Ok(Ok(ref message)) if message == "done"));
    }

    /// A caller whose wait ended leaves the key free: a second wait under
    /// the same key is taken, carries the next waiter id, and is answered
    /// with its own job's outcome once the dispatcher runs. The test holds
    /// the dispatcher's one permit, so the queued jobs wait, and lets go of
    /// it once the second wait holds the key.
    #[tokio::test]
    async fn a_second_wait_under_a_freed_key_is_taken_and_answered_with_its_own_outcome() {
        use crate::app_context::testing::app_context_for_tests;

        let context = app_context_for_tests(
            RouterConfig::default(),
            Arc::new(WorkerRegistry::new()),
            None,
        );
        let queue = JobQueue::new(
            JobQueueConfig {
                queue_capacity: 1,
                max_concurrent_jobs: 1,
            },
            Arc::downgrade(&context),
        );
        let key = "http://worker:8000".to_string();
        let refresh = |url: &str| Job::RefreshWorkerLabels {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
        };
        let held = Arc::clone(QueueTest::state(&queue).concurrency_limit)
            .acquire_owned()
            .await
            .expect("the dispatcher's permit");

        // The first wait's job is taken off the channel and parks at the
        // permit; the wait ends on its bound and the key is free again.
        let first = queue
            .submit_and_wait(refresh(&key), Duration::from_millis(20))
            .await;
        assert!(
            matches!(first, Err(JobOutcomeError::TimedOut(_))),
            "{first:?}"
        );
        assert!(
            QueueTest::state(&queue).outcomes.is_empty(),
            "the first caller's waiter left with it"
        );

        // The second wait under the key is taken with the second id; once
        // the dispatcher runs, the first job's late outcome reaches nobody
        // and the second wait is answered by its own job.
        let second = queue.submit_and_wait(refresh(&key), Duration::from_secs(5));
        let release = async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert_eq!(
                QueueTest::state(&queue)
                    .outcomes
                    .get(&key)
                    .map(|waiter| waiter.id),
                Some(2),
                "the second wait holds the key with the next id"
            );
            drop(held);
        };
        let (second, ()) = tokio::join!(second, release);
        assert!(
            matches!(second, Err(JobOutcomeError::Failed(ref reason)) if reason.contains("not initialized")),
            "{second:?}"
        );
        assert!(QueueTest::state(&queue).outcomes.is_empty());
    }

    /// A finished job's outcome reaches the waiter under its key once, and
    /// the waiter is gone; a key without a waiter is a no-op.
    #[tokio::test]
    async fn an_outcome_reaches_its_waiter_once() {
        let outcomes: Outcomes = Arc::new(DashMap::new());
        let (tx, rx) = oneshot::channel();
        outcomes.insert("job-1".to_string(), Waiter { id: 1, sender: tx });
        QueueTest::deliver_outcome(&outcomes, "job-2", Some(1), Ok("stray".to_string()));
        QueueTest::deliver_outcome(&outcomes, "job-1", Some(1), Ok("done".to_string()));
        assert!(matches!(rx.await, Ok(Ok(ref message)) if message == "done"));
        assert!(outcomes.is_empty());
    }

    /// A job's outcome goes to the waiter it was queued for alone: after that
    /// waiter left and a later caller took the key, the late outcome reaches
    /// nobody, and a job queued without a waiter delivers nothing.
    #[tokio::test]
    async fn a_late_outcome_does_not_reach_a_later_waiter() {
        let outcomes: Outcomes = Arc::new(DashMap::new());
        let (tx, mut rx) = oneshot::channel();
        outcomes.insert("job-1".to_string(), Waiter { id: 2, sender: tx });
        QueueTest::deliver_outcome(
            &outcomes,
            "job-1",
            Some(1),
            Ok("the first job's".to_string()),
        );
        QueueTest::deliver_outcome(&outcomes, "job-1", None, Ok("a plain submit's".to_string()));
        assert_eq!(outcomes.get("job-1").map(|waiter| waiter.id), Some(2));
        assert!(matches!(
            rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        QueueTest::deliver_outcome(&outcomes, "job-1", Some(2), Ok("its own".to_string()));
        assert!(matches!(rx.await, Ok(Ok(ref message)) if message == "its own"));
        assert!(outcomes.is_empty());
    }

    /// A submission the queue refuses leaves no waiter behind and says why.
    #[tokio::test]
    async fn a_refused_submission_leaves_no_waiter() {
        let queue = JobQueue::new(JobQueueConfig::default(), Weak::new());
        let job = Job::RemoveWasmModule {
            request: Box::new(WasmModuleRemovalRequest::new(uuid::Uuid::now_v7())),
        };
        let outcome = queue.submit_and_wait(job, Duration::from_secs(1)).await;
        assert!(
            matches!(outcome, Err(JobOutcomeError::NotQueued(ref reason)) if reason.contains("shutting down")),
            "{outcome:?}"
        );
        assert!(QueueTest::state(&queue).outcomes.is_empty());
    }

    /// A caller leaving while its submit is parked on a slot (the queue full)
    /// takes its waiter with it: the waiter is under the key while the submit
    /// waits, gone when the caller is, and a later caller takes the key.
    #[tokio::test]
    async fn a_wait_cancelled_under_a_full_queue_leaves_no_waiter() {
        use crate::{
            app_context::testing::app_context_for_tests, routers::common::snapshot::WorkerRegistry,
        };

        let context = app_context_for_tests(
            RouterConfig::builder().build_unchecked(),
            Arc::new(WorkerRegistry::new()),
            None,
        );
        let queue = JobQueue::new(
            JobQueueConfig {
                queue_capacity: 1,
                max_concurrent_jobs: 1,
            },
            Arc::downgrade(&context),
        );
        // The queue's one slot, held for the whole test: every submit parks.
        let slot = QueueTest::reserve_slot(&queue)
            .await
            .expect("the queue is open");
        let key = "http://127.0.0.1:8000".to_string();

        let mut wait = Box::pin(queue.submit_and_wait(
            Job::RefreshWorkerLabels {
                url: key.clone(),
                asker: WorkerJobAsker::Api,
            },
            Duration::from_secs(5),
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut wait)
                .await
                .is_err(),
            "the submit parks on the held slot"
        );
        assert!(
            QueueTest::state(&queue).outcomes.contains_key(&key),
            "the waiter is under the key while the submit waits"
        );
        drop(wait);
        assert!(
            QueueTest::state(&queue).outcomes.is_empty(),
            "the caller leaving took its waiter"
        );

        let mut later = Box::pin(queue.submit_and_wait(
            Job::RefreshWorkerLabels {
                url: key.clone(),
                asker: WorkerJobAsker::Api,
            },
            Duration::from_secs(5),
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut later)
                .await
                .is_err(),
            "the later submit parks too"
        );
        assert!(
            QueueTest::state(&queue).outcomes.contains_key(&key),
            "a later caller takes the key"
        );
        drop(later);
        drop(slot);
    }

    fn spec_for(url: &str, router_config: &RouterConfig) -> WorkerSpec {
        let mut spec = WorkerSpec::new(url);
        apply_startup_worker_config(&mut spec, router_config);
        spec
    }

    /// A grouped-ZMQ fleet stamps `dp_size` only on the `ipc://` workers:
    /// `dp_size` on an HTTP/gRPC worker means rank-aware DP routing, a
    /// different topology entirely.
    #[test]
    fn engine_count_only_reaches_zmq_workers_in_a_mixed_fleet() {
        let config = RouterConfig {
            zmq_engine_count: Some(4),
            ..RouterConfig::default()
        };

        assert_eq!(spec_for("ipc:///tmp/smg/engine", &config).dp_size, Some(4));
        assert_eq!(spec_for("grpc://127.0.0.1:30000", &config).dp_size, None);
        assert_eq!(spec_for("http://127.0.0.1:8000", &config).dp_size, None);
    }

    /// One engine is an ungrouped worker, not a group of one: the ZMQ client
    /// awaits a single engine by default, and `Some(1)` must not turn the
    /// worker into a dp-aware one.
    #[test]
    fn a_single_engine_leaves_dp_size_unset() {
        let mut config = RouterConfig {
            zmq_engine_count: Some(1),
            ..RouterConfig::default()
        };
        assert_eq!(spec_for("ipc:///tmp/smg/engine", &config).dp_size, None);

        config.zmq_engine_count = None;
        assert_eq!(spec_for("ipc:///tmp/smg/engine", &config).dp_size, None);
    }

    /// `--backend` pins the ZMQ wire protocol (the handshake can't be probed);
    /// without it the spec keeps its default so `detect_backend` can probe.
    #[test]
    fn startup_runtime_is_pinned_only_when_configured() {
        let default_runtime = WorkerSpec::new("ipc:///tmp/smg/engine").runtime_type;

        let mut config = RouterConfig {
            startup_worker_runtime_type: Some(RuntimeType::Vllm),
            ..RouterConfig::default()
        };
        assert_eq!(
            spec_for("ipc:///tmp/smg/engine", &config).runtime_type,
            RuntimeType::Vllm
        );

        config.startup_worker_runtime_type = None;
        assert_eq!(
            spec_for("ipc:///tmp/smg/engine", &config).runtime_type,
            default_runtime
        );
    }

    /// Pending/processing statuses are the discovery reconciler's in-flight
    /// guard: purging them mid-wave makes every pass resubmit duplicate jobs
    /// for workers still registering, multiplying the wave's work.
    #[test]
    fn cleanup_never_expires_in_flight_statuses() {
        let pending = JobStatus::pending("AddWorker", "http://w:8000");
        let processing = JobStatus::processing("AddWorker", "http://w:8000");
        let far_future = pending.timestamp + 100_000;

        assert!(!QueueTest::status_expired(&pending, far_future, 300));
        assert!(!QueueTest::status_expired(&processing, far_future, 300));
    }

    /// Terminal statuses still age out so a failed worker retries on a later
    /// reconcile pass and the map stays bounded.
    #[test]
    fn cleanup_expires_terminal_statuses_after_the_ttl() {
        let failed = JobStatus::failed("AddWorker", "http://w:8000", "boom".to_string());

        assert!(!QueueTest::status_expired(
            &failed,
            failed.timestamp + 299,
            300
        ));
        assert!(QueueTest::status_expired(
            &failed,
            failed.timestamp + 300,
            300
        ));
    }

    /// Concurrent creates for one URL must enqueue exactly one AddWorker, and a
    /// terminal status must not block the next attempt (#1533).
    #[test]
    fn claim_admits_one_in_flight_job_per_url() {
        let map = Arc::new(DashMap::new());
        let won: Vec<bool> = (0..16)
            .map(|_| {
                let map = Arc::clone(&map);
                std::thread::spawn(move || QueueTest::claim(&map, "http://w:8000", "AddWorker"))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        assert_eq!(won.iter().filter(|&&w| w).count(), 1);

        map.insert(
            "http://w:8000".to_string(),
            JobStatus::failed("AddWorker", "http://w:8000", "boom".to_string()),
        );
        assert!(QueueTest::claim(&map, "http://w:8000", "AddWorker"));

        // A different job type under the same key is not a competing claim.
        map.insert(
            "http://w:8000".to_string(),
            JobStatus::pending("RegisterMcpServer", "http://w:8000"),
        );
        assert!(QueueTest::claim(&map, "http://w:8000", "AddWorker"));
    }

    /// A claim that yields to any in-flight job never replaces that job's
    /// status under the URL; a terminal status under it is taken over.
    #[test]
    fn claim_unless_in_flight_yields_to_a_job_of_another_type() {
        let map: DashMap<String, JobStatus> = DashMap::new();
        map.insert(
            "http://w:8000".to_string(),
            JobStatus::processing("AddWorker", "http://w:8000"),
        );
        assert!(!QueueTest::claim_unless_in_flight(
            &map,
            "http://w:8000",
            "RefreshWorkerLabels"
        ));
        assert_eq!(map.get("http://w:8000").unwrap().job_type, "AddWorker");

        map.insert(
            "http://w:8000".to_string(),
            JobStatus::failed("AddWorker", "http://w:8000", "boom".to_string()),
        );
        assert!(QueueTest::claim_unless_in_flight(
            &map,
            "http://w:8000",
            "RefreshWorkerLabels"
        ));
        assert_eq!(
            map.get("http://w:8000").unwrap().job_type,
            "RefreshWorkerLabels"
        );
    }

    /// A job's processing and completion writes touch only its own entry: a
    /// job of another type that took the URL after it was queued keeps its
    /// status through the first job's processing, failure and success.
    #[test]
    fn a_jobs_status_writes_leave_another_jobs_entry_alone() {
        let map: DashMap<String, JobStatus> = DashMap::new();
        let url = "http://w:8000";
        map.insert(url.to_string(), JobStatus::processing("RemoveWorker", url));
        for status in [
            Some(JobStatus::processing("RefreshWorkerLabels", url)),
            Some(JobStatus::failed(
                "RefreshWorkerLabels",
                url,
                "boom".to_string(),
            )),
            None,
        ] {
            QueueTest::record_own_status(&map, url, "RefreshWorkerLabels", status);
            let entry = map.get(url).expect("the other job's entry stays");
            assert_eq!(entry.job_type, "RemoveWorker");
            assert_eq!(entry.status, "processing");
        }

        // The job's own entry takes its writes, and a vacant one is written.
        map.insert(
            url.to_string(),
            JobStatus::pending("RefreshWorkerLabels", url),
        );
        QueueTest::record_own_status(
            &map,
            url,
            "RefreshWorkerLabels",
            Some(JobStatus::processing("RefreshWorkerLabels", url)),
        );
        assert_eq!(map.get(url).unwrap().status, "processing");
        QueueTest::record_own_status(&map, url, "RefreshWorkerLabels", None);
        assert!(map.get(url).is_none());
        QueueTest::record_own_status(
            &map,
            url,
            "RefreshWorkerLabels",
            Some(JobStatus::failed(
                "RefreshWorkerLabels",
                url,
                "boom".to_string(),
            )),
        );
        assert_eq!(map.get(url).unwrap().status, "failed");
    }

    /// The pending status a claimed submission writes once its slot is held
    /// goes over its own claim alone: a job of another type that took the URL
    /// during the wait keeps its entry, where a plain submission's write takes
    /// the key; a vacant entry is written either way.
    #[test]
    fn a_claimed_submissions_pending_write_leaves_another_jobs_entry_alone() {
        let map: DashMap<String, JobStatus> = DashMap::new();
        let url = "http://w:8000";
        map.insert(url.to_string(), JobStatus::pending("AddWorker", url));
        QueueTest::write_pending(&map, url, "RefreshWorkerLabels", true);
        assert_eq!(map.get(url).unwrap().job_type, "AddWorker");
        QueueTest::write_pending(&map, url, "RefreshWorkerLabels", false);
        assert_eq!(map.get(url).unwrap().job_type, "RefreshWorkerLabels");
        map.remove(url);
        QueueTest::write_pending(&map, url, "RefreshWorkerLabels", true);
        let entry = map.get(url).expect("a vacant entry is written");
        assert_eq!(entry.job_type, "RefreshWorkerLabels");
        assert_eq!(entry.status, "pending");
    }

    /// A claimed submission that fails releases its claim, so the caller's
    /// next occasion submits again; the release is a write of its own entry
    /// (`record_own_status`), so another job's entry under the URL stays.
    #[tokio::test]
    async fn failed_submit_unless_in_flight_releases_its_claim() {
        let queue = JobQueue::new(JobQueueConfig::default(), Weak::new());
        let url = "http://w:8000";
        let job = || Job::RefreshWorkerLabels {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
        };
        assert!(queue.submit_unless_in_flight(job()).await.is_err());
        assert!(QueueTest::state(&queue).status_map.get(url).is_none());
        assert!(queue.submit_unless_in_flight(job()).await.is_err());
    }

    /// A job that finds another in flight under its URL is kept and runs on
    /// that job's heels instead of being dropped: a label refresh whose
    /// worker's registration job is still ending (its handshake completed
    /// during the last steps) would otherwise wait for a transition into
    /// service a healthy worker never makes. The holder's status is left to
    /// it; the kept job runs under its own once the holder ends. A test
    /// context starts no workflow engine, so a job that runs ends failed
    /// under its own status: that status is the run's evidence.
    #[tokio::test]
    async fn a_job_kept_behind_the_in_flight_one_runs_when_it_ends() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        let url = "http://w:8000";
        QueueTest::hold_url_for_tests(&queue, url, "AddWorker");

        let removal = Job::RemoveWorker {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
            expected_revisions: None,
        };
        assert_eq!(queue.submit_or_defer(removal).await, Ok(false));
        assert!(
            QueueTest::kept_for(&queue, url),
            "the job is kept, not dropped"
        );
        assert_eq!(
            queue.get_status(url).expect("the holder's status").job_type,
            "AddWorker",
            "the holder's status is left to it"
        );

        QueueTest::record_job_completion(
            "AddWorker",
            url,
            Duration::ZERO,
            &Ok("done".to_string()),
            QueueTest::state(&queue).status_map,
        );
        QueueTest::run_kept(
            url,
            &context,
            QueueTest::state(&queue).status_map,
            QueueTest::state(&queue).kept,
            QueueTest::state(&queue).cancelled_before_start,
        )
        .await;
        assert!(!QueueTest::kept_for(&queue, url));
        let status = queue.get_status(url).expect("the kept job's own status");
        assert_eq!(status.job_type, "RemoveWorker");
        assert_eq!(
            status.status, "failed",
            "the kept job ran once the holder ended"
        );
    }

    /// The queue as the worker's health manager holds it ([`WorkerJobs`]): a
    /// label refresh asked through the trait is the queue's own refresh job.
    /// With nothing in flight under the address it is submitted under its
    /// kind; behind a job that holds the address it is kept, the holder's
    /// status left to it, and runs under its own kind when the holder ends.
    #[tokio::test]
    async fn a_refresh_asked_through_the_workers_trait_is_the_queues_refresh() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        let jobs: Arc<dyn WorkerJobs> = queue.clone();

        let free = "http://free:8000";
        assert_eq!(
            jobs.refresh_worker_labels(free.to_string(), WorkerJobAsker::Health)
                .await,
            Ok(true)
        );
        assert_eq!(
            queue
                .get_status(free)
                .expect("the refresh's status")
                .job_type,
            "RefreshWorkerLabels"
        );

        let held = "http://held:8000";
        QueueTest::hold_url_for_tests(&queue, held, "AddWorker");
        assert_eq!(
            jobs.refresh_worker_labels(held.to_string(), WorkerJobAsker::Health)
                .await,
            Ok(false)
        );
        assert!(
            QueueTest::kept_for(&queue, held),
            "kept behind the job that holds it"
        );
        assert_eq!(
            queue
                .get_status(held)
                .expect("the holder's status")
                .job_type,
            "AddWorker",
            "the holder's status is left to it"
        );
        QueueTest::record_job_completion(
            "AddWorker",
            held,
            Duration::ZERO,
            &Ok("done".to_string()),
            QueueTest::state(&queue).status_map,
        );
        QueueTest::run_kept(
            held,
            &context,
            QueueTest::state(&queue).status_map,
            QueueTest::state(&queue).kept,
            QueueTest::state(&queue).cancelled_before_start,
        )
        .await;
        assert!(!QueueTest::kept_for(&queue, held));
        assert_eq!(
            queue
                .get_status(held)
                .expect("the kept job's status")
                .job_type,
            "RefreshWorkerLabels",
            "the kept job is the refresh"
        );
    }

    /// The queue as a discovery pass reads it ([`WorkerJobs::job_in_flight`]):
    /// an address reads as in flight while a job of it is pending or
    /// processing, whatever the job, and no longer once that job has ended,
    /// done or failed, so the next pass retries what failed.
    #[tokio::test]
    async fn an_address_reads_as_in_flight_until_its_job_ends() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        let jobs: Arc<dyn WorkerJobs> = queue.clone();
        let url = "http://held:8000";
        assert!(!jobs.job_in_flight(url), "no job of the address");

        let ends: [Result<String, String>; 2] =
            [Ok("done".to_string()), Err("refused".to_string())];
        for (job_type, end) in ["AddWorker", "RemoveWorker"].into_iter().zip(ends) {
            QueueTest::hold_url_for_tests(&queue, url, job_type);
            assert!(jobs.job_in_flight(url), "{job_type} is processing");
            QueueTest::record_job_completion(
                job_type,
                url,
                Duration::ZERO,
                &end,
                QueueTest::state(&queue).status_map,
            );
            assert!(!jobs.job_in_flight(url), "{job_type} ended: {end:?}");
        }

        QueueTest::state(&queue)
            .status_map
            .insert(url.to_string(), JobStatus::pending("AddWorker", url));
        assert!(jobs.job_in_flight(url), "a pending job is in flight");
    }

    /// The wiring at `process_job`: a kept job runs at the completion of the
    /// job that held the URL, through the dispatcher and not by hand. A hold
    /// by hand keeps the first submission; a plain submission for the same
    /// URL writes its pending status over the hold and becomes the holder;
    /// when the dispatcher finishes it, the kept job runs under its own
    /// status. Neither job's workflow runs in a test context, so both end
    /// failed, and the URL's last status is the kept job's.
    #[tokio::test]
    #[tracing_test::traced_test]
    async fn a_kept_job_runs_at_the_dispatcher_completion_of_its_holder() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        let url = "http://w:8001";
        QueueTest::hold_url_for_tests(&queue, url, "AddWorker");
        let kept = Job::RemoveWorker {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
            expected_revisions: None,
        };
        assert_eq!(queue.submit_or_defer(kept).await, Ok(false));
        assert!(QueueTest::kept_for(&queue, url));

        queue
            .submit(Job::RefreshWorkerLabels {
                asker: WorkerJobAsker::Api,
                url: url.to_string(),
            })
            .await
            .expect("the holder is queued");
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let last = queue
                .get_status(url)
                .map(|status| (status.job_type.to_string(), status.status.to_string()));
            if !QueueTest::kept_for(&queue, url)
                && last == Some(("RemoveWorker".to_string(), "failed".to_string()))
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the kept job did not run at the holder's completion: kept={} last={last:?}",
                QueueTest::kept_for(&queue, url)
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        logs_assert(|lines| {
            for message in [
                "Processing job: type=RefreshWorkerLabels",
                "Running the RemoveWorker job kept",
            ] {
                if !lines
                    .iter()
                    .any(|line| line.contains(message) && line.contains("asker=Api"))
                {
                    return Err(format!("missing asker on {message}"));
                }
            }
            Ok(())
        });
    }

    /// A label refresh kept while a DELETE's removal is kept for the URL is
    /// kept beside it (one slot per kind), and both run at the holder's end,
    /// the removal first: the refresh's worker may be the one leaving, or
    /// the one a guarded removal leaves standing. Each runs under its own
    /// status, so the URL's last status is the refresh's. Before this the
    /// one slot per URL held the latest job, and a refresh kept while the
    /// DELETE waited dropped the accepted DELETE.
    #[tokio::test]
    async fn a_refresh_kept_beside_a_kept_removal_runs_after_it() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        let url = "http://w:8002";
        QueueTest::hold_url_for_tests(&queue, url, "AddWorker");
        let removal = Job::RemoveWorker {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
            expected_revisions: None,
        };
        assert_eq!(queue.submit_or_defer(removal).await, Ok(false));
        let refresh = Job::RefreshWorkerLabels {
            asker: WorkerJobAsker::Health,
            url: url.to_string(),
        };
        assert_eq!(
            queue.submit_or_defer(refresh).await,
            Ok(false),
            "the refresh is kept, neither submitted nor dropped"
        );
        assert!(QueueTest::kept_for(&queue, url));

        QueueTest::record_job_completion(
            "AddWorker",
            url,
            Duration::ZERO,
            &Ok("done".to_string()),
            QueueTest::state(&queue).status_map,
        );
        QueueTest::run_kept(
            url,
            &context,
            QueueTest::state(&queue).status_map,
            QueueTest::state(&queue).kept,
            QueueTest::state(&queue).cancelled_before_start,
        )
        .await;
        assert!(!QueueTest::kept_for(&queue, url), "both kept jobs ran");
        assert_eq!(
            queue
                .get_status(url)
                .expect("the last kept job's own status")
                .job_type,
            "RefreshWorkerLabels",
            "the refresh ran after the removal, under its own status"
        );
    }

    /// A DELETE's removal kept while a label refresh is kept for the URL
    /// takes the removal's slot beside it (the refresh stays), and the run
    /// order is the removal first, then the refresh, whichever was kept
    /// first.
    #[tokio::test]
    async fn a_removal_kept_beside_a_kept_refresh_runs_first() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        let url = "http://w:8003";
        QueueTest::hold_url_for_tests(&queue, url, "AddWorker");
        let refresh = Job::RefreshWorkerLabels {
            asker: WorkerJobAsker::Health,
            url: url.to_string(),
        };
        assert_eq!(queue.submit_or_defer(refresh).await, Ok(false));
        let removal = Job::RemoveWorker {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
            expected_revisions: None,
        };
        assert_eq!(queue.submit_or_defer(removal).await, Ok(false));
        assert!(QueueTest::kept_for(&queue, url));
        // The run order, read off the slots and put back as it was.
        let first = QueueTest::take_next(QueueTest::state(&queue).kept, url, |_| true)
            .expect("the removal is next");
        let second = QueueTest::take_next(QueueTest::state(&queue).kept, url, |_| true)
            .expect("the refresh follows");
        assert_eq!(
            (first.job_type(), second.job_type()),
            ("RemoveWorker", "RefreshWorkerLabels"),
            "the removal runs first, the refresh after it"
        );
        assert!(
            QueueTest::take_next(QueueTest::state(&queue).kept, url, |_| true).is_none()
                && !QueueTest::kept_for(&queue, url),
            "two slots, nothing else"
        );
        QueueTest::put_back(QueueTest::state(&queue).kept, url, second);
        QueueTest::put_back(QueueTest::state(&queue).kept, url, first);

        QueueTest::record_job_completion(
            "AddWorker",
            url,
            Duration::ZERO,
            &Ok("done".to_string()),
            QueueTest::state(&queue).status_map,
        );
        QueueTest::run_kept(
            url,
            &context,
            QueueTest::state(&queue).status_map,
            QueueTest::state(&queue).kept,
            QueueTest::state(&queue).cancelled_before_start,
        )
        .await;
        assert!(!QueueTest::kept_for(&queue, url), "both kept jobs ran");
        assert_eq!(
            queue
                .get_status(url)
                .expect("the last kept job's own status")
                .job_type,
            "RefreshWorkerLabels",
            "the refresh ran last: the removal ran first and did not displace it"
        );
    }

    /// A repeated DELETE's removal, pinned to the id it saw, kept behind the
    /// job that took the URL after the first removal, and the refresh of
    /// the worker registered since under a fresh id, kept behind the same
    /// job: both are kept, the removal first, and both run at the holder's
    /// end. Before this the removal took the one slot and the refresh was
    /// dropped, and the worker the removal's pin then skipped kept its
    /// provisional labels until another transition into service.
    #[tokio::test]
    async fn a_fresh_workers_refresh_kept_beside_an_old_id_removal_runs_after_it() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        let url = "http://w:8005";
        QueueTest::hold_url_for_tests(&queue, url, "UpdateWorker");
        let old_id_removal = Job::RemoveWorker {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
            expected_revisions: Some(HashMap::from([("old-id".to_string(), None)])),
        };
        assert_eq!(queue.submit_or_defer(old_id_removal).await, Ok(false));
        let fresh_workers_refresh = Job::RefreshWorkerLabels {
            asker: WorkerJobAsker::Health,
            url: url.to_string(),
        };
        assert_eq!(
            queue.submit_or_defer(fresh_workers_refresh).await,
            Ok(false),
            "the refresh is kept beside the removal, not dropped for it"
        );
        let next = QueueTest::take_next(QueueTest::state(&queue).kept, url, |_| true)
            .expect("the removal is next");
        assert!(
            matches!(
                next,
                Job::RemoveWorker {
                    expected_revisions: Some(_),
                    ..
                }
            ),
            "the old-id removal runs first"
        );
        QueueTest::put_back(QueueTest::state(&queue).kept, url, next);

        QueueTest::record_job_completion(
            "UpdateWorker",
            url,
            Duration::ZERO,
            &Ok("done".to_string()),
            QueueTest::state(&queue).status_map,
        );
        QueueTest::run_kept(
            url,
            &context,
            QueueTest::state(&queue).status_map,
            QueueTest::state(&queue).kept,
            QueueTest::state(&queue).cancelled_before_start,
        )
        .await;
        assert!(
            !QueueTest::kept_for(&queue, url),
            "the removal and the refresh both ran"
        );
        assert_eq!(
            queue
                .get_status(url)
                .expect("the last kept job's own status")
                .job_type,
            "RefreshWorkerLabels",
            "the fresh worker's refresh ran after the removal"
        );
    }

    /// A kept job whose claim the runner could not take goes back to the
    /// slot of its kind: a job of the other kind kept meanwhile keeps its own
    /// slot, and a removal kept meanwhile merges with the one going back (an
    /// unguarded side wins). The URL's entry goes with its last job.
    #[test]
    fn a_kept_job_put_back_merges_with_the_removal_kept_meanwhile() {
        let url = "http://w:8004";
        let removal = || Job::RemoveWorker {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
            expected_revisions: None,
        };
        let refresh = || Job::RefreshWorkerLabels {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
        };
        let kinds = |kept: &DashMap<String, Kept<Job>>| {
            kept.get(url).map_or((None, None), |slots| {
                (
                    slots.removals.front().map(Job::job_type),
                    slots.other.as_ref().map(Job::job_type),
                )
            })
        };

        let kept: DashMap<String, Kept<Job>> = DashMap::new();
        QueueTest::put_back(&kept, url, refresh());
        assert_eq!(
            kinds(&kept),
            (None, Some("RefreshWorkerLabels")),
            "an empty slot takes the job back"
        );

        QueueTest::put_back(
            &kept,
            url,
            Job::RemoveWorker {
                asker: WorkerJobAsker::Api,
                url: url.to_string(),
                expected_revisions: Some(HashMap::new()),
            },
        );
        assert_eq!(
            kinds(&kept),
            (Some("RemoveWorker"), Some("RefreshWorkerLabels")),
            "a removal put back beside a kept refresh keeps both"
        );

        QueueTest::keep(
            &kept,
            url,
            Job::RemoveWorker {
                asker: WorkerJobAsker::Api,
                url: url.to_string(),
                expected_revisions: Some(HashMap::new()),
            },
        );
        QueueTest::put_back(&kept, url, removal());
        assert!(
            kept.get(url).is_some_and(|slots| matches!(
                slots.removals.front(),
                Some(Job::RemoveWorker {
                    expected_revisions: None,
                    ..
                })
            )),
            "a removal put back merges with the removal kept meanwhile, the unguarded side winning"
        );
        assert_eq!(
            kinds(&kept).1,
            Some("RefreshWorkerLabels"),
            "the other kind's slot is untouched"
        );

        assert_eq!(
            QueueTest::take_next(&kept, url, |_| true).map(|job| job.job_type()),
            Some("RemoveWorker"),
            "the removal is taken first"
        );
        assert_eq!(
            QueueTest::take_next(&kept, url, |_| true).map(|job| job.job_type()),
            Some("RefreshWorkerLabels"),
            "the refresh follows"
        );
        assert!(
            QueueTest::take_next(&kept, url, |_| true).is_none() && !kept.contains_key(url),
            "the URL's entry goes with its last job"
        );
    }

    /// A same-URL registration job of another kind started under a cancelled
    /// `POST`'s reservation (a `PUT` queued before the `POST`, a discovery
    /// pass over the URL) owns no mark and leaves the `POST`'s for the
    /// `POST`'s own job, which takes it.
    #[test]
    fn a_put_or_discovery_job_under_a_cancelled_posts_reservation_leaves_its_mark() {
        let registry = WorkerRegistry::new();
        let url = "http://w:8000";
        let reserved = registry.reserve_id_for_url(url);
        let marks: DashSet<WorkerId> = DashSet::new();
        marks.insert(reserved.clone());
        let job = |mode, asker| Job::AddWorker {
            asker,
            config: Box::new(WorkerSpec::new(url)),
            registration_mode: mode,
        };
        for (kind, mode, asker) in [
            (
                "a PUT's",
                WorkerRegistrationMode::ReplaceById {
                    worker_id: reserved.as_str().to_string(),
                    expected_revision: 1,
                },
                WorkerJobAsker::Api,
            ),
            (
                "a discovery",
                WorkerRegistrationMode::Upsert,
                WorkerJobAsker::Discovery,
            ),
        ] {
            let own = own_mark_id(&job(mode, asker), &registry);
            assert_eq!(own, None, "{kind} job owns no mark");
            assert!(!take_mark(&marks, own.as_ref()));
            assert!(marks.contains(&reserved), "{kind} job took the POST's mark");
        }
        let own = own_mark_id(
            &job(WorkerRegistrationMode::CreateOnly, WorkerJobAsker::Api),
            &registry,
        );
        assert_eq!(
            own.as_ref(),
            Some(&reserved),
            "the POST's job owns its reserved id"
        );
        assert!(take_mark(&marks, own.as_ref()));
        assert!(marks.is_empty());
    }

    /// A registration's cleanup removal is one per URL at a time: a second
    /// submission (a DELETE's, or a second end) finds the first in flight and
    /// is kept to run on its heels with its own pins; the entry under the URL
    /// is the first removal's own.
    #[tokio::test]
    async fn a_registrations_cleanup_removal_is_queued_once_per_url() {
        use crate::{
            app_context::testing::app_context_for_tests, routers::common::snapshot::WorkerRegistry,
        };
        let registry = Arc::new(WorkerRegistry::new());
        let context = app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None);
        // A queue that accepts jobs and never runs one: no slot to run one in.
        let queue = JobQueue::new(
            JobQueueConfig {
                max_concurrent_jobs: 0,
                ..JobQueueConfig::default()
            },
            Arc::downgrade(&context),
        );
        context.worker_job_queue.set(queue).ok();
        let url = "http://w:8000";
        let entries = || HashMap::from([("worker-id".to_string(), 1u64)]);
        assert_eq!(
            queue_removal(&context, url, entries(), WorkerJobAsker::Api).await,
            Ok(CleanupRemoval::Queued)
        );
        assert_eq!(
            queue_removal(&context, url, entries(), WorkerJobAsker::Api).await,
            Ok(CleanupRemoval::Behind),
            "a removal in flight owns the URL; the second runs on its heels"
        );
        let status = context
            .worker_job_queue
            .get()
            .and_then(|queue| queue.get_status(url))
            .expect("the removal's entry");
        assert_eq!(status.job_type, "RemoveWorker");
        assert_eq!(status.status, "pending");
        let queue = context.worker_job_queue.get().expect("the queue");
        let slots = QueueTest::state(queue)
            .kept
            .get(url)
            .expect("a removal kept behind the one in flight");
        let Some(Job::RemoveWorker {
            expected_revisions: Some(pins),
            ..
        }) = slots.removals.front()
        else {
            panic!("the kept job is the cleanup's pinned removal")
        };
        assert_eq!(
            pins,
            &HashMap::from([("worker-id".to_string(), Some(1u64))]),
            "the kept removal carries the second cleanup's entries"
        );
    }

    /// A removal in flight at the URL pinned to other workers does not own a
    /// failed registration's: the cleanup's removal is kept to run on its
    /// heels with its own pins, and a later cleanup's entries join the kept
    /// removal's pins instead of displacing it.
    #[tokio::test]
    async fn a_cleanup_behind_a_removal_of_other_workers_keeps_its_own_pins() {
        use crate::{
            app_context::testing::app_context_for_tests, routers::common::snapshot::WorkerRegistry,
        };
        let registry = Arc::new(WorkerRegistry::new());
        let context = app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None);
        let queue = JobQueue::new(
            JobQueueConfig {
                max_concurrent_jobs: 0,
                ..JobQueueConfig::default()
            },
            Arc::downgrade(&context),
        );
        context.worker_job_queue.set(queue).ok();
        let url = "http://w:8000";
        let one = |id: &str, revision: u64| HashMap::from([(id.to_string(), revision)]);
        assert_eq!(
            queue_removal(&context, url, one("old", 1), WorkerJobAsker::Api).await,
            Ok(CleanupRemoval::Queued)
        );
        assert_eq!(
            queue_removal(&context, url, one("new", 3), WorkerJobAsker::Api).await,
            Ok(CleanupRemoval::Behind),
            "a removal of other workers in flight: the cleanup runs on its heels"
        );
        assert_eq!(
            queue_removal(&context, url, one("third", 5), WorkerJobAsker::Api).await,
            Ok(CleanupRemoval::Behind),
            "a later cleanup joins the kept removal"
        );
        let queue = context.worker_job_queue.get().expect("the queue");
        let slots = QueueTest::state(queue)
            .kept
            .get(url)
            .expect("a removal kept behind the one in flight");
        let Some(Job::RemoveWorker {
            expected_revisions: Some(pins),
            ..
        }) = slots.removals.front()
        else {
            panic!("the kept job is a pinned removal")
        };
        assert_eq!(
            pins,
            &HashMap::from([
                ("new".to_string(), Some(3u64)),
                ("third".to_string(), Some(5u64)),
            ]),
            "both cleanups' entries, none of the in-flight removal's"
        );
    }

    /// A submission that fails must not leave its claim behind, or the next
    /// create for the URL would be told a job is in flight when none is (#1533).
    #[tokio::test]
    async fn a_failed_claimed_submission_releases_its_claim() {
        let queue = JobQueue::new(JobQueueConfig::default(), Weak::new());
        let job = || Job::AddWorker {
            asker: WorkerJobAsker::Api,
            config: Box::new(WorkerSpec::new("http://w:8000")),
            registration_mode: WorkerRegistrationMode::CreateOnly,
        };
        assert!(queue.claim_if_idle(&job()));
        assert!(queue.submit_claimed(job()).await.is_err());
        // Before the rollback the claim stayed: a 202 with no job behind it.
        assert!(
            queue.claim_if_idle(&job()),
            "a failed submission left its claim"
        );
    }

    /// The queue channel and dispatch semaphore are sized from the config,
    /// not hardcoded.
    #[tokio::test]
    async fn queue_sizing_comes_from_the_config() {
        let context: Weak<AppContext> = Weak::new();
        let queue = JobQueue::new(
            JobQueueConfig {
                queue_capacity: 7,
                max_concurrent_jobs: 3,
            },
            context,
        );

        assert_eq!(QueueTest::capacity(&queue), 7);
        let (queue_depth, available_permits) = queue.get_load_info();
        assert_eq!(queue_depth, 0);
        assert_eq!(available_permits, 3);
    }

    /// The connection budget scales with the health-check success threshold,
    /// and a zero threshold must not collapse it to zero attempts. A startup
    /// worker and an external endpoint share the configured budget, the one
    /// derivation on `RouterConfig`.
    #[test]
    fn connection_attempts_scale_with_the_success_threshold_floor() {
        let mut config = RouterConfig::default();
        config.health_check.success_threshold = 0;
        assert_eq!(
            spec_for("http://127.0.0.1:8000", &config).max_connection_attempts,
            10
        );
        assert_eq!(
            build_external_worker_config("https://api.example.com", None, &config)
                .max_connection_attempts,
            10
        );

        config.health_check.success_threshold = 3;
        assert_eq!(
            spec_for("http://127.0.0.1:8000", &config).max_connection_attempts,
            30
        );
        assert_eq!(
            build_external_worker_config("https://api.example.com", None, &config)
                .max_connection_attempts,
            config.connection_attempt_budget(RouterConfig::CONFIGURED_WORKER_ATTEMPTS_PER_SUCCESS)
        );
    }

    /// A PATCH's submission unless a registration is in flight: the check and
    /// the claim are one entry operation, and the update's writes stay on its
    /// own entry. A registration in flight refuses the update and keeps its
    /// entry; a registration that claims the URL after the update's claim
    /// keeps its entry through the update's writes.
    #[tokio::test]
    async fn an_update_submitted_unless_registering_leaves_a_registrations_entry() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        // A queue that accepts jobs and never runs one: no slot to run one in.
        let queue = JobQueue::new(
            JobQueueConfig {
                max_concurrent_jobs: 0,
                ..JobQueueConfig::default()
            },
            Arc::downgrade(&context),
        );
        let url = "http://w:8003";
        let update = || Job::UpdateWorker {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
            update: Box::new(WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            }),
        };
        QueueTest::hold_url_for_tests(&queue, url, "AddWorker");
        assert_eq!(queue.submit_unless_registering(update()).await, Ok(false));
        let entry = queue.get_status(url).expect("the registration's entry");
        assert_eq!(
            (entry.job_type.as_str(), entry.status.as_str()),
            ("AddWorker", "processing")
        );
        QueueTest::record_job_completion(
            "AddWorker",
            url,
            Duration::ZERO,
            &Ok("done".to_string()),
            QueueTest::state(&queue).status_map,
        );
        assert_eq!(queue.submit_unless_registering(update()).await, Ok(true));
        let entry = queue.get_status(url).expect("the update's entry");
        assert_eq!(
            (entry.job_type.as_str(), entry.status.as_str()),
            ("UpdateWorker", "pending")
        );
        let registration = Job::AddWorker {
            asker: WorkerJobAsker::Api,
            config: Box::new(WorkerSpec::new(url)),
            registration_mode: WorkerRegistrationMode::CreateOnly,
        };
        assert!(
            queue.claim_if_idle(&registration),
            "the registration takes the entry"
        );
        QueueTest::record_job_completion(
            "UpdateWorker",
            url,
            Duration::ZERO,
            &Ok("done".to_string()),
            QueueTest::state(&queue).status_map,
        );
        let entry = queue.get_status(url).expect("the registration's entry");
        assert_eq!(
            (entry.job_type.as_str(), entry.status.as_str()),
            ("AddWorker", "pending")
        );
    }

    /// An update job for `url`, the plain shape the tests below submit.
    fn an_update(url: &str) -> Job {
        Job::UpdateWorker {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
            update: Box::new(WorkerUpdateRequest {
                priority: None,
                cost: None,
                labels: None,
                api_key: None,
                health: None,
            }),
        }
    }

    /// A one-slot queue that is full and runs no job: the dispatcher parks the
    /// first job on the concurrency limit of zero, the second fills the slot,
    /// and every later submission waits.
    async fn a_full_one_slot_queue(context: &Arc<AppContext>) -> Arc<JobQueue> {
        let queue = JobQueue::new(
            JobQueueConfig {
                queue_capacity: 1,
                max_concurrent_jobs: 0,
            },
            Arc::downgrade(context),
        );
        queue
            .submit(an_update("http://w:8006"))
            .await
            .expect("the first job");
        for _ in 0..1000 {
            if queue.get_load_info().0 == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            queue.get_load_info().0,
            0,
            "the dispatcher took the first job"
        );
        queue
            .submit(an_update("http://w:8007"))
            .await
            .expect("the slot");
        assert_eq!(queue.get_load_info().0, 1, "the queue is full");
        queue
    }

    /// A claimed submission dropped inside its wait for a slot (a cancelled
    /// request on a full queue) leaves no entry under its URL: the claim an
    /// update made against a registration in flight and a registration's
    /// typed claim both go with the dropped caller, and the URL takes the
    /// next claim.
    #[tokio::test]
    async fn a_claimed_submission_dropped_in_its_slot_wait_leaves_no_entry() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = a_full_one_slot_queue(&context).await;
        let url = "http://w:8008";
        let wait = Duration::from_millis(50);
        let dropped =
            tokio::time::timeout(wait, queue.submit_unless_registering(an_update(url))).await;
        assert!(dropped.is_err(), "the update's submission waits for a slot");
        assert!(
            queue.get_status(url).is_none(),
            "the update's claim went with its caller"
        );

        let registration = Job::AddWorker {
            asker: WorkerJobAsker::Api,
            config: Box::new(WorkerSpec::new(url)),
            registration_mode: WorkerRegistrationMode::CreateOnly,
        };
        assert!(
            queue.claim_if_idle(&registration),
            "the URL takes the next claim"
        );
        let dropped = tokio::time::timeout(wait, queue.submit_claimed(registration)).await;
        assert!(
            dropped.is_err(),
            "the registration's submission waits for a slot"
        );
        assert!(
            queue.get_status(url).is_none(),
            "the registration's claim went with its caller"
        );
    }

    /// A plain submission touches no entry before it holds a slot: the entry
    /// another job holds under the URL survives a plain submission of the
    /// same type dropped inside its wait on a full queue.
    #[tokio::test]
    async fn a_plain_submission_dropped_in_its_slot_wait_leaves_the_entry_it_found() {
        use crate::app_context::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = a_full_one_slot_queue(&context).await;
        let url = "http://w:8009";
        QueueTest::hold_url_for_tests(&queue, url, "UpdateWorker");
        let dropped =
            tokio::time::timeout(Duration::from_millis(50), queue.submit(an_update(url))).await;
        assert!(dropped.is_err(), "the plain submission waits for a slot");
        let entry = queue
            .get_status(url)
            .expect("the entry the other job holds");
        assert_eq!(
            (entry.job_type.as_str(), entry.status.as_str()),
            ("UpdateWorker", "processing"),
            "the entry survives the dropped plain submission"
        );
    }

    /// A `DELETE` whose cancel is written while the registration's registry
    /// step runs (past the step's cancel read, before its registry write)
    /// finds nothing registered to remove, and the step then writes its
    /// worker: the job's end reads it from the registry and it leaves through
    /// the removal the end queues, so the URL takes a new registration. The
    /// window is held open by the registry step's test hold; on the base the
    /// worker stayed under the URL, under a fresh id.
    #[tokio::test]
    async fn a_registration_cancelled_as_its_registry_step_writes_leaves_no_worker_under_its_url() {
        use http::StatusCode;

        use crate::{
            app_context::testing::app_context_for_tests,
            routers::common::snapshot::WorkerRegistry,
            workflow::{
                steps::{local::test_support::serve_json_routes, shared::registration_hold},
                WorkflowEngines,
            },
        };

        let url = serve_json_routes(vec![
            ("/health", StatusCode::OK, serde_json::json!({})),
            (
                "/get_server_info",
                StatusCode::OK,
                serde_json::json!({"model_path": "org/model"}),
            ),
            (
                "/v1/models",
                StatusCode::OK,
                serde_json::json!({"data": [{"id": "served"}]}),
            ),
        ])
        .await;
        let router_config = RouterConfig::default();
        let registry = Arc::new(WorkerRegistry::new());
        let context = app_context_for_tests(router_config.clone(), Arc::clone(&registry), None);
        assert!(
            context
                .workflow_engines
                .set(WorkflowEngines::new(&router_config))
                .is_ok(),
            "the engines are set once"
        );
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        assert!(
            context.worker_job_queue.set(Arc::clone(&queue)).is_ok(),
            "the queue is set once"
        );
        let hold = registration_hold::place(&url);

        // The POST: the id it reserves, its job, and a waiter for the job's
        // outcome.
        let id = registry.reserve_id_for_url(&url);
        let mut spec = WorkerSpec::new(url.as_str());
        spec.runtime_type = RuntimeType::Sglang;
        let post = queue.submit_and_wait(
            Job::AddWorker {
                asker: WorkerJobAsker::Api,
                config: Box::new(spec),
                registration_mode: WorkerRegistrationMode::CreateOnly,
            },
            Duration::from_secs(60),
        );
        // The DELETE lands in the window: the registration's cancel is
        // written while its registry step runs, and nothing is registered to
        // remove.
        let delete = async {
            tokio::time::timeout(Duration::from_secs(30), hold.reached.notified())
                .await
                .expect("the registration reaches its registry step");
            assert!(
                registry.get(&id).is_none(),
                "the registry step wrote before its hold"
            );
            let deleted = context
                .worker_service
                .delete_worker(id.as_str())
                .await
                .expect("the DELETE ends a registration in flight");
            assert!(
                registry.get(&id).is_none(),
                "the DELETE found a worker to remove"
            );
            // Whatever the DELETE set in motion is over before the step
            // writes: a removal it queued under the URL (none on this head,
            // where the cancel answers it) must not meet the worker's write.
            while queue.get_status(&url).is_some_and(|status| {
                status.job_type == "RemoveWorker"
                    && matches!(status.status.as_str(), "pending" | "processing")
            }) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            hold.release.notify_one();
            deleted
        };
        let (outcome, deleted) = tokio::join!(post, delete);
        assert_eq!(deleted.url, url);

        // The worker the step wrote leaves through the removal the job's end
        // queued (a job of its own), and the URL is free.
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while (registry.get_by_url(&url).is_some() || queue.get_status(&url).is_some())
            && std::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(
            registry.get_by_url(&url).is_none(),
            "the worker stays under the URL after the DELETE: {outcome:?}"
        );
        assert!(
            registry.get_url_by_id(&id).is_none(),
            "the id still names the URL"
        );
        assert!(
            queue.get_status(&url).is_none(),
            "a job entry is left under the URL"
        );
        let failure = match outcome {
            Err(JobOutcomeError::Failed(reason)) => reason,
            other => panic!("the cancelled registration's job ended otherwise: {other:?}"),
        };
        assert!(
            failure.contains("leave through a queued removal"),
            "the outcome does not name the removal: {failure}"
        );
    }

    /// A `POST`'s registration job whose wait ends before its run cancels
    /// the run: left going, the run probed on and registered a worker that
    /// answered later, under a job that had already failed. The run of any
    /// other registration is left to finish. The cancel finds the run's
    /// state whatever it reads at the wait's end: in the third case a
    /// `DELETE` cancelled the run half a second earlier with a probe still
    /// out, a state the engine's own timeout reclaims. The worker here
    /// accepts and never answers, so the run outlasts the wait.
    #[tokio::test(start_paused = true)]
    async fn only_a_posts_registration_job_ends_the_run_it_stops_waiting_for() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use crate::{
            app_context::testing::app_context_for_tests, routers::common::snapshot::WorkerRegistry,
            workflow::WorkflowEngines,
        };

        for (kind, registration_mode, asker, delay_secs, deleted_at, cancels) in [
            (
                "a POST's",
                WorkerRegistrationMode::CreateOnly,
                WorkerJobAsker::Api,
                0,
                None,
                true,
            ),
            (
                "a startup",
                WorkerRegistrationMode::Upsert,
                WorkerJobAsker::Startup,
                0,
                None,
                false,
            ),
            (
                "a POST's, cancelled by a DELETE as its wait ends,",
                WorkerRegistrationMode::CreateOnly,
                WorkerJobAsker::Api,
                0,
                Some(Duration::from_millis(49_500)),
                true,
            ),
            // The engine counts a step's delay on the real clock, which the
            // paused clock here outruns: this run is cancelled before its
            // first probe, and the case holds the length of the wait alone.
            (
                "a POST's, under a startup delay,",
                WorkerRegistrationMode::CreateOnly,
                WorkerJobAsker::Api,
                5,
                None,
                true,
            ),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("a loopback port of the system's choice");
            let url = format!(
                "http://{}",
                listener.local_addr().expect("the bound address")
            );
            let probes = Arc::new(AtomicUsize::new(0));
            let seen = Arc::clone(&probes);
            #[expect(
                clippy::disallowed_methods,
                reason = "a test fixture: the stub lives as long as the test's runtime"
            )]
            tokio::spawn(async move {
                let mut held = Vec::new();
                while let Ok((connection, _)) = listener.accept().await {
                    seen.fetch_add(1, Ordering::SeqCst);
                    held.push(connection);
                }
            });
            // A startup timeout of twenty seconds: the job waits fifty, and
            // the case's startup delay on top.
            let router_config = RouterConfig {
                worker_startup_timeout_secs: 20,
                worker_startup_check_interval_secs: 2,
                worker_startup_delay_secs: delay_secs,
                ..RouterConfig::default()
            };
            let context =
                app_context_for_tests(router_config.clone(), Arc::new(WorkerRegistry::new()), None);
            assert!(
                context
                    .workflow_engines
                    .set(WorkflowEngines::new(&router_config))
                    .is_ok(),
                "the engines are set once"
            );
            // Each probe outlasts its step's bound of twenty seconds: with
            // the pause of two seconds between probes, the third is out from
            // second 44 to second 64, across the end of the wait.
            let mut spec = WorkerSpec::new(url.as_str());
            spec.health.timeout_secs = Some(30);
            let job = Job::AddWorker {
                asker,
                config: Box::new(spec),
                registration_mode,
            };

            let marks = DashSet::new();
            let (job_end, ()) = tokio::join!(execute_job(&job, &context, &marks, None), async {
                if let Some(at) = deleted_at {
                    tokio::time::sleep(at).await;
                    let engine = &workflow_engines(&context)
                        .expect("set above")
                        .worker_registration;
                    let runs = engine
                        .state_store()
                        .list_all()
                        .await
                        .expect("the store lists its states");
                    engine
                        .cancel_workflow(runs.first().expect("the run's state").instance_id)
                        .await
                        .expect("the run is in the store");
                }
            });
            let failure = job_end.expect_err("the wait ends before the run");

            // A job that cancelled its run has read the run's end, and that
            // read takes the state out of the store.
            let runs_left = workflow_engines(&context)
                .expect("set above")
                .worker_registration
                .state_store()
                .list_all()
                .await
                .expect("the store lists its states")
                .len();
            let probes_at_the_jobs_end = probes.load(Ordering::SeqCst);
            tokio::time::sleep(Duration::from_secs(60)).await;
            let probed_on = probes.load(Ordering::SeqCst) > probes_at_the_jobs_end;
            // The engine's words for a wait that ended first, which a POST's
            // job writes itself.
            let waited_out = format!(
                "Workflow timeout after {}s for {}",
                50 + delay_secs,
                Job::workflow_label(&url)
            );
            let expected = if cancels {
                format!("{waited_out}; its run was cancelled")
            } else {
                waited_out
            };
            assert_eq!(
                (failure.as_str(), probed_on, runs_left),
                (expected.as_str(), !cancels, usize::from(!cancels)),
                "{kind} registration after its job failed"
            );
        }
    }

    /// What `a_delete_setting` builds: the context, its queue and registry,
    /// the worker under the URL and that worker's id.
    type DeleteSetting = (
        Arc<AppContext>,
        Arc<JobQueue>,
        Arc<WorkerRegistry>,
        Arc<dyn Worker>,
        WorkerId,
    );

    /// A running queue over live engines, a registry, and a worker under
    /// `url` with no drain window, the setting a DELETE's kept removal runs in.
    fn a_delete_setting(url: &str) -> DeleteSetting {
        use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
        use smg_worker::identity::basic::{Worker, WorkerType};

        use crate::{
            app_context::testing::app_context_for_tests,
            routers::{common::snapshot::WorkerRegistry, grpc::backend_client::BasicWorkerBuilder},
            workflow::WorkflowEngines,
        };
        let router_config = RouterConfig::default();
        let registry = Arc::new(WorkerRegistry::new());
        let context = app_context_for_tests(router_config.clone(), Arc::clone(&registry), None);
        assert!(
            context
                .workflow_engines
                .set(WorkflowEngines::new(&router_config))
                .is_ok(),
            "the engines are set once"
        );
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        assert!(
            context.worker_job_queue.set(Arc::clone(&queue)).is_ok(),
            "the queue is set once"
        );
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("served"))
                .health_config(HealthCheckConfig {
                    drain_settle_secs: 0,
                    ..Default::default()
                })
                .build(),
        );
        let id = registry
            .register(Arc::clone(&worker))
            .expect("the worker registers");
        (context, queue, registry, worker, id)
    }

    /// A DELETE's removal kept behind a removal in flight removes the worker
    /// the DELETE saw once it runs: the worker a guarded cleanup skipped as
    /// replaced since its snapshot leaves through the DELETE's own removal.
    #[tokio::test]
    async fn a_kept_delete_removal_removes_the_worker_the_delete_saw() {
        let url = "http://10.0.0.9:8000";
        let (context, queue, registry, _worker, id) = a_delete_setting(url);
        QueueTest::hold_url_for_tests(&queue, url, "RemoveWorker");
        context
            .worker_service
            .delete_worker(id.as_str())
            .await
            .expect("the DELETE is admitted");
        assert!(
            QueueTest::kept_for(&queue, url),
            "the DELETE's removal waits for the holder"
        );
        QueueTest::record_job_completion(
            "RemoveWorker",
            url,
            Duration::ZERO,
            &Ok("done".to_string()),
            QueueTest::state(&queue).status_map,
        );
        QueueTest::run_kept(
            url,
            &context,
            QueueTest::state(&queue).status_map,
            QueueTest::state(&queue).kept,
            QueueTest::state(&queue).cancelled_before_start,
        )
        .await;
        assert!(!QueueTest::kept_for(&queue, url), "the kept removal ran");
        assert!(
            registry.get_by_url(url).is_none(),
            "the DELETE's removal removed the worker it saw"
        );
    }

    /// A DELETE's removal kept behind a job in flight is pinned to the ids
    /// the DELETE saw and to no revision: the job it waited for (an update
    /// here; a PUT's replacement and a label refresh rebuild the same way)
    /// replaces the worker under its id, and the kept removal takes the
    /// rebuild out, where a revision pin skipped it every time.
    #[tokio::test]
    async fn a_kept_delete_removal_removes_the_worker_rebuilt_under_its_id_since_the_delete() {
        use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
        use smg_worker::identity::basic::{Worker, WorkerType};

        use crate::routers::grpc::backend_client::BasicWorkerBuilder;
        let url = "http://10.0.0.10:8000";
        let (context, queue, registry, first, id) = a_delete_setting(url);
        QueueTest::hold_url_for_tests(&queue, url, "UpdateWorker");
        context
            .worker_service
            .delete_worker(id.as_str())
            .await
            .expect("the DELETE is admitted");
        assert!(
            QueueTest::kept_for(&queue, url),
            "the DELETE's removal waits for the holder"
        );
        // The update rebuilds the worker under its id before the kept removal
        // runs.
        let rebuilt: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("served"))
                .health_config(HealthCheckConfig {
                    drain_settle_secs: 0,
                    ..Default::default()
                })
                .build(),
        );
        assert!(
            registry.replace_claiming_local(&id, Arc::clone(&rebuilt), first.revision()),
            "the update replaces the worker"
        );
        QueueTest::record_job_completion(
            "UpdateWorker",
            url,
            Duration::ZERO,
            &Ok("done".to_string()),
            QueueTest::state(&queue).status_map,
        );
        QueueTest::run_kept(
            url,
            &context,
            QueueTest::state(&queue).status_map,
            QueueTest::state(&queue).kept,
            QueueTest::state(&queue).cancelled_before_start,
        )
        .await;
        assert!(!QueueTest::kept_for(&queue, url), "the kept removal ran");
        assert!(
            registry.get_by_url(url).is_none(),
            "the DELETE's removal removed the worker rebuilt under the id it named"
        );
    }

    /// A DELETE's removal kept behind a job in flight leaves a worker
    /// registered under a fresh id since the DELETE: the holder took the
    /// worker the DELETE named out, and the registration that followed under
    /// the URL is not the one the DELETE named.
    #[tokio::test]
    async fn a_kept_delete_removal_leaves_a_worker_registered_under_a_fresh_id_since_the_delete() {
        use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
        use smg_worker::identity::basic::{Worker, WorkerType};

        use crate::routers::grpc::backend_client::BasicWorkerBuilder;
        let url = "http://10.0.0.11:8000";
        let (context, queue, registry, _first, id) = a_delete_setting(url);
        QueueTest::hold_url_for_tests(&queue, url, "RemoveWorker");
        context
            .worker_service
            .delete_worker(id.as_str())
            .await
            .expect("the DELETE is admitted");
        assert!(
            QueueTest::kept_for(&queue, url),
            "the DELETE's removal waits for the holder"
        );
        // The holder takes the worker out, and a registration lands under the
        // URL with a fresh id before the kept removal runs.
        assert!(
            registry.remove(&id).is_some(),
            "the holder removes the worker the DELETE named"
        );
        let fresh: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("served"))
                .health_config(HealthCheckConfig {
                    drain_settle_secs: 0,
                    ..Default::default()
                })
                .build(),
        );
        let fresh_id = registry
            .register(Arc::clone(&fresh))
            .expect("the fresh registration lands");
        assert_ne!(
            fresh_id.as_str(),
            id.as_str(),
            "the registration has a fresh id"
        );
        QueueTest::record_job_completion(
            "RemoveWorker",
            url,
            Duration::ZERO,
            &Ok("done".to_string()),
            QueueTest::state(&queue).status_map,
        );
        QueueTest::run_kept(
            url,
            &context,
            QueueTest::state(&queue).status_map,
            QueueTest::state(&queue).kept,
            QueueTest::state(&queue).cancelled_before_start,
        )
        .await;
        assert!(!QueueTest::kept_for(&queue, url), "the kept removal ran");
        let standing = registry
            .get_by_url(url)
            .expect("the fresh worker stays under the URL");
        assert!(
            Arc::ptr_eq(&standing, &fresh),
            "the worker standing is the fresh registration"
        );
    }

    fn removal(url: &str, pins: Option<&[(&str, Option<u64>)]>) -> Job {
        Job::RemoveWorker {
            asker: WorkerJobAsker::Api,
            url: url.to_string(),
            expected_revisions: pins.map(|pins| {
                pins.iter()
                    .map(|(id, revision)| (id.to_string(), *revision))
                    .collect()
            }),
        }
    }

    fn pins(pins: &[(&str, Option<u64>)]) -> HashMap<String, Option<u64>> {
        pins.iter()
            .map(|(id, revision)| (id.to_string(), *revision))
            .collect()
    }

    fn kept_removal_pins(
        kept: &DashMap<String, Kept<Job>>,
        url: &str,
    ) -> Option<HashMap<String, Option<u64>>> {
        let slots = kept.get(url).expect("an entry kept for the URL");
        let Some(Job::RemoveWorker {
            expected_revisions, ..
        }) = slots.removals.front()
        else {
            panic!("a removal kept for the URL");
        };
        expected_revisions.clone()
    }

    #[test]
    fn a_removal_kept_over_a_kept_removal_merges_the_pins_the_wider_pin_staying() {
        let kept = DashMap::new();
        let url = "http://w:8000";
        QueueTest::keep(
            &kept,
            url,
            removal(url, Some(&[("a", Some(1)), ("b", None)])),
        );
        QueueTest::keep(
            &kept,
            url,
            removal(url, Some(&[("b", Some(2)), ("c", Some(3))])),
        );
        assert_eq!(
            kept_removal_pins(&kept, url),
            Some(pins(&[("a", Some(1)), ("b", None), ("c", Some(3))]))
        );
        // A `None` pin on the newer side widens the held one.
        QueueTest::keep(&kept, url, removal(url, Some(&[("a", None)])));
        assert_eq!(
            kept_removal_pins(&kept, url),
            Some(pins(&[("a", None), ("b", None), ("c", Some(3))]))
        );
    }

    #[test]
    fn an_unguarded_removal_kept_in_either_order_leaves_the_kept_removal_unguarded() {
        let url = "http://w:8000";
        // The `DELETE`'s unguarded removal first, the cleanup's pins over it.
        let kept = DashMap::new();
        QueueTest::keep(&kept, url, removal(url, None));
        QueueTest::keep(&kept, url, removal(url, Some(&[("a", Some(1))])));
        assert_eq!(kept_removal_pins(&kept, url), None);
        // The cleanup's pins first, the `DELETE`'s unguarded removal over them.
        let kept = DashMap::new();
        QueueTest::keep(&kept, url, removal(url, Some(&[("a", Some(1))])));
        QueueTest::keep(&kept, url, removal(url, None));
        assert_eq!(kept_removal_pins(&kept, url), None);
    }

    #[test]
    fn a_removal_put_back_merges_with_the_removal_kept_since() {
        let kept = DashMap::new();
        let url = "http://w:8000";
        QueueTest::keep(&kept, url, removal(url, Some(&[("b", Some(2))])));
        QueueTest::put_back(&kept, url, removal(url, Some(&[("a", Some(1))])));
        assert_eq!(
            kept_removal_pins(&kept, url),
            Some(pins(&[("a", Some(1)), ("b", Some(2))]))
        );
    }

    /// The health manager's removal reaches the removal workflow under the
    /// revisions it names: a worker named at another revision stays
    /// registered, and it goes once it is named at its own.
    #[tokio::test]
    async fn a_health_removal_runs_under_the_revisions_it_names() {
        use crate::{
            app_context::testing::live_app_context,
            routers::grpc::backend_client::BasicWorkerBuilder,
        };

        async fn ended(queue: &JobQueue, url: &str) {
            while queue
                .get_status(url)
                .is_some_and(|status| matches!(status.status.as_str(), "pending" | "processing"))
            {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }

        let context = live_app_context();
        let registry = &context.worker_registry;
        let queue = context.worker_job_queue.get().expect("the live queue");
        let url = "http://w:8000";
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new(url).build());
        let id = registry
            .register(Arc::clone(&worker))
            .expect("the worker registers");
        for (revision, stays) in [(worker.revision() + 1, true), (worker.revision(), false)] {
            queue
                .remove_worker(url.to_string(), &id, revision, WorkerJobAsker::Health)
                .await
                .expect("the removal is queued");
            tokio::time::timeout(Duration::from_secs(30), ended(queue, url))
                .await
                .expect("the removal ends");
            assert_eq!(
                registry.get_by_url(url).is_some(),
                stays,
                "named at revision {revision}"
            );
        }
    }

    /// A removal job lasts as long as its run. A worker's own drain window
    /// past the old wait of 630 seconds ends in the removal, and the job
    /// holds the URL until then: it reports no failure while its run still
    /// means to remove the worker.
    #[tokio::test(start_paused = true)]
    async fn a_removal_job_holds_its_url_through_a_long_drain_window() {
        use openai_protocol::worker::HealthCheckConfig;
        use smg_worker::identity::basic::Worker;

        use crate::{
            app_context::testing::live_app_context,
            routers::grpc::backend_client::BasicWorkerBuilder,
        };

        let context = live_app_context();
        let registry = &context.worker_registry;
        let queue = context.worker_job_queue.get().expect("the live queue");
        let url = "http://w:8000";
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .health_config(HealthCheckConfig {
                    drain_settle_secs: 900,
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        registry
            .register(Arc::clone(&worker))
            .expect("the worker registers");
        queue
            .submit(removal(url, None))
            .await
            .expect("the removal is queued");

        tokio::time::sleep(Duration::from_secs(700)).await;
        let status = queue.get_status(url).expect("the job holds the URL");
        assert_eq!(status.status, "processing");
        assert!(
            registry.get_by_url(url).is_some(),
            "the worker drains through its window"
        );

        tokio::time::sleep(Duration::from_secs(300)).await;
        assert!(
            queue.get_status(url).is_none(),
            "the job ended without a failure"
        );
        assert!(
            registry.get_by_url(url).is_none(),
            "the removal took the worker"
        );
    }

    /// The wait counts every step: the selection and the registry removal,
    /// 10 seconds each, the settle bound, then the policy cleanup under the
    /// workflow's default of 300 seconds and a 10 second tokenizer release.
    #[test]
    fn a_removal_job_waits_for_the_bound_of_every_step() {
        use smg_worker::workflow::SETTLE_STEP_TIMEOUT;

        assert_eq!(
            removal_wait(),
            SETTLE_STEP_TIMEOUT + Duration::from_secs(330)
        );
    }

    /// A step that may run twice counts twice, so the wait holds through a
    /// retry of the policy cleanup.
    #[test]
    fn a_runs_bound_counts_every_attempt_of_a_step() {
        use wfaas::{BackoffStrategy, RetryPolicy, StepId};

        let once = run_bound(&create_worker_removal_workflow());
        // The cleanup step's attempts, and its bounds past the first. The
        // engine runs a step once even under a policy of no attempts.
        for (attempts, more) in [(2, 1), (1, 0), (0, 0)] {
            let mut workflow = create_worker_removal_workflow();
            let cleanup = workflow
                .steps
                .iter_mut()
                .find(|step| step.id == StepId::new("remove_from_policy_registry"))
                .expect("the policy cleanup step");
            cleanup.retry_policy = Some(RetryPolicy {
                max_attempts: attempts,
                backoff: BackoffStrategy::Fixed(Duration::ZERO),
            });

            assert_eq!(
                run_bound(&workflow),
                once + Duration::from_secs(300) * more,
                "{attempts} attempts"
            );
        }
    }

    /// The wait adds up bounds, and a step that retries without end has none.
    #[test]
    fn no_removal_step_retries_without_end() {
        use wfaas::FailureAction;

        for step in &create_worker_removal_workflow().steps {
            assert!(
                step.on_failure != FailureAction::RetryIndefinitely,
                "{} retries without end",
                step.name
            );
        }
    }
}
