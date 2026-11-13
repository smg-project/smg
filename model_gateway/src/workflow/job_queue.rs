//! Async job queue for control plane operations
//!
//! Provides non-blocking worker management by queuing operations and processing
//! them asynchronously in background worker tasks.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Weak,
    },
    time::{Duration, SystemTime},
};

use dashmap::{mapref::entry::Entry, DashMap};
use openai_protocol::worker::{
    ConnectionMode, JobStatus, RuntimeType, WorkerSpec, WorkerType, WorkerUpdateRequest,
};
use smg_mcp::McpConfig;
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{debug, error, info, warn};
use wfaas::WorkflowId;

use crate::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    worker::endpoint::Endpoint,
    workflow::{
        create_mcp_workflow_data, create_tokenizer_workflow_data,
        create_wasm_registration_workflow_data, create_wasm_removal_workflow_data,
        create_worker_removal_workflow_data, create_worker_update_workflow_data,
        create_worker_workflow_data, note_tokenizer_load_outcome, note_tokenizer_removed,
        McpServerConfigRequest, TokenizerConfigRequest, TokenizerRemovalRequest,
        WasmModuleConfigRequest, WasmModuleRemovalRequest, WorkerRegistrationMode,
    },
};

/// Job types for control plane operations
#[derive(Debug, Clone)]
pub enum Job {
    AddWorker {
        config: Box<WorkerSpec>,
        registration_mode: WorkerRegistrationMode,
    },
    UpdateWorker {
        url: String,
        update: Box<WorkerUpdateRequest>,
    },
    /// Re-read a registered worker's labels from its engine and rebuild the
    /// worker if they changed: the worker came into service, as a restarted
    /// engine on the same address or one that finished loading after it was
    /// registered on provisional information. Runs through the update
    /// workflow.
    RefreshWorkerLabels {
        url: String,
    },
    RemoveWorker {
        url: String,
        /// Per-worker revision guards keyed by registry worker id. A DP group
        /// shares one canonical URL but each rank holds its own revision, so
        /// one scalar cannot guard the group. `None` removes every match
        /// unguarded.
        expected_revisions: Option<HashMap<String, u64>>,
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
    /// Get job type as string for logging
    pub fn job_type(&self) -> &'static str {
        match self {
            Job::AddWorker { .. } => "AddWorker",
            Job::UpdateWorker { .. } => "UpdateWorker",
            Job::RefreshWorkerLabels { .. } => "RefreshWorkerLabels",
            Job::RemoveWorker { .. } => "RemoveWorker",
            Job::InitializeWorkersFromConfig { .. } => "InitializeWorkersFromConfig",
            Job::InitializeMcpServers { .. } => "InitializeMcpServers",
            Job::RegisterMcpServer { .. } => "RegisterMcpServer",
            Job::AddWasmModule { .. } => "AddWasmModule",
            Job::RemoveWasmModule { .. } => "RemoveWasmModule",
            Job::AddTokenizer { .. } => "AddTokenizer",
            Job::RemoveTokenizer { .. } => "RemoveTokenizer",
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
            | "AddTokenizer"
            | "RemoveTokenizer" => worker_url.to_string(),
            _ => Endpoint::redacted_str(worker_url),
        }
    }

    /// Get worker URL, MCP server name, WASM module, or tokenizer identifier for logging and status tracking
    pub fn worker_url(&self) -> &str {
        match self {
            Job::AddWorker { config, .. } => &config.url,
            Job::UpdateWorker { url, .. } => url,
            Job::RefreshWorkerLabels { url } => url,
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

/// Job queue configuration
#[derive(Clone, Debug)]
pub struct JobQueueConfig {
    /// Maximum pending jobs in queue
    pub queue_capacity: usize,
    /// Maximum number of jobs executing concurrently
    pub max_concurrent_jobs: usize,
}

impl Default for JobQueueConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1000,
            max_concurrent_jobs: 200,
        }
    }
}

/// Job queue manager for worker validation and removal operations
pub struct JobQueue {
    /// Channel for submitting jobs, each with the id of the waiter for its
    /// outcome when a caller waits on it
    tx: mpsc::Sender<Queued>,
    /// Weak reference to AppContext to avoid circular dependencies
    context: Weak<AppContext>,
    /// Job status tracking by worker URL
    status_map: Arc<DashMap<String, JobStatus>>,
    /// Semaphore to limit concurrent job execution
    concurrency_limit: Arc<Semaphore>,
    /// The waiters of [`Self::submit_and_wait`], by job key: a job's outcome
    /// is handed to its waiter when the job ends.
    outcomes: Outcomes,
    /// The next waiter's id: what tells a caller's waiter from a later one
    /// under the same key.
    next_waiter_id: AtomicU64,
    /// One job per URL, kept while another holds the URL
    /// (`submit_or_defer`) and run by that job's completion (`run_kept`).
    kept: Arc<DashMap<String, Job>>,
}

/// A job handed to the dispatcher with the id of the waiter for its outcome,
/// when a caller waits on it: the outcome goes to that waiter alone, never to
/// a later caller's under the same key.
struct Queued {
    job: Job,
    waiter: Option<u64>,
}

/// The caller waiting on a job's outcome under the job's key, one per key
/// at a time; `id` is the caller's own, so a caller leaving takes its own
/// entry alone.
struct Waiter {
    id: u64,
    sender: oneshot::Sender<Result<String, String>>,
}

/// The waiters for a job's outcome, by job key.
type Outcomes = Arc<DashMap<String, Waiter>>;

/// Removes the caller's waiter when the caller leaves
/// [`JobQueue::submit_and_wait`] by any path: a submit refused, a wait
/// cancelled while the queue is full, a timeout, or the outcome delivered
/// (which removed it already). The entry goes only while it is still this
/// caller's, so a later waiter under the same key stays.
struct WaiterSlot {
    outcomes: Outcomes,
    key: String,
    id: u64,
}

impl Drop for WaiterSlot {
    fn drop(&mut self) {
        self.outcomes
            .remove_if(&self.key, |_, waiter| waiter.id == self.id);
    }
}

/// Why [`JobQueue::submit_and_wait`] returned no success message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobOutcomeError {
    /// The queue took no job: full, or shutting down.
    NotQueued(String),
    /// The job ran and failed, with its reason.
    Failed(String),
    /// The job did not end within the wait; it keeps running.
    TimedOut(Duration),
    /// A caller is waiting on a job under this key already: the job runs
    /// once and its outcome goes to that caller. The key is written as a
    /// log line writes it, a worker's address with its userinfo masked.
    InFlight(String),
}

impl std::fmt::Display for JobOutcomeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotQueued(reason) => write!(f, "Failed to submit job: {reason}"),
            Self::Failed(reason) => f.write_str(reason),
            Self::TimedOut(wait) => write!(f, "Job timeout after {}s", wait.as_secs()),
            Self::InFlight(key) => write!(f, "A job for {key} is in flight; wait for its outcome"),
        }
    }
}

impl std::error::Error for JobOutcomeError {}

impl std::fmt::Debug for JobQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobQueue")
            .field("status_count", &self.status_map.len())
            .finish()
    }
}

impl JobQueue {
    /// Create a new job queue with semaphore-based concurrency control
    ///
    /// Takes a Weak reference to AppContext to avoid circular strong references.
    /// Spawns a single dispatcher task that spawns individual job tasks with semaphore control.
    pub fn new(config: JobQueueConfig, context: Weak<AppContext>) -> Arc<Self> {
        let (tx, mut rx) = mpsc::channel(config.queue_capacity);

        debug!(
            "Initializing job queue: capacity={}, max_concurrent={}",
            config.queue_capacity, config.max_concurrent_jobs
        );

        let status_map = Arc::new(DashMap::new());
        let concurrency_limit = Arc::new(Semaphore::new(config.max_concurrent_jobs));
        let outcomes: Outcomes = Arc::new(DashMap::new());
        let kept: Arc<DashMap<String, Job>> = Arc::new(DashMap::new());

        let queue = Arc::new(Self {
            tx,
            context: context.clone(),
            status_map: status_map.clone(),
            concurrency_limit: concurrency_limit.clone(),
            outcomes: outcomes.clone(),
            next_waiter_id: AtomicU64::new(1),
            kept: kept.clone(),
        });

        // Single dispatcher task
        let ctx = context.clone();
        let status = status_map.clone();
        let sem = concurrency_limit.clone();
        let waiters = outcomes.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "Core job dispatcher loop: runs for the lifetime of the gateway and drains cleanly when the channel sender is dropped on shutdown"
        )]
        tokio::spawn(async move {
            while let Some(queued) = rx.recv().await {
                // Acquire permit (blocks if at concurrency limit)
                let Ok(permit) = sem.clone().acquire_owned().await else {
                    error!("Semaphore closed, stopping dispatcher");
                    break;
                };

                let ctx_clone = ctx.clone();
                let status_clone = status.clone();
                let waiters_clone = waiters.clone();
                let kept_clone = kept.clone();

                #[expect(
                    clippy::disallowed_methods,
                    reason = "Job processing task: bounded by the semaphore permit which is dropped when the task completes"
                )]
                tokio::spawn(async move {
                    Self::process_job(
                        queued,
                        ctx_clone,
                        status_clone,
                        waiters_clone,
                        kept_clone,
                        permit,
                    )
                    .await;
                });
            }

            debug!("Job dispatcher stopped");
        });

        // Spawn cleanup task for old job statuses (TTL 5 minutes)
        let cleanup_status_map = status_map.clone();
        #[expect(
            clippy::disallowed_methods,
            reason = "Background cleanup loop: runs periodically to evict expired job statuses, bounded by the DashMap's own TTL logic"
        )]
        tokio::spawn(async move {
            Self::cleanup_old_statuses(cleanup_status_map).await;
        });

        queue
    }

    /// Get current queue and concurrency status
    pub fn get_load_info(&self) -> (usize, usize) {
        let queue_depth = self.tx.max_capacity() - self.tx.capacity();
        let available_permits = self.concurrency_limit.available_permits();
        (queue_depth, available_permits)
    }

    /// Submit unless a job for this URL is already pending or processing.
    /// Returns `Ok(false)` when an earlier submission is still in flight. The
    /// status-map entry is taken atomically, so concurrent callers enqueue
    /// exactly one job per URL (#1533).
    pub async fn submit_if_idle(&self, job: Job) -> Result<bool, String> {
        let worker_url = job.worker_url().to_string();
        let job_type = job.job_type();
        if !Self::claim(&self.status_map, &worker_url, job_type) {
            return Ok(false);
        }
        self.submit_inner(job, true, None)
            .await
            .map(|()| true)
            .inspect_err(|_| {
                // A failed submission owns no job, so its claim must not block the
                // next one; only its own entry goes.
                Self::record_own_status(&self.status_map, &worker_url, job_type, None);
            })
    }

    /// Only an in-flight job of the same type counts: the status map is keyed
    /// by name across job types, so an MCP server or WASM module named like a
    /// worker URL must not block that worker's AddWorker.
    fn claim(status_map: &DashMap<String, JobStatus>, url: &str, job_type: &'static str) -> bool {
        match status_map.entry(url.to_string()) {
            Entry::Occupied(e)
                if e.get().job_type == job_type
                    && matches!(e.get().status.as_str(), "pending" | "processing") =>
            {
                false
            }
            entry => {
                entry.insert(JobStatus::pending(job_type, url));
                true
            }
        }
    }

    /// Submit unless a job of any type is pending or processing for this URL.
    /// For a best-effort submission that must not take an in-flight job's
    /// status under the key: a label refresh while the worker's registration
    /// is still processing would take that job's entry (and let the in-flight
    /// guard admit a second registration). Returns `Ok(false)` when one is in
    /// flight; the caller's next occasion submits again. A job of another type
    /// submitted after the claim keeps its entry too: the pending status
    /// written once a slot is held, the rollback of a failed submission and a
    /// job's later status writes touch only this job's own entry
    /// (`record_own_status`).
    pub async fn submit_unless_in_flight(&self, job: Job) -> Result<bool, String> {
        let worker_url = job.worker_url().to_string();
        let job_type = job.job_type();
        if !Self::claim_unless_in_flight(&self.status_map, &worker_url, job_type) {
            return Ok(false);
        }
        self.submit_inner(job, true, None)
            .await
            .map(|()| true)
            .inspect_err(|_| {
                // A failed submission owns no job, so its claim must not block the
                // next one; only its own entry goes (another job's, taken during
                // the wait for a slot, stays with that job).
                Self::record_own_status(&self.status_map, &worker_url, job_type, None);
            })
    }

    /// `submit_unless_in_flight` for a submission that must not be lost: a
    /// job that finds another of any type in flight under its URL is kept
    /// (one per URL, the latest) and run by that job's completion
    /// (`run_kept`), after it and never beside it. Returns `Ok(false)` for a
    /// kept job. A job still kept once nothing is in flight (the holder ended
    /// between the claim and the keeping, its completion finding nothing) is
    /// this call's to submit.
    pub async fn submit_or_defer(&self, mut job: Job) -> Result<bool, String> {
        let worker_url = job.worker_url().to_string();
        let job_type = job.job_type();
        loop {
            if Self::claim_unless_in_flight(&self.status_map, &worker_url, job_type) {
                return self
                    .submit_inner(job, true, None)
                    .await
                    .map(|()| true)
                    .inspect_err(|_| {
                        Self::record_own_status(&self.status_map, &worker_url, job_type, None);
                    });
            }
            self.kept.insert(worker_url.clone(), job);
            if Self::has_in_flight(&self.status_map, &worker_url) {
                return Ok(false);
            }
            match self.kept.remove(&worker_url) {
                Some((_, taken_back)) => job = taken_back,
                None => return Ok(false),
            }
        }
    }

    /// Whether a job of any type is pending or processing under `url`.
    fn has_in_flight(status_map: &DashMap<String, JobStatus>, url: &str) -> bool {
        status_map
            .get(url)
            .is_some_and(|entry| matches!(entry.status.as_str(), "pending" | "processing"))
    }

    /// The job kept for `worker_url` (`submit_or_defer`), run on the heels of
    /// the job that held the URL, on its task and under its own status; a job
    /// queued for the URL in the meantime holds it first, and its completion
    /// runs the kept one. Nothing waits on a kept job's outcome.
    async fn run_kept(
        worker_url: &str,
        context: &Arc<AppContext>,
        status_map: &Arc<DashMap<String, JobStatus>>,
        kept: &DashMap<String, Job>,
    ) {
        while let Some((_, job)) = kept.remove(worker_url) {
            let job_type = job.job_type();
            if !Self::claim_unless_in_flight(status_map, worker_url, job_type) {
                kept.insert(worker_url.to_string(), job);
                if Self::has_in_flight(status_map, worker_url) {
                    return;
                }
                continue;
            }
            debug!(
                "Running the {} job kept for worker={} while another job held the URL",
                job_type,
                Job::shown_worker(job_type, worker_url)
            );
            Self::record_own_status(
                status_map,
                worker_url,
                job_type,
                Some(JobStatus::processing(job_type, worker_url)),
            );
            let start = std::time::Instant::now();
            let result = Self::execute_job(&job, context).await;
            Self::record_job_completion(job_type, worker_url, start.elapsed(), &result, status_map);
        }
    }

    /// A job of `job_type` written as processing under `url`, standing in for
    /// one the dispatcher is running.
    #[cfg(test)]
    pub(crate) fn hold_url_for_tests(&self, url: &str, job_type: &'static str) {
        self.status_map
            .insert(url.to_string(), JobStatus::processing(job_type, url));
    }

    /// Whether a job is kept for `url` (`submit_or_defer`).
    #[cfg(test)]
    pub(crate) fn kept_for(&self, url: &str) -> bool {
        self.kept.contains_key(url)
    }

    /// Any in-flight job under the key blocks the claim, whatever its type.
    fn claim_unless_in_flight(
        status_map: &DashMap<String, JobStatus>,
        url: &str,
        job_type: &'static str,
    ) -> bool {
        match status_map.entry(url.to_string()) {
            Entry::Occupied(e) if matches!(e.get().status.as_str(), "pending" | "processing") => {
                false
            }
            entry => {
                entry.insert(JobStatus::pending(job_type, url));
                true
            }
        }
    }

    /// Submit a job with detailed queue status
    pub async fn submit(&self, job: Job) -> Result<(), String> {
        self.submit_inner(job, false, None).await
    }

    /// `submit`, with the pending status written over this job's own claim
    /// alone when `claimed` ([`Self::submit_if_idle`],
    /// [`Self::submit_unless_in_flight`]): a job of another type that took the
    /// URL during the wait for a slot keeps its entry, where a plain submission
    /// writes its own over whatever is there. `waiter` is the id of the caller
    /// waiting on the outcome, [`Self::submit_and_wait`]'s own; none otherwise.
    async fn submit_inner(
        &self,
        job: Job,
        claimed: bool,
        waiter: Option<u64>,
    ) -> Result<(), String> {
        // Check if context is still alive before accepting jobs
        let Some(context) = self.context.upgrade() else {
            return Err("Job queue shutting down: AppContext dropped".to_string());
        };

        // Extract values before moving job
        let job_type = job.job_type();
        let worker_url = job.worker_url().to_string();
        let tokenizer_load = match &job {
            Job::AddTokenizer { config } => Some(config.name.clone()),
            _ => None,
        };

        // A slot in the queue first: the wait for one (a full queue) can be
        // cancelled with the caller and must leave nothing behind, so nothing
        // is recorded before a slot is held; from the slot on, the hand-over
        // is synchronous and cannot fail.
        let Ok(permit) = self.tx.reserve().await else {
            let (queue_depth, _) = self.get_load_info();
            return Err(format!(
                "Job queue full: {} jobs pending (capacity: {})",
                queue_depth,
                self.tx.max_capacity()
            ));
        };

        // Record pending status
        Self::write_pending(&self.status_map, &worker_url, job_type, claimed);
        // A tokenizer load counts as in flight for its model from here (the
        // model is pending until the job ends, whatever was recorded): with
        // the slot held, the job is handed over right after the count, so a
        // job that ends at once finds its count to release.
        if let Some(model) = &tokenizer_load {
            context.tokenizer_load_failures.load_started(model);
        }
        permit.send(Queued { job, waiter });

        let (queue_depth, available_permits) = self.get_load_info();
        debug!(
            "Job submitted: type={}, worker={}, queue_depth={}, available_slots={}",
            job_type,
            Job::shown_worker(job_type, &worker_url),
            queue_depth,
            available_permits
        );
        Ok(())
    }

    /// Submit `job` and wait for its outcome: the job's own success message
    /// or its failure, as the status map records them, without a poll.
    /// `timeout` bounds the wait; a job that outlives it keeps running and
    /// its outcome is dropped.
    pub async fn submit_and_wait(
        &self,
        job: Job,
        timeout: Duration,
    ) -> Result<String, JobOutcomeError> {
        let key = job.worker_url().to_string();
        let (tx, rx) = oneshot::channel();
        let id = self.next_waiter_id.fetch_add(1, Ordering::Relaxed);
        // One waiter per key: a second caller for a job in flight is refused,
        // not handed the first caller's outcome (and the first its loss).
        match self.outcomes.entry(key.clone()) {
            Entry::Occupied(_) => {
                let shown = Job::shown_worker(job.job_type(), &key);
                return Err(JobOutcomeError::InFlight(shown));
            }
            Entry::Vacant(entry) => {
                entry.insert(Waiter { id, sender: tx });
            }
        }
        // The waiter leaves with the caller, whichever way the caller leaves.
        let _slot = WaiterSlot {
            outcomes: Arc::clone(&self.outcomes),
            key,
            id,
        };
        if let Err(reason) = self.submit_inner(job, false, Some(id)).await {
            return Err(JobOutcomeError::NotQueued(reason));
        }
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(Ok(message))) => Ok(message),
            Ok(Ok(Err(reason))) => Err(JobOutcomeError::Failed(reason)),
            Ok(Err(_)) => Err(JobOutcomeError::Failed(
                "the job ended without an outcome".to_string(),
            )),
            Err(_) => Err(JobOutcomeError::TimedOut(timeout)),
        }
    }

    /// Hands a finished job's outcome to the waiter the job was queued for,
    /// while that waiter is still under the key. A job queued without a
    /// waiter, or whose waiter left and a later caller took the key, delivers
    /// nothing; a waiter that gave up (its receiver dropped) is a no-op.
    fn deliver_outcome(
        outcomes: &Outcomes,
        key: &str,
        waiter: Option<u64>,
        outcome: Result<String, String>,
    ) {
        let Some(id) = waiter else { return };
        if let Some((_, waiter)) = outcomes.remove_if(key, |_, waiter| waiter.id == id) {
            let _ = waiter.sender.send(outcome);
        }
    }

    /// Get job status by worker URL
    pub fn get_status(&self, worker_url: &str) -> Option<JobStatus> {
        self.status_map.get(worker_url).map(|entry| entry.clone())
    }

    /// Remove job status (called when worker is deleted)
    pub fn remove_status(&self, worker_url: &str) {
        self.status_map.remove(worker_url);
    }

    /// Process a single job with status tracking and error handling
    async fn process_job(
        queued: Queued,
        context: Weak<AppContext>,
        status_map: Arc<DashMap<String, JobStatus>>,
        outcomes: Outcomes,
        kept: Arc<DashMap<String, Job>>,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        let Queued { job, waiter } = queued;
        let job_type = job.job_type();
        let worker_url = job.worker_url().to_string();
        let start = std::time::Instant::now();

        // Update to processing: this job's entry only (a job of another type
        // submitted for the URL since this one was queued keeps its status).
        Self::record_own_status(
            &status_map,
            &worker_url,
            job_type,
            Some(JobStatus::processing(job_type, &worker_url)),
        );

        debug!(
            "Processing job: type={}, worker={}",
            job_type,
            Job::shown_worker(job_type, &worker_url)
        );

        // Release concurrency slot immediately. The semaphore bounds how many
        // jobs can be dequeued concurrently (preventing thundering herd), but
        // long-running waits (e.g. 30-min worker health checks) must not hold
        // a slot or they starve the queue for all other job types.
        drop(permit);

        // Execute job
        match context.upgrade() {
            Some(ctx) => {
                let result = Self::execute_job(&job, &ctx).await;
                let duration = start.elapsed();
                // Only `POST /workers` (CreateOnly) reserves an id for its 202; a failed
                // upsert from discovery or startup owns no reservation to drop. Release
                // before the status turns terminal, so a retry cannot reserve the same
                // mapping in between and have it deleted from under it (#1533).
                if result.is_err()
                    && matches!(
                        job,
                        Job::AddWorker {
                            registration_mode: WorkerRegistrationMode::CreateOnly,
                            ..
                        }
                    )
                {
                    ctx.worker_registry.release_reservation(&worker_url);
                }
                Self::record_job_completion(job_type, &worker_url, duration, &result, &status_map);
                Self::deliver_outcome(&outcomes, &worker_url, waiter, result);
                Self::run_kept(&worker_url, &ctx, &status_map, &kept).await;
            }
            None => {
                let error_msg = "AppContext dropped".to_string();
                status_map.insert(
                    worker_url.clone(),
                    JobStatus::failed(job_type, &worker_url, error_msg.clone()),
                );
                Self::deliver_outcome(&outcomes, &worker_url, waiter, Err(error_msg));
                error!(
                    "AppContext dropped, cannot process job: type={}, worker={}",
                    job_type,
                    Job::shown_worker(job_type, &worker_url)
                );
            }
        }
    }

    /// Execute a specific job
    async fn execute_job(job: &Job, context: &Arc<AppContext>) -> Result<String, String> {
        match job {
            Job::AddWorker {
                config,
                registration_mode,
            } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let timeout_duration =
                    Duration::from_secs(context.router_config.worker_startup_timeout_secs + 30);

                let workflow_data = create_worker_workflow_data(
                    (**config).clone(),
                    registration_mode.clone(),
                    Arc::clone(context),
                );
                let instance_id = engines
                    .worker_registration
                    .start_workflow(WorkflowId::new("worker_registration"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start worker registration workflow: {e:?}"))?;

                debug!(
                    "Started worker registration workflow for {} (instance: {})",
                    Endpoint::redacted_str(&config.url),
                    instance_id
                );

                engines
                    .worker_registration
                    .wait_for_completion(instance_id, &config.url, timeout_duration)
                    .await
            }
            Job::UpdateWorker { url, update } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data = create_worker_update_workflow_data(
                    url.to_string(),
                    (**update).clone(),
                    Arc::clone(context),
                );

                let instance_id = engines
                    .worker_update
                    .start_workflow(WorkflowId::new("worker_update"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start worker update workflow: {e:?}"))?;

                debug!(
                    "Started worker update workflow for {} (instance: {})",
                    Endpoint::redacted_str(url),
                    instance_id
                );

                let timeout_duration = Duration::from_secs(30);

                engines
                    .worker_update
                    .wait_for_completion(instance_id, url, timeout_duration)
                    .await
            }
            Job::RefreshWorkerLabels { url } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

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
                );
                workflow_data.refresh_discovered_labels = true;

                let instance_id = engines
                    .worker_update
                    .start_workflow(WorkflowId::new("worker_update"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start worker label refresh workflow: {e:?}"))?;

                debug!(
                    "Started worker label refresh workflow for {} (instance: {})",
                    Endpoint::redacted_str(url),
                    instance_id
                );

                engines
                    .worker_update
                    .wait_for_completion(instance_id, url, Duration::from_secs(30))
                    .await
            }
            Job::RemoveWorker {
                url,
                expected_revisions,
            } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data = create_worker_removal_workflow_data(
                    url.to_string(),
                    expected_revisions.clone(),
                    Arc::clone(context),
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

                // Caller wait must cover the worst-case `DrainWorkersStep`
                // sleep, which is `max(per-worker drain_settle_secs)`. We
                // can't see per-worker overrides from here without scanning
                // the registry, so use a generous floor: at least
                // `MAX_DRAIN_WAIT_SECS` (large enough for realistic
                // overrides), and at least the global default if it's
                // higher. 30s on top covers the other removal steps.
                const MAX_DRAIN_WAIT_SECS: u64 = 600;
                let drain_wait_secs = context
                    .router_config
                    .health_check
                    .drain_settle_secs
                    .max(MAX_DRAIN_WAIT_SECS);
                let timeout_duration = Duration::from_secs(30 + drain_wait_secs);

                let result = engines
                    .worker_removal
                    .wait_for_completion(instance_id, url, timeout_duration)
                    .await;

                // Clean up job status when removing worker
                if let Some(queue) = context.worker_job_queue.get() {
                    queue.remove_status(url);
                }

                result
            }
            Job::AddWasmModule { config } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data =
                    create_wasm_registration_workflow_data(*config.clone(), Arc::clone(context));

                let instance_id = engines
                    .wasm_registration
                    .start_workflow(WorkflowId::new("wasm_module_registration"), workflow_data)
                    .await
                    .map_err(|e| {
                        format!("Failed to start WASM module registration workflow: {e:?}")
                    })?;

                debug!(
                    "Started WASM module registration workflow for {} (instance: {})",
                    config.descriptor.name, instance_id
                );

                let timeout_duration = Duration::from_secs(300); // 5 minutes

                engines
                    .wasm_registration
                    .wait_for_completion(instance_id, &config.descriptor.name, timeout_duration)
                    .await
            }
            Job::RemoveWasmModule { request } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data =
                    create_wasm_removal_workflow_data(*request.clone(), Arc::clone(context));

                let instance_id = engines
                    .wasm_removal
                    .start_workflow(WorkflowId::new("wasm_module_removal"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start WASM module removal workflow: {e:?}"))?;

                debug!(
                    "Started WASM module removal workflow for {} (instance: {})",
                    request.module_uuid, instance_id
                );

                let timeout_duration = Duration::from_secs(60); // 1 minute

                engines
                    .wasm_removal
                    .wait_for_completion(
                        instance_id,
                        &request.module_uuid.to_string(),
                        timeout_duration,
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
                    };

                    if let Some(queue) = context.worker_job_queue.get() {
                        queue.submit(job).await.map_err(|e| {
                            format!(
                                "Failed to submit AddWorker job for {worker_type} worker {url_for_error}: {e}"
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
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data =
                    create_mcp_workflow_data((**config).clone(), Arc::clone(context));

                let instance_id = engines
                    .mcp
                    .start_workflow(WorkflowId::new("mcp_registration"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start MCP registration workflow: {e:?}"))?;

                debug!(
                    "Started MCP registration workflow for {} (instance: {})",
                    config.name, instance_id
                );

                let timeout_duration = Duration::from_secs(7200 + 30); // 2hr + margin

                engines
                    .mcp
                    .wait_for_completion(instance_id, &config.name, timeout_duration)
                    .await
            }
            Job::AddTokenizer { config } => {
                // The load was counted in flight at submission; its end is
                // noted for the model (a failure of the model's last load
                // answers its requests 503 tokenizer_load_failed).
                let result = async {
                    let engines = context
                        .workflow_engines
                        .get()
                        .ok_or_else(|| "Workflow engines not initialized".to_string())?;

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

    /// Update job status on completion
    fn record_job_completion(
        job_type: &'static str,
        worker_url: &str,
        _duration: Duration,
        result: &Result<String, String>,
        status_map: &Arc<DashMap<String, JobStatus>>,
    ) {
        match result {
            Ok(message) => {
                Self::record_own_status(status_map, worker_url, job_type, None);
                debug!(
                    "Completed job: type={}, worker={}, result={}",
                    job_type,
                    Job::shown_worker(job_type, worker_url),
                    message
                );
            }
            Err(error) => {
                Self::record_own_status(
                    status_map,
                    worker_url,
                    job_type,
                    Some(JobStatus::failed(job_type, worker_url, error.clone())),
                );
                warn!(
                    "Failed job: type={}, worker={}, error={}",
                    job_type,
                    Job::shown_worker(job_type, worker_url),
                    error
                );
            }
        }
    }

    /// The pending status under `url` once a slot is held: written over
    /// whatever is there, or, for a claimed submission (`own_only`), over this
    /// job's own claim alone (`record_own_status`), so a job of another type
    /// that took the URL during the wait keeps its entry and the in-flight
    /// guard that reads it.
    fn write_pending(
        status_map: &DashMap<String, JobStatus>,
        url: &str,
        job_type: &'static str,
        own_only: bool,
    ) {
        if own_only {
            Self::record_own_status(
                status_map,
                url,
                job_type,
                Some(JobStatus::pending(job_type, url)),
            );
        } else {
            status_map.insert(url.to_string(), JobStatus::pending(job_type, url));
        }
    }

    /// Writes a job's status under its URL (`None` removes it) unless the
    /// entry there is another job's: a job of another type submitted for the
    /// URL after this one was queued keeps its status for its lifetime, since
    /// the in-flight guards (the reconciler's, `submit_unless_in_flight`) read
    /// that entry. A vacant entry is written as before.
    fn record_own_status(
        status_map: &DashMap<String, JobStatus>,
        url: &str,
        job_type: &'static str,
        status: Option<JobStatus>,
    ) {
        match status_map.entry(url.to_string()) {
            Entry::Occupied(entry) if entry.get().job_type != job_type => {
                debug!(
                    "Job status of type={} for worker={} left to the {} job under the key",
                    job_type,
                    Job::shown_worker(job_type, url),
                    entry.get().job_type
                );
            }
            Entry::Occupied(mut entry) => match status {
                Some(status) => {
                    entry.insert(status);
                }
                None => {
                    entry.remove();
                }
            },
            Entry::Vacant(entry) => {
                if let Some(status) = status {
                    entry.insert(status);
                }
            }
        }
    }

    /// A status is reclaimed only once terminal and older than the TTL.
    /// Pending/processing entries are the reconciler's in-flight guard: a
    /// long registration wave holds jobs in those states well past any
    /// fixed TTL, and purging them makes every discovery pass resubmit
    /// duplicate jobs for workers already mid-registration.
    fn status_expired(status: &JobStatus, now: u64, ttl: u64) -> bool {
        !matches!(status.status.as_str(), "pending" | "processing")
            && now.saturating_sub(status.timestamp) >= ttl
    }

    /// Cleanup old terminal job statuses (TTL 5 minutes)
    async fn cleanup_old_statuses(status_map: Arc<DashMap<String, JobStatus>>) {
        const CLEANUP_INTERVAL: Duration = Duration::from_secs(60); // Run every minute
        const STATUS_TTL: u64 = 300; // 5 minutes in seconds

        loop {
            tokio::time::sleep(CLEANUP_INTERVAL).await;

            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            status_map.retain(|_key, value| !Self::status_expired(value, now, STATUS_TTL));

            debug!(
                "Cleaned up old job statuses, remaining: {}",
                status_map.len()
            );
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
    // Grouped ZMQ worker: the handshake awaits this many DP engines on one
    // socket set. Only ZMQ URLs — dp_size on an HTTP/gRPC startup worker would
    // misread as dp-awareness.
    if let Some(count) = router_config.zmq_engine_count.filter(|&n| n > 1) {
        if ConnectionMode::from_url(&spec.url) == Some(ConnectionMode::Zmq) {
            spec.dp_size = Some(count);
        }
    }
    // Health config is resolved at worker build time from router defaults +
    // per-worker overrides (spec.health). No need to set spec.health here since
    // these workers have no overrides.
    spec.max_connection_attempts = router_config.health_check.success_threshold.max(1) * 10;
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
        };

        if let Some(queue) = context.worker_job_queue.get() {
            queue.submit(job).await.map_err(|e| {
                format!("Failed to submit AddWorker job for {provider_name} endpoint {url_for_error}: {e}")
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
    spec.max_connection_attempts = router_config.health_check.success_threshold.max(1) * 10;
    spec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::{WasmModuleDescriptor, WasmModuleType};

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
        queue.outcomes.insert(
            key.clone(),
            Waiter {
                id: 1,
                sender: first,
            },
        );
        let second = queue
            .submit_and_wait(
                Job::RefreshWorkerLabels { url: key.clone() },
                Duration::from_secs(1),
            )
            .await;
        let shown = Job::shown_worker("RefreshWorkerLabels", &key);
        assert!(!shown.contains("secret"), "{shown}");
        assert_eq!(second, Err(JobOutcomeError::InFlight(shown)));
        assert_eq!(queue.outcomes.get(&key).map(|waiter| waiter.id), Some(1));
        JobQueue::deliver_outcome(&queue.outcomes, &key, Some(1), Ok("done".to_string()));
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
            Arc::new(crate::worker::WorkerRegistry::new()),
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
            url: url.to_string(),
        };
        let held = Arc::clone(&queue.concurrency_limit)
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
            queue.outcomes.is_empty(),
            "the first caller's waiter left with it"
        );

        // The second wait under the key is taken with the second id; once
        // the dispatcher runs, the first job's late outcome reaches nobody
        // and the second wait is answered by its own job.
        let second = queue.submit_and_wait(refresh(&key), Duration::from_secs(5));
        let release = async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert_eq!(
                queue.outcomes.get(&key).map(|waiter| waiter.id),
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
        assert!(queue.outcomes.is_empty());
    }

    /// A finished job's outcome reaches the waiter under its key once, and
    /// the waiter is gone; a key without a waiter is a no-op.
    #[tokio::test]
    async fn an_outcome_reaches_its_waiter_once() {
        let outcomes: Outcomes = Arc::new(DashMap::new());
        let (tx, rx) = oneshot::channel();
        outcomes.insert("job-1".to_string(), Waiter { id: 1, sender: tx });
        JobQueue::deliver_outcome(&outcomes, "job-2", Some(1), Ok("stray".to_string()));
        JobQueue::deliver_outcome(&outcomes, "job-1", Some(1), Ok("done".to_string()));
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
        JobQueue::deliver_outcome(
            &outcomes,
            "job-1",
            Some(1),
            Ok("the first job's".to_string()),
        );
        JobQueue::deliver_outcome(&outcomes, "job-1", None, Ok("a plain submit's".to_string()));
        assert_eq!(outcomes.get("job-1").map(|waiter| waiter.id), Some(2));
        assert!(matches!(
            rx.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ));
        JobQueue::deliver_outcome(&outcomes, "job-1", Some(2), Ok("its own".to_string()));
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
        assert!(queue.outcomes.is_empty());
    }

    /// A caller leaving while its submit is parked on a slot (the queue full)
    /// takes its waiter with it: the waiter is under the key while the submit
    /// waits, gone when the caller is, and a later caller takes the key.
    #[tokio::test]
    async fn a_wait_cancelled_under_a_full_queue_leaves_no_waiter() {
        use crate::{app_context::testing::app_context_for_tests, worker::WorkerRegistry};

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
        let slot = queue.tx.reserve().await.expect("the queue is open");
        let key = "http://127.0.0.1:8000".to_string();

        let mut wait = Box::pin(queue.submit_and_wait(
            Job::RefreshWorkerLabels { url: key.clone() },
            Duration::from_secs(5),
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut wait)
                .await
                .is_err(),
            "the submit parks on the held slot"
        );
        assert!(
            queue.outcomes.contains_key(&key),
            "the waiter is under the key while the submit waits"
        );
        drop(wait);
        assert!(
            queue.outcomes.is_empty(),
            "the caller leaving took its waiter"
        );

        let mut later = Box::pin(queue.submit_and_wait(
            Job::RefreshWorkerLabels { url: key.clone() },
            Duration::from_secs(5),
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(50), &mut later)
                .await
                .is_err(),
            "the later submit parks too"
        );
        assert!(
            queue.outcomes.contains_key(&key),
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

        assert!(!JobQueue::status_expired(&pending, far_future, 300));
        assert!(!JobQueue::status_expired(&processing, far_future, 300));
    }

    /// Terminal statuses still age out so a failed worker retries on a later
    /// reconcile pass and the map stays bounded.
    #[test]
    fn cleanup_expires_terminal_statuses_after_the_ttl() {
        let failed = JobStatus::failed("AddWorker", "http://w:8000", "boom".to_string());

        assert!(!JobQueue::status_expired(
            &failed,
            failed.timestamp + 299,
            300
        ));
        assert!(JobQueue::status_expired(
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
                std::thread::spawn(move || JobQueue::claim(&map, "http://w:8000", "AddWorker"))
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
        assert!(JobQueue::claim(&map, "http://w:8000", "AddWorker"));

        // A different job type under the same key is not a competing claim.
        map.insert(
            "http://w:8000".to_string(),
            JobStatus::pending("RegisterMcpServer", "http://w:8000"),
        );
        assert!(JobQueue::claim(&map, "http://w:8000", "AddWorker"));
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
        assert!(!JobQueue::claim_unless_in_flight(
            &map,
            "http://w:8000",
            "RefreshWorkerLabels"
        ));
        assert_eq!(map.get("http://w:8000").unwrap().job_type, "AddWorker");

        map.insert(
            "http://w:8000".to_string(),
            JobStatus::failed("AddWorker", "http://w:8000", "boom".to_string()),
        );
        assert!(JobQueue::claim_unless_in_flight(
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
            JobQueue::record_own_status(&map, url, "RefreshWorkerLabels", status);
            let entry = map.get(url).expect("the other job's entry stays");
            assert_eq!(entry.job_type, "RemoveWorker");
            assert_eq!(entry.status, "processing");
        }

        // The job's own entry takes its writes, and a vacant one is written.
        map.insert(
            url.to_string(),
            JobStatus::pending("RefreshWorkerLabels", url),
        );
        JobQueue::record_own_status(
            &map,
            url,
            "RefreshWorkerLabels",
            Some(JobStatus::processing("RefreshWorkerLabels", url)),
        );
        assert_eq!(map.get(url).unwrap().status, "processing");
        JobQueue::record_own_status(&map, url, "RefreshWorkerLabels", None);
        assert!(map.get(url).is_none());
        JobQueue::record_own_status(
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
        JobQueue::write_pending(&map, url, "RefreshWorkerLabels", true);
        assert_eq!(map.get(url).unwrap().job_type, "AddWorker");
        JobQueue::write_pending(&map, url, "RefreshWorkerLabels", false);
        assert_eq!(map.get(url).unwrap().job_type, "RefreshWorkerLabels");
        map.remove(url);
        JobQueue::write_pending(&map, url, "RefreshWorkerLabels", true);
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
            url: url.to_string(),
        };
        assert!(queue.submit_unless_in_flight(job()).await.is_err());
        assert!(queue.status_map.get(url).is_none());
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
        use crate::service_discovery::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        let url = "http://w:8000";
        queue.hold_url_for_tests(url, "AddWorker");

        let removal = Job::RemoveWorker {
            url: url.to_string(),
            expected_revisions: None,
        };
        assert_eq!(queue.submit_or_defer(removal).await, Ok(false));
        assert!(queue.kept_for(url), "the job is kept, not dropped");
        assert_eq!(
            queue.get_status(url).expect("the holder's status").job_type,
            "AddWorker",
            "the holder's status is left to it"
        );

        JobQueue::record_job_completion(
            "AddWorker",
            url,
            Duration::ZERO,
            &Ok("done".to_string()),
            &queue.status_map,
        );
        JobQueue::run_kept(url, &context, &queue.status_map, &queue.kept).await;
        assert!(!queue.kept_for(url));
        let status = queue.get_status(url).expect("the kept job's own status");
        assert_eq!(status.job_type, "RemoveWorker");
        assert_eq!(
            status.status, "failed",
            "the kept job ran once the holder ended"
        );
    }

    /// The wiring at `process_job`: a kept job runs at the completion of the
    /// job that held the URL, through the dispatcher and not by hand. A hold
    /// by hand keeps the first submission; a plain submission for the same
    /// URL writes its pending status over the hold and becomes the holder;
    /// when the dispatcher finishes it, the kept job runs under its own
    /// status. Neither job's workflow runs in a test context, so both end
    /// failed, and the URL's last status is the kept job's.
    #[tokio::test]
    async fn a_kept_job_runs_at_the_dispatcher_completion_of_its_holder() {
        use crate::service_discovery::testing::create_test_app_context;
        let context = create_test_app_context();
        let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&context));
        let url = "http://w:8001";
        queue.hold_url_for_tests(url, "AddWorker");
        let kept = Job::RemoveWorker {
            url: url.to_string(),
            expected_revisions: None,
        };
        assert_eq!(queue.submit_or_defer(kept).await, Ok(false));
        assert!(queue.kept_for(url));

        queue
            .submit(Job::RefreshWorkerLabels {
                url: url.to_string(),
            })
            .await
            .expect("the holder is queued");
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let last = queue
                .get_status(url)
                .map(|status| (status.job_type.to_string(), status.status.to_string()));
            if !queue.kept_for(url)
                && last == Some(("RemoveWorker".to_string(), "failed".to_string()))
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the kept job did not run at the holder's completion: kept={} last={last:?}",
                queue.kept_for(url)
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// A submission that fails must not leave its claim behind, or the next
    /// create for the URL would be told a job is in flight when none is (#1533).
    #[tokio::test]
    async fn failed_submit_if_idle_releases_its_claim() {
        let queue = JobQueue::new(JobQueueConfig::default(), Weak::new());
        let job = || Job::AddWorker {
            config: Box::new(WorkerSpec::new("http://w:8000")),
            registration_mode: WorkerRegistrationMode::CreateOnly,
        };
        assert!(queue.submit_if_idle(job()).await.is_err());
        // Before the rollback this returned Ok(false): a 202 with no job behind it.
        assert!(queue.submit_if_idle(job()).await.is_err());
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

        assert_eq!(queue.tx.max_capacity(), 7);
        let (queue_depth, available_permits) = queue.get_load_info();
        assert_eq!(queue_depth, 0);
        assert_eq!(available_permits, 3);
    }

    /// The connection budget scales with the health-check success threshold,
    /// and a zero threshold must not collapse it to zero attempts.
    #[test]
    fn connection_attempts_scale_with_the_success_threshold_floor() {
        let mut config = RouterConfig::default();
        config.health_check.success_threshold = 0;
        assert_eq!(
            spec_for("http://127.0.0.1:8000", &config).max_connection_attempts,
            10
        );

        config.health_check.success_threshold = 3;
        assert_eq!(
            spec_for("http://127.0.0.1:8000", &config).max_connection_attempts,
            30
        );
    }
}
