//! Async job queue for control plane operations
//!
//! Provides non-blocking worker management by queuing operations and processing
//! them asynchronously in background worker tasks.

use std::{
    sync::{Arc, Weak},
    time::{Duration, SystemTime},
};

use dashmap::DashMap;
use openai_protocol::worker::{
    ConnectionMode, JobStatus, RuntimeType, WorkerSpec, WorkerType, WorkerUpdateRequest,
};
use smg_mcp::McpConfig;
use tokio::sync::{mpsc, Semaphore};
use tracing::{debug, error, info, warn};
use wfaas::WorkflowId;

use crate::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    workflow::{
        create_mcp_workflow_data, create_tokenizer_workflow_data,
        create_wasm_registration_workflow_data, create_wasm_removal_workflow_data,
        create_worker_removal_workflow_data, create_worker_update_workflow_data,
        create_worker_workflow_data, McpServerConfigRequest, TokenizerConfigRequest,
        TokenizerRemovalRequest, WasmModuleConfigRequest, WasmModuleRemovalRequest,
        WorkerRegistrationMode,
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
    RemoveWorker {
        url: String,
        expected_revision: Option<u64>,
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

    /// Get worker URL, MCP server name, WASM module, or tokenizer identifier for logging and status tracking
    pub fn worker_url(&self) -> &str {
        match self {
            Job::AddWorker { config, .. } => &config.url,
            Job::UpdateWorker { url, .. } => url,
            Job::RemoveWorker { url, .. } => url,
            Job::InitializeWorkersFromConfig { .. } => "startup",
            Job::InitializeMcpServers { .. } => "startup",
            Job::RegisterMcpServer { config } => &config.name,
            Job::AddWasmModule { config } => &config.descriptor.name,
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
    /// Channel for submitting jobs
    tx: mpsc::Sender<Job>,
    /// Weak reference to AppContext to avoid circular dependencies
    context: Weak<AppContext>,
    /// Job status tracking by worker URL
    status_map: Arc<DashMap<String, JobStatus>>,
    /// Semaphore to limit concurrent job execution
    concurrency_limit: Arc<Semaphore>,
}

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

        let queue = Arc::new(Self {
            tx,
            context: context.clone(),
            status_map: status_map.clone(),
            concurrency_limit: concurrency_limit.clone(),
        });

        // Single dispatcher task
        let ctx = context.clone();
        let status = status_map.clone();
        let sem = concurrency_limit.clone();

        #[expect(
            clippy::disallowed_methods,
            reason = "Core job dispatcher loop: runs for the lifetime of the gateway and drains cleanly when the channel sender is dropped on shutdown"
        )]
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                // Acquire permit (blocks if at concurrency limit)
                let Ok(permit) = sem.clone().acquire_owned().await else {
                    error!("Semaphore closed, stopping dispatcher");
                    break;
                };

                let ctx_clone = ctx.clone();
                let status_clone = status.clone();

                #[expect(
                    clippy::disallowed_methods,
                    reason = "Job processing task: bounded by the semaphore permit which is dropped when the task completes"
                )]
                tokio::spawn(async move {
                    Self::process_job(job, ctx_clone, status_clone, permit).await;
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

    /// Submit a job with detailed queue status
    pub async fn submit(&self, job: Job) -> Result<(), String> {
        // Check if context is still alive before accepting jobs
        if self.context.upgrade().is_none() {
            return Err("Job queue shutting down: AppContext dropped".to_string());
        }

        // Extract values before moving job
        let job_type = job.job_type();
        let worker_url = job.worker_url().to_string();

        // Record pending status
        self.status_map.insert(
            worker_url.clone(),
            JobStatus::pending(job_type, &worker_url),
        );

        match self.tx.send(job).await {
            Ok(()) => {
                let (queue_depth, available_permits) = self.get_load_info();
                debug!(
                    "Job submitted: type={}, worker={}, queue_depth={}, available_slots={}",
                    job_type, worker_url, queue_depth, available_permits
                );
                Ok(())
            }
            Err(_) => {
                self.status_map.remove(&worker_url);
                let (queue_depth, _) = self.get_load_info();
                Err(format!(
                    "Job queue full: {} jobs pending (capacity: {})",
                    queue_depth,
                    self.tx.max_capacity()
                ))
            }
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
        job: Job,
        context: Weak<AppContext>,
        status_map: Arc<DashMap<String, JobStatus>>,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        let job_type = job.job_type();
        let worker_url = job.worker_url().to_string();
        let start = std::time::Instant::now();

        // Update to processing
        status_map.insert(
            worker_url.clone(),
            JobStatus::processing(job_type, &worker_url),
        );

        debug!("Processing job: type={}, worker={}", job_type, worker_url);

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
                Self::record_job_completion(job_type, &worker_url, duration, &result, &status_map);
            }
            None => {
                let error_msg = "AppContext dropped".to_string();
                status_map.insert(
                    worker_url.clone(),
                    JobStatus::failed(job_type, &worker_url, error_msg),
                );
                error!(
                    "AppContext dropped, cannot process job: type={}, worker={}",
                    job_type, worker_url
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
                    config.url, instance_id
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
                    url, instance_id
                );

                let timeout_duration = Duration::from_secs(30);

                engines
                    .worker_update
                    .wait_for_completion(instance_id, url, timeout_duration)
                    .await
            }
            Job::RemoveWorker {
                url,
                expected_revision,
            } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data = create_worker_removal_workflow_data(
                    url.to_string(),
                    context.router_config.dp_aware,
                    *expected_revision,
                    Arc::clone(context),
                );

                let instance_id = engines
                    .worker_removal
                    .start_workflow(WorkflowId::new("worker_removal"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start worker removal workflow: {e:?}"))?;

                debug!(
                    "Started worker removal workflow for {} (instance: {})",
                    url, instance_id
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

                // Create iterator of (url, worker_type, bootstrap_port, is_long_pool)
                // tuples. The 4th element marks whether the worker gets pool=long
                // label, determined by matching the prefill index against
                // router_config.long_prefill_indices.
                let long_indices = &router_config.long_prefill_indices;
                let workers: Vec<(String, &str, Option<u16>, bool)> = match &router_config.mode {
                    RoutingMode::Regular { worker_urls } => worker_urls
                        .iter()
                        .map(|url| (url.clone(), "regular", None, false))
                        .collect(),
                    RoutingMode::PrefillDecode {
                        prefill_urls,
                        decode_urls,
                        ..
                    } => {
                        let prefill_workers =
                            prefill_urls.iter().enumerate().map(|(i, (url, port))| {
                                (url.clone(), "prefill", *port, is_long_pool_index(i, long_indices))
                            });

                        let decode_workers = decode_urls
                            .iter()
                            .map(|url| (url.clone(), "decode", None, false));

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
                            .map(|(url, port)| (url.clone(), "encode", *port, false));
                        let prefill_workers =
                            prefill_urls.iter().enumerate().map(|(i, (url, port))| {
                                (url.clone(), "prefill", *port, is_long_pool_index(i, long_indices))
                            });
                        let decode_workers = decode_urls
                            .iter()
                            .map(|url| (url.clone(), "decode", None, false));

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
                for (url, worker_type, bootstrap_port, is_long_pool) in workers {
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
                    if is_long_pool {
                        spec.labels.insert("pool".to_string(), "long".to_string());
                    }
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
            Job::RemoveTokenizer { request } => {
                // Tokenizer removal is synchronous and fast
                if let Some(entry) = context.tokenizer_registry.remove_by_id(&request.id) {
                    context.multimodal_config_registry.remove(&entry.id);
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
                status_map.remove(worker_url);
                debug!(
                    "Completed job: type={}, worker={}, result={}",
                    job_type, worker_url, message
                );
            }
            Err(error) => {
                status_map.insert(
                    worker_url.to_string(),
                    JobStatus::failed(job_type, worker_url, error.clone()),
                );
                warn!(
                    "Failed job: type={}, worker={}, error={}",
                    job_type, worker_url, error
                );
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

/// Whether prefill worker at `index` belongs to the long pool, determined by
/// membership in `long_prefill_indices`.
fn is_long_pool_index(index: usize, long_indices: &[usize]) -> bool {
    long_indices.contains(&index)
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

    /// `--long-prefill-indices 3,4` marks the 4th and 5th prefill workers with
    /// `pool=long`; other prefill workers and all decode workers stay unlabelled.
    #[test]
    fn long_prefill_indices_tags_correct_workers() {
        let prefill_urls: Vec<(String, Option<u16>)> = vec![
            ("http://p1:8000".into(), None),
            ("http://p2:8000".into(), None),
            ("http://p3:8000".into(), None),
            ("http://p4:8000".into(), None),
            ("http://p5:8000".into(), None),
        ];
        let long_indices = vec![3, 4];

        // Simulate the worker-tuple construction from InitializeWorkersFromConfig
        let workers: Vec<(String, &str, Option<u16>, bool)> = prefill_urls
            .iter()
            .enumerate()
            .map(|(i, (url, port))| {
                (url.clone(), "prefill", *port, is_long_pool_index(i, &long_indices))
            })
            .collect();

        // P1 (index 0), P2 (index 1), P3 (index 2) → short pool (is_long_pool=false)
        assert!(!workers[0].3, "P1 should be short pool");
        assert!(!workers[1].3, "P2 should be short pool");
        assert!(!workers[2].3, "P3 should be short pool");
        // P4 (index 3), P5 (index 4) → long pool (is_long_pool=true)
        assert!(workers[3].3, "P4 should be long pool");
        assert!(workers[4].3, "P5 should be long pool");

        // Verify the label is actually applied to the WorkerSpec
        for (_, _, _, is_long_pool) in &workers {
            let mut spec = WorkerSpec::new("http://test:8000");
            if *is_long_pool {
                spec.labels.insert("pool".to_string(), "long".to_string());
            }
            if *is_long_pool {
                assert_eq!(
                    spec.labels.get("pool").map(|s| s.as_str()),
                    Some("long"),
                    "long-pool worker must have pool=long label"
                );
            } else {
                assert!(
                    spec.labels.get("pool").is_none(),
                    "short-pool worker must NOT have pool label"
                );
            }
        }
    }

    /// Empty `long_prefill_indices` means no long pool — all prefill workers
    /// are short pool.
    #[test]
    fn empty_long_prefill_indices_means_all_short() {
        let prefill_urls: Vec<(String, Option<u16>)> = vec![
            ("http://p1:8000".into(), None),
            ("http://p2:8000".into(), None),
        ];
        let long_indices: Vec<usize> = vec![];

        let workers: Vec<(String, &str, Option<u16>, bool)> = prefill_urls
            .iter()
            .enumerate()
            .map(|(i, (url, port))| {
                (url.clone(), "prefill", *port, is_long_pool_index(i, &long_indices))
            })
            .collect();

        assert!(!workers[0].3, "no long indices → all short pool");
        assert!(!workers[1].3, "no long indices → all short pool");
    }
}
