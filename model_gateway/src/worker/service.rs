//! Worker Service - Business logic layer for worker operations
//!
//! This module provides a clean separation between HTTP concerns (in routers)
//! and business logic for worker management. The service orchestrates
//! WorkerRegistry and JobQueue operations.

use std::{collections::HashMap, sync::Arc};

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::worker::{WorkerErrorResponse, WorkerInfo, WorkerSpec, WorkerUpdateRequest};
use serde_json::json;
use tracing::warn;

use crate::{
    config::{validate_worker_url, RouterConfig},
    routers::provider_support,
    worker::{registry::WorkerId, worker::worker_to_info, WorkerRegistry, WorkerType},
    workflow::{Job, JobQueue, WorkerRegistrationMode},
};

/// Validate the URL of an incoming worker-management request, translating a
/// config-layer rejection into `400 Bad Request`. Must run before
/// `WorkerRegistry::reserve_id_for_url` so rejected input can never leave an
/// orphaned reservation (#1533).
fn validate_worker_url_request(url: &str) -> Result<(), WorkerServiceError> {
    validate_worker_url(url).map_err(|e| WorkerServiceError::BadRequest {
        message: e.to_string(),
    })
}

/// Longest URL a registration takes: longer ones are not addresses of a
/// worker but a mistake or a probe, and every log line and metric label
/// derived from the URL would carry them.
const MAX_WORKER_URL_BYTES: usize = 2048;
/// Most labels a worker carries (an engine reports a few dozen); held at
/// registration, at the admin check of a PATCH and again where the PATCH
/// merges into the worker.
pub(crate) const MAX_WORKER_LABELS: usize = 128;
const MAX_LABEL_KEY_BYTES: usize = 128;
const MAX_LABEL_VALUE_BYTES: usize = 4096;
/// Labels the routers read as the decode window, a count from 1 to 65535
/// (`Worker::max_running_requests`); any other value is read as "not
/// reported", silently.
const WINDOW_LABELS: [&str; 2] = ["max_running_requests", "max_num_seqs"];
/// Labels the routers read as a positive count.
const COUNT_LABELS: [&str; 1] = ["dp_size"];
/// Labels the routers read as a boolean, spelled `"true"` or `"false"`; any
/// other spelling is read as unset, silently.
const BOOLEAN_LABELS: [&str; 2] = ["supports_vision", "realtime"];

fn bad_request(message: String) -> WorkerServiceError {
    WorkerServiceError::BadRequest { message }
}

/// The bounds a registration body must satisfy before any job is queued:
/// the URL's length; the labels' count, charset and size and the shape of
/// the labels the routers read as numbers or booleans; the fields the routers
/// rely on (a positive, finite cost; a KV block size and a data-parallel size
/// of at least 1; a rank only with a size and below it; a bootstrap port
/// that is a port). A violation is a `400 Bad Request` naming the field.
fn validate_spec(spec: &WorkerSpec) -> Result<(), WorkerServiceError> {
    if spec.url.len() > MAX_WORKER_URL_BYTES {
        return Err(bad_request(format!(
            "url is {} bytes long; at most {MAX_WORKER_URL_BYTES} are accepted",
            spec.url.len()
        )));
    }
    validate_labels(&spec.labels)?;
    validate_cost(spec.cost)?;
    if spec.kv_block_size == Some(0) {
        return Err(bad_request(
            "kv_block_size must be at least 1 (tokens per KV block)".to_string(),
        ));
    }
    if spec.bootstrap_port == Some(0) {
        return Err(bad_request(
            "bootstrap_port must be a port from 1 to 65535".to_string(),
        ));
    }
    match (spec.dp_rank, spec.dp_size) {
        (Some(_), None) => Err(bad_request(
            "dp_rank needs dp_size: the rank is a position in a data-parallel group of that size"
                .to_string(),
        )),
        (_, Some(0)) => Err(bad_request("dp_size must be at least 1".to_string())),
        (Some(rank), Some(size)) if rank >= size => Err(bad_request(format!(
            "dp_rank {rank} is not below dp_size {size}"
        ))),
        _ => Ok(()),
    }
}

/// The bounds of an update: the fields it may change, under the same rules.
fn validate_update(update: &WorkerUpdateRequest) -> Result<(), WorkerServiceError> {
    if let Some(cost) = update.cost {
        validate_cost(cost)?;
    }
    if let Some(labels) = &update.labels {
        validate_labels(labels)?;
    }
    Ok(())
}

/// The cost weighs the worker in the policies' arithmetic: finite and above 0.
fn validate_cost(cost: f32) -> Result<(), WorkerServiceError> {
    if !cost.is_finite() || cost <= 0.0 {
        return Err(bad_request(format!(
            "cost must be a finite number above 0 (baseline 1.0), not {cost}"
        )));
    }
    Ok(())
}

/// A rejected key as the error shows it: whole when short, else its first
/// characters and its length, so a key the size of the payload limit does
/// not come back as a body (or a log line) of that size.
fn shown_key(key: &str) -> String {
    const SHOWN: usize = 64;
    if key.chars().count() <= SHOWN {
        return format!("{key:?}");
    }
    let head: String = key.chars().take(SHOWN).collect();
    format!("{head:?}... ({} bytes)", key.len())
}

/// The labels a PATCH would leave on the worker: the request's merged into
/// the existing ones (`UpdateWorkerPropertiesStep` keeps every key it does not
/// name), held to the same cap as a registration, so repeated updates cannot
/// grow a worker past it; an update that adds no key (a rewrite of keys the
/// worker has) passes, so a worker over the cap through the labels discovered
/// at registration can still be updated. This is the early check against the
/// labels the worker has when the request arrives, for a clear 400; the merge
/// runs later in the queued job, where `UpdateWorkerPropertiesStep` holds the
/// cap again, under the same condition, over the labels it actually merges,
/// so PATCHes queued together cannot pass it between them.
fn validate_merged_labels(
    existing: &HashMap<String, String>,
    update: &HashMap<String, String>,
) -> Result<(), WorkerServiceError> {
    let added = update
        .keys()
        .filter(|key| !existing.contains_key(*key))
        .count();
    let merged = existing.len() + added;
    if added > 0 && merged > MAX_WORKER_LABELS {
        return Err(bad_request(format!(
            "{merged} labels on the worker after this update ({} now, {added} new); at most \
             {MAX_WORKER_LABELS} are accepted",
            existing.len()
        )));
    }
    Ok(())
}

fn validate_labels(labels: &HashMap<String, String>) -> Result<(), WorkerServiceError> {
    if labels.len() > MAX_WORKER_LABELS {
        return Err(bad_request(format!(
            "{} labels; at most {MAX_WORKER_LABELS} are accepted",
            labels.len()
        )));
    }
    for (key, value) in labels {
        if key.is_empty()
            || key.len() > MAX_LABEL_KEY_BYTES
            || !key.chars().all(|c| c.is_ascii_graphic())
        {
            return Err(bad_request(format!(
                "label key {} must be 1 to {MAX_LABEL_KEY_BYTES} printable ASCII characters \
                 without spaces",
                shown_key(key)
            )));
        }
        if value.len() > MAX_LABEL_VALUE_BYTES || value.chars().any(char::is_control) {
            return Err(bad_request(format!(
                "label '{key}' must be at most {MAX_LABEL_VALUE_BYTES} bytes without control \
                 characters"
            )));
        }
        let key_str = key.as_str();
        if WINDOW_LABELS.contains(&key_str) && !value.parse::<u16>().is_ok_and(|n| n >= 1) {
            return Err(bad_request(format!(
                "label '{key}' must be a count from 1 to 65535, not {value:?}"
            )));
        }
        if COUNT_LABELS.contains(&key_str) && !value.parse::<u64>().is_ok_and(|n| n >= 1) {
            return Err(bad_request(format!(
                "label '{key}' must be a positive integer, not {value:?}"
            )));
        }
        if BOOLEAN_LABELS.contains(&key_str) && value != "true" && value != "false" {
            return Err(bad_request(format!(
                "label '{key}' must be \"true\" or \"false\", not {value:?}"
            )));
        }
    }
    Ok(())
}

/// Error types for worker service operations
#[derive(Debug)]
pub enum WorkerServiceError {
    /// Worker with given ID was not found
    NotFound { worker_id: String },
    /// Invalid worker ID format (expected UUID)
    InvalidId { raw: String, message: String },
    /// Bad request (e.g., URL mismatch in PUT)
    BadRequest { message: String },
    /// Worker with this URL already exists (duplicate POST)
    Conflict { url: String, worker_id: WorkerId },
    /// A create for this URL is still in flight (concurrent POST)
    CreateInProgress { url: String, worker_id: WorkerId },
    /// The spec targets a provider whose router this build does not carry
    ProviderNotCompiled {
        url: String,
        family: &'static str,
        feature: &'static str,
    },
    /// Job queue not initialized
    QueueNotInitialized,
    /// Failed to submit job to queue
    QueueSubmitFailed { message: String },
}

impl WorkerServiceError {
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::NotFound { .. } => "WORKER_NOT_FOUND",
            Self::InvalidId { .. } => "BAD_REQUEST",
            Self::BadRequest { .. } => "BAD_REQUEST",
            Self::Conflict { .. } => "WORKER_ALREADY_EXISTS",
            Self::CreateInProgress { .. } => "WORKER_CREATE_IN_PROGRESS",
            Self::ProviderNotCompiled { .. } => "PROVIDER_NOT_COMPILED",
            Self::QueueNotInitialized => "INTERNAL_SERVER_ERROR",
            Self::QueueSubmitFailed { .. } => "INTERNAL_SERVER_ERROR",
        }
    }

    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound { .. } => StatusCode::NOT_FOUND,
            Self::InvalidId { .. } => StatusCode::BAD_REQUEST,
            Self::BadRequest { .. } => StatusCode::BAD_REQUEST,
            Self::Conflict { .. } => StatusCode::CONFLICT,
            Self::CreateInProgress { .. } => StatusCode::CONFLICT,
            Self::ProviderNotCompiled { .. } => StatusCode::BAD_REQUEST,
            Self::QueueNotInitialized => StatusCode::INTERNAL_SERVER_ERROR,
            Self::QueueSubmitFailed { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl std::fmt::Display for WorkerServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { worker_id } => write!(f, "Worker {worker_id} not found"),
            Self::InvalidId { raw, message } => {
                write!(
                    f,
                    "Invalid worker_id '{raw}' (expected UUID). Error: {message}"
                )
            }
            Self::BadRequest { message } => write!(f, "{message}"),
            Self::Conflict { url, worker_id } => {
                let id = worker_id.as_str();
                write!(
                    f,
                    "Worker already exists at URL '{url}' with ID {id}. \
                    Use PUT /workers/{id} to replace or PATCH /workers/{id} to update."
                )
            }
            Self::CreateInProgress { url, worker_id } => {
                let id = worker_id.as_str();
                write!(
                    f,
                    "Worker creation already in progress for URL '{url}' with ID {id}. \
                    Poll GET /workers/{id}."
                )
            }
            Self::ProviderNotCompiled {
                url,
                family,
                feature,
            } => write!(
                f,
                "Worker '{url}' targets the {family} provider, but this build carries no \
                 {family} router; rebuild with the `{feature}` Cargo feature to admit it."
            ),
            Self::QueueNotInitialized => write!(f, "Job queue not initialized"),
            Self::QueueSubmitFailed { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for WorkerServiceError {}

impl IntoResponse for WorkerServiceError {
    fn into_response(self) -> Response {
        let error = WorkerErrorResponse {
            error: self.to_string(),
            code: self.error_code().to_string(),
        };
        (self.status_code(), Json(error)).into_response()
    }
}

/// Result of creating a worker (async job submission)
#[derive(Debug)]
pub struct CreateWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
    pub location: String,
}

impl IntoResponse for CreateWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "url": self.url,
            "location": self.location,
            "message": "Worker addition queued for background processing"
        });
        (
            StatusCode::ACCEPTED,
            [(http::header::LOCATION, self.location)],
            Json(response),
        )
            .into_response()
    }
}

/// Result of deleting a worker (async job submission)
#[derive(Debug)]
pub struct DeleteWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
}

impl IntoResponse for DeleteWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "message": "Worker removal queued for background processing"
        });
        (StatusCode::ACCEPTED, Json(response)).into_response()
    }
}

/// Result of updating a worker (async job submission)
#[derive(Debug)]
pub struct UpdateWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
}

impl IntoResponse for UpdateWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "message": "Worker update queued for background processing"
        });
        (StatusCode::ACCEPTED, Json(response)).into_response()
    }
}

/// Result of listing workers.
#[derive(Debug)]
pub struct ListWorkersResult {
    pub workers: Vec<WorkerInfo>,
    pub total: usize,
    pub prefill_count: usize,
    pub decode_count: usize,
    pub regular_count: usize,
}

impl IntoResponse for ListWorkersResult {
    fn into_response(self) -> Response {
        // Serialize the typed response directly with `Json` (a single
        // `serde_json::to_vec` pass) instead of building an intermediate
        // `serde_json::Value` via `json!`.
        #[derive(serde::Serialize)]
        struct Body {
            workers: Vec<WorkerInfo>,
            total: usize,
            stats: Stats,
        }
        #[derive(serde::Serialize)]
        struct Stats {
            prefill_count: usize,
            decode_count: usize,
            regular_count: usize,
        }
        Json(Body {
            workers: self.workers,
            total: self.total,
            stats: Stats {
                prefill_count: self.prefill_count,
                decode_count: self.decode_count,
                regular_count: self.regular_count,
            },
        })
        .into_response()
    }
}

/// Wrapper for WorkerInfo to implement IntoResponse
pub struct GetWorkerResponse(pub WorkerInfo);

impl IntoResponse for GetWorkerResponse {
    fn into_response(self) -> Response {
        Json(self.0).into_response()
    }
}

/// Worker Service - Orchestrates worker business logic
///
/// This service provides a clean API for worker operations, separating
/// business logic from HTTP concerns. Handlers in server.rs become thin
/// wrappers that translate between HTTP and this service.
pub struct WorkerService {
    worker_registry: Arc<WorkerRegistry>,
    job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
    router_config: RouterConfig,
}

impl WorkerService {
    /// Create a new WorkerService
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
        router_config: RouterConfig,
    ) -> Self {
        Self {
            worker_registry,
            job_queue,
            router_config,
        }
    }

    /// Parse and validate a worker ID string
    pub fn parse_worker_id(raw: &str) -> Result<WorkerId, WorkerServiceError> {
        uuid::Uuid::parse_str(raw)
            .map(|_| WorkerId::from_string(raw.to_string()))
            .map_err(|e| WorkerServiceError::InvalidId {
                raw: raw.to_string(),
                message: e.to_string(),
            })
    }

    /// Get the job queue, returning an error if not initialized
    fn get_job_queue(&self) -> Result<&Arc<JobQueue>, WorkerServiceError> {
        self.job_queue
            .get()
            .ok_or(WorkerServiceError::QueueNotInitialized)
    }

    pub async fn create_worker(
        &self,
        config: WorkerSpec,
    ) -> Result<CreateWorkerResult, WorkerServiceError> {
        validate_spec(&config)?;
        validate_worker_url_request(&config.url)?;
        Self::require_provider_router(&config)?;

        if self.router_config.api_key.is_some() && config.api_key.is_none() {
            warn!(
                "Adding worker {} without API key while router has API key configured. \
                Worker will be accessible without authentication. \
                If the worker requires the same API key as the router, please specify it explicitly.",
                config.url
            );
        }

        let worker_url = config.url.clone();
        let queue = self.get_job_queue()?;

        // Reserve (or retrieve) a stable ID for the 202 response.
        // If this URL already has an active worker, reject with 409.
        let worker_id = self.worker_registry.reserve_id_for_url(&worker_url);
        if self.worker_registry.get(&worker_id).is_some() {
            return Err(WorkerServiceError::Conflict {
                url: worker_url,
                worker_id,
            });
        }

        let job = Job::AddWorker {
            config: Box::new(config),
            registration_mode: WorkerRegistrationMode::CreateOnly,
        };

        // Exactly one create per URL is in flight at a time, so that attempt alone
        // owns the reservation and a failed one can safely release it. A second
        // create is told to poll the first rather than silently merged into it,
        // since its spec may differ (#1533).
        let submitted = queue.submit_if_idle(job).await.map_err(|e| {
            // No 202 goes out, so nothing may keep this URL reserved (#1533).
            self.worker_registry.release_reservation(&worker_url);
            WorkerServiceError::QueueSubmitFailed { message: e }
        })?;
        if !submitted {
            return Err(WorkerServiceError::CreateInProgress {
                url: worker_url,
                worker_id,
            });
        }

        let location = format!("/workers/{}", worker_id.as_str());

        Ok(CreateWorkerResult {
            worker_id,
            url: worker_url,
            location,
        })
    }

    /// A worker that targets a provider is reachable only through that
    /// provider's router, which exists only in a build that compiled it in.
    /// Admitting one otherwise would leave it routable by nothing, so refuse
    /// here rather than after the 202, inside the background workflow.
    fn require_provider_router(config: &WorkerSpec) -> Result<(), WorkerServiceError> {
        match provider_support::missing_router(config) {
            Some(missing) => Err(WorkerServiceError::ProviderNotCompiled {
                url: config.url.clone(),
                family: missing.label,
                feature: missing.feature,
            }),
            None => Ok(()),
        }
    }

    /// Replace a worker by ID (full replace, re-runs registration workflow)
    pub async fn replace_worker(
        &self,
        worker_id_raw: &str,
        config: WorkerSpec,
    ) -> Result<UpdateWorkerResult, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;

        let existing =
            self.worker_registry
                .get(&worker_id)
                .ok_or_else(|| WorkerServiceError::NotFound {
                    worker_id: worker_id_raw.to_string(),
                })?;
        let url = existing.url().to_string();
        validate_spec(&config)?;
        Self::require_provider_router(&config)?;

        // A data-parallel router expands one spec into one worker per rank,
        // each registered under a rank-suffixed URL. Re-running registration
        // for a single ID cannot express that, so refuse here instead of
        // answering 202 and failing in the background.
        if self.router_config.dp_aware {
            return Err(WorkerServiceError::BadRequest {
                message: format!(
                    "Worker '{url}' belongs to a data-parallel group. \
                    PUT is not supported for data-parallel workers. \
                    Use DELETE + POST instead."
                ),
            });
        }

        // Validate that the URL in the request body matches the existing worker.
        // URL changes are not supported via replace — use DELETE + POST instead.
        if config.url != url {
            return Err(WorkerServiceError::BadRequest {
                message: format!(
                    "URL mismatch: worker has URL '{url}' but request body has '{}'. \
                    URL changes are not supported via PUT. Use DELETE + POST instead.",
                    config.url
                ),
            });
        }

        // Re-run the full registration workflow (model discovery, etc.).
        // The workflow registers with overwrite-then-diff, bound to the ID and
        // revision validated above: the job runs after this call returns 202,
        // so a concurrent DELETE, DELETE + POST, or second PUT must make the
        // write fail rather than resurrect this worker, overwrite its
        // successor, or restore this specification over a newer one.
        let job = Job::AddWorker {
            config: Box::new(config),
            registration_mode: WorkerRegistrationMode::ReplaceById {
                worker_id: worker_id.as_str().to_string(),
                expected_revision: existing.revision(),
            },
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(UpdateWorkerResult { worker_id, url })
    }

    /// List all workers with their info, optionally filtered to workers
    /// serving `model`. Building each `WorkerInfo` is cheap: `WorkerSpec`
    /// is shared via `Arc`, so there is no per-worker spec deep clone.
    ///
    /// The counts reflect the returned (filtered) list, not the whole
    /// registry.
    pub fn list_workers(&self, model: Option<&str>) -> ListWorkersResult {
        let mut worker_infos = Vec::new();
        let mut prefill_count = 0;
        let mut decode_count = 0;
        let mut regular_count = 0;

        for (worker_id, worker) in self.worker_registry.get_all_with_ids() {
            if let Some(model) = model {
                if !worker.supports_model(model) {
                    continue;
                }
            }
            match worker.worker_type() {
                WorkerType::Prefill => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
                WorkerType::Regular => regular_count += 1,
                // EPD encode workers are a distinct pool: counted in `total` and
                // listed, but not in the P/D/Regular sub-counts.
                WorkerType::Encode => {}
            }
            let mut info = worker_to_info(&worker);
            info.id = worker_id.as_str().to_string();
            worker_infos.push(info);
        }

        ListWorkersResult {
            total: worker_infos.len(),
            workers: worker_infos,
            prefill_count,
            decode_count,
            regular_count,
        }
    }

    pub fn get_worker(&self, worker_id_raw: &str) -> Result<GetWorkerResponse, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;
        let job_queue = self.get_job_queue()?;

        if let Some(worker) = self.worker_registry.get(&worker_id) {
            let worker_url = worker.url().to_string();
            let mut worker_info = worker_to_info(&worker);
            worker_info.id = worker_id.as_str().to_string();
            if let Some(status) = job_queue.get_status(&worker_url) {
                worker_info.job_status = Some(status);
            }
            return Ok(GetWorkerResponse(worker_info));
        }

        if let Some(worker_url) = self.worker_registry.get_url_by_id(&worker_id) {
            if let Some(status) = job_queue.get_status(&worker_url) {
                return Ok(GetWorkerResponse(WorkerInfo::pending(
                    worker_id.as_str(),
                    worker_url,
                    Some(status),
                )));
            }
        }

        Err(WorkerServiceError::NotFound {
            worker_id: worker_id_raw.to_string(),
        })
    }

    /// Delete a worker by ID (submits async job)
    pub async fn delete_worker(
        &self,
        worker_id_raw: &str,
    ) -> Result<DeleteWorkerResult, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        let job = Job::RemoveWorker {
            url: url.clone(),
            expected_revisions: None,
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(DeleteWorkerResult { worker_id, url })
    }

    /// Update a worker by ID (submits async job)
    pub async fn update_worker(
        &self,
        worker_id_raw: &str,
        update: WorkerUpdateRequest,
    ) -> Result<UpdateWorkerResult, WorkerServiceError> {
        let worker_id = Self::parse_worker_id(worker_id_raw)?;
        validate_update(&update)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;
        if let (Some(labels), Some(worker)) =
            (&update.labels, self.worker_registry.get_by_url(&url))
        {
            validate_merged_labels(&worker.metadata().spec.labels, labels)?;
        }

        let job = Job::UpdateWorker {
            url: url.clone(),
            update: Box::new(update),
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(UpdateWorkerResult { worker_id, url })
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::model_card::ModelCard;

    use super::*;
    use crate::{
        worker::{BasicWorkerBuilder, WorkerType},
        workflow::JobQueueConfig,
    };

    fn make_service(registry: Arc<WorkerRegistry>) -> WorkerService {
        WorkerService::new(
            registry,
            Arc::new(std::sync::OnceLock::new()),
            RouterConfig::default(),
        )
    }

    fn register_worker(registry: &WorkerRegistry, url: &str, model: Option<&str>) {
        let mut builder = BasicWorkerBuilder::new(url).worker_type(WorkerType::Regular);
        if let Some(model) = model {
            builder = builder.model(ModelCard::new(model));
        }
        registry.register(Arc::new(builder.build())).unwrap();
    }

    #[test]
    fn test_list_workers_unfiltered_returns_all() {
        let registry = Arc::new(WorkerRegistry::new());
        register_worker(&registry, "http://kimi:30000", Some("moonshotai/Kimi-K2.5"));
        register_worker(
            &registry,
            "http://llama:30000",
            Some("meta-llama/Llama-3.1-8B"),
        );
        let service = make_service(registry);

        let result = service.list_workers(None);
        assert_eq!(result.workers.len(), 2);
        assert_eq!(result.total, 2);
        assert_eq!(result.regular_count, 2);
    }

    #[test]
    fn test_list_workers_filters_by_model() {
        let registry = Arc::new(WorkerRegistry::new());
        register_worker(&registry, "http://kimi:30000", Some("moonshotai/Kimi-K2.5"));
        register_worker(
            &registry,
            "http://llama:30000",
            Some("meta-llama/Llama-3.1-8B"),
        );
        let service = make_service(registry);

        let result = service.list_workers(Some("moonshotai/Kimi-K2.5"));
        assert_eq!(result.workers.len(), 1);
        assert_eq!(
            result.workers[0].model_id.as_deref(),
            Some("moonshotai/Kimi-K2.5")
        );
        assert_eq!(result.total, 1);
        assert_eq!(result.regular_count, 1);

        let result = service.list_workers(Some("no-such-model"));
        assert!(result.workers.is_empty());
        assert_eq!(result.total, 0);
        assert_eq!(result.regular_count, 0);
    }

    #[test]
    fn test_list_workers_wildcard_worker_matches_any_model() {
        let registry = Arc::new(WorkerRegistry::new());
        register_worker(&registry, "http://wildcard:30000", None);
        let service = make_service(registry);

        let result = service.list_workers(Some("moonshotai/Kimi-K2.5"));
        assert_eq!(result.workers.len(), 1);
        assert_eq!(result.total, 1);
    }

    fn worker_spec(url: &str) -> WorkerSpec {
        serde_json::from_value(json!({ "url": url })).expect("worker spec")
    }

    #[tokio::test]
    async fn create_worker_rejects_schemeless_url_before_reserving() {
        let registry = Arc::new(WorkerRegistry::new());
        let service = make_service(registry);

        let err = service
            .create_worker(worker_spec("10.0.0.5:8000"))
            .await
            .expect_err("schemeless URL must be rejected at the boundary");

        assert!(matches!(err, WorkerServiceError::BadRequest { .. }));
        assert_eq!(err.error_code(), "BAD_REQUEST");
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn create_worker_rejects_mixed_case_scheme() {
        let registry = Arc::new(WorkerRegistry::new());
        let service = make_service(registry);

        let err = service
            .create_worker(worker_spec("HTTP://10.0.0.5:8000"))
            .await
            .expect_err("mixed-case scheme must be rejected at the boundary");

        assert!(matches!(err, WorkerServiceError::BadRequest { .. }));
    }

    #[tokio::test]
    async fn create_worker_accepts_schemed_url() {
        let registry = Arc::new(WorkerRegistry::new());
        let service = make_service(registry);

        // The harness leaves the job queue uninitialized, so a valid URL
        // passes boundary validation and stops at queue submission —
        // proving validation did not reject it.
        let err = service
            .create_worker(worker_spec("grpc://10.0.0.5:8000"))
            .await
            .expect_err("queue is uninitialized in the test harness");

        assert!(matches!(err, WorkerServiceError::QueueNotInitialized));
    }

    #[tokio::test]
    async fn create_worker_releases_reservation_when_submission_fails() {
        let registry = Arc::new(WorkerRegistry::new());
        // A queue whose AppContext is gone rejects every submission.
        let queue = Arc::new(std::sync::OnceLock::new());
        queue
            .set(JobQueue::new(
                JobQueueConfig::default(),
                std::sync::Weak::new(),
            ))
            .ok();
        let service = WorkerService::new(Arc::clone(&registry), queue, RouterConfig::default());
        // Idempotent, so this is the id create_worker will hand out.
        let reserved = registry.reserve_id_for_url("grpc://10.0.0.5:8000");

        let err = service
            .create_worker(worker_spec("grpc://10.0.0.5:8000"))
            .await
            .expect_err("submission must fail without an AppContext");
        assert!(matches!(err, WorkerServiceError::QueueSubmitFailed { .. }));

        // The 202 was never sent, so nothing may keep the URL reserved (#1533).
        assert!(registry.get_url_by_id(&reserved).is_none());
    }

    /// A registration whose labels or fields the routers could not read is
    /// refused at the boundary with a 400 naming the field, before any job
    /// is queued; one they can read passes the boundary (and stops at the
    /// uninitialized queue of this harness).
    #[tokio::test]
    async fn create_worker_refuses_labels_and_fields_the_routers_cannot_read() {
        let service = make_service(Arc::new(WorkerRegistry::new()));
        let url = "grpc://10.0.0.5:8000";
        let thousand: serde_json::Map<String, serde_json::Value> =
            (0..1000).map(|i| (format!("k{i}"), json!("v"))).collect();
        let refused = [
            (
                json!({"url": url, "labels": {"max_running_requests": "abc"}}),
                "max_running_requests",
            ),
            (
                json!({"url": url, "labels": {"max_running_requests": "0"}}),
                "max_running_requests",
            ),
            (
                json!({"url": url, "labels": {"max_num_seqs": "70000"}}),
                "max_num_seqs",
            ),
            (
                json!({"url": url, "labels": {"dp_size": "zero"}}),
                "dp_size",
            ),
            (
                json!({"url": url, "labels": {"supports_vision": "maybe"}}),
                "supports_vision",
            ),
            (
                json!({"url": url, "labels": {"note": "x".repeat(4097)}}),
                "note",
            ),
            (json!({"url": url, "labels": thousand}), "labels"),
            (
                json!({"url": url, "labels": {"key with space\nnewline": "v"}}),
                "label key",
            ),
            (json!({"url": url, "cost": 0}), "cost"),
            (json!({"url": url, "cost": -1.0}), "cost"),
            (json!({"url": url, "kv_block_size": 0}), "kv_block_size"),
            (json!({"url": url, "dp_rank": 1}), "dp_rank"),
            (json!({"url": url, "dp_rank": 2, "dp_size": 2}), "dp_rank"),
            (json!({"url": url, "bootstrap_port": 0}), "bootstrap_port"),
            (
                json!({"url": format!("grpc://{}:1", "a".repeat(2049))}),
                "url",
            ),
        ];
        for (body, field) in refused {
            let spec: WorkerSpec =
                serde_json::from_value(body.clone()).expect("a spec the type takes");
            let err = service
                .create_worker(spec)
                .await
                .expect_err(&body.to_string());
            assert!(
                matches!(err, WorkerServiceError::BadRequest { .. }),
                "{body}: {err}"
            );
            assert!(err.to_string().contains(field), "{body}: {err}");
        }
        let spec: WorkerSpec = serde_json::from_value(json!({
            "url": url, "cost": 2.5, "kv_block_size": 16, "dp_rank": 1, "dp_size": 2,
            "labels": {
                "max_running_requests": "256", "supports_vision": "true", "served_model_name": "m"
            }
        }))
        .expect("a valid spec");
        assert!(matches!(
            service
                .create_worker(spec)
                .await
                .expect_err("queue is uninitialized"),
            WorkerServiceError::QueueNotInitialized
        ));
    }

    /// An update is held to the same rules for the fields it may change.
    #[tokio::test]
    async fn update_worker_refuses_a_cost_or_labels_the_routers_cannot_read() {
        let registry = Arc::new(WorkerRegistry::new());
        register_worker(&registry, "grpc://10.0.0.5:8000", Some("m"));
        let id = registry.reserve_id_for_url("grpc://10.0.0.5:8000");
        let service = make_service(Arc::clone(&registry));
        for body in [
            json!({"cost": 0}),
            json!({"cost": -2.0}),
            json!({"labels": {"supports_vision": "yes"}}),
            json!({"labels": {"max_running_requests": "many"}}),
        ] {
            let update: WorkerUpdateRequest =
                serde_json::from_value(body.clone()).expect("an update the type takes");
            let err = service
                .update_worker(id.as_str(), update)
                .await
                .expect_err(&body.to_string());
            assert!(
                matches!(err, WorkerServiceError::BadRequest { .. }),
                "{body}: {err}"
            );
        }
        let update: WorkerUpdateRequest =
            serde_json::from_value(json!({"cost": 1.5, "labels": {"max_running_requests": "64"}}))
                .expect("a valid update");
        assert!(matches!(
            service
                .update_worker(id.as_str(), update)
                .await
                .expect_err("queue is uninitialized"),
            WorkerServiceError::QueueNotInitialized
        ));
    }

    /// The label cap holds for the worker, not the request: a PATCH merges
    /// its labels into the existing ones, so one that would leave more than
    /// the cap on the worker is refused, while one that only rewrites keys
    /// the worker has is taken.
    #[tokio::test]
    async fn update_worker_holds_the_label_cap_over_the_merged_labels() {
        let registry = Arc::new(WorkerRegistry::new());
        let full: HashMap<String, String> = (0..MAX_WORKER_LABELS)
            .map(|i| (format!("k{i}"), "v".to_string()))
            .collect();
        let worker = BasicWorkerBuilder::new("grpc://10.0.0.6:8000")
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .labels(full)
            .build();
        registry.register(Arc::new(worker)).unwrap();
        let id = registry.reserve_id_for_url("grpc://10.0.0.6:8000");
        let service = make_service(Arc::clone(&registry));
        let one_more: WorkerUpdateRequest =
            serde_json::from_value(json!({"labels": {"k_new": "v"}})).expect("an update");
        let err = service
            .update_worker(id.as_str(), one_more)
            .await
            .expect_err("a 129th label");
        assert!(
            matches!(&err, WorkerServiceError::BadRequest { message } if message.contains("129 labels on the worker")),
            "{err}"
        );
        let rewrite: WorkerUpdateRequest =
            serde_json::from_value(json!({"labels": {"k0": "w", "k1": "w"}})).expect("an update");
        assert!(matches!(
            service
                .update_worker(id.as_str(), rewrite)
                .await
                .expect_err("queue is uninitialized"),
            WorkerServiceError::QueueNotInitialized
        ));
    }

    /// A worker over the cap through the labels discovered at registration
    /// still takes an update that adds no key through the endpoint: a rewrite
    /// of keys it has reaches the queue, a new key is refused by name.
    #[tokio::test]
    async fn update_worker_over_the_cap_takes_a_rewrite_through_the_endpoint() {
        let registry = Arc::new(WorkerRegistry::new());
        let over: HashMap<String, String> = (0..MAX_WORKER_LABELS + 20)
            .map(|i| (format!("k{i}"), "v".to_string()))
            .collect();
        let worker = BasicWorkerBuilder::new("grpc://10.0.0.7:8000")
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .labels(over)
            .build();
        registry.register(Arc::new(worker)).unwrap();
        let id = registry.reserve_id_for_url("grpc://10.0.0.7:8000");
        let service = make_service(Arc::clone(&registry));
        let rewrite: WorkerUpdateRequest =
            serde_json::from_value(json!({"labels": {"k3": "w"}})).expect("an update");
        assert!(matches!(
            service
                .update_worker(id.as_str(), rewrite)
                .await
                .expect_err("queue is uninitialized"),
            WorkerServiceError::QueueNotInitialized
        ));
        let adds: WorkerUpdateRequest =
            serde_json::from_value(json!({"labels": {"k_new": "v"}})).expect("an update");
        let err = service
            .update_worker(id.as_str(), adds)
            .await
            .expect_err("a new key on a worker over the cap");
        assert!(
            matches!(&err, WorkerServiceError::BadRequest { message } if message.contains("149 labels on the worker") && message.contains("1 new")),
            "{err}"
        );
    }

    /// A rejected key is not echoed whole: a key the size of the payload
    /// limit comes back as its first characters and its length.
    #[test]
    fn a_rejected_label_key_is_shown_truncated() {
        let long = "k".repeat(1 << 20);
        let labels: HashMap<String, String> = [(long.clone(), "v".to_string())].into();
        let err = validate_labels(&labels).expect_err("a key over the cap");
        let WorkerServiceError::BadRequest { message } = err else {
            panic!("expected a bad request, got {err}");
        };
        assert!(message.len() < 512, "{} bytes", message.len());
        assert!(
            message.contains(&format!("({} bytes)", long.len())),
            "{message}"
        );
        let short: HashMap<String, String> = [("has space".to_string(), "v".to_string())].into();
        let err = validate_labels(&short).expect_err("a key with a space");
        assert!(err.to_string().contains("\"has space\""), "{err}");
    }
}
