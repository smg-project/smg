//! Worker Service - Business logic layer for worker operations
//!
//! This module provides a clean separation between HTTP concerns (in routers)
//! and business logic for worker management. The service orchestrates
//! WorkerRegistry and JobQueue operations.

use std::sync::Arc;

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

        self.get_job_queue()?
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

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
            expected_revision: None,
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

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

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
    use crate::worker::{BasicWorkerBuilder, WorkerType};

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
}
