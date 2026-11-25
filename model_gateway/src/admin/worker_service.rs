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
use openai_protocol::worker::{
    HealthCheckUpdate, WorkerErrorResponse, WorkerInfo, WorkerListOrigin, WorkerSpec, WorkerStatus,
    WorkerUpdateRequest,
};
use serde_json::json;
use smg_auth::AuditResource;
use smg_service_discovery::{DISCOVERY_LABEL_PREFIX, DISCOVERY_PROVIDER_LABEL};
use smg_worker::{
    health::manager::MAX_PROBE_DELAY,
    identity::{
        basic::{merge_labels, worker_to_info, Worker, WorkerType},
        endpoint::Endpoint,
    },
    jobs::{queue::QueueJob, WorkerJobAsker},
    labels::has_control_character,
    registry::WorkerId,
    workflow::find_workers_by_url,
};
use tracing::warn;

use crate::{
    config::{
        unused_credentials_warning, validate_worker_url, validation::ConfigValidator,
        CircuitBreakerConfig, RouterConfig,
    },
    routers::{common::snapshot::WorkerRegistry, provider_support},
    workflow::{
        job_queue::{add_worker_in_flight, cancel_registration},
        Job, JobQueue, WorkerRegistrationMode,
    },
};

/// Validate the URL of an incoming worker-management request, translating a
/// config-layer rejection into `400 Bad Request`. Must run before
/// `WorkerRegistry::reserve_id_for_url` so rejected input can never leave an
/// orphaned reservation (#1533).
fn validate_worker_url_request(url: &str) -> Result<(), WorkerServiceError> {
    validate_worker_url(url).map_err(|e| WorkerServiceError::BadRequest {
        message: e.to_string(),
    })?;
    if let Some(warning) = unused_credentials_warning(url) {
        warn!("{warning}");
    }
    Ok(())
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
/// Labels the routers read as the decode window, a positive count
/// (`Worker::max_running_requests`, which saturates a figure wider than
/// 65535); any other value is read as "not reported", silently.
const WINDOW_LABELS: [&str; 2] = ["max_running_requests", "max_num_seqs"];
/// Labels the routers read as a positive count.
const COUNT_LABELS: [&str; 1] = ["dp_size"];
/// Labels the routers read as a boolean, spelled `"true"` or `"false"`; any
/// other spelling is read as unset, silently.
const BOOLEAN_LABELS: [&str; 2] = ["supports_vision", "realtime"];

fn bad_request(message: String) -> WorkerServiceError {
    WorkerServiceError::BadRequest { message }
}

/// A worker's view with its address as a read shows it: a plain address
/// exactly as registered, any other in the form of the log lines
/// ([`WorkerSpec::redacted_url`]). Applied where the answer is written: the
/// engine load and the manifest export read the registered address.
fn redacted(mut info: WorkerInfo) -> WorkerInfo {
    let url = info.spec.redacted_url();
    // A rank's address is its group's base and `@rank`; a base that is no
    // part of the url is shown by its own value.
    let base = info
        .spec
        .dp_base_url
        .as_deref()
        .map(|base| match url.rsplit_once('@') {
            Some((shown, _)) if info.spec.url_is_ranked() => shown.to_string(),
            _ => WorkerSpec::new(base).redacted_url(),
        });
    // A status kept under another registration's key shows that one's
    // address, by its own value.
    if let Some(status) = &mut info.job_status {
        if status.worker_url == info.spec.url {
            status.worker_url.clone_from(&url);
        } else {
            status.worker_url = WorkerSpec::new(status.worker_url.as_str()).redacted_url();
        }
    }
    if url != info.spec.url || base != info.spec.dp_base_url {
        let spec = Arc::make_mut(&mut info.spec);
        spec.dp_base_url = base;
        spec.url = url;
    }
    info
}

/// The strings of a model card that log lines and a failed job's text write
/// as they came, each under its field's name in a registration body.
pub(crate) fn logged_strings(
    card: &openai_protocol::model_card::ModelCard,
) -> [(&'static str, Option<&str>); 3] {
    [
        ("models[].id", Some(card.id.as_str())),
        (
            "models[].reasoning_parser",
            card.reasoning_parser.as_deref(),
        ),
        ("models[].tool_parser", card.tool_parser.as_deref()),
    ]
}

/// The refusal for the first of `strings` that carries a control character.
pub(crate) fn control_character_refusal<'a>(
    strings: impl IntoIterator<Item = (&'static str, Option<&'a str>)>,
) -> Option<String> {
    strings
        .into_iter()
        .find(|(_, text)| text.is_some_and(has_control_character))
        .map(|(field, _)| format!("{field} must not contain control characters"))
}

/// The bounds a registration body must satisfy before any job is queued:
/// the URL's length, and no control character in it, in a model's id or in
/// a parser's name; the labels' count, charset and size, the shape of the
/// labels the routers read as numbers or booleans and none under the prefix
/// reserved for discovery; the fields the routers rely on (a positive,
/// finite cost; a KV block size and a data-parallel size of at least 1; a
/// rank only with a size and below it; a bootstrap port that is a port); a
/// probe interval and a probe timeout the health schedule can hold; breaker
/// overrides at the floors of the global settings. A violation is a
/// `400 Bad Request` naming the field.
fn validate_spec(spec: &WorkerSpec) -> Result<(), WorkerServiceError> {
    if spec.url.len() > MAX_WORKER_URL_BYTES {
        return Err(bad_request(format!(
            "url is {} bytes long; at most {MAX_WORKER_URL_BYTES} are accepted",
            spec.url.len()
        )));
    }
    // Log lines and a failed job's text write these as they came, so a line
    // break in one would start a line of the caller's choosing.
    let of_models = spec.models.iter().flat_map(logged_strings);
    let strings = std::iter::once(("url", Some(spec.url.as_str()))).chain(of_models);
    if let Some(refusal) = control_character_refusal(strings) {
        return Err(bad_request(refusal));
    }
    validate_labels(&spec.labels)?;
    validate_cost(spec.cost)?;
    validate_health(&spec.health)?;
    // The start checks the effective settings; here a given zero is refused
    // whatever the breaker's switches say, and an override left out passes.
    let breaker = &spec.resilience;
    ConfigValidator::validate_circuit_breaker(
        [
            "resilience.cb_failure_threshold",
            "resilience.cb_success_threshold",
            "resilience.cb_timeout_secs",
            "resilience.cb_window_secs",
        ],
        &CircuitBreakerConfig {
            failure_threshold: breaker.cb_failure_threshold.unwrap_or(1),
            success_threshold: breaker.cb_success_threshold.unwrap_or(1),
            timeout_duration_secs: breaker.cb_timeout_secs.unwrap_or(1),
            window_duration_secs: breaker.cb_window_secs.unwrap_or(1),
        },
    )
    .map_err(|e| bad_request(e.to_string()))?;
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
    if let Some(health) = &update.health {
        validate_health(health)?;
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

/// The health settings a caller sends: a probe interval and a probe timeout
/// the schedule can hold. The schedule adds each to its clock.
fn validate_health(health: &HealthCheckUpdate) -> Result<(), WorkerServiceError> {
    let max = MAX_PROBE_DELAY.as_secs();
    let delays = [
        ("check_interval_secs", health.check_interval_secs),
        ("timeout_secs", health.timeout_secs),
    ];
    for (field, secs) in delays {
        if let Some(secs) = secs.filter(|secs| *secs > max) {
            return Err(bad_request(format!(
                "health.{field} is {secs}; at most {max} (thirty years) are accepted"
            )));
        }
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
/// name, except a running window under the other name when the request sets
/// one, see `merge_labels`), held to the same cap as a registration, so repeated updates cannot
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
    let merged = merge_labels(existing, update).len();
    if merged > existing.len() && merged > MAX_WORKER_LABELS {
        return Err(bad_request(format!(
            "{merged} labels on the worker after this update ({} now, {} new); at most \
             {MAX_WORKER_LABELS} are accepted",
            existing.len(),
            merged - existing.len()
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
        // Discovery decides which workers it owns, and the health checker
        // which it may remove, by these labels: one set here would hand a
        // manual worker to a provider, or forge a discovered worker's record.
        if key.starts_with(DISCOVERY_LABEL_PREFIX) {
            return Err(bad_request(format!(
                "label '{key}' is under '{DISCOVERY_LABEL_PREFIX}', a prefix reserved for \
                 worker discovery"
            )));
        }
        if value.len() > MAX_LABEL_VALUE_BYTES || has_control_character(value) {
            return Err(bad_request(format!(
                "label '{key}' must be at most {MAX_LABEL_VALUE_BYTES} bytes without control \
                 characters"
            )));
        }
        let key_str = key.as_str();
        if WINDOW_LABELS.contains(&key_str) && !value.parse::<u64>().is_ok_and(|n| n >= 1) {
            return Err(bad_request(format!(
                "label '{key}' must be a positive count, not {value:?}"
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

/// A worker a discovery provider registered takes its spec and labels from
/// its discovery source, which is authoritative: a PUT, or a PATCH of its
/// labels, would disagree with the source until the source next changes the
/// worker and overwrites the edit. What no provider publishes (priority,
/// cost, API key, health checks) is still a PATCH's to change, and a DELETE
/// stays allowed; the provider adds the worker back while it still
/// publishes it.
fn refuse_discovery_owned(worker: &dyn Worker) -> Result<(), WorkerServiceError> {
    match discovery_provider(worker) {
        Some(provider) => Err(WorkerServiceError::DiscoveryOwned {
            url: worker.redacted_url().to_string(),
            provider: provider.clone(),
        }),
        None => Ok(()),
    }
}

/// The provider that owns `worker`, from the provenance label the
/// reconciler writes; `None` for a worker registered locally.
fn discovery_provider(worker: &dyn Worker) -> Option<&String> {
    worker.metadata().spec.labels.get(DISCOVERY_PROVIDER_LABEL)
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
    /// A worker at this URL is being removed: its removal drains it for
    /// `drain_secs` before the URL is free (a POST of the URL or a PUT of
    /// the worker inside that window)
    RemovalInProgress {
        url: String,
        worker_id: WorkerId,
        drain_secs: u64,
    },
    /// A create for this URL is still in flight (concurrent POST)
    /// `worker_id` is the reservation of the `POST` in flight, or `None` when
    /// the registration in flight has no id to poll: a startup or discovery
    /// one, or a `POST`'s before its reservation or after its release.
    CreateInProgress {
        url: String,
        worker_id: Option<WorkerId>,
    },
    /// A discovery provider owns the worker (a PUT, or a PATCH of labels, of a
    /// discovered worker)
    DiscoveryOwned { url: String, provider: String },
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
            // The worker is still registered at the URL while it drains: the
            // code the window answered before; the message says what holds.
            Self::RemovalInProgress { .. } => "WORKER_ALREADY_EXISTS",
            Self::CreateInProgress { .. } => "WORKER_CREATE_IN_PROGRESS",
            Self::DiscoveryOwned { .. } => "WORKER_DISCOVERY_OWNED",
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
            Self::RemovalInProgress { .. } => StatusCode::CONFLICT,
            Self::CreateInProgress { .. } => StatusCode::CONFLICT,
            Self::DiscoveryOwned { .. } => StatusCode::CONFLICT,
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
            Self::RemovalInProgress {
                url,
                worker_id,
                drain_secs,
            } => {
                let id = worker_id.as_str();
                write!(
                    f,
                    "Worker at URL '{url}' with ID {id} is being removed; its removal drains \
                    it for {drain_secs}s, the worker's configured drain, before the URL is \
                    free. Retry once GET /workers/{id} answers 404 or its job status reads \
                    failed."
                )
            }
            Self::CreateInProgress {
                url,
                worker_id: Some(worker_id),
            } => {
                let id = worker_id.as_str();
                write!(
                    f,
                    "Worker creation already in progress for URL '{url}' with ID {id}. \
                    Poll GET /workers/{id}."
                )
            }
            Self::CreateInProgress {
                url,
                worker_id: None,
            } => write!(
                f,
                "Worker registration already in progress for URL '{url}', with no id to \
                poll yet. Retry once it has ended."
            ),
            Self::DiscoveryOwned { url, provider } => write!(
                f,
                "Worker '{url}' is owned by the '{provider}' discovery provider. \
                Its spec and labels come from its discovery source: change them there, \
                not with a PUT or a PATCH of labels."
            ),
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
        // A registration refused for a removal in progress says when to
        // retry: the drain window, a second at least.
        let retry_after = if let Self::RemovalInProgress { drain_secs, .. } = &self {
            Some(http::HeaderValue::from((*drain_secs).max(1)))
        } else {
            None
        };
        let error = WorkerErrorResponse {
            error: self.to_string(),
            code: self.error_code().to_string(),
        };
        let mut response = (self.status_code(), Json(error)).into_response();
        if let Some(retry_after) = retry_after {
            response
                .headers_mut()
                .insert(http::header::RETRY_AFTER, retry_after);
        }
        response
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
        AuditResource::name(
            (
                StatusCode::ACCEPTED,
                [(http::header::LOCATION, self.location)],
                Json(response),
            )
                .into_response(),
            self.worker_id.as_str(),
        )
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
        AuditResource::name(
            (StatusCode::ACCEPTED, Json(response)).into_response(),
            self.worker_id.as_str(),
        )
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
        AuditResource::name(
            (StatusCode::ACCEPTED, Json(response)).into_response(),
            self.worker_id.as_str(),
        )
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
            workers: self.workers.into_iter().map(redacted).collect(),
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
        Json(redacted(self.0)).into_response()
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
                "Adding worker {} without an API key while the gateway has one configured: \
                the gateway's key ends at its auth middleware, so requests reach this worker \
                with no credential. If the worker requires a key (the gateway's or its own), \
                register it with that key explicitly.",
                Endpoint::redacted_str(&config.url)
            );
        }

        let worker_url = config.url.clone();
        let queue = self.get_job_queue()?;

        // A URL with a worker behind its mapping is a 409. A worker its
        // removal is draining holds the URL for the drain window; that answer
        // says so and when to retry, instead of offering a PUT or a PATCH of
        // a worker that is leaving. A data-parallel group stands under its
        // rank URLs, none under the id its POST answered: a rank that stays
        // answers for the group, else the rank whose drain its removal waits
        // for, the longest. Not while a registration of the URL is in flight:
        // its id may never reach the ranks found (ranks registered at startup
        // fail it, and its end releases the id). That POST is told to poll
        // here, not at the claim below: a registration that ended before the
        // claim would have this POST admitted over the ranks it registered.
        let live = match self
            .worker_registry
            .get_id_by_url_or_reservation(&worker_url)
        {
            Some(id) => match self.worker_registry.get(&id) {
                Some(worker) => Some((id, worker)),
                None if add_worker_in_flight(queue, &worker_url) => {
                    return Err(WorkerServiceError::CreateInProgress {
                        url: worker_url,
                        worker_id: Some(id),
                    });
                }
                None => find_workers_by_url(&self.worker_registry, &worker_url)
                    .into_iter()
                    .max_by_key(|rank| {
                        (
                            rank.status() != WorkerStatus::Draining,
                            rank.metadata().health_config.drain_settle_secs,
                        )
                    })
                    .map(|rank| (id, rank)),
            },
            None => None,
        };
        if let Some((worker_id, existing)) = live {
            let url = config.redacted_url();
            return Err(if existing.status() == WorkerStatus::Draining {
                WorkerServiceError::RemovalInProgress {
                    url,
                    worker_id,
                    drain_secs: existing.metadata().health_config.drain_settle_secs,
                }
            } else {
                WorkerServiceError::Conflict { url, worker_id }
            });
        }

        let job = Job::AddWorker {
            config: Box::new(config),
            registration_mode: WorkerRegistrationMode::CreateOnly,
            asker: WorkerJobAsker::Api,
        };

        // Exactly one create per URL is in flight at a time, so that attempt
        // alone owns the reservation and a failed one can safely release it;
        // a second create is told to poll the first rather than silently
        // merged into it, since its spec may differ (#1533). The queue's claim
        // comes before the reservation: a POST refused here reserves nothing,
        // so no id without a job behind it is left for a DELETE to mark, and
        // names the id in flight when the registration is a POST's (a startup
        // or discovery one reserves none; a POST's before its reservation or
        // after its release leaves none to name).
        if !queue.claim_if_idle(&job) {
            let worker_id = self
                .worker_registry
                .get_id_by_url_or_reservation(&worker_url);
            return Err(WorkerServiceError::CreateInProgress {
                url: WorkerSpec::new(worker_url).redacted_url(),
                worker_id,
            });
        }

        // The stable id for the 202, reserved behind the claim. A mapping
        // found here has no job behind it: a reservation left over (a
        // data-parallel registration leaves its base URL's once its ranks are
        // registered under their own URLs), adopted as it always was.
        let worker_id = self.worker_registry.reserve_id_for_url(&worker_url);

        queue.submit_claimed(job).await.map_err(|e| {
            // No 202 goes out, so nothing may keep this URL reserved (#1533).
            self.worker_registry.release_reservation(&worker_url);
            WorkerServiceError::QueueSubmitFailed { message: e }
        })?;

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
                url: config.redacted_url(),
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
        // Before the body is read: a discovered worker's own spec, sent back,
        // carries the reserved labels, and the useful answer is where to make
        // the change, not which labels to strip.
        refuse_discovery_owned(existing.as_ref())?;
        let url = existing.url().to_string();
        let shown_url = existing.redacted_url();
        validate_spec(&config)?;
        Self::require_provider_router(&config)?;

        // A data-parallel engine registers as one worker per rank, each
        // under a rank-suffixed URL. Re-running registration for the ID of
        // one rank cannot express that, so refuse here instead of answering
        // 202 and failing in the background.
        if existing.is_dp_aware() {
            return Err(WorkerServiceError::BadRequest {
                message: format!(
                    "Worker '{shown_url}' belongs to a data-parallel group. \
                    PUT is not supported for data-parallel workers. \
                    Use DELETE + POST instead."
                ),
            });
        }

        // Validate that the URL in the request body matches the existing worker.
        // URL changes are not supported via replace — use DELETE + POST instead.
        if config.url != url {
            return Err(WorkerServiceError::BadRequest {
                message: "URL mismatch: a PUT carries the worker's URL exactly as it was \
                    registered (a read may show it shortened). URL changes are not \
                    supported via PUT. Use DELETE + POST instead."
                    .to_string(),
            });
        }

        // A worker that a removal is draining is leaving: a PUT gets the
        // answer a POST of its URL gets. A replacement would share the
        // Draining status and be served again inside the drain.
        if existing.status() == WorkerStatus::Draining {
            return Err(WorkerServiceError::RemovalInProgress {
                url: shown_url.to_string(),
                worker_id,
                drain_secs: existing.metadata().health_config.drain_settle_secs,
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
            asker: WorkerJobAsker::Api,
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(UpdateWorkerResult { worker_id, url })
    }

    /// List all workers with their info, optionally filtered to workers
    /// serving `model` and to the workers of one `origin` (registered
    /// locally, or by a discovery provider). Building each `WorkerInfo` is
    /// cheap: `WorkerSpec` is shared via `Arc`, so there is no per-worker
    /// spec deep clone.
    ///
    /// The counts reflect the returned (filtered) list, not the whole
    /// registry.
    pub fn list_workers(
        &self,
        model: Option<&str>,
        origin: Option<WorkerListOrigin>,
    ) -> ListWorkersResult {
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
            if let Some(origin) = origin {
                let discovered = discovery_provider(worker.as_ref()).is_some();
                if discovered != (origin == WorkerListOrigin::Discovery) {
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

        // A registration in flight holds its id. One that failed has released
        // it: the queue keeps its failed status under the id until its
        // finished statuses age out.
        let url = self.worker_registry.get_url_by_id(&worker_id);
        let in_flight = url
            .as_deref()
            .and_then(|worker_url| job_queue.get_status(worker_url));
        let failed = || {
            job_queue
                .get_status(worker_id.as_str())
                .filter(|status| status.job_type == Job::REGISTRATION_TYPE)
        };
        if let Some(status) = in_flight.or_else(failed) {
            return Ok(GetWorkerResponse(WorkerInfo::pending(
                worker_id.as_str(),
                status.worker_url.clone(),
                Some(status),
            )));
        }

        // A registered data-parallel group stands under its rank URLs, none
        // under the id its POST answered. The read of that id answers with
        // the least ready rank, so a client that polls it reads healthy once
        // every rank is. Each rank is read once, into its view, so the
        // health compared is the health answered.
        let group = url.and_then(|url| {
            find_workers_by_url(&self.worker_registry, &url)
                .iter()
                .map(worker_to_info)
                .min_by_key(|rank| rank.is_healthy)
        });
        if let Some(mut worker_info) = group {
            worker_info.id = worker_id.as_str().to_string();
            return Ok(GetWorkerResponse(worker_info));
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
        let job_queue = self.get_job_queue()?;

        // An id with no worker behind it is a POST's registration still in
        // flight: queued, or polling an engine unreachable or still starting
        // for the startup timeout. Its removal is that poll's cancellation; the
        // cancelled job releases the id and the URL when it ends, and nothing
        // is registered to drain or remove. A registration that reached the
        // registry meanwhile is removed below as any worker. A startup or
        // discovery registration reserves no id, and a POST reserves only
        // once it holds the queue's claim (one refused for a registration in
        // flight reserves nothing), so no DELETE reaches this path for such a
        // registration before its worker exists.
        if self.worker_registry.get(&worker_id).is_none() {
            let cancelled = cancel_registration(job_queue, &url, &worker_id)
                .await
                .map_err(|message| WorkerServiceError::QueueSubmitFailed { message })?;
            if cancelled > 0 && self.worker_registry.get(&worker_id).is_none() {
                return Ok(DeleteWorkerResult { worker_id, url });
            }
        }

        // The removal is pinned to the worker this DELETE named, by the id
        // of the request itself, and to the rest of its group by theirs, at
        // no revision, as the removal step's guard reads it: a worker
        // rebuilt under its id before the removal runs (a PATCH or a PUT
        // answered earlier and still in flight, a label refresh) is the one
        // this DELETE named and goes; a worker registered under a fresh id
        // since is not and stays. A revision pin held the removal behind
        // the job that moves the revision and then skipped the worker,
        // every time.
        let mut entries = find_workers_by_url(&self.worker_registry, &url)
            .into_iter()
            .filter_map(|worker| {
                let id = self.worker_registry.get_id_by_url(worker.url())?;
                Some((id.as_str().to_string(), None))
            })
            .collect::<HashMap<String, Option<u64>>>();
        // The named worker's own id, not one read back through the URL: a
        // registration landing at the URL between the two reads is not this
        // request's to remove.
        entries.insert(worker_id.as_str().to_string(), None);
        let job = Job::RemoveWorker {
            url: url.clone(),
            expected_revisions: Some(entries),
            asker: WorkerJobAsker::Api,
        };
        // One removal per URL at a time: a registration's own end queues one
        // for the workers it registered and did not finish, and a repeated
        // DELETE would queue another; the second would find the worker
        // draining and remove it without the drain wait the first sits out.
        // The removal in flight holds the URL; this DELETE's removal is kept
        // and runs on its heels (`run_kept`), after it and never beside it.
        // A repeated DELETE's removal finds nothing left then; one behind a
        // guarded cleanup that skipped a worker replaced since the cleanup's
        // snapshot removes that worker, where a DELETE answered by the
        // cleanup alone removed nothing. A kept label refresh runs after
        // the kept removals, on any worker they leave standing.
        job_queue
            .submit_or_defer(job)
            .await
            .map_err(|message| WorkerServiceError::QueueSubmitFailed { message })?;

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
        let worker = self.worker_registry.get_by_url(&url);
        // Only a PATCH of labels edits what a provider publishes. Decided
        // before the body is validated, as for a PUT, so reserved labels sent
        // to a discovered worker are told where to make the change.
        if let (Some(_), Some(worker)) = (&update.labels, &worker) {
            refuse_discovery_owned(worker.as_ref())?;
        }
        validate_update(&update)?;
        if let (Some(labels), Some(worker)) = (&update.labels, &worker) {
            validate_merged_labels(&worker.metadata().spec.labels, labels)?;
        }
        let job = Job::UpdateWorker {
            url: url.clone(),
            update: Box::new(update),
            asker: WorkerJobAsker::Api,
        };
        let job_queue = self.get_job_queue()?;
        if worker.is_some() {
            job_queue
                .submit(job)
                .await
                .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;
        } else {
            // An id whose URL's registration job is still in the queue is a
            // registration in flight: it is polled (GET) or ended (DELETE), not
            // edited. An update job for it would write its status over the
            // registration's entry in the job queue, and with it the claim that
            // keeps a second POST of the URL off the same id. An id with no
            // worker under its URL and no such job is a completed data-parallel
            // group (its ranks stand under the rank URLs, the POST's id maps to
            // the base), whose update reaches every rank through the URL. The
            // check and the update's claim are one entry operation, and the
            // update writes only over its own entry once it holds a slot, so a
            // registration that claims the URL in between keeps its entry.
            let admitted = job_queue
                .submit_unless_registering(job)
                .await
                .map_err(|message| WorkerServiceError::QueueSubmitFailed { message })?;
            if !admitted {
                return Err(WorkerServiceError::CreateInProgress {
                    url: WorkerSpec::new(url).redacted_url(),
                    worker_id: Some(worker_id),
                });
            }
        }

        Ok(UpdateWorkerResult { worker_id, url })
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::model_card::ModelCard;
    use smg_worker::{identity::basic::WorkerType, jobs::queue::testing::QueueTest};

    use super::*;
    use crate::{
        app_context::testing::app_context_for_tests,
        routers::grpc::backend_client::BasicWorkerBuilder,
        workflow::{job_queue::recording_for_tests, JobQueueConfig},
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

        let result = service.list_workers(None, None);
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

        let result = service.list_workers(Some("moonshotai/Kimi-K2.5"), None);
        assert_eq!(result.workers.len(), 1);
        assert_eq!(
            result.workers[0].model_id.as_deref(),
            Some("moonshotai/Kimi-K2.5")
        );
        assert_eq!(result.total, 1);
        assert_eq!(result.regular_count, 1);

        let result = service.list_workers(Some("no-such-model"), None);
        assert!(result.workers.is_empty());
        assert_eq!(result.total, 0);
        assert_eq!(result.regular_count, 0);
    }

    #[test]
    fn test_list_workers_wildcard_worker_matches_any_model() {
        let registry = Arc::new(WorkerRegistry::new());
        register_worker(&registry, "http://wildcard:30000", None);
        let service = make_service(registry);

        let result = service.list_workers(Some("moonshotai/Kimi-K2.5"), None);
        assert_eq!(result.workers.len(), 1);
        assert_eq!(result.total, 1);
    }

    fn worker_spec(url: &str) -> WorkerSpec {
        serde_json::from_value(json!({ "url": url })).expect("worker spec")
    }

    #[tokio::test]
    async fn the_worker_api_asks_as_api() {
        use crate::app_context::testing::create_test_app_context;

        let context = create_test_app_context();
        let (queue, mut next_job) = recording_for_tests(&context);
        context.worker_job_queue.set(queue).unwrap();
        let registry = &context.worker_registry;
        let registered = |url| {
            register_worker(registry, url, Some("served"));
            registry.get_id_by_url(url).unwrap()
        };
        let replace_url = "http://replace:8000";
        let replace = registered(replace_url);
        let update = registered("http://update:8000");
        let delete = registered("http://delete:8000");
        let service = &context.worker_service;

        service
            .create_worker(worker_spec("http://create:8000"))
            .await
            .unwrap();
        assert!(matches!(
            next_job(),
            Job::AddWorker {
                registration_mode: WorkerRegistrationMode::CreateOnly,
                asker: WorkerJobAsker::Api,
                ..
            }
        ));
        service
            .replace_worker(replace.as_str(), worker_spec(replace_url))
            .await
            .unwrap();
        assert!(matches!(
            next_job(),
            Job::AddWorker {
                registration_mode: WorkerRegistrationMode::ReplaceById { .. },
                asker: WorkerJobAsker::Api,
                ..
            }
        ));
        let request = serde_json::from_value(json!({"priority": 2})).unwrap();
        service
            .update_worker(update.as_str(), request)
            .await
            .unwrap();
        assert!(matches!(
            next_job(),
            Job::UpdateWorker {
                asker: WorkerJobAsker::Api,
                ..
            }
        ));
        service.delete_worker(delete.as_str()).await.unwrap();
        assert!(matches!(
            next_job(),
            Job::RemoveWorker {
                asker: WorkerJobAsker::Api,
                ..
            }
        ));
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

    /// A worker URL that ends in `@` and digits is refused with a 400 and
    /// its reason: every line would name a rank of the host before the
    /// digits while the probe dialled the digits as the host. The answer
    /// shows the scheme of the address and no other part: the last two are
    /// a token alone, and a user with a password of digits, which the
    /// gateway's own rendering reads as a host and writes whole.
    #[tokio::test]
    async fn create_worker_refuses_an_address_read_two_ways() {
        let registry = Arc::new(WorkerRegistry::new());
        let service = make_service(registry);
        for url in [
            "http://127.0.0.1:35099@2130706433",
            "http://user:s3cret@127.0.0.1:52605@2130706433",
            "grpc://127.0.0.1:61845@2130706433",
            "http://s3cret-token@8080",
            "http://s3cret-user:8080@2130706433",
        ] {
            let err = service
                .create_worker(worker_spec(url))
                .await
                .expect_err(url);
            assert_eq!(err.status_code(), StatusCode::BAD_REQUEST, "{url}");
            let WorkerServiceError::BadRequest { message } = err else {
                panic!("expected a bad request, got {err}");
            };
            assert!(
                message.contains("cannot end in '@' and digits"),
                "{url}: {message}"
            );
            assert!(!message.contains("s3cret"), "{url}: {message}");
        }
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

    /// A registered data-parallel group stands under its rank URLs, none
    /// under the id its `POST` answered. The read of that id answers from
    /// the ranks, the least ready one, where it answered 404.
    #[tokio::test]
    async fn get_worker_reads_a_registered_data_parallel_group_by_the_id_of_its_post() {
        let registry = Arc::new(WorkerRegistry::new());
        let queue = Arc::new(std::sync::OnceLock::new());
        queue
            .set(JobQueue::new(
                JobQueueConfig::default(),
                std::sync::Weak::new(),
            ))
            .ok();
        let service = WorkerService::new(Arc::clone(&registry), queue, RouterConfig::default());
        let url = "grpc://10.0.0.7:8000";
        let id = registry.reserve_id_for_url(url);
        for rank in 0..2 {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("m"))
                .dp_config(rank, 2)
                .build();
            worker.set_status(WorkerStatus::Ready);
            registry
                .register(Arc::new(worker))
                .expect("a rank registers");
        }

        let GetWorkerResponse(group) = service
            .get_worker(id.as_str())
            .expect("the ranks answer for the id of the POST");
        assert_eq!(group.id, id.as_str());
        assert!(group.is_healthy, "every rank is ready");

        // The lookup's second rank, so that a read of the first one found
        // would still say healthy.
        let not_ready = find_workers_by_url(&registry, url)[1].clone();
        not_ready.set_status(WorkerStatus::NotReady);
        let GetWorkerResponse(group) = service
            .get_worker(id.as_str())
            .expect("the ranks answer for the id of the POST");
        assert!(!group.is_healthy, "one rank is not ready");
        assert_eq!(
            group.spec.url,
            not_ready.url(),
            "the view is the not-ready rank's"
        );
    }

    /// The job a `POST` of `url` claims the queue with (the claim reads the
    /// job's type and URL, not its registration mode).
    fn registration(url: &str) -> Job {
        Job::AddWorker {
            config: Box::new(worker_spec(url)),
            registration_mode: WorkerRegistrationMode::CreateOnly,
            asker: WorkerJobAsker::Api,
        }
    }

    /// A registration in flight holds the queue's claim for its URL, and a
    /// POST's has reserved the id its 202 named: a POST refused by that claim
    /// is told to poll that id and leaves the mapping and the queue alone (a
    /// queue with no context behind it fails any submission, so reaching it
    /// would show).
    #[tokio::test]
    async fn a_post_under_a_posts_registration_in_flight_is_told_to_poll_its_id_and_leaves_it() {
        let registry = Arc::new(WorkerRegistry::new());
        let url = "grpc://10.0.0.5:8000";
        let queue = JobQueue::new(JobQueueConfig::default(), std::sync::Weak::new());
        // The first POST's claim, then its reservation: create_worker's order.
        assert!(queue.claim_if_idle(&registration(url)));
        let reserved = registry.reserve_id_for_url(url);
        let held = Arc::new(std::sync::OnceLock::new());
        held.set(queue).ok();
        let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());
        let err = service
            .create_worker(worker_spec(url))
            .await
            .expect_err("another POST's registration is in flight");
        let WorkerServiceError::CreateInProgress { worker_id, .. } = err else {
            panic!("expected CreateInProgress, got {err:?}");
        };
        assert_eq!(
            worker_id,
            Some(reserved.clone()),
            "the id in flight is the one to poll"
        );
        assert_eq!(
            registry.get_id_by_url_or_reservation(url),
            Some(reserved),
            "the refused POST released a mapping that is not its own"
        );
    }

    /// A POST refused by the queue's claim reserves nothing: the registration
    /// in flight here is a startup or discovery one, which reserves no id, so
    /// there is none to poll, and none outlives the refusal for a DELETE to
    /// find.
    #[tokio::test]
    async fn a_post_refused_for_a_registration_of_another_kind_leaves_no_reservation() {
        let registry = Arc::new(WorkerRegistry::new());
        let url = "grpc://10.0.0.5:8000";
        let queue = JobQueue::new(JobQueueConfig::default(), std::sync::Weak::new());
        // The discovery registration's claim, with no reservation behind it.
        assert!(queue.claim_if_idle(&registration(url)));
        let held = Arc::new(std::sync::OnceLock::new());
        held.set(queue).ok();
        let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());
        let err = service
            .create_worker(worker_spec(url))
            .await
            .expect_err("the discovery registration is in flight");
        let WorkerServiceError::CreateInProgress { worker_id, .. } = err else {
            panic!("expected CreateInProgress, got {err:?}");
        };
        assert_eq!(worker_id, None, "no id of a POST's is in flight to poll");
        assert_eq!(
            registry.get_id_by_url_or_reservation(url),
            None,
            "the refused POST's reservation outlived the refusal"
        );
    }

    /// A mapping with no job behind it is a reservation left over (a
    /// data-parallel registration leaves its base URL's once its ranks are
    /// registered under their own URLs, and nothing releases it): the next
    /// POST adopts it and is admitted under that id, as it was before the
    /// queue's claim moved in front of the reservation.
    #[tokio::test]
    async fn a_leftover_reservation_with_no_job_in_flight_is_admitted_under_its_id() {
        let registry = Arc::new(WorkerRegistry::new());
        let url = "grpc://10.0.0.5:8000";
        let reserved = registry.reserve_id_for_url(url);
        let context = app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None);
        // A queue that accepts a job and never runs it: no slot to run one in.
        let queue = JobQueue::new(
            JobQueueConfig {
                max_concurrent_jobs: 0,
                ..JobQueueConfig::default()
            },
            Arc::downgrade(&context),
        );
        let held = Arc::new(std::sync::OnceLock::new());
        held.set(queue).ok();
        let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());
        let admitted = service
            .create_worker(worker_spec(url))
            .await
            .expect("a leftover reservation is no registration in flight");
        assert_eq!(admitted.worker_id, reserved, "the 202 names the adopted id");
        assert_eq!(registry.get_id_by_url_or_reservation(url), Some(reserved));
    }

    /// An id with no worker behind it is a registration in flight: a PATCH
    /// of it is told to poll the id and submits nothing, so the registration's
    /// entry in the job queue, and with it the claim that keeps a second POST
    /// off its id, stay its own. Before this, the update job's status took
    /// the entry and the second POST was admitted under the same id; a DELETE
    /// then cancelled the first registration alone, whose end released the
    /// shared reservation, and the second registered under a fresh id.
    #[tokio::test]
    async fn a_patch_of_a_registration_in_flight_is_told_to_poll_it_and_leaves_its_claim() {
        let registry = Arc::new(WorkerRegistry::new());
        let url = "grpc://10.0.0.5:8000";
        let context = app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None);
        // A queue that accepts jobs and never runs one: no slot to run one in.
        let queue = JobQueue::new(
            JobQueueConfig {
                max_concurrent_jobs: 0,
                ..JobQueueConfig::default()
            },
            Arc::downgrade(&context),
        );
        let held = Arc::new(std::sync::OnceLock::new());
        held.set(queue).ok();
        let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());
        let id = service
            .create_worker(worker_spec(url))
            .await
            .expect("the first POST is admitted")
            .worker_id;
        let patch: WorkerUpdateRequest =
            serde_json::from_value(json!({"labels": {"tier": "a"}})).expect("an update");
        let err = service
            .update_worker(id.as_str(), patch)
            .await
            .expect_err("the registration is in flight");
        assert!(
            matches!(&err, WorkerServiceError::CreateInProgress { worker_id: Some(in_flight), .. } if *in_flight == id),
            "{err:?}"
        );
        let err = service
            .create_worker(worker_spec(url))
            .await
            .expect_err("the registration still holds the URL");
        assert!(
            matches!(&err, WorkerServiceError::CreateInProgress { worker_id: Some(in_flight), .. } if *in_flight == id),
            "a second POST was admitted beside the registration: {err:?}"
        );
    }

    /// A successful data-parallel POST leaves its id mapped to the base URL,
    /// the ranks registered under the rank URLs and no job in flight: a PATCH
    /// by that id is admitted and reaches every rank, as before the in-flight
    /// guard, which read the empty base as a registration in flight and
    /// refused the group's PATCH on every try.
    #[tokio::test]
    async fn a_patch_of_a_completed_data_parallel_group_by_its_post_id_reaches_every_rank() {
        use crate::workflow::WorkflowEngines;

        let registry = Arc::new(WorkerRegistry::new());
        let url = "http://10.0.0.6:8000";
        let router_config = RouterConfig::default();
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
        // The group as its POST leaves it: the id reserved for the base URL,
        // the ranks registered under the rank URLs, the job over.
        let id = registry.reserve_id_for_url(url);
        for rank in 0..2 {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("served"))
                .dp_config(rank, 2)
                .build();
            registry
                .register(Arc::new(worker))
                .expect("a rank registers");
        }
        assert!(
            registry.get_by_url(url).is_none(),
            "nothing stands under the base URL itself"
        );
        assert_eq!(find_workers_by_url(&registry, url).len(), 2);

        let patch: WorkerUpdateRequest =
            serde_json::from_value(json!({"labels": {"tier": "a"}})).expect("an update");
        let updated = context
            .worker_service
            .update_worker(id.as_str(), patch)
            .await
            .expect("the group's PATCH is admitted");
        assert_eq!(updated.worker_id, id);

        // The update job reaches both ranks.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let ranks = find_workers_by_url(&registry, url);
            if ranks.len() == 2
                && ranks.iter().all(|worker| {
                    worker
                        .metadata()
                        .spec
                        .labels
                        .get("tier")
                        .map(String::as_str)
                        == Some("a")
                })
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the update did not reach every rank: {:?}",
                queue.get_status(url)
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    /// A DELETE behind a removal in flight for its URL keeps its removal for
    /// that removal's end (`run_kept`), where before it was answered by the
    /// removal in flight alone: a guarded cleanup (the entries a
    /// registration's end saw) skips a worker replaced since its snapshot,
    /// and a DELETE answered by it removed nothing.
    #[tokio::test]
    async fn a_delete_behind_a_removal_in_flight_keeps_its_removal() {
        let registry = Arc::new(WorkerRegistry::new());
        let url = "http://10.0.0.7:8000";
        let context = app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None);
        // A queue that accepts jobs and never runs one: no slot to run one in.
        let queue = JobQueue::new(
            JobQueueConfig {
                max_concurrent_jobs: 0,
                ..JobQueueConfig::default()
            },
            Arc::downgrade(&context),
        );
        let held = Arc::new(std::sync::OnceLock::new());
        held.set(Arc::clone(&queue)).ok();
        let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("served"))
            .build();
        let id = registry
            .register(Arc::new(worker))
            .expect("the worker registers");
        // The cleanup removal of a registration's end, pinned to the entry it
        // saw, holds the URL.
        let cleanup = Job::RemoveWorker {
            url: url.to_string(),
            expected_revisions: Some(HashMap::from([(id.as_str().to_string(), Some(1))])),
            asker: WorkerJobAsker::Api,
        };
        assert!(queue.claim_if_idle(&cleanup), "the cleanup holds the URL");
        let deleted = service
            .delete_worker(id.as_str())
            .await
            .expect("the DELETE is admitted");
        assert_eq!(deleted.worker_id, id);
        assert!(
            QueueTest::kept_for(&queue, url),
            "the DELETE's removal is kept for the removal in flight to run on its heels"
        );
    }

    /// The status, the body and the Retry-After header of an error's answer.
    async fn answer(err: WorkerServiceError) -> (StatusCode, serde_json::Value, Option<String>) {
        let response = err.into_response();
        let status = response.status();
        let retry_after = response
            .headers()
            .get(http::header::RETRY_AFTER)
            .map(|value| value.to_str().expect("a visible header").to_string());
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("a body");
        (
            status,
            serde_json::from_slice(&bytes).expect("a JSON body"),
            retry_after,
        )
    }

    /// A registration of a URL whose worker is being removed is refused for
    /// the drain window under the code it had, with an answer that says what
    /// holds: the removal in progress, the drain, when to retry, and a
    /// Retry-After of the drain. A live worker's refusal keeps its hint to
    /// replace or update it.
    #[tokio::test]
    async fn create_worker_names_the_removal_in_progress_of_a_draining_worker() {
        use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

        let registry = Arc::new(WorkerRegistry::new());
        let queue = Arc::new(std::sync::OnceLock::new());
        queue
            .set(JobQueue::new(
                JobQueueConfig::default(),
                std::sync::Weak::new(),
            ))
            .ok();
        let service = WorkerService::new(Arc::clone(&registry), queue, RouterConfig::default());
        let url = "grpc://10.0.0.8:8000";
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("m"))
                .health_config(HealthCheckConfig {
                    drain_settle_secs: 7,
                    ..Default::default()
                })
                .build(),
        );
        registry.register(Arc::clone(&worker)).unwrap();
        let id = registry.reserve_id_for_url(url);

        let (status, body, retry_after) = answer(
            service
                .create_worker(worker_spec(url))
                .await
                .expect_err("a live worker holds the URL"),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
        assert_eq!(body["code"], "WORKER_ALREADY_EXISTS", "{body}");
        let message = body["error"].as_str().expect("a message");
        assert!(
            message.contains(&format!("Use PUT /workers/{}", id.as_str())),
            "a live worker's refusal hints at the replacement: {message}"
        );
        assert_eq!(retry_after, None, "a live worker's refusal has no retry");

        worker.set_status(WorkerStatus::Draining);
        let (status, body, retry_after) = answer(
            service
                .create_worker(worker_spec(url))
                .await
                .expect_err("a draining worker holds the URL for its drain"),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{body}");
        assert_eq!(body["code"], "WORKER_ALREADY_EXISTS", "{body}");
        let message = body["error"].as_str().expect("a message");
        assert!(
            message.contains("is being removed")
                && message.contains("for 7s, the worker's configured drain"),
            "the refusal names the removal and its drain: {message}"
        );
        assert!(
            message.contains(&format!("GET /workers/{} answers 404", id.as_str())),
            "the refusal says when to retry: {message}"
        );
        assert!(
            !message.contains("PUT"),
            "no replacement of a worker that is leaving: {message}"
        );
        assert_eq!(
            retry_after.as_deref(),
            Some("7"),
            "Retry-After carries the drain window"
        );
    }

    /// A PUT re-runs one registration under one id, which a rank of a
    /// data-parallel engine cannot be: the PUT of a rank worker is refused by
    /// name, on a gateway given no setting for it. The PUT of a plain worker
    /// passes that test: here it stops at the next thing it needs, the job
    /// queue this service was given none of.
    #[tokio::test]
    async fn a_put_is_refused_for_a_rank_worker_and_for_no_other() {
        let registry = Arc::new(WorkerRegistry::new());
        let service = make_service(Arc::clone(&registry));
        let plain_url = "http://10.0.0.9:8000";
        register_worker(&registry, plain_url, Some("m"));
        let plain = registry.get_id_by_url(plain_url).expect("the plain worker");
        let engine = "http://10.0.0.10:8000";
        let rank = registry
            .register(Arc::new(
                BasicWorkerBuilder::new(engine)
                    .worker_type(WorkerType::Regular)
                    .model(ModelCard::new("m"))
                    .dp_config(1, 2)
                    .build(),
            ))
            .expect("a rank registers");

        let refused = service
            .replace_worker(rank.as_str(), worker_spec(engine))
            .await;
        assert!(
            matches!(
                &refused,
                Err(WorkerServiceError::BadRequest { message })
                    if message.contains("belongs to a data-parallel group")
            ),
            "a rank worker's PUT was not refused as one: {refused:?}"
        );
        let passed = service
            .replace_worker(plain.as_str(), worker_spec(plain_url))
            .await;
        assert!(
            matches!(&passed, Err(WorkerServiceError::QueueNotInitialized)),
            "a plain worker's PUT stopped short of the queue: {passed:?}"
        );
    }

    /// A worker that a removal is draining is leaving, and a PUT does not
    /// bring it back to serve inside the drain.
    #[tokio::test]
    async fn a_put_does_not_bring_a_draining_worker_back() {
        use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

        use crate::workflow::{
            create_worker_workflow_data, ActivateWorkersStep, RegisterWorkersStep,
        };

        let registry = Arc::new(WorkerRegistry::new());
        let service = make_service(Arc::clone(&registry));
        let url = "http://10.0.0.8:8000";
        let build = || -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new(url)
                    .worker_type(WorkerType::Regular)
                    .model(ModelCard::new("m"))
                    .build(),
            )
        };
        let worker = build();
        let id = registry.register(Arc::clone(&worker)).unwrap();
        // The run of a PUT answered now, bound to the revision it read.
        let run_of_a_put = |replacement: &Arc<dyn Worker>| {
            let mut data = create_worker_workflow_data(
                worker_spec(url),
                WorkerRegistrationMode::ReplaceById {
                    worker_id: id.as_str().to_string(),
                    expected_revision: worker.revision(),
                },
                app_context_for_tests(RouterConfig::default(), Arc::clone(&registry), None),
                WorkerJobAsker::Api,
            );
            data.actual_workers = Some(vec![Arc::clone(replacement)]);
            WorkflowContext::new(WorkflowInstanceId::new(), data)
        };

        let written = build();
        let mut run = run_of_a_put(&written);
        RegisterWorkersStep
            .execute(&mut run)
            .await
            .expect("the PUT writes its replacement");
        // A probe promotes the worker and a removal marks it after the PUT's
        // activation chose Ready, before it takes the worker's lock.
        let lost = registry.with_another_write_before_the_lock(
            |_| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("a runtime for the step")
                    .block_on(ActivateWorkersStep.execute(&mut run))
            },
            |registry| {
                registry.transition_status(&id, WorkerStatus::Ready);
                registry.transition_status(&id, WorkerStatus::Draining);
            },
        );
        assert!(
            matches!(&lost, Err(error) if error.to_string().contains("was not activated"))
                && written.status() == WorkerStatus::Draining,
            "a removal's mark behind the write of a PUT stands and fails its activation"
        );

        assert!(
            matches!(
                service.replace_worker(id.as_str(), worker_spec(url)).await,
                Err(WorkerServiceError::RemovalInProgress { .. })
            ),
            "a PUT of a draining worker is refused as a POST of its URL is"
        );

        let dropped = RegisterWorkersStep
            .execute(&mut run_of_a_put(&build()))
            .await;
        assert!(
            matches!(&dropped, Err(error) if error.to_string().contains("is being removed")),
            "the registry step of a PUT answered before the removal drops its replacement"
        );

        // The same revision is written without the Draining test: the refusal
        // above was the removal's, not a stale revision's.
        let rebuilt = build();
        assert!(registry.replace_claiming_local(&id, Arc::clone(&rebuilt), worker.revision()));
        ActivateWorkersStep
            .execute(&mut run)
            .await
            .expect("a superseded PUT has nothing to activate");
        assert_eq!(
            rebuilt.status(),
            WorkerStatus::Draining,
            "the activation of a PUT superseded since leaves its successor's status alone"
        );
    }

    /// A registered data-parallel group stands under its rank URLs, none
    /// under the id its `POST` answered. A second `POST` of its address is
    /// refused with that id while a rank stays, and as a removal with the
    /// longest drain once every rank is leaving. A registration of the
    /// address in flight is still the answer to poll.
    #[tokio::test]
    async fn create_worker_refuses_the_address_of_a_registered_data_parallel_group() {
        use openai_protocol::worker::HealthCheckConfig;

        let registry = Arc::new(WorkerRegistry::new());
        let queue = Arc::new(std::sync::OnceLock::new());
        queue
            .set(JobQueue::new(
                JobQueueConfig::default(),
                std::sync::Weak::new(),
            ))
            .ok();
        let service = WorkerService::new(Arc::clone(&registry), queue, RouterConfig::default());
        let url = "grpc://10.0.0.9:8000";
        let id = registry.reserve_id_for_url(url);
        for (rank, drain_settle_secs) in [3, 9].into_iter().enumerate() {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .model(ModelCard::new("m"))
                .health_config(HealthCheckConfig {
                    drain_settle_secs,
                    ..Default::default()
                })
                .dp_config(rank, 2)
                .build();
            registry
                .register(Arc::new(worker))
                .expect("a rank registers");
        }
        let ranks = find_workers_by_url(&registry, url);

        ranks[0].set_status(WorkerStatus::Draining);
        let err = service
            .create_worker(worker_spec(url))
            .await
            .expect_err("the rank that stays holds the address");
        assert!(
            matches!(&err, WorkerServiceError::Conflict { worker_id, .. } if *worker_id == id),
            "the refusal of a registered worker, with the id of the first POST: {err}"
        );

        ranks[1].set_status(WorkerStatus::Draining);
        let err = service
            .create_worker(worker_spec(url))
            .await
            .expect_err("the draining ranks hold the address");
        assert!(
            matches!(
                &err,
                WorkerServiceError::RemovalInProgress { worker_id, drain_secs: 9, .. }
                    if *worker_id == id
            ),
            "a group whose every rank is leaving is being removed, for its longest drain: {err}"
        );

        assert!(service
            .get_job_queue()
            .expect("the queue is set")
            .claim_if_idle(&registration(url)));
        let err = service
            .create_worker(worker_spec(url))
            .await
            .expect_err("a registration in flight holds the address");
        assert!(
            matches!(&err, WorkerServiceError::CreateInProgress { worker_id: Some(in_flight), .. } if *in_flight == id),
            "a registration in flight answers with its id to poll, not for the ranks: {err}"
        );
    }

    /// A control character in a string the log writes as it came is refused
    /// at the door, on POST and PUT: the URL, a model's id, a parser's name.
    #[tokio::test]
    async fn a_control_character_in_a_string_the_log_writes_is_refused_on_post_and_put() {
        let registry = Arc::new(WorkerRegistry::new());
        register_worker(&registry, "grpc://10.0.0.8:8000", Some("m"));
        let id = registry.reserve_id_for_url("grpc://10.0.0.8:8000");
        let service = make_service(Arc::clone(&registry));
        let url = "grpc://10.0.0.9:8000";
        let refused = [
            (
                "url",
                json!({"url": "grpc://10.0.0.9:8000/\nINJECTED level=ERROR forged"}),
            ),
            ("url", json!({"url": "grpc://10.0.0.9:8000/\rINJECTED"})),
            (
                "models[].id",
                json!({"url": url, "models": [{"id": "m\nINJECTED"}]}),
            ),
            (
                "models[].reasoning_parser",
                json!({"url": url, "models": [{"id": "m", "reasoning_parser": "p\nINJECTED"}]}),
            ),
            (
                "models[].tool_parser",
                json!({"url": url, "models": [{"id": "m", "tool_parser": "p\nINJECTED"}]}),
            ),
        ];
        for (field, body) in refused {
            let wanted = format!("{field} must not contain control characters");
            let spec = || serde_json::from_value::<WorkerSpec>(body.clone()).expect("a spec");
            let post = service.create_worker(spec()).await.expect_err("POST");
            let put = service
                .replace_worker(id.as_str(), spec())
                .await
                .expect_err("PUT");
            for (verb, err) in [("POST", post), ("PUT", put)] {
                assert!(
                    matches!(&err, WorkerServiceError::BadRequest { message } if *message == wanted),
                    "{verb} {field}: {err}"
                );
            }
        }
        let taken: WorkerSpec = serde_json::from_value(json!({
            "url": url,
            "models": [{"id": "m", "reasoning_parser": "p", "tool_parser": "p"}]
        }))
        .expect("a spec");
        assert!(matches!(
            service
                .create_worker(taken)
                .await
                .expect_err("queue is uninitialized"),
            WorkerServiceError::QueueNotInitialized
        ));
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
                json!({"url": url, "resilience": {"cb_failure_threshold": 0}}),
                "'resilience.cb_failure_threshold': 0 - Must be >= 1 (set to u32::MAX to effectively disable CB)",
            ),
            (
                json!({"url": url, "resilience": {"cb_success_threshold": 0}}),
                "'resilience.cb_success_threshold': 0 - Must be >= 1",
            ),
            (
                json!({"url": url, "resilience": {"cb_timeout_secs": 0}}),
                "'resilience.cb_timeout_secs': 0 - Must be > 0",
            ),
            (
                json!({"url": url, "resilience": {"cb_window_secs": 0}}),
                "'resilience.cb_window_secs': 0 - Must be > 0",
            ),
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
        // A window wider than 65535 passes the boundary: the routers saturate
        // it. A window set here is the worker's only window, kept as set on
        // every dp-aware rank worker; only a discovered one is divided per rank.
        let spec: WorkerSpec = serde_json::from_value(json!({
            "url": url, "cost": 2.5, "kv_block_size": 16, "dp_rank": 1, "dp_size": 2,
            "resilience": {"cb_failure_threshold": 1, "cb_success_threshold": 1, "cb_timeout_secs": 1, "cb_window_secs": 1},
            "labels": {
                "max_running_requests": "256", "max_num_seqs": "70000",
                "supports_vision": "true", "served_model_name": "m"
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

    /// The early check counts the labels the merge leaves: a window set by
    /// the update replaces a discovered one under the other name, so swapping
    /// the window's name on a full worker passes (as the step would take it),
    /// while a new key on a full worker is refused.
    #[test]
    fn the_early_cap_check_counts_the_labels_the_merge_leaves() {
        let mut full: HashMap<String, String> = (1..MAX_WORKER_LABELS)
            .map(|i| (format!("k{i}"), "v".to_string()))
            .collect();
        full.insert("max_num_seqs".to_string(), "256".to_string());
        assert_eq!(full.len(), MAX_WORKER_LABELS);
        let swap = HashMap::from([("max_running_requests".to_string(), "512".to_string())]);
        assert!(validate_merged_labels(&full, &swap).is_ok());
        assert_eq!(merge_labels(&full, &swap).len(), MAX_WORKER_LABELS);
        let new_key = HashMap::from([("k_new".to_string(), "v".to_string())]);
        let message = validate_merged_labels(&full, &new_key)
            .expect_err("a 129th label")
            .to_string();
        assert!(
            message.contains("129 labels on the worker after this update (128 now, 1 new)"),
            "{message}"
        );
        assert!(
            message.contains("new); at most 128 are accepted"),
            "{message}"
        );
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

    /// No request can write the labels discovery reads ownership from: a
    /// key under the reserved prefix is a 400 naming it on a POST, on a PUT
    /// and on a PATCH of a manual worker, before any job is queued; a key
    /// that only resembles the prefix passes.
    #[tokio::test]
    async fn a_label_under_the_discovery_prefix_is_refused_on_post_put_and_patch() {
        let registry = Arc::new(WorkerRegistry::new());
        register_worker(&registry, "grpc://10.0.0.8:8000", Some("m"));
        let id = registry.reserve_id_for_url("grpc://10.0.0.8:8000");
        let service = make_service(Arc::clone(&registry));
        let refused = |err: WorkerServiceError, key: &str| {
            assert!(
                matches!(&err, WorkerServiceError::BadRequest { message }
                    if message.contains(&format!("'{key}'"))
                        && message.contains("reserved for worker discovery")),
                "{key}: {err}"
            );
        };
        for key in [
            "smg.ai/discovery-provider",
            "smg.ai/discovery-id",
            "smg.ai/discovery-spec-hash",
            "smg.ai/discovery-anything",
        ] {
            let labels = json!({ key: "file" });
            let post: WorkerSpec =
                serde_json::from_value(json!({"url": "grpc://10.0.0.9:8000", "labels": labels}))
                    .expect("a spec");
            refused(service.create_worker(post).await.expect_err("POST"), key);
            let put: WorkerSpec =
                serde_json::from_value(json!({"url": "grpc://10.0.0.8:8000", "labels": labels}))
                    .expect("a spec");
            refused(
                service
                    .replace_worker(id.as_str(), put)
                    .await
                    .expect_err("PUT"),
                key,
            );
            let patch: WorkerUpdateRequest =
                serde_json::from_value(json!({ "labels": labels })).expect("an update");
            refused(
                service
                    .update_worker(id.as_str(), patch)
                    .await
                    .expect_err("PATCH"),
                key,
            );
        }
        let lookalikes = json!({
            "smg.ai/discovery": "v",
            "smg.ai/discoveryid": "v",
            "x/smg.ai/discovery-id": "v",
        });
        let post: WorkerSpec =
            serde_json::from_value(json!({"url": "grpc://10.0.0.9:8000", "labels": lookalikes}))
                .expect("a spec");
        assert!(matches!(
            service
                .create_worker(post)
                .await
                .expect_err("queue is uninitialized"),
            WorkerServiceError::QueueNotInitialized
        ));
    }

    /// A probe interval or timeout the health schedule cannot hold is refused
    /// at the door, on POST, PUT and PATCH; the longest it can hold is taken.
    #[tokio::test]
    async fn a_probe_delay_past_the_schedules_bound_is_refused_on_post_put_and_patch() {
        let registry = Arc::new(WorkerRegistry::new());
        register_worker(&registry, "grpc://10.0.0.8:8000", Some("m"));
        let id = registry.reserve_id_for_url("grpc://10.0.0.8:8000");
        let service = make_service(Arc::clone(&registry));
        for field in ["check_interval_secs", "timeout_secs"] {
            let refused = |err: WorkerServiceError, verb: &str| {
                assert!(
                    matches!(&err, WorkerServiceError::BadRequest { message }
                        if message.contains(&format!("health.{field} is"))),
                    "{verb} {field}: {err}"
                );
            };
            for secs in [946_080_001, u64::MAX] {
                let health = json!({ field: secs });
                let post: WorkerSpec = serde_json::from_value(
                    json!({"url": "grpc://10.0.0.9:8000", "health": health}),
                )
                .expect("a spec");
                refused(service.create_worker(post).await.expect_err("POST"), "POST");
                let put: WorkerSpec = serde_json::from_value(
                    json!({"url": "grpc://10.0.0.8:8000", "health": health}),
                )
                .expect("a spec");
                refused(
                    service
                        .replace_worker(id.as_str(), put)
                        .await
                        .expect_err("PUT"),
                    "PUT",
                );
                let patch: WorkerUpdateRequest =
                    serde_json::from_value(json!({ "health": health })).expect("an update");
                refused(
                    service
                        .update_worker(id.as_str(), patch)
                        .await
                        .expect_err("PATCH"),
                    "PATCH",
                );
            }
            let longest: WorkerSpec = serde_json::from_value(json!({
                "url": "grpc://10.0.0.9:8000",
                "health": { field: 946_080_000_u64 }
            }))
            .expect("a spec");
            assert!(matches!(
                service
                    .create_worker(longest)
                    .await
                    .expect_err("queue is uninitialized"),
                WorkerServiceError::QueueNotInitialized
            ));
        }
    }

    /// A discovered worker's spec and labels are changed at its discovery
    /// source: a PUT of it is a 409 naming its provider, whatever the body —
    /// the worker's own spec sent back included — and so is a PATCH that
    /// carries labels, reserved ones included. A PATCH of what no provider
    /// publishes, and a DELETE, are taken (the provider adds a deleted
    /// worker back while it still publishes it).
    #[tokio::test]
    async fn put_and_a_label_patch_of_a_discovered_worker_are_refused_naming_its_provider() {
        let registry = Arc::new(WorkerRegistry::new());
        let url = "grpc://10.0.0.10:8000";
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .model(ModelCard::new("m"))
            .label(DISCOVERY_PROVIDER_LABEL, "consul")
            .label("smg.ai/discovery-id", "node-1:8000")
            .build();
        registry.register(Arc::new(worker)).unwrap();
        let id = registry.reserve_id_for_url(url);
        let service = make_service(Arc::clone(&registry));
        let owned = |err: WorkerServiceError| {
            assert_eq!(err.status_code(), StatusCode::CONFLICT, "{err}");
            assert_eq!(err.error_code(), "WORKER_DISCOVERY_OWNED");
            let message = err.to_string();
            assert!(message.contains("'consul' discovery provider"), "{message}");
            assert!(message.contains("discovery source"), "{message}");
        };
        let own_spec = (*registry.get(&id).expect("registered").metadata().spec).clone();
        for put in [worker_spec(url), own_spec] {
            owned(
                service
                    .replace_worker(id.as_str(), put)
                    .await
                    .expect_err("PUT"),
            );
        }
        for body in [
            json!({"labels": {"env": "prod"}}),
            json!({"priority": 5, "labels": {"env": "prod"}}),
            json!({"labels": {DISCOVERY_PROVIDER_LABEL: "file"}}),
        ] {
            let patch: WorkerUpdateRequest = serde_json::from_value(body).expect("an update");
            owned(
                service
                    .update_worker(id.as_str(), patch)
                    .await
                    .expect_err("PATCH"),
            );
        }
        for body in [
            json!({"priority": 5}),
            json!({"cost": 2.0}),
            json!({"api_key": "rotated"}),
            json!({"health": {"timeout_secs": 9}}),
        ] {
            let patch: WorkerUpdateRequest =
                serde_json::from_value(body.clone()).expect("an update");
            assert!(
                matches!(
                    service.update_worker(id.as_str(), patch).await,
                    Err(WorkerServiceError::QueueNotInitialized)
                ),
                "{body} reaches the queue"
            );
        }
        assert!(matches!(
            service
                .delete_worker(id.as_str())
                .await
                .expect_err("queue is uninitialized"),
            WorkerServiceError::QueueNotInitialized
        ));
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

    #[test]
    fn list_workers_by_origin_tells_local_from_discovered() {
        let registry = Arc::new(WorkerRegistry::new());
        register_worker(&registry, "http://local:30000", None);
        let discovered: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://discovered:30000")
                .worker_type(WorkerType::Regular)
                .label(DISCOVERY_PROVIDER_LABEL, "file")
                .build(),
        );
        registry.register(discovered).unwrap();
        let service = make_service(registry);
        let urls = |result: ListWorkersResult| -> Vec<String> {
            result.workers.iter().map(|w| w.spec.url.clone()).collect()
        };

        assert_eq!(
            urls(service.list_workers(None, Some(WorkerListOrigin::Local))),
            ["http://local:30000"],
            "local: the worker the API registered"
        );
        assert_eq!(
            urls(service.list_workers(None, Some(WorkerListOrigin::Discovery))),
            ["http://discovered:30000"],
            "discovery: the worker a provider owns"
        );
        assert_eq!(service.list_workers(None, None).total, 2, "no origin: both");
    }

    #[tokio::test]
    async fn the_reads_of_a_worker_show_an_address_that_is_not_plain_in_the_log_form() {
        let registry = Arc::new(WorkerRegistry::new());
        let live = "http://name:made-up@10.0.0.5:8000";
        register_worker(&registry, live, None);
        let queue = JobQueue::new(JobQueueConfig::default(), std::sync::Weak::new());
        assert!(queue.claim_if_idle(&registration(live)));
        let held = Arc::new(std::sync::OnceLock::new());
        held.set(queue).ok();
        let service = WorkerService::new(Arc::clone(&registry), held, RouterConfig::default());

        let listed = service.list_workers(None, None);
        let id = listed.workers[0].id.clone();
        let read = service.get_worker(&id).expect("a read");
        let refused = service
            .create_worker(worker_spec(live))
            .await
            .expect_err("the address is registered")
            .to_string();
        let put = service
            .replace_worker(&id, worker_spec("http://10.0.0.6:8000"))
            .await
            .expect_err("a PUT with another address is refused")
            .to_string();

        assert_eq!(
            (
                listed.workers[0].spec.url.as_str(),
                read.0.spec.url.as_str()
            ),
            (live, live),
            "the load and the export read the registered address"
        );
        let mut written = Vec::new();
        for response in [listed.into_response(), read.into_response()] {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("a body");
            let body: serde_json::Value = serde_json::from_slice(&bytes).expect("a JSON body");
            let row = body.get("workers").map_or(&body, |rows| &rows[0]);
            written.push(row["url"].clone());
            written.extend(row["job_status"].get("worker_url").cloned());
        }
        let shown = worker_spec(live).redacted_url();
        assert_eq!(
            written,
            [json!(shown), json!(shown), json!(shown)],
            "url of the list's row, then url and job_status.worker_url of the read"
        );
        assert!(
            refused.contains(&shown) && !refused.contains("made-up"),
            "the refusal of a second registration: {refused}"
        );
        assert!(
            put.contains("URL mismatch") && !put.contains("10.0.0."),
            "the refusal of a PUT with another address names neither: {put}"
        );
    }
}
