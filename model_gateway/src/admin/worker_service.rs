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
#[path = "../../tests/unit/admin/worker_service/tests.rs"]
mod tests;
