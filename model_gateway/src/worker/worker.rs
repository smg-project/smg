use std::{
    any::Any,
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use arc_swap::{ArcSwap, ArcSwapOption};
use async_trait::async_trait;
use axum::body::Body;
// Re-export protocol types as the canonical types for the gateway
pub use openai_protocol::worker::{ConnectionMode, ProfileOptions, RuntimeType, WorkerType};
use openai_protocol::{
    model_card::ModelCard,
    model_type::{Endpoint, ModelType},
    worker::{HealthCheckConfig, ProviderType, WorkerInfo, WorkerModels, WorkerSpec, WorkerStatus},
};
use smg_grpc_client::common_proto;
use tokio::{
    sync::{mpsc, OnceCell},
    task::AbortHandle,
    time,
};

use super::{
    event::WorkerConnected, overload::OverloadThresholds, CircuitBreaker, ResolvedResilience,
    WorkerError, WorkerResult, UNKNOWN_MODEL_ID,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::header_utils::extract_routing_key,
        grpc::{backend_client::BackendClient, client::GrpcClient, zmq_client},
    },
};

/// Default HTTP client timeout for worker requests (in seconds)
pub const DEFAULT_WORKER_HTTP_TIMEOUT_SECS: u64 = 30;

/// A worker's HTTP client handle, materialized on first use.
///
/// Registration hands in a shared client from the worker client cache. A
/// worker built without one whose connection mode never speaks HTTP (ZMQ:
/// local health check, admin ops rejected up front) would otherwise pay for a
/// connector and idle pool it can never use, so the fallback client is built
/// only when a caller actually asks for it.
pub struct LazyHttpClient {
    cell: OnceLock<Arc<reqwest::Client>>,
}

impl LazyHttpClient {
    /// Wrap an already-built client (the registration paths hand one in).
    /// The strong handle is what keeps the client's cache entry alive.
    pub fn ready(client: Arc<reqwest::Client>) -> Self {
        let cell = OnceLock::new();
        let _ = cell.set(client);
        Self { cell }
    }

    /// Defer construction until [`Self::client`] is first called.
    pub fn deferred() -> Self {
        Self {
            cell: OnceLock::new(),
        }
    }

    /// Whether the client is still unbuilt (test-only observation of laziness).
    #[cfg(test)]
    pub(crate) fn cell_is_empty(&self) -> bool {
        self.cell.get().is_none()
    }

    /// The client, building the default one on first use.
    pub fn client(&self) -> &reqwest::Client {
        self.init()
    }

    /// A strong handle to the client if one was materialized, for a
    /// replacement worker to adopt. Never forces the lazy cell: a worker
    /// that never spoke HTTP hands its replacement a still-deferred slot.
    pub fn handle_if_initialized(&self) -> Option<Arc<reqwest::Client>> {
        self.cell.get().map(Arc::clone)
    }

    fn init(&self) -> &Arc<reqwest::Client> {
        self.cell.get_or_init(|| {
            Arc::new(
                reqwest::Client::builder()
                    .timeout(Duration::from_secs(DEFAULT_WORKER_HTTP_TIMEOUT_SECS))
                    .pool_max_idle_per_host(8)
                    .build()
                    .unwrap_or_else(|e| {
                        tracing::warn!(
                            error = %e,
                            "failed to build the default per-worker HTTP client; \
                             falling back to reqwest defaults (no request timeout)"
                        );
                        reqwest::Client::new()
                    }),
            )
        })
    }
}

/// Timeout for worker HTTP `flush_cache` requests. Matches the gRPC
/// client's local flush deadline.
const FLUSH_HTTP_TIMEOUT: Duration = Duration::from_secs(45);

/// Timeout for worker HTTP profile requests. Stopping a profile can take
/// a long time while the backend serializes large traces. Matches the
/// gRPC client's profile deadline.
const PROFILE_HTTP_TIMEOUT: Duration = Duration::from_secs(630);

/// Connect the ZMQ backend for this worker URL. All connect mechanics —
/// address derivation, socket-dir prep, EOS resolution, handshake — live in
/// the client layer ([`zmq_client::connect_for_worker`]); the worker layer
/// only wraps the client into its slot and error type.
async fn connect_zmq_backend(
    base_url: String,
    model_id: String,
    runtime: RuntimeType,
    handshake_override: Option<String>,
    engine_count: usize,
) -> WorkerResult<Arc<BackendClient>> {
    zmq_client::connect_for_worker(
        &base_url,
        model_id,
        runtime,
        handshake_override.as_deref(),
        engine_count,
    )
    .await
    .map(|client| Arc::new(BackendClient::Zmq(client)))
    .map_err(|reason| WorkerError::ConnectionFailed {
        url: base_url,
        reason,
    })
}

/// Default bootstrap port for PD disaggregation (used by SGLang and vLLM Mooncake)
pub const DEFAULT_BOOTSTRAP_PORT: u16 = 8998;

/// vLLM Mooncake KV connector name
pub const MOONCAKE_CONNECTOR: &str = "MooncakeConnector";

/// vLLM NIXL KV connector name
pub const NIXL_CONNECTOR: &str = "NixlConnector";

/// POST an admin endpoint on an HTTP worker and map the outcome to a
/// [`WorkerResult`].
async fn admin_http_post(
    client: &reqwest::Client,
    url: String,
    api_key: Option<&String>,
    body: Option<serde_json::Value>,
    operation: &str,
    timeout: Duration,
) -> WorkerResult<()> {
    let mut req = client.post(&url).timeout(timeout);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    if let Some(body) = body {
        req = req.json(&body);
    }
    match req.send().await {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(resp) => Err(WorkerError::OperationFailed {
            url,
            operation: operation.to_string(),
            reason: format!("HTTP {}", resp.status()),
        }),
        Err(e) => Err(WorkerError::OperationFailed {
            url,
            operation: operation.to_string(),
            reason: e.to_string(),
        }),
    }
}

/// Unwrap the optional gRPC client for an admin op, erroring when absent.
fn require_backend_client(
    url: &str,
    operation: &str,
    client: Option<Arc<BackendClient>>,
) -> WorkerResult<Arc<BackendClient>> {
    client.ok_or_else(|| WorkerError::OperationFailed {
        url: url.to_string(),
        operation: operation.to_string(),
        reason: "no backend client available".to_string(),
    })
}

/// EngineCore exposes no admin RPCs over ZMQ, so these ops can never succeed for
/// a ZMQ worker. Reject them up front rather than dispatching to client
/// acquisition, which would otherwise drive the (up to 10-minute) lazy handshake
/// only to return `unimplemented`.
fn zmq_admin_unsupported(url: &str, operation: &str) -> WorkerError {
    WorkerError::OperationFailed {
        url: url.to_string(),
        operation: operation.to_string(),
        reason: format!("{operation} is not supported over ZMQ"),
    }
}

/// Map a gRPC admin-op outcome (`success` flag plus message) to a
/// [`WorkerResult`].
fn admin_grpc_result(
    url: &str,
    operation: &str,
    result: Result<(bool, String), tonic::Status>,
) -> WorkerResult<()> {
    match result {
        Ok((true, _)) => Ok(()),
        Ok((false, message)) => Err(WorkerError::OperationFailed {
            url: url.to_string(),
            operation: operation.to_string(),
            reason: if message.is_empty() {
                "backend reported failure".to_string()
            } else {
                message
            },
        }),
        Err(status) => Err(WorkerError::OperationFailed {
            url: url.to_string(),
            operation: operation.to_string(),
            reason: status.to_string(),
        }),
    }
}

pub struct WorkerRoutingKeyLoad {
    url: String,
    active_routing_keys: dashmap::DashMap<String, usize>,
}

impl WorkerRoutingKeyLoad {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            active_routing_keys: dashmap::DashMap::new(),
        }
    }

    pub fn value(&self) -> usize {
        self.active_routing_keys.len()
    }

    /// In-flight requests for one routing key.
    pub fn key_inflight(&self, routing_key: &str) -> usize {
        self.active_routing_keys
            .get(routing_key)
            .map_or(0, |count| *count)
    }

    pub fn increment(&self, routing_key: &str) {
        *self
            .active_routing_keys
            .entry(routing_key.to_string())
            .or_insert(0) += 1;
        self.update_metrics();
    }

    pub fn decrement(&self, routing_key: &str) {
        use dashmap::mapref::entry::Entry;

        match self.active_routing_keys.entry(routing_key.to_string()) {
            Entry::Occupied(mut entry) => {
                let counter = entry.get_mut();
                if *counter > 0 {
                    *counter -= 1;
                    if *counter == 0 {
                        entry.remove();
                    }
                } else {
                    tracing::warn!(
                        worker_url = %self.url,
                        routing_key = %routing_key,
                        "Attempted to decrement routing key counter that is already at 0"
                    );
                }
            }
            Entry::Vacant(_) => {
                tracing::warn!(
                    worker_url = %self.url,
                    routing_key = %routing_key,
                    "Attempted to decrement non-existent routing key"
                );
            }
        }
        self.update_metrics();
    }

    fn update_metrics(&self) {
        Metrics::set_worker_routing_keys_active(&self.url, self.value());
    }
}

impl fmt::Debug for WorkerRoutingKeyLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerRoutingKeyLoad")
            .field("url", &self.url)
            .field("active_routing_keys", &self.value())
            .finish()
    }
}

/// Core worker abstraction that represents a backend service
#[async_trait]
pub trait Worker: Send + Sync + fmt::Debug + 'static {
    /// Downcast support for same-URL replace state sharing.
    fn as_any(&self) -> &dyn Any;

    /// Get the worker's URL
    fn url(&self) -> &str;
    /// Get the worker's API key
    fn api_key(&self) -> Option<&String>;
    /// Get the worker's type (Regular, Prefill, or Decode)
    /// Returns a reference to avoid cloning on every access
    fn worker_type(&self) -> &WorkerType;

    /// Get the worker's connection mode (HTTP or gRPC)
    /// Returns a reference to avoid cloning on every access
    fn connection_mode(&self) -> &ConnectionMode;

    /// Get the worker's lifecycle status.
    fn status(&self) -> WorkerStatus;

    /// Get the current monotonic worker revision.
    ///
    /// Same-URL `replace()` increments the revision so stale probe outcomes
    /// can be discarded without mutating the newly installed worker object.
    fn revision(&self) -> u64 {
        0
    }

    /// Set the worker's lifecycle status.
    fn set_status(&self, status: WorkerStatus);

    /// Adopt shared mutable runtime state from a previous worker object.
    ///
    /// Used by same-URL `replace()` so in-flight traffic and counters remain
    /// attached to a single shared runtime across the old and new objects.
    fn inherit_shared_state_from(&self, _other: &dyn Worker) -> bool {
        false
    }

    /// Check if the worker is currently healthy (status == Ready).
    ///
    /// This is a routing predicate — returns true only for `Ready` workers.
    /// A `Pending` worker is not "unhealthy", just unverified.
    fn is_healthy(&self) -> bool {
        self.status() == WorkerStatus::Ready
    }

    /// Perform an async health check on the worker.
    ///
    /// Pure probe — does not mutate worker status, does not increment
    /// counters beyond `consecutive_*` totals exposed via the accessors
    /// below. The state machine lives in `WorkerManager`, which reads the
    /// counters and applies transitions via `WorkerRegistry::transition_status()`.
    async fn check_health_async(&self) -> WorkerResult<()>;

    // ── Health check counter accessors (used by WorkerManager state machine) ──

    /// Increment `consecutive_failures` and return the new value.
    fn consecutive_failures_increment(&self) -> usize;

    /// Reset `consecutive_failures` to 0.
    fn consecutive_failures_reset(&self);

    /// Increment `consecutive_successes` and return the new value.
    fn consecutive_successes_increment(&self) -> usize;

    /// Reset `consecutive_successes` to 0.
    fn consecutive_successes_reset(&self);

    /// Read `total_pending_probes` (lifetime probe attempts in Pending state).
    fn total_pending_probes(&self) -> usize;

    /// Increment `total_pending_probes` and return the new value.
    fn total_pending_probes_increment(&self) -> usize;

    /// Reset `total_pending_probes` to 0 (called when promoting Pending → Ready).
    fn total_pending_probes_reset(&self);

    /// Get the current load (number of active requests)
    fn load(&self) -> usize;

    /// Increment the load counter
    fn increment_load(&self);

    /// Decrement the load counter
    fn decrement_load(&self);

    /// Get the current routing-key load cardinality.
    fn routing_key_load(&self) -> usize;

    /// In-flight requests for one routing key on this worker.
    fn routing_key_inflight(&self, routing_key: &str) -> usize;

    /// Increment the routing-key load tracker for an active key.
    fn increment_routing_key_load(&self, routing_key: &str);

    /// Decrement the routing-key load tracker for a completed key.
    fn decrement_routing_key_load(&self, routing_key: &str);

    /// Get the number of processed requests
    fn processed_requests(&self) -> usize;

    /// Increment the processed requests counter
    fn increment_processed(&self);

    /// Get worker-specific metadata
    fn metadata(&self) -> &WorkerMetadata;

    /// Worker-reported in-flight capacity, if available.
    ///
    /// Reads the `max_running_requests` label populated by the metadata
    /// discovery pipeline (Step 4 of the worker lifecycle). Returns
    /// `None` when the worker hasn't reported a value or reports zero
    /// (zero is meaningless for capacity accounting).
    ///
    /// `WorkerCapacity` uses this to derive total fleet capacity when
    /// every worker reports; falls back to a configured per-worker
    /// estimate otherwise.
    fn max_running_requests(&self) -> Option<u16> {
        self.metadata()
            .spec
            .labels
            .get("max_running_requests")
            .and_then(|s| s.parse::<u16>().ok())
            .filter(|n| *n > 0)
    }

    /// Whether this worker can serve the Realtime API (WS/WebRTC/REST
    /// relay). Reads the `realtime` label (`"true"`) populated via
    /// discovery, worker registration, or static config. Defaults to
    /// `false` so only explicitly-marked workers receive realtime traffic.
    fn is_realtime_capable(&self) -> bool {
        self.metadata()
            .spec
            .labels
            .get("realtime")
            .is_some_and(|v| v == "true")
    }

    /// Get the current circuit breaker state for observability/debugging.
    fn circuit_breaker_state(&self) -> super::circuit_breaker::CircuitState;

    /// Check whether the current circuit breaker state allows execution.
    fn circuit_breaker_can_execute(&self) -> bool;

    /// Record a request outcome against the circuit breaker.
    fn record_circuit_breaker_outcome(&self, success: bool);

    /// Check if the worker is available (healthy + circuit closed/half-open +
    /// not vetoed by the absolute overload guard).
    fn is_available(&self) -> bool {
        self.is_healthy() && self.circuit_breaker_can_execute() && !self.is_overloaded()
    }

    /// [`Self::is_healthy`] fused with the overload veto. For the hash policies,
    /// which route on health alone and never consult the circuit breaker;
    /// `BasicWorker` overrides it to read both under a single runtime guard.
    fn is_healthy_and_eligible(&self) -> bool {
        self.is_healthy() && !self.is_overloaded()
    }

    /// Whether the absolute overload guard currently vetoes this worker.
    ///
    /// Written only by the load monitor, once per ingested load report, and
    /// always `false` while overload protection is unconfigured.
    fn is_overloaded(&self) -> bool {
        false
    }

    /// Set the overload veto, returning `true` when the flag actually changed.
    ///
    /// Route writes through [`WorkerRegistry::set_worker_overloaded`] instead of
    /// calling this directly: the per-model counters and the
    /// `smg_workers_overloaded` gauge move only on transitions.
    fn set_overloaded(&self, _overloaded: bool) -> bool {
        false
    }

    /// One-shot routing snapshot for the per-request O(workers) selection loops:
    /// reads status, load, processed and the overload veto together so the hot
    /// path takes one `ArcSwap` guard per backing cell per worker instead of one
    /// per accessor (that guard traffic is a large share of routing CPU at
    /// scale). `BasicWorker` overrides this to share the runtime guard.
    fn routing_state(&self) -> RoutingState {
        RoutingState {
            healthy: self.is_healthy(),
            can_execute: self.circuit_breaker_can_execute(),
            load: self.load(),
            processed: self.processed_requests(),
            overloaded: self.is_overloaded(),
        }
    }

    /// Record the outcome of a request based on the HTTP status code.
    ///
    /// Statuses in the per-worker `capacity_status_codes` set (default: 429)
    /// record nothing at all — neither failure nor success. Any other status
    /// is a circuit-breaker failure when it appears in `retryable_status_codes`,
    /// which by default leaves 408, 500, 502, 503 and 504 tripping the breaker.
    /// 429 is in that set too, but the capacity check returns before it is read.
    /// Callers just pass the status — no need to interpret it.
    ///
    /// For transport/connection errors where no HTTP response is received,
    /// pass the status code returned to the client (e.g., 502 for a send
    /// error, 504 for a timeout).
    fn record_outcome(&self, status_code: u16) {
        let resilience = self.resilience();
        // Capacity pushback (429 by default) is a routing signal, not a
        // worker fault: the request is retried elsewhere, but no
        // circuit-breaker sample is recorded in either direction — opening
        // the breaker on backpressure would amplify a load spike into
        // unavailability, and crediting a success would close a half-open
        // breaker on a request the worker refused.
        if resilience.capacity_status_codes.contains(&status_code) {
            return;
        }
        let is_failure = resilience.retryable_status_codes.contains(&status_code);
        self.record_circuit_breaker_outcome(!is_failure);
    }

    /// Get the resolved resilience config for this worker.
    fn resilience(&self) -> &ResolvedResilience;

    /// Get the per-worker HTTP client.
    fn http_client(&self) -> &reqwest::Client;

    /// Strong handle to the worker's HTTP client, if one was materialized.
    /// Must not force a deferred client into existence; cache-fed workers
    /// return the shared handle so a replacement worker keeps the cache
    /// entry alive.
    fn http_client_handle_if_initialized(&self) -> Option<Arc<reqwest::Client>>;

    // ── Metadata convenience delegates ──────────────────────────────
    //
    // These default impls forward to the canonical implementation on
    // [`WorkerMetadata`] so callers can write `worker.foo()` instead
    // of the longer `worker.metadata().foo()`. Adding a new metadata
    // accessor means adding it to `impl WorkerMetadata` first and
    // then forwarding it here. Implementors of `Worker` should never
    // need to override these — `BasicWorker` and the FFI workers
    // both rely on the defaults.

    /// Get the bootstrap hostname for PD mode.
    fn bootstrap_host(&self) -> &str {
        self.metadata().bootstrap_host()
    }

    /// Get the bootstrap port for PD mode.
    fn bootstrap_port(&self) -> Option<u16> {
        self.metadata().bootstrap_port()
    }

    /// Get the base URL without any DP rank suffix.
    fn base_url(&self) -> &str {
        self.metadata().base_url()
    }

    /// Compose an endpoint URL for a specific route.
    fn endpoint_url(&self, route: &str) -> String {
        self.metadata().endpoint_url(route)
    }

    /// Check if this worker is DP-aware.
    fn is_dp_aware(&self) -> bool {
        self.metadata().is_dp_aware()
    }

    /// Get DP rank if this is a DP-aware worker.
    fn dp_rank(&self) -> Option<usize> {
        self.metadata().dp_rank()
    }

    /// Get DP size if this worker is part of a DP group.
    fn dp_size(&self) -> Option<usize> {
        self.metadata().dp_size()
    }

    /// Transform a request for DP-aware routing.
    ///
    /// When the worker has a `dp_rank`, injects `data_parallel_rank`
    /// into the request body. Otherwise returns the request unchanged.
    ///
    /// Any override that edits the body must also override
    /// [`Worker::mutates_request`] to return `true`.
    fn prepare_request(&self, req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        self.metadata().prepare_request(req)
    }

    /// Whether [`Worker::prepare_request`] rewrites the body. The HTTP proxy
    /// path skips the `serde_json::Value` round-trip when this is `false`.
    fn mutates_request(&self) -> bool {
        self.metadata().mutates_request()
    }

    /// Get the model ID this worker serves.
    fn model_id(&self) -> &str {
        self.metadata().model_id()
    }

    /// Get the priority of this worker (higher value = higher priority).
    fn priority(&self) -> u32 {
        self.metadata().priority()
    }

    /// Get the cost factor of this worker (baseline = 1.0).
    fn cost(&self) -> f32 {
        self.metadata().cost()
    }

    /// Get the default provider type for this worker.
    /// `None` means native/passthrough.
    fn default_provider(&self) -> Option<&ProviderType> {
        self.metadata().default_provider()
    }

    /// Get the provider for a specific model. Priority:
    /// `ModelCard.provider` > `worker.default_provider()`.
    fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.metadata().provider_for_model(model_id)
    }

    /// Check if this worker supports an endpoint for a given model.
    /// Falls back to LLM capabilities if the model is not registered.
    fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        self.metadata().supports_endpoint(model_id, endpoint)
    }

    /// Check if this worker supports a specific model.
    ///
    /// `BasicWorker` overrides this to consult its lazy-discovered
    /// `models_override`; the default delegates to the underlying
    /// [`WorkerMetadata::supports_model`].
    fn supports_model(&self, model_id: &str) -> bool {
        self.metadata().supports_model(model_id)
    }

    /// Get all models this worker can serve.
    fn models(&self) -> Vec<ModelCard> {
        self.metadata().spec.models.all().to_vec()
    }

    /// Set models for this worker (for lazy discovery).
    /// Default implementation does nothing - only BasicWorker supports this.
    fn set_models(&self, _models: Vec<ModelCard>) {
        // Default: no-op. BasicWorker overrides this.
    }

    /// Check if models have been discovered for this worker.
    /// Returns true if models were set via set_models() or if metadata has models.
    fn has_models_discovered(&self) -> bool {
        !self.metadata().spec.models.is_wildcard()
    }

    /// Get or create a gRPC client for this worker
    /// Returns None for HTTP workers, Some(client) for gRPC workers
    async fn get_backend_client(&self) -> WorkerResult<Option<Arc<BackendClient>>>;

    /// Reset the gRPC client connection (for reconnection scenarios)
    /// No-op for HTTP workers
    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        Ok(())
    }

    /// Release background work owned by this worker instance.
    ///
    /// Called by the registry when the worker leaves it (removal) or is
    /// superseded by a replacement. `BasicWorker` aborts the detached ZMQ
    /// handshake driver: it holds the handshake and data-plane socket binds
    /// until it lands, so an orphaned driver would keep them for up to the
    /// connect timeout and collide with a same-URL re-registration.
    fn abort_background_tasks(&self) {}
    async fn grpc_health_check(&self) -> WorkerResult<bool>;
    /// Liveness check for a ZMQ worker. Unlike gRPC there is no health RPC on
    /// the raw wire: liveness is local (handshake completed and the engine has
    /// not signalled `ENGINE_CORE_DEAD`).
    async fn zmq_health_check(&self) -> WorkerResult<bool>;
    async fn http_health_check(&self) -> WorkerResult<bool>;

    // ── Admin operations ────────────────────────────────────────────
    //
    // Dual-path dispatch on connection mode, mirroring
    // `check_health_async`: HTTP workers receive a POST to the engine's
    // native admin endpoint, gRPC workers receive the corresponding RPC.
    // Backends without the RPC surface the dispatcher's `Unimplemented`
    // status as an `OperationFailed` error.

    /// Flush the KV cache on this worker's backend.
    async fn flush_cache(&self) -> WorkerResult<()> {
        match self.connection_mode() {
            ConnectionMode::Http => {
                admin_http_post(
                    self.http_client(),
                    self.endpoint_url("/flush_cache"),
                    self.api_key(),
                    None,
                    "flush_cache",
                    FLUSH_HTTP_TIMEOUT,
                )
                .await
            }
            ConnectionMode::Zmq => Err(zmq_admin_unsupported(self.url(), "flush_cache")),
            ConnectionMode::Grpc => {
                let client = require_backend_client(
                    self.url(),
                    "flush_cache",
                    self.get_backend_client().await?,
                )?;
                let result = client.flush_cache(0.0).await;
                admin_grpc_result(
                    self.url(),
                    "flush_cache",
                    result.map(|r| (r.success, r.message)),
                )
            }
        }
    }

    /// Start a profiling run on this worker's backend.
    async fn start_profile(&self, options: &ProfileOptions) -> WorkerResult<()> {
        match self.connection_mode() {
            ConnectionMode::Http => {
                let body =
                    serde_json::to_value(options).map_err(|e| WorkerError::OperationFailed {
                        url: self.url().to_string(),
                        operation: "start_profile".to_string(),
                        reason: format!("failed to serialize profile options: {e}"),
                    })?;
                admin_http_post(
                    self.http_client(),
                    self.endpoint_url("/start_profile"),
                    self.api_key(),
                    Some(body),
                    "start_profile",
                    PROFILE_HTTP_TIMEOUT,
                )
                .await
            }
            ConnectionMode::Zmq => Err(zmq_admin_unsupported(self.url(), "start_profile")),
            ConnectionMode::Grpc => {
                let client = require_backend_client(
                    self.url(),
                    "start_profile",
                    self.get_backend_client().await?,
                )?;
                let req = common_proto::StartProfileRequest {
                    output_dir: options.output_dir.clone(),
                    start_step: options.start_step,
                    num_steps: options.num_steps,
                    activities: options.activities.clone().unwrap_or_default(),
                    with_stack: options.with_stack,
                    record_shapes: options.record_shapes,
                    profile_by_stage: options.profile_by_stage,
                };
                let result = client.start_profile(req).await;
                admin_grpc_result(
                    self.url(),
                    "start_profile",
                    result.map(|r| (r.success, r.message)),
                )
            }
        }
    }

    /// Stop the in-flight profiling run on this worker's backend and
    /// export traces.
    async fn stop_profile(&self) -> WorkerResult<()> {
        match self.connection_mode() {
            ConnectionMode::Http => {
                admin_http_post(
                    self.http_client(),
                    self.endpoint_url("/stop_profile"),
                    self.api_key(),
                    None,
                    "stop_profile",
                    PROFILE_HTTP_TIMEOUT,
                )
                .await
            }
            ConnectionMode::Zmq => Err(zmq_admin_unsupported(self.url(), "stop_profile")),
            ConnectionMode::Grpc => {
                let client = require_backend_client(
                    self.url(),
                    "stop_profile",
                    self.get_backend_client().await?,
                )?;
                let result = client.stop_profile().await;
                admin_grpc_result(
                    self.url(),
                    "stop_profile",
                    result.map(|r| (r.success, r.message)),
                )
            }
        }
    }
}

/// Extension trait for model_gateway-specific ConnectionMode methods.
pub(crate) trait ConnectionModeExt {
    fn as_metric_label(&self) -> &'static str;
}

impl ConnectionModeExt for ConnectionMode {
    fn as_metric_label(&self) -> &'static str {
        match self {
            ConnectionMode::Http => metrics_labels::CONNECTION_HTTP,
            ConnectionMode::Grpc => metrics_labels::CONNECTION_GRPC,
            ConnectionMode::Zmq => metrics_labels::CONNECTION_ZMQ,
        }
    }
}

/// Extension trait for model_gateway-specific WorkerType methods.
pub(crate) trait WorkerTypeExt {
    fn as_metric_label(&self) -> &'static str;
}

impl WorkerTypeExt for WorkerType {
    fn as_metric_label(&self) -> &'static str {
        match self {
            WorkerType::Regular => metrics_labels::WORKER_REGULAR,
            WorkerType::Prefill => metrics_labels::WORKER_PREFILL,
            WorkerType::Decode => metrics_labels::WORKER_DECODE,
            WorkerType::Encode => metrics_labels::WORKER_ENCODE,
        }
    }
}

/// Metadata associated with a worker.
///
/// Embeds [`WorkerSpec`] for identity/config fields shared with the
/// protocol layer, plus internal-only fields for health checking and
/// endpoint routing.
#[derive(Debug, Clone)]
pub struct WorkerMetadata {
    /// Protocol-level worker identity and configuration.
    ///
    /// Behind an `Arc` so it can be shared into response `WorkerInfo`s (e.g.
    /// `GET /workers`) without a per-worker deep clone. Set once at construction
    /// and not mutated afterwards.
    pub spec: Arc<WorkerSpec>,
    /// Resolved health check config (router defaults + per-worker overrides).
    /// This is the concrete config used at runtime; `spec.health` only stores
    /// the partial overrides from the API layer.
    pub health_config: HealthCheckConfig,
    /// Health check endpoint path (internal-only, from router config).
    pub health_endpoint: String,
    /// Effective absolute overload thresholds (per signal: `spec.overload`
    /// override, else gateway default). Resolved once at registration; the
    /// load monitor's ingestion predicate scores every report against these,
    /// so nothing on a request path re-resolves them.
    pub overload: OverloadThresholds,
}

impl WorkerMetadata {
    // ── Identity / transport ────────────────────────────────────────

    /// Get the bootstrap hostname for PD mode (parsed from URL at
    /// construction time).
    pub fn bootstrap_host(&self) -> &str {
        &self.spec.bootstrap_host
    }

    /// Get the bootstrap port for PD mode.
    pub fn bootstrap_port(&self) -> Option<u16> {
        self.spec.bootstrap_port
    }

    /// Get the base URL without any DP rank suffix.
    pub fn base_url(&self) -> &str {
        self.spec
            .dp_base_url
            .as_deref()
            .unwrap_or(self.spec.url.as_str())
    }

    /// Compose an endpoint URL for a specific route.
    pub fn endpoint_url(&self, route: &str) -> String {
        format!("{}{}", self.base_url(), route)
    }

    // ── DP awareness ────────────────────────────────────────────────

    /// Check if this worker is DP-aware.
    pub fn is_dp_aware(&self) -> bool {
        self.spec.dp_rank.is_some()
    }

    /// Get DP rank if this is a DP-aware worker.
    pub fn dp_rank(&self) -> Option<usize> {
        self.spec.dp_rank
    }

    /// Get DP size if this worker is part of a DP group.
    pub fn dp_size(&self) -> Option<usize> {
        self.spec.dp_size
    }

    /// Number of ZMQ engines this worker's handshake awaits. A grouped ZMQ
    /// worker carries `dp_size = Some(N)` with no `dp_rank` — one worker, one
    /// socket set, N engines dialing in (the connector balances across them
    /// and drives the wave protocol for lockstep groups). Rank-expanded DP
    /// workers never reach ZMQ connect (rejected at registration).
    pub fn zmq_engine_count(&self) -> usize {
        self.spec.dp_size.unwrap_or(1).max(1)
    }

    /// Transform a request for DP-aware routing.
    ///
    /// When the worker has a `dp_rank`, injects `data_parallel_rank`
    /// into the request body. Otherwise returns the request unchanged.
    /// Sync because the body is pure JSON manipulation — the previous
    /// `async` on the trait method had no `await` inside.
    pub fn prepare_request(&self, mut req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        if let Some(rank) = self.spec.dp_rank {
            if let Some(map) = req.as_object_mut() {
                map.insert("data_parallel_rank".to_string(), serde_json::json!(rank));
                Ok(req)
            } else {
                Err(WorkerError::InvalidConfiguration {
                    message: "Request must be a JSON object for DP-aware routing".to_string(),
                })
            }
        } else {
            Ok(req)
        }
    }

    /// True when [`Self::prepare_request`] would modify the request.
    pub fn mutates_request(&self) -> bool {
        self.spec.dp_rank.is_some()
    }

    // ── Routing priorities / model lookup ───────────────────────────

    /// Get the model ID this worker serves.
    ///
    /// Checks `ModelCards` first, then falls back to the `model_id`
    /// label, and finally [`UNKNOWN_MODEL_ID`] if nothing is set.
    pub fn model_id(&self) -> &str {
        self.spec
            .models
            .primary()
            .map(|m| m.id.as_str())
            .or_else(|| self.spec.labels.get("model_id").map(|s| s.as_str()))
            .unwrap_or(UNKNOWN_MODEL_ID)
    }

    /// Get the priority of this worker (higher value = higher priority).
    pub fn priority(&self) -> u32 {
        self.spec.priority
    }

    /// Get the cost factor of this worker (baseline = 1.0).
    pub fn cost(&self) -> f32 {
        self.spec.cost
    }

    /// Get the default provider type for this worker.
    /// `None` means native/passthrough.
    pub fn default_provider(&self) -> Option<&ProviderType> {
        self.spec.provider.as_ref()
    }

    // ── Model lookups ───────────────────────────────────────────────

    /// Find a model card by ID (including aliases)
    pub fn find_model(&self, model_id: &str) -> Option<&ModelCard> {
        self.spec.models.find(model_id)
    }

    /// Check if this worker can serve a given model.
    /// Wildcard workers accept any model.
    pub fn supports_model(&self, model_id: &str) -> bool {
        self.spec.models.supports(model_id)
    }

    /// Check if this worker supports an endpoint for a given model.
    /// Falls back to LLM capabilities if model not found — this is safe because
    /// non-LLM workers (embeddings, rerank) are always registered with explicit
    /// models via discovery, never as wildcards.
    pub fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        if let Some(model) = self.find_model(model_id) {
            model.supports_endpoint(endpoint)
        } else {
            ModelType::LLM.supports_endpoint(endpoint)
        }
    }

    /// Get the provider for a given model.
    /// Returns the model's provider if found, otherwise the worker's default provider.
    pub fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.find_model(model_id)
            .and_then(|m| m.provider.as_ref())
            .or(self.spec.provider.as_ref())
    }

    /// Get all model IDs this worker can serve
    pub fn model_ids(&self) -> impl Iterator<Item = &str> {
        self.spec.models.iter().map(|m| m.id.as_str())
    }

    /// Check if this worker is in wildcard mode (accepts any model).
    pub fn is_wildcard(&self) -> bool {
        self.spec.models.is_wildcard()
    }
}

/// One-shot routing snapshot — see [`Worker::routing_state`].
#[derive(Clone, Copy, Debug)]
pub struct RoutingState {
    /// `status == Ready`.
    pub healthy: bool,
    /// Circuit breaker permits execution (closed or half-open).
    pub can_execute: bool,
    /// Active in-flight request count.
    pub load: usize,
    /// Lifetime processed-request count (min-load tie-break).
    pub processed: usize,
    /// Absolute overload veto, set by the load monitor at ingestion time.
    pub overloaded: bool,
}

impl RoutingState {
    /// The full routing eligibility test. Costs nothing beyond the reads the
    /// gather pass already performed: every field rides the one guard
    /// [`Worker::routing_state`] took.
    pub const fn eligible(self) -> bool {
        self.healthy && self.can_execute && !self.overloaded
    }
}

/// Shared mutable worker state preserved across same-URL replacements.
#[derive(Debug)]
pub struct WorkerRuntime {
    status: AtomicU8,
    consecutive_failures: AtomicUsize,
    consecutive_successes: AtomicUsize,
    total_pending_probes: AtomicUsize,
    load_counter: AtomicUsize,
    processed_counter: AtomicUsize,
    worker_routing_key_load: WorkerRoutingKeyLoad,
    revision: AtomicU64,
    /// Absolute overload veto. Lives here rather than in the load-snapshot map
    /// so selection reads it under the guard it already holds, and so a
    /// same-URL replacement inherits it with the rest of the shared runtime.
    overloaded: AtomicBool,
}

impl WorkerRuntime {
    pub fn new(url: &str, initial_status: WorkerStatus) -> Self {
        Self {
            status: AtomicU8::new(initial_status as u8),
            consecutive_failures: AtomicUsize::new(0),
            consecutive_successes: AtomicUsize::new(0),
            total_pending_probes: AtomicUsize::new(0),
            load_counter: AtomicUsize::new(0),
            processed_counter: AtomicUsize::new(0),
            worker_routing_key_load: WorkerRoutingKeyLoad::new(url),
            revision: AtomicU64::new(0),
            overloaded: AtomicBool::new(false),
        }
    }

    // ── Lifecycle status ────────────────────────────────────────────

    pub fn status(&self) -> WorkerStatus {
        WorkerStatus::from_u8(self.status.load(Ordering::Acquire))
    }

    pub fn set_status(&self, status: WorkerStatus) {
        self.status.store(status as u8, Ordering::Release);
    }

    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    pub fn bump_revision(&self) -> u64 {
        self.revision.fetch_add(1, Ordering::AcqRel) + 1
    }

    // ── Health-check counters ───────────────────────────────────────

    pub fn consecutive_failures_increment(&self) -> usize {
        self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn consecutive_failures_reset(&self) {
        self.consecutive_failures.store(0, Ordering::Release);
    }

    pub fn consecutive_successes_increment(&self) -> usize {
        self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1
    }

    pub fn consecutive_successes_reset(&self) {
        self.consecutive_successes.store(0, Ordering::Release);
    }

    pub fn total_pending_probes(&self) -> usize {
        self.total_pending_probes.load(Ordering::Relaxed)
    }

    pub fn total_pending_probes_increment(&self) -> usize {
        self.total_pending_probes.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn total_pending_probes_reset(&self) {
        self.total_pending_probes.store(0, Ordering::Relaxed);
    }

    // ── Load counter ────────────────────────────────────────────────

    pub fn load(&self) -> usize {
        self.load_counter.load(Ordering::Relaxed)
    }

    pub fn increment_load(&self) {
        self.load_counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Saturating decrement. Returns `true` if the counter was decremented,
    /// `false` if it was already zero — callers can log when that happens.
    pub fn try_decrement_load(&self) -> bool {
        self.load_counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            })
            .is_ok()
    }

    // ── Routing-key load ────────────────────────────────────────────

    pub fn routing_key_load(&self) -> usize {
        self.worker_routing_key_load.value()
    }

    pub fn routing_key_inflight(&self, routing_key: &str) -> usize {
        self.worker_routing_key_load.key_inflight(routing_key)
    }

    pub fn increment_routing_key_load(&self, routing_key: &str) {
        self.worker_routing_key_load.increment(routing_key);
    }

    pub fn decrement_routing_key_load(&self, routing_key: &str) {
        self.worker_routing_key_load.decrement(routing_key);
    }

    // ── Processed-request counter ───────────────────────────────────

    pub fn processed_requests(&self) -> usize {
        self.processed_counter.load(Ordering::Relaxed)
    }

    pub fn increment_processed(&self) {
        self.processed_counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn is_overloaded(&self) -> bool {
        self.overloaded.load(Ordering::Relaxed)
    }

    /// Returns `true` when the flag transitioned, so the caller can move the
    /// per-model counters and gauge exactly once per edge.
    pub fn set_overloaded(&self, overloaded: bool) -> bool {
        self.overloaded.swap(overloaded, Ordering::Relaxed) != overloaded
    }
}

/// Basic worker implementation
pub struct BasicWorker {
    pub metadata: WorkerMetadata,
    pub runtime: ArcSwap<WorkerRuntime>,
    pub circuit_breaker: ArcSwap<CircuitBreaker>,
    /// Lazily initialized backend client (gRPC or ZMQ) for local workers.
    /// The inner `OnceCell` keeps reads lock-free after initialization and
    /// dedupes concurrent connects; the outer `ArcSwap` lets the ZMQ health
    /// probe evict a dead client by swapping in a fresh cell (gRPC never
    /// swaps — a failed gRPC worker is removed and re-added instead).
    pub backend_client: Arc<ArcSwap<OnceCell<Arc<BackendClient>>>>,
    /// Guards the one-shot background ZMQ handshake driver so the health probe
    /// never cancels a long (model-load) handshake. Self-clears on failure to
    /// allow a retry. Unused for HTTP/gRPC.
    pub zmq_connect_started: Arc<AtomicBool>,
    /// Abort handle for that driver, so removing or replacing the worker
    /// releases the sockets the in-flight handshake has bound instead of
    /// leaving them held until it times out. `None` until a driver is spawned.
    pub zmq_connect_abort: Arc<ArcSwapOption<AbortHandle>>,
    /// Wakes the manager the instant the ZMQ handshake completes so it can
    /// promote the worker without waiting for the next health poll. Set only
    /// for ZMQ workers built through the registration path; `None` elsewhere
    /// (HTTP/gRPC, tests), where promotion stays poll-driven.
    pub connect_signal_tx: Option<mpsc::UnboundedSender<WorkerConnected>>,
    /// Runtime-mutable models override (for lazy discovery).
    /// When not `Wildcard`, overrides metadata.models for routing decisions.
    /// Uses `ArcSwap` for lock-free reads on the hot path (`supports_model`).
    pub models_override: Arc<ArcSwap<WorkerModels>>,
    /// Worker-directed HTTP client, shared across same-config workers, built
    /// on first use (see [`LazyHttpClient`]).
    pub http_client: Arc<LazyHttpClient>,
    /// Resolved resilience config (retry + circuit breaker settings).
    pub resilience: ResolvedResilience,
}

impl Clone for BasicWorker {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            runtime: ArcSwap::from(self.runtime.load_full()),
            circuit_breaker: ArcSwap::from(self.circuit_breaker.load_full()),
            backend_client: Arc::clone(&self.backend_client),
            zmq_connect_started: Arc::clone(&self.zmq_connect_started),
            zmq_connect_abort: Arc::clone(&self.zmq_connect_abort),
            connect_signal_tx: self.connect_signal_tx.clone(),
            models_override: Arc::clone(&self.models_override),
            http_client: Arc::clone(&self.http_client),
            resilience: self.resilience.clone(),
        }
    }
}

impl fmt::Debug for BasicWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let runtime = self.runtime.load();
        f.debug_struct("BasicWorker")
            .field("metadata", &self.metadata)
            .field("status", &runtime.status())
            .field("revision", &runtime.revision())
            .field("circuit_breaker_state", &self.circuit_breaker_state())
            .field("grpc_client", &"<OnceCell>")
            .finish()
    }
}

impl BasicWorker {
    fn update_running_requests_metrics(&self) {
        let load = self.load();
        Metrics::set_worker_requests_active(self.url(), load);
    }

    fn shared_runtime(&self) -> Arc<WorkerRuntime> {
        self.runtime.load_full()
    }

    /// Start the one-shot background ZMQ handshake driver for `cell` unless one
    /// is already in flight.
    ///
    /// The handshake is never driven inline: it can take as long as a model
    /// load, so any caller awaiting it (request pipeline, load monitor, health
    /// probe) would block far past its own deadline. Callers peek the cell and
    /// report unavailable until the driver lands.
    fn spawn_zmq_connect_driver(&self, cell: &Arc<OnceCell<Arc<BackendClient>>>) {
        if self.zmq_connect_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let cell = Arc::clone(cell);
        let started = Arc::clone(&self.zmq_connect_started);
        let base_url = self.metadata.base_url().to_string();
        let model_id = self.metadata.model_id().to_string();
        let url = self.metadata.spec.url.clone();
        let runtime = self.metadata.spec.runtime_type;
        let handshake_override = self.metadata.spec.zmq_handshake_address.clone();
        let engine_count = self.metadata.zmq_engine_count();
        // Capture the readiness signal and the revision at hand-off. The
        // manager only promotes if this revision still matches, so a
        // same-URL replacement racing the handshake is discarded.
        let signal_tx = self.connect_signal_tx.clone();
        let revision = self.revision();
        #[expect(
            clippy::disallowed_methods,
            reason = "detached one-shot handshake driver; the OnceCell dedupes with the request path and the guard self-clears on failure to allow a retry"
        )]
        // Detached: only the AbortHandle is kept, so the task keeps running
        // until it completes or the worker leaves the registry.
        let handle = tokio::spawn(async move {
            match cell
                .get_or_try_init(|| {
                    connect_zmq_backend(
                        base_url,
                        model_id,
                        runtime,
                        handshake_override,
                        engine_count,
                    )
                })
                .await
            {
                Ok(_) => {
                    // Handshake landed: wake the manager to promote now
                    // rather than on the next poll. A dropped signal (no
                    // manager, or receiver gone) is harmless — polling
                    // still promotes on the success threshold.
                    if let Some(tx) = &signal_tx {
                        let _ = tx.send(WorkerConnected { url, revision });
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "ZMQ backend handshake failed for {url}: {e}; will retry on the next health probe"
                    );
                    started.store(false, Ordering::SeqCst);
                }
            }
        });
        self.zmq_connect_abort
            .store(Some(Arc::new(handle.abort_handle())));
    }

    fn install_shared_state_from_basic(&self, other: &BasicWorker) {
        let shared_runtime = other.shared_runtime();
        shared_runtime.bump_revision();
        let shared_status = shared_runtime.status();
        self.runtime.store(shared_runtime);
        Metrics::set_worker_health(self.url(), shared_status == WorkerStatus::Ready);

        let existing_cb = self.circuit_breaker.load();
        let other_cb = other.circuit_breaker.load_full();
        if other_cb.config() == existing_cb.config() {
            self.circuit_breaker.store(other_cb);
        }

        self.adopt_backend_client_from(other);
    }

    /// Adopt the replaced worker's backend-client cell so a same-URL
    /// replacement (a metadata-only update, say) keeps talking over the
    /// connection the old worker already established. Decisive for ZMQ: SMG
    /// binds the sockets and the engine handshakes only at its own startup, so
    /// a replacement starting from an empty cell would unbind the live sockets,
    /// rebind, and wait for a HELLO that never arrives.
    ///
    /// `zmq_connect_started` is deliberately left cleared rather than copied:
    /// the shared cell already dedupes a handshake in flight, and a cleared
    /// guard lets the replacement retry (and signal under its own revision) if
    /// that handshake fails.
    fn adopt_backend_client_from(&self, other: &BasicWorker) {
        // A transport or runtime change means a different wire protocol, and a
        // replacement that arrived with its own client keeps it.
        if self.metadata.spec.connection_mode != other.metadata.spec.connection_mode
            || self.metadata.spec.runtime_type != other.metadata.spec.runtime_type
            || self.backend_client.load().get().is_some()
        {
            return;
        }
        self.backend_client.store(other.backend_client.load_full());
    }
}

#[async_trait]
impl Worker for BasicWorker {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn url(&self) -> &str {
        &self.metadata.spec.url
    }

    fn api_key(&self) -> Option<&String> {
        self.metadata.spec.api_key.as_ref()
    }

    fn worker_type(&self) -> &WorkerType {
        &self.metadata.spec.worker_type
    }

    fn connection_mode(&self) -> &ConnectionMode {
        &self.metadata.spec.connection_mode
    }

    fn status(&self) -> WorkerStatus {
        self.runtime.load().status()
    }

    fn revision(&self) -> u64 {
        self.runtime.load().revision()
    }

    fn set_status(&self, status: WorkerStatus) {
        self.runtime.load().set_status(status);
        Metrics::set_worker_health(self.url(), status == WorkerStatus::Ready);
    }

    fn inherit_shared_state_from(&self, other: &dyn Worker) -> bool {
        let Some(other) = other.as_any().downcast_ref::<BasicWorker>() else {
            return false;
        };
        self.install_shared_state_from_basic(other);
        true
    }

    async fn check_health_async(&self) -> WorkerResult<()> {
        if self.metadata.health_config.disable_health_check {
            return Ok(());
        }

        let probe_ok = match &self.metadata.spec.connection_mode {
            ConnectionMode::Http => self.http_health_check().await?,
            ConnectionMode::Grpc => self.grpc_health_check().await?,
            ConnectionMode::Zmq => self.zmq_health_check().await?,
        };

        if probe_ok {
            Ok(())
        } else {
            Err(WorkerError::HealthCheckFailed {
                url: self.metadata.spec.url.clone(),
                reason: "health probe returned non-success".to_string(),
            })
        }
    }

    fn consecutive_failures_increment(&self) -> usize {
        self.runtime.load().consecutive_failures_increment()
    }

    fn consecutive_failures_reset(&self) {
        self.runtime.load().consecutive_failures_reset();
    }

    fn consecutive_successes_increment(&self) -> usize {
        self.runtime.load().consecutive_successes_increment()
    }

    fn consecutive_successes_reset(&self) {
        self.runtime.load().consecutive_successes_reset();
    }

    fn total_pending_probes(&self) -> usize {
        self.runtime.load().total_pending_probes()
    }

    fn total_pending_probes_increment(&self) -> usize {
        self.runtime.load().total_pending_probes_increment()
    }

    fn total_pending_probes_reset(&self) {
        self.runtime.load().total_pending_probes_reset();
    }

    fn load(&self) -> usize {
        self.runtime.load().load()
    }

    fn increment_load(&self) {
        self.runtime.load().increment_load();
        self.update_running_requests_metrics();
    }

    fn decrement_load(&self) {
        if !self.runtime.load().try_decrement_load() {
            tracing::warn!(
                worker_url = %self.metadata.spec.url,
                "Attempted to decrement load counter that is already at 0"
            );
        }
        self.update_running_requests_metrics();
    }

    fn routing_key_load(&self) -> usize {
        self.runtime.load().routing_key_load()
    }

    fn routing_key_inflight(&self, routing_key: &str) -> usize {
        self.runtime.load().routing_key_inflight(routing_key)
    }

    fn increment_routing_key_load(&self, routing_key: &str) {
        self.runtime.load().increment_routing_key_load(routing_key);
    }

    fn decrement_routing_key_load(&self, routing_key: &str) {
        self.runtime.load().decrement_routing_key_load(routing_key);
    }

    fn processed_requests(&self) -> usize {
        self.runtime.load().processed_requests()
    }

    fn increment_processed(&self) {
        self.runtime.load().increment_processed();
    }

    fn metadata(&self) -> &WorkerMetadata {
        &self.metadata
    }

    fn circuit_breaker_state(&self) -> super::circuit_breaker::CircuitState {
        self.circuit_breaker.load().state()
    }

    fn circuit_breaker_can_execute(&self) -> bool {
        self.circuit_breaker.load().can_execute()
    }

    fn is_overloaded(&self) -> bool {
        self.runtime.load().is_overloaded()
    }

    fn set_overloaded(&self, overloaded: bool) -> bool {
        self.runtime.load().set_overloaded(overloaded)
    }

    fn is_available(&self) -> bool {
        // Same two guards the pre-veto version took (`is_healthy` +
        // `circuit_breaker_can_execute`): the veto rides the runtime guard.
        let rt = self.runtime.load();
        rt.status() == WorkerStatus::Ready
            && !rt.is_overloaded()
            && self.circuit_breaker.load().can_execute()
    }

    fn is_healthy_and_eligible(&self) -> bool {
        let rt = self.runtime.load();
        rt.status() == WorkerStatus::Ready && !rt.is_overloaded()
    }

    fn routing_state(&self) -> RoutingState {
        // One runtime guard covers status + load + processed + the overload veto
        // (all live in `self.runtime`); the circuit breaker is a separate
        // ArcSwap, so it needs its own guard.
        let rt = self.runtime.load();
        RoutingState {
            healthy: rt.status() == WorkerStatus::Ready,
            can_execute: self.circuit_breaker.load().can_execute(),
            load: rt.load(),
            processed: rt.processed_requests(),
            overloaded: rt.is_overloaded(),
        }
    }

    fn record_circuit_breaker_outcome(&self, success: bool) {
        self.circuit_breaker.load().record_outcome(success);
    }

    fn resilience(&self) -> &ResolvedResilience {
        &self.resilience
    }

    fn http_client(&self) -> &reqwest::Client {
        self.http_client.client()
    }

    fn http_client_handle_if_initialized(&self) -> Option<Arc<reqwest::Client>> {
        self.http_client.handle_if_initialized()
    }

    fn supports_model(&self, model_id: &str) -> bool {
        let overridden = self.models_override.load();
        if !overridden.is_wildcard() {
            return overridden.supports(model_id);
        }
        self.metadata.supports_model(model_id)
    }

    fn models(&self) -> Vec<ModelCard> {
        let overridden = self.models_override.load();
        let source = if overridden.is_wildcard() {
            self.metadata.spec.models.all()
        } else {
            overridden.all()
        };
        source.to_vec()
    }

    fn set_models(&self, models: Vec<ModelCard>) {
        tracing::debug!(
            "Setting {} models for worker {} via lazy discovery",
            models.len(),
            self.metadata.spec.url
        );
        self.models_override
            .store(Arc::new(WorkerModels::from(models)));
    }

    fn has_models_discovered(&self) -> bool {
        !self.models_override.load().is_wildcard() || !self.metadata.spec.models.is_wildcard()
    }

    async fn get_backend_client(&self) -> WorkerResult<Option<Arc<BackendClient>>> {
        match self.metadata.spec.connection_mode {
            ConnectionMode::Http => Ok(None),
            ConnectionMode::Grpc => {
                // OnceCell provides lock-free reads after initialization.
                // get_or_try_init only acquires internal lock on first call.
                let cell = self.backend_client.load_full();
                let client = cell
                    .get_or_try_init(|| async {
                        let runtime_str = self.metadata.spec.runtime_type.to_string();
                        tracing::info!(
                            "Lazily initializing gRPC client ({}) for worker: {}",
                            runtime_str,
                            self.metadata.spec.url
                        );
                        // DP-expanded workers carry a `{base}@{rank}` URL; connect to the base
                        match GrpcClient::connect(self.metadata.base_url(), &runtime_str).await {
                            Ok(client) => {
                                tracing::info!(
                                    "Successfully connected gRPC client ({}) for worker: {}",
                                    runtime_str,
                                    self.metadata.spec.url
                                );
                                Ok(Arc::new(BackendClient::Grpc(client)))
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Failed to connect gRPC client for worker {}: {}",
                                    self.metadata.spec.url,
                                    e
                                );
                                Err(WorkerError::ConnectionFailed {
                                    url: self.metadata.spec.url.clone(),
                                    reason: format!("Failed to connect to gRPC server: {e}"),
                                })
                            }
                        }
                    })
                    .await?;
                Ok(Some(Arc::clone(client)))
            }
            ConnectionMode::Zmq => {
                // SMG binds the handshake + data-plane sockets; the
                // operator-launched engine dials them. Peek only: the handshake
                // is owned by the background driver (it can outlast any
                // caller's deadline — see spawn_zmq_connect_driver), so fail
                // fast instead of blocking a request or a load poll behind it.
                // Kicking the driver off here also covers workers whose health
                // checks are disabled, where no probe ever runs.
                let cell = self.backend_client.load_full();
                if let Some(client) = cell.get() {
                    return Ok(Some(Arc::clone(client)));
                }
                self.spawn_zmq_connect_driver(&cell);
                Err(WorkerError::ConnectionFailed {
                    url: self.metadata.spec.url.clone(),
                    reason: "ZMQ backend handshake has not completed yet".to_string(),
                })
            }
        }
    }

    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        // Intentional no-op: a failed gRPC worker is removed and re-added rather
        // than reconnected in place. (The ZMQ path differs: its health probe
        // evicts a dead client and reconnects — see zmq_health_check.)
        tracing::debug!(
            "reset_grpc_client called for {} (no-op for gRPC workers)",
            self.metadata.spec.url
        );
        Ok(())
    }

    async fn grpc_health_check(&self) -> WorkerResult<bool> {
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);
        let maybe = self.get_backend_client().await?;
        let Some(backend_client) = maybe else {
            tracing::error!(
                "Worker {} is not a gRPC worker but connection mode is gRPC",
                self.metadata.spec.url
            );
            return Ok(false);
        };

        match time::timeout(timeout, backend_client.health_check()).await {
            Ok(Ok(resp)) => {
                tracing::debug!(
                    "gRPC health OK for {}: healthy={}",
                    self.metadata.spec.url,
                    resp.healthy
                );
                Ok(resp.healthy)
            }
            Ok(Err(err)) => {
                tracing::warn!(
                    "gRPC health RPC error for {}: {err:?}",
                    self.metadata.spec.url
                );
                Ok(false)
            }
            Err(_) => {
                tracing::warn!("gRPC health timed out for {}", self.metadata.spec.url);
                Ok(false)
            }
        }
    }

    async fn zmq_health_check(&self) -> WorkerResult<bool> {
        // Never drive (or cancel) the handshake from the probe: model load can
        // take far longer than the health timeout, and cancelling it mid-flight
        // would rebind the sockets on every probe and never connect. Peek the
        // cached client instead — if it isn't there yet, kick the handshake off
        // once in the background and report not-ready until it lands. Thereafter
        // liveness is local (the connector marks the client closed on
        // ENGINE_CORE_DEAD or a transport failure); there is no health RPC on
        // the raw ZMQ wire.
        let cell = self.backend_client.load_full();
        if let Some(backend_client) = cell.get() {
            if backend_client.is_alive() {
                return Ok(true);
            }
            // The engine behind this client died, and a ZMQ client cannot
            // reconnect in place (liveness is latched). Evict the dead client
            // and reset the handshake guard so the next probe re-runs
            // connect_zmq_backend, rebinding the sockets for a replacement
            // engine to dial into. Eviction cannot race an in-flight connect:
            // a cell only becomes non-empty once its connect finished, and
            // compare_and_swap keeps a lagging probe from clobbering a fresh
            // cell another probe already swapped in.
            tracing::warn!(
                "ZMQ engine for {} is no longer alive; evicting the dead client and rebinding on a later probe",
                self.metadata.spec.url
            );
            self.backend_client
                .compare_and_swap(&cell, Arc::new(OnceCell::new()));
            self.zmq_connect_started.store(false, Ordering::SeqCst);
            return Ok(false);
        }
        self.spawn_zmq_connect_driver(&cell);
        Ok(false)
    }

    fn abort_background_tasks(&self) {
        if let Some(handle) = self.zmq_connect_abort.swap(None) {
            handle.abort();
            // The bound sockets are released with the cancelled future; reset
            // the guard so any instance still sharing this state can retry.
            self.zmq_connect_started.store(false, Ordering::SeqCst);
        }
    }

    async fn http_health_check(&self) -> WorkerResult<bool> {
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);

        let health_url = format!("{}{}", self.base_url(), self.metadata.health_endpoint);

        let mut req = self.http_client.client().get(&health_url).timeout(timeout);
        if let Some(api_key) = &self.metadata.spec.api_key {
            req = req.bearer_auth(api_key);
        }

        match req.send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    Ok(true)
                } else {
                    tracing::warn!(
                        "HTTP health check returned non-success status for {}: {}",
                        health_url,
                        status
                    );
                    Ok(false)
                }
            }
            Err(err) => {
                tracing::warn!("HTTP health check failed for {}: {err:?}", health_url);
                Ok(false)
            }
        }
    }
}

/// RAII guard for worker load management
///
/// Automatically decrements worker load when dropped. Can be attached to
/// an axum Response to tie the guard's lifetime to the response body,
/// which is essential for streaming responses where the function returns
/// immediately but the stream continues in the background.
pub struct WorkerLoadGuard {
    worker: Arc<dyn Worker>,
    routing_key: Option<String>,
}

impl WorkerLoadGuard {
    pub fn new(worker: Arc<dyn Worker>, headers: Option<&http::HeaderMap>) -> Self {
        let key = extract_routing_key(headers).map(String::from);
        Self::with_key(worker, key.as_deref())
    }

    /// Guard keyed by the caller-resolved effective sticky key (rid-derived
    /// wins over the header), so keyed-load accounting matches selection.
    pub fn with_key(worker: Arc<dyn Worker>, routing_key: Option<&str>) -> Self {
        worker.increment_load();

        let routing_key = routing_key.map(String::from);
        if let Some(ref key) = routing_key {
            worker.increment_routing_key_load(key);
        }

        Self {
            worker,
            routing_key,
        }
    }
}

impl Drop for WorkerLoadGuard {
    fn drop(&mut self) {
        self.worker.decrement_load();
        if let Some(ref key) = self.routing_key {
            self.worker.decrement_routing_key_load(key);
        }
    }
}

/// Body wrapper that holds an attached value.
///
/// When this body is dropped (stream ends or client disconnects),
/// the attached value is dropped automatically. This is useful for RAII guards
/// like WorkerLoadGuard that need to be tied to a response body's lifetime.
pub struct AttachedBody<T> {
    inner: Body,
    _attached: T,
}

impl<T> AttachedBody<T> {
    pub fn new(inner: Body, attached: T) -> Self {
        Self {
            inner,
            _attached: attached,
        }
    }
}

impl<T: Send + Unpin + 'static> AttachedBody<T> {
    pub fn wrap_response(
        response: axum::response::Response,
        attached: T,
    ) -> axum::response::Response {
        let (parts, body) = response.into_parts();
        axum::response::Response::from_parts(parts, Body::new(Self::new(body, attached)))
    }
}

impl<T: Send + Unpin + 'static> http_body::Body for AttachedBody<T> {
    type Data = bytes::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        std::pin::Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// Helper to convert Worker trait object to WorkerInfo struct.
///
/// Both `is_healthy` and `status` are derived from the same atomic snapshot
/// to avoid TOCTOU between the two fields. The `status` field exposes the
/// real lifecycle state (Pending/Ready/NotReady/Failed) so API consumers
/// can distinguish "starting up" from "broken" — `is_healthy` collapses
/// everything to a routability bool for backwards compatibility.
pub fn worker_to_info(worker: &Arc<dyn Worker>) -> WorkerInfo {
    let metadata = worker.metadata();
    let spec = metadata.spec.clone();
    let status = worker.status();

    WorkerInfo {
        id: worker.url().to_string(),
        model_id: spec.models.primary().map(|m| m.id.clone()),
        spec,
        is_healthy: status == WorkerStatus::Ready,
        status: Some(status),
        load: worker.load(),
        job_status: None,
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::worker::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        BasicWorkerBuilder,
    };

    /// Health config that skips health checks — workers start Ready immediately.
    /// Use in tests that don't test the health check lifecycle.
    fn no_health_check() -> HealthCheckConfig {
        HealthCheckConfig {
            disable_health_check: true,
            ..HealthCheckConfig::default()
        }
    }

    #[test]
    fn test_worker_type_display() {
        assert_eq!(WorkerType::Regular.to_string(), "regular");
        assert_eq!(WorkerType::Prefill.to_string(), "prefill");
        assert_eq!(WorkerType::Decode.to_string(), "decode");
    }

    #[test]
    fn test_worker_type_equality() {
        assert_eq!(WorkerType::Regular, WorkerType::Regular);
        assert_ne!(WorkerType::Regular, WorkerType::Decode);
        assert_eq!(WorkerType::Prefill, WorkerType::Prefill);
    }

    #[test]
    fn test_worker_type_clone() {
        let original = WorkerType::Prefill;
        let cloned = original;
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_health_config_default() {
        use openai_protocol::worker::HealthCheckConfig;
        let config = HealthCheckConfig::default();
        assert_eq!(config.timeout_secs, 30);
        assert_eq!(config.check_interval_secs, 60);
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.success_threshold, 2);
        assert!(!config.disable_health_check);
    }

    #[test]
    fn test_health_config_custom() {
        use openai_protocol::worker::HealthCheckConfig;
        let config = HealthCheckConfig {
            timeout_secs: 10,
            check_interval_secs: 60,
            failure_threshold: 5,
            success_threshold: 3,
            disable_health_check: true,
            drain_settle_secs: 5,
        };
        assert_eq!(config.timeout_secs, 10);
        assert_eq!(config.check_interval_secs, 60);
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.success_threshold, 3);
        assert!(config.disable_health_check);
    }

    #[test]
    fn test_basic_worker_creation() {
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();
        assert_eq!(worker.url(), "http://test:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
        assert!(worker.is_healthy());
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.processed_requests(), 0);
    }

    #[test]
    fn test_worker_with_labels() {
        let mut labels = std::collections::HashMap::new();
        labels.insert("env".to_string(), "prod".to_string());
        labels.insert("zone".to_string(), "us-west".to_string());

        use crate::worker::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .labels(labels.clone())
            .build();

        assert_eq!(worker.metadata().spec.labels, labels);
    }

    #[test]
    fn test_max_running_requests_parses_from_label() {
        use crate::worker::BasicWorkerBuilder;
        let mut labels = std::collections::HashMap::new();
        labels.insert("max_running_requests".to_string(), "256".to_string());
        let worker = BasicWorkerBuilder::new("http://w:9000")
            .labels(labels)
            .build();
        assert_eq!(worker.max_running_requests(), Some(256));
    }

    #[test]
    fn test_max_running_requests_returns_none_when_label_missing() {
        use crate::worker::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://w:9000").build();
        assert_eq!(worker.max_running_requests(), None);
    }

    #[test]
    fn test_max_running_requests_returns_none_on_bad_value() {
        use crate::worker::BasicWorkerBuilder;
        let mut labels = std::collections::HashMap::new();
        labels.insert(
            "max_running_requests".to_string(),
            "not-a-number".to_string(),
        );
        let worker = BasicWorkerBuilder::new("http://w:9000")
            .labels(labels)
            .build();
        assert_eq!(worker.max_running_requests(), None);
    }

    #[test]
    fn test_max_running_requests_zero_treated_as_none() {
        use crate::worker::BasicWorkerBuilder;
        let mut labels = std::collections::HashMap::new();
        labels.insert("max_running_requests".to_string(), "0".to_string());
        let worker = BasicWorkerBuilder::new("http://w:9000")
            .labels(labels)
            .build();
        // Zero is meaningless for capacity; treat as "not reported".
        assert_eq!(worker.max_running_requests(), None);
    }

    #[test]
    fn test_is_realtime_capable_true_from_label() {
        use crate::worker::BasicWorkerBuilder;
        let mut labels = std::collections::HashMap::new();
        labels.insert("realtime".to_string(), "true".to_string());
        let worker = BasicWorkerBuilder::new("http://w:9000")
            .labels(labels)
            .build();
        assert!(worker.is_realtime_capable());
    }

    #[test]
    fn test_is_realtime_capable_false_when_label_missing() {
        use crate::worker::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://w:9000").build();
        assert!(!worker.is_realtime_capable());
    }

    #[test]
    fn test_is_realtime_capable_false_on_non_true_value() {
        use crate::worker::BasicWorkerBuilder;
        let mut labels = std::collections::HashMap::new();
        labels.insert("realtime".to_string(), "1".to_string());
        let worker = BasicWorkerBuilder::new("http://w:9000")
            .labels(labels)
            .build();
        // Only the exact string "true" enables realtime.
        assert!(!worker.is_realtime_capable());
    }

    #[test]
    fn test_worker_with_health_config() {
        use openai_protocol::worker::HealthCheckConfig;
        let custom_config = HealthCheckConfig {
            timeout_secs: 15,
            check_interval_secs: 45,
            failure_threshold: 4,
            success_threshold: 2,
            disable_health_check: false,
            drain_settle_secs: 5,
        };

        use crate::worker::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .health_config(custom_config.clone())
            .health_endpoint("/custom-health")
            .build();

        assert_eq!(worker.metadata().health_config.timeout_secs, 15);
        assert_eq!(worker.metadata().health_config.check_interval_secs, 45);
        assert_eq!(worker.metadata().health_endpoint, "/custom-health");
    }

    #[test]
    fn test_worker_url() {
        use crate::worker::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(worker.url(), "http://worker1:8080");
    }

    #[test]
    fn test_worker_type_getter() {
        use crate::worker::BasicWorkerBuilder;
        let regular = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(regular.worker_type(), &WorkerType::Regular);

        let prefill = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Prefill)
            .bootstrap_port(Some(9090))
            .build();
        assert_eq!(prefill.worker_type(), &WorkerType::Prefill);
        assert_eq!(prefill.bootstrap_port(), Some(9090));

        let decode = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Decode)
            .build();
        assert_eq!(decode.worker_type(), &WorkerType::Decode);
    }

    #[test]
    fn test_health_status() {
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();

        assert!(worker.is_healthy());
        assert_eq!(worker.status(), WorkerStatus::Ready);

        worker.set_status(WorkerStatus::NotReady);
        assert!(!worker.is_healthy());
        assert_eq!(worker.status(), WorkerStatus::NotReady);

        worker.set_status(WorkerStatus::Ready);
        assert!(worker.is_healthy());
        assert_eq!(worker.status(), WorkerStatus::Ready);
    }

    #[test]
    fn test_pending_worker_not_routable() {
        // Default health config: health checks enabled → starts Pending
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(worker.status(), WorkerStatus::Pending);
        assert!(!worker.is_healthy()); // Pending is not routable
        assert!(!worker.is_available()); // Pending is not available
    }

    #[test]
    fn test_load_counter_operations() {
        use crate::worker::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(worker.load(), 0);

        worker.increment_load();
        assert_eq!(worker.load(), 1);

        worker.increment_load();
        worker.increment_load();
        assert_eq!(worker.load(), 3);

        worker.decrement_load();
        assert_eq!(worker.load(), 2);

        worker.decrement_load();
        worker.decrement_load();
        assert_eq!(worker.load(), 0);

        worker.decrement_load();
        assert_eq!(worker.load(), 0);
    }

    #[test]
    fn test_processed_counter() {
        use crate::worker::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(worker.processed_requests(), 0);

        for i in 1..=100 {
            worker.increment_processed();
            assert_eq!(worker.processed_requests(), i);
        }
    }

    #[tokio::test]
    async fn test_concurrent_load_increments() {
        use crate::worker::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut handles = vec![];

        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            #[expect(
                clippy::disallowed_methods,
                reason = "Test helper: short-lived tasks joined before test ends"
            )]
            let handle = tokio::spawn(async move {
                worker_clone.increment_load();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(worker.load(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_load_decrements() {
        use crate::worker::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        for _ in 0..100 {
            worker.increment_load();
        }
        assert_eq!(worker.load(), 100);

        let mut handles = vec![];

        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            #[expect(
                clippy::disallowed_methods,
                reason = "Test helper: short-lived tasks joined before test ends"
            )]
            let handle = tokio::spawn(async move {
                worker_clone.decrement_load();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_health_updates() {
        use crate::worker::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut handles = vec![];

        for i in 0..100 {
            let worker_clone = Arc::clone(&worker);
            #[expect(
                clippy::disallowed_methods,
                reason = "Test helper: short-lived tasks joined before test ends"
            )]
            let handle = tokio::spawn(async move {
                let status = if i % 2 == 0 {
                    WorkerStatus::Ready
                } else {
                    WorkerStatus::NotReady
                };
                worker_clone.set_status(status);
                time::sleep(Duration::from_micros(10)).await;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }
    }

    #[test]
    fn test_create_regular_worker() {
        use crate::worker::BasicWorkerBuilder;
        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        assert_eq!(worker.url(), "http://regular:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
    }

    #[test]
    fn test_create_prefill_worker() {
        use crate::worker::BasicWorkerBuilder;
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill)
                .bootstrap_port(Some(9090))
                .build(),
        );
        assert_eq!(worker1.url(), "http://prefill:8080");
        assert_eq!(worker1.worker_type(), &WorkerType::Prefill);
        assert_eq!(worker1.bootstrap_port(), Some(9090));

        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        assert_eq!(worker2.worker_type(), &WorkerType::Prefill);
        assert_eq!(worker2.bootstrap_port(), None);
    }

    #[test]
    fn test_create_decode_worker() {
        use crate::worker::BasicWorkerBuilder;
        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://decode:8080")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        assert_eq!(worker.url(), "http://decode:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Decode);
    }

    #[tokio::test]
    async fn test_check_health_async() {
        use crate::worker::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        // Health check should fail since there's no actual server
        let result = worker.check_health_async().await;
        assert!(result.is_err());
    }

    #[test]
    #[expect(clippy::print_stderr)]
    fn test_load_counter_performance() {
        use std::time::Instant;

        use crate::worker::BasicWorkerBuilder;

        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        let iterations = 1_000_000;

        let start = Instant::now();
        for _ in 0..iterations {
            worker.increment_load();
        }
        let duration = start.elapsed();

        let ops_per_sec = iterations as f64 / duration.as_secs_f64();
        eprintln!("Load counter operations per second: {ops_per_sec:.0}");

        // Lower bound is intentionally generous so this microbench does
        // not flake on CI runners under contention. A relaxed Acquire/
        // Release atomic increment should comfortably exceed this on any
        // reasonable hardware — observed CI floor is around 1M ops/sec,
        // so 500k gives a 2x safety margin.
        assert!(ops_per_sec > 500_000.0);
    }

    #[test]
    fn test_dp_aware_worker_creation() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(2, 4)
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@2");
        assert_eq!(dp_worker.base_url(), "http://worker1:8080");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.dp_rank(), Some(2));
        assert_eq!(dp_worker.dp_size(), Some(4));
        assert_eq!(dp_worker.worker_type(), &WorkerType::Regular);
    }

    #[test]
    fn test_dp_aware_worker_creation_prefill() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(1, 2)
            .worker_type(WorkerType::Prefill)
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@1");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.worker_type(), &WorkerType::Prefill);
    }

    #[test]
    fn test_dp_aware_worker_creation_decode() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(0, 4)
            .worker_type(WorkerType::Decode)
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@0");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.worker_type(), &WorkerType::Decode);
    }

    #[test]
    fn test_dp_aware_prepare_request() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(3, 8)
            .worker_type(WorkerType::Regular)
            .build();

        let original_req = serde_json::json!({
            "prompt": "Hello",
            "max_tokens": 100
        });

        let prepared_req = dp_worker.prepare_request(original_req).unwrap();

        assert_eq!(prepared_req["prompt"], "Hello");
        assert_eq!(prepared_req["max_tokens"], 100);
        assert_eq!(prepared_req["data_parallel_rank"], 3);
    }

    #[test]
    fn test_dp_aware_prepare_request_invalid() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(0, 4)
            .worker_type(WorkerType::Regular)
            .build();

        // Non-object JSON should fail
        let invalid_req = serde_json::json!("not an object");
        let result = dp_worker.prepare_request(invalid_req);

        assert!(result.is_err());
        match result.unwrap_err() {
            WorkerError::InvalidConfiguration { message } => {
                assert!(message.contains("JSON object"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_dp_aware_endpoint_url() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(1, 4)
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(
            dp_worker.endpoint_url("/generate"),
            "http://worker1:8080/generate"
        );
        assert_eq!(
            dp_worker.endpoint_url("/health"),
            "http://worker1:8080/health"
        );
    }

    #[test]
    fn test_dp_aware_worker_delegated_methods() {
        let dp_worker = BasicWorkerBuilder::new("http://worker1:8080")
            .dp_config(0, 2)
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();

        assert!(dp_worker.is_healthy());
        dp_worker.set_status(WorkerStatus::NotReady);
        assert!(!dp_worker.is_healthy());

        assert_eq!(dp_worker.load(), 0);
        dp_worker.increment_load();
        assert_eq!(dp_worker.load(), 1);
        dp_worker.decrement_load();
        assert_eq!(dp_worker.load(), 0);

        assert_eq!(dp_worker.processed_requests(), 0);
        dp_worker.increment_processed();
        assert_eq!(dp_worker.processed_requests(), 1);
    }

    #[test]
    fn test_worker_circuit_breaker() {
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();

        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);

        worker.record_outcome(503);
        worker.record_outcome(503);

        assert!(worker.is_available());

        worker.record_outcome(503);
        worker.record_outcome(503);
        worker.record_outcome(503);

        assert!(!worker.is_available());
        assert!(worker.is_healthy());
        assert!(!worker.circuit_breaker_can_execute());
    }

    #[test]
    fn test_capacity_pushback_never_trips_circuit_breaker() {
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();

        // A storm of 429 capacity pushback must not open the breaker...
        for _ in 0..20 {
            worker.record_outcome(429);
        }
        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);

        // ...while genuine failures still do.
        for _ in 0..5 {
            worker.record_outcome(500);
        }
        assert!(!worker.is_available());
    }

    #[test]
    fn test_capacity_pushback_does_not_close_half_open_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration: Duration::from_millis(50),
            window_duration: Duration::from_secs(60),
        };
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .circuit_breaker_config(config)
            .health_config(no_health_check())
            .build();

        worker.record_outcome(500);
        worker.record_outcome(500);
        assert!(!worker.is_available());
        thread::sleep(Duration::from_millis(80));
        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);

        // 429 records no sample: the breaker must stay half-open, not close.
        worker.record_outcome(429);
        assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);

        worker.record_outcome(200);
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    }

    #[test]
    fn test_worker_with_circuit_breaker_config() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration: Duration::from_millis(100),
            window_duration: Duration::from_secs(60),
        };

        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .circuit_breaker_config(config)
            .health_config(no_health_check())
            .build();

        worker.record_outcome(503);
        assert!(worker.is_available());
        worker.record_outcome(503);
        assert!(!worker.is_available());

        thread::sleep(Duration::from_millis(150));

        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker_state(), CircuitState::HalfOpen);

        worker.record_outcome(200);
        assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
    }

    #[test]
    fn test_dp_aware_worker_circuit_breaker() {
        let dp_worker = BasicWorkerBuilder::new("http://worker:8080")
            .dp_config(0, 2)
            .worker_type(WorkerType::Regular)
            .health_config(no_health_check())
            .build();

        assert!(dp_worker.is_available());

        for _ in 0..5 {
            dp_worker.record_outcome(503);
        }

        assert!(!dp_worker.is_available());
        assert_eq!(dp_worker.circuit_breaker_state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_mixed_worker_types() {
        let hc = no_health_check();
        let regular: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8080")
                .worker_type(WorkerType::Regular)
                .health_config(hc.clone())
                .build(),
        );
        let prefill: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill)
                .bootstrap_port(Some(9090))
                .health_config(hc.clone())
                .build(),
        );
        let decode: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://decode:8080")
                .worker_type(WorkerType::Decode)
                .health_config(hc.clone())
                .build(),
        );
        let dp_aware_regular: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://dp:8080")
                .dp_config(0, 2)
                .worker_type(WorkerType::Regular)
                .health_config(hc.clone())
                .api_key("test_api_key")
                .build(),
        );
        let dp_aware_prefill: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://dp-prefill:8080")
                .dp_config(1, 2)
                .worker_type(WorkerType::Prefill)
                .health_config(hc.clone())
                .api_key("test_api_key")
                .build(),
        );
        let dp_aware_decode: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://dp-decode:8080")
                .dp_config(0, 4)
                .worker_type(WorkerType::Decode)
                .health_config(hc.clone())
                .api_key("test_api_key")
                .build(),
        );

        let workers: Vec<Box<dyn Worker>> = vec![
            regular,
            prefill,
            decode,
            dp_aware_regular,
            dp_aware_prefill,
            dp_aware_decode,
        ];

        for worker in &workers {
            assert!(worker.is_healthy());
            assert_eq!(worker.load(), 0);
            assert_eq!(worker.processed_requests(), 0);
        }

        assert!(!workers[0].is_dp_aware());
        assert!(!workers[1].is_dp_aware());
        assert!(!workers[2].is_dp_aware());
        assert!(workers[3].is_dp_aware());
        assert!(workers[4].is_dp_aware());
        assert!(workers[5].is_dp_aware());

        assert_eq!(workers[0].worker_type(), &WorkerType::Regular);
        assert_eq!(workers[1].worker_type(), &WorkerType::Prefill);
        assert_eq!(workers[2].worker_type(), &WorkerType::Decode);
        assert_eq!(workers[3].worker_type(), &WorkerType::Regular);
        assert_eq!(workers[4].worker_type(), &WorkerType::Prefill);
        assert_eq!(workers[5].worker_type(), &WorkerType::Decode);
    }

    // === Phase 1.3: WorkerMetadata model methods tests ===

    #[test]
    fn test_worker_metadata_empty_models_accepts_all() {
        let metadata = WorkerMetadata {
            spec: Arc::new(WorkerSpec::new("http://test:8080")),
            health_config: HealthCheckConfig::default(),
            health_endpoint: "/health".to_string(),
            overload: OverloadThresholds::default(),
        };

        // Empty models list should accept any model
        assert!(metadata.supports_model("any-model"));
        assert!(metadata.supports_model("gpt-4"));
        assert!(metadata.supports_model("llama-3.1"));
    }

    #[test]
    fn test_worker_metadata_find_model() {
        use super::ModelCard;

        let model1 = ModelCard::new("meta-llama/Llama-3.1-8B")
            .with_alias("llama-3.1-8b")
            .with_alias("llama3.1");
        let model2 = ModelCard::new("gpt-4o");

        let mut spec = WorkerSpec::new("http://test:8080");
        spec.models = WorkerModels::from(vec![model1, model2]);
        let metadata = WorkerMetadata {
            spec: Arc::new(spec),
            health_config: HealthCheckConfig::default(),
            health_endpoint: "/health".to_string(),
            overload: OverloadThresholds::default(),
        };

        // Find by primary ID
        assert!(metadata.find_model("meta-llama/Llama-3.1-8B").is_some());
        assert!(metadata.find_model("gpt-4o").is_some());

        // Find by alias
        assert!(metadata.find_model("llama-3.1-8b").is_some());
        assert!(metadata.find_model("llama3.1").is_some());

        // Not found
        assert!(metadata.find_model("unknown-model").is_none());
    }

    #[test]
    fn test_worker_routing_key_load_increment_decrement() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        assert_eq!(load.value(), 0);

        load.increment("key1");
        assert_eq!(load.value(), 1);

        load.increment("key2");
        assert_eq!(load.value(), 2);

        load.increment("key1");
        assert_eq!(load.value(), 2);

        load.decrement("key1");
        assert_eq!(load.value(), 2);

        load.decrement("key1");
        assert_eq!(load.value(), 1);

        load.decrement("key2");
        assert_eq!(load.value(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_cleanup_on_zero() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");

        load.increment("key1");
        load.increment("key2");
        load.increment("key3");
        assert_eq!(load.active_routing_keys.len(), 3);

        load.decrement("key1");
        assert_eq!(load.active_routing_keys.len(), 2);

        load.decrement("key2");
        assert_eq!(load.active_routing_keys.len(), 1);

        load.decrement("key3");
        assert_eq!(load.active_routing_keys.len(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_multiple_requests_same_key() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");

        load.increment("key-1");
        load.increment("key-1");
        load.increment("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 1);

        load.decrement("key-1");
        assert_eq!(load.value(), 0);
        assert_eq!(load.active_routing_keys.len(), 0);
    }

    #[test]
    fn test_worker_routing_key_load_decrement_nonexistent() {
        let load = WorkerRoutingKeyLoad::new("http://test:8000");
        load.decrement("nonexistent");
        assert_eq!(load.value(), 0);
    }

    #[test]
    fn test_worker_load_guard_with_routing_key() {
        use crate::worker::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", "key-123".parse().unwrap());

        {
            let _guard = WorkerLoadGuard::new(worker.clone(), Some(&headers));
            assert_eq!(worker.load(), 1);
            assert_eq!(worker.routing_key_load(), 1);
        }

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);
    }

    #[test]
    fn test_worker_load_guard_without_routing_key() {
        use crate::worker::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);

        {
            let _guard = WorkerLoadGuard::new(worker.clone(), None);
            assert_eq!(worker.load(), 1);
            assert_eq!(worker.routing_key_load(), 0);
        }

        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);
    }

    #[test]
    fn test_worker_load_guard_multiple_same_routing_key() {
        use crate::worker::BasicWorkerBuilder;

        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://test:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut headers = http::HeaderMap::new();
        headers.insert("x-smg-routing-key", "key-123".parse().unwrap());

        let guard1 = WorkerLoadGuard::new(worker.clone(), Some(&headers));
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.routing_key_load(), 1);

        let guard2 = WorkerLoadGuard::new(worker.clone(), Some(&headers));
        assert_eq!(worker.load(), 2);
        assert_eq!(worker.routing_key_load(), 1);

        drop(guard1);
        assert_eq!(worker.load(), 1);
        assert_eq!(worker.routing_key_load(), 1);

        drop(guard2);
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.routing_key_load(), 0);
    }

    #[test]
    fn test_lazy_discovered_models_override_wildcard() {
        let worker = BasicWorkerBuilder::new("http://test:8080").build();

        // Wildcard worker starts with no models listed, but accepts any model
        assert!(worker.models().is_empty());
        assert!(!worker.has_models_discovered());
        assert!(worker.supports_model("gpt-4o-mini")); // wildcard accepts anything

        // Simulate lazy discovery via set_models
        let discovered = vec![
            ModelCard::new("gpt-4o-mini"),
            ModelCard::new("text-embedding-3-small"),
        ];
        worker.set_models(discovered);

        let ids: Vec<String> = worker.models().into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["gpt-4o-mini", "text-embedding-3-small"]);
        assert!(worker.supports_model("gpt-4o-mini"));
        assert!(worker.supports_model("text-embedding-3-small"));
        assert!(!worker.supports_model("non-existent-model"));
        assert!(worker.has_models_discovered());
    }

    #[test]
    fn replacement_adopts_the_backend_client_only_on_a_matching_transport() {
        // The cell object itself is the connection: sharing it is what keeps a
        // replaced ZMQ worker talking to the engine that already handshook.
        let build = |mode: ConnectionMode| {
            BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
                .connection_mode(mode)
                .health_config(no_health_check())
                .build()
        };

        let old = build(ConnectionMode::Zmq);
        let new = build(ConnectionMode::Zmq);
        assert!(new.inherit_shared_state_from(&old));
        assert!(Arc::ptr_eq(
            &new.backend_client.load_full(),
            &old.backend_client.load_full()
        ));

        // A transport change means a different wire protocol — no adoption.
        let retyped = build(ConnectionMode::Http);
        assert!(retyped.inherit_shared_state_from(&old));
        assert!(!Arc::ptr_eq(
            &retyped.backend_client.load_full(),
            &old.backend_client.load_full()
        ));
    }

    /// A ZMQ client whose engine dies must be evicted by the health probe (the
    /// connection can't reconnect in place — liveness is latched), and the
    /// handshake guard reset so a later probe rebinds the sockets for a
    /// replacement engine. A live client stays cached across probes.
    #[tokio::test]
    async fn zmq_health_check_evicts_a_dead_client_and_resets_the_guard() {
        use engine_zmq_client::{
            mock_engine::{connect_to_frontend, default_ready_response},
            EngineId, ENGINE_CORE_DEAD_SENTINEL,
        };

        use crate::routers::grpc::zmq_client::{EosTokenIds, ZmqEngineClient};

        let base = tempfile::tempdir().unwrap();
        let ep = |name: &str| format!("ipc://{}", base.path().join(name).display());
        let (handshake, input, output) = (ep("hs.sock"), ep("in.sock"), ep("out.sock"));

        let (client, engine) = tokio::join!(
            ZmqEngineClient::connect(
                &handshake,
                &input,
                &output,
                1,
                "m".to_string(),
                EosTokenIds::default(),
                RuntimeType::Vllm,
                Duration::from_secs(10)
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = client.expect("adapter connect");
        let mut engine = engine.expect("mock engine");

        let worker = BasicWorkerBuilder::new(ep("ts0.ipc"))
            .connection_mode(ConnectionMode::Zmq)
            .health_config(no_health_check())
            .build();

        // Inject a completed connect and mark the one-shot guard as set.
        let cell = worker.backend_client.load_full();
        cell.set(Arc::new(BackendClient::Zmq(client)))
            .ok()
            .expect("cell empty");
        worker.zmq_connect_started.store(true, Ordering::SeqCst);

        // A live client stays cached and reports healthy.
        assert!(worker.zmq_health_check().await.unwrap());
        assert!(worker.backend_client.load().get().is_some());

        // Kill the engine and wait for the dispatcher to latch the client closed.
        engine
            .send_output(vec![bytes::Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)])
            .await
            .unwrap();
        let observed_dead = time::timeout(Duration::from_secs(5), async {
            loop {
                let alive = worker
                    .backend_client
                    .load()
                    .get()
                    .map(|c| c.is_alive())
                    .unwrap_or(false);
                if !alive {
                    break;
                }
                time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            observed_dead.is_ok(),
            "client never observed ENGINE_CORE_DEAD"
        );

        // The next probe evicts the dead client and resets the guard so a later
        // probe re-runs the handshake for a replacement engine.
        assert!(!worker.zmq_health_check().await.unwrap());
        assert!(
            worker.backend_client.load().get().is_none(),
            "dead client must be evicted from the cell"
        );
        assert!(
            !worker.zmq_connect_started.load(Ordering::SeqCst),
            "handshake guard must reset to allow a reconnect"
        );
    }

    /// Build a ZMQ worker whose sockets no engine will ever dial, plus the
    /// data-plane socket path its handshake driver binds.
    fn unattended_zmq_worker(dir: &std::path::Path) -> (BasicWorker, std::path::PathBuf) {
        let worker = BasicWorkerBuilder::new(format!("ipc://{}", dir.join("ts0.ipc").display()))
            .connection_mode(ConnectionMode::Zmq)
            .health_config(no_health_check())
            .build();
        (worker, dir.join("ts0.ipc-in.sock"))
    }

    /// Poll `cond` until it holds, up to five seconds.
    async fn wait_for(cond: impl Fn() -> bool) -> bool {
        time::timeout(Duration::from_secs(5), async {
            while !cond() {
                time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .is_ok()
    }

    /// The ZMQ handshake can run as long as a model load, so acquisition on a
    /// request or load-monitor path must hand it to the background driver and
    /// fail fast instead of awaiting it inline.
    #[tokio::test]
    async fn zmq_get_backend_client_hands_the_handshake_to_the_background_driver() {
        let base = tempfile::tempdir().unwrap();
        let (worker, input_socket) = unattended_zmq_worker(base.path());

        let acquired = time::timeout(Duration::from_secs(5), worker.get_backend_client())
            .await
            .expect("acquisition must not await the handshake");
        assert!(
            matches!(acquired, Err(WorkerError::ConnectionFailed { .. })),
            "acquisition must report the backend as not connected yet"
        );
        assert!(
            worker.backend_client.load().get().is_none(),
            "no client can be cached before the handshake lands"
        );

        let handle = worker
            .zmq_connect_abort
            .load_full()
            .expect("acquisition must start the background handshake driver");
        assert!(
            wait_for(|| input_socket.exists()).await,
            "driver never bound the data-plane sockets"
        );
        assert!(
            !handle.is_finished(),
            "driver must still be waiting for the engine"
        );

        // A second acquisition rides the in-flight driver instead of spawning
        // another one (which would rebind the same sockets).
        assert!(worker.get_backend_client().await.is_err());
        assert!(Arc::ptr_eq(
            &handle,
            &worker
                .zmq_connect_abort
                .load_full()
                .expect("driver still recorded")
        ));

        worker.abort_background_tasks();
    }

    /// The driver holds the ipc data-plane and TCP handshake binds until it
    /// lands, so worker removal/replacement must abort it — otherwise a
    /// same-URL re-registration collides with an orphan for up to the connect
    /// timeout.
    #[tokio::test]
    async fn abort_background_tasks_cancels_the_zmq_handshake_driver() {
        let base = tempfile::tempdir().unwrap();
        let (worker, input_socket) = unattended_zmq_worker(base.path());

        assert!(
            !worker.zmq_health_check().await.unwrap(),
            "worker is not ready until the handshake lands"
        );
        let handle = worker
            .zmq_connect_abort
            .load_full()
            .expect("probe must start the background handshake driver");
        assert!(
            wait_for(|| input_socket.exists()).await,
            "driver never bound the data-plane sockets"
        );

        worker.abort_background_tasks();

        assert!(
            wait_for(|| handle.is_finished()).await,
            "abort must cancel the in-flight handshake"
        );
        assert!(
            worker.zmq_connect_abort.load_full().is_none(),
            "the aborted driver must not stay recorded"
        );
        assert!(
            !worker.zmq_connect_started.load(Ordering::SeqCst),
            "handshake guard must reset so a later probe can retry"
        );
    }
}
