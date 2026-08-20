#[cfg(test)]
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::{borrow::Cow, sync::Arc, time::Duration};

use dashmap::DashMap;
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use once_cell::sync::Lazy;

// Interned strings are never freed; only intern low-cardinality, server-controlled
// labels (model IDs, worker URLs, normalized paths), never user-controlled input.

/// Global string interner for metric labels.
/// Uses DashMap for lock-free concurrent access.
static STRING_INTERNER: Lazy<DashMap<String, Arc<str>>> = Lazy::new(DashMap::new);

/// Intern a string, returning a cheaply-cloneable Arc<str>.
///
/// This function is designed for high-throughput scenarios where the same
/// strings (model IDs, worker URLs) appear repeatedly. The first call allocates,
/// subsequent calls just clone the Arc (very cheap - just a ref count increment).
pub(crate) fn intern_string(s: &str) -> Arc<str> {
    // Fast path: check if already interned
    if let Some(entry) = STRING_INTERNER.get(s) {
        return Arc::clone(entry.value());
    }

    // Slow path: intern the string
    // Use entry API to avoid TOCTOU race
    STRING_INTERNER
        .entry(s.to_string())
        .or_insert_with(|| Arc::from(s))
        .clone()
}

#[cfg(test)]
pub(crate) fn interner_size() -> usize {
    STRING_INTERNER.len()
}

/// Sentinel substituted once a bounded interner reaches its cap, so a flood of
/// client-controlled label values (model IDs, MCP tool names) cannot mint an
/// unbounded number of interner entries or Prometheus series.
const BOUNDED_LABEL_SENTINEL: &str = "other";

/// Max distinct client-supplied model labels retained before collapsing to the
/// sentinel. A gateway fronts far fewer real models than this; the cap only bites
/// on adversarial or unvalidated input.
const MAX_MODEL_LABELS: usize = 1024;

/// Max distinct client/model-controlled MCP tool-name labels.
const MAX_TOOL_LABELS: usize = 1024;

static MODEL_LABELS: Lazy<DashMap<String, Arc<str>>> = Lazy::new(DashMap::new);
static TOOL_LABELS: Lazy<DashMap<String, Arc<str>>> = Lazy::new(DashMap::new);
static BOUNDED_LABEL_SENTINEL_ARC: Lazy<Arc<str>> = Lazy::new(|| Arc::from(BOUNDED_LABEL_SENTINEL));

/// Intern a client-controlled label with a hard cardinality cap.
///
/// Distinct values beyond `cap` collapse to a shared sentinel, so untrusted input
/// (client-supplied model names, model-generated tool names) cannot grow the
/// interner — or the metric's Prometheus series set — without bound. Unlike an LRU,
/// admitted values are never evicted and re-admitted, which would keep minting new
/// series in the recorder even as the map churned.
fn intern_bounded_label(map: &DashMap<String, Arc<str>>, cap: usize, s: &str) -> Arc<str> {
    if let Some(entry) = map.get(s) {
        return Arc::clone(entry.value());
    }
    // Best-effort cap; a small concurrent overshoot is harmless.
    if map.len() >= cap {
        return Arc::clone(&BOUNDED_LABEL_SENTINEL_ARC);
    }
    map.entry(s.to_string())
        .or_insert_with(|| Arc::from(s))
        .clone()
}

/// Intern a client-supplied model label, bounded by [`MAX_MODEL_LABELS`].
fn intern_model_label(model_id: &str) -> Arc<str> {
    intern_bounded_label(&MODEL_LABELS, MAX_MODEL_LABELS, model_id)
}

/// Intern a client/model-controlled MCP tool-name label, bounded by
/// [`MAX_TOOL_LABELS`].
fn intern_tool_label(tool_name: &str) -> Arc<str> {
    intern_bounded_label(&TOOL_LABELS, MAX_TOOL_LABELS, tool_name)
}

// =============================================================================
// STATIC STRING CONSTANTS
// =============================================================================

/// Static string constants for boolean labels to avoid allocations.
pub const STREAMING_TRUE: &str = "true";
pub const STREAMING_FALSE: &str = "false";

pub const fn bool_to_static_str(b: bool) -> &'static str {
    if b {
        STREAMING_TRUE
    } else {
        STREAMING_FALSE
    }
}

/// Static lookup table for common HTTP status codes to avoid allocations.
/// Returns a static string for known codes, or None for unknown codes.
#[inline]
pub fn status_code_to_static_str(code: u16) -> Option<&'static str> {
    // Using a match with explicit arms is faster than a lookup table for this size
    match code {
        200 => Some("200"),
        201 => Some("201"),
        204 => Some("204"),
        400 => Some("400"),
        401 => Some("401"),
        403 => Some("403"),
        404 => Some("404"),
        408 => Some("408"),
        422 => Some("422"),
        429 => Some("429"),
        500 => Some("500"),
        502 => Some("502"),
        503 => Some("503"),
        504 => Some("504"),
        _ => None,
    }
}

/// Static HTTP method strings to avoid allocations on every request.
pub(crate) mod http_methods {
    pub const GET: &str = "GET";
    pub const POST: &str = "POST";
    pub const PUT: &str = "PUT";
    pub const DELETE: &str = "DELETE";
    pub const PATCH: &str = "PATCH";
    pub const HEAD: &str = "HEAD";
    pub const OPTIONS: &str = "OPTIONS";
}

/// Convert HTTP method to static string. Returns the method as-is for unknown methods.
#[inline]
pub fn method_to_static_str(method: &str) -> &'static str {
    match method {
        "GET" => http_methods::GET,
        "POST" => http_methods::POST,
        "PUT" => http_methods::PUT,
        "DELETE" => http_methods::DELETE,
        "PATCH" => http_methods::PATCH,
        "HEAD" => http_methods::HEAD,
        "OPTIONS" => http_methods::OPTIONS,
        _ => "OTHER",
    }
}

/// Get status code as Cow - static for common codes, allocated for rare ones.
#[inline]
pub fn status_code_to_cow(code: u16) -> Cow<'static, str> {
    match status_code_to_static_str(code) {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(code.to_string()),
    }
}

#[derive(Debug, Clone)]
pub struct PrometheusConfig {
    pub port: u16,
    pub host: String,
    pub duration_buckets: Option<Vec<f64>>,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "0.0.0.0".to_string(),
            duration_buckets: None,
        }
    }
}

/// Upkeep interval for histogram maintenance. Must match the value passed to
/// `PrometheusBuilder::upkeep_timeout()` in `start_prometheus`.
pub(crate) const UPKEEP_INTERVAL_SECS: u64 = 5 * 60;

/// Marks jemalloc as the final artifact's Rust global allocator.
///
/// Call this before [`start_prometheus`] only from a binary or extension that
/// declares `tikv_jemallocator::Jemalloc` with `#[global_allocator]`. Keeping
/// this registration at the artifact boundary prevents `smg` rlib consumers
/// that use the system allocator from publishing statistics for an unused
/// linked jemalloc instance.
pub fn register_jemalloc_as_global_allocator() {
    #[cfg(all(
        feature = "jemalloc-stats",
        not(target_env = "msvc"),
        not(target_env = "musl")
    ))]
    allocator_stats::register_global_allocator();
}

pub(crate) fn init_metrics() {
    #[cfg(all(
        feature = "jemalloc-stats",
        not(target_env = "msvc"),
        not(target_env = "musl")
    ))]
    allocator_stats::describe();

    // Layer 1: HTTP metrics
    describe_counter!(
        "smg_http_requests_total",
        "Total HTTP requests by method and path"
    );
    describe_histogram!(
        "smg_http_request_duration_seconds",
        "HTTP request duration by method and path"
    );
    describe_gauge!(
        "smg_http_inflight_request_age_count",
        "In-flight HTTP requests per age bucket (gt < age <= le, non-cumulative)"
    );
    describe_counter!(
        "smg_http_responses_total",
        "Total HTTP responses by path, status_code and error_code"
    );
    describe_gauge!(
        "smg_http_connections_active",
        "Currently active HTTP connections"
    );
    describe_counter!(
        "smg_http_rate_limit_total",
        "Rate limiting decisions by result (allowed/rejected)"
    );
    describe_gauge!(
        "smg_admission_queue_depth",
        "Requests currently parked in the admission queue"
    );
    describe_counter!(
        "smg_admission_queue_rejected_total",
        "Requests rejected at admission by reason (full/timeout)"
    );
    describe_gauge!(
        "smg_admission_inflight",
        "Requests currently holding an admission token"
    );

    // Layer 2: Router metrics
    describe_counter!(
        "smg_router_requests_total",
        "Total routed requests by router_type, backend_type, connection_mode, model, endpoint, streaming"
    );
    describe_histogram!(
        "smg_router_request_duration_seconds",
        "Router request duration by router_type, backend_type, connection_mode, model, endpoint"
    );
    describe_counter!(
        "smg_router_request_errors_total",
        "Router errors by router_type, backend_type, connection_mode, model, endpoint, error_type"
    );
    describe_histogram!(
        "smg_router_stage_duration_seconds",
        "Pipeline stage duration by router_type and stage (gRPC only)"
    );
    describe_counter!(
        "smg_router_upstream_responses_total",
        "Upstream backend HTTP responses by router_type, status_code, error_code"
    );
    describe_counter!(
        "smg_router_request_buffers_released_early_bytes_total",
        "Serialized size of request buffers freed at dispatch instead of response completion (retries disabled)"
    );

    // Layer 2: Router inference metrics (gRPC only)
    describe_histogram!(
        "smg_router_ttft_seconds",
        "Time to first token by router_type, backend_type, model, endpoint (gRPC only)"
    );
    describe_histogram!(
        "smg_router_tpot_seconds",
        "Time per output token by router_type, backend_type, model, endpoint (gRPC only)"
    );
    describe_counter!(
        "smg_router_tokens_total",
        "Total tokens processed by router_type, backend_type, model, endpoint, token_type (gRPC only)"
    );
    describe_histogram!(
        "smg_router_generation_duration_seconds",
        "Total generation time by router_type, backend_type, model, endpoint (gRPC only)"
    );

    // Layer 2: PD disaggregation metrics (signals only SMG can measure — it is the
    // only component that observes both the prefill and decode legs of a request).
    describe_histogram!(
        "smg_pd_prefill_duration_seconds",
        "Prefill-leg RPC duration by backend_type, model, runtime"
    );
    describe_histogram!(
        "smg_pd_kv_transfer_duration_seconds",
        "KV-transfer window (prefill drain to decode send) by backend_type, model, runtime (vLLM sequential PD)"
    );
    describe_histogram!(
        "smg_pd_ttft_seconds",
        "Honest end-to-end TTFT (prefill start to first decode token) by backend_type, model, runtime"
    );
    describe_counter!(
        "smg_pd_kv_connector_mode_total",
        "KV connector mode decisions by mode (mooncake/nixl/passthrough)"
    );
    describe_counter!(
        "smg_pd_bootstrap_failures_total",
        "PD bootstrap injection failures"
    );
    describe_counter!(
        "smg_pd_kv_transfer_failures_total",
        "PD KV-transfer failures (missing connector params at decode handoff)"
    );

    // Layer 3: Worker metrics
    describe_gauge!(
        "smg_worker_pool_size",
        "Current worker pool size by worker_type, connection_mode, model"
    );
    describe_gauge!(
        "smg_worker_connections_active",
        "Active connections to workers by worker_type, connection_mode"
    );
    describe_gauge!(
        "smg_worker_requests_active",
        "Currently running requests per worker"
    );
    describe_gauge!(
        "smg_worker_health",
        "Worker health status (1=healthy, 0=unhealthy)"
    );
    describe_counter!(
        "smg_worker_health_checks_total",
        "Health check results by worker_type and result"
    );
    describe_counter!(
        "smg_worker_selection_total",
        "Worker selection events by worker_type, connection_mode, model, policy"
    );
    describe_counter!(
        "smg_worker_errors_total",
        "Worker-level errors by worker_type, connection_mode, error_type"
    );
    describe_counter!(
        "smg_kv_event_subscription_failures_total",
        "KV event subscription task failures by worker and reason \
         (panic, join_error, intern_failed)"
    );
    describe_gauge!(
        "smg_workers_overloaded",
        "Workers currently flagged overloaded and excluded from routing, by model"
    );
    describe_counter!(
        "smg_worker_overload_shed_total",
        "Requests shed because every worker for the model is overloaded, by stage \
         (selection, dispatch)"
    );
    describe_gauge!(
        "smg_manual_policy_cache_entries",
        "Number of routing entries in manual policy cache"
    );
    describe_gauge!(
        "smg_cache_tree_chars",
        "Cache-aware string tree cached characters by model (summed across tenants)"
    );
    describe_gauge!(
        "smg_cache_tree_tokens",
        "Cache-aware token tree cached tokens by model (summed across tenants)"
    );
    describe_gauge!(
        "smg_cache_tree_tenants",
        "Cache-aware tree tenant count by model and tree (string/token)"
    );
    describe_gauge!(
        "smg_cache_placement_entries",
        "Cache-aware hash-index placement entries by model (keys with a live holder)"
    );

    // Layer 3: Worker resilience metrics (circuit breaker)
    describe_gauge!(
        "smg_worker_cb_state",
        "Circuit breaker state per worker (0=closed, 1=open, 2=half_open)"
    );
    describe_counter!(
        "smg_worker_cb_transitions_total",
        "Circuit breaker state transitions by worker, from, to"
    );
    describe_counter!(
        "smg_worker_cb_outcomes_total",
        "Circuit breaker outcomes by worker and outcome (success/failure)"
    );
    describe_gauge!(
        "smg_worker_cb_consecutive_failures",
        "Current consecutive failure count per worker"
    );
    describe_gauge!(
        "smg_worker_cb_consecutive_successes",
        "Current consecutive success count per worker"
    );

    // Layer 3: Worker resilience metrics (retry)
    describe_counter!(
        "smg_worker_retries_total",
        "Total retry attempts by worker_type and endpoint"
    );
    describe_counter!(
        "smg_worker_retries_exhausted_total",
        "Requests that exhausted all retries by worker_type and endpoint"
    );
    describe_histogram!(
        "smg_worker_retry_backoff_seconds",
        "Retry backoff duration by attempt number"
    );

    // Layer 3: Engine load re-export (from the GetLoads poll loop)
    describe_gauge!(
        "smg_engine_running_requests",
        "Engine-reported running requests by worker, model, dp_rank"
    );
    describe_gauge!(
        "smg_engine_waiting_requests",
        "Engine-reported waiting requests by worker, model, dp_rank"
    );
    describe_gauge!(
        "smg_engine_token_usage",
        "Engine-reported KV token usage ratio (0.0-1.0) by worker, model, dp_rank"
    );
    describe_gauge!(
        "smg_engine_gen_throughput",
        "Engine-reported generation throughput (tokens/s) by worker, model, dp_rank"
    );
    describe_gauge!(
        "smg_engine_cache_hit_rate",
        "Engine-reported prefix cache hit rate (0.0-1.0) by worker, model, dp_rank"
    );
    describe_gauge!(
        "smg_engine_pd_kv_transfer_latency_ms",
        "Engine-reported PD KV transfer latency (ms) by worker, role, dp_rank"
    );
    describe_gauge!(
        "smg_engine_pd_kv_transfer_speed_gb_s",
        "Engine-reported PD KV transfer speed (GB/s) by worker, role, dp_rank"
    );
    describe_gauge!(
        "smg_engine_pd_prefill_queue_reqs",
        "Engine-reported PD prefill queue depth by worker, role, dp_rank"
    );
    describe_gauge!(
        "smg_engine_pd_decode_queue_reqs",
        "Engine-reported PD decode queue depth by worker, role, dp_rank"
    );

    // Layer 4: Discovery metrics
    describe_counter!(
        "smg_discovery_registrations_total",
        "Worker registration attempts by source and result"
    );
    describe_counter!(
        "smg_discovery_deregistrations_total",
        "Worker deregistration events by source and reason"
    );
    describe_histogram!(
        "smg_discovery_sync_duration_seconds",
        "Discovery sync duration by source"
    );
    describe_gauge!(
        "smg_discovery_workers_discovered",
        "Workers known via discovery by source"
    );

    // Layer 5: MCP metrics
    describe_counter!(
        "smg_mcp_tool_calls_total",
        "Total MCP tool invocations by model, tool_name, result"
    );
    describe_histogram!(
        "smg_mcp_tool_duration_seconds",
        "MCP tool execution duration by model, tool_name"
    );
    describe_gauge!("smg_mcp_servers_active", "Active MCP server connections");
    describe_counter!(
        "smg_mcp_tool_iterations_total",
        "Tool loop iterations in Responses API by model"
    );

    // Layer 6: Database metrics
    describe_counter!(
        "smg_db_operations_total",
        "Total database operations by storage_type, operation, result"
    );
    describe_histogram!(
        "smg_db_operation_duration_seconds",
        "Database operation duration by storage_type, operation"
    );
    describe_gauge!(
        "smg_db_connections_active",
        "Active database connections by storage_type"
    );
    describe_counter!("smg_db_items_stored", "Total items stored by storage_type");

    // Multimodal tensor transport (shm vs inline), labeled by runtime.
    describe_counter!(
        "smg_mm_tensors_total",
        "Multimodal tensors sent, by runtime and transport path (shm/inline)"
    );
    describe_counter!(
        "smg_mm_tensor_bytes_total",
        "Multimodal tensor bytes sent, by runtime and transport path (shm/inline)"
    );
    describe_counter!(
        "smg_mm_shm_write_failures_total",
        "SHM tensor write attempts that failed and fell back to inline, by runtime"
    );

    // Layer 0: Tokio runtime self-observability (event-loop canary + sampler).
    super::runtime_metrics::describe();

    // Initialize mesh metrics
    smg_mesh::init_mesh_metrics();

    // Priority scheduler metrics (no-op at scrape time unless the scheduler
    // is enabled and recording).
    use crate::middleware::scheduler::metrics as scheduler_metrics;
    scheduler_metrics::describe();
}

#[expect(
    clippy::expect_used,
    reason = "startup initialization — metrics exporter must be installed or the process cannot serve metrics"
)]
pub fn start_prometheus(config: PrometheusConfig) -> PrometheusHandle {
    init_metrics();

    let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
    let duration_bucket: Vec<f64> = config.duration_buckets.unwrap_or_else(|| {
        vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0, 300.0, 480.0, 900.0, 1200.0, 1800.0, 2700.0, 3600.0,
            5400.0, 7200.0,
        ]
    });

    // The event-loop canary needs its own buckets: its name does not end in
    // `duration_seconds`, and the request-latency buckets above are far too
    // coarse for 0-1s wake drift. Without explicit buckets the recorder would
    // render it as a summary.
    let canary_matcher = Matcher::Full(super::runtime_metrics::EVENT_LOOP_DELAY_SECONDS.into());

    // TTFT and TPOT (per-request mean inter-token latency) end in `_seconds`
    // but NOT `duration_seconds`, so without explicit buckets the recorder
    // renders them as summaries (quantile lines only) — not heatmap-able. Reuse
    // the request-latency buckets: they span 0.001-7200s, fine for both the
    // sub-second-to-seconds TTFT and the tens-of-ms TPOT.
    let ttft_matcher = Matcher::Suffix(String::from("ttft_seconds"));
    let tpot_matcher = Matcher::Suffix(String::from("tpot_seconds"));

    PrometheusBuilder::new()
        .upkeep_timeout(Duration::from_secs(UPKEEP_INTERVAL_SECS))
        .set_buckets_for_metric(duration_matcher, &duration_bucket)
        .expect("failed to set duration bucket")
        .set_buckets_for_metric(ttft_matcher, &duration_bucket)
        .expect("failed to set ttft bucket")
        .set_buckets_for_metric(tpot_matcher, &duration_bucket)
        .expect("failed to set tpot bucket")
        .set_buckets_for_metric(
            canary_matcher,
            super::runtime_metrics::EVENT_LOOP_DELAY_BUCKETS,
        )
        .expect("failed to set event loop delay buckets")
        .install_recorder()
        .inspect(|_| {
            #[cfg(all(
                feature = "jemalloc-stats",
                not(target_env = "msvc"),
                not(target_env = "musl")
            ))]
            allocator_stats::start_reporting();
        })
        .expect("failed to install Prometheus recorder")
}

#[cfg(all(
    feature = "jemalloc-stats",
    not(target_env = "msvc"),
    not(target_env = "musl")
))]
pub(crate) mod allocator_stats {
    use std::sync::atomic::{AtomicBool, Ordering};

    use metrics::{describe_gauge, gauge};

    static JEMALLOC_IS_GLOBAL: AtomicBool = AtomicBool::new(false);

    pub(super) fn register_global_allocator() {
        JEMALLOC_IS_GLOBAL.store(true, Ordering::Release);
    }

    fn is_global_allocator() -> bool {
        JEMALLOC_IS_GLOBAL.load(Ordering::Acquire)
    }

    pub(crate) fn describe() {
        if !is_global_allocator() {
            return;
        }
        describe_gauge!(
            "smg_allocator_allocated_bytes",
            "Bytes in live Rust allocations managed by SMG's jemalloc instance"
        );
        describe_gauge!(
            "smg_allocator_active_bytes",
            "Bytes in active pages for SMG's Rust jemalloc heap"
        );
        describe_gauge!(
            "smg_allocator_resident_bytes",
            "Upper bound on resident bytes for SMG's Rust jemalloc heap"
        );
        describe_gauge!(
            "smg_allocator_metadata_bytes",
            "Metadata bytes for SMG's Rust jemalloc instance"
        );
    }

    fn record() {
        use tikv_jemalloc_ctl::{epoch, stats};
        if epoch::advance().is_err() {
            return;
        }
        if let Ok(v) = stats::allocated::read() {
            gauge!("smg_allocator_allocated_bytes").set(v as f64);
        }
        if let Ok(v) = stats::active::read() {
            gauge!("smg_allocator_active_bytes").set(v as f64);
        }
        if let Ok(v) = stats::resident::read() {
            gauge!("smg_allocator_resident_bytes").set(v as f64);
        }
        if let Ok(v) = stats::metadata::read() {
            gauge!("smg_allocator_metadata_bytes").set(v as f64);
        }
    }

    /// Registration at the final-artifact boundary keeps these gauges tied to
    /// Rust's actual global allocator.
    pub(crate) fn start_reporting() {
        if !is_global_allocator() {
            return;
        }
        record();
        // Plain thread: metrics must not depend on a runtime being alive.
        let _ = std::thread::Builder::new()
            .name("smg-allocator-stats".into())
            .spawn(|| loop {
                std::thread::sleep(std::time::Duration::from_secs(60));
                record();
            });
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn jemalloc_stats_interface_readable() {
            use tikv_jemalloc_ctl::{epoch, stats};
            epoch::advance().expect("epoch advance");
            stats::allocated::read().expect("stats.allocated readable");
            stats::active::read().expect("stats.active readable");
            stats::resident::read().expect("stats.resident readable");
            stats::metadata::read().expect("stats.metadata readable");
        }
    }
}

/// Label constants for consistent metric labeling
pub mod metrics_labels {
    // Router types
    pub const ROUTER_OPENAI: &str = "openai";
    pub const ROUTER_HTTP: &str = "http";
    pub const ROUTER_GRPC: &str = "grpc";

    // Backend types
    pub const BACKEND_REGULAR: &str = "regular";
    pub const BACKEND_PD: &str = "pd";
    pub const BACKEND_EXTERNAL: &str = "external";
    pub const BACKEND_HARMONY: &str = "harmony";

    // Connection modes
    pub const CONNECTION_HTTP: &str = "http";
    pub const CONNECTION_GRPC: &str = "grpc";
    pub const CONNECTION_ZMQ: &str = "zmq";

    // Endpoints
    pub const ENDPOINT_CHAT: &str = "chat";
    pub const ENDPOINT_GENERATE: &str = "generate";
    pub const ENDPOINT_RESPONSES: &str = "responses";
    pub const ENDPOINT_COMPLETIONS: &str = "completions";
    pub const ENDPOINT_RERANK: &str = "rerank";
    pub const ENDPOINT_EMBEDDINGS: &str = "embeddings";
    pub const ENDPOINT_CLASSIFY: &str = "classify";
    pub const ENDPOINT_MESSAGES: &str = "messages";
    pub const ENDPOINT_REALTIME: &str = "realtime";
    pub const ENDPOINT_REALTIME_SESSIONS: &str = "realtime_sessions";
    pub const ENDPOINT_REALTIME_CLIENT_SECRETS: &str = "realtime_client_secrets";
    pub const ENDPOINT_REALTIME_TRANSCRIPTION: &str = "realtime_transcription";
    pub const ENDPOINT_AUDIO_TRANSCRIPTIONS: &str = "audio_transcriptions";

    // Connection modes
    pub const CONNECTION_WEBSOCKET: &str = "websocket";
    pub const CONNECTION_WEBRTC: &str = "webrtc";

    // Worker types
    pub const WORKER_REGULAR: &str = "regular";
    pub const WORKER_PREFILL: &str = "prefill";
    pub const WORKER_DECODE: &str = "decode";
    pub const WORKER_ENCODE: &str = "encode";
    pub const WORKER_HTTP: &str = "http";
    pub const WORKER_GRPC: &str = "grpc";

    // Token types
    pub const TOKEN_INPUT: &str = "input";
    pub const TOKEN_OUTPUT: &str = "output";

    // PD KV connector modes (smg_pd_kv_connector_mode_total)
    pub const KV_CONNECTOR_MOONCAKE: &str = "mooncake";
    pub const KV_CONNECTOR_NIXL: &str = "nixl";
    pub const KV_CONNECTOR_PASSTHROUGH: &str = "passthrough";

    // Storage types
    pub const STORAGE_RESPONSE: &str = "response";
    pub const STORAGE_CONVERSATION: &str = "conversation";
    pub const STORAGE_CONVERSATION_ITEM: &str = "conversation_item";

    // Database operations
    pub const DB_OP_GET: &str = "get";
    pub const DB_OP_PUT: &str = "put";
    pub const DB_OP_DELETE: &str = "delete";
    pub const DB_OP_LIST: &str = "list";

    // Result types
    pub const RESULT_SUCCESS: &str = "success";
    pub const RESULT_ERROR: &str = "error";
    pub const RESULT_TIMEOUT: &str = "timeout";
    pub const RESULT_NOT_FOUND: &str = "not_found";

    // Discovery sources
    pub const DISCOVERY_STATIC: &str = "static";
    pub const DISCOVERY_KUBERNETES: &str = "kubernetes";
    pub const DISCOVERY_CONSUL: &str = "consul";
    pub const DISCOVERY_MANUAL: &str = "manual";

    // Discovery registration results
    pub const REGISTRATION_SUCCESS: &str = "success";
    pub const REGISTRATION_FAILED: &str = "failed";
    pub const DEREGISTRATION_RECONCILED: &str = "reconciled";

    // Rate limit results
    pub const RATE_LIMIT_ALLOWED: &str = "allowed";
    pub const RATE_LIMIT_REJECTED: &str = "rejected";

    // Admission rejection reasons
    pub const ADMISSION_REJECTED_FULL: &str = "full";
    pub const ADMISSION_REJECTED_TIMEOUT: &str = "timeout";

    // Circuit breaker states
    pub const CB_CLOSED: &str = "closed";
    pub const CB_OPEN: &str = "open";
    pub const CB_HALF_OPEN: &str = "half_open";

    // Circuit breaker outcomes
    pub const CB_SUCCESS: &str = "success";
    pub const CB_FAILURE: &str = "failure";

    // Router error types
    pub const ERROR_NO_WORKERS: &str = "no_workers";
    pub const ERROR_TIMEOUT: &str = "timeout";
    pub const ERROR_BACKEND: &str = "backend_error";
    pub const ERROR_VALIDATION: &str = "validation_error";
    pub const ERROR_INTERNAL: &str = "internal_error";
}

/// SMG Metrics helper struct for the new layered metrics architecture.
///
/// Design principles for low overhead:
/// - Dynamic labels use string interning (single allocation per unique value)
/// - Static labels use the metrics crate's internal caching
pub struct Metrics;

/// Parameters for recording streaming metrics.
pub struct StreamingMetricsParams<'a> {
    /// Router type label (e.g., "grpc", "http")
    pub router_type: &'static str,
    /// Backend type label (e.g., "regular", "pd")
    pub backend_type: &'static str,
    /// Model identifier (will be converted to owned String for metrics)
    pub model_id: &'a str,
    /// Endpoint label (e.g., "chat", "generate")
    pub endpoint: &'static str,
    /// Time to first token (None if no tokens were generated)
    pub ttft: Option<Duration>,
    /// Total generation time
    pub generation_duration: Duration,
    /// Input token count (None for endpoints that don't track this)
    pub input_tokens: Option<u64>,
    /// Output token count
    pub output_tokens: u64,
}

impl Metrics {
    /// Record an HTTP request.
    /// Here we want a metric to directly reflect user's experience ("I am sending a request")
    /// when viewing the router as a blackbox, and is bumped immediately when the request arrives.
    pub fn record_http_request(method: &'static str, path: &str) {
        let path_interned = intern_string(path);
        counter!(
            "smg_http_requests_total",
            "method" => method,
            "path" => path_interned,
        )
        .increment(1);
    }

    /// Record HTTP request duration.
    /// For best performance, pass static strings for method.
    pub fn record_http_duration(method: &'static str, path: &str, duration: Duration) {
        let path_interned = intern_string(path);
        histogram!(
            "smg_http_request_duration_seconds",
            "method" => method,
            "path" => path_interned
        )
        .record(duration.as_secs_f64());
    }

    /// Set active HTTP connections count
    pub fn set_http_connections_active(count: usize) {
        gauge!("smg_http_connections_active").set(count as f64);
    }

    /// Record HTTP response.
    pub fn record_http_response(path: &str, status_code: u16, error_code: &str) {
        let path_interned = intern_string(path);
        let status_str: Cow<'static, str> = status_code_to_cow(status_code);
        let error_interned = intern_string(error_code);
        counter!(
            "smg_http_responses_total",
            "path" => path_interned,
            "status_code" => status_str,
            "error_code" => error_interned
        )
        .increment(1);
    }

    /// Record rate limit decision.
    pub fn record_http_rate_limit(result: &'static str) {
        counter!(
            "smg_http_rate_limit_total",
            "result" => result
        )
        .increment(1);
    }

    /// Track a request entering the admission queue.
    pub fn record_admission_queue_entered() {
        gauge!("smg_admission_queue_depth").increment(1.0);
    }

    /// Track a request leaving the admission queue (admitted, rejected, or cancelled).
    pub fn record_admission_queue_exited() {
        gauge!("smg_admission_queue_depth").decrement(1.0);
    }

    /// Record a request rejected at admission.
    pub fn record_admission_rejected(reason: &'static str) {
        counter!(
            "smg_admission_queue_rejected_total",
            "reason" => reason
        )
        .increment(1);
    }

    /// Track acquisition of an admission token.
    pub fn record_admission_inflight_acquired() {
        gauge!("smg_admission_inflight").increment(1.0);
    }

    /// Track release of an admission token.
    pub fn record_admission_inflight_released() {
        gauge!("smg_admission_inflight").decrement(1.0);
    }

    /// Record one multimodal tensor sent over `path` ("inline"|"shm"|"remote") for `runtime`.
    pub fn record_mm_tensor(runtime: &'static str, path: &'static str, nbytes: usize) {
        counter!("smg_mm_tensors_total", "runtime" => runtime, "path" => path).increment(1);
        counter!("smg_mm_tensor_bytes_total", "runtime" => runtime, "path" => path)
            .increment(nbytes as u64);
    }

    /// Record a SHM tensor write that failed and fell back to inline, for `runtime`.
    pub fn record_mm_shm_write_failure(runtime: &'static str) {
        counter!("smg_mm_shm_write_failures_total", "runtime" => runtime).increment(1);
    }

    // ========================================================================
    // Layer 2: Router metrics
    // ========================================================================

    /// Record a routed request.
    ///
    /// Uses string interning for model_id to avoid repeated allocations.
    ///
    /// # Arguments
    /// * `streaming` - Use `bool_to_static_str(request.stream)` or the constants
    pub fn record_router_request(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        streaming: &'static str,
    ) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_router_requests_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "endpoint" => endpoint,
            "streaming" => streaming
        )
        .increment(1);
    }

    /// Record router request duration.
    /// Uses string interning for model_id.
    pub fn record_router_duration(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_router_request_duration_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record a router error.
    /// Uses string interning for model_id.
    pub fn record_router_error(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        error_type: &'static str,
    ) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_router_request_errors_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "endpoint" => endpoint,
            "error_type" => error_type
        )
        .increment(1);
    }

    /// Record pipeline stage duration (gRPC only).
    /// All labels are static, so this is very fast.
    pub fn record_router_stage_duration(
        router_type: &'static str,
        stage: &'static str,
        duration: Duration,
    ) {
        histogram!(
            "smg_router_stage_duration_seconds",
            "router_type" => router_type,
            "stage" => stage
        )
        .record(duration.as_secs_f64());
    }

    /// Record a single-shot resend after a pre-response transport failure.
    pub fn record_upstream_send_retry(router_type: &'static str) {
        counter!(
            "smg_router_upstream_send_retries_total",
            "router_type" => router_type
        )
        .increment(1);
    }

    /// Record request buffers freed at dispatch (retries disabled), sized by
    /// the serialized upstream body.
    pub fn record_request_buffers_released_early(bytes: usize) {
        counter!("smg_router_request_buffers_released_early_bytes_total").increment(bytes as u64);
    }

    /// Record upstream backend response.
    /// Uses static strings for common status codes and interning for error_code.
    pub fn record_router_upstream_response(
        router_type: &'static str,
        status_code: u16,
        error_code: &str,
    ) {
        let status_str: Cow<'static, str> = status_code_to_cow(status_code);
        let error_interned = intern_string(error_code);
        counter!(
            "smg_router_upstream_responses_total",
            "router_type" => router_type,
            "status_code" => status_str,
            "error_code" => error_interned
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 2: Router inference metrics (gRPC only)
    // ========================================================================

    /// Record time to first token.
    /// Uses string interning for model_id.
    pub fn record_router_ttft(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_router_ttft_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record time per output token
    pub fn record_router_tpot(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_router_tpot_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record tokens processed
    pub fn record_router_tokens(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        token_type: &'static str,
        count: u64,
    ) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_router_tokens_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint,
            "token_type" => token_type
        )
        .increment(count);
    }

    /// Record total generation duration.
    /// Uses string interning for model_id.
    pub fn record_router_generation_duration(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_router_generation_duration_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record all streaming metrics in a single batch call.
    ///
    /// This consolidates TTFT, TPOT, generation duration, and token metrics
    /// into one function, handling TPOT calculation internally.
    pub fn record_streaming_metrics(params: StreamingMetricsParams<'_>) {
        let StreamingMetricsParams {
            router_type,
            backend_type,
            model_id,
            endpoint,
            ttft,
            generation_duration,
            input_tokens,
            output_tokens,
        } = params;

        // Intern model string once - Arc::clone is just a ref count increment
        let model = intern_model_label(model_id);

        // TTFT and TPOT (only if we have a first token time)
        if let Some(ttft_duration) = ttft {
            histogram!(
                "smg_router_ttft_seconds",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => Arc::clone(&model),
                "endpoint" => endpoint
            )
            .record(ttft_duration.as_secs_f64());

            // TPOT - only meaningful with >1 output token
            if output_tokens > 1 {
                let time_after_first = generation_duration.saturating_sub(ttft_duration);
                let tpot = time_after_first / (output_tokens as u32 - 1);
                histogram!(
                    "smg_router_tpot_seconds",
                    "router_type" => router_type,
                    "backend_type" => backend_type,
                    "model" => Arc::clone(&model),
                    "endpoint" => endpoint
                )
                .record(tpot.as_secs_f64());
            }
        }

        // Generation duration (always recorded)
        histogram!(
            "smg_router_generation_duration_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => Arc::clone(&model),
            "endpoint" => endpoint
        )
        .record(generation_duration.as_secs_f64());

        // Input tokens (if available)
        if let Some(input) = input_tokens {
            counter!(
                "smg_router_tokens_total",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => Arc::clone(&model),
                "endpoint" => endpoint,
                "token_type" => metrics_labels::TOKEN_INPUT
            )
            .increment(input);
        }

        // Output tokens (always recorded - move model on final use)
        counter!(
            "smg_router_tokens_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint,
            "token_type" => metrics_labels::TOKEN_OUTPUT
        )
        .increment(output_tokens);
    }

    // ========================================================================
    // Layer 2: PD disaggregation metrics
    //
    // Per-request, engine-agnostic signals that no backend can self-report: SMG
    // is the only component that sees both the prefill and decode legs. All
    // durations come from a monotonic clock and are recorded once per request
    // (never per retry attempt).
    // ========================================================================

    /// Record prefill-leg RPC duration.
    /// Uses string interning for model_id; runtime is a static label.
    pub fn record_pd_prefill_duration(
        backend_type: &'static str,
        model_id: &str,
        runtime: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_pd_prefill_duration_seconds",
            "backend_type" => backend_type,
            "model" => model,
            "runtime" => runtime
        )
        .record(duration.as_secs_f64());
    }

    /// Record the KV-transfer window (prefill drain to decode send) for vLLM
    /// sequential PD. Uses string interning for model_id; runtime is a static label.
    pub fn record_pd_kv_transfer_duration(
        backend_type: &'static str,
        model_id: &str,
        runtime: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_pd_kv_transfer_duration_seconds",
            "backend_type" => backend_type,
            "model" => model,
            "runtime" => runtime
        )
        .record(duration.as_secs_f64());
    }

    /// Record honest end-to-end TTFT: prefill start to first decode token.
    ///
    /// INVARIANT: this is the user-facing complement to
    /// `smg_router_ttft_seconds{backend_type="pd"}`, which measures only the
    /// decode leg (first decode token minus decode-send). For sequential PD the
    /// two differ by the prefill + KV-transfer time; both are kept on purpose.
    /// Uses string interning for model_id; runtime is a static label.
    pub fn record_pd_ttft(
        backend_type: &'static str,
        model_id: &str,
        runtime: &'static str,
        duration: Duration,
    ) {
        let model = intern_model_label(model_id);
        histogram!(
            "smg_pd_ttft_seconds",
            "backend_type" => backend_type,
            "model" => model,
            "runtime" => runtime
        )
        .record(duration.as_secs_f64());
    }

    /// Record a KV connector mode decision (mooncake/nixl/passthrough).
    pub fn record_pd_kv_connector_mode(mode: &'static str) {
        counter!(
            "smg_pd_kv_connector_mode_total",
            "mode" => mode
        )
        .increment(1);
    }

    /// Record a PD bootstrap injection failure.
    pub fn record_pd_bootstrap_failure() {
        counter!("smg_pd_bootstrap_failures_total").increment(1);
    }

    /// Record a PD KV-transfer failure (missing connector params at handoff).
    pub fn record_pd_kv_transfer_failure() {
        counter!("smg_pd_kv_transfer_failures_total").increment(1);
    }

    // ========================================================================
    // Layer 3: Worker metrics
    // ========================================================================

    /// Set worker pool size
    pub fn set_worker_pool_size(
        worker_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        size: usize,
    ) {
        let model = intern_model_label(model_id);
        gauge!(
            "smg_worker_pool_size",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model
        )
        .set(size as f64);
    }

    /// Set active worker connections
    pub fn set_worker_connections_active(
        worker_type: &'static str,
        connection_mode: &'static str,
        count: usize,
    ) {
        gauge!(
            "smg_worker_connections_active",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode
        )
        .set(count as f64);
    }

    /// Record health check result
    pub fn record_worker_health_check(worker_type: &'static str, result: &'static str) {
        counter!(
            "smg_worker_health_checks_total",
            "worker_type" => worker_type,
            "result" => result
        )
        .increment(1);
    }

    /// Record worker selection
    pub fn record_worker_selection(
        worker_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        policy: &'static str,
    ) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_worker_selection_total",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "policy" => policy
        )
        .increment(1);
    }

    /// Record worker error
    pub fn record_worker_error(
        worker_type: &'static str,
        connection_mode: &'static str,
        error_type: &'static str,
    ) {
        counter!(
            "smg_worker_errors_total",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "error_type" => error_type
        )
        .increment(1);
    }

    /// Set the count of workers a model currently has vetoed by the absolute
    /// overload guard. Written only when a worker's flag transitions.
    pub fn set_workers_overloaded(model_id: &str, count: usize) {
        let model = intern_model_label(model_id);
        gauge!("smg_workers_overloaded", "model" => model).set(count as f64);
    }

    /// Record a request shed because every worker for the model is overloaded.
    /// `stage` is "selection" or "dispatch".
    pub fn record_worker_overload_shed(stage: &'static str) {
        counter!(
            "smg_worker_overload_shed_total",
            "stage" => stage
        )
        .increment(1);
    }

    /// Record manual policy execution branch for routing decisions
    pub fn record_worker_manual_policy_branch(branch: &'static str) {
        counter!(
            "smg_manual_policy_branch_total",
            "branch" => branch
        )
        .increment(1);
    }

    /// Set manual policy cache entries count
    pub fn set_manual_policy_cache_entries(count: usize) {
        gauge!("smg_manual_policy_cache_entries").set(count as f64);
    }

    /// Record which source supplied the sticky routing key for a keyed request
    pub fn record_routing_key_source(source: &'static str) {
        counter!(
            "smg_routing_key_source_total",
            "source" => source
        )
        .increment(1);
    }

    /// Set cache-aware string-tree cached characters for a model
    pub fn set_cache_tree_chars(model_id: &str, chars: usize) {
        let model = intern_model_label(model_id);
        gauge!("smg_cache_tree_chars", "model" => model).set(chars as f64);
    }

    /// Set cache-aware token-tree cached tokens for a model
    pub fn set_cache_tree_tokens(model_id: &str, tokens: usize) {
        let model = intern_model_label(model_id);
        gauge!("smg_cache_tree_tokens", "model" => model).set(tokens as f64);
    }

    /// Set cache-aware tree tenant count for a model and tree kind ("string"/"token")
    pub fn set_cache_tree_tenants(model_id: &str, tree: &'static str, count: usize) {
        let model = intern_model_label(model_id);
        gauge!("smg_cache_tree_tenants", "model" => model, "tree" => tree).set(count as f64);
    }

    /// Set cache-aware hash-index placement entry count for a model
    pub fn set_cache_placement_entries(model_id: &str, count: usize) {
        let model = intern_model_label(model_id);
        gauge!("smg_cache_placement_entries", "model" => model).set(count as f64);
    }

    /// Record consistent hashing policy execution branch for routing decisions
    pub fn record_worker_consistent_hashing_policy_branch(branch: &'static str) {
        counter!(
            "smg_consistent_hashing_policy_branch_total",
            "branch" => branch
        )
        .increment(1);
    }

    /// Record prefix hash policy execution branch for routing decisions
    pub fn record_worker_prefix_hash_policy_branch(branch: &'static str) {
        counter!(
            "smg_prefix_hash_policy_branch_total",
            "branch" => branch
        )
        .increment(1);
    }

    /// Set running requests per worker
    pub fn set_worker_requests_active(worker: &str, count: usize) {
        let worker_interned = intern_string(worker);
        gauge!(
            "smg_worker_requests_active",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set active routing keys per worker
    pub fn set_worker_routing_keys_active(worker: &str, count: usize) {
        let worker_interned = intern_string(worker);
        gauge!(
            "smg_worker_routing_keys_active",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set worker health status
    pub fn set_worker_health(worker_url: &str, healthy: bool) {
        let worker_interned = intern_string(worker_url);
        gauge!(
            "smg_worker_health",
            "worker" => worker_interned
        )
        .set(if healthy { 1.0 } else { 0.0 });
    }

    /// Record a KV event subscription task failure (panic, join error, or
    /// worker-id intern failure)
    pub fn record_kv_event_subscription_failure(worker_url: &str, reason: &'static str) {
        let worker_interned = intern_string(worker_url);
        counter!(
            "smg_kv_event_subscription_failures_total",
            "worker" => worker_interned,
            "reason" => reason
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 3: Worker resilience metrics (circuit breaker)
    // ========================================================================

    /// Set circuit breaker state (0=closed, 1=open, 2=half_open)
    pub fn set_worker_cb_state(worker: &str, state_code: u8) {
        let worker_interned = intern_string(worker);
        gauge!(
            "smg_worker_cb_state",
            "worker" => worker_interned
        )
        .set(state_code as f64);
    }

    /// Record circuit breaker state transition
    pub fn record_worker_cb_transition(worker: &str, from: &'static str, to: &'static str) {
        let worker_interned = intern_string(worker);
        counter!(
            "smg_worker_cb_transitions_total",
            "worker" => worker_interned,
            "from" => from,
            "to" => to
        )
        .increment(1);
    }

    /// Record circuit breaker outcome
    pub fn record_worker_cb_outcome(worker: &str, outcome: &'static str) {
        let worker_interned = intern_string(worker);
        counter!(
            "smg_worker_cb_outcomes_total",
            "worker" => worker_interned,
            "outcome" => outcome
        )
        .increment(1);
    }

    /// Set circuit breaker consecutive failures
    pub fn set_worker_cb_consecutive_failures(worker: &str, count: u32) {
        let worker_interned = intern_string(worker);
        gauge!(
            "smg_worker_cb_consecutive_failures",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set circuit breaker consecutive successes
    pub fn set_worker_cb_consecutive_successes(worker: &str, count: u32) {
        let worker_interned = intern_string(worker);
        gauge!(
            "smg_worker_cb_consecutive_successes",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    // ========================================================================
    // Layer 3: Worker resilience metrics (retry)
    // ========================================================================

    /// Record retry attempt
    pub fn record_worker_retry(worker_type: &'static str, endpoint: &'static str) {
        counter!(
            "smg_worker_retries_total",
            "worker_type" => worker_type,
            "endpoint" => endpoint
        )
        .increment(1);
    }

    /// Record retries exhausted
    pub fn record_worker_retries_exhausted(worker_type: &'static str, endpoint: &'static str) {
        counter!(
            "smg_worker_retries_exhausted_total",
            "worker_type" => worker_type,
            "endpoint" => endpoint
        )
        .increment(1);
    }

    /// Record retry backoff duration.
    pub fn record_worker_retry_backoff(attempt: u32, duration: Duration) {
        let attempt_str: Cow<'static, str> = match attempt {
            1 => Cow::Borrowed("1"),
            2 => Cow::Borrowed("2"),
            3 => Cow::Borrowed("3"),
            4 => Cow::Borrowed("4"),
            5 => Cow::Borrowed("5"),
            _ => Cow::Owned(attempt.to_string()),
        };
        histogram!(
            "smg_worker_retry_backoff_seconds",
            "attempt" => attempt_str
        )
        .record(duration.as_secs_f64());
    }

    // ========================================================================
    // Layer 4: Discovery metrics
    // ========================================================================

    /// Record worker registration attempt
    pub fn record_discovery_registration(source: &'static str, result: &'static str) {
        counter!(
            "smg_discovery_registrations_total",
            "source" => source,
            "result" => result
        )
        .increment(1);
    }

    /// Record worker deregistration
    pub fn record_discovery_deregistration(source: &'static str, reason: &'static str) {
        counter!(
            "smg_discovery_deregistrations_total",
            "source" => source,
            "reason" => reason
        )
        .increment(1);
    }

    /// Record discovery sync duration
    pub fn record_discovery_sync_duration(source: &'static str, duration: Duration) {
        histogram!(
            "smg_discovery_sync_duration_seconds",
            "source" => source
        )
        .record(duration.as_secs_f64());
    }

    /// Set workers discovered count
    pub fn set_discovery_workers_discovered(source: &'static str, count: usize) {
        gauge!(
            "smg_discovery_workers_discovered",
            "source" => source
        )
        .set(count as f64);
    }

    // ========================================================================
    // Layer 5: MCP metrics
    // ========================================================================

    /// Record MCP tool call
    pub fn record_mcp_tool_call(model_id: &str, tool_name: &str, result: &'static str) {
        let model = intern_model_label(model_id);
        let tool = intern_tool_label(tool_name);
        counter!(
            "smg_mcp_tool_calls_total",
            "model" => model,
            "tool_name" => tool,
            "result" => result
        )
        .increment(1);
    }

    /// Record MCP tool execution duration
    pub fn record_mcp_tool_duration(model_id: &str, tool_name: &str, duration: Duration) {
        let model = intern_model_label(model_id);
        let tool = intern_tool_label(tool_name);
        histogram!(
            "smg_mcp_tool_duration_seconds",
            "model" => model,
            "tool_name" => tool
        )
        .record(duration.as_secs_f64());
    }

    /// Set active MCP servers count
    pub fn set_mcp_servers_active(count: usize) {
        gauge!("smg_mcp_servers_active").set(count as f64);
    }

    /// Record MCP tool loop iteration
    pub fn record_mcp_tool_iteration(model_id: &str) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_mcp_tool_iterations_total",
            "model" => model
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 6: Database metrics
    // ========================================================================

    /// Record database operation
    pub fn record_db_operation(
        storage_type: &'static str,
        operation: &'static str,
        result: &'static str,
    ) {
        counter!(
            "smg_db_operations_total",
            "storage_type" => storage_type,
            "operation" => operation,
            "result" => result
        )
        .increment(1);
    }

    /// Record database operation duration
    pub fn record_db_operation_duration(
        storage_type: &'static str,
        operation: &'static str,
        duration: Duration,
    ) {
        histogram!(
            "smg_db_operation_duration_seconds",
            "storage_type" => storage_type,
            "operation" => operation
        )
        .record(duration.as_secs_f64());
    }

    /// Set active database connections
    pub fn set_db_connections_active(storage_type: &'static str, count: usize) {
        gauge!(
            "smg_db_connections_active",
            "storage_type" => storage_type
        )
        .set(count as f64);
    }

    /// Record item stored
    pub fn increment_db_items_stored(storage_type: &'static str) {
        counter!(
            "smg_db_items_stored",
            "storage_type" => storage_type
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 3: Engine load re-export
    // ========================================================================

    /// Re-export a worker's `GetLoads` snapshot as `smg_engine_*` gauges.
    ///
    /// Core gauges are per DP rank (`dp_rank` bounded by dp_size). PD gauges
    /// are emitted only for ranks that carry a `disagg` section, labeled by the
    /// engine-reported role (`prefill`/`decode`/`null`).
    pub fn record_engine_load(
        worker_url: &str,
        model_id: &str,
        response: &openai_protocol::worker::WorkerLoadResponse,
    ) {
        let worker = intern_string(worker_url);
        let model = intern_model_label(model_id);

        for load in &response.loads {
            let dp_rank = intern_string(&load.dp_rank.to_string());

            gauge!(
                "smg_engine_running_requests",
                "worker" => Arc::clone(&worker),
                "model" => Arc::clone(&model),
                "dp_rank" => Arc::clone(&dp_rank),
            )
            .set(load.num_running_reqs as f64);
            gauge!(
                "smg_engine_waiting_requests",
                "worker" => Arc::clone(&worker),
                "model" => Arc::clone(&model),
                "dp_rank" => Arc::clone(&dp_rank),
            )
            .set(load.num_waiting_reqs as f64);
            gauge!(
                "smg_engine_token_usage",
                "worker" => Arc::clone(&worker),
                "model" => Arc::clone(&model),
                "dp_rank" => Arc::clone(&dp_rank),
            )
            .set(load.token_usage);
            gauge!(
                "smg_engine_gen_throughput",
                "worker" => Arc::clone(&worker),
                "model" => Arc::clone(&model),
                "dp_rank" => Arc::clone(&dp_rank),
            )
            .set(load.gen_throughput);
            gauge!(
                "smg_engine_cache_hit_rate",
                "worker" => Arc::clone(&worker),
                "model" => Arc::clone(&model),
                "dp_rank" => Arc::clone(&dp_rank),
            )
            .set(load.cache_hit_rate);

            // PD gauges only when the engine reported a disagg section. Labeled
            // by dp_rank too, so DP ranks sharing a role don't overwrite.
            let Some(role) = load.disagg_mode.as_deref() else {
                continue;
            };
            let role = intern_string(role);
            if let Some(latency) = load.kv_transfer_latency_ms {
                gauge!(
                    "smg_engine_pd_kv_transfer_latency_ms",
                    "worker" => Arc::clone(&worker),
                    "role" => Arc::clone(&role),
                    "dp_rank" => Arc::clone(&dp_rank),
                )
                .set(latency);
            }
            if let Some(speed) = load.kv_transfer_speed_gb_s {
                gauge!(
                    "smg_engine_pd_kv_transfer_speed_gb_s",
                    "worker" => Arc::clone(&worker),
                    "role" => Arc::clone(&role),
                    "dp_rank" => Arc::clone(&dp_rank),
                )
                .set(speed);
            }
            if let Some(reqs) = load.prefill_queue_reqs {
                gauge!(
                    "smg_engine_pd_prefill_queue_reqs",
                    "worker" => Arc::clone(&worker),
                    "role" => Arc::clone(&role),
                    "dp_rank" => Arc::clone(&dp_rank),
                )
                .set(reqs as f64);
            }
            if let Some(reqs) = load.decode_queue_reqs {
                gauge!(
                    "smg_engine_pd_decode_queue_reqs",
                    "worker" => Arc::clone(&worker),
                    "role" => role,
                    "dp_rank" => dp_rank,
                )
                .set(reqs as f64);
            }
        }
    }

    // ========================================================================
    // Worker cleanup
    // ========================================================================

    pub fn remove_worker_metrics(worker_url: &str) {
        // Intern once, clone (cheap) for each metric
        let worker = intern_string(worker_url);

        gauge!("smg_worker_cb_consecutive_failures", "worker" => Arc::clone(&worker)).set(0.0);
        gauge!("smg_worker_cb_consecutive_successes", "worker" => Arc::clone(&worker)).set(0.0);
        gauge!("smg_worker_requests_active", "worker" => Arc::clone(&worker)).set(0.0);

        // Zero for these metrics have special valid meaning, thus we set to -1 temporarily
        // (and will remove them completely after https://github.com/metrics-rs/metrics/issues/653)
        gauge!("smg_worker_cb_state", "worker" => Arc::clone(&worker)).set(-1.0);
        gauge!("smg_worker_health", "worker" => worker).set(-1.0);
    }

    /// Sentinel-out `smg_engine_*` series for a removed worker.
    ///
    /// metrics-rs cannot delete series, so per the `remove_worker_metrics`
    /// convention we set each to -1 (an impossible value for these gauges, whose
    /// 0 is meaningful) until <https://github.com/metrics-rs/metrics/issues/653>.
    /// `dp_size` bounds the rank labels; the role label is unknown at teardown,
    /// so the full `dp_rank` × role (prefill/decode/null) space is cleared. The
    /// label set must exactly match `record_engine_load` (including `model` and
    /// `dp_rank`) or a fresh series is created instead of overwriting the live one.
    pub fn remove_engine_load_metrics(worker_url: &str, model_id: &str, dp_size: usize) {
        let worker = intern_string(worker_url);
        let model = intern_model_label(model_id);

        for rank in 0..dp_size.max(1) {
            let dp_rank = intern_string(&rank.to_string());
            for name in [
                "smg_engine_running_requests",
                "smg_engine_waiting_requests",
                "smg_engine_token_usage",
                "smg_engine_gen_throughput",
                "smg_engine_cache_hit_rate",
            ] {
                gauge!(
                    name,
                    "worker" => Arc::clone(&worker),
                    "model" => Arc::clone(&model),
                    "dp_rank" => Arc::clone(&dp_rank),
                )
                .set(-1.0);
            }

            // PD gauges are labeled {worker, role, dp_rank}; the role is unknown
            // at teardown, so clear every role for this rank.
            for role in ["prefill", "decode", "null"] {
                for name in [
                    "smg_engine_pd_kv_transfer_latency_ms",
                    "smg_engine_pd_kv_transfer_speed_gb_s",
                    "smg_engine_pd_prefill_queue_reqs",
                    "smg_engine_pd_decode_queue_reqs",
                ] {
                    gauge!(
                        name,
                        "worker" => Arc::clone(&worker),
                        "role" => role,
                        "dp_rank" => Arc::clone(&dp_rank),
                    )
                    .set(-1.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use metrics_exporter_prometheus::PrometheusBuilder;
    use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};

    use super::*;

    #[test]
    fn intern_bounded_label_caps_cardinality_with_sentinel() {
        let map: DashMap<String, Arc<str>> = DashMap::new();

        let a = intern_bounded_label(&map, 2, "m1");
        let b = intern_bounded_label(&map, 2, "m2");
        assert_eq!(map.len(), 2);

        // Repeats return the same interned Arc without growing the map.
        let a2 = intern_bounded_label(&map, 2, "m1");
        assert!(Arc::ptr_eq(&a, &a2));
        assert_eq!(map.len(), 2);

        // A distinct value past the cap collapses to the sentinel and does not
        // grow the map, so no new Prometheus series is minted for it.
        let c = intern_bounded_label(&map, 2, "m3");
        assert_eq!(&*c, BOUNDED_LABEL_SENTINEL);
        assert_eq!(map.len(), 2);

        // Already-admitted values still resolve normally after the cap is hit.
        let b2 = intern_bounded_label(&map, 2, "m2");
        assert!(Arc::ptr_eq(&b, &b2));
        assert_ne!(&*a, BOUNDED_LABEL_SENTINEL);
    }

    /// Run `f` under a thread-local Prometheus recorder and return the
    /// rendered `/metrics` text — the same scrape output the :29000 endpoint
    /// serves in production.
    fn render_with_recorder(f: impl FnOnce()) -> String {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, f);
        handle.render()
    }

    /// Core engine gauges share these labels for the snapshot fixtures.
    const CORE_LABELS: [&str; 3] = ["dp_rank=\"2\"", "model=\"m\"", "worker=\"http://w:1\""];

    /// Assert the rendered scrape has a line for `name` carrying every label in
    /// `labels` and ending in `value`. Label order is exporter-defined, so this
    /// matches on substrings rather than a fixed label set.
    fn assert_metric(rendered: &str, name: &str, labels: &[&str], value: &str) {
        let line = rendered
            .lines()
            .find(|l| l.starts_with(&format!("{name}{{")))
            .unwrap_or_else(|| panic!("metric {name} missing; rendered:\n{rendered}"));
        for label in labels {
            assert!(line.contains(label), "{name} missing label {label}: {line}");
        }
        assert!(
            line.ends_with(&format!(" {value}")),
            "{name} expected value {value}: {line}"
        );
    }

    #[test]
    fn record_engine_load_sets_core_gauges_per_dp_rank() {
        let response = WorkerLoadResponse {
            timestamp: "t".to_string(),
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: 2,
                num_running_reqs: 7,
                num_waiting_reqs: 3,
                token_usage: 0.5,
                gen_throughput: 42.0,
                cache_hit_rate: 0.25,
                ..Default::default()
            }],
        };

        let rendered = render_with_recorder(|| {
            Metrics::record_engine_load("http://w:1", "m", &response);
        });

        // The exporter renders labels in insertion order, so assert on the
        // metric line's components rather than a fixed label ordering.
        assert_metric(&rendered, "smg_engine_running_requests", &CORE_LABELS, "7");
        assert_metric(&rendered, "smg_engine_waiting_requests", &CORE_LABELS, "3");
        assert_metric(&rendered, "smg_engine_gen_throughput", &CORE_LABELS, "42");
        // PD gauges absent when no disagg section was reported.
        assert!(
            !rendered.contains("smg_engine_pd_"),
            "PD gauges must not appear without a disagg section; rendered:\n{rendered}"
        );
    }

    #[test]
    fn record_engine_load_sets_pd_gauges_when_disagg_present() {
        let response = WorkerLoadResponse {
            timestamp: "t".to_string(),
            dp_rank_count: 1,
            loads: vec![SchedulerLoadSnapshot {
                dp_rank: 0,
                disagg_mode: Some("prefill".to_string()),
                kv_transfer_latency_ms: Some(3.5),
                kv_transfer_speed_gb_s: Some(12.0),
                prefill_queue_reqs: Some(9),
                decode_queue_reqs: Some(4),
                ..Default::default()
            }],
        };

        let rendered = render_with_recorder(|| {
            Metrics::record_engine_load("http://w:1", "m", &response);
        });

        let pd_labels = ["role=\"prefill\"", "worker=\"http://w:1\"", "dp_rank=\"0\""];
        assert_metric(
            &rendered,
            "smg_engine_pd_kv_transfer_latency_ms",
            &pd_labels,
            "3.5",
        );
        assert_metric(
            &rendered,
            "smg_engine_pd_prefill_queue_reqs",
            &pd_labels,
            "9",
        );
        assert_metric(
            &rendered,
            "smg_engine_pd_decode_queue_reqs",
            &pd_labels,
            "4",
        );
    }

    #[test]
    fn cache_tree_setters_emit_per_model_gauges() {
        let rendered = render_with_recorder(|| {
            Metrics::set_cache_tree_chars("m", 120);
            Metrics::set_cache_tree_tokens("m", 64);
            Metrics::set_cache_tree_tenants("m", "string", 3);
            Metrics::set_cache_tree_tenants("m", "token", 2);
        });

        assert_metric(&rendered, "smg_cache_tree_chars", &["model=\"m\""], "120");
        assert_metric(&rendered, "smg_cache_tree_tokens", &["model=\"m\""], "64");
        // Two tenant series (one per tree kind); series order within the
        // family is exporter-defined, so match each line independently.
        for (tree, value) in [("string", "3"), ("token", "2")] {
            let label = format!("tree=\"{tree}\"");
            assert!(
                rendered.lines().any(|l| {
                    l.starts_with("smg_cache_tree_tenants{")
                        && l.contains("model=\"m\"")
                        && l.contains(&label)
                        && l.ends_with(&format!(" {value}"))
                }),
                "smg_cache_tree_tenants {tree} series missing; rendered:\n{rendered}"
            );
        }
    }

    #[test]
    fn test_prometheus_config_default() {
        let config = PrometheusConfig::default();
        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_prometheus_config_custom() {
        let config = PrometheusConfig {
            port: 8080,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        };
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_prometheus_config_clone() {
        let config = PrometheusConfig {
            port: 9090,
            host: "192.168.1.1".to_string(),
            duration_buckets: None,
        };
        let cloned = config.clone();
        assert_eq!(cloned.port, config.port);
        assert_eq!(cloned.host, config.host);
    }

    #[test]
    fn test_valid_ipv4_parsing() {
        let test_cases = vec!["127.0.0.1", "192.168.1.1", "0.0.0.0"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            assert!(matches!(ip_addr, IpAddr::V4(_)));
        }
    }

    #[test]
    fn test_valid_ipv6_parsing() {
        let test_cases = vec!["::1", "2001:db8::1", "::"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            assert!(matches!(ip_addr, IpAddr::V6(_)));
        }
    }

    #[test]
    fn test_invalid_ip_parsing() {
        let test_cases = vec!["invalid", "256.256.256.256", "hostname"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config
                .host
                .parse()
                .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));

            assert_eq!(ip_addr, IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
        }
    }

    #[test]
    fn test_socket_addr_creation() {
        let test_cases = vec![("127.0.0.1", 8080), ("0.0.0.0", 29000), ("::1", 9090)];

        for (host, port) in test_cases {
            let config = PrometheusConfig {
                port,
                host: host.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
            assert_eq!(socket_addr.ip().to_string(), host);
        }
    }

    #[test]
    fn test_socket_addr_with_different_ports() {
        let ports = vec![0, 80, 8080, 65535];

        for port in ports {
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
        }
    }

    #[test]
    fn test_duration_bucket_coverage() {
        let test_cases: [(f64, &str); 7] = [
            (0.0005, "sub-millisecond"),
            (0.005, "5ms"),
            (0.05, "50ms"),
            (1.0, "1s"),
            (10.0, "10s"),
            (60.0, "1m"),
            (240.0, "4m"),
        ];

        let buckets: [f64; 20] = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        for (duration, label) in test_cases {
            let bucket_found = buckets
                .iter()
                .any(|&b| (b - duration).abs() < 0.0001 || b > duration);
            assert!(bucket_found, "No bucket found for {duration} ({label})");
        }
    }

    #[test]
    fn test_duration_suffix_matcher() {
        let matcher = Matcher::Suffix(String::from("duration_seconds"));

        let _matching_metrics = [
            "request_duration_seconds",
            "response_duration_seconds",
            "smg_request_duration_seconds",
        ];

        let _non_matching_metrics = ["duration_total", "duration_seconds_total", "other_metric"];

        match matcher {
            Matcher::Suffix(suffix) => assert_eq!(suffix, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    #[test]
    fn test_prometheus_builder_configuration() {
        let _config = PrometheusConfig::default();

        let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
        let duration_bucket = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        assert_eq!(duration_bucket.len(), 20);

        match duration_matcher {
            Matcher::Suffix(s) => assert_eq!(s, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    #[test]
    fn test_upkeep_timeout_duration() {
        let timeout = Duration::from_secs(5 * 60);
        assert_eq!(timeout.as_secs(), 300);
    }

    #[test]
    fn test_custom_buckets_for_different_metrics() {
        let request_buckets = [0.001, 0.01, 0.1, 1.0, 10.0];
        let generate_buckets = [0.1, 0.5, 1.0, 5.0, 30.0, 60.0];

        assert_eq!(request_buckets.len(), 5);
        assert_eq!(generate_buckets.len(), 6);

        for i in 1..request_buckets.len() {
            assert!(request_buckets[i] > request_buckets[i - 1]);
        }

        for i in 1..generate_buckets.len() {
            assert!(generate_buckets[i] > generate_buckets[i - 1]);
        }
    }

    #[test]
    fn test_port_already_in_use() {
        let port = 29123;

        if let Ok(_listener) = TcpListener::bind(("127.0.0.1", port)) {
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
                duration_buckets: None,
            };

            assert_eq!(config.port, port);
        }
    }

    #[test]
    fn test_metrics_endpoint_accessibility() {
        let config = PrometheusConfig {
            port: 29000,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        };

        let ip_addr: IpAddr = config.host.parse().unwrap();
        let socket_addr = SocketAddr::new(ip_addr, config.port);

        assert_eq!(socket_addr.to_string(), "127.0.0.1:29000");
    }

    // ========================================================================
    // String interning tests
    // ========================================================================

    #[test]
    fn test_intern_string_returns_same_arc() {
        let s1 = intern_string("test_model");
        let s2 = intern_string("test_model");

        // Should return the same Arc (pointer equality)
        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(&*s1, "test_model");
    }

    #[test]
    fn test_intern_string_different_strings() {
        let s1 = intern_string("model_a");
        let s2 = intern_string("model_b");

        // Different strings should have different Arcs
        assert!(!Arc::ptr_eq(&s1, &s2));
        assert_eq!(&*s1, "model_a");
        assert_eq!(&*s2, "model_b");
    }

    #[test]
    fn test_intern_string_empty() {
        let s1 = intern_string("");
        let s2 = intern_string("");

        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(&*s1, "");
    }

    #[test]
    fn test_interner_size_grows() {
        let initial_size = interner_size();

        // Intern some unique strings
        let unique = format!("unique_test_string_{initial_size}");
        intern_string(&unique);

        assert!(interner_size() > initial_size);
    }

    #[test]
    fn test_bool_to_static_str() {
        assert_eq!(bool_to_static_str(true), "true");
        assert_eq!(bool_to_static_str(false), "false");
    }

    #[test]
    fn test_status_code_to_static_str() {
        // Common codes should return static strings
        assert_eq!(status_code_to_static_str(200), Some("200"));
        assert_eq!(status_code_to_static_str(404), Some("404"));
        assert_eq!(status_code_to_static_str(500), Some("500"));

        // Uncommon codes should return None
        assert_eq!(status_code_to_static_str(418), None);
        assert_eq!(status_code_to_static_str(999), None);
    }

    #[test]
    fn test_status_code_to_cow() {
        // Common codes should be borrowed
        let cow_200 = status_code_to_cow(200);
        assert!(matches!(cow_200, Cow::Borrowed(_)));
        assert_eq!(cow_200, "200");

        // Uncommon codes should be owned
        let cow_418 = status_code_to_cow(418);
        assert!(matches!(cow_418, Cow::Owned(_)));
        assert_eq!(cow_418, "418");
    }

    #[test]
    fn test_method_to_static_str() {
        assert_eq!(method_to_static_str("GET"), "GET");
        assert_eq!(method_to_static_str("POST"), "POST");
        assert_eq!(method_to_static_str("UNKNOWN"), "OTHER");
    }

    // ========================================================================
    // PD disaggregation metric tests
    // ========================================================================

    /// Run `f` with a Prometheus recorder installed thread-locally and return
    /// the rendered /metrics text. Mirrors the helper in `runtime_metrics`.
    fn with_test_recorder<T>(f: impl FnOnce() -> T) -> (String, T) {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let result = metrics::with_local_recorder(&recorder, f);
        (handle.render(), result)
    }

    #[test]
    fn test_record_pd_prefill_duration_emits_histogram() {
        let (rendered, ()) = with_test_recorder(|| {
            Metrics::record_pd_prefill_duration(
                metrics_labels::BACKEND_PD,
                "test-model",
                "vllm",
                Duration::from_millis(42),
            );
        });
        assert!(
            rendered.contains("smg_pd_prefill_duration_seconds_count{")
                && rendered.contains(r#"backend_type="pd""#)
                && rendered.contains(r#"model="test-model""#)
                && rendered.contains(r#"runtime="vllm""#),
            "prefill duration histogram not emitted; rendered:\n{rendered}"
        );
    }

    #[test]
    fn test_record_pd_kv_transfer_duration_emits_histogram() {
        let (rendered, ()) = with_test_recorder(|| {
            Metrics::record_pd_kv_transfer_duration(
                metrics_labels::BACKEND_PD,
                "m",
                "vllm",
                Duration::from_millis(7),
            );
        });
        assert!(
            rendered.contains("smg_pd_kv_transfer_duration_seconds_count"),
            "kv transfer histogram not emitted; rendered:\n{rendered}"
        );
    }

    #[test]
    fn test_record_pd_ttft_emits_histogram() {
        let (rendered, ()) = with_test_recorder(|| {
            Metrics::record_pd_ttft(
                metrics_labels::BACKEND_PD,
                "m",
                "sglang",
                Duration::from_millis(123),
            );
        });
        assert!(
            rendered.contains("smg_pd_ttft_seconds_count")
                && rendered.contains(r#"runtime="sglang""#),
            "pd ttft histogram not emitted; rendered:\n{rendered}"
        );
    }

    #[test]
    fn test_record_pd_kv_connector_mode_counts_by_mode() {
        let (rendered, ()) = with_test_recorder(|| {
            Metrics::record_pd_kv_connector_mode(metrics_labels::KV_CONNECTOR_MOONCAKE);
            Metrics::record_pd_kv_connector_mode(metrics_labels::KV_CONNECTOR_MOONCAKE);
            Metrics::record_pd_kv_connector_mode(metrics_labels::KV_CONNECTOR_NIXL);
        });
        assert!(
            rendered.contains(r#"smg_pd_kv_connector_mode_total{mode="mooncake"} 2"#),
            "mooncake connector counter wrong; rendered:\n{rendered}"
        );
        assert!(
            rendered.contains(r#"smg_pd_kv_connector_mode_total{mode="nixl"} 1"#),
            "nixl connector counter wrong; rendered:\n{rendered}"
        );
    }

    #[test]
    fn test_record_pd_failure_counters() {
        let (rendered, ()) = with_test_recorder(|| {
            Metrics::record_pd_bootstrap_failure();
            Metrics::record_pd_kv_transfer_failure();
            Metrics::record_pd_kv_transfer_failure();
        });
        assert!(
            rendered.contains("smg_pd_bootstrap_failures_total 1"),
            "bootstrap failure counter wrong; rendered:\n{rendered}"
        );
        assert!(
            rendered.contains("smg_pd_kv_transfer_failures_total 2"),
            "kv transfer failure counter wrong; rendered:\n{rendered}"
        );
    }
}
