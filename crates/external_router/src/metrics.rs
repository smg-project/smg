//! The metrics a router emits. They live here so the gateway and every
//! external router record the same names with the same labels.

use std::{
    borrow::Cow,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use dashmap::{mapref::entry::Entry, DashMap};
use metrics::{counter, histogram};
use once_cell::sync::Lazy;

const BOUNDED_LABEL_SENTINEL: &str = "other";

/// Max distinct client-supplied model labels retained before collapsing to the
/// sentinel. A gateway fronts far fewer real models than this; the cap only bites
/// on adversarial or unvalidated input.
const MAX_MODEL_LABELS: usize = 1024;

/// Max distinct client/model-controlled MCP tool-name labels.
const MAX_TOOL_LABELS: usize = 1024;

static MODEL_LABELS: Lazy<BoundedLabels> = Lazy::new(|| BoundedLabels::new(MAX_MODEL_LABELS));
static TOOL_LABELS: Lazy<BoundedLabels> = Lazy::new(|| BoundedLabels::new(MAX_TOOL_LABELS));
static BOUNDED_LABEL_SENTINEL_ARC: Lazy<Arc<str>> = Lazy::new(|| Arc::from(BOUNDED_LABEL_SENTINEL));

/// Intern a client-controlled label with a hard cardinality cap.
///
/// Distinct values beyond `cap` collapse to a shared sentinel, so untrusted input
/// (client-supplied model names, model-generated tool names) cannot grow the
/// interner — or the metric's Prometheus series set — without bound. Unlike an LRU,
/// admitted values are never evicted and re-admitted, which would keep minting new
/// series in the recorder even as the map churned.
/// A label set that admits at most `cap` distinct values; everything past
/// the cap maps to the sentinel. Admission reserves a slot before touching
/// the map, so a burst of distinct labels cannot overshoot the cap.
struct BoundedLabels {
    map: DashMap<String, Arc<str>>,
    admitted: AtomicUsize,
    cap: usize,
}

impl BoundedLabels {
    fn new(cap: usize) -> Self {
        Self {
            map: DashMap::new(),
            admitted: AtomicUsize::new(0),
            cap,
        }
    }

    fn intern(&self, s: &str) -> Arc<str> {
        if let Some(entry) = self.map.get(s) {
            return Arc::clone(entry.value());
        }
        let mut admitted = self.admitted.load(Ordering::Relaxed);
        loop {
            if admitted >= self.cap {
                return Arc::clone(&BOUNDED_LABEL_SENTINEL_ARC);
            }
            match self.admitted.compare_exchange_weak(
                admitted,
                admitted + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(now) => admitted = now,
            }
        }
        match self.map.entry(s.to_string()) {
            Entry::Occupied(entry) => {
                // Someone else admitted this label first; give the slot back.
                self.admitted.fetch_sub(1, Ordering::AcqRel);
                Arc::clone(entry.get())
            }
            Entry::Vacant(entry) => Arc::clone(&*entry.insert(Arc::from(s))),
        }
    }
}

/// Intern a client-supplied model label, bounded by [`MAX_MODEL_LABELS`].
pub fn intern_model_label(model_id: &str) -> Arc<str> {
    MODEL_LABELS.intern(model_id)
}

/// Intern a client/model-controlled MCP tool-name label, bounded by
/// [`MAX_TOOL_LABELS`].
pub fn intern_tool_label(tool_name: &str) -> Arc<str> {
    TOOL_LABELS.intern(tool_name)
}

pub const STREAMING_TRUE: &str = "true";
pub const STREAMING_FALSE: &str = "false";

pub const fn bool_to_static_str(b: bool) -> &'static str {
    if b {
        STREAMING_TRUE
    } else {
        STREAMING_FALSE
    }
}

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

/// Recorders for the metrics a router emits.
pub struct Metrics;

impl Metrics {
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
    /// Record MCP tool loop iteration
    pub fn record_mcp_tool_iteration(model_id: &str) {
        let model = intern_model_label(model_id);
        counter!(
            "smg_mcp_tool_iterations_total",
            "model" => model
        )
        .increment(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_labels_cap_cardinality_with_a_sentinel() {
        let labels = BoundedLabels::new(2);

        let a = labels.intern("m1");
        let b = labels.intern("m2");
        assert_eq!(labels.map.len(), 2);

        // Repeats return the same interned Arc without growing the map.
        let a2 = labels.intern("m1");
        assert!(Arc::ptr_eq(&a, &a2));
        assert_eq!(labels.map.len(), 2);

        // A distinct value past the cap collapses to the sentinel and does not
        // grow the map, so no new Prometheus series is minted for it.
        let c = labels.intern("m3");
        assert_eq!(&*c, BOUNDED_LABEL_SENTINEL);
        assert_eq!(labels.map.len(), 2);

        // Already-admitted values still resolve normally after the cap is hit.
        let b2 = labels.intern("m2");
        assert!(Arc::ptr_eq(&b, &b2));
        assert_ne!(&*a, BOUNDED_LABEL_SENTINEL);
    }
}
