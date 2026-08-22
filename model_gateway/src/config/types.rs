use std::collections::HashMap;

use openai_protocol::worker::HealthCheckConfig as ProtocolHealthCheckConfig;
pub use openai_protocol::worker::TransportMode;
use serde::{Deserialize, Serialize};
// Re-export storage config types from data_connector
pub use smg_data_connector::{
    HistoryBackend, OracleConfig, PostgresConfig, RedisConfig, SchemaConfig,
};

use super::{validation::ConfigValidator, ConfigResult};
use crate::{
    tenant::DEFAULT_TENANT_HEADER_NAME,
    worker::{ConnectionMode, RuntimeType},
};

/// Main router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub mode: RoutingMode,
    #[serde(default)]
    pub connection_mode: ConnectionMode,
    /// 0-based indices of `--prefill` URLs that belong to the long pool
    /// (get `pool=long` label for cache_aware_length). Empty = no long pool.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub long_prefill_indices: Vec<usize>,
    /// Explicit runtime for the startup workers (`--worker-urls`), set from
    /// `--backend` when the connection mode is ZMQ. The ZMQ handshake is shared
    /// across engine runtimes, so the wire protocol cannot be probed and must
    /// be declared up front; HTTP/gRPC workers keep auto-detection and ignore
    /// this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_worker_runtime_type: Option<RuntimeType>,
    /// DP engines per startup ZMQ worker: each `--worker-urls` ZMQ worker
    /// becomes a grouped worker whose handshake awaits this many engines on
    /// one socket set (`dp_size` on the worker spec, no rank). `None`/1 keeps
    /// today's one-engine workers; HTTP/gRPC workers ignore this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zmq_engine_count: Option<usize>,
    pub policy: PolicyConfig,
    /// Token positions at which serving engines retain reusable prefix state;
    /// cache-affinity policies hash request heads at the deepest applicable
    /// boundary. Ascending; empty disables boundary-based keying.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cache_boundaries: Vec<usize>,
    /// Per-request sticky-session routing (rid-lineage keys, header fallback).
    #[serde(default, alias = "sticky_sessions")]
    pub routing_key_override: RoutingKeyOverrideConfig,
    pub host: String,
    pub port: u16,
    /// Dedicated port for the isolated Kubernetes liveness/readiness/health
    /// probe listener. `None` means the dedicated listener is off; the probe
    /// routes always remain available on the main `port` regardless.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub health_check_port: Option<u16>,
    /// Explicit async runtime worker-thread count. `None` uses tokio's default
    /// (`available_parallelism()`), which already honors the cgroup CPU quota on
    /// Rust 1.95+ and is therefore container-aware. `Some` pins a count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_worker_threads: Option<usize>,
    pub max_payload_size: usize,
    /// Forward request bodies larger than this many bytes to the worker as a
    /// raw stream instead of buffering, when the route's policy needs no
    /// request text — or the request carries a valid routing hint header
    /// that stands in for it — and the worker applies no body mutation.
    /// Streamed bodies cannot be replayed, so those requests bypass
    /// router-level retries; bodies without a Content-Length header always
    /// buffer. `0` disables.
    #[serde(default)]
    pub stream_request_bodies_over: u64,
    /// Abort a streamed request body once the upstream sender has waited on
    /// the client for this many seconds (408). The clock pauses while the
    /// worker applies backpressure, so a slow worker read never trips it.
    /// `0` disables the watchdog.
    #[serde(default = "default_stream_body_stall_timeout_secs")]
    pub stream_body_stall_timeout_secs: u64,
    pub request_timeout_secs: u64,
    /// Idle timeout for pooled upstream connections. Must stay below the
    /// backend HTTP server's keep-alive timeout (vLLM and SGLang default to
    /// 5s), or the pool hands out connections the server has already closed
    /// and non-idempotent sends fail. `0` keeps idle connections forever.
    #[serde(default = "default_upstream_pool_idle_timeout_secs")]
    pub upstream_pool_idle_timeout_secs: u64,
    pub worker_startup_timeout_secs: u64,
    /// Grace period before the first worker-startup check fires. The engine is
    /// left alone for this long, then polled every
    /// `worker_startup_check_interval_secs`.
    #[serde(default)]
    pub worker_startup_delay_secs: u64,
    pub worker_startup_check_interval_secs: u64,
    /// Control-plane job queue: max pending jobs. Size to fleet scale so a
    /// discovery reconcile pass can enqueue every worker without blocking.
    #[serde(default = "default_job_queue_capacity")]
    pub job_queue_capacity: usize,
    /// Control-plane job queue: max jobs dispatched concurrently.
    #[serde(default = "default_job_queue_concurrency")]
    pub job_queue_concurrency: usize,
    #[serde(default = "default_load_monitor_interval_secs")]
    pub load_monitor_interval_secs: u64,
    /// Restore the conditional load-monitor poll gate: only poll worker groups
    /// when a load-aware routing policy, `engine_metrics`, or overload
    /// protection needs the data. Default `false` — the monitor polls every
    /// group unconditionally from registration onward. A load-aware policy is
    /// always fed regardless of this flag.
    #[serde(default)]
    pub disable_load_monitoring: bool,
    /// Enable absolute worker overload protection with the gateway default of
    /// `worker_overload_token_usage = 0.9` (KV token usage is engine-universal;
    /// a waiting-requests default would be workload-dependent, so that signal
    /// stays unset). Redundant when either explicit threshold below is set —
    /// those enable protection on their own, exactly as before this flag.
    #[serde(default)]
    pub worker_overload_protection: bool,
    /// Queued-request count at or above which a worker is considered
    /// overloaded and excluded from routing until the signal recovers; when all
    /// workers are overloaded, requests are shed immediately rather than
    /// queued. Evaluated once per ingested load report, never per request.
    /// `None` (default) disables this signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_overload_waiting_requests: Option<usize>,
    /// KV-cache token usage (0.0-1.0, averaged across DP ranks) at or above
    /// which a worker is considered overloaded — the same signal
    /// `balance_token_usage_threshold` reads, applied as an absolute per-worker
    /// ceiling instead of a fleet-relative spread. `None` (default) disables
    /// this signal; with both signals unset, overload protection is off and
    /// routing behaves exactly as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_overload_token_usage: Option<f64>,
    /// TTL in seconds for entries in the event-driven cache-aware positional
    /// indexer: entries neither stored to nor read by a query within this
    /// window are evicted by a periodic background prune. Bounds index growth
    /// when a backend stops emitting removal events (crash, stream downgrade).
    /// `None`/`0` disables the TTL pass (default, preserving unbounded growth).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_indexer_ttl_secs: Option<u64>,
    /// Capacity ceiling per model for the positional indexer, enforced by the
    /// same periodic prune: beyond it, oldest-touched entries are evicted down
    /// to 90% of the ceiling. `None`/`0` disables the ceiling (default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_indexer_max_entries: Option<usize>,
    /// Re-export engine `GetLoads` signals as `smg_engine_*` gauges, polling
    /// even when no load-aware routing policy is active. Decouples engine
    /// observability from routing.
    #[serde(default)]
    pub engine_metrics: bool,
    /// Global multimodal tensor transport mode (`inline` | `shm` | `auto` | `rdma`).
    /// Per-worker `WorkerSpec.multimodal_tensor_transport` overrides this; when
    /// unset, falls back to `SMG_MM_TENSOR_TRANSPORT`, then `inline`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multimodal_tensor_transport: Option<TransportMode>,
    /// Global minimum multimodal tensor size (bytes) before SHM transport is used.
    /// Per-worker `WorkerSpec.multimodal_shm_min_bytes` overrides this; falls back
    /// to `SMG_MM_SHM_MIN_BYTES`, then 64 KiB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multimodal_shm_min_bytes: Option<usize>,
    pub dp_aware: bool,
    #[serde(default)]
    pub dp_minimum_tokens_scheduler: bool,
    pub api_key: Option<String>,
    /// Per-tenant API keys for serving-path auth, layered on top of
    /// `api_key` rather than replacing it.
    #[serde(default)]
    pub tenant_api_keys: Vec<TenantApiKeyEntry>,
    pub discovery: Option<DiscoveryConfig>,
    pub metrics: Option<MetricsConfig>,
    pub trace_config: Option<TraceConfig>,
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub request_id_headers: Option<Vec<String>>,
    #[serde(default)]
    pub storage_context_headers: HashMap<String, String>,
    #[serde(default)]
    pub tenant_resolution: TenantResolutionConfig,
    /// Standing-concurrency cap; -1 disables. Each admission permit is
    /// held for the full response, including streaming bodies.
    pub max_concurrent_requests: i32,
    pub queue_size: usize,
    pub queue_timeout_secs: u64,
    /// Unset or 0 = no refill: `max_concurrent_requests` bounds standing
    /// concurrency alone.
    pub rate_limit_tokens_per_second: Option<i32>,
    /// Enable the priority-aware admission scheduler. When false (default),
    /// the legacy concurrency-limit middleware stays wired — zero behavior
    /// change for existing deployments.
    #[serde(default)]
    pub priority_scheduler_enabled: bool,
    /// Max priority class applied to tenants not listed in the scheduler
    /// YAML (`system` | `interactive` | `default` | `bulk`).
    #[serde(default = "default_priority_scheduler_max_class")]
    pub priority_scheduler_default_max_class: String,
    /// Optional path to the priority-scheduler YAML (per-class + per-tenant
    /// overrides). Absent → built-in defaults, empty tenant policy map.
    #[serde(default)]
    pub priority_scheduler_config: Option<String>,
    /// Cap on per-tenant scheduler metric label cardinality (top-N tenants
    /// by inflight; the remainder bucket under `tenant="other"`).
    #[serde(default = "default_priority_scheduler_tenant_metric_top_n")]
    pub priority_scheduler_tenant_metric_top_n: u32,
    /// Enable per-tenant LLM token/request rate limiting. When false
    /// (default), no rate limiter is constructed — zero behavior change
    /// for existing deployments.
    #[serde(default)]
    pub tenant_rate_limit_enabled: bool,
    /// Path to the tenant-rate-limit YAML (default + per-tenant policies,
    /// optionally further restricted per-model). Required when
    /// `tenant_rate_limit_enabled` is true.
    #[serde(default)]
    pub tenant_rate_limit_config: Option<String>,
    pub cors_allowed_origins: Vec<String>,
    pub retry: RetryConfig,
    pub circuit_breaker: CircuitBreakerConfig,
    /// When true, overrides retry.max_retries to 1
    #[serde(default)]
    pub disable_retries: bool,
    /// When true, overrides circuit_breaker.failure_threshold to u32::MAX
    #[serde(default)]
    pub disable_circuit_breaker: bool,
    pub health_check: HealthCheckConfig,
    #[serde(default)]
    pub enable_igw: bool,
    /// Can be a HuggingFace model ID or local path
    pub model_path: Option<String>,
    /// Overrides model_path tokenizer if provided
    pub tokenizer_path: Option<String>,
    pub chat_template: Option<String>,
    /// Alias → canonical model ID map. At worker creation, every alias whose
    /// canonical ID equals the worker's model ID is added to that model's
    /// `ModelCard.aliases`. This is the configuration entry for deployments
    /// whose workers are registered automatically (startup URLs or Kubernetes
    /// service discovery), where no caller supplies a `ModelCard` by hand.
    /// Lookup stays case-sensitive; entries name exact client-sent strings.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub model_aliases: HashMap<String, String>,
    /// Disable automatic tokenizer loading at startup and worker registration
    #[serde(default)]
    pub disable_tokenizer_autoload: bool,
    #[serde(default = "default_history_backend")]
    pub history_backend: HistoryBackend,
    /// Required when history_backend = "oracle"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oracle: Option<OracleConfig>,
    /// Required when history_backend = "postgres"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postgres: Option<PostgresConfig>,
    /// Required when history_backend = "redis"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redis: Option<RedisConfig>,
    /// For reasoning models (e.g., deepseek-r1, qwen3)
    pub reasoning_parser: Option<String>,
    /// For tool-call interactions
    pub tool_call_parser: Option<String>,
    #[serde(default)]
    pub tokenizer_cache: TokenizerCacheConfig,
    /// Server TLS certificate (PEM)
    #[serde(skip)]
    pub server_cert: Option<Vec<u8>>,
    /// Server TLS private key (PEM)
    #[serde(skip)]
    pub server_key: Option<Vec<u8>>,
    /// Combined certificate + key in PEM format, loaded from client_cert_path and client_key_path during config creation
    #[serde(skip)]
    pub client_identity: Option<Vec<u8>>,
    /// PEM format, loaded from ca_cert_paths during config creation
    #[serde(default)]
    pub ca_certificates: Vec<Vec<u8>>,
    /// Speak HTTP/2 to workers via prior knowledge (h2c on cleartext) on all
    /// engine-directed connections — request dispatch and health/probe traffic
    /// alike — multiplexing every request to a worker over one connection
    /// instead of one TCP connection per in-flight request. Requires every
    /// HTTP worker to serve HTTP/2 without an upgrade handshake.
    #[serde(default)]
    pub upstream_http2: bool,
    /// Loaded from mcp_config_path during config creation
    #[serde(skip)]
    pub mcp_config: Option<smg_mcp::McpConfig>,
    /// Enable WASM support
    #[serde(default)]
    pub enable_wasm: bool,
    /// Path to a WASM component implementing storage hooks.
    /// When set, wraps all storage backends with hook-based interceptors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_hook_wasm_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TenantResolutionConfig {
    pub trust_tenant_header: bool,
    pub tenant_header_name: String,
}

/// A single tenant-scoped API key for serving-path authentication.
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TenantApiKeyEntry {
    /// Resolves to tenant key `auth:<tenant_id>`, e.g. `team-red`.
    pub tenant_id: String,
    pub key: String,
}

impl std::fmt::Debug for TenantApiKeyEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantApiKeyEntry")
            .field("tenant_id", &self.tenant_id)
            .field("key", &"<redacted>")
            .finish()
    }
}

impl Default for TenantResolutionConfig {
    fn default() -> Self {
        Self {
            trust_tenant_header: false,
            tenant_header_name: DEFAULT_TENANT_HEADER_NAME.to_string(),
        }
    }
}

/// Tokenizer cache configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerCacheConfig {
    /// Whole-string exact match cache
    #[serde(default = "default_enable_l0")]
    pub enable_l0: bool,
    #[serde(default = "default_l0_max_entries")]
    pub l0_max_entries: usize,
    /// Prefix matching at fixed boundaries
    #[serde(default = "default_enable_l1")]
    pub enable_l1: bool,
    #[serde(default = "default_l1_max_memory")]
    pub l1_max_memory: usize,
}

fn default_load_monitor_interval_secs() -> u64 {
    10
}

fn default_job_queue_capacity() -> usize {
    1000
}

fn default_job_queue_concurrency() -> usize {
    200
}

fn default_enable_l0() -> bool {
    false
}

fn default_l0_max_entries() -> usize {
    10_000
}

fn default_enable_l1() -> bool {
    false
}

fn default_l1_max_memory() -> usize {
    50 * 1024 * 1024 // 50MB
}

impl TokenizerCacheConfig {
    /// Returns Some(self) if any caching is enabled, None otherwise.
    /// Use this when passing cache config to tokenizer registration workflow.
    pub fn to_option(&self) -> Option<Self> {
        if self.enable_l0 || self.enable_l1 {
            Some(self.clone())
        } else {
            None
        }
    }
}

impl Default for TokenizerCacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: default_enable_l0(),
            l0_max_entries: default_l0_max_entries(),
            enable_l1: default_enable_l1(),
            l1_max_memory: default_l1_max_memory(),
        }
    }
}

fn default_priority_scheduler_max_class() -> String {
    "default".to_string()
}

fn default_priority_scheduler_tenant_metric_top_n() -> u32 {
    32
}

fn default_history_backend() -> HistoryBackend {
    HistoryBackend::Memory
}

/// Routing mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RoutingMode {
    #[serde(rename = "regular")]
    Regular { worker_urls: Vec<String> },
    #[serde(rename = "prefill_decode")]
    PrefillDecode {
        /// With optional bootstrap ports
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill_policy: Option<PolicyConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        decode_policy: Option<PolicyConfig>,
    },
    #[serde(rename = "encode_prefill_decode")]
    EncodePrefillDecode {
        /// Encode worker urls (run the vision tower); optional Mooncake
        /// bootstrap ports.
        encode_urls: Vec<(String, Option<u16>)>,
        /// Prefill worker urls with optional bootstrap ports.
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encode_policy: Option<PolicyConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill_policy: Option<PolicyConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        decode_policy: Option<PolicyConfig>,
    },
    #[serde(rename = "openai")]
    OpenAI { worker_urls: Vec<String> },
    #[serde(rename = "anthropic")]
    Anthropic { worker_urls: Vec<String> },
    #[serde(rename = "gemini")]
    Gemini { worker_urls: Vec<String> },
}

impl RoutingMode {
    pub fn is_pd_mode(&self) -> bool {
        matches!(self, RoutingMode::PrefillDecode { .. })
    }

    pub fn worker_count(&self) -> usize {
        match self {
            RoutingMode::Regular { worker_urls } => worker_urls.len(),
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => prefill_urls.len() + decode_urls.len(),
            RoutingMode::EncodePrefillDecode {
                encode_urls,
                prefill_urls,
                decode_urls,
                ..
            } => encode_urls.len() + prefill_urls.len() + decode_urls.len(),
            RoutingMode::OpenAI { worker_urls } => worker_urls.len(),
            RoutingMode::Anthropic { worker_urls } => worker_urls.len(),
            RoutingMode::Gemini { worker_urls } => worker_urls.len(),
        }
    }

    /// Get the effective prefill policy for PD mode
    /// Falls back to the main policy if no specific prefill policy is set
    pub fn get_prefill_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::PrefillDecode { prefill_policy, .. }
            | RoutingMode::EncodePrefillDecode { prefill_policy, .. } => {
                prefill_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }

    /// Get the effective decode policy for PD mode
    /// Falls back to the main policy if no specific decode policy is set
    pub fn get_decode_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::PrefillDecode { decode_policy, .. }
            | RoutingMode::EncodePrefillDecode { decode_policy, .. } => {
                decode_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }

    /// Get the effective encode policy for EPD mode. The default is
    /// consistent_hashing because encode routing is item-cache affinity, not the
    /// request-level main policy.
    pub fn get_encode_policy<'a>(
        &'a self,
        default_encode_policy: &'a PolicyConfig,
    ) -> &'a PolicyConfig {
        match self {
            RoutingMode::EncodePrefillDecode { encode_policy, .. } => {
                encode_policy.as_ref().unwrap_or(default_encode_policy)
            }
            _ => default_encode_policy,
        }
    }
}

/// Assignment mode for manual policy when encountering a new routing key
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ManualAssignmentMode {
    /// Random selection (default)
    #[default]
    Random,
    /// Select worker with minimum running requests
    MinLoad,
    /// Select worker with minimum active routing keys
    MinGroup,
    /// Delegate the first assignment for a key to the underlying routing
    /// policy, then pin. With `--policy manual` (no underlying policy to
    /// delegate to) this falls back to min-load.
    Delegate,
}

/// Per-request sticky-routing override: when a sticky key is present, any
/// eligible policy routes via manual sticky-map semantics. Reuses the manual
/// policy knobs for the sticky map; eviction defaults match the manual policy so
/// config-file users with only `enabled: true` still get TTL eviction (no leak).
///
/// Key priority is fixed: a key derived from the typed body's `rid` (per-turn
/// `_t<n>` and per-retry `_r<n>` suffixes stripped, so every turn of a
/// conversation shares one key) wins over the routing-key headers; the first
/// configured header carrying a valid value is the fallback when no rid is
/// present. Raw-streamed requests have no readable body and therefore derive
/// keys from the headers only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingKeyOverrideConfig {
    /// When false, policies are used unchanged.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_manual_eviction_interval_secs")]
    pub eviction_interval_secs: u64,
    #[serde(
        default = "default_manual_max_idle_secs",
        alias = "sticky_key_idle_secs"
    )]
    pub max_idle_secs: u64,
    /// Defaults to `delegate`: first-seen keys route via the underlying
    /// policy, then pin.
    #[serde(default = "default_override_assignment_mode")]
    pub assignment_mode: ManualAssignmentMode,
    /// Ordered header names consulted for the routing key; the first header
    /// present with a valid value (non-empty UTF-8 within the byte cap) wins.
    /// When the override is enabled, header keys get the same per-turn /
    /// per-retry suffix stripping as rid-derived keys.
    #[serde(default = "default_routing_key_headers")]
    pub headers: Vec<String>,
}

fn default_override_assignment_mode() -> ManualAssignmentMode {
    ManualAssignmentMode::Delegate
}

fn default_routing_key_headers() -> Vec<String> {
    vec!["x-smg-routing-key".to_string()]
}

impl Default for RoutingKeyOverrideConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            eviction_interval_secs: default_manual_eviction_interval_secs(),
            max_idle_secs: default_manual_max_idle_secs(),
            assignment_mode: default_override_assignment_mode(),
            headers: default_routing_key_headers(),
        }
    }
}

/// Under-layer index the cache_aware policy keeps per model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CacheIndexKind {
    /// Radix prefix tree (default).
    #[default]
    Tree,
    /// TTL'd exact-match placement map keyed on quantized request heads
    /// (`cache_boundaries`); the radix tree is neither consulted nor populated.
    Hash,
}

/// Policy configuration for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PolicyConfig {
    #[serde(rename = "random")]
    Random,

    #[serde(rename = "round_robin")]
    RoundRobin,

    /// Forward every request to the single backend with no load balancing,
    /// load monitoring, or KV-event subscription. Intended for single-worker
    /// gateways. See `policies/passthrough.rs`.
    #[serde(rename = "passthrough")]
    Passthrough,

    #[serde(rename = "cache_aware")]
    CacheAware {
        /// Minimum matched-prefix share before a request pins to a holder.
        #[serde(alias = "cache_match_threshold")]
        cache_threshold: f32,
        /// Spill gate, absolute part: the selected worker spills to
        /// least-loaded when its load exceeds the healthy-fleet mean by this
        /// many requests AND by `balance_rel_threshold`.
        #[serde(alias = "spill_abs_threshold")]
        balance_abs_threshold: usize,
        /// Spill gate, relative part (multiple of the healthy-fleet mean);
        /// fires only together with `balance_abs_threshold`.
        #[serde(alias = "spill_rel_threshold")]
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        #[serde(default = "default_block_size")]
        block_size: usize,
        /// KV-usage spread (hottest minus coldest backend, 0.0–1.0) above which
        /// cache affinity is abandoned for shortest-queue. `>= 1.0` disables.
        #[serde(default = "default_balance_token_usage_threshold")]
        balance_token_usage_threshold: f32,
        /// Backend KV-utilization ceiling (0.0–1.0): a single engine above it
        /// triggers shedding regardless of spread. `>= 1.0` disables (default).
        #[serde(default = "default_balance_token_usage_threshold")]
        overload_token_usage_threshold: f32,
        /// Anti-hotspot decay for event-driven overlap credit: each
        /// candidate's overlap score is divided by `1 + overlap_decay * x`,
        /// with `x` the worker's waiting-prefill backlog (blocks in excess of
        /// the candidate minimum) per request block. `0.0` disables (default).
        #[serde(default = "default_overlap_decay")]
        overlap_decay: f32,
        /// Softmax temperature for event-driven selection over min-max
        /// normalized scores. `0.0` (default) is exact argmax with the
        /// existing tie-breaks; larger values spread picks across candidates.
        #[serde(default = "default_selection_temperature")]
        selection_temperature: f32,
        /// Index under-layer: `tree` (default) or `hash` (TTL'd exact-match
        /// placement map over `cache_boundaries` heads).
        #[serde(default)]
        cache_index: CacheIndexKind,
        /// Seconds a cache-affinity placement stays routable; should
        /// approximate serving-engine cache retention.
        #[serde(default = "default_cache_ttl_secs")]
        cache_ttl_secs: u64,
        /// Boundary token positions for the hash index (copied from the
        /// shared `cache_boundaries` config).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        cache_boundaries: Vec<usize>,
    },

    /// Cache-aware length policy: a full superset of `cache_aware` that adds
    /// a long/short pool split on the no-cache branch, driven by the `pool`
    /// worker label (`pool=long` → long pool, otherwise short pool). Inherits
    /// all cache_aware features (string tree, token tree, event-driven
    /// routing, hash index, mesh sync, KV pressure). See
    /// `policies/cache_aware_length.rs`.
    #[serde(rename = "cache_aware_length")]
    CacheAwareLength {
        // --- Inherited from cache_aware ---
        #[serde(alias = "cache_match_threshold")]
        #[serde(default = "default_cal_cache_threshold")]
        cache_threshold: f32,
        #[serde(alias = "spill_abs_threshold")]
        #[serde(default = "default_cal_balance_abs_threshold")]
        balance_abs_threshold: usize,
        #[serde(alias = "spill_rel_threshold")]
        #[serde(default = "default_cal_balance_rel_threshold")]
        balance_rel_threshold: f32,
        #[serde(default = "default_cal_eviction_interval_secs")]
        eviction_interval_secs: u64,
        #[serde(default = "default_cal_max_tree_size")]
        max_tree_size: usize,
        #[serde(default = "default_block_size")]
        block_size: usize,
        #[serde(default = "default_balance_token_usage_threshold")]
        balance_token_usage_threshold: f32,
        #[serde(default = "default_balance_token_usage_threshold")]
        overload_token_usage_threshold: f32,
        #[serde(default = "default_overlap_decay")]
        overlap_decay: f32,
        #[serde(default = "default_selection_temperature")]
        selection_temperature: f32,
        #[serde(default)]
        cache_index: CacheIndexKind,
        #[serde(default = "default_cache_ttl_secs")]
        cache_ttl_secs: u64,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        cache_boundaries: Vec<usize>,
        // --- Length-specific ---
        #[serde(default = "default_cal_chars_per_token")]
        chars_per_token: usize,
        #[serde(default = "default_cal_long_prefill_threshold")]
        long_prefill_threshold: usize,
        #[serde(default = "default_cal_long_pool_max_load")]
        long_pool_max_load: usize,
        #[serde(default = "default_cal_short_pool_max_load")]
        short_pool_max_load: usize,
    },

    /// Power-of-two choices policy: samples two workers and routes to the one
    /// with the lower expected wait, scored like `least_load`
    /// (`(queued_tokens + inflight_tokens) / throughput + kv_pressure_weight * k/(1-k)`).
    /// TODO: Implement per-policy load monitoring intervals.
    /// Currently, load_check_interval_secs is populated from RouterConfig.load_monitor_interval_secs,
    /// but WorkerMonitor does not yet use per-policy intervals. This field is reserved for
    /// future support of different polling cadences per policy.
    #[serde(rename = "power_of_two")]
    PowerOfTwo { load_check_interval_secs: u64 },

    /// Least-(token-)work policy: routes to the worker minimizing the expected
    /// wait `(queued_tokens + inflight_tokens) / throughput + kv_pressure_weight * k/(1-k)`
    /// — token-work drain time plus a convex KV-cache pressure barrier, computed
    /// from the load monitor with in-flight correction. See `policies/least_load.rs`.
    #[serde(rename = "least_load")]
    LeastLoad {
        /// TODO: Implement per-policy load monitoring intervals.
        /// Currently, load_check_interval_secs is populated from RouterConfig.load_monitor_interval_secs,
        /// but WorkerMonitor does not yet use per-policy intervals. This field is reserved for
        /// future support of different polling cadences per policy.
        #[serde(default = "default_least_load_interval")]
        load_check_interval_secs: u64,
        /// KV-pressure weight `λ_t` (seconds): the time-cost of KV contention,
        /// commensurate with the expected-queue-wait term.
        #[serde(default = "default_least_load_kv_pressure_weight")]
        kv_pressure_weight: f64,
        /// Mean prefill length (tokens) used to estimate in-flight token-work
        /// when a request's token count is unknown at routing time.
        #[serde(default = "default_least_load_mean_prefill")]
        mean_prefill_tokens: u32,
        /// Fallback generation throughput (tokens/s) for the expected-wait term
        /// when a backend reports no live `gen_throughput`. Set to the fleet's
        /// per-replica generation rate; co-tunes with `kv_pressure_weight`.
        #[serde(default = "default_least_load_throughput")]
        default_throughput: f64,
        /// Per-worker waiting-queue cap: skip workers whose reported waiting
        /// requests, plus dispatches since their last poll, have reached this
        /// count; when every candidate is at the cap, selection fails and the
        /// request falls to the router's admission queue. `0` disables. Set
        /// below the engine's max batch size.
        #[serde(default)]
        max_waiting_requests: u32,
    },

    #[serde(rename = "bucket")]
    Bucket {
        /// Absolute load difference threshold for load balancing
        #[serde(alias = "spill_abs_threshold")]
        balance_abs_threshold: usize,
        /// Relative load ratio threshold for load balancing
        #[serde(alias = "spill_rel_threshold")]
        balance_rel_threshold: f32,
        /// Interval between bucket boundary adjustment cycles (seconds)
        bucket_adjust_interval_secs: usize,
    },

    /// Manual routing policy with sticky sessions using DashMap.
    /// - X-SMG-Routing-Key: Routes to a cached worker or assigns a new one
    /// - Provides true sticky sessions with zero key redistribution on worker add
    /// - Falls back to random selection if no routing key is provided
    /// - Supports LRU eviction when cache size exceeds max_entries
    #[serde(rename = "manual")]
    Manual {
        /// Interval between TTL eviction cycles (seconds, default: 60)
        #[serde(default = "default_manual_eviction_interval_secs")]
        eviction_interval_secs: u64,
        /// Maximum idle time before a key is evicted (seconds, default:
        /// 14400 = 4 hours)
        #[serde(
            default = "default_manual_max_idle_secs",
            alias = "sticky_key_idle_secs"
        )]
        max_idle_secs: u64,
        /// Assignment mode for new routing keys (default: random)
        #[serde(default)]
        assignment_mode: ManualAssignmentMode,
    },

    /// Consistent hashing policy using hash ring for session affinity:
    /// - X-SMG-Target-Worker: Direct routing to a specific worker by URL
    /// - X-SMG-Routing-Key: Consistent hash routing for session affinity
    /// - Provides O(log n) lookup with minimal redistribution (~1/N keys) on topology change
    #[serde(rename = "consistent_hashing")]
    ConsistentHashing,

    /// Prefix hash policy for KV cache-aware load balancing.
    /// A lightweight alternative to cache_aware radix tree.
    /// Routes requests based on prefix token hash for cache locality.
    /// - Uses consistent hash ring with bounded load balancing
    /// - Diverts to the least loaded worker when the hashed one is overloaded
    /// - O(log n) lookup instead of O(prefix_len) radix tree traversal
    #[serde(rename = "prefix_hash")]
    PrefixHash {
        /// Number of prefix tokens to hash, or four times as many characters
        /// of the prompt when the request is untokenized (default: 256)
        #[serde(default = "default_prefix_token_count")]
        prefix_token_count: usize,
        /// Relative load threshold - a worker is overloaded when its load
        /// exceeds both avg * factor and the absolute margin (default: 1.25)
        #[serde(default = "default_load_factor")]
        load_factor: f64,
        /// Absolute load difference over average a worker must also exceed
        /// before it counts as overloaded (default: 10)
        #[serde(default = "default_prefix_hash_balance_abs_threshold")]
        balance_abs_threshold: usize,
        /// Resolved copy of `RouterConfig::cache_boundaries`: ascending token
        /// positions; requests hash at the deepest boundary they reach.
        /// Empty = hash at `prefix_token_count` only.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        cache_boundaries: Vec<usize>,
    },
}

fn default_block_size() -> usize {
    16
}

fn default_balance_token_usage_threshold() -> f32 {
    1.0
}

fn default_overlap_decay() -> f32 {
    0.0
}

fn default_selection_temperature() -> f32 {
    0.0
}

fn default_cache_ttl_secs() -> u64 {
    180
}

// cache_aware_length defaults (kept aligned with CacheAwareLengthConfig::default).
fn default_cal_cache_threshold() -> f32 {
    0.3
}
fn default_cal_balance_abs_threshold() -> usize {
    32
}
fn default_cal_balance_rel_threshold() -> f32 {
    1.1
}
fn default_cal_eviction_interval_secs() -> u64 {
    30
}
fn default_cal_max_tree_size() -> usize {
    10000
}
fn default_cal_chars_per_token() -> usize {
    4
}
fn default_cal_long_prefill_threshold() -> usize {
    100_000
}
fn default_cal_long_pool_max_load() -> usize {
    4
}
fn default_cal_short_pool_max_load() -> usize {
    32
}

fn default_prefix_token_count() -> usize {
    256
}

fn default_load_factor() -> f64 {
    1.25
}

fn default_upstream_pool_idle_timeout_secs() -> u64 {
    3
}

fn default_stream_body_stall_timeout_secs() -> u64 {
    300
}

fn default_prefix_hash_balance_abs_threshold() -> usize {
    10
}

fn default_manual_eviction_interval_secs() -> u64 {
    60
}

fn default_manual_max_idle_secs() -> u64 {
    4 * 3600
}

fn default_least_load_interval() -> u64 {
    10
}

fn default_least_load_kv_pressure_weight() -> f64 {
    0.15
}

fn default_least_load_mean_prefill() -> u32 {
    1024
}

fn default_least_load_throughput() -> f64 {
    2000.0
}

impl PolicyConfig {
    pub fn name(&self) -> &'static str {
        match self {
            PolicyConfig::Random => "random",
            PolicyConfig::RoundRobin => "round_robin",
            PolicyConfig::Passthrough => "passthrough",
            PolicyConfig::CacheAware { .. } => "cache_aware",
            PolicyConfig::CacheAwareLength { .. } => "cache_aware_length",
            PolicyConfig::PowerOfTwo { .. } => "power_of_two",
            PolicyConfig::LeastLoad { .. } => "least_load",
            PolicyConfig::Bucket { .. } => "bucket",
            PolicyConfig::Manual { .. } => "manual",
            PolicyConfig::ConsistentHashing => "consistent_hashing",
            PolicyConfig::PrefixHash { .. } => "prefix_hash",
        }
    }
}

/// Service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub enabled: bool,
    /// None = all namespaces
    pub namespace: Option<String>,
    pub port: u16,
    pub check_interval_secs: u64,
    /// Regular mode
    pub selector: HashMap<String, String>,
    /// EPD mode encode
    #[serde(default)]
    pub encode_selector: HashMap<String, String>,
    /// PD mode prefill
    pub prefill_selector: HashMap<String, String>,
    /// PD mode decode
    pub decode_selector: HashMap<String, String>,
    pub bootstrap_port_annotation: String,
    /// Annotation listing a pod's worker data ports (comma-separated).
    /// Absent on a pod = single worker at `port`.
    #[serde(default = "default_worker_ports_annotation")]
    pub worker_ports_annotation: String,
    /// Router node discovery for HA (Kubernetes label selector)
    #[serde(default)]
    pub router_selector: HashMap<String, String>,
    /// Annotation key to read mesh port from Router Pods
    #[serde(default = "default_router_mesh_port_annotation")]
    pub router_mesh_port_annotation: String,
    /// Source for per-worker model_id override: "namespace", "label:<key>", or "annotation:<key>"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id_source: Option<String>,
}

fn default_router_mesh_port_annotation() -> String {
    "sglang.ai/mesh-port".to_string()
}

fn default_worker_ports_annotation() -> String {
    "smg.ai/worker-ports".to_string()
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            namespace: None,
            port: 8000,
            check_interval_secs: 120,
            selector: HashMap::new(),
            encode_selector: HashMap::new(),
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            worker_ports_annotation: default_worker_ports_annotation(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: default_router_mesh_port_annotation(),
            model_id_source: None,
        }
    }
}

/// Retry configuration for request handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f32,
    /// D' = D * (1 + U[-j, +j]) where j is jitter factor
    #[serde(default = "default_retry_jitter_factor")]
    pub jitter_factor: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 50,
            max_backoff_ms: 30000,
            backoff_multiplier: 1.5,
            jitter_factor: 0.2,
        }
    }
}

fn default_retry_jitter_factor() -> f32 {
    0.2
}

/// Health check configuration for worker monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_secs: u64,
    pub check_interval_secs: u64,
    pub endpoint: String,
    pub disable_health_check: bool,
    /// Let workers recover after prolonged failure: removal re-enters them
    /// through service discovery once their engine returns.
    #[serde(default, alias = "worker_auto_recovery")]
    pub remove_unhealthy_workers: bool,
    /// Seconds to keep a Ready worker in `Draining` after `RemoveWorker`
    /// is submitted before the registry entry is removed. Lets in-flight
    /// requests complete naturally. Set to `0` to skip draining and
    /// remove immediately. Default: 5.
    #[serde(default = "default_drain_settle_secs")]
    pub drain_settle_secs: u64,
}

fn default_drain_settle_secs() -> u64 {
    5
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 2,
            timeout_secs: 5,
            check_interval_secs: 60,
            endpoint: "/health".to_string(),
            disable_health_check: false,
            remove_unhealthy_workers: false,
            drain_settle_secs: default_drain_settle_secs(),
        }
    }
}

impl HealthCheckConfig {
    /// Convert to protocol-level health check config (without endpoint).
    pub fn to_protocol_config(&self) -> ProtocolHealthCheckConfig {
        ProtocolHealthCheckConfig {
            timeout_secs: self.timeout_secs,
            check_interval_secs: self.check_interval_secs,
            success_threshold: self.success_threshold,
            failure_threshold: self.failure_threshold,
            disable_health_check: self.disable_health_check,
            drain_settle_secs: self.drain_settle_secs,
        }
    }
}

/// Circuit breaker configuration for worker reliability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_duration_secs: u64,
    pub window_duration_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 10,
            success_threshold: 3,
            timeout_duration_secs: 60,
            window_duration_secs: 120,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub port: u16,
    pub host: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "0.0.0.0".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    pub enable_trace: bool,
    pub otlp_traces_endpoint: String,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            enable_trace: false,
            otlp_traces_endpoint: "localhost:4317".to_string(),
        }
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::Random,
            cache_boundaries: Vec::new(),
            long_prefill_indices: Vec::new(),
            routing_key_override: RoutingKeyOverrideConfig::default(),
            host: "0.0.0.0".to_string(),
            port: 3001,
            health_check_port: None,
            runtime_worker_threads: None,
            max_payload_size: 536_870_912, // 512MB
            stream_request_bodies_over: 0,
            stream_body_stall_timeout_secs: default_stream_body_stall_timeout_secs(),
            request_timeout_secs: 1800, // 30 minutes
            upstream_pool_idle_timeout_secs: default_upstream_pool_idle_timeout_secs(),
            worker_startup_timeout_secs: 1800, // 30 minutes for large model loading
            worker_startup_delay_secs: 0,
            worker_startup_check_interval_secs: 30,
            job_queue_capacity: default_job_queue_capacity(),
            job_queue_concurrency: default_job_queue_concurrency(),
            load_monitor_interval_secs: 10,
            disable_load_monitoring: false,
            worker_overload_protection: false,
            worker_overload_waiting_requests: None,
            worker_overload_token_usage: None,
            kv_indexer_ttl_secs: None,
            kv_indexer_max_entries: None,
            engine_metrics: false,
            multimodal_tensor_transport: None,
            multimodal_shm_min_bytes: None,
            dp_aware: false,
            dp_minimum_tokens_scheduler: false,
            api_key: None,
            tenant_api_keys: Vec::new(),
            discovery: None,
            metrics: None,
            trace_config: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            storage_context_headers: HashMap::new(),
            tenant_resolution: TenantResolutionConfig::default(),
            max_concurrent_requests: -1,
            queue_size: 100,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            priority_scheduler_enabled: false,
            priority_scheduler_default_max_class: default_priority_scheduler_max_class(),
            priority_scheduler_config: None,
            priority_scheduler_tenant_metric_top_n: default_priority_scheduler_tenant_metric_top_n(
            ),
            tenant_rate_limit_enabled: false,
            tenant_rate_limit_config: None,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: HealthCheckConfig::default(),
            enable_igw: false,
            connection_mode: ConnectionMode::Http,
            startup_worker_runtime_type: None,
            zmq_engine_count: None,
            model_path: None,
            tokenizer_path: None,
            chat_template: None,
            disable_tokenizer_autoload: false,
            model_aliases: HashMap::new(),
            history_backend: default_history_backend(),
            oracle: None,
            postgres: None,
            redis: None,
            reasoning_parser: None,
            tool_call_parser: None,
            tokenizer_cache: TokenizerCacheConfig::default(),
            client_identity: None,
            ca_certificates: vec![],
            upstream_http2: false,
            mcp_config: None,
            enable_wasm: false,
            storage_hook_wasm_path: None,
            server_cert: None,
            server_key: None,
        }
    }
}

impl RouterConfig {
    /// Create a new configuration with mode and policy
    pub fn new(mode: RoutingMode, policy: PolicyConfig) -> Self {
        Self {
            mode,
            policy,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> ConfigResult<()> {
        ConfigValidator::validate(self)
    }

    /// Get the routing mode type as a string
    pub fn mode_type(&self) -> &'static str {
        match self.mode {
            RoutingMode::Regular { .. } => "regular",
            RoutingMode::PrefillDecode { .. } => "prefill_decode",
            RoutingMode::EncodePrefillDecode { .. } => "encode_prefill_decode",
            RoutingMode::OpenAI { .. } => "openai",
            RoutingMode::Anthropic { .. } => "anthropic",
            RoutingMode::Gemini { .. } => "gemini",
        }
    }

    /// Check if service discovery is enabled
    pub fn has_service_discovery(&self) -> bool {
        self.discovery.as_ref().is_some_and(|d| d.enabled)
    }

    /// Check if metrics are enabled
    pub fn has_metrics(&self) -> bool {
        self.metrics.is_some()
    }

    /// Check if tracing is enabled
    pub fn has_tracing(&self) -> bool {
        match &self.trace_config {
            Some(trace_config) => trace_config.enable_trace,
            None => false,
        }
    }

    /// Compute the effective retry config considering disable flag
    pub fn effective_retry_config(&self) -> RetryConfig {
        let mut cfg = self.retry.clone();
        if self.disable_retries {
            cfg.max_retries = 1;
        }
        cfg
    }

    /// Compute the effective circuit breaker config considering disable flag
    pub fn effective_circuit_breaker_config(&self) -> CircuitBreakerConfig {
        let mut cfg = self.circuit_breaker.clone();
        if self.disable_circuit_breaker {
            cfg.failure_threshold = u32::MAX;
        }
        cfg
    }

    /// Check if running in IGW (Inference Gateway) mode
    pub fn is_igw_mode(&self) -> bool {
        self.enable_igw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();

        assert!(
            matches!(config.mode, RoutingMode::Regular { worker_urls } if worker_urls.is_empty())
        );
        assert!(matches!(config.policy, PolicyConfig::Random));
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3001);
        assert_eq!(config.max_payload_size, 536_870_912);
        assert_eq!(config.stream_request_bodies_over, 0);
        assert_eq!(config.request_timeout_secs, 1800);
        assert_eq!(config.upstream_pool_idle_timeout_secs, 3);
        assert_eq!(config.worker_startup_timeout_secs, 1800);
        assert_eq!(config.worker_startup_check_interval_secs, 30);
        assert_eq!(config.load_monitor_interval_secs, 10);
        assert!(config.discovery.is_none());
        assert!(config.metrics.is_none());
        assert!(config.trace_config.is_none());
        assert!(config.log_dir.is_none());
        assert!(config.log_level.is_none());
        assert!(!config.tenant_resolution.trust_tenant_header);
        assert_eq!(
            config.tenant_resolution.tenant_header_name,
            DEFAULT_TENANT_HEADER_NAME
        );
    }

    #[test]
    fn test_router_config_new() {
        let mode = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string(), "http://worker2".to_string()],
        };
        let policy = PolicyConfig::RoundRobin;

        let config = RouterConfig::new(mode, policy);

        match config.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls.len(), 2);
                assert_eq!(worker_urls[0], "http://worker1");
                assert_eq!(worker_urls[1], "http://worker2");
            }
            _ => panic!("Expected Regular mode"),
        }

        assert!(matches!(config.policy, PolicyConfig::RoundRobin));
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3001);
    }

    #[test]
    fn test_router_config_serialization() {
        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker1".to_string()])
            .random_policy()
            .host("0.0.0.0")
            .port(8080)
            .log_dir("/var/log")
            .log_level("debug")
            .build_unchecked();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.host, deserialized.host);
        assert_eq!(config.port, deserialized.port);
        assert_eq!(config.max_payload_size, deserialized.max_payload_size);
        assert_eq!(config.log_dir, deserialized.log_dir);
        assert_eq!(config.log_level, deserialized.log_level);
        assert!(deserialized.discovery.is_none());
        assert!(deserialized.metrics.is_none());
        assert!(deserialized.trace_config.is_none());
    }

    #[test]
    fn test_health_check_port_serde_roundtrip_and_backward_compat() {
        // Default: dedicated probe listener off, and `skip_serializing_if`
        // keeps the key out of serialized output entirely.
        let config = RouterConfig::default();
        assert_eq!(config.health_check_port, None);
        let json = serde_json::to_string(&config).unwrap();
        assert!(
            !json.contains("health_check_port"),
            "None health_check_port must be omitted from serialized config"
        );

        // Existing config files predating the field deserialize cleanly via
        // `#[serde(default)]` (→ None).
        let without: RouterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(without.health_check_port, None);

        // When set, the value round-trips.
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .health_check_port(Some(8081))
            .build_unchecked();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("health_check_port"));
        let with: RouterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(with.health_check_port, Some(8081));
    }

    #[test]
    fn test_stream_request_bodies_over_serde_roundtrip_and_backward_compat() {
        // Config files predating the field deserialize to the disabled default.
        let mut json: serde_json::Value = serde_json::to_value(RouterConfig::default()).unwrap();
        json.as_object_mut()
            .unwrap()
            .remove("stream_request_bodies_over")
            .unwrap();
        let without: RouterConfig = serde_json::from_value(json).unwrap();
        assert_eq!(without.stream_request_bodies_over, 0);

        // When set, the value round-trips.
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .stream_request_bodies_over(4 * 1024 * 1024)
            .build_unchecked();
        let json = serde_json::to_string(&config).unwrap();
        let with: RouterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(with.stream_request_bodies_over, 4 * 1024 * 1024);
    }

    #[test]
    fn test_job_queue_sizing_serde_default_and_roundtrip() {
        // Config files predating the fields deserialize to today's values.
        let mut json: serde_json::Value = serde_json::to_value(RouterConfig::default()).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.remove("job_queue_capacity").unwrap();
        obj.remove("job_queue_concurrency").unwrap();
        let without: RouterConfig = serde_json::from_value(json).unwrap();
        assert_eq!(without.job_queue_capacity, 1000);
        assert_eq!(without.job_queue_concurrency, 200);

        // When set, the values round-trip.
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .job_queue_capacity(20_000)
            .job_queue_concurrency(500)
            .build_unchecked();
        let json = serde_json::to_string(&config).unwrap();
        let with: RouterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(with.job_queue_capacity, 20_000);
        assert_eq!(with.job_queue_concurrency, 500);
    }

    #[test]
    fn test_job_queue_sizing_rejects_zero() {
        // Config-file values bypass the CLI parsers, so validation is the
        // backstop against a zero-capacity channel panic at startup and
        // mirrors the CLI upper bounds.
        for (capacity, concurrency) in [(0, 200), (1000, 0), (1_000_001, 200), (1000, 100_001)] {
            let config = RouterConfig::builder()
                .regular_mode(vec![])
                .job_queue_capacity(capacity)
                .job_queue_concurrency(concurrency)
                .build_unchecked();
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn test_stream_body_stall_timeout_serde_default_and_roundtrip() {
        // Config files predating the field deserialize to the 300s default.
        let mut json: serde_json::Value = serde_json::to_value(RouterConfig::default()).unwrap();
        json.as_object_mut()
            .unwrap()
            .remove("stream_body_stall_timeout_secs")
            .unwrap();
        let without: RouterConfig = serde_json::from_value(json).unwrap();
        assert_eq!(without.stream_body_stall_timeout_secs, 300);

        // The disabling zero round-trips instead of reverting to the default.
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .stream_body_stall_timeout_secs(0)
            .build_unchecked();
        let json = serde_json::to_string(&config).unwrap();
        let with: RouterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(with.stream_body_stall_timeout_secs, 0);
    }

    #[test]
    fn alias_field_spellings_deserialize_and_serialize_canonically() {
        // Config files may use the intent-revealing spellings; alias in,
        // canonical out.
        let mut json = serde_json::to_value(RouterConfig::default()).unwrap();
        let obj = json.as_object_mut().unwrap();
        let v = obj.remove("routing_key_override").unwrap();
        obj.insert("sticky_sessions".to_string(), v);
        let hc = obj
            .get_mut("health_check")
            .unwrap()
            .as_object_mut()
            .unwrap();
        hc.remove("remove_unhealthy_workers").unwrap();
        hc.insert(
            "worker_auto_recovery".to_string(),
            serde_json::Value::Bool(true),
        );
        let cfg: RouterConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.health_check.remove_unhealthy_workers);
        let out = serde_json::to_string(&cfg).unwrap();
        assert!(out.contains("routing_key_override"));
        assert!(out.contains("remove_unhealthy_workers"));
        assert!(!out.contains("sticky_sessions"));
        assert!(!out.contains("worker_auto_recovery"));
    }

    #[test]
    fn policy_alias_field_spellings_deserialize_identically() {
        let canonical: PolicyConfig = serde_json::from_value(serde_json::json!({
            "type": "cache_aware",
            "cache_threshold": 0.6,
            "balance_abs_threshold": 8,
            "balance_rel_threshold": 1.2,
            "eviction_interval_secs": 60,
            "max_tree_size": 1024,
        }))
        .unwrap();
        let aliased: PolicyConfig = serde_json::from_value(serde_json::json!({
            "type": "cache_aware",
            "cache_match_threshold": 0.6,
            "spill_abs_threshold": 8,
            "spill_rel_threshold": 1.2,
            "eviction_interval_secs": 60,
            "max_tree_size": 1024,
        }))
        .unwrap();
        assert_eq!(format!("{canonical:?}"), format!("{aliased:?}"));
        let out = serde_json::to_string(&aliased).unwrap();
        assert!(!out.contains("cache_match_threshold"));
        assert!(!out.contains("spill_abs_threshold"));

        let manual: PolicyConfig = serde_json::from_value(serde_json::json!({
            "type": "manual",
            "sticky_key_idle_secs": 123,
        }))
        .unwrap();
        match manual {
            PolicyConfig::Manual { max_idle_secs, .. } => assert_eq!(max_idle_secs, 123),
            other => panic!("expected manual policy, got {other:?}"),
        }

        let override_cfg: RoutingKeyOverrideConfig = serde_json::from_value(serde_json::json!({
            "enabled": true,
            "sticky_key_idle_secs": 321,
        }))
        .unwrap();
        assert_eq!(override_cfg.max_idle_secs, 321);
    }

    #[test]
    fn test_routing_key_override_serde_default_and_roundtrip() {
        // Config files with only `enabled` deserialize to the defaults.
        let json: serde_json::Value = serde_json::json!({ "enabled": true });
        let cfg: RoutingKeyOverrideConfig = serde_json::from_value(json).unwrap();
        assert!(cfg.enabled);
        assert_eq!(cfg.assignment_mode, ManualAssignmentMode::Delegate);
        assert_eq!(cfg.headers, vec!["x-smg-routing-key".to_string()]);

        let cfg = RoutingKeyOverrideConfig {
            enabled: true,
            assignment_mode: ManualAssignmentMode::MinLoad,
            headers: vec!["x-routing-key".into(), "x-smg-routing-key".into()],
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let roundtripped: RoutingKeyOverrideConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtripped.assignment_mode, ManualAssignmentMode::MinLoad);
        assert_eq!(
            roundtripped.headers,
            vec!["x-routing-key".to_string(), "x-smg-routing-key".to_string()]
        );
    }

    #[test]
    fn delegate_assignment_mode_survives_serde_roundtrip() {
        let json = serde_json::to_string(&ManualAssignmentMode::Delegate).unwrap();
        assert_eq!(json, "\"delegate\"");
        let back: ManualAssignmentMode = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ManualAssignmentMode::Delegate);

        let cfg = RoutingKeyOverrideConfig {
            enabled: true,
            assignment_mode: ManualAssignmentMode::Delegate,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let roundtripped: RoutingKeyOverrideConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtripped.assignment_mode, ManualAssignmentMode::Delegate);
    }

    #[test]
    fn assignment_mode_defaults_split_by_context() {
        // Manual policy standalone keeps random; the override sticky map
        // defaults to delegate. Both are operator-visible defaults.
        assert_eq!(
            ManualAssignmentMode::default(),
            ManualAssignmentMode::Random
        );
        assert_eq!(
            RoutingKeyOverrideConfig::default().assignment_mode,
            ManualAssignmentMode::Delegate
        );
    }

    #[test]
    fn test_routing_mode_is_pd_mode() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        assert!(!regular.is_pd_mode());

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), Some(8001))],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };
        assert!(pd.is_pd_mode());
    }

    #[test]
    fn test_routing_mode_worker_count() {
        let regular = RoutingMode::Regular {
            worker_urls: vec![
                "http://worker1".to_string(),
                "http://worker2".to_string(),
                "http://worker3".to_string(),
            ],
        };
        assert_eq!(regular.worker_count(), 3);

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![
                ("http://prefill1".to_string(), Some(8001)),
                ("http://prefill2".to_string(), None),
            ],
            decode_urls: vec![
                "http://decode1".to_string(),
                "http://decode2".to_string(),
                "http://decode3".to_string(),
            ],
            prefill_policy: None,
            decode_policy: None,
        };
        assert_eq!(pd.worker_count(), 5);

        let empty_regular = RoutingMode::Regular {
            worker_urls: vec![],
        };
        assert_eq!(empty_regular.worker_count(), 0);
    }

    #[test]
    fn test_routing_mode_serialization() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        let json = serde_json::to_string(&regular).unwrap();
        assert!(json.contains("\"type\":\"regular\""));
        assert!(json.contains("\"worker_urls\""));

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), Some(8001))],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };
        let json = serde_json::to_string(&pd).unwrap();
        assert!(json.contains("\"type\":\"prefill_decode\""));
        assert!(json.contains("\"prefill_urls\""));
        assert!(json.contains("\"decode_urls\""));
    }

    #[test]
    fn test_policy_config_name() {
        assert_eq!(PolicyConfig::Random.name(), "random");
        assert_eq!(PolicyConfig::RoundRobin.name(), "round_robin");
        assert_eq!(PolicyConfig::Passthrough.name(), "passthrough");

        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.8,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 1000,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
        };
        assert_eq!(cache_aware.name(), "cache_aware");

        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        };
        assert_eq!(power_of_two.name(), "power_of_two");
    }

    #[test]
    fn test_policy_config_serialization() {
        let random = PolicyConfig::Random;
        let json = serde_json::to_string(&random).unwrap();
        assert_eq!(json, r#"{"type":"random"}"#);

        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.8,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 1000,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
        };
        let json = serde_json::to_string(&cache_aware).unwrap();
        assert!(json.contains("\"type\":\"cache_aware\""));
        assert!(json.contains("\"cache_threshold\":0.8"));
        assert!(json.contains("\"balance_abs_threshold\":10"));

        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        };
        let json = serde_json::to_string(&power_of_two).unwrap();
        assert!(json.contains("\"type\":\"power_of_two\""));
        assert!(json.contains("\"load_check_interval_secs\":60"));
    }

    #[test]
    fn test_cache_aware_parameters() {
        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.75,
            balance_abs_threshold: 20,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 600,
            max_tree_size: 5000,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
        };

        match cache_aware {
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
                ..
            } => {
                assert!((cache_threshold - 0.75).abs() < 0.0001);
                assert_eq!(balance_abs_threshold, 20);
                assert!((balance_rel_threshold - 2.0).abs() < 0.0001);
                assert_eq!(eviction_interval_secs, 600);
                assert_eq!(max_tree_size, 5000);
            }
            _ => panic!("Expected CacheAware"),
        }
    }

    #[test]
    fn test_cache_aware_pressure_knobs_default_off_when_absent() {
        // Config files written before the knobs existed must keep parsing,
        // with both knobs off (behavior-preserving defaults).
        let json = r#"{
            "type": "cache_aware",
            "cache_threshold": 0.5,
            "balance_abs_threshold": 32,
            "balance_rel_threshold": 1.1,
            "eviction_interval_secs": 60,
            "max_tree_size": 1000
        }"#;
        let policy: PolicyConfig = serde_json::from_str(json).unwrap();
        match policy {
            PolicyConfig::CacheAware {
                overlap_decay,
                selection_temperature,
                ..
            } => {
                assert_eq!(overlap_decay, 0.0);
                assert_eq!(selection_temperature, 0.0);
            }
            _ => panic!("Expected CacheAware"),
        }
    }

    #[test]
    fn test_cache_aware_index_fields_default_when_absent() {
        // Config files written before the hash index existed must keep
        // parsing as tree mode with the default TTL and no boundaries.
        let json = r#"{
            "type": "cache_aware",
            "cache_threshold": 0.5,
            "balance_abs_threshold": 32,
            "balance_rel_threshold": 1.1,
            "eviction_interval_secs": 60,
            "max_tree_size": 1000
        }"#;
        let policy: PolicyConfig = serde_json::from_str(json).unwrap();
        match policy {
            PolicyConfig::CacheAware {
                cache_index,
                cache_ttl_secs,
                cache_boundaries,
                ..
            } => {
                assert_eq!(cache_index, CacheIndexKind::Tree);
                assert_eq!(cache_ttl_secs, 180);
                assert!(cache_boundaries.is_empty());
            }
            _ => panic!("Expected CacheAware"),
        }
    }

    #[test]
    fn test_cache_aware_index_fields_round_trip() {
        let json = r#"{
            "type": "cache_aware",
            "cache_threshold": 0.5,
            "balance_abs_threshold": 32,
            "balance_rel_threshold": 1.1,
            "eviction_interval_secs": 60,
            "max_tree_size": 1000,
            "cache_index": "hash",
            "cache_ttl_secs": 90,
            "cache_boundaries": [2048, 8192]
        }"#;
        let policy: PolicyConfig = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&policy).unwrap();
        assert!(serialized.contains("\"cache_index\":\"hash\""));
        assert!(serialized.contains("\"cache_ttl_secs\":90"));
        assert!(serialized.contains("\"cache_boundaries\":[2048,8192]"));
        match serde_json::from_str::<PolicyConfig>(&serialized).unwrap() {
            PolicyConfig::CacheAware {
                cache_index,
                cache_ttl_secs,
                cache_boundaries,
                ..
            } => {
                assert_eq!(cache_index, CacheIndexKind::Hash);
                assert_eq!(cache_ttl_secs, 90);
                assert_eq!(cache_boundaries, vec![2048, 8192]);
            }
            _ => panic!("Expected CacheAware"),
        }
    }

    #[test]
    fn test_router_config_cache_boundaries_default_and_skip() {
        // Absent in old configs → empty; empty is skipped on serialize.
        let config = RouterConfig::default();
        assert!(config.cache_boundaries.is_empty());
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("cache_boundaries"));

        let with_boundaries = RouterConfig {
            cache_boundaries: vec![16, 64],
            ..Default::default()
        };
        let json = serde_json::to_string(&with_boundaries).unwrap();
        assert!(json.contains("\"cache_boundaries\":[16,64]"));
        let parsed: RouterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cache_boundaries, vec![16, 64]);
    }

    #[test]
    fn test_power_of_two_parameters() {
        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 120,
        };

        match power_of_two {
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            } => {
                assert_eq!(load_check_interval_secs, 120);
            }
            _ => panic!("Expected PowerOfTwo"),
        }
    }

    #[test]
    fn test_bucket_parameters() {
        let bucket = PolicyConfig::Bucket {
            balance_abs_threshold: 20,
            balance_rel_threshold: 2.0,
            bucket_adjust_interval_secs: 5,
        };

        match bucket {
            PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                assert_eq!(balance_abs_threshold, 20);
                assert!((balance_rel_threshold - 2.0).abs() < 0.0001);
                assert_eq!(bucket_adjust_interval_secs, 5);
            }
            _ => panic!("Expected Bucket"),
        }
    }

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();

        assert!(!config.enabled);
        assert!(config.namespace.is_none());
        assert_eq!(config.port, 8000);
        assert_eq!(config.check_interval_secs, 120);
        assert!(config.selector.is_empty());
        assert!(config.encode_selector.is_empty());
        assert!(config.prefill_selector.is_empty());
        assert!(config.decode_selector.is_empty());
        assert_eq!(config.bootstrap_port_annotation, "sglang.ai/bootstrap-port");
    }

    #[test]
    fn test_discovery_config_with_selectors() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());
        selector.insert("role".to_string(), "worker".to_string());

        let config = DiscoveryConfig {
            enabled: true,
            namespace: Some("default".to_string()),
            port: 9000,
            check_interval_secs: 30,
            selector: selector.clone(),
            encode_selector: selector.clone(),
            prefill_selector: selector.clone(),
            decode_selector: selector.clone(),
            bootstrap_port_annotation: "custom.io/port".to_string(),
            worker_ports_annotation: "smg.ai/worker-ports".to_string(),
            router_selector: HashMap::new(),
            router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
            model_id_source: None,
        };

        assert!(config.enabled);
        assert_eq!(config.namespace, Some("default".to_string()));
        assert_eq!(config.port, 9000);
        assert_eq!(config.selector.len(), 2);
        assert_eq!(config.selector.get("app"), Some(&"sglang".to_string()));
    }

    #[test]
    fn test_discovery_config_namespace() {
        let config = DiscoveryConfig {
            namespace: None,
            ..Default::default()
        };
        assert!(config.namespace.is_none());

        let config = DiscoveryConfig {
            namespace: Some("production".to_string()),
            ..Default::default()
        };
        assert_eq!(config.namespace, Some("production".to_string()));
    }

    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();

        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_metrics_config_custom() {
        let config = MetricsConfig {
            port: 9090,
            host: "0.0.0.0".to_string(),
        };

        assert_eq!(config.port, 9090);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_trace_config_default() {
        let config = TraceConfig::default();

        assert!(!config.enable_trace);
        assert_eq!(config.otlp_traces_endpoint, "localhost:4317");
    }

    #[test]
    fn test_trace_config_custom() {
        let config = TraceConfig {
            enable_trace: true,
            otlp_traces_endpoint: "otel-collector:4317".to_string(),
        };

        assert!(config.enable_trace);
        assert_eq!(config.otlp_traces_endpoint, "otel-collector:4317");
    }

    #[test]
    fn test_mode_type() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .build_unchecked();
        assert_eq!(config.mode_type(), "regular");

        let config = RouterConfig::builder()
            .prefill_decode_mode(vec![], vec![])
            .build_unchecked();
        assert_eq!(config.mode_type(), "prefill_decode");
    }

    #[test]
    fn test_has_service_discovery() {
        let config = RouterConfig::default();
        assert!(!config.has_service_discovery());

        let config = RouterConfig::builder()
            .discovery_config(DiscoveryConfig {
                enabled: false,
                ..Default::default()
            })
            .build_unchecked();
        assert!(!config.has_service_discovery());

        let config = RouterConfig::builder().enable_discovery().build_unchecked();
        assert!(config.has_service_discovery());
    }

    #[test]
    fn test_has_metrics() {
        let config = RouterConfig::default();
        assert!(!config.has_metrics());

        let config = RouterConfig::builder()
            .metrics_config(MetricsConfig::default())
            .build_unchecked();
        assert!(config.has_metrics());
    }

    #[test]
    fn test_has_tracing() {
        let config = RouterConfig::default();
        assert!(!config.has_tracing());

        let config = RouterConfig::builder()
            .enable_trace("localhost:4317")
            .build_unchecked();
        assert!(config.has_tracing());
    }

    #[test]
    fn test_large_worker_lists() {
        let large_urls: Vec<String> = (0..1000).map(|i| format!("http://worker{i}")).collect();

        let config = RouterConfig::builder()
            .regular_mode(large_urls.clone())
            .build_unchecked();

        assert_eq!(config.mode.worker_count(), 1000);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        match deserialized.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls.len(), 1000);
            }
            _ => panic!("Expected Regular mode"),
        }
    }

    #[test]
    fn test_unicode_in_config() {
        let config = RouterConfig::builder()
            .regular_mode(vec![
                "http://работник1".to_string(),
                "http://工作者2".to_string(),
            ])
            .log_dir("/日志/目录")
            .build_unchecked();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        match deserialized.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls[0], "http://работник1");
                assert_eq!(worker_urls[1], "http://工作者2");
            }
            _ => panic!("Expected Regular mode"),
        }

        assert_eq!(deserialized.log_dir, Some("/日志/目录".to_string()));
    }

    #[test]
    fn test_empty_string_fields() {
        let config = RouterConfig::builder()
            .host("")
            .log_dir("")
            .log_level("")
            .build_unchecked();

        assert_eq!(config.host, "");
        assert_eq!(config.log_dir, Some(String::new()));
        assert_eq!(config.log_level, Some(String::new()));
    }

    #[test]
    fn test_full_pd_mode_config() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://prefill1:8000".to_string(), Some(8001)),
                    ("http://prefill2:8000".to_string(), None),
                ],
                vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
            )
            .power_of_two_policy(30)
            .host("0.0.0.0")
            .port(3000)
            .max_payload_size(1048576)
            .request_timeout_secs(120)
            .worker_startup_timeout_secs(60)
            .worker_startup_check_interval_secs(5)
            .discovery_config(DiscoveryConfig {
                enabled: true,
                namespace: Some("sglang".to_string()),
                ..Default::default()
            })
            .enable_metrics("0.0.0.0", 9090)
            .enable_trace("localhost:4317")
            .log_dir("/var/log/sglang")
            .log_level("info")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 4);
        assert_eq!(config.policy.name(), "power_of_two");
        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
        assert!(config.has_tracing());
    }

    #[test]
    fn test_full_regular_mode_config() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());

        let config = RouterConfig::builder()
            .regular_mode(vec![
                "http://worker1:8000".to_string(),
                "http://worker2:8000".to_string(),
                "http://worker3:8000".to_string(),
            ])
            .cache_aware_policy(0.9, 5, 1.2, 600, 10000)
            .host("0.0.0.0")
            .port(3001)
            .max_payload_size(536870912)
            .request_timeout_secs(300)
            .worker_startup_timeout_secs(180)
            .worker_startup_check_interval_secs(15)
            .discovery_config(DiscoveryConfig {
                enabled: true,
                namespace: None,
                port: 8080,
                check_interval_secs: 45,
                selector,
                ..Default::default()
            })
            .metrics_config(MetricsConfig::default())
            .enable_trace("localhost:4317")
            .log_level("debug")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(!config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 3);
        assert_eq!(config.policy.name(), "cache_aware");
        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
        assert!(config.has_tracing());
    }

    #[test]
    fn test_config_with_all_options() {
        let mut selectors = HashMap::new();
        selectors.insert("env".to_string(), "prod".to_string());
        selectors.insert("version".to_string(), "v1".to_string());

        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker1".to_string()])
            .round_robin_policy()
            .host("::1") // IPv6
            .port(8888)
            .max_payload_size(1024 * 1024 * 512) // 512MB
            .request_timeout_secs(900)
            .worker_startup_timeout_secs(600)
            .worker_startup_check_interval_secs(20)
            .discovery_config(DiscoveryConfig {
                enabled: true,
                namespace: Some("production".to_string()),
                port: 8443,
                check_interval_secs: 120,
                selector: selectors.clone(),
                encode_selector: selectors.clone(),
                prefill_selector: selectors.clone(),
                decode_selector: selectors,
                bootstrap_port_annotation: "mycompany.io/bootstrap".to_string(),
                worker_ports_annotation: "smg.ai/worker-ports".to_string(),
                router_selector: HashMap::new(),
                router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
                model_id_source: None,
            })
            .enable_metrics("::", 9999) // IPv6 any
            .enable_trace("localhost:4317")
            .log_dir("/opt/logs/sglang")
            .log_level("trace")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
        assert!(config.has_tracing());
        assert_eq!(config.mode_type(), "regular");

        let json = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.host, "::1");
        assert_eq!(deserialized.port, 8888);
        assert_eq!(
            deserialized.discovery.unwrap().namespace,
            Some("production".to_string())
        );
    }

    #[test]
    fn test_pd_policy_fallback_both_specified() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: Some(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
                block_size: 16,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                overlap_decay: 0.0,
                selection_temperature: 0.0,
                cache_index: Default::default(),
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
            }),
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
        };

        let main_policy = PolicyConfig::Random;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {}
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {}
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_prefill() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: Some(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
                block_size: 16,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                overlap_decay: 0.0,
                selection_temperature: 0.0,
                cache_index: Default::default(),
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
            }),
            decode_policy: None,
        };

        let main_policy = PolicyConfig::RoundRobin;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {}
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_decode() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
        };

        let main_policy = PolicyConfig::Random;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::Random => {}
            _ => panic!("Expected Random for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {}
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_none_specified() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };

        let main_policy = PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 20,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 2000,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
        };

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.7).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.7).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware for decode"),
        }
    }

    #[test]
    fn test_regular_mode_policy_fallback() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };

        let main_policy = PolicyConfig::RoundRobin;

        match regular.get_prefill_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for regular mode"),
        }

        match regular.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for regular mode"),
        }
    }
}
