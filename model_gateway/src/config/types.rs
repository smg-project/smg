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
    /// Explicit runtime for the startup workers (`--worker-urls`), set from
    /// `--backend` when the connection mode is ZMQ. The ZMQ handshake is shared
    /// across engine runtimes, so the wire protocol cannot be probed and must
    /// be declared up front; HTTP/gRPC workers keep auto-detection and ignore
    /// this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub startup_worker_runtime_type: Option<RuntimeType>,
    pub policy: PolicyConfig,
    /// Per-request sticky-routing override (honors `X-SMG-Routing-Key`).
    #[serde(default)]
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
    pub request_timeout_secs: u64,
    pub worker_startup_timeout_secs: u64,
    /// Grace period before the first worker-startup check fires. The engine is
    /// left alone for this long, then polled every
    /// `worker_startup_check_interval_secs`.
    #[serde(default)]
    pub worker_startup_delay_secs: u64,
    pub worker_startup_check_interval_secs: u64,
    #[serde(default = "default_load_monitor_interval_secs")]
    pub load_monitor_interval_secs: u64,
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
    /// Set to -1 to disable rate limiting
    pub max_concurrent_requests: i32,
    pub queue_size: usize,
    pub queue_timeout_secs: u64,
    /// If not set, defaults to max_concurrent_requests
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
}

/// Per-request sticky-routing override: when `X-SMG-Routing-Key` is present, any
/// eligible policy routes via manual sticky-map semantics. Reuses the manual
/// policy knobs for the sticky map; eviction defaults match the manual policy so
/// config-file users with only `enabled: true` still get TTL eviction (no leak).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingKeyOverrideConfig {
    /// When false, policies are used unchanged.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_manual_eviction_interval_secs")]
    pub eviction_interval_secs: u64,
    #[serde(default = "default_manual_max_idle_secs")]
    pub max_idle_secs: u64,
    #[serde(default)]
    pub assignment_mode: ManualAssignmentMode,
}

impl Default for RoutingKeyOverrideConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            eviction_interval_secs: default_manual_eviction_interval_secs(),
            max_idle_secs: default_manual_max_idle_secs(),
            assignment_mode: ManualAssignmentMode::default(),
        }
    }
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
        cache_threshold: f32,
        balance_abs_threshold: usize,
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
    },

    #[serde(rename = "power_of_two")]
    PowerOfTwo { load_check_interval_secs: u64 },

    /// Least-(token-)work policy: routes to the worker minimizing the expected
    /// wait `(queued_tokens + inflight_tokens) / throughput + kv_pressure_weight * k/(1-k)`
    /// — token-work drain time plus a convex KV-cache pressure barrier, computed
    /// from the load monitor with in-flight correction. See `policies/least_load.rs`.
    #[serde(rename = "least_load")]
    LeastLoad {
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
    },

    #[serde(rename = "bucket")]
    Bucket {
        /// Absolute load difference threshold for load balancing
        balance_abs_threshold: usize,
        /// Relative load ratio threshold for load balancing
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
        /// Maximum idle time before eviction (seconds, default: 14400 = 4 hours)
        #[serde(default = "default_manual_max_idle_secs")]
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
    /// - Walks ring if worker is overloaded (load > avg * load_factor)
    /// - O(log n) lookup instead of O(prefix_len) radix tree traversal
    #[serde(rename = "prefix_hash")]
    PrefixHash {
        /// Number of prefix tokens to hash (default: 256)
        #[serde(default = "default_prefix_token_count")]
        prefix_token_count: usize,
        /// Load factor threshold - walk ring if load > avg * factor (default: 1.25)
        #[serde(default = "default_load_factor")]
        load_factor: f64,
    },
}

fn default_block_size() -> usize {
    16
}

fn default_balance_token_usage_threshold() -> f32 {
    1.0
}

fn default_prefix_token_count() -> usize {
    256
}

fn default_load_factor() -> f64 {
    1.25
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
    #[serde(default)]
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
            routing_key_override: RoutingKeyOverrideConfig::default(),
            host: "0.0.0.0".to_string(),
            port: 3001,
            health_check_port: None,
            runtime_worker_threads: None,
            max_payload_size: 536_870_912,     // 512MB
            request_timeout_secs: 1800,        // 30 minutes
            worker_startup_timeout_secs: 1800, // 30 minutes for large model loading
            worker_startup_delay_secs: 0,
            worker_startup_check_interval_secs: 30,
            load_monitor_interval_secs: 10,
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
        assert_eq!(config.request_timeout_secs, 1800);
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
