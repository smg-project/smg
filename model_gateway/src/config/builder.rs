use std::{collections::HashMap, io::Read};

use llm_multimodal::jpeg_turbo::TurboJpegPolicy;
use openai_protocol::worker::{MmProcessingMode, TransportMode};
use smg_mcp::McpConfig;
use smg_policy_controller::document::operator::{SpecRejected, MAX_DOCUMENT_BYTES};
use smg_worker::identity::basic::{ConnectionMode, RuntimeType};

use super::{
    validation::ConfigValidator, AccessLogField, CircuitBreakerConfig, ConfigError, ConfigResult,
    DiscoveryConfig, GroupedDpSticky, HealthCheckConfig, HistoryBackend, KubernetesDiscoveryConfig,
    MemoryConfig, MetricsConfig, OracleConfig, ParserChoice, PdPairingMode, PdReadinessConfig,
    PolicyConfig, PostgresConfig, RedisConfig, RetryConfig, RouterConfig, RoutingKeyOverrideConfig,
    RoutingMode, TenantApiKeyEntry, TokenizerCacheConfig, TraceConfig,
};

/// A policy document's text from its opened source, where `stated` is the
/// size the file's metadata states. A file over the controller's bound on
/// a document ([`MAX_DOCUMENT_BYTES`]) is refused by that size, unread, in
/// the controller's words. A file may grow behind its metadata, and a
/// source that is no regular file states no size: the read takes one byte
/// past the bound and no more, and refuses a source that reaches past it.
fn bounded_document(stated: u64, source: impl Read) -> std::io::Result<String> {
    let too_large = |len: u64| {
        std::io::Error::new(
            std::io::ErrorKind::FileTooLarge,
            SpecRejected::TooLarge(len).to_string(),
        )
    };
    if stated > MAX_DOCUMENT_BYTES {
        return Err(too_large(stated));
    }
    let mut bytes = Vec::new();
    source
        .take(MAX_DOCUMENT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    let read = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if read > MAX_DOCUMENT_BYTES {
        return Err(too_large(read));
    }
    let mut text = String::new();
    bytes.as_slice().read_to_string(&mut text)?;
    Ok(text)
}

/// Builder for RouterConfig that wraps the config itself
/// This eliminates field duplication and stays in sync automatically
#[derive(Debug, Clone, Default)]
pub struct RouterConfigBuilder {
    config: RouterConfig,
    // Temporary fields for certificate paths (read during build)
    client_cert_path: Option<String>,
    client_key_path: Option<String>,
    ca_cert_paths: Vec<String>,
    server_cert_path: Option<String>,
    server_key_path: Option<String>,
    mcp_config_path: Option<String>,
}

impl RouterConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    // ==================== Routing Mode ====================

    pub fn regular_mode(mut self, worker_urls: Vec<String>) -> Self {
        self.config.mode = RoutingMode::Regular { worker_urls };
        self
    }

    pub fn prefill_decode_mode(
        mut self,
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
    ) -> Self {
        self.config.mode = RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            prefill_policy: None,
            decode_policy: None,
        };
        self
    }

    pub fn openai_mode(mut self, worker_urls: Vec<String>) -> Self {
        self.config.mode = RoutingMode::OpenAI { worker_urls };
        self
    }

    pub fn mode(mut self, mode: RoutingMode) -> Self {
        self.config.mode = mode;
        self
    }

    // ==================== Policy ====================

    pub fn policy(mut self, policy: PolicyConfig) -> Self {
        self.config.policy = policy;
        self
    }

    /// Choose the policy controller for the whole process when `on`.
    pub fn policy_controller(mut self, on: bool) -> Self {
        self.config.policy_controller = on;
        self
    }

    pub fn cache_boundaries(mut self, boundaries: Vec<usize>) -> Self {
        self.config.cache_boundaries = boundaries;
        self
    }

    pub fn random_policy(mut self) -> Self {
        self.config.policy = PolicyConfig::Random;
        self
    }

    pub fn round_robin_policy(mut self) -> Self {
        self.config.policy = PolicyConfig::RoundRobin;
        self
    }

    pub fn cache_aware_policy(
        mut self,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
    ) -> Self {
        self.config.policy = PolicyConfig::cache_aware(
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
        );
        self
    }

    pub fn power_of_two_policy(mut self) -> Self {
        self.config.policy = PolicyConfig::PowerOfTwo {
            load_metric: Default::default(),
        };
        self
    }

    // ==================== Connection ====================

    /// Explicit runtime for startup workers over ZMQ (`None` keeps
    /// auto-detection for HTTP/gRPC; see `RouterConfig::startup_worker_runtime_type`).
    pub fn startup_worker_runtime_type(mut self, runtime: Option<RuntimeType>) -> Self {
        self.config.startup_worker_runtime_type = runtime;
        self
    }

    /// DP engines per startup ZMQ worker (grouped worker; see
    /// `RouterConfig::zmq_engine_count`).
    pub fn zmq_engine_count(mut self, count: Option<usize>) -> Self {
        self.config.zmq_engine_count = count;
        self
    }

    pub fn connection_mode(mut self, mode: ConnectionMode) -> Self {
        self.config.connection_mode = mode;
        self
    }

    pub fn grpc_connection(mut self) -> Self {
        self.config.connection_mode = ConnectionMode::Grpc;
        self
    }

    pub fn host<S: Into<String>>(mut self, host: S) -> Self {
        self.config.host = host.into();
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    pub fn health_check_port(mut self, health_check_port: Option<u16>) -> Self {
        self.config.health_check_port = health_check_port;
        self
    }

    /// Set optional per-role capacity thresholds for PD/EPD readiness.
    pub fn pd_readiness(mut self, pd_readiness: Option<PdReadinessConfig>) -> Self {
        self.config.pd_readiness = pd_readiness;
        self
    }

    /// The admin listener's port; `None` keeps the control plane on the
    /// data listener.
    pub fn admin_port(mut self, admin_port: Option<u16>) -> Self {
        self.config.admin_port = admin_port;
        self
    }

    /// The admin listener's bind host; `None` binds it on the server host.
    pub fn admin_host(mut self, admin_host: Option<String>) -> Self {
        self.config.admin_host = admin_host;
        self
    }

    pub fn runtime_worker_threads(mut self, threads: Option<usize>) -> Self {
        self.config.runtime_worker_threads = threads;
        self
    }

    // ==================== Request ====================

    pub fn max_payload_size(mut self, size: usize) -> Self {
        self.config.max_payload_size = size;
        self
    }

    pub fn max_buffered_request_bytes(mut self, bytes: u64) -> Self {
        self.config.max_buffered_request_bytes = bytes;
        self
    }

    pub fn stream_body_stall_timeout_secs(mut self, secs: u64) -> Self {
        self.config.stream_body_stall_timeout_secs = secs;
        self
    }

    pub fn header_read_timeout_secs(mut self, secs: u64) -> Self {
        self.config.header_read_timeout_secs = secs;
        self
    }

    pub fn idle_connection_timeout_secs(mut self, secs: u64) -> Self {
        self.config.idle_connection_timeout_secs = secs;
        self
    }

    /// Most client connections the HTTP listener holds open at once; `0` is
    /// no bound.
    pub fn max_connections(mut self, max_connections: usize) -> Self {
        self.config.max_connections = max_connections;
        self
    }

    /// Most client connections the admin listener holds open at once; `0`
    /// is no bound.
    pub fn admin_max_connections(mut self, admin_max_connections: usize) -> Self {
        self.config.admin_max_connections = admin_max_connections;
        self
    }

    pub fn upstream_pool_idle_timeout_secs(mut self, secs: u64) -> Self {
        self.config.upstream_pool_idle_timeout_secs = secs;
        self
    }

    pub fn request_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.request_timeout_secs = timeout;
        self
    }

    /// The bound on a request to a gRPC worker, from its dispatch to its last
    /// message; `0` arms no deadline.
    pub fn grpc_request_timeout_secs(mut self, secs: u64) -> Self {
        self.config.grpc_request_timeout_secs = secs;
        self
    }

    pub fn sse_keepalive_secs(mut self, secs: u64) -> Self {
        self.config.sse_keepalive_secs = secs;
        self
    }

    pub fn worker_startup_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.worker_startup_timeout_secs = timeout;
        self
    }

    pub fn zmq_startup_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.zmq_startup_timeout_secs = timeout;
        self
    }

    pub fn worker_startup_delay_secs(mut self, delay: u64) -> Self {
        self.config.worker_startup_delay_secs = delay;
        self
    }

    pub fn worker_startup_check_interval_secs(mut self, interval: u64) -> Self {
        self.config.worker_startup_check_interval_secs = interval;
        self
    }

    pub fn job_queue_capacity(mut self, capacity: usize) -> Self {
        self.config.job_queue_capacity = capacity;
        self
    }

    pub fn job_queue_concurrency(mut self, concurrency: usize) -> Self {
        self.config.job_queue_concurrency = concurrency;
        self
    }

    pub fn load_monitor_interval_secs(mut self, interval: u64) -> Self {
        self.config.load_monitor_interval_secs = interval;
        self
    }

    pub fn load_monitor_interval_ms(mut self, interval: Option<u64>) -> Self {
        self.config.load_monitor_interval_ms = interval;
        self
    }

    pub fn worker_stall_secs(mut self, secs: u64) -> Self {
        self.config.worker_stall_secs = secs;
        self
    }

    pub fn worker_wedge_secs(mut self, secs: u64) -> Self {
        self.config.worker_wedge_secs = secs;
        self
    }

    pub fn worker_stale_secs(mut self, secs: u64) -> Self {
        self.config.worker_stale_secs = secs;
        self
    }

    pub fn worker_warmup(
        mut self,
        secs: u64,
        share: f32,
        blocks: usize,
        thin_ratio: f32,
        divert_every: u64,
    ) -> Self {
        self.config.worker_warmup_secs = secs;
        self.config.worker_warmup_share = share;
        self.config.worker_warmup_blocks = blocks;
        self.config.worker_warmup_thin_ratio = thin_ratio;
        self.config.worker_warmup_divert_every = divert_every;
        self
    }

    pub fn pd_admission_wait_secs(mut self, secs: u64) -> Self {
        self.config.pd_admission_wait_secs = secs;
        self
    }

    pub fn disable_load_monitoring(mut self, disabled: bool) -> Self {
        self.config.disable_load_monitoring = disabled;
        self
    }

    pub fn worker_overload_protection(mut self, enabled: bool) -> Self {
        self.config.worker_overload_protection = enabled;
        self
    }

    pub fn worker_overload_waiting_requests(mut self, threshold: Option<usize>) -> Self {
        self.config.worker_overload_waiting_requests = threshold;
        self
    }

    pub fn worker_overload_shed(mut self, shed: bool) -> Self {
        self.config.worker_overload_shed = shed;
        self
    }

    pub fn worker_overload_token_usage(mut self, threshold: Option<f64>) -> Self {
        self.config.worker_overload_token_usage = threshold;
        self
    }

    pub fn engine_metrics(mut self, enabled: bool) -> Self {
        self.config.engine_metrics = enabled;
        self
    }

    /// Global multimodal tensor transport mode (per-worker specs can override).
    pub fn multimodal_tensor_transport(mut self, mode: Option<TransportMode>) -> Self {
        self.config.multimodal_tensor_transport = mode;
        self
    }

    /// Global minimum multimodal tensor size (bytes) before SHM transport is used.
    pub fn multimodal_shm_min_bytes(mut self, bytes: Option<usize>) -> Self {
        self.config.multimodal_shm_min_bytes = bytes;
        self
    }

    /// Most bytes of preprocessed media held in flight for engines at once.
    pub fn multimodal_max_inflight_bytes(mut self, bytes: Option<usize>) -> Self {
        self.config.multimodal_max_inflight_bytes = bytes;
        self
    }

    /// Per-request image-count limit replacing each model spec's built-in limit.
    pub fn mm_per_request_image_limit(mut self, limit: Option<usize>) -> Self {
        self.config.mm_per_request_image_limit = limit;
        self
    }

    /// Where media for one engine's gRPC workers is fetched and preprocessed.
    pub fn mm_processing(mut self, mode: Option<MmProcessingMode>) -> Self {
        self.config.mm_processing = mode;
        self
    }

    /// Host-DRAM budget (MiB) for router-side preprocessed media.
    pub fn mm_pixel_cache_mb(mut self, mb: Option<usize>) -> Self {
        self.config.mm_pixel_cache_mb = mb;
        self
    }

    /// Image/audio download timeout in milliseconds; unset uses env/default.
    pub fn mm_fetch_timeout_ms(mut self, timeout: Option<u64>) -> Self {
        self.config.mm_fetch_timeout_ms = timeout;
        self
    }

    /// Video download timeout in milliseconds; unset uses env/default.
    pub fn mm_video_fetch_timeout_ms(mut self, timeout: Option<u64>) -> Self {
        self.config.mm_video_fetch_timeout_ms = timeout;
        self
    }

    /// Serve cached pixels over RDMA (legacy switch for the RDMA lane).
    pub fn mm_pixel_rdma(mut self, enabled: bool) -> Self {
        self.config.mm_pixel_rdma = enabled;
        self
    }

    /// What the router's own media pipeline does when libjpeg-turbo cannot be
    /// loaded: `prefer` (the fallback decoder) or `require` (no start-up).
    pub fn mm_turbojpeg(mut self, policy: Option<TurboJpegPolicy>) -> Self {
        self.config.mm_turbojpeg = policy;
        self
    }

    /// Hosts the router's own media pipeline may fetch client-supplied media
    /// URLs from, by name (every redirect hop included); `None` is any host
    /// the address policy passes.
    pub fn mm_allowed_media_domains(mut self, domains: Option<Vec<String>>) -> Self {
        self.config.mm_allowed_media_domains = domains;
        self
    }

    /// Allow client-supplied media URLs on loopback, link-local or private
    /// addresses (a deployment whose media origin is internal).
    pub fn mm_allow_private_media_hosts(mut self, allowed: bool) -> Self {
        self.config.mm_allow_private_media_hosts = allowed;
        self
    }

    /// Listener IP for the RDMA metadata exchange.
    pub fn rdma_listen_ip(mut self, ip: Option<impl Into<String>>) -> Self {
        self.config.rdma_listen_ip = ip.map(Into::into);
        self
    }

    /// Full-TTL override (seconds) for leased RDMA pixel slots.
    pub fn rdma_slot_ttl_s(mut self, secs: Option<u64>) -> Self {
        self.config.rdma_slot_ttl_s = secs;
        self
    }

    /// Emit per-request multimodal timing at INFO.
    pub fn log_mm_timing(mut self, enabled: bool) -> Self {
        self.config.log_mm_timing = enabled;
        self
    }

    // ==================== Rate Limiting ====================

    pub fn max_concurrent_requests(mut self, max: i32) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }

    pub fn queue_size(mut self, size: usize) -> Self {
        self.config.queue_size = size;
        self
    }

    pub fn queue_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.queue_timeout_secs = timeout;
        self
    }

    pub fn prefill_max_inflight_requests_per_worker(mut self, max: i32) -> Self {
        self.config.prefill_max_inflight_requests_per_worker = max;
        self
    }

    pub fn prefill_queue_size(mut self, size: Option<usize>) -> Self {
        self.config.prefill_queue_size = size;
        self
    }

    pub fn prefill_queue_timeout_secs(mut self, timeout: Option<u64>) -> Self {
        self.config.prefill_queue_timeout_secs = timeout;
        self
    }

    pub fn rate_limit_tokens_per_second(mut self, tokens: i32) -> Self {
        self.config.rate_limit_tokens_per_second = Some(tokens);
        self
    }

    /// Most background Responses in flight at once (0 = background mode off).
    pub fn max_background_responses(mut self, max: usize) -> Self {
        self.config.max_background_responses = max;
        self
    }

    // ==================== Priority Scheduler ====================

    pub fn priority_scheduler_enabled(mut self, enabled: bool) -> Self {
        self.config.priority_scheduler_enabled = enabled;
        self
    }

    pub fn priority_scheduler_default_max_class(mut self, class: impl Into<String>) -> Self {
        self.config.priority_scheduler_default_max_class = class.into();
        self
    }

    pub fn priority_scheduler_config(mut self, path: Option<String>) -> Self {
        self.config.priority_scheduler_config = path;
        self
    }

    pub fn priority_scheduler_tenant_metric_top_n(mut self, n: u32) -> Self {
        self.config.priority_scheduler_tenant_metric_top_n = n;
        self
    }

    // ==================== Tenant Rate Limit ====================

    pub fn tenant_rate_limit_enabled(mut self, enabled: bool) -> Self {
        self.config.tenant_rate_limit_enabled = enabled;
        self
    }

    pub fn tenant_rate_limit_config(mut self, path: Option<String>) -> Self {
        self.config.tenant_rate_limit_config = path;
        self
    }

    // ==================== Security & CORS ====================

    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    pub fn tenant_api_keys(mut self, keys: Vec<TenantApiKeyEntry>) -> Self {
        self.config.tenant_api_keys = keys;
        self
    }

    pub fn cors_allowed_origins(mut self, origins: Vec<String>) -> Self {
        self.config.cors_allowed_origins = origins;
        self
    }

    // ==================== Retry ====================

    pub fn retry_config(mut self, retry: RetryConfig) -> Self {
        self.config.retry = retry;
        self
    }

    pub fn disable_retries(mut self) -> Self {
        self.config.disable_retries = true;
        self
    }

    // ==================== Circuit Breaker ====================

    pub fn circuit_breaker_config(mut self, circuit_breaker: CircuitBreakerConfig) -> Self {
        self.config.circuit_breaker = circuit_breaker;
        self
    }

    pub fn disable_circuit_breaker(mut self) -> Self {
        self.config.disable_circuit_breaker = true;
        self
    }

    // ==================== Health Check ====================

    pub fn health_check_config(mut self, health_check: HealthCheckConfig) -> Self {
        self.config.health_check = health_check;
        self
    }

    // ==================== Discovery ====================

    /// Kubernetes discovery with default settings
    pub fn enable_discovery(mut self) -> Self {
        self.config.discovery = Some(KubernetesDiscoveryConfig::default().into());
        self
    }

    // ==================== Metrics ====================

    pub fn metrics_config(mut self, metrics: MetricsConfig) -> Self {
        self.config.metrics = Some(metrics);
        self
    }

    pub fn enable_metrics<S: Into<String>>(mut self, host: S, port: u16) -> Self {
        self.config.metrics = Some(MetricsConfig {
            host: host.into(),
            port,
        });
        self
    }

    // ===================== Otel Trace ====================

    pub fn enable_trace<S: Into<String>>(mut self, endpoint: S) -> Self {
        self.config.trace_config = Some(TraceConfig {
            enable_trace: true,
            otlp_traces_endpoint: endpoint.into(),
        });
        self
    }

    // ==================== Logging ====================

    pub fn log_dir<S: Into<String>>(mut self, dir: S) -> Self {
        self.config.log_dir = Some(dir.into());
        self
    }

    pub fn log_level<S: Into<String>>(mut self, level: S) -> Self {
        self.config.log_level = Some(level.into());
        self
    }

    /// The optional fields of the access-log record (none by default).
    pub fn access_log_fields(mut self, fields: Vec<AccessLogField>) -> Self {
        self.config.access_log_fields = fields;
        self
    }

    pub fn request_id_headers(mut self, headers: Vec<String>) -> Self {
        self.config.request_id_headers = Some(headers);
        self
    }

    pub fn storage_context_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.config.storage_context_headers = headers;
        self
    }

    pub fn trust_tenant_header(mut self, trust: bool) -> Self {
        self.config.tenant_resolution.trust_tenant_header = trust;
        self
    }

    pub fn tenant_header_name<S: Into<String>>(mut self, header_name: S) -> Self {
        self.config.tenant_resolution.tenant_header_name = header_name.into();
        self
    }

    // ==================== IGW Mode ====================

    pub fn enable_igw(mut self) -> Self {
        self.config.enable_igw = true;
        self
    }

    pub fn upstream_http2(mut self, enable: bool) -> Self {
        self.config.upstream_http2 = enable;
        self
    }

    // ==================== WASM ====================

    pub fn enable_wasm(mut self, enable: bool) -> Self {
        self.config.enable_wasm = enable;
        self
    }

    pub fn maybe_storage_hook_wasm_path(mut self, path: Option<&str>) -> Self {
        self.config.storage_hook_wasm_path = path.map(|p| p.to_string());
        self
    }

    pub fn model_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.model_path = Some(path.into());
        self
    }

    /// Overrides model_path tokenizer
    pub fn tokenizer_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.tokenizer_path = Some(path.into());
        self
    }

    pub fn chat_template<S: Into<String>>(mut self, path: S) -> Self {
        self.config.chat_template = Some(path.into());
        self
    }

    /// Alias → canonical model ID map applied to locally created workers.
    pub fn model_aliases(mut self, aliases: HashMap<String, String>) -> Self {
        self.config.model_aliases = aliases;
        self
    }

    pub fn disable_tokenizer_autoload(mut self, disable: bool) -> Self {
        self.config.disable_tokenizer_autoload = disable;
        self
    }

    /// Which parser reads a chat completion's output unless a request's header says otherwise.
    pub fn parser(mut self, parser: ParserChoice) -> Self {
        self.config.parser = parser;
        self
    }

    // ==================== History Backend ====================

    pub fn history_backend(mut self, backend: HistoryBackend) -> Self {
        self.config.history_backend = backend;
        self
    }

    pub fn oracle_history(mut self, oracle_config: OracleConfig) -> Self {
        self.config.history_backend = HistoryBackend::Oracle;
        self.config.oracle = Some(oracle_config);
        self
    }

    // ==================== Parsers ====================

    pub fn reasoning_parser<S: Into<String>>(mut self, parser: S) -> Self {
        self.config.reasoning_parser = Some(parser.into());
        self
    }

    pub fn tool_call_parser<S: Into<String>>(mut self, parser: S) -> Self {
        self.config.tool_call_parser = Some(parser.into());
        self
    }

    // ==================== Tokenizer Cache ====================

    pub fn tokenizer_cache(mut self, cache: TokenizerCacheConfig) -> Self {
        self.config.tokenizer_cache = cache;
        self
    }

    // ==================== Boolean Setters ====================
    // Accept bool parameters to conditionally set flags without if statements

    /// Deprecated, accepted and ignored: every registration reads an
    /// engine's data-parallel size and registers one worker per rank.
    pub fn dp_aware(mut self, enable: bool) -> Self {
        self.config.dp_aware = enable;
        self
    }

    pub fn routing_key_override(mut self, config: RoutingKeyOverrideConfig) -> Self {
        self.config.routing_key_override = config;
        self
    }

    pub fn pd_pairing_mode(mut self, mode: PdPairingMode) -> Self {
        self.config.pd_pairing_mode = mode;
        self
    }

    pub fn grouped_dp_sticky(mut self, mode: GroupedDpSticky) -> Self {
        self.config.grouped_dp_sticky = mode;
        self
    }

    /// Inverse of disable_retries field
    pub fn retries(mut self, enable: bool) -> Self {
        self.config.disable_retries = !enable;
        self
    }

    /// Inverse of disable_circuit_breaker field
    pub fn circuit_breaker(mut self, enable: bool) -> Self {
        self.config.disable_circuit_breaker = !enable;
        self
    }

    pub fn igw(mut self, enable: bool) -> Self {
        self.config.enable_igw = enable;
        self
    }

    pub fn rl(mut self, rl: smg_rl::RlConfig) -> Self {
        self.config.rl = rl;
        self
    }

    pub fn dp_minimum_tokens_scheduler(mut self, enable: bool) -> Self {
        self.config.dp_minimum_tokens_scheduler = enable;
        self
    }

    // ==================== Option Setters ====================
    // Accept Option<T> and only set if Some

    pub fn maybe_api_key(mut self, key: Option<impl Into<String>>) -> Self {
        if let Some(k) = key {
            self.config.api_key = Some(k.into());
        }
        self
    }

    pub fn maybe_discovery(mut self, discovery: Option<DiscoveryConfig>) -> Self {
        self.config.discovery = discovery;
        self
    }

    pub fn maybe_metrics(mut self, metrics: Option<MetricsConfig>) -> Self {
        self.config.metrics = metrics;
        self
    }

    pub fn maybe_trace(mut self, trace_config: Option<TraceConfig>) -> Self {
        self.config.trace_config = trace_config;
        self
    }

    pub fn maybe_log_dir(mut self, dir: Option<impl Into<String>>) -> Self {
        self.config.log_dir = dir.map(|d| d.into());
        self
    }

    pub fn maybe_jemalloc_prof_dir(mut self, dir: Option<impl Into<String>>) -> Self {
        self.config.jemalloc_prof_dir = dir.map(|d| d.into());
        self
    }

    pub fn maybe_log_level(mut self, level: Option<impl Into<String>>) -> Self {
        self.config.log_level = level.map(|l| l.into());
        self
    }

    pub fn maybe_request_id_headers(mut self, headers: Option<Vec<String>>) -> Self {
        self.config.request_id_headers = headers;
        self
    }

    pub fn maybe_storage_context_headers(
        mut self,
        headers: Option<HashMap<String, String>>,
    ) -> Self {
        self.config.storage_context_headers = headers.unwrap_or_default();
        self
    }

    pub fn maybe_rate_limit_tokens_per_second(mut self, tokens: Option<i32>) -> Self {
        self.config.rate_limit_tokens_per_second = tokens;
        self
    }

    pub fn maybe_model_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.config.model_path = path.map(|p| p.into());
        self
    }

    pub fn maybe_tokenizer_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.config.tokenizer_path = path.map(|p| p.into());
        self
    }

    pub fn maybe_chat_template(mut self, template: Option<impl Into<String>>) -> Self {
        self.config.chat_template = template.map(|t| t.into());
        self
    }

    pub fn maybe_oracle(mut self, oracle: Option<OracleConfig>) -> Self {
        if let Some(cfg) = oracle {
            self.config.history_backend = HistoryBackend::Oracle;
            self.config.oracle = Some(cfg);
        }
        self
    }

    pub fn maybe_postgres(mut self, postgres: Option<PostgresConfig>) -> Self {
        if let Some(cfg) = postgres {
            self.config.history_backend = HistoryBackend::Postgres;
            self.config.postgres = Some(cfg);
        }
        self
    }

    pub fn maybe_redis(mut self, redis: Option<RedisConfig>) -> Self {
        if let Some(cfg) = redis {
            self.config.history_backend = HistoryBackend::Redis;
            self.config.redis = Some(cfg);
        }
        self
    }

    /// Bounds of the memory history backend.
    pub fn memory_config(mut self, memory: MemoryConfig) -> Self {
        self.config.memory = Some(memory);
        self
    }

    pub fn maybe_reasoning_parser(mut self, parser: Option<impl Into<String>>) -> Self {
        self.config.reasoning_parser = parser.map(|p| p.into());
        self
    }

    pub fn maybe_tool_call_parser(mut self, parser: Option<impl Into<String>>) -> Self {
        self.config.tool_call_parser = parser.map(|p| p.into());
        self
    }

    // ==================== mTLS ====================

    /// Files read during build()
    pub fn maybe_client_cert_and_key(
        mut self,
        cert_path: Option<impl Into<String>>,
        key_path: Option<impl Into<String>>,
    ) -> Self {
        self.client_cert_path = cert_path.map(|p| p.into());
        self.client_key_path = key_path.map(|p| p.into());
        self
    }

    /// Files read during build()
    pub fn add_ca_certificates<S: Into<String>>(mut self, ca_cert_paths: Vec<S>) -> Self {
        self.ca_cert_paths
            .extend(ca_cert_paths.into_iter().map(|p| p.into()));
        self
    }

    // ==================== Server TLS ====================

    /// Files read during build()
    pub fn maybe_server_cert_and_key(
        mut self,
        cert_path: Option<impl Into<String>>,
        key_path: Option<impl Into<String>>,
    ) -> Self {
        self.server_cert_path = cert_path.map(|p| p.into());
        self.server_key_path = key_path.map(|p| p.into());
        self
    }

    // ==================== MCP ====================

    /// Config file loaded during build()
    pub fn mcp_config_path<S: Into<String>>(mut self, path: S) -> Self {
        self.mcp_config_path = Some(path.into());
        self
    }

    /// Config file loaded during build()
    pub fn maybe_mcp_config_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.mcp_config_path = path.map(|p| p.into());
        self
    }

    /// Load the policy document from this path during `build()`.
    pub fn maybe_policy_spec_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.config.policy_spec_path = path.map(Into::into);
        self
    }

    // ==================== Build ====================

    pub fn build(self) -> ConfigResult<RouterConfig> {
        self.build_with_validation(true)
    }

    pub fn build_unchecked(self) -> RouterConfig {
        self.into()
    }

    /// The configuration of a builder that holds the routing part alone,
    /// checked by the validator's routing rules alone. No file is read: no
    /// certificate, no key, no MCP configuration and no policy document.
    /// What `--print-policy-spec` converts.
    pub fn build_routing(self) -> ConfigResult<RouterConfig> {
        let config: RouterConfig = self.into();
        ConfigValidator::validate_routing(&config)?;
        Ok(config)
    }

    pub fn build_with_validation(mut self, validate: bool) -> ConfigResult<RouterConfig> {
        // Read mTLS certificates from paths if provided
        self = self.read_mtls_certificates()?;

        // Read Server TLS certificates from paths if provided
        self = self.read_server_certificates()?;

        // Read MCP config from path if provided
        self = self.read_mcp_config()?;
        self = self.read_policy_spec()?;

        let config: RouterConfig = self.into();
        if validate {
            config.validate()?;
        }
        Ok(config)
    }

    /// Internal method to read mTLS certificates from paths
    fn read_mtls_certificates(mut self) -> ConfigResult<Self> {
        let pair = cert_and_key(
            self.client_cert_path.as_deref(),
            self.client_key_path.as_deref(),
            "--client-cert-path and --client-key-path",
        )?;
        if let Some((cert_path, key_path)) = pair {
            let cert = read_file(cert_path, "client certificate")?;
            let key = read_file(key_path, "client key")?;

            // Combine cert and key into single PEM for reqwest::Identity
            // When using rustls, certificate must come first, then key
            // Ensure proper PEM formatting with newlines
            let mut combined = cert;
            if !combined.ends_with(b"\n") {
                combined.push(b'\n');
            }
            combined.extend_from_slice(&key);
            if !combined.ends_with(b"\n") {
                combined.push(b'\n');
            }

            self.config.client_identity = Some(combined);
            self.config.client_key_path = Some(key_path.to_string());
        }

        for path in &self.ca_cert_paths {
            let cert = read_file(path, "CA certificate")?;
            self.config.ca_certificates.push(cert);
        }

        Ok(self)
    }

    /// Internal method to read Server TLS certificates from paths
    fn read_server_certificates(mut self) -> ConfigResult<Self> {
        let pair = cert_and_key(
            self.server_cert_path.as_deref(),
            self.server_key_path.as_deref(),
            "--tls-cert-path and --tls-key-path",
        )?;
        if let Some((cert_path, key_path)) = pair {
            self.config.server_cert = Some(read_file(cert_path, "server certificate")?);
            self.config.server_key = Some(read_file(key_path, "server key")?);
            self.config.server_key_path = Some(key_path.to_string());
        }
        Ok(self)
    }

    /// Internal method to read MCP config from path
    fn read_mcp_config(mut self) -> ConfigResult<Self> {
        if let Some(mcp_config_path) = &self.mcp_config_path {
            let contents = std::fs::read_to_string(mcp_config_path).map_err(|e| {
                ConfigError::ValidationFailed {
                    reason: format!("Failed to read MCP config from {mcp_config_path}: {e}"),
                }
            })?;
            let mcp_config: McpConfig =
                serde_yaml::from_str(&contents).map_err(|e| ConfigError::ValidationFailed {
                    reason: format!("Failed to parse MCP config from {mcp_config_path}: {e}"),
                })?;
            self.config.mcp_config = Some(mcp_config);
        }

        Ok(self)
    }

    fn read_policy_spec(mut self) -> ConfigResult<Self> {
        if let Some(path) = &self.config.policy_spec_path {
            let text = std::fs::File::open(path)
                .and_then(|file| Ok((file.metadata()?.len(), file)))
                .and_then(|(stated, file)| bounded_document(stated, file))
                .map_err(|error| ConfigError::ValidationFailed {
                    reason: format!("{path}: could not read the policy document ({error})"),
                })?;
            self.config.policy_spec = Some(serde_yaml::from_str(&text).map_err(|error| {
                ConfigError::ValidationFailed {
                    reason: format!("{path}: {error}"),
                }
            })?);
        }
        Ok(self)
    }
}

fn cert_and_key<'a>(
    cert_path: Option<&'a str>,
    key_path: Option<&'a str>,
    flags: &str,
) -> ConfigResult<Option<(&'a str, &'a str)>> {
    match (cert_path, key_path) {
        (Some(cert_path), Some(key_path)) => Ok(Some((cert_path, key_path))),
        (None, None) => Ok(None),
        _ => Err(ConfigError::ValidationFailed {
            reason: format!("Both {flags} must be specified together"),
        }),
    }
}

fn read_file(path: &str, what: &str) -> ConfigResult<Vec<u8>> {
    std::fs::read(path).map_err(|e| ConfigError::ValidationFailed {
        reason: format!("Failed to read {what} from {path}: {e}"),
    })
}

impl From<RouterConfigBuilder> for RouterConfig {
    fn from(builder: RouterConfigBuilder) -> Self {
        builder.config
    }
}

impl RouterConfig {
    /// Create a builder for RouterConfig
    pub fn builder() -> RouterConfigBuilder {
        RouterConfigBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use smg_policy_controller::document::PolicySpec;
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn a_document_that_cannot_be_read_stops_the_build_by_its_path() {
        let dir = tempdir().expect("a temporary directory");
        let path = dir.path().join("missing.yaml");
        let path = path.to_str().expect("a UTF-8 path");
        let io_error = std::fs::read_to_string(path).expect_err("the file is absent");
        let refusal = RouterConfig::builder()
            .policy_controller(true)
            .maybe_policy_spec_path(Some(path))
            .build()
            .expect_err("an unreadable file is refused");
        assert!(matches!(
            refusal,
            ConfigError::ValidationFailed { reason } if reason.contains(path)
                && reason.contains(&io_error.to_string())
        ));
    }

    #[test]
    fn a_document_the_reader_refuses_stops_the_build_in_its_words() {
        let dir = tempdir().expect("a temporary directory");
        let path = dir.path().join("invalid.yaml");
        let path = path.to_str().expect("a UTF-8 path");
        let text = "apiVersion: smg.policy/v1\nprofiles: {}\nunknown: true\n";
        std::fs::write(path, text).expect("the document is written");
        let read_error = serde_yaml::from_str::<PolicySpec>(text).expect_err("unknown field");
        let refusal = RouterConfig::builder()
            .policy_controller(true)
            .maybe_policy_spec_path(Some(path))
            .build()
            .expect_err("an invalid document is refused");
        assert!(matches!(
            refusal,
            ConfigError::ValidationFailed { reason } if reason.contains(path)
                && reason.contains(&read_error.to_string())
        ));
    }

    #[test]
    fn a_document_with_the_switch_off_is_refused() {
        let dir = tempdir().expect("a temporary directory");
        let path = dir.path().join("routing.yaml");
        let path = path.to_str().expect("a UTF-8 path");
        std::fs::write(
            path,
            "apiVersion: smg.policy/v1\nprofiles: {default: {pick: random}}\n",
        )
        .expect("the document is written");
        let refusal = RouterConfig::builder()
            .igw(true)
            .maybe_policy_spec_path(Some(path))
            .build()
            .expect_err("the document needs the controller");
        assert!(matches!(
            refusal,
            ConfigError::InvalidValue { field, value, reason }
                if field == "policy_spec" && value == path
                    && reason == "a policy document is run by the policy controller: set --policy-controller on"
        ));
    }

    #[test]
    fn a_document_given_without_its_path_needs_the_switch_too() {
        // A host hands the document itself, and no launcher stands before it.
        let document = PolicySpec::preset("random").expect("the preset");
        let config = RouterConfig {
            enable_igw: true,
            policy_spec: Some(document),
            ..RouterConfig::default()
        };
        assert!(matches!(
            config.validate().expect_err("the document needs the controller"),
            ConfigError::InvalidValue { field, value, reason }
                if field == "policy_spec" && value == "the given document"
                    && reason == "a policy document is run by the policy controller: set --policy-controller on"
        ));
    }

    #[test]
    fn a_path_with_no_loaded_document_is_refused() {
        let config = RouterConfig {
            enable_igw: true,
            policy_controller: true,
            policy_spec_path: Some("routing.yaml".to_owned()),
            ..RouterConfig::default()
        };
        assert!(matches!(
            config.validate().expect_err("the document was never read"),
            ConfigError::InvalidValue { field, value, reason }
                if field == "policy_spec" && value == "routing.yaml"
                    && reason.starts_with("the policy document of this path is not loaded")
        ));
    }

    /// A policy document of `len` bytes: a small one, then one YAML comment
    /// that fills the file to the length.
    fn document_of(len: usize) -> String {
        let document = "apiVersion: smg.policy/v1\nprofiles: {default: {pick: random}}\n";
        let filler = "x".repeat(len - document.len() - 2);
        format!("{document}#{filler}\n")
    }

    #[test]
    fn a_document_at_the_size_bound_is_read() {
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        let dir = tempdir().expect("a temporary directory");
        let path = dir.path().join("bound.yaml");
        let path = path.to_str().expect("a UTF-8 path");
        std::fs::write(path, document_of(bound)).expect("the document is written");
        let config = RouterConfig::builder()
            .policy_controller(true)
            .maybe_policy_spec_path(Some(path))
            .build_with_validation(false)
            .expect("a document at the bound is read");
        assert!(config.policy_spec.is_some());
    }

    #[test]
    fn a_document_over_the_size_bound_stops_the_build_by_its_size() {
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        let dir = tempdir().expect("a temporary directory");
        let path = dir.path().join("over.yaml");
        let path = path.to_str().expect("a UTF-8 path");
        // A document the reader would take, five bytes past the bound: the
        // refusal states the file's whole size, so the file was not read.
        let over = bound + 5;
        std::fs::write(path, document_of(over)).expect("the document is written");
        let refusal = RouterConfig::builder()
            .policy_controller(true)
            .maybe_policy_spec_path(Some(path))
            .build_with_validation(false)
            .expect_err("a document over the bound is refused");
        assert!(matches!(
            refusal,
            ConfigError::ValidationFailed { reason } if reason == format!(
                "{path}: could not read the policy document (the document is {over} bytes; at most {bound})"
            )
        ));
    }

    #[test]
    fn a_source_that_holds_more_than_it_states_is_read_one_byte_past_the_bound() {
        let bound = usize::try_from(MAX_DOCUMENT_BYTES).expect("the bound fits");
        // A source that states no size and holds ten bytes past the bound.
        let source = std::io::repeat(b'#').take(MAX_DOCUMENT_BYTES + 10);
        let refusal = bounded_document(0, source).expect_err("the source is past the bound");
        assert_eq!(refusal.kind(), std::io::ErrorKind::FileTooLarge);
        assert_eq!(
            refusal.to_string(),
            format!("the document is {} bytes; at most {bound}", bound + 1)
        );
    }

    /// Test complex routing mode helper method
    #[test]
    fn test_builder_prefill_decode_mode() {
        let config = RouterConfigBuilder::new()
            .prefill_decode_mode(
                vec![("http://prefill:8000".to_string(), Some(8001))],
                vec!["http://decode:8000".to_string()],
            )
            .power_of_two_policy()
            .build()
            .unwrap();

        assert!(config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 2);
    }

    /// Test complex policy helper method with multiple parameters
    #[test]
    fn test_builder_cache_aware_policy() {
        let config = RouterConfigBuilder::new()
            .regular_mode(vec!["http://worker1:8000".to_string()])
            .cache_aware_policy(0.8, 10, 1.5, 300, 1000)
            .build()
            .unwrap();

        match config.policy {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.8).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware policy"),
        }
    }

    #[test]
    fn test_builder_storage_context_headers_round_trip() {
        let headers = HashMap::from([("x-tenant-id".to_string(), "tenant_id".to_string())]);

        let config = RouterConfigBuilder::new()
            .regular_mode(vec!["http://worker1:8000".to_string()])
            .storage_context_headers(headers.clone())
            .build()
            .unwrap();

        assert_eq!(config.storage_context_headers, headers);
    }
}
