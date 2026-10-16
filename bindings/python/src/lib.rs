use std::collections::HashMap;

use once_cell::sync::OnceCell;
use pyo3::prelude::*;

// Jemalloc for all Rust-side allocations in the extension. Prefixed symbols
// leave CPython's allocators untouched; disable_initial_exec_tls is required
// for a dlopen'd cdylib.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[global_allocator]
static GLOBAL_ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// The gateway's jemalloc options (`SERVER_MALLOC_CONF`). The extension is a
// final artifact of its own, so it exports them as jemalloc's
// application-provided `malloc_conf` like the `smg` executable does;
// without this the router launched from Python ran jemalloc's stock decay
// and kept a traffic burst's freed pages resident.
#[cfg(all(not(target_env = "msvc"), not(target_env = "musl")))]
#[expect(
    unsafe_code,
    reason = "jemalloc reads its options from this exported symbol; a NUL-terminated byte string nothing in Rust dereferences"
)]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 72] = observability::metrics::SERVER_MALLOC_CONF;
use smg::*;
use smg_auth as auth;

mod servicer;
use servicer::{
    init_servicer_tracing, PySglangGrpcServer, PyTokenSpeedGrpcServer, PyVllmGrpcServer,
};

// Define the enums with PyO3 bindings
#[pyclass(eq, from_py_object)]
#[derive(Clone, PartialEq, Debug)]
pub enum PolicyType {
    Random,
    RoundRobin,
    Passthrough,
    CacheAware,
    PowerOfTwo,
    LeastLoad,
    Bucket,
    Manual,
    ConsistentHashing,
    PrefixHash,
}

#[pyclass(eq, from_py_object)]
#[derive(Clone, PartialEq, Debug)]
pub enum BackendType {
    Sglang,
    Openai,
    Anthropic,
    /// vLLM engine. Routing behaves like the default; over ZMQ this pins the
    /// startup workers' wire protocol to vLLM EngineCore.
    Vllm,
    /// TokenSpeed engine. Routing behaves like the default; over ZMQ this pins
    /// the startup workers' wire protocol to TokenSpeed.
    Tokenspeed,
}

#[pyclass(eq, from_py_object)]
#[derive(Clone, PartialEq, Debug)]
pub enum HistoryBackendType {
    Memory,
    None,
    Oracle,
    Postgres,
    Redis,
}

#[pyclass(eq, from_py_object)]
#[derive(Clone, PartialEq, Debug, Default)]
pub enum PyRole {
    Admin,
    #[default]
    User,
}

impl PyRole {
    pub fn to_auth_role(&self) -> auth::Role {
        match self {
            PyRole::Admin => auth::Role::Admin,
            PyRole::User => auth::Role::User,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct PyApiKeyEntry {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub key: String,
    #[pyo3(get, set)]
    pub role: PyRole,
}

#[pymethods]
impl PyApiKeyEntry {
    #[new]
    #[pyo3(signature = (id, name, key, role = PyRole::User))]
    fn new(id: String, name: String, key: String, role: PyRole) -> Self {
        PyApiKeyEntry {
            id,
            name,
            key,
            role,
        }
    }
}

impl PyApiKeyEntry {
    pub fn to_auth_api_key_entry(&self) -> auth::ApiKeyEntry {
        auth::ApiKeyEntry::new(&self.id, &self.name, &self.key, self.role.to_auth_role())
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq)]
pub struct PyJwtConfig {
    #[pyo3(get, set)]
    pub issuer: String,
    #[pyo3(get, set)]
    pub audience: String,
    #[pyo3(get, set)]
    pub jwks_uri: Option<String>,
    #[pyo3(get, set)]
    pub role_mapping: HashMap<String, String>,
}

#[pymethods]
impl PyJwtConfig {
    #[new]
    #[pyo3(signature = (
        issuer,
        audience,
        jwks_uri = None,
        role_mapping = HashMap::new(),
    ))]
    fn new(
        issuer: String,
        audience: String,
        jwks_uri: Option<String>,
        role_mapping: HashMap<String, String>,
    ) -> Self {
        PyJwtConfig {
            issuer,
            audience,
            jwks_uri,
            role_mapping,
        }
    }
}

impl PyJwtConfig {
    pub fn to_auth_jwt_config(&self) -> auth::JwtConfig {
        let mut config = auth::JwtConfig::new(&self.issuer, &self.audience);

        // Conditionally set JWKS URI
        if let Some(ref uri) = self.jwks_uri {
            config = config.with_jwks_uri(uri);
        }

        // Add role mappings
        for (idp_role, gateway_role) in &self.role_mapping {
            let role = match gateway_role.to_lowercase().as_str() {
                "admin" => auth::Role::Admin,
                _ => auth::Role::User,
            };
            config = config.with_role_mapping(idp_role, role);
        }

        config
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyControlPlaneAuthConfig {
    #[pyo3(get, set)]
    pub jwt: Option<PyJwtConfig>,
    #[pyo3(get, set)]
    pub api_keys: Vec<PyApiKeyEntry>,
    #[pyo3(get, set)]
    pub audit_enabled: bool,
}

#[pymethods]
impl PyControlPlaneAuthConfig {
    #[new]
    #[pyo3(signature = (
        jwt = None,
        api_keys = vec![],
        audit_enabled = true,
    ))]
    fn new(jwt: Option<PyJwtConfig>, api_keys: Vec<PyApiKeyEntry>, audit_enabled: bool) -> Self {
        PyControlPlaneAuthConfig {
            jwt,
            api_keys,
            audit_enabled,
        }
    }
}

impl PyControlPlaneAuthConfig {
    pub fn to_auth_control_plane_config(&self) -> auth::ControlPlaneAuthConfig {
        auth::ControlPlaneAuthConfig {
            jwt: self.jwt.as_ref().map(|j| j.to_auth_jwt_config()),
            api_keys: self
                .api_keys
                .iter()
                .map(|k| k.to_auth_api_key_entry())
                .collect(),
            audit_enabled: self.audit_enabled,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyOracleConfig {
    #[pyo3(get, set)]
    pub wallet_path: Option<String>,
    #[pyo3(get, set)]
    pub connect_descriptor: Option<String>,
    #[pyo3(get, set)]
    pub external_auth: bool,
    #[pyo3(get, set)]
    pub username: Option<String>,
    #[pyo3(get, set)]
    pub password: Option<String>,
    #[pyo3(get, set)]
    pub pool_min: usize,
    #[pyo3(get, set)]
    pub pool_max: usize,
    #[pyo3(get, set)]
    pub pool_timeout_secs: u64,
}

impl std::fmt::Debug for PyOracleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyOracleConfig")
            .field("wallet_path", &self.wallet_path)
            .field("connect_descriptor", &"<redacted>")
            .field("external_auth", &self.external_auth)
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .field("pool_min", &self.pool_min)
            .field("pool_max", &self.pool_max)
            .field("pool_timeout_secs", &self.pool_timeout_secs)
            .finish()
    }
}

#[pymethods]
impl PyOracleConfig {
    #[expect(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (
        password = None,
        username = None,
        connect_descriptor = None,
        wallet_path = None,
        external_auth = false,
        pool_min = 1,
        pool_max = 16,
        pool_timeout_secs = 30,
    ))]
    fn new(
        password: Option<String>,
        username: Option<String>,
        connect_descriptor: Option<String>,
        wallet_path: Option<String>,
        external_auth: bool,
        pool_min: usize,
        pool_max: usize,
        pool_timeout_secs: u64,
    ) -> PyResult<Self> {
        if pool_min == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pool_min must be at least 1",
            ));
        }
        if pool_max < pool_min {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pool_max must be >= pool_min",
            ));
        }

        Ok(PyOracleConfig {
            wallet_path,
            connect_descriptor,
            external_auth,
            username,
            password,
            pool_min,
            pool_max,
            pool_timeout_secs,
        })
    }
}

impl PyOracleConfig {
    pub fn to_config_oracle(&self) -> config::OracleConfig {
        config::OracleConfig {
            wallet_path: self.wallet_path.clone(),
            connect_descriptor: self.connect_descriptor.clone().unwrap_or_default(),
            external_auth: self.external_auth,
            username: self.username.clone().unwrap_or_default(),
            password: self.password.clone().unwrap_or_default(),
            pool_min: self.pool_min,
            pool_max: self.pool_max,
            pool_timeout_secs: self.pool_timeout_secs,
            schema: None,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyRedisConfig {
    #[pyo3(get, set)]
    pub url: String,
    #[pyo3(get, set)]
    pub pool_max: usize,
    #[pyo3(get, set)]
    pub retention_days: Option<u64>,
    /// Bound in seconds on each call to the store (default 5).
    #[pyo3(get, set)]
    pub op_timeout_secs: u64,
}

#[pymethods]
impl PyRedisConfig {
    #[new]
    #[pyo3(signature = (url, pool_max = 16, retention_days = Some(30), op_timeout_secs = 5))]
    #[expect(
        clippy::unnecessary_wraps,
        reason = "PyO3 #[new] method signature requires PyResult"
    )]
    fn new(
        url: String,
        pool_max: usize,
        retention_days: Option<u64>,
        op_timeout_secs: u64,
    ) -> PyResult<Self> {
        Ok(PyRedisConfig {
            url,
            pool_max,
            retention_days,
            op_timeout_secs,
        })
    }
}

impl PyRedisConfig {
    pub fn to_config_redis(&self) -> config::RedisConfig {
        config::RedisConfig {
            url: self.url.clone(),
            pool_max: self.pool_max,
            op_timeout_secs: self.op_timeout_secs,
            retention_days: self.retention_days,
            schema: None,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct PyPostgresConfig {
    #[pyo3(get, set)]
    pub db_url: Option<String>,

    #[pyo3(get, set)]
    pub pool_max: usize,

    /// Bound in seconds on each call to the store (default 5).
    #[pyo3(get, set)]
    pub op_timeout_secs: u64,
}

#[pymethods]
impl PyPostgresConfig {
    #[new]
    #[pyo3(signature = (db_url = None, pool_max = 16, op_timeout_secs = 5))]
    #[expect(
        clippy::unnecessary_wraps,
        reason = "PyO3 #[new] method signature requires PyResult"
    )]
    fn new(db_url: Option<String>, pool_max: usize, op_timeout_secs: u64) -> PyResult<Self> {
        Ok(PyPostgresConfig {
            db_url,
            pool_max,
            op_timeout_secs,
        })
    }
}

impl PyPostgresConfig {
    pub fn to_config_postgres(&self) -> config::PostgresConfig {
        config::PostgresConfig {
            db_url: self.db_url.clone().unwrap_or_default(),
            pool_max: self.pool_max,
            op_timeout_secs: self.op_timeout_secs,
            schema: None,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone, PartialEq)]
struct Router {
    // The constructor's positional parameters, in their order.
    worker_urls: Vec<String>,
    policy: PolicyType,
    host: String,
    port: u16,
    // The keyword-only parameters, in alphabetical order, as the `#[pyo3(signature)]`
    // below lists them: a new one goes at its place in each of the four lists (this
    // struct, the signature, the parameters of `new`, the literal in `new`), never at
    // the tail, so two additions do not touch the same lines.
    /// The optional fields of the access-log record.
    access_log_fields: Vec<String>,
    admin_host: Option<String>,
    admin_port: Option<u16>,
    api_key: Option<String>,
    assignment_mode: Option<String>,
    backend: BackendType,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    balance_token_usage_threshold: f32,
    block_size: usize,
    bootstrap_port_annotation: String,
    bucket_adjust_interval_secs: usize,
    ca_cert_paths: Vec<String>,
    cache_boundaries: Vec<usize>,
    cache_index: String,
    cache_threshold: f32,
    cache_ttl_secs: u64,
    cb_failure_threshold: u32,
    cb_success_threshold: u32,
    cb_timeout_duration_secs: u64,
    cb_window_duration_secs: u64,
    chat_template: Option<String>,
    client_cert_path: Option<String>,
    client_key_path: Option<String>,
    control_plane_auth: Option<PyControlPlaneAuthConfig>,
    cors_allowed_origins: Vec<String>,
    decode_policy: Option<PolicyType>,
    decode_selector: HashMap<String, String>,
    decode_urls: Option<Vec<String>>,
    disable_circuit_breaker: bool,
    disable_health_check: bool,
    disable_load_monitoring: bool,
    disable_retries: bool,
    disable_tokenizer_autoload: bool,
    /// The `discovery` mapping, read by the same rules as
    /// `RouterConfig.discovery`.
    discovery: Option<config::DiscoveryConfig>,
    dp_aware: bool,
    dp_minimum_tokens_scheduler: bool,
    drain_settle_secs: u64,
    enable_igw: bool,
    enable_mesh: bool,
    enable_rl: bool,
    enable_trace: bool,
    enable_wasm: bool,
    encode_policy: Option<PolicyType>,
    encode_selector: HashMap<String, String>,
    encode_urls: Option<Vec<(String, Option<u16>)>>,
    epd_disaggregation: bool,
    eviction_interval_secs: u64,
    header_read_timeout_secs: u64,
    health_check_endpoint: String,
    health_check_interval_secs: u64,
    health_check_port: Option<u16>,
    health_check_timeout_secs: u64,
    health_failure_threshold: u32,
    health_success_threshold: u32,
    history_backend: HistoryBackendType,
    history_memory_max_items: usize,
    history_memory_max_responses: usize,
    history_memory_retention_secs: u64,
    idle_connection_timeout_secs: u64,
    jemalloc_prof_dir: Option<String>,
    job_queue_capacity: usize,
    job_queue_concurrency: usize,
    kv_connector_annotation: String,
    kv_engine_id_annotation: String,
    kv_index: Option<String>,
    least_load_default_throughput: f64,
    least_load_kv_pressure_weight: f64,
    least_load_max_waiting_requests: u32,
    least_load_mean_prefill_tokens: u32,
    load_monitor_interval: u64,
    log_dir: Option<String>,
    log_json: bool,
    log_level: Option<String>,
    log_mm_timing: bool,
    max_background_responses: usize,
    max_buffered_request_bytes: u64,
    max_concurrent_requests: i32,
    max_idle_secs: u64,
    max_payload_size: usize,
    max_tree_size: usize,
    mcp_config_path: Option<String>,
    mesh_advertise_host: Option<String>,
    mesh_host: String,
    mesh_peer_urls: Vec<String>,
    mesh_port: u16,
    mesh_server_name: Option<String>,
    /// The mesh CA, this node's certificate and key (PEM
    /// paths); all three or none.
    mesh_tls_ca_cert: Option<String>,
    mesh_tls_cert: Option<String>,
    mesh_tls_key: Option<String>,
    mm_allow_private_media_hosts: bool,
    mm_allowed_media_domains: Option<Vec<String>>,
    mm_per_request_image_limit: Option<usize>,
    mm_pixel_cache_mb: Option<usize>,
    mm_pixel_rdma: bool,
    mm_processing: Option<String>,
    model_aliases: HashMap<String, String>,
    model_id_from: Option<String>,
    model_path: Option<String>,
    multimodal_max_inflight_bytes: Option<usize>,
    multimodal_shm_min_bytes: Option<usize>,
    multimodal_tensor_transport: Option<String>,
    oracle_config: Option<PyOracleConfig>,
    otlp_traces_endpoint: String,
    overlap_decay: f32,
    overload_token_usage_threshold: f32,
    /// `legacy` or `symphony`: which parser reads a chat completion's output.
    parser: String,
    pd_admission_wait_secs: u64,
    pd_disaggregation: bool,
    postgres_config: Option<PyPostgresConfig>,
    power_of_two_load_metric: String,
    prefill_max_inflight_requests_per_worker: i32,
    prefill_policy: Option<PolicyType>,
    prefill_queue_size: Option<usize>,
    prefill_queue_timeout_secs: Option<u64>,
    prefill_selector: HashMap<String, String>,
    prefill_urls: Option<Vec<(String, Option<u16>)>>,
    prefix_hash_balance_abs_threshold: usize,
    prefix_hash_load_factor: f64,
    prefix_token_count: usize,
    priority_scheduler_config: Option<String>,
    priority_scheduler_default_max_class: String,
    priority_scheduler_enabled: bool,
    priority_scheduler_tenant_metric_top_n: u32,
    prometheus_duration_buckets: Option<Vec<f64>>,
    prometheus_host: Option<String>,
    prometheus_port: Option<u16>,
    queue_size: usize,
    queue_timeout_secs: u64,
    rate_limit_tokens_per_second: Option<i32>,
    rdma_listen_ip: Option<String>,
    rdma_slot_ttl_s: Option<u64>,
    reasoning_parser: Option<String>,
    redis_config: Option<PyRedisConfig>,
    remove_unhealthy_workers: Option<bool>,
    request_id_headers: Option<Vec<String>>,
    request_timeout_secs: u64,
    retry_backoff_multiplier: f32,
    retry_initial_backoff_ms: u64,
    retry_jitter_factor: f32,
    retry_max_backoff_ms: u64,
    retry_max_retries: u32,
    rl_control_timeout_secs: u64,
    rl_fanout_concurrency: usize,
    router_selector: HashMap<String, String>,
    routing_key_headers: Vec<String>,
    routing_key_override: bool,
    schema_config: Option<String>,
    selection_accounting_ttl_ms: u64,
    selection_policy: String,
    selection_temperature: f32,
    selector: HashMap<String, String>,
    server_cert_path: Option<String>,
    server_key_path: Option<String>,
    service_discovery: bool,
    service_discovery_namespace: Option<String>,
    service_discovery_port: u16,
    shutdown_grace_period_secs: u64,
    sse_keepalive_secs: u64,
    storage_context_headers: HashMap<String, String>,
    storage_hook_wasm_path: Option<String>,
    stream_body_stall_timeout_secs: u64,
    /// Per-tenant data-plane keys as `(tenant_id, key)` pairs; each resolves
    /// to its own tenant identity (`auth:<tenant_id>`) on top of `api_key`.
    tenant_api_keys: Vec<(String, String)>,
    tenant_header_name: String,
    tenant_rate_limit_config: Option<String>,
    tenant_rate_limit_enabled: bool,
    tokenizer_cache_enable_l0: bool,
    tokenizer_cache_enable_l1: bool,
    tokenizer_cache_l0_max_entries: usize,
    tokenizer_cache_l0_max_memory: usize,
    tokenizer_cache_l1_max_memory: usize,
    tokenizer_path: Option<String>,
    tool_call_parser: Option<String>,
    trust_tenant_header: bool,
    upstream_http2: bool,
    upstream_pool_idle_timeout_secs: u64,
    worker_overload_protection: bool,
    worker_overload_shed: bool,
    worker_overload_token_usage: Option<f64>,
    worker_overload_waiting_requests: Option<usize>,
    worker_ports_annotation: String,
    worker_stale_secs: u64,
    worker_stall_secs: u64,
    worker_startup_check_interval: u64,
    worker_startup_delay: u64,
    worker_startup_timeout_secs: u64,
    worker_warmup_blocks: usize,
    worker_warmup_divert_every: u64,
    worker_warmup_secs: u64,
    worker_warmup_share: f32,
    worker_warmup_thin_ratio: f32,
    worker_wedge_secs: u64,
    /// DP engines per startup ZMQ worker (grouped worker; None/1 = ungrouped).
    zmq_engine_count: Option<usize>,
    // Derived from the URL lists; not a parameter.
    connection_mode: worker::ConnectionMode,
}

/// Read the keyword-only `discovery` mapping by the same rules as
/// `RouterConfig.discovery`. It goes through JSON, so nested mappings and
/// lists convert without a hand-written walker.
fn parse_discovery(mapping: &Bound<'_, PyAny>) -> PyResult<Option<config::DiscoveryConfig>> {
    let invalid = |e: serde_json::Error| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid discovery mapping: {e}"))
    };
    let json: String = PyModule::import(mapping.py(), "json")?
        .call_method1("dumps", (mapping,))?
        .extract()?;
    let value: serde_json::Value = serde_json::from_str(&json).map_err(invalid)?;
    config::deserialize_discovery(value).map_err(invalid)
}

impl Router {
    fn determine_connection_mode(worker_urls: &[String]) -> worker::ConnectionMode {
        use worker::ConnectionMode;
        // First worker URL that declares ipc:// or grpc:// wins; http:// and bare
        // host:port fall through to the HTTP default. See ConnectionMode::from_url.
        worker_urls
            .iter()
            .find_map(|url| match ConnectionMode::from_url(url) {
                mode @ (Some(ConnectionMode::Zmq) | Some(ConnectionMode::Grpc)) => mode,
                _ => None,
            })
            .unwrap_or(ConnectionMode::Http)
    }

    /// The metrics bind host: `prometheus_host`, or the unspecified address
    /// of `host`'s family when it is not given.
    fn metrics_host(&self) -> String {
        self.prometheus_host
            .clone()
            .unwrap_or_else(|| config::MetricsConfig::default_host_for(&self.host))
    }

    /// The mesh mTLS configuration from `mesh_tls_ca_cert`, `mesh_tls_cert`
    /// and `mesh_tls_key`: all three or none, each a readable file.
    fn mesh_mtls_config(&self) -> PyResult<Option<smg_mesh::MTLSConfig>> {
        let (ca, cert, key) = match (
            &self.mesh_tls_ca_cert,
            &self.mesh_tls_cert,
            &self.mesh_tls_key,
        ) {
            (None, None, None) => return Ok(None),
            (Some(ca), Some(cert), Some(key)) => (ca, cert, key),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "mesh_tls_ca_cert, mesh_tls_cert and mesh_tls_key come together: give all \
                     three or none",
                ))
            }
        };
        for (field, path) in [
            ("mesh_tls_ca_cert", ca),
            ("mesh_tls_cert", cert),
            ("mesh_tls_key", key),
        ] {
            std::fs::metadata(path).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid value for {field}='{path}': cannot read the file: {e}"
                ))
            })?;
        }
        Ok(Some(smg_mesh::MTLSConfig {
            ca_cert_path: ca.into(),
            server_cert_path: cert.into(),
            server_key_path: key.into(),
            ..smg_mesh::MTLSConfig::default()
        }))
    }

    fn parse_mesh_socket_addr(
        host: &str,
        port: u16,
        field: &str,
    ) -> PyResult<std::net::SocketAddr> {
        config::bind_socket_addr(host, port).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid value for {field}='{host}': invalid mesh socket address: {e}"
            ))
        })
    }

    fn parse_cache_index(&self) -> Result<config::CacheIndexKind, config::ConfigError> {
        match self.cache_index.as_str() {
            "tree" => Ok(config::CacheIndexKind::Tree),
            "hash" => Ok(config::CacheIndexKind::Hash),
            other => Err(config::ConfigError::InvalidValue {
                field: "cache_index".to_string(),
                value: other.to_string(),
                reason: "expected 'tree' or 'hash'".to_string(),
            }),
        }
    }

    fn parse_assignment_mode(
        &self,
        default: config::ManualAssignmentMode,
    ) -> Result<config::ManualAssignmentMode, config::ConfigError> {
        let Some(mode) = self.assignment_mode.as_deref() else {
            return Ok(default);
        };
        match mode {
            "random" => Ok(config::ManualAssignmentMode::Random),
            "min_load" => Ok(config::ManualAssignmentMode::MinLoad),
            "min_group" => Ok(config::ManualAssignmentMode::MinGroup),
            "delegate" => Ok(config::ManualAssignmentMode::Delegate),
            other => Err(config::ConfigError::InvalidValue {
                field: "assignment_mode".to_string(),
                value: other.to_string(),
                reason: "expected 'random', 'min_load', 'min_group', or 'delegate'".to_string(),
            }),
        }
    }

    pub fn to_router_config(&self) -> config::ConfigResult<config::RouterConfig> {
        use config::{
            DiscoveryConfig, KubernetesDiscoveryConfig, MetricsConfig,
            PolicyConfig as ConfigPolicyConfig, RoutingMode,
        };

        // Validate the transport mode up front. The CLI (value_parser) and the
        // argparse path (choices) already reject bad values; this covers direct
        // programmatic `RouterArgs` use, matching the CLI/Rust parsing contract.
        let multimodal_tensor_transport = self
            .multimodal_tensor_transport
            .as_deref()
            .map(|value| {
                config::TransportMode::parse(value).ok_or_else(|| {
                    config::ConfigError::InvalidValue {
                        field: "multimodal_tensor_transport".to_string(),
                        value: value.to_string(),
                        reason: "expected 'inline', 'shm', 'auto', or 'rdma'".to_string(),
                    }
                })
            })
            .transpose()?;
        let mm_processing = self
            .mm_processing
            .as_deref()
            .map(|value| {
                config::MmProcessingMode::parse(value).ok_or_else(|| {
                    config::ConfigError::InvalidValue {
                        field: "mm_processing".to_string(),
                        value: value.to_string(),
                        reason: "expected 'auto', 'router', or 'worker'".to_string(),
                    }
                })
            })
            .transpose()?;

        // The access-log fields by name, each spelled as the record keys it.
        let access_log_fields = self
            .access_log_fields
            .iter()
            .map(|name| {
                name.parse::<config::AccessLogField>().map_err(|reason| {
                    config::ConfigError::InvalidValue {
                        field: "access_log_fields".to_string(),
                        value: name.clone(),
                        reason,
                    }
                })
            })
            .collect::<config::ConfigResult<Vec<_>>>()?;

        let power_of_two_load_metric = config::PowerOfTwoLoadMetric::parse(
            &self.power_of_two_load_metric,
        )
        .ok_or_else(|| config::ConfigError::InvalidValue {
            field: "power_of_two_load_metric".to_string(),
            value: self.power_of_two_load_metric.clone(),
            reason: "expected expected_wait, requests or least_requests".to_string(),
        })?;

        let convert_policy = |policy: &PolicyType| -> config::ConfigResult<ConfigPolicyConfig> {
            Ok(match policy {
                PolicyType::Random => ConfigPolicyConfig::Random,
                PolicyType::RoundRobin => ConfigPolicyConfig::RoundRobin,
                PolicyType::Passthrough => ConfigPolicyConfig::Passthrough,
                PolicyType::CacheAware => ConfigPolicyConfig::CacheAware {
                    cache_threshold: self.cache_threshold,
                    balance_abs_threshold: self.balance_abs_threshold,
                    balance_rel_threshold: self.balance_rel_threshold,
                    eviction_interval_secs: self.eviction_interval_secs,
                    max_tree_size: self.max_tree_size,
                    block_size: self.block_size,
                    balance_token_usage_threshold: self.balance_token_usage_threshold,
                    overload_token_usage_threshold: self.overload_token_usage_threshold,
                    overlap_decay: self.overlap_decay,
                    selection_temperature: self.selection_temperature,
                    cache_index: self.parse_cache_index()?,
                    cache_ttl_secs: self.cache_ttl_secs,
                    cache_boundaries: self.cache_boundaries.clone(),
                    selection_policy: (self.selection_policy != policies::cost::DEFAULT_POLICY)
                        .then(|| self.selection_policy.clone()),
                    selection_accounting_ttl_ms: self.selection_accounting_ttl_ms,
                },
                PolicyType::PowerOfTwo => ConfigPolicyConfig::PowerOfTwo {
                    load_check_interval_secs: self.load_monitor_interval,
                    load_metric: power_of_two_load_metric,
                },
                PolicyType::LeastLoad => ConfigPolicyConfig::LeastLoad {
                    load_check_interval_secs: self.load_monitor_interval,
                    kv_pressure_weight: self.least_load_kv_pressure_weight,
                    mean_prefill_tokens: self.least_load_mean_prefill_tokens,
                    default_throughput: self.least_load_default_throughput,
                    max_waiting_requests: self.least_load_max_waiting_requests,
                },
                PolicyType::Bucket => ConfigPolicyConfig::Bucket {
                    balance_abs_threshold: self.balance_abs_threshold,
                    balance_rel_threshold: self.balance_rel_threshold,
                    bucket_adjust_interval_secs: self.bucket_adjust_interval_secs,
                },
                PolicyType::Manual => ConfigPolicyConfig::Manual {
                    eviction_interval_secs: self.eviction_interval_secs,
                    max_idle_secs: self.max_idle_secs,
                    assignment_mode: self
                        .parse_assignment_mode(config::ManualAssignmentMode::Random)?,
                },
                PolicyType::ConsistentHashing => ConfigPolicyConfig::ConsistentHashing,
                PolicyType::PrefixHash => ConfigPolicyConfig::PrefixHash {
                    prefix_token_count: self.prefix_token_count,
                    load_factor: self.prefix_hash_load_factor,
                    balance_abs_threshold: self.prefix_hash_balance_abs_threshold,
                    cache_boundaries: self.cache_boundaries.clone(),
                },
            })
        };

        // IGW does not override backend or disaggregated modes; IGW-only keeps
        // the Python binding's existing empty startup-worker behavior.
        let mode = if matches!(self.backend, BackendType::Openai) {
            RoutingMode::OpenAI {
                worker_urls: self.worker_urls.clone(),
            }
        } else if matches!(self.backend, BackendType::Anthropic) {
            RoutingMode::Anthropic {
                worker_urls: self.worker_urls.clone(),
            }
        } else if self.epd_disaggregation {
            RoutingMode::EncodePrefillDecode {
                encode_urls: self.encode_urls.clone().unwrap_or_default(),
                prefill_urls: self.prefill_urls.clone().unwrap_or_default(),
                decode_urls: self.decode_urls.clone().unwrap_or_default(),
                encode_policy: self
                    .encode_policy
                    .as_ref()
                    .map(convert_policy)
                    .transpose()?,
                prefill_policy: self
                    .prefill_policy
                    .as_ref()
                    .map(convert_policy)
                    .transpose()?,
                decode_policy: self
                    .decode_policy
                    .as_ref()
                    .map(convert_policy)
                    .transpose()?,
            }
        } else if self.pd_disaggregation {
            RoutingMode::PrefillDecode {
                prefill_urls: self.prefill_urls.clone().unwrap_or_default(),
                decode_urls: self.decode_urls.clone().unwrap_or_default(),
                prefill_policy: self
                    .prefill_policy
                    .as_ref()
                    .map(convert_policy)
                    .transpose()?,
                decode_policy: self
                    .decode_policy
                    .as_ref()
                    .map(convert_policy)
                    .transpose()?,
            }
        } else {
            RoutingMode::Regular {
                worker_urls: if self.enable_igw {
                    vec![]
                } else {
                    self.worker_urls.clone()
                },
            }
        };

        let policy = convert_policy(&self.policy)?;

        let discovery = if self.service_discovery {
            Some(DiscoveryConfig::Kubernetes(KubernetesDiscoveryConfig {
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: self.selector.clone(),
                encode_selector: self.encode_selector.clone(),
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
                bootstrap_port_annotation: self.bootstrap_port_annotation.clone(),
                worker_ports_annotation: self.worker_ports_annotation.clone(),
                kv_connector_annotation: self.kv_connector_annotation.clone(),
                kv_engine_id_annotation: self.kv_engine_id_annotation.clone(),
                router_selector: self.router_selector.clone(),
                router_mesh_port_annotation: "sglang.ai/mesh-port".to_string(),
                model_id_source: self.model_id_from.clone(),
            }))
        } else {
            self.discovery.clone()
        };
        let has_discovery = discovery.is_some();

        // The metrics listener `start` creates: its port when unset is the
        // exporter's default, so the validator's port checks (the admin,
        // probe and data ports against it) see the listener that will bind.
        let metrics = Some(MetricsConfig {
            port: self.prometheus_port.unwrap_or(29000),
            host: self.metrics_host(),
        });

        let trace_config = Some(config::TraceConfig {
            enable_trace: self.enable_trace,
            otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
        });

        let parser = config::ParserChoice::parse(&self.parser).ok_or_else(|| {
            config::ConfigError::InvalidValue {
                field: "parser".to_string(),
                value: self.parser.clone(),
                reason: "expected legacy or symphony".to_string(),
            }
        })?;
        let history_backend = match self.history_backend {
            HistoryBackendType::Memory => config::HistoryBackend::Memory,
            HistoryBackendType::None => config::HistoryBackend::None,
            HistoryBackendType::Oracle => config::HistoryBackend::Oracle,
            HistoryBackendType::Postgres => config::HistoryBackend::Postgres,
            HistoryBackendType::Redis => config::HistoryBackend::Redis,
        };

        // Load schema config from YAML file if provided
        let schema = if let Some(ref path) = self.schema_config {
            let content = std::fs::read_to_string(path).map_err(|e| {
                config::ConfigError::ValidationFailed {
                    reason: format!("Failed to read schema config file '{path}': {e}"),
                }
            })?;
            let schema: config::SchemaConfig = serde_yaml::from_str(&content).map_err(|e| {
                config::ConfigError::ValidationFailed {
                    reason: format!("Failed to parse schema config file '{path}': {e}"),
                }
            })?;
            Some(schema)
        } else {
            None
        };

        let oracle = if matches!(self.history_backend, HistoryBackendType::Oracle) {
            self.oracle_config.as_ref().map(|cfg| {
                let mut c = cfg.to_config_oracle();
                c.schema.clone_from(&schema);
                c
            })
        } else {
            None
        };

        let postgres_config = if matches!(self.history_backend, HistoryBackendType::Postgres) {
            self.postgres_config.as_ref().map(|cfg| {
                let mut c = cfg.to_config_postgres();
                c.schema.clone_from(&schema);
                c
            })
        } else {
            None
        };

        let redis_config = if matches!(self.history_backend, HistoryBackendType::Redis) {
            self.redis_config.as_ref().map(|cfg| {
                let mut c = cfg.to_config_redis();
                c.schema = schema;
                c
            })
        } else {
            None
        };

        // `backend` normally only steers the routing mode. Over ZMQ it
        // additionally pins the startup workers' runtime: the shared EngineCore
        // handshake carries no engine identity, so the wire protocol cannot be
        // probed. HTTP/gRPC keep auto-detection (None). Mirrors
        // `to_router_config` in model_gateway/src/main.rs.
        let startup_worker_runtime_type =
            if matches!(self.connection_mode, worker::ConnectionMode::Zmq) {
                match self.backend {
                    BackendType::Vllm => Some(worker::RuntimeType::Vllm),
                    BackendType::Tokenspeed => Some(worker::RuntimeType::TokenSpeed),
                    BackendType::Sglang => Some(worker::RuntimeType::Sglang),
                    _ => None,
                }
            } else {
                None
            };

        config::RouterConfig::builder()
            .mode(mode)
            .policy(policy)
            .cache_boundaries(self.cache_boundaries.clone())
            .host(&self.host)
            .port(self.port)
            .health_check_port(self.health_check_port)
            .admin_port(self.admin_port)
            .admin_host(self.admin_host.clone())
            .connection_mode(self.connection_mode)
            .startup_worker_runtime_type(startup_worker_runtime_type)
            .zmq_engine_count(self.zmq_engine_count)
            .max_payload_size(self.max_payload_size)
            .request_timeout_secs(self.request_timeout_secs)
            .worker_startup_timeout_secs(self.worker_startup_timeout_secs)
            .worker_startup_delay_secs(self.worker_startup_delay)
            .worker_startup_check_interval_secs(self.worker_startup_check_interval)
            .job_queue_capacity(self.job_queue_capacity)
            .job_queue_concurrency(self.job_queue_concurrency)
            .worker_overload_waiting_requests(self.worker_overload_waiting_requests)
            .worker_overload_token_usage(self.worker_overload_token_usage)
            .worker_overload_protection(self.worker_overload_protection)
            .worker_overload_shed(self.worker_overload_shed)
            .worker_stall_secs(self.worker_stall_secs)
            .worker_wedge_secs(self.worker_wedge_secs)
            .worker_stale_secs(self.worker_stale_secs)
            .worker_warmup(
                self.worker_warmup_secs,
                self.worker_warmup_share,
                self.worker_warmup_blocks,
                self.worker_warmup_thin_ratio,
                self.worker_warmup_divert_every,
            )
            // Deprecated: unset by default; `chain` is the only value, `positional` fails validation.
            .kv_index(self.kv_index.clone())
            .disable_load_monitoring(self.disable_load_monitoring)
            .load_monitor_interval_secs(self.load_monitor_interval)
            .pd_admission_wait_secs(self.pd_admission_wait_secs)
            .max_concurrent_requests(self.max_concurrent_requests)
            .queue_size(self.queue_size)
            .queue_timeout_secs(self.queue_timeout_secs)
            .priority_scheduler_enabled(self.priority_scheduler_enabled)
            .priority_scheduler_default_max_class(self.priority_scheduler_default_max_class.clone())
            .priority_scheduler_config(self.priority_scheduler_config.clone())
            .priority_scheduler_tenant_metric_top_n(self.priority_scheduler_tenant_metric_top_n)
            .tenant_rate_limit_enabled(self.tenant_rate_limit_enabled)
            .tenant_rate_limit_config(self.tenant_rate_limit_config.clone())
            .prefill_max_inflight_requests_per_worker(self.prefill_max_inflight_requests_per_worker)
            .prefill_queue_size(self.prefill_queue_size)
            .prefill_queue_timeout_secs(self.prefill_queue_timeout_secs)
            .cors_allowed_origins(self.cors_allowed_origins.clone())
            .retry_config(config::RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            })
            .circuit_breaker_config(config::CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            })
            .health_check_config(config::HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
                disable_health_check: self.disable_health_check,
                // Explicit setting wins; otherwise recovery-by-removal follows
                // service discovery, which is what re-adds a removed worker.
                remove_unhealthy_workers: config::resolve_worker_auto_recovery(
                    self.remove_unhealthy_workers,
                    has_discovery,
                ),
                drain_settle_secs: self.drain_settle_secs,
            })
            .tokenizer_cache(config::TokenizerCacheConfig {
                enable_l0: self.tokenizer_cache_enable_l0,
                l0_max_entries: self.tokenizer_cache_l0_max_entries,
                l0_max_memory: self.tokenizer_cache_l0_max_memory,
                enable_l1: self.tokenizer_cache_enable_l1,
                l1_max_memory: self.tokenizer_cache_l1_max_memory,
            })
            .disable_tokenizer_autoload(self.disable_tokenizer_autoload)
            .parser(parser)
            .history_backend(history_backend)
            .memory_config(config::MemoryConfig {
                max_responses: self.history_memory_max_responses,
                max_items: self.history_memory_max_items,
                retention_secs: (self.history_memory_retention_secs > 0)
                    .then_some(self.history_memory_retention_secs),
            })
            .maybe_api_key(self.api_key.as_ref())
            .tenant_api_keys(
                self.tenant_api_keys
                    .iter()
                    .map(|(tenant_id, key)| config::TenantApiKeyEntry {
                        tenant_id: tenant_id.clone(),
                        key: key.clone(),
                    })
                    .collect(),
            )
            .maybe_discovery(discovery)
            .maybe_metrics(metrics)
            .maybe_trace(trace_config)
            .maybe_log_dir(self.log_dir.as_ref())
            .maybe_jemalloc_prof_dir(self.jemalloc_prof_dir.as_ref())
            .maybe_log_level(self.log_level.as_ref())
            .maybe_request_id_headers(self.request_id_headers.clone())
            .trust_tenant_header(self.trust_tenant_header)
            .tenant_header_name(&self.tenant_header_name)
            .maybe_storage_context_headers(
                (!self.storage_context_headers.is_empty())
                    .then(|| self.storage_context_headers.clone()),
            )
            .maybe_rate_limit_tokens_per_second(self.rate_limit_tokens_per_second)
            .max_background_responses(self.max_background_responses)
            .sse_keepalive_secs(self.sse_keepalive_secs)
            .maybe_model_path(self.model_path.as_ref())
            .maybe_tokenizer_path(self.tokenizer_path.as_ref())
            .maybe_chat_template(self.chat_template.as_ref())
            .model_aliases(self.model_aliases.clone())
            .maybe_oracle(oracle)
            .maybe_postgres(postgres_config)
            .maybe_redis(redis_config)
            .maybe_reasoning_parser(self.reasoning_parser.as_ref())
            .maybe_tool_call_parser(self.tool_call_parser.as_ref())
            .maybe_mcp_config_path(self.mcp_config_path.as_ref())
            .maybe_storage_hook_wasm_path(self.storage_hook_wasm_path.as_deref())
            .enable_wasm(self.enable_wasm)
            .dp_aware(self.dp_aware)
            .upstream_http2(self.upstream_http2)
            .upstream_pool_idle_timeout_secs(self.upstream_pool_idle_timeout_secs)
            .max_buffered_request_bytes(self.max_buffered_request_bytes)
            .stream_body_stall_timeout_secs(self.stream_body_stall_timeout_secs)
            .header_read_timeout_secs(self.header_read_timeout_secs)
            .idle_connection_timeout_secs(self.idle_connection_timeout_secs)
            .multimodal_tensor_transport(multimodal_tensor_transport)
            .multimodal_shm_min_bytes(self.multimodal_shm_min_bytes)
            .multimodal_max_inflight_bytes(self.multimodal_max_inflight_bytes)
            .mm_per_request_image_limit(self.mm_per_request_image_limit)
            .mm_processing(mm_processing)
            .mm_pixel_cache_mb(self.mm_pixel_cache_mb)
            .mm_pixel_rdma(self.mm_pixel_rdma)
            .mm_allowed_media_domains(self.mm_allowed_media_domains.clone())
            .mm_allow_private_media_hosts(self.mm_allow_private_media_hosts)
            .rdma_listen_ip(self.rdma_listen_ip.clone())
            .rdma_slot_ttl_s(self.rdma_slot_ttl_s)
            .log_mm_timing(self.log_mm_timing)
            .access_log_fields(access_log_fields)
            .routing_key_override(config::RoutingKeyOverrideConfig {
                enabled: self.routing_key_override,
                eviction_interval_secs: self.eviction_interval_secs,
                max_idle_secs: self.max_idle_secs,
                assignment_mode: self
                    .parse_assignment_mode(config::ManualAssignmentMode::Delegate)?,
                headers: self.routing_key_headers.clone(),
            })
            .retries(!self.disable_retries)
            .circuit_breaker(!self.disable_circuit_breaker)
            .igw(self.enable_igw)
            .rl(smg_rl::RlConfig {
                enabled: self.enable_rl,
                control_timeout_secs: self.rl_control_timeout_secs,
                fanout_concurrency: self.rl_fanout_concurrency,
            })
            .maybe_client_cert_and_key(
                self.client_cert_path.as_ref(),
                self.client_key_path.as_ref(),
            )
            .add_ca_certificates(self.ca_cert_paths.clone())
            .maybe_server_cert_and_key(
                self.server_cert_path.as_ref(),
                self.server_key_path.as_ref(),
            )
            .dp_minimum_tokens_scheduler(self.dp_minimum_tokens_scheduler)
            .build()
    }
}

#[pymethods]
impl Router {
    #[new]
    #[pyo3(signature = (
        worker_urls,
        policy = PolicyType::RoundRobin,
        host = String::from("0.0.0.0"),
        port = 3001,
        // Every other parameter is keyword-only, in alphabetical order (the struct's
        // note says where a new one goes).
        *,
        access_log_fields = vec![],
        admin_host = None,
        admin_port = None,
        api_key = None,
        assignment_mode = None,
        backend = BackendType::Sglang,
        balance_abs_threshold = 64,
        balance_rel_threshold = 1.5,
        balance_token_usage_threshold = 1.0,
        block_size = 16,
        bootstrap_port_annotation = String::from("sglang.ai/bootstrap-port"),
        bucket_adjust_interval_secs = 5,
        ca_cert_paths = vec![],
        cache_boundaries = vec![],
        cache_index = String::from("tree"),
        cache_threshold = 0.3,
        cache_ttl_secs = 180,
        cb_failure_threshold = 10,
        cb_success_threshold = 3,
        cb_timeout_duration_secs = 60,
        cb_window_duration_secs = 120,
        chat_template = None,
        client_cert_path = None,
        client_key_path = None,
        control_plane_auth = None,
        cors_allowed_origins = vec![],
        decode_policy = None,
        decode_selector = HashMap::new(),
        decode_urls = None,
        disable_circuit_breaker = false,
        disable_health_check = false,
        disable_load_monitoring = false,
        disable_retries = false,
        disable_tokenizer_autoload = false,
        discovery = None,
        dp_aware = false,
        dp_minimum_tokens_scheduler = false,
        drain_settle_secs = 5,
        enable_igw = false,
        enable_mesh = false,
        enable_rl = false,
        enable_trace = false,
        enable_wasm = false,
        encode_policy = None,
        encode_selector = HashMap::new(),
        encode_urls = None,
        epd_disaggregation = false,
        eviction_interval_secs = 120,
        header_read_timeout_secs = 30,
        health_check_endpoint = String::from("/health"),
        health_check_interval_secs = 60,
        health_check_port = None,
        health_check_timeout_secs = 5,
        health_failure_threshold = 3,
        health_success_threshold = 2,
        history_backend = HistoryBackendType::Memory,
        history_memory_max_items = 100000,
        history_memory_max_responses = 10000,
        history_memory_retention_secs = 86400,
        idle_connection_timeout_secs = 90,
        jemalloc_prof_dir = None,
        job_queue_capacity = 1000,
        job_queue_concurrency = 200,
        kv_connector_annotation = String::from("smg.ai/kv-connector"),
        kv_engine_id_annotation = String::from("smg.ai/kv-engine-id"),
        kv_index = None,
        least_load_default_throughput = 2000.0,
        least_load_kv_pressure_weight = 0.15,
        least_load_max_waiting_requests = 0,
        least_load_mean_prefill_tokens = 1024,
        load_monitor_interval = 10,
        log_dir = None,
        log_json = false,
        log_level = None,
        log_mm_timing = false,
        max_background_responses = 1024,
        max_buffered_request_bytes = 1_048_576,
        max_concurrent_requests = -1,
        max_idle_secs = 14400,
        max_payload_size = 512 * 1024 * 1024,
        max_tree_size = 2usize.pow(26),
        mcp_config_path = None,
        mesh_advertise_host = None,
        mesh_host = String::from("0.0.0.0"),
        mesh_peer_urls = vec![],
        mesh_port = 39527u16,
        mesh_server_name = None,
        mesh_tls_ca_cert = None,
        mesh_tls_cert = None,
        mesh_tls_key = None,
        mm_allow_private_media_hosts = false,
        mm_allowed_media_domains = None,
        mm_per_request_image_limit = None,
        mm_pixel_cache_mb = None,
        mm_pixel_rdma = false,
        mm_processing = None,
        model_aliases = HashMap::new(),
        model_id_from = None,
        model_path = None,
        multimodal_max_inflight_bytes = None,
        multimodal_shm_min_bytes = None,
        multimodal_tensor_transport = None,
        oracle_config = None,
        otlp_traces_endpoint = String::from("localhost:4317"),
        overlap_decay = 0.0,
        overload_token_usage_threshold = 1.0,
        parser = String::from("legacy"),
        pd_admission_wait_secs = 30,
        pd_disaggregation = false,
        postgres_config = None,
        power_of_two_load_metric = String::from("expected_wait"),
        prefill_max_inflight_requests_per_worker = -1,
        prefill_policy = None,
        prefill_queue_size = None,
        prefill_queue_timeout_secs = None,
        prefill_selector = HashMap::new(),
        prefill_urls = None,
        prefix_hash_balance_abs_threshold = 10,
        prefix_hash_load_factor = 1.25,
        prefix_token_count = 256,
        priority_scheduler_config = None,
        priority_scheduler_default_max_class = String::from("default"),
        priority_scheduler_enabled = false,
        priority_scheduler_tenant_metric_top_n = 32,
        prometheus_duration_buckets = None,
        prometheus_host = None,
        prometheus_port = None,
        queue_size = 100,
        queue_timeout_secs = 60,
        rate_limit_tokens_per_second = None,
        rdma_listen_ip = None,
        rdma_slot_ttl_s = None,
        reasoning_parser = None,
        redis_config = None,
        remove_unhealthy_workers = None,
        request_id_headers = None,
        request_timeout_secs = 1800,
        retry_backoff_multiplier = 1.5,
        retry_initial_backoff_ms = 50,
        retry_jitter_factor = 0.2,
        retry_max_backoff_ms = 30_000,
        retry_max_retries = 5,
        rl_control_timeout_secs = 600,
        rl_fanout_concurrency = 32,
        router_selector = HashMap::new(),
        routing_key_headers = vec![String::from("x-smg-routing-key")],
        routing_key_override = false,
        schema_config = None,
        selection_accounting_ttl_ms = 0,
        selection_policy = String::from("cache-aware-default"),
        selection_temperature = 0.0,
        selector = HashMap::new(),
        server_cert_path = None,
        server_key_path = None,
        service_discovery = false,
        service_discovery_namespace = None,
        service_discovery_port = 80,
        shutdown_grace_period_secs = 180,
        sse_keepalive_secs = 15,
        storage_context_headers = HashMap::new(),
        storage_hook_wasm_path = None,
        stream_body_stall_timeout_secs = 300,
        tenant_api_keys = vec![],
        tenant_header_name = String::from("x-smg-tenant-id"),
        tenant_rate_limit_config = None,
        tenant_rate_limit_enabled = false,
        tokenizer_cache_enable_l0 = false,
        tokenizer_cache_enable_l1 = false,
        tokenizer_cache_l0_max_entries = 10000,
        tokenizer_cache_l0_max_memory = 268435456,
        tokenizer_cache_l1_max_memory = 52428800,
        tokenizer_path = None,
        tool_call_parser = None,
        trust_tenant_header = false,
        upstream_http2 = false,
        upstream_pool_idle_timeout_secs = 3,
        worker_overload_protection = true,
        worker_overload_shed = false,
        worker_overload_token_usage = Some(0.8),
        worker_overload_waiting_requests = Some(8),
        worker_ports_annotation = String::from("smg.ai/worker-ports"),
        worker_stale_secs = 15,
        worker_stall_secs = 2,
        worker_startup_check_interval = 30,
        worker_startup_delay = 0,
        worker_startup_timeout_secs = 600,
        worker_warmup_blocks = 1024,
        worker_warmup_divert_every = 8,
        worker_warmup_secs = 60,
        worker_warmup_share = 0.25,
        worker_warmup_thin_ratio = 0.5,
        worker_wedge_secs = 3,
        zmq_engine_count = None,
    ))]
    #[expect(clippy::too_many_arguments)]
    fn new(
        worker_urls: Vec<String>,
        policy: PolicyType,
        host: String,
        port: u16,
        // Keyword-only, in the signature's order (pyo3 wants the two orders to agree).
        access_log_fields: Vec<String>,
        admin_host: Option<String>,
        admin_port: Option<u16>,
        api_key: Option<String>,
        assignment_mode: Option<String>,
        backend: BackendType,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        balance_token_usage_threshold: f32,
        block_size: usize,
        bootstrap_port_annotation: String,
        bucket_adjust_interval_secs: usize,
        ca_cert_paths: Vec<String>,
        cache_boundaries: Vec<usize>,
        cache_index: String,
        cache_threshold: f32,
        cache_ttl_secs: u64,
        cb_failure_threshold: u32,
        cb_success_threshold: u32,
        cb_timeout_duration_secs: u64,
        cb_window_duration_secs: u64,
        chat_template: Option<String>,
        client_cert_path: Option<String>,
        client_key_path: Option<String>,
        control_plane_auth: Option<PyControlPlaneAuthConfig>,
        cors_allowed_origins: Vec<String>,
        decode_policy: Option<PolicyType>,
        decode_selector: HashMap<String, String>,
        decode_urls: Option<Vec<String>>,
        disable_circuit_breaker: bool,
        disable_health_check: bool,
        disable_load_monitoring: bool,
        disable_retries: bool,
        disable_tokenizer_autoload: bool,
        discovery: Option<Bound<'_, PyAny>>,
        dp_aware: bool,
        dp_minimum_tokens_scheduler: bool,
        drain_settle_secs: u64,
        enable_igw: bool,
        enable_mesh: bool,
        enable_rl: bool,
        enable_trace: bool,
        enable_wasm: bool,
        encode_policy: Option<PolicyType>,
        encode_selector: HashMap<String, String>,
        encode_urls: Option<Vec<(String, Option<u16>)>>,
        epd_disaggregation: bool,
        eviction_interval_secs: u64,
        header_read_timeout_secs: u64,
        health_check_endpoint: String,
        health_check_interval_secs: u64,
        health_check_port: Option<u16>,
        health_check_timeout_secs: u64,
        health_failure_threshold: u32,
        health_success_threshold: u32,
        history_backend: HistoryBackendType,
        history_memory_max_items: usize,
        history_memory_max_responses: usize,
        history_memory_retention_secs: u64,
        idle_connection_timeout_secs: u64,
        jemalloc_prof_dir: Option<String>,
        job_queue_capacity: usize,
        job_queue_concurrency: usize,
        kv_connector_annotation: String,
        kv_engine_id_annotation: String,
        kv_index: Option<String>,
        least_load_default_throughput: f64,
        least_load_kv_pressure_weight: f64,
        least_load_max_waiting_requests: u32,
        least_load_mean_prefill_tokens: u32,
        load_monitor_interval: u64,
        log_dir: Option<String>,
        log_json: bool,
        log_level: Option<String>,
        log_mm_timing: bool,
        max_background_responses: usize,
        max_buffered_request_bytes: u64,
        max_concurrent_requests: i32,
        max_idle_secs: u64,
        max_payload_size: usize,
        max_tree_size: usize,
        mcp_config_path: Option<String>,
        mesh_advertise_host: Option<String>,
        mesh_host: String,
        mesh_peer_urls: Vec<String>,
        mesh_port: u16,
        mesh_server_name: Option<String>,
        mesh_tls_ca_cert: Option<String>,
        mesh_tls_cert: Option<String>,
        mesh_tls_key: Option<String>,
        mm_allow_private_media_hosts: bool,
        mm_allowed_media_domains: Option<Vec<String>>,
        mm_per_request_image_limit: Option<usize>,
        mm_pixel_cache_mb: Option<usize>,
        mm_pixel_rdma: bool,
        mm_processing: Option<String>,
        model_aliases: HashMap<String, String>,
        model_id_from: Option<String>,
        model_path: Option<String>,
        multimodal_max_inflight_bytes: Option<usize>,
        multimodal_shm_min_bytes: Option<usize>,
        multimodal_tensor_transport: Option<String>,
        oracle_config: Option<PyOracleConfig>,
        otlp_traces_endpoint: String,
        overlap_decay: f32,
        overload_token_usage_threshold: f32,
        parser: String,
        pd_admission_wait_secs: u64,
        pd_disaggregation: bool,
        postgres_config: Option<PyPostgresConfig>,
        power_of_two_load_metric: String,
        prefill_max_inflight_requests_per_worker: i32,
        prefill_policy: Option<PolicyType>,
        prefill_queue_size: Option<usize>,
        prefill_queue_timeout_secs: Option<u64>,
        prefill_selector: HashMap<String, String>,
        prefill_urls: Option<Vec<(String, Option<u16>)>>,
        prefix_hash_balance_abs_threshold: usize,
        prefix_hash_load_factor: f64,
        prefix_token_count: usize,
        priority_scheduler_config: Option<String>,
        priority_scheduler_default_max_class: String,
        priority_scheduler_enabled: bool,
        priority_scheduler_tenant_metric_top_n: u32,
        prometheus_duration_buckets: Option<Vec<f64>>,
        prometheus_host: Option<String>,
        prometheus_port: Option<u16>,
        queue_size: usize,
        queue_timeout_secs: u64,
        rate_limit_tokens_per_second: Option<i32>,
        rdma_listen_ip: Option<String>,
        rdma_slot_ttl_s: Option<u64>,
        reasoning_parser: Option<String>,
        redis_config: Option<PyRedisConfig>,
        remove_unhealthy_workers: Option<bool>,
        request_id_headers: Option<Vec<String>>,
        request_timeout_secs: u64,
        retry_backoff_multiplier: f32,
        retry_initial_backoff_ms: u64,
        retry_jitter_factor: f32,
        retry_max_backoff_ms: u64,
        retry_max_retries: u32,
        rl_control_timeout_secs: u64,
        rl_fanout_concurrency: usize,
        router_selector: HashMap<String, String>,
        routing_key_headers: Vec<String>,
        routing_key_override: bool,
        schema_config: Option<String>,
        selection_accounting_ttl_ms: u64,
        selection_policy: String,
        selection_temperature: f32,
        selector: HashMap<String, String>,
        server_cert_path: Option<String>,
        server_key_path: Option<String>,
        service_discovery: bool,
        service_discovery_namespace: Option<String>,
        service_discovery_port: u16,
        shutdown_grace_period_secs: u64,
        sse_keepalive_secs: u64,
        storage_context_headers: HashMap<String, String>,
        storage_hook_wasm_path: Option<String>,
        stream_body_stall_timeout_secs: u64,
        tenant_api_keys: Vec<(String, String)>,
        tenant_header_name: String,
        tenant_rate_limit_config: Option<String>,
        tenant_rate_limit_enabled: bool,
        tokenizer_cache_enable_l0: bool,
        tokenizer_cache_enable_l1: bool,
        tokenizer_cache_l0_max_entries: usize,
        tokenizer_cache_l0_max_memory: usize,
        tokenizer_cache_l1_max_memory: usize,
        tokenizer_path: Option<String>,
        tool_call_parser: Option<String>,
        trust_tenant_header: bool,
        upstream_http2: bool,
        upstream_pool_idle_timeout_secs: u64,
        worker_overload_protection: bool,
        worker_overload_shed: bool,
        worker_overload_token_usage: Option<f64>,
        worker_overload_waiting_requests: Option<usize>,
        worker_ports_annotation: String,
        worker_stale_secs: u64,
        worker_stall_secs: u64,
        worker_startup_check_interval: u64,
        worker_startup_delay: u64,
        worker_startup_timeout_secs: u64,
        worker_warmup_blocks: usize,
        worker_warmup_divert_every: u64,
        worker_warmup_secs: u64,
        worker_warmup_share: f32,
        worker_warmup_thin_ratio: f32,
        worker_wedge_secs: u64,
        zmq_engine_count: Option<usize>,
    ) -> PyResult<Self> {
        // Two spellings of one choice: refuse both rather than pick one.
        if service_discovery && discovery.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Pass service_discovery=True or a discovery mapping, not both",
            ));
        }
        let discovery = discovery
            .as_ref()
            .map(parse_discovery)
            .transpose()?
            .flatten();

        let mut all_urls = worker_urls.clone();

        if let Some(ref encode_urls) = encode_urls {
            for (url, _) in encode_urls {
                all_urls.push(url.clone());
            }
        }

        if let Some(ref prefill_urls) = prefill_urls {
            for (url, _) in prefill_urls {
                all_urls.push(url.clone());
            }
        }

        if let Some(ref decode_urls) = decode_urls {
            all_urls.extend(decode_urls.clone());
        }

        let connection_mode = Self::determine_connection_mode(&all_urls);

        Ok(Router {
            worker_urls,
            policy,
            host,
            port,
            access_log_fields,
            admin_host,
            admin_port,
            api_key,
            assignment_mode,
            backend,
            balance_abs_threshold,
            balance_rel_threshold,
            balance_token_usage_threshold,
            block_size,
            bootstrap_port_annotation,
            bucket_adjust_interval_secs,
            ca_cert_paths,
            cache_boundaries,
            cache_index,
            cache_threshold,
            cache_ttl_secs,
            cb_failure_threshold,
            cb_success_threshold,
            cb_timeout_duration_secs,
            cb_window_duration_secs,
            chat_template,
            client_cert_path,
            client_key_path,
            control_plane_auth,
            cors_allowed_origins,
            decode_policy,
            decode_selector,
            decode_urls,
            disable_circuit_breaker,
            disable_health_check,
            disable_load_monitoring,
            disable_retries,
            disable_tokenizer_autoload,
            discovery,
            dp_aware,
            dp_minimum_tokens_scheduler,
            drain_settle_secs,
            enable_igw,
            enable_mesh,
            enable_rl,
            enable_trace,
            enable_wasm,
            encode_policy,
            encode_selector,
            encode_urls,
            epd_disaggregation,
            eviction_interval_secs,
            header_read_timeout_secs,
            health_check_endpoint,
            health_check_interval_secs,
            health_check_port,
            health_check_timeout_secs,
            health_failure_threshold,
            health_success_threshold,
            history_backend,
            history_memory_max_items,
            history_memory_max_responses,
            history_memory_retention_secs,
            idle_connection_timeout_secs,
            jemalloc_prof_dir,
            job_queue_capacity,
            job_queue_concurrency,
            kv_connector_annotation,
            kv_engine_id_annotation,
            kv_index,
            least_load_default_throughput,
            least_load_kv_pressure_weight,
            least_load_max_waiting_requests,
            least_load_mean_prefill_tokens,
            load_monitor_interval,
            log_dir,
            log_json,
            log_level,
            log_mm_timing,
            max_background_responses,
            max_buffered_request_bytes,
            max_concurrent_requests,
            max_idle_secs,
            max_payload_size,
            max_tree_size,
            mcp_config_path,
            mesh_advertise_host,
            mesh_host,
            mesh_peer_urls,
            mesh_port,
            mesh_server_name,
            mesh_tls_ca_cert,
            mesh_tls_cert,
            mesh_tls_key,
            mm_allow_private_media_hosts,
            mm_allowed_media_domains,
            mm_per_request_image_limit,
            mm_pixel_cache_mb,
            mm_pixel_rdma,
            mm_processing,
            model_aliases,
            model_id_from,
            model_path,
            multimodal_max_inflight_bytes,
            multimodal_shm_min_bytes,
            multimodal_tensor_transport,
            oracle_config,
            otlp_traces_endpoint,
            overlap_decay,
            overload_token_usage_threshold,
            parser,
            pd_admission_wait_secs,
            pd_disaggregation,
            postgres_config,
            power_of_two_load_metric,
            prefill_max_inflight_requests_per_worker,
            prefill_policy,
            prefill_queue_size,
            prefill_queue_timeout_secs,
            prefill_selector,
            prefill_urls,
            prefix_hash_balance_abs_threshold,
            prefix_hash_load_factor,
            prefix_token_count,
            priority_scheduler_config,
            priority_scheduler_default_max_class,
            priority_scheduler_enabled,
            priority_scheduler_tenant_metric_top_n,
            prometheus_duration_buckets,
            prometheus_host,
            prometheus_port,
            queue_size,
            queue_timeout_secs,
            rate_limit_tokens_per_second,
            rdma_listen_ip,
            rdma_slot_ttl_s,
            reasoning_parser,
            redis_config,
            remove_unhealthy_workers,
            request_id_headers,
            request_timeout_secs,
            retry_backoff_multiplier,
            retry_initial_backoff_ms,
            retry_jitter_factor,
            retry_max_backoff_ms,
            retry_max_retries,
            rl_control_timeout_secs,
            rl_fanout_concurrency,
            router_selector,
            routing_key_headers,
            routing_key_override,
            schema_config,
            selection_accounting_ttl_ms,
            selection_policy,
            selection_temperature,
            selector,
            server_cert_path,
            server_key_path,
            service_discovery,
            service_discovery_namespace,
            service_discovery_port,
            shutdown_grace_period_secs,
            sse_keepalive_secs,
            storage_context_headers,
            storage_hook_wasm_path,
            stream_body_stall_timeout_secs,
            tenant_api_keys,
            tenant_header_name,
            tenant_rate_limit_config,
            tenant_rate_limit_enabled,
            tokenizer_cache_enable_l0,
            tokenizer_cache_enable_l1,
            tokenizer_cache_l0_max_entries,
            tokenizer_cache_l0_max_memory,
            tokenizer_cache_l1_max_memory,
            tokenizer_path,
            tool_call_parser,
            trust_tenant_header,
            upstream_http2,
            upstream_pool_idle_timeout_secs,
            worker_overload_protection,
            worker_overload_shed,
            worker_overload_token_usage,
            worker_overload_waiting_requests,
            worker_ports_annotation,
            worker_stale_secs,
            worker_stall_secs,
            worker_startup_check_interval,
            worker_startup_delay,
            worker_startup_timeout_secs,
            worker_warmup_blocks,
            worker_warmup_divert_every,
            worker_warmup_secs,
            worker_warmup_share,
            worker_warmup_thin_ratio,
            worker_wedge_secs,
            zmq_engine_count,
            connection_mode,
        })
    }

    fn start(&self, py: Python<'_>) -> PyResult<()> {
        use observability::metrics::PrometheusConfig;

        let router_config = self.to_router_config().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Configuration error: {e}"))
        })?;

        router_config.validate().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Configuration validation failed: {e}"))
        })?;

        let service_discovery_config = router_config
            .discovery
            .as_ref()
            .map(|discovery| {
                service_discovery::RuntimeDiscoveryConfig::from_config(
                    discovery,
                    &router_config.mode,
                )
            })
            .transpose()
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Configuration error: {e}"))
            })?;
        let mesh_discovery_config = router_config
            .discovery
            .as_ref()
            .and_then(mesh_discovery::MeshDiscoveryConfig::from_discovery);

        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port.unwrap_or(29000),
            host: self.metrics_host(),
            duration_buckets: self.prometheus_duration_buckets.clone(),
        });

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Release the GIL while the server runs so Python threads can make progress.
        py.detach(|| {
            runtime.block_on(async move {
            Box::pin(server::startup(server::ServerConfig {
                host: self.host.clone(),
                port: self.port,
                health_check_port: self.health_check_port,
                runtime_worker_threads: None,
                router_config,
                max_payload_size: self.max_payload_size,
                log_dir: self.log_dir.clone(),
                log_level: self.log_level.clone(),
                log_json: self.log_json,
                service_discovery_config,
                mesh_discovery_config,
                prometheus_config,
                request_timeout_secs: self.request_timeout_secs,
                request_id_headers: self.request_id_headers.clone(),
                shutdown_grace_period_secs: self.shutdown_grace_period_secs,
                control_plane_auth: self
                    .control_plane_auth
                    .as_ref()
                    .map(|c| c.to_auth_control_plane_config()),
                mesh_server_config: if self.enable_mesh {
                    let self_name = self.mesh_server_name.clone().unwrap_or_else(|| {
                        use rand::{distr::Alphanumeric, RngExt};
                        let random_string: String = (0..4)
                            .map(|_| rand::rng().sample(Alphanumeric) as char)
                            .collect();
                        format!("Mesh_{random_string}")
                    });
                    let peer = self
                        .mesh_peer_urls
                        .first()
                        .map(|url| {
                            url.parse::<std::net::SocketAddr>().map_err(|e| {
                                pyo3::exceptions::PyValueError::new_err(format!(
                                    "Invalid mesh peer URL '{url}': {e}"
                                ))
                            })
                        })
                        .transpose()?;
                    // Mirrors the CLI check in `main.rs`: port 0 parses as a
                    // socket address but is undialable once gossiped to peers,
                    // and reaches router discovery as the fallback mesh port
                    // for Pods with no usable annotation.
                    if self.mesh_port == 0 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Invalid value for mesh_port='0': mesh port cannot be 0; peers dial \
                             the advertised port, so it must be a fixed, routable one",
                        ));
                    }
                    let bind_addr =
                        Self::parse_mesh_socket_addr(&self.mesh_host, self.mesh_port, "mesh_host")?;
                    let (advertise_host, advertise_field) =
                        if let Some(host) = self.mesh_advertise_host.as_deref() {
                            (host, "mesh_advertise_host")
                        } else {
                            (self.mesh_host.as_str(), "mesh_host")
                        };
                    let advertise_addr = Self::parse_mesh_socket_addr(
                        advertise_host,
                        self.mesh_port,
                        advertise_field,
                    )?;
                    if advertise_addr.ip().is_unspecified() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid value for {advertise_field}='{advertise_host}': mesh advertise address cannot be unspecified; set mesh_advertise_host to a routable node IP"
                        )));
                    }
                    Some(smg_mesh::MeshServerConfig {
                        self_name,
                        bind_addr,
                        advertise_addr,
                        init_peer: peer,
                        mtls_config: self.mesh_mtls_config()?,
                    })
                } else {
                    None
                },
                webrtc_bind_addr: None,
                webrtc_stun_server: None,
                // The launcher resolved the secrets' sources in Python; the
                // record is the CLI's.
                secret_sources: Default::default(),
            }))
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            })
        })
    }
}

/// Get simple version string (default for --version)
#[pyfunction]
fn get_version_string() -> String {
    version::get_version_string()
}

/// Get verbose version information string with full build details (for --version-verbose)
#[pyfunction]
fn get_verbose_version_string() -> String {
    version::get_verbose_version_string()
}

/// Print the startup banner with braille art and key configuration info.
#[pyfunction]
fn print_banner(host: &str, port: u16, mode: &str) {
    version::print_banner(host, port, mode);
}

/// Get the list of available tool call parsers from the Rust factory.
#[pyfunction]
fn get_available_tool_call_parsers() -> Vec<String> {
    static PARSERS: OnceCell<Vec<String>> = OnceCell::new();
    PARSERS
        .get_or_init(|| parser_choices(|tools, _| tools.list_parsers()))
        .clone()
}

/// The choices of one parser flag: the names `legacy` picks from the two legacy registries (the
/// flag's own kind), Symphony's tables' (`symphony::formats::Table::name`), which `--parser
/// symphony` reads with when both flags name one, the names a table answers to that no legacy
/// parser of either kind has (`symphony::formats::Table::ALIASES`: a launcher's name for a model
/// family whose table has no older parser, which the gateway takes in either flag), and `hf`,
/// the table a model's response template compiles into, sorted, each once.
fn parser_choices(
    legacy: impl FnOnce(&tool_parser::ParserFactory, &reasoning_parser::ParserFactory) -> Vec<String>,
) -> Vec<String> {
    let tools = tool_parser::ParserFactory::new();
    let reasoning = reasoning_parser::ParserFactory::new();
    let mut names = legacy(&tools, &reasoning);
    names.extend(
        symphony::formats::Table::ALL
            .iter()
            .map(|table| table.name().to_string()),
    );
    names.extend(
        symphony::formats::Table::ALIASES
            .iter()
            .map(|(alias, _)| *alias)
            .filter(|alias| !tools.has_parser(alias) && !reasoning.registry().has_parser(alias))
            .map(str::to_string),
    );
    names.push(config::TEMPLATE_PARSER.to_string());
    names.sort();
    names.dedup();
    names
}

/// Get the list of available reasoning parsers from the Rust factory.
#[pyfunction]
fn get_available_reasoning_parsers() -> Vec<String> {
    static PARSERS: OnceCell<Vec<String>> = OnceCell::new();
    PARSERS
        .get_or_init(|| parser_choices(|_, reasoning| reasoning.list_parsers()))
        .clone()
}

#[pymodule]
fn smg_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    observability::metrics::register_jemalloc_as_global_allocator();
    // The process-level TLS crypto provider, installed once when the module
    // is imported: every HTTP client a binding builds takes it, whether or
    // not a router is started in this process.
    tls::install_crypto_provider();

    m.add_class::<PolicyType>()?;
    m.add_class::<BackendType>()?;
    m.add_class::<HistoryBackendType>()?;
    m.add_class::<PyRole>()?;
    m.add_class::<PyApiKeyEntry>()?;
    m.add_class::<PyJwtConfig>()?;
    m.add_class::<PyControlPlaneAuthConfig>()?;
    m.add_class::<PyOracleConfig>()?;
    m.add_class::<PyPostgresConfig>()?;
    m.add_class::<PyRedisConfig>()?;
    m.add_class::<Router>()?;
    m.add_class::<PyVllmGrpcServer>()?;
    m.add_class::<PyTokenSpeedGrpcServer>()?;
    m.add_class::<PySglangGrpcServer>()?;
    m.add_function(wrap_pyfunction!(get_version_string, m)?)?;
    m.add_function(wrap_pyfunction!(get_verbose_version_string, m)?)?;
    m.add_function(wrap_pyfunction!(print_banner, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_tool_call_parsers, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_reasoning_parsers, m)?)?;
    m.add_function(wrap_pyfunction!(init_servicer_tracing, m)?)?;
    Ok(())
}
