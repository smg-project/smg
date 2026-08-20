use std::{collections::HashMap, sync::Arc};

use arc_swap::{ArcSwap, ArcSwapOption};
use openai_protocol::{
    model_card::ModelCard,
    worker::{HealthCheckConfig, OverloadUpdate, WorkerModels, WorkerSpec, WorkerStatus},
};
use tokio::sync::mpsc;

use super::{
    circuit_breaker::{CircuitBreaker, CircuitBreakerConfig},
    event::WorkerConnected,
    overload::OverloadThresholds,
    resilience::ResolvedResilience,
    worker::{
        BasicWorker, ConnectionMode, LazyHttpClient, RuntimeType, WorkerMetadata, WorkerRuntime,
        WorkerType,
    },
};
use crate::{observability::metrics::Metrics, routers::grpc::backend_client::BackendClient};

/// Builder for creating BasicWorker instances with fluent API.
///
/// Internally stores a [`WorkerSpec`] for identity/config fields.
/// Callers with a pre-built `WorkerSpec` can use [`from_spec()`](Self::from_spec).
pub struct BasicWorkerBuilder {
    spec: WorkerSpec,
    /// Resolved health config (router defaults + per-worker overrides).
    /// If not set, falls back to `HealthCheckConfig::default()`.
    health_config: Option<HealthCheckConfig>,
    health_endpoint: String,
    circuit_breaker_config: CircuitBreakerConfig,
    backend_client: Option<BackendClient>,
    /// Pre-built worker-directed HTTP client (if not set, a default is created).
    http_client: Option<Arc<reqwest::Client>>,
    /// Resolved resilience config (if not set, defaults are used).
    resilience: Option<ResolvedResilience>,
    /// Initial lifecycle status. If unset, defaults to `Pending` for
    /// health-checked workers and `Ready` for `disable_health_check == true`.
    /// Callers replacing an existing worker (e.g. metadata updates) should
    /// pass the old worker's status to avoid kicking it back to Pending.
    initial_status: Option<WorkerStatus>,
    /// Connect-readiness signal sender (ZMQ registration path only).
    connect_signal_tx: Option<mpsc::UnboundedSender<WorkerConnected>>,
    /// Gateway-level overload thresholds the spec's `overload` block resolves
    /// against. Default empty: a spec block alone still enables protection
    /// for this worker.
    overload_defaults: OverloadThresholds,
}

impl BasicWorkerBuilder {
    /// Create a new builder with only the URL (uses default WorkerSpec)
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            spec: WorkerSpec::new(url),
            health_config: None,
            health_endpoint: "/health".to_string(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            backend_client: None,
            http_client: None,
            resilience: None,
            initial_status: None,
            connect_signal_tx: None,
            overload_defaults: OverloadThresholds::default(),
        }
    }

    /// Create a builder from an existing WorkerSpec.
    pub fn from_spec(spec: WorkerSpec) -> Self {
        Self {
            spec,
            health_config: None,
            health_endpoint: "/health".to_string(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            backend_client: None,
            http_client: None,
            resilience: None,
            initial_status: None,
            connect_signal_tx: None,
            overload_defaults: OverloadThresholds::default(),
        }
    }

    /// Create a new builder with URL and worker type (for backwards compatibility)
    pub fn new_with_type(url: impl Into<String>, worker_type: WorkerType) -> Self {
        let mut spec = WorkerSpec::new(url);
        spec.worker_type = worker_type;
        Self {
            spec,
            health_config: None,
            health_endpoint: "/health".to_string(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            backend_client: None,
            http_client: None,
            resilience: None,
            initial_status: None,
            connect_signal_tx: None,
            overload_defaults: OverloadThresholds::default(),
        }
    }

    /// Set the bootstrap port (for prefill workers in PD disaggregation)
    pub fn bootstrap_port(mut self, port: Option<u16>) -> Self {
        self.spec.bootstrap_port = port;
        self
    }

    /// Set the API key
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.spec.api_key = Some(api_key.into());
        self
    }

    /// Set the worker type (Regular, Prefill, or Decode)
    pub fn worker_type(mut self, worker_type: WorkerType) -> Self {
        self.spec.worker_type = worker_type;
        self
    }

    /// Set the connection mode (HTTP or gRPC)
    pub fn connection_mode(mut self, mode: ConnectionMode) -> Self {
        self.spec.connection_mode = mode;
        self
    }

    /// Set the runtime type (SGLang or vLLM)
    pub fn runtime_type(mut self, runtime_type: RuntimeType) -> Self {
        self.spec.runtime_type = runtime_type;
        self
    }

    /// Set the explicit ZMQ handshake bind address (replaces the address
    /// derived from the ipc:// worker URL). Only meaningful for ZMQ workers.
    pub fn zmq_handshake_address(mut self, address: impl Into<String>) -> Self {
        self.spec.zmq_handshake_address = Some(address.into());
        self
    }

    /// Set labels for worker identification
    pub fn labels(mut self, labels: HashMap<String, String>) -> Self {
        self.spec.labels = labels;
        self
    }

    /// Add a single label
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.spec.labels.insert(key.into(), value.into());
        self
    }

    /// Set the resolved health check configuration.
    ///
    /// This is the fully-resolved config (router defaults + per-worker overrides)
    /// stored on `WorkerMetadata` for runtime use.
    pub fn health_config(mut self, config: HealthCheckConfig) -> Self {
        self.health_config = Some(config);
        self
    }

    /// Override the initial lifecycle status.
    ///
    /// By default, `build()` chooses `Pending` for health-checked workers and
    /// `Ready` for workers with `disable_health_check == true`. Callers that
    /// replace an existing worker (e.g. metadata-only updates via
    /// `register_or_replace`) should pass the old worker's status here to
    /// avoid kicking a healthy worker back to Pending and causing avoidable
    /// 503s while it re-proves itself.
    pub fn status(mut self, status: WorkerStatus) -> Self {
        self.initial_status = Some(status);
        self
    }

    /// Set health check endpoint path (internal-only, from router config).
    pub fn health_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.health_endpoint = endpoint.into();
        self
    }

    /// Set circuit breaker configuration
    pub fn circuit_breaker_config(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker_config = config;
        self
    }

    /// Set the backend client (gRPC or ZMQ) for a local worker.
    pub fn backend_client(mut self, client: BackendClient) -> Self {
        self.backend_client = Some(client);
        self
    }

    /// Wire the connect-readiness signal sender (from the registry) so a ZMQ
    /// worker can wake the manager the instant its handshake completes. Only
    /// meaningful for ZMQ workers; HTTP/gRPC promotion stays poll-driven.
    pub fn connect_signal_tx(mut self, tx: mpsc::UnboundedSender<WorkerConnected>) -> Self {
        self.connect_signal_tx = Some(tx);
        self
    }

    /// Set a pre-built worker-directed HTTP client. The strong handle keeps
    /// the client's shared cache entry alive for the worker's lifetime.
    pub fn http_client(mut self, client: Arc<reqwest::Client>) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Set the resolved resilience config.
    pub fn resilience(mut self, resilience: ResolvedResilience) -> Self {
        self.resilience = Some(resilience);
        self
    }

    /// Set per-worker overload threshold overrides (carried on the spec so
    /// they survive replacement and show up in `GET /workers`).
    pub fn overload(mut self, overrides: OverloadUpdate) -> Self {
        self.spec.overload = overrides;
        self
    }

    /// Set the gateway-level overload thresholds the spec overrides resolve
    /// against (per signal: worker override, else this default).
    pub fn overload_defaults(mut self, defaults: OverloadThresholds) -> Self {
        self.overload_defaults = defaults;
        self
    }

    /// Set KV connector type (e.g., "MooncakeConnector", "NixlConnector")
    pub fn kv_connector(mut self, connector: impl Into<String>) -> Self {
        self.spec.kv_connector = Some(connector.into());
        self
    }

    /// Set KV role (e.g., "kv_producer", "kv_consumer", "kv_both")
    pub fn kv_role(mut self, role: impl Into<String>) -> Self {
        self.spec.kv_role = Some(role.into());
        self
    }

    /// Set KV transfer engine id (vLLM `kv_transfer_config.engine_id`)
    pub fn kv_engine_id(mut self, engine_id: impl Into<String>) -> Self {
        self.spec.kv_engine_id = Some(engine_id.into());
        self
    }

    /// Set worker priority (higher value = higher priority)
    pub fn priority(mut self, priority: u32) -> Self {
        self.spec.priority = priority;
        self
    }

    /// Set worker cost factor (baseline = 1.0)
    pub fn cost(mut self, cost: f32) -> Self {
        self.spec.cost = cost;
        self
    }

    /// Set models this worker can serve
    pub fn models(mut self, models: impl Into<WorkerModels>) -> Self {
        self.spec.models = models.into();
        self
    }

    /// Set a single model this worker can serve
    pub fn model(mut self, model: ModelCard) -> Self {
        self.spec.models = WorkerModels::Single(Box::new(model));
        self
    }

    /// Configure data-parallel routing.
    /// Captures the current URL as the base URL, then formats it as `{base}@{rank}`.
    pub fn dp_config(mut self, rank: usize, size: usize) -> Self {
        let base_url = self.spec.url.clone();
        self.spec.url = format!("{base_url}@{rank}");
        self.spec.dp_base_url = Some(base_url);
        self.spec.dp_rank = Some(rank);
        self.spec.dp_size = Some(size);
        self
    }

    /// Configure a grouped ZMQ DP worker: one worker and one socket set that
    /// `size` engines dial into. Sets `dp_size` without a rank — the URL is
    /// untouched and the worker is not rank-pinned; the connector balances
    /// across the group's engines internally.
    pub fn zmq_engine_group(mut self, size: usize) -> Self {
        self.spec.dp_size = Some(size);
        self
    }

    /// Build the BasicWorker instance
    pub fn build(mut self) -> BasicWorker {
        use std::sync::atomic::AtomicBool;

        use tokio::sync::OnceCell;

        // bootstrap_host is a PD/TCP-disaggregation concept (a host:port peer),
        // so derive it only for transports that carry a host. An ipc:// ZMQ
        // worker has no host; leave bootstrap_host empty rather than forcing the
        // URL through host:port parsing (which would warn and default to
        // localhost).
        if self.spec.connection_mode != ConnectionMode::Zmq {
            self.spec.bootstrap_host = parse_bootstrap_host(&self.spec.url);
        }

        // Resolve health config: use explicit config if set, otherwise
        // apply per-worker overrides from spec.health to defaults.
        let health_config = self
            .health_config
            .unwrap_or_else(|| self.spec.health.apply_to(&HealthCheckConfig::default()));

        let metadata = WorkerMetadata {
            overload: OverloadThresholds::resolve(&self.spec.overload, self.overload_defaults),
            spec: Arc::new(self.spec),
            health_config,
            health_endpoint: self.health_endpoint,
        };

        // OnceCell for lock-free client access after initialization; ArcSwap so
        // the ZMQ health probe can evict a dead client (see BasicWorker docs).
        let backend_client = {
            let cell = OnceCell::new();
            if let Some(client) = self.backend_client {
                // Pre-set the client if provided (set on a fresh cell cannot fail)
                cell.set(Arc::new(client)).ok();
            }
            Arc::new(ArcSwap::from_pointee(cell))
        };

        // Caller can override the initial status (e.g. when replacing an
        // existing worker, to preserve its prior status). Otherwise:
        // - workers with health checks disabled start Ready (routable)
        // - workers with health checks enabled start Pending (not routable
        //   until the health checker promotes them after success_threshold)
        let initial_status =
            self.initial_status
                .unwrap_or(if metadata.health_config.disable_health_check {
                    WorkerStatus::Ready
                } else {
                    WorkerStatus::Pending
                });
        Metrics::set_worker_health(&metadata.spec.url, initial_status == WorkerStatus::Ready);

        let http_client = Arc::new(match self.http_client {
            Some(client) => LazyHttpClient::ready(client),
            None => LazyHttpClient::deferred(),
        });

        let resilience = self.resilience.unwrap_or_default();

        BasicWorker {
            runtime: ArcSwap::from_pointee(WorkerRuntime::new(&metadata.spec.url, initial_status)),
            circuit_breaker: ArcSwap::from_pointee(CircuitBreaker::with_config_and_label(
                self.circuit_breaker_config,
                metadata.spec.url.clone(),
            )),
            metadata,
            backend_client,
            zmq_connect_started: Arc::new(AtomicBool::new(false)),
            zmq_connect_abort: Arc::new(ArcSwapOption::empty()),
            connect_signal_tx: self.connect_signal_tx,
            models_override: Arc::new(ArcSwap::from_pointee(WorkerModels::Wildcard)),
            http_client,
            resilience,
        }
    }
}

/// Parse bootstrap hostname from a URL, falling back to "localhost".
///
/// Handles DP-aware URLs like `http://host:8080@3` by stripping the `@rank`
/// suffix before parsing, since `@` is otherwise interpreted as a userinfo
/// delimiter per RFC 3986.
fn parse_bootstrap_host(url: &str) -> String {
    // Strip DP rank suffix (e.g., "http://host:8080@3" -> "http://host:8080")
    let clean_url = match url.rfind('@') {
        Some(at_pos)
            if !url[at_pos + 1..].is_empty()
                && url[at_pos + 1..].chars().all(|c| c.is_ascii_digit()) =>
        {
            &url[..at_pos]
        }
        _ => url,
    };

    // Try parsing as-is first. If the URL lacks a scheme (e.g., "worker1:8080"),
    // Url::parse may treat the host as a scheme — detect this via missing host_str()
    // and fall back to prefixing "http://".
    let try_parse = |u: &str| -> Option<String> {
        url::Url::parse(u)
            .ok()
            .and_then(|p| p.host_str().map(|h| h.to_string()))
    };

    if let Some(host) = try_parse(clean_url) {
        host
    } else if !clean_url.contains("://") {
        try_parse(&format!("http://{clean_url}")).unwrap_or_else(|| {
            tracing::warn!("Failed to parse URL '{}', defaulting to localhost", url);
            "localhost".to_string()
        })
    } else {
        tracing::warn!("Failed to parse URL '{}', defaulting to localhost", url);
        "localhost".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::worker::worker::Worker;

    #[test]
    fn zmq_engine_group_sets_size_without_rank_or_url_rewrite() {
        // A grouped ZMQ worker is one worker awaiting N engines: dp_size set,
        // no rank, URL untouched — the opposite of dp_config's rank expansion.
        let worker = BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
            .connection_mode(ConnectionMode::Zmq)
            .zmq_engine_group(4)
            .build();
        assert_eq!(worker.metadata().spec.url, "ipc:///tmp/w.ipc");
        assert_eq!(worker.metadata().spec.dp_size, Some(4));
        assert_eq!(worker.metadata().spec.dp_rank, None);
        assert_eq!(worker.metadata().zmq_engine_count(), 4);
    }

    #[test]
    fn ungrouped_worker_awaits_one_engine() {
        let worker = BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
            .connection_mode(ConnectionMode::Zmq)
            .build();
        assert_eq!(worker.metadata().zmq_engine_count(), 1);
    }

    #[test]
    fn http_client_is_deferred_until_first_use_and_shared_across_clones() {
        // A ZMQ worker never issues an HTTP request, so nothing is built at
        // construction; asking for it materializes one usable client, and a
        // clone keeps pointing at the same lazy slot.
        let worker = BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
            .connection_mode(ConnectionMode::Zmq)
            .build();
        assert!(worker.http_client.cell_is_empty());
        let clone = worker.clone();
        let client = worker.http_client();
        assert!(!worker.http_client.cell_is_empty());
        assert!(std::ptr::eq(client, clone.http_client()));
    }

    #[test]
    fn provided_http_client_is_used_as_is() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .http_client(Arc::new(reqwest::Client::new()))
            .build();
        assert!(!worker.http_client.cell_is_empty());
    }

    #[test]
    fn overload_overrides_resolve_per_signal_against_gateway_defaults() {
        let worker = BasicWorkerBuilder::new("http://w:1")
            .overload(OverloadUpdate {
                waiting_requests: None,
                token_usage: Some(0.5),
            })
            .overload_defaults(OverloadThresholds {
                waiting_requests: Some(16),
                token_usage: Some(0.9),
            })
            .build();
        // Worker override wins its signal; the other keeps the gateway value.
        assert_eq!(
            worker.metadata().overload,
            OverloadThresholds {
                waiting_requests: Some(16),
                token_usage: Some(0.5),
            }
        );
        // The block itself rides the spec, so it survives spec-based rebuilds
        // and appears in `GET /workers`.
        assert_eq!(worker.metadata().spec.overload.token_usage, Some(0.5));
    }

    #[test]
    fn spec_overload_block_alone_enables_protection() {
        let mut spec = WorkerSpec::new("http://w:1");
        spec.overload.waiting_requests = Some(4);
        let worker = BasicWorkerBuilder::from_spec(spec).build();
        assert!(worker.metadata().overload.is_enabled());
        assert_eq!(worker.metadata().overload.waiting_requests, Some(4));

        // No block, no defaults: protection stays off for this worker.
        let plain = BasicWorkerBuilder::new("http://w:2").build();
        assert!(!plain.metadata().overload.is_enabled());
    }

    #[test]
    fn zmq_handshake_address_reaches_the_built_spec() {
        // The connect path reads the override from the built worker's spec, so
        // dropping it here would silently fall back to the derived address.
        let worker = BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
            .connection_mode(ConnectionMode::Zmq)
            .zmq_handshake_address("tcp://127.0.0.1:30500")
            .build();
        assert_eq!(
            worker.metadata().spec.zmq_handshake_address.as_deref(),
            Some("tcp://127.0.0.1:30500")
        );
    }

    #[test]
    fn test_basic_worker_builder_minimal() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080").build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
        assert_eq!(worker.connection_mode(), &ConnectionMode::Http);
        // Health-checked workers start Pending (not routable until health checker promotes)
        assert!(!worker.is_healthy());
        assert_eq!(worker.status(), WorkerStatus::Pending);
    }

    #[test]
    fn test_basic_worker_builder_with_type() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .worker_type(WorkerType::Decode)
            .build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Decode);
        assert_eq!(worker.connection_mode(), &ConnectionMode::Http);
        assert!(!worker.is_healthy());
    }

    #[test]
    fn test_basic_worker_builder_full() {
        let mut labels = HashMap::new();
        labels.insert("env".to_string(), "prod".to_string());
        labels.insert("region".to_string(), "us-east".to_string());

        let health_config = HealthCheckConfig {
            timeout_secs: 30,
            check_interval_secs: 60,
            failure_threshold: 3,
            success_threshold: 2,
            disable_health_check: false,
            drain_settle_secs: 5,
        };

        let cb_config = CircuitBreakerConfig {
            failure_threshold: 10,
            success_threshold: 5,
            timeout_duration: Duration::from_millis(2000),
            window_duration: Duration::from_millis(30000),
        };

        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .worker_type(WorkerType::Prefill)
            .connection_mode(ConnectionMode::Grpc)
            .labels(labels.clone())
            .health_config(health_config.clone())
            .health_endpoint("/health")
            .circuit_breaker_config(cb_config)
            .build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Prefill);
        assert_eq!(worker.connection_mode(), &ConnectionMode::Grpc);
        assert_eq!(worker.metadata().spec.labels, labels);
        assert_eq!(worker.metadata().health_endpoint, "/health");
        assert_eq!(
            worker.metadata().health_config.timeout_secs,
            health_config.timeout_secs
        );
        assert_eq!(
            worker.metadata().health_config.check_interval_secs,
            health_config.check_interval_secs
        );
        assert_eq!(
            worker.metadata().health_config.failure_threshold,
            health_config.failure_threshold
        );
        assert_eq!(
            worker.metadata().health_config.success_threshold,
            health_config.success_threshold
        );
    }

    #[test]
    fn test_basic_worker_builder_with_single_label() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .worker_type(WorkerType::Decode)
            .label("env", "staging")
            .label("version", "v1.2.3")
            .build();

        assert_eq!(
            worker.metadata().spec.labels.get("env"),
            Some(&"staging".to_string())
        );
        assert_eq!(
            worker.metadata().spec.labels.get("version"),
            Some(&"v1.2.3".to_string())
        );
    }

    #[test]
    fn test_dp_aware_worker_builder_minimal() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .dp_config(2, 8)
            .build();

        assert_eq!(worker.url(), "http://localhost:8080@2");
        assert_eq!(worker.dp_rank(), Some(2));
        assert_eq!(worker.dp_size(), Some(8));
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
    }

    #[test]
    fn test_dp_aware_worker_builder_full() {
        let mut labels = HashMap::new();
        labels.insert("cluster".to_string(), "main".to_string());

        let health_config = HealthCheckConfig {
            timeout_secs: 20,
            check_interval_secs: 45,
            failure_threshold: 5,
            success_threshold: 3,
            disable_health_check: false,
            drain_settle_secs: 5,
        };

        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .dp_config(3, 16)
            .worker_type(WorkerType::Prefill)
            .bootstrap_port(Some(9090))
            .connection_mode(ConnectionMode::Http)
            .labels(labels.clone())
            .health_config(health_config.clone())
            .health_endpoint("/status")
            .api_key("test_api_key")
            .build();

        assert_eq!(worker.url(), "http://localhost:8080@3");
        assert_eq!(worker.dp_rank(), Some(3));
        assert_eq!(worker.dp_size(), Some(16));
        assert_eq!(worker.metadata().spec.labels, labels);
        assert_eq!(worker.metadata().health_endpoint, "/status");
        assert_eq!(
            worker.metadata().health_config.timeout_secs,
            health_config.timeout_secs
        );
        assert_eq!(
            worker.metadata().health_config.check_interval_secs,
            health_config.check_interval_secs
        );
        assert_eq!(
            worker.metadata().health_config.failure_threshold,
            health_config.failure_threshold
        );
        assert_eq!(
            worker.metadata().health_config.success_threshold,
            health_config.success_threshold
        );
    }

    #[test]
    fn test_dp_aware_worker_with_grpc() {
        let worker = BasicWorkerBuilder::new("grpc://cluster.local")
            .dp_config(1, 4)
            .worker_type(WorkerType::Decode)
            .connection_mode(ConnectionMode::Grpc)
            .label("transport", "grpc")
            .build();

        assert_eq!(worker.url(), "grpc://cluster.local@1");
        assert_eq!(worker.dp_rank(), Some(1));
        assert_eq!(worker.dp_size(), Some(4));
        assert_eq!(worker.worker_type(), &WorkerType::Decode);
        assert_eq!(worker.connection_mode(), &ConnectionMode::Grpc);
        assert_eq!(
            worker.metadata().spec.labels.get("transport"),
            Some(&"grpc".to_string())
        );
    }

    #[test]
    fn test_parse_bootstrap_host_normal_url() {
        assert_eq!(parse_bootstrap_host("http://worker1:8080"), "worker1");
        assert_eq!(parse_bootstrap_host("https://10.0.0.5:443"), "10.0.0.5");
        assert_eq!(
            parse_bootstrap_host("grpc://cluster.local"),
            "cluster.local"
        );
    }

    #[test]
    fn test_parse_bootstrap_host_dp_aware_url() {
        // DP-aware URLs use @rank suffix — must extract host, not rank
        assert_eq!(parse_bootstrap_host("http://worker1:8080@0"), "worker1");
        assert_eq!(parse_bootstrap_host("http://worker1:8080@3"), "worker1");
        assert_eq!(
            parse_bootstrap_host("grpc://prefill.local@7"),
            "prefill.local"
        );
    }

    #[test]
    fn test_parse_bootstrap_host_bare_host() {
        assert_eq!(parse_bootstrap_host("worker1:8080"), "worker1");
        assert_eq!(parse_bootstrap_host("localhost"), "localhost");
    }

    #[test]
    fn test_dp_aware_worker_bootstrap_host() {
        let worker = BasicWorkerBuilder::new("http://prefill1:8080")
            .dp_config(3, 8)
            .worker_type(WorkerType::Prefill)
            .build();

        // bootstrap_host should be "prefill1", not "3"
        assert_eq!(worker.metadata().spec.bootstrap_host, "prefill1");
    }
}
