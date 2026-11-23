use std::{collections::HashMap, marker::PhantomData, sync::Arc, time::Duration};

use arc_swap::{ArcSwap, ArcSwapOption};
use tokio::sync::mpsc;

use super::{
    basic::{BasicWorker, ConnectionMode, RuntimeType, WorkerMetadata, WorkerRuntime, WorkerType},
    endpoint::Endpoint,
    metadata::DecodingLabels,
};
use crate::{
    connection::{
        engine::{
            ConnectorDefault, EngineConnection, EngineConnectionCache, EngineConnector, NoConnector,
        },
        http_client::LazyHttpClient,
    },
    health::{
        circuit_breaker::{CircuitBreaker, CircuitBreakerConfig},
        overload::OverloadThresholds,
        resilience::ResolvedResilience,
    },
    identity::basic::Worker,
    metrics::writers::WorkerHandles,
    model_card::ModelCard,
    pd::pairing::PdPairing,
    registry::event::WorkerSignal,
    settings::DEFAULT_ZMQ_STARTUP_TIMEOUT_SECS,
    worker::{
        ConstrainedDecoding, HealthCheckConfig, OverloadUpdate, WorkerModels, WorkerSpec,
        WorkerStatus,
    },
};

/// The URL of the rank `rank` of the engine at `base`.
pub fn rank_url(base: &str, rank: usize) -> String {
    format!("{base}@{rank}")
}

/// Builder for creating BasicWorker instances with fluent API.
///
/// Internally stores a [`WorkerSpec`] for identity/config fields.
/// Callers with a pre-built `WorkerSpec` can use [`from_spec()`](Self::from_spec).
///
/// `D` names the connector of a worker the builder is handed none for, as a
/// map's type names its hasher: none here, and a host names the builder with
/// its own ([`ConnectorDefault`]).
pub struct BasicWorkerBuilder<D = NoConnector> {
    spec: WorkerSpec,
    /// Resolved health config (router defaults + per-worker overrides).
    /// If not set, falls back to `HealthCheckConfig::default()`.
    health_config: Option<HealthCheckConfig>,
    health_endpoint: String,
    /// The per-stage silence bound of a ZMQ engine's startup handshake; the
    /// router config's default when not set.
    zmq_startup_timeout: Option<Duration>,
    /// The bound on a request over the worker's ZMQ link, from its dispatch
    /// to its last message (the worker's own `http_pool.timeout_secs`, else
    /// the router config's `request_timeout_secs`, as the registration step
    /// resolves it); the link's streams are unbounded when not set.
    zmq_request_timeout: Option<Duration>,
    circuit_breaker_config: CircuitBreakerConfig,
    /// An engine connection that is already made (if not set, the worker
    /// connects on first use).
    engine_connection: Option<Arc<dyn EngineConnection>>,
    /// The connection slot of the rank workers of one ZMQ engine group (if
    /// not set, the worker has a slot of its own).
    shared_engine_connection: Option<Arc<EngineConnectionCache>>,
    /// What makes the worker's engine connection (if not set, the default
    /// `D` names).
    engine_connector: Option<Arc<dyn EngineConnector>>,
    /// The type that names the default; no value is held.
    default_connector: PhantomData<fn() -> D>,
    /// Pre-built worker-directed HTTP client (if not set, a default is created).
    http_client: Option<Arc<reqwest::Client>>,
    /// Whether `http_client` speaks HTTP/2 prior knowledge.
    http2: bool,
    /// Resolved resilience config (if not set, defaults are used).
    resilience: Option<ResolvedResilience>,
    /// Initial lifecycle status. If unset, defaults to `Pending` for
    /// health-checked workers and `Ready` for `disable_health_check == true`.
    /// Callers replacing an existing worker (e.g. metadata updates) should
    /// pass the old worker's status to avoid kicking it back to Pending.
    initial_status: Option<WorkerStatus>,
    /// The registry's signal sender: the worker's connect signal and the
    /// moves of its liveness veto ride it to the manager's loop.
    connect_signal_tx: Option<mpsc::UnboundedSender<WorkerSignal>>,
    /// Gateway-level overload thresholds the spec's `overload` block resolves
    /// against. Default empty: a spec block alone still enables protection
    /// for this worker.
    overload_defaults: OverloadThresholds,
    /// The labels discovery read from the engine, recorded on the worker so a
    /// later refresh replaces exactly these; empty when none were discovered.
    discovered_labels: HashMap<String, String>,
}

impl<D: ConnectorDefault> BasicWorkerBuilder<D> {
    /// Create a new builder with only the URL (uses default WorkerSpec)
    pub fn new(url: impl Into<String>) -> Self {
        Self::from_spec(WorkerSpec::new(url))
    }

    /// Create a builder from an existing WorkerSpec, every other field at its default: no health
    /// config and `/health` as the endpoint, the default circuit breaker and overload thresholds,
    /// no clients, HTTP/1, no resilience, no initial status, no connect signal.
    pub fn from_spec(spec: WorkerSpec) -> Self {
        Self {
            spec,
            health_config: None,
            health_endpoint: "/health".to_string(),
            zmq_startup_timeout: None,
            zmq_request_timeout: None,
            circuit_breaker_config: CircuitBreakerConfig::default(),
            engine_connection: None,
            shared_engine_connection: None,
            engine_connector: None,
            default_connector: PhantomData,
            http_client: None,
            http2: false,
            resilience: None,
            initial_status: None,
            connect_signal_tx: None,
            overload_defaults: OverloadThresholds::default(),
            discovered_labels: HashMap::new(),
        }
    }

    /// A builder over a live worker's resolved state: the spec, HTTP/2, the
    /// discovered labels, the health config and endpoint, the resilience,
    /// the breaker config, the connector, the ZMQ bounds, the overload block, a
    /// materialized HTTP client and the current status, so a rebuild keeps
    /// what registration resolved. The caller applies its own changes, the
    /// gateway's overload defaults and the connect signal, then shares the
    /// KV engine state with the worker it replaces.
    pub fn carrying(worker: &Arc<dyn Worker>) -> Self {
        let metadata = worker.metadata();
        let mut builder = Self::from_spec((*metadata.spec).clone())
            .http2(worker.http2())
            .discovered_labels((*metadata.discovered_labels).clone())
            .health_config(metadata.health_config.clone())
            .health_endpoint(metadata.health_endpoint.clone())
            .resilience(worker.resilience().clone())
            // The resolved thresholds stand as the defaults the spec's block
            // re-resolves against, so a rebuild without the gateway config at
            // hand keeps the effective thresholds.
            .overload(metadata.spec.overload)
            .overload_defaults(metadata.overload)
            .zmq_startup_timeout(metadata.zmq_startup_timeout)
            .status(worker.status());
        // `from_spec` starts on the built-in breaker config; with the live one
        // the registry's replacement adopts the live breaker and its state.
        if let Some(config) = worker
            .as_any()
            .downcast_ref::<BasicWorker>()
            .map(BasicWorker::circuit_breaker_config)
        {
            builder = builder.circuit_breaker_config(config);
        }
        // The rebuild connects the way the live worker does.
        if let Some(connector) = worker.engine_connector() {
            builder = builder.engine_connector(connector);
        }
        if let Some(timeout) = metadata.zmq_request_timeout {
            builder = builder.zmq_request_timeout(timeout);
        }
        // A never-HTTP worker keeps a deferred client slot.
        if let Some(client) = worker.http_client_handle_if_initialized() {
            builder = builder.http_client(client);
        }
        builder
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

    /// Set the runtime type
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

    /// Record the labels discovery read from the engine for this worker
    /// (before the operator's labels were merged over them). A later refresh
    /// replaces exactly these; `labels` stays the merged set the worker shows.
    pub fn discovered_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.discovered_labels = labels;
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

    /// Set the longest any one stage of a ZMQ engine's startup handshake may
    /// stay silent (internal-only, from router config): it alone covers the
    /// engine's init between its HELLO and its READY.
    pub fn zmq_startup_timeout(mut self, timeout: Duration) -> Self {
        self.zmq_startup_timeout = Some(timeout);
        self
    }

    /// Set the bound on a request over the worker's ZMQ link, from its
    /// dispatch to its last message (internal-only: the registration step
    /// resolves it as the HTTP worker client resolves its per-request
    /// timeout, the worker's own `http_pool.timeout_secs` else the router's
    /// `--request-timeout-secs`): a stream the engine leaves silent ends at
    /// it with the gateway's timeout and the engine request is aborted. Unset
    /// leaves the link's streams open for as long as the client reads them.
    pub fn zmq_request_timeout(mut self, timeout: Duration) -> Self {
        self.zmq_request_timeout = Some(timeout);
        self
    }

    /// Set circuit breaker configuration
    pub fn circuit_breaker_config(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker_config = config;
        self
    }

    /// Hand the worker an engine connection that is already made (gRPC or
    /// ZMQ), so its first use needs no connect.
    pub fn engine_connection(mut self, connection: impl EngineConnection) -> Self {
        self.engine_connection = Some(Arc::new(connection));
        self
    }

    /// Share one connection slot with the other rank workers of a ZMQ engine
    /// group: the group's engines dial one socket set, so its rank workers
    /// run one handshake between them and follow its reconnects together.
    #[cfg(any(feature = "workflow", test))]
    pub(crate) fn shared_engine_connection(mut self, slot: Arc<EngineConnectionCache>) -> Self {
        self.shared_engine_connection = Some(slot);
        self
    }

    /// Set what makes the worker's engine connection, in place of the default
    /// `D` names. A gRPC or ZMQ worker with neither refuses its connect.
    pub fn engine_connector(mut self, connector: Arc<dyn EngineConnector>) -> Self {
        self.engine_connector = Some(connector);
        self
    }

    /// Wire the signal sender (from the registry). The worker says two
    /// things on it to the manager's loop. Its connection completed: a ZMQ
    /// worker is promoted the instant its handshake lands, and a demoted
    /// worker is probed at its first contact. And its liveness veto moved,
    /// which the loop tells the health sink.
    pub fn connect_signal_tx(mut self, tx: mpsc::UnboundedSender<WorkerSignal>) -> Self {
        self.connect_signal_tx = Some(tx);
        self
    }

    /// Set a pre-built worker-directed HTTP client. The strong handle keeps
    /// the client's shared cache entry alive for the worker's lifetime.
    pub fn http_client(mut self, client: Arc<reqwest::Client>) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Record that the worker's HTTP client speaks HTTP/2 prior knowledge.
    pub fn http2(mut self, http2: bool) -> Self {
        self.http2 = http2;
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

    /// Set KV transfer engine id (the engine's `kv_transfer_config.engine_id`)
    pub fn kv_engine_id(mut self, engine_id: impl Into<String>) -> Self {
        self.spec.kv_engine_id = Some(engine_id.into());
        self
    }

    /// Set when the gateway hands this worker's engine a grammar: on every turn,
    /// or only when the request asks ([`ConstrainedDecoding`]). Unset, the
    /// worker's `constrained_decoding` label decides.
    pub fn constrained_decoding(mut self, policy: ConstrainedDecoding) -> Self {
        self.spec.constrained_decoding = Some(policy);
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
        self.spec.url = rank_url(&base_url, rank);
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

    /// Build the BasicWorker instance.
    ///
    /// Building writes no metric series: a worker publishes its own where it
    /// enters a registry ([`Worker::publish_series`]). A worker that is built
    /// and never enters leaves the series of its address as they stood.
    pub fn build(mut self) -> BasicWorker {
        use std::sync::atomic::AtomicBool;

        use tokio::sync::OnceCell;

        // bootstrap_host is a PD/TCP-disaggregation concept (a host:port peer),
        // so derive it only for transports that carry a host. An ipc:// ZMQ
        // worker has no host; leave bootstrap_host empty rather than forcing the
        // URL through host:port parsing (which would warn and default to
        // localhost).
        if self.spec.connection_mode != ConnectionMode::Zmq {
            self.spec.bootstrap_host = bootstrap_host(&self.spec);
        }

        // Resolve health config: use explicit config if set, otherwise
        // apply per-worker overrides from spec.health to defaults.
        let health_config = self
            .health_config
            .unwrap_or_else(|| self.spec.health.apply_to(&HealthCheckConfig::default()));

        let metadata = WorkerMetadata {
            overload: OverloadThresholds::resolve(&self.spec.overload, self.overload_defaults),
            pd_pairing: PdPairing::derive(&self.spec),
            redacted_url: self.spec.redacted_url(),
            decoding: DecodingLabels::of(&self.spec),
            spec: Arc::new(self.spec),
            health_config,
            health_endpoint: self.health_endpoint,
            zmq_startup_timeout: self
                .zmq_startup_timeout
                .unwrap_or(Duration::from_secs(DEFAULT_ZMQ_STARTUP_TIMEOUT_SECS)),
            zmq_request_timeout: self.zmq_request_timeout,
            http2: self.http2,
            discovered_labels: Arc::new(self.discovered_labels),
        };

        // OnceCell for lock-free client access after initialization; ArcSwap so
        // the ZMQ health probe can evict a dead client (see BasicWorker docs).
        let engine_connection = if let Some(slot) = self.shared_engine_connection {
            slot
        } else {
            let cell = OnceCell::new();
            if let Some(connection) = self.engine_connection {
                // Pre-set the connection if provided (set on a fresh cell cannot fail)
                cell.set(connection).ok();
            }
            Arc::new(EngineConnectionCache::new(cell))
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

        let http_client = Arc::new(match self.http_client {
            Some(client) => LazyHttpClient::ready(client),
            None => LazyHttpClient::deferred(),
        });

        let resilience = self.resilience.unwrap_or_default();

        let metric_handles = Arc::new(WorkerHandles::new(metadata.spec.url.as_str()));

        BasicWorker {
            kv_engine_id: Arc::new(ArcSwapOption::new(
                metadata.spec.kv_engine_id.clone().map(Arc::new),
            )),
            kv_engine_id_unconfirmed: Arc::new(AtomicBool::new(false)),
            runtime: ArcSwap::from_pointee(WorkerRuntime::new(&metadata.spec.url, initial_status)),
            circuit_breaker: ArcSwap::from_pointee(CircuitBreaker::with_config_and_label(
                self.circuit_breaker_config,
                metadata.spec.url.clone(),
            )),
            metadata,
            engine_connection,
            engine_connector: self.engine_connector.or_else(D::default_connector),
            load_source: Arc::new(OnceCell::new()),
            zmq_connect_started: Arc::new(AtomicBool::new(false)),
            zmq_connect_abort: Arc::new(ArcSwapOption::empty()),
            superseded: AtomicBool::new(false),
            connect_signal_tx: self.connect_signal_tx,
            models_override: Arc::new(ArcSwap::from_pointee(WorkerModels::Wildcard)),
            http_client,
            resilience,
            completion_sink: Arc::new(std::sync::RwLock::new(None)),
            metric_handles,
        }
    }
}

/// The host a worker's partners reach it at for the bootstrap handshake:
/// the host of the address requests dial, as a client reads it, falling
/// back to "localhost" for an address no client reads.
///
/// The address requests dial is [`WorkerMetadata::base_url`]'s, and the
/// rule is read there: the base a spec holds, which the registration keeps
/// beside the address it writes as `{base}@{rank}`, and the address as
/// written for a spec that holds none. In an address as written an `@` and
/// digits at the end are a userinfo and a host, and no rank to strip.
/// Stripped as one, they leave what stands before them as the host, and
/// that may be a token: `http://token@8080` is a userinfo and the host
/// `0.0.31.144`, the number `8080` as a client reads an HTTP host.
fn bootstrap_host(spec: &WorkerSpec) -> String {
    // The address requests dial (`WorkerMetadata::base_url`): a spec's base
    // where it holds one, its address as written where it holds none.
    let dialled = spec.dp_base_url.as_deref().unwrap_or(spec.url.as_str());
    let host_of = |address: &str| {
        url::Url::parse(address)
            .ok()
            .and_then(|parsed| parsed.host_str().map(str::to_string))
    };
    // Without a scheme (`worker1:8080`) the URL library takes the host for
    // one and finds no host: such an address is read again behind `http://`.
    let host = host_of(dialled).or_else(|| {
        (!dialled.contains("://"))
            .then(|| host_of(&format!("http://{dialled}")))
            .flatten()
    });
    host.unwrap_or_else(|| {
        // Named as it was read, with no rank read out of it. The mask a log
        // line names a worker by reads digits behind the last `@` as a rank
        // and writes what stands before them as the host: a token, where
        // the digits are a host no client reads.
        let shown = Endpoint::parse(dialled).map_or_else(
            |_| "<unparsable address>".to_string(),
            |address| address.redacted(),
        );
        tracing::warn!("Failed to parse URL '{shown}', defaulting to localhost");
        "localhost".to_string()
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::{connection::client_factory::http_client, identity::basic::Worker};

    /// The builder without a default connector, as this module has it.
    type BasicWorkerBuilder = super::BasicWorkerBuilder;

    /// `new(url)` is `from_spec` of the default spec for `url`: the two constructors give the same
    /// worker, every default written once.
    #[test]
    fn a_builder_from_a_url_is_the_builder_from_its_default_spec() {
        let from_url = BasicWorkerBuilder::new("http://w:1").build();
        let from_spec = BasicWorkerBuilder::from_spec(WorkerSpec::new("http://w:1")).build();
        assert_eq!(
            format!("{:?}", from_url.metadata().spec),
            format!("{:?}", from_spec.metadata().spec)
        );
        assert_eq!(
            format!("{:?}", from_url.metadata().health_config),
            format!("{:?}", from_spec.metadata().health_config)
        );
        assert_eq!(from_url.metadata().health_endpoint, "/health");
        assert_eq!(
            from_url.metadata().health_endpoint,
            from_spec.metadata().health_endpoint
        );
        assert_eq!(from_url.worker_type(), from_spec.worker_type());
        assert_eq!(from_url.is_healthy(), from_spec.is_healthy());
    }

    /// The rendered URL is computed once at build: a URL without userinfo is
    /// the URL byte for byte (its spelling kept), one with userinfo masks it
    /// whole and keeps the DP rank.
    #[test]
    fn the_rendered_url_masks_a_userinfo_and_keeps_a_plain_url_byte_for_byte() {
        let plain = BasicWorkerBuilder::new("http://Worker:8080/").build();
        assert_eq!(plain.redacted_url(), plain.url());
        let with_secret = BasicWorkerBuilder::new("http://user:secret@worker:8080")
            .dp_config(2, 4)
            .build();
        assert_eq!(with_secret.url(), "http://user:secret@worker:8080@2");
        assert_eq!(with_secret.redacted_url(), "http://***@worker:8080@2");
    }

    /// A rank is read out of an address by its spec's own fields alone. An
    /// address that ends in an `@` and digits beside no base, or beside a
    /// base it is not built from, is a userinfo before a host: it is
    /// rendered with the userinfo masked, never raw.
    #[test]
    fn the_rendered_url_reads_no_rank_out_of_an_address_its_spec_does_not_rank() {
        let alone = BasicWorkerBuilder::new("http://pebble@2130706433").build();
        assert_eq!(alone.redacted_url(), "http://***@2130706433");
        let mut spec = WorkerSpec::new("http://pebble@2130706433");
        spec.dp_base_url = Some("http://other:8000".to_string());
        let beside = BasicWorkerBuilder::from_spec(spec).build();
        assert_eq!(beside.redacted_url(), "http://***@2130706433");
    }

    /// An IPC address holds no userinfo: an `@` and digits at its end are
    /// bytes of its path. A socket its spec does not rank renders as it
    /// stands, so two such sockets render apart.
    #[test]
    fn the_rendered_url_keeps_a_socket_path_that_ends_like_a_rank() {
        for url in ["ipc:///tmp/w", "ipc:///tmp/w@0", "ipc:///tmp/w@00"] {
            let alone = BasicWorkerBuilder::new(url).build();
            assert_eq!(alone.redacted_url(), url);
            let mut spec = WorkerSpec::new(url);
            spec.dp_base_url = Some("ipc:///tmp/g".to_string());
            let beside = BasicWorkerBuilder::from_spec(spec).build();
            assert_eq!(beside.redacted_url(), url);
        }
        // A socket's rank still renders with its base.
        let mut spec = WorkerSpec::new("ipc:///tmp/g@1");
        spec.dp_base_url = Some("ipc:///tmp/g".to_string());
        let ranked = BasicWorkerBuilder::from_spec(spec).build();
        assert_eq!(ranked.redacted_url(), "ipc:///tmp/g@1");
    }

    #[test]
    fn a_ranks_url_is_formed_in_one_place() {
        let base = "http://worker:8000";
        for (rank, url) in [
            "http://worker:8000@0",
            "http://worker:8000@1",
            "http://worker:8000@2",
        ]
        .into_iter()
        .enumerate()
        {
            let worker = BasicWorkerBuilder::new(base).dp_config(rank, 3).build();
            assert_eq!(rank_url(base, rank), url);
            assert_eq!(worker.url(), url);
        }
    }

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
        assert_eq!(worker.metadata().zmq_engine_count(), Some(4));
    }

    #[test]
    fn ungrouped_worker_awaits_the_engines_it_is_dialed_by() {
        // No count: the handshake awaits the group the first engine declares
        // (one engine for an engine that declares none).
        let worker = BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
            .connection_mode(ConnectionMode::Zmq)
            .build();
        assert_eq!(worker.metadata().zmq_engine_count(), None);
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
            .http_client(Arc::new(http_client()))
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

    /// The connect driver reads the request bound from the built worker's
    /// metadata; a builder that sets none leaves the link's streams unbounded.
    #[test]
    fn zmq_request_timeout_reaches_the_built_metadata() {
        let bounded = BasicWorkerBuilder::new("ipc:///tmp/w.ipc")
            .connection_mode(ConnectionMode::Zmq)
            .zmq_request_timeout(Duration::from_secs(90))
            .build();
        assert_eq!(
            bounded.metadata().zmq_request_timeout,
            Some(Duration::from_secs(90))
        );
        let unbounded = BasicWorkerBuilder::new("ipc:///tmp/w2.ipc")
            .connection_mode(ConnectionMode::Zmq)
            .build();
        assert_eq!(unbounded.metadata().zmq_request_timeout, None);
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

    /// The built worker's health check reads back `health_config`'s timeout,
    /// interval and thresholds.
    fn assert_health_config(worker: &BasicWorker, health_config: &HealthCheckConfig) {
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
        assert_health_config(&worker, &health_config);
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
        assert_health_config(&worker, &health_config);
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

    /// The bootstrap host of a worker built from `url` alone, with no rank.
    fn bootstrap_host_of(url: &str) -> String {
        let worker = BasicWorkerBuilder::new(url).build();
        worker.metadata().spec.bootstrap_host.clone()
    }

    #[test]
    fn test_parse_bootstrap_host_normal_url() {
        assert_eq!(bootstrap_host_of("http://worker1:8080"), "worker1");
        assert_eq!(bootstrap_host_of("https://10.0.0.5:443"), "10.0.0.5");
        assert_eq!(bootstrap_host_of("grpc://cluster.local"), "cluster.local");
    }

    /// A rank-expanded worker's bootstrap host is the host of its base: the
    /// rank the registration appended is no part of it.
    #[test]
    fn test_parse_bootstrap_host_dp_aware_url() {
        let rank_of = |base: &str, rank: usize| {
            let worker = BasicWorkerBuilder::new(base).dp_config(rank, 8).build();
            worker.metadata().spec.bootstrap_host.clone()
        };
        assert_eq!(rank_of("http://worker1:8080", 0), "worker1");
        assert_eq!(rank_of("http://worker1:8080", 3), "worker1");
        assert_eq!(rank_of("grpc://prefill.local", 7), "prefill.local");
    }

    #[test]
    fn test_parse_bootstrap_host_bare_host() {
        assert_eq!(bootstrap_host_of("worker1:8080"), "worker1");
        assert_eq!(bootstrap_host_of("localhost"), "localhost");
    }

    /// A worker the registration did not expand has no rank to strip: an
    /// `@` and digits at the end of its address are a userinfo and a host,
    /// and its bootstrap host is that host, as a client dials it. Stripped
    /// as a rank, they left the userinfo as the host: a token, written in
    /// the registration probe's line and handed to the worker's partners.
    #[test]
    fn the_bootstrap_host_of_a_worker_with_no_rank_is_the_host_a_client_dials() {
        for (url, host) in [
            ("http://s3cret-token@2130706433", "127.0.0.1"),
            ("https://s3cret-token@2130706433", "127.0.0.1"),
            ("grpc://s3cret-token@2130706433", "2130706433"),
            ("http://adm1n:s3cret@2130706433", "127.0.0.1"),
            ("s3cret-token@2130706433", "127.0.0.1"),
            ("http://s3cret-token@2130706433/", "127.0.0.1"),
            ("http://adm1n:s3cret@2130706433/", "127.0.0.1"),
        ] {
            assert_eq!(bootstrap_host_of(url), host, "{url}");
        }
    }

    /// An address no client reads falls back to "localhost", and the
    /// fallback's warning names it with no rank read out of it. The digits
    /// are a host here: read as a rank, they left the token before them to
    /// be written whole, as the host. An address the address type reads
    /// with no rank is named as before, its userinfo masked: the third
    /// spelling, whose digits fit no rank, and a socket path.
    #[test]
    #[tracing_test::traced_test]
    fn the_bootstrap_host_of_an_address_no_client_reads_is_localhost() {
        for url in [
            "http://s3cret-token@99999999999",
            "s3cret-token@99999999999",
            "http://s3cret-token@18446744073709551616",
        ] {
            assert_eq!(bootstrap_host_of(url), "localhost", "{url}");
        }
        assert!(!logs_contain("s3cret-token"));
        assert!(logs_contain(
            "Failed to parse URL '<unparsable address>', defaulting to localhost"
        ));
        assert!(logs_contain(
            "Failed to parse URL 'http://***@18446744073709551616', defaulting to localhost"
        ));

        assert_eq!(bootstrap_host_of("ipc:///tmp/engine.sock"), "localhost");
        assert!(logs_contain(
            "Failed to parse URL 'ipc:///tmp/engine.sock', defaulting to localhost"
        ));
    }

    /// The bootstrap host is the host of the address requests dial
    /// (`base_url`), whatever a spec holds of a base and a rank. The
    /// registration writes both. A spec a peer hands over may hold a base
    /// and no rank, or a base that is not its address less the rank, and
    /// requests dial that base.
    #[test]
    fn the_bootstrap_host_is_the_host_of_the_address_requests_dial() {
        let base = "http://prefill1:8080";
        let ranked = "http://prefill1:8080@3";
        for (url, holds_base, holds_rank, host) in [
            (base, false, false, "prefill1"),
            (ranked, true, true, "prefill1"),
            (ranked, true, false, "prefill1"),
            (ranked, false, true, "0.0.0.3"),
            ("http://other:9000@3", true, true, "prefill1"),
        ] {
            let mut spec = WorkerSpec::new(url);
            spec.dp_base_url = holds_base.then(|| base.to_string());
            spec.dp_rank = holds_rank.then_some(3);
            let worker = BasicWorkerBuilder::from_spec(spec).build();

            let dialled = url::Url::parse(worker.base_url()).unwrap();
            let bootstrap_host = worker.metadata().spec.bootstrap_host.as_str();
            let cell = format!("{url}, a base {holds_base}, a rank {holds_rank}");
            assert_eq!(Some(bootstrap_host), dialled.host_str(), "{cell}");
            assert_eq!(bootstrap_host, host, "{cell}");
        }
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

    /// A URL registered with a trailing slash keeps it as the registry's key
    /// (`url()`), and every join reads one slash: the base URL carries none,
    /// for the worker and for a data-parallel rank of it.
    #[test]
    fn a_trailing_slash_on_the_registered_url_is_not_in_the_join_base() {
        let worker = BasicWorkerBuilder::new("http://h:8000/").build();
        assert_eq!(worker.url(), "http://h:8000/");
        assert_eq!(worker.base_url(), "http://h:8000");
        assert_eq!(
            worker.endpoint_url("/get_server_info"),
            "http://h:8000/get_server_info"
        );
        assert_eq!(
            format!("{}/{}", worker.base_url(), "get_server_info"),
            "http://h:8000/get_server_info"
        );

        let rank = BasicWorkerBuilder::new("http://h:8000/")
            .dp_config(3, 8)
            .build();
        assert_eq!(rank.base_url(), "http://h:8000");
        assert_eq!(
            rank.endpoint_url("/get_server_info"),
            "http://h:8000/get_server_info"
        );
    }

    /// A socket address is not a base routes join to: the ZMQ adapter hashes
    /// the exact path for its handshake port and socket files, so a trailing
    /// slash on an `ipc://` or `tcp://` registration stays in `base_url()`,
    /// for the worker and for a data-parallel rank of it.
    #[test]
    fn a_trailing_slash_on_a_socket_address_stays_in_the_base() {
        let ipc = BasicWorkerBuilder::new("ipc:///tmp/engine/")
            .connection_mode(ConnectionMode::Zmq)
            .build();
        assert_eq!(ipc.url(), "ipc:///tmp/engine/");
        assert_eq!(ipc.base_url(), "ipc:///tmp/engine/");

        let rank = BasicWorkerBuilder::new("ipc:///tmp/engine/")
            .connection_mode(ConnectionMode::Zmq)
            .dp_config(3, 8)
            .build();
        assert_eq!(rank.base_url(), "ipc:///tmp/engine/");

        let tcp = BasicWorkerBuilder::new("tcp://127.0.0.1:5555/")
            .connection_mode(ConnectionMode::Zmq)
            .build();
        assert_eq!(tcp.base_url(), "tcp://127.0.0.1:5555/");
    }
}
