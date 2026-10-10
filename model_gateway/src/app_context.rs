use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use llm_tokenizer::registry::TokenizerRegistry;
use openai_protocol::profile::ProviderProfile;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use reqwest::Client;
use smg_data_connector::{
    create_storage, ConversationItemStorage, ConversationStorage, ResponseStorage,
    StorageFactoryConfig,
};
use smg_external_router::realtime::webrtc::default_bind_addr;
use smg_mcp::McpOrchestrator;
use tokio::sync::broadcast::error::RecvError;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{debug, info, warn};

use crate::{
    config::{KvIndexKind, RouterConfig},
    middleware::{AuthConfig, TokenBucket},
    observability::inflight_tracker::InFlightRequestTracker,
    policies::PolicyRegistry,
    rate_limit::RateLimitManager,
    routers::{
        common::{
            openai_bridge::FormatRegistry, overload, pd_admission, realtime::RealtimeRegistry,
        },
        gateway::Gateway,
        grpc::multimodal::MultimodalConfigRegistry,
    },
    wasm::{config::WasmRuntimeConfig, module_manager::WasmModuleManager},
    worker::{
        http_client::build_client, liveness, KvEventMonitor, PrefillAdmission,
        WorkerHttpClientCache, WorkerMonitor, WorkerRegistry, WorkerService,
    },
    workflow::{JobQueue, WorkflowEngines},
};

/// Error type for AppContext builder
#[derive(Debug)]
pub enum AppContextBuildError {
    MissingField(&'static str),
    InvalidConfig(String),
}

impl std::fmt::Display for AppContextBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "Missing required field: {field}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
        }
    }
}

impl std::error::Error for AppContextBuildError {}

#[derive(Clone)]
pub struct AppContext {
    pub client: Client,
    pub router_config: RouterConfig,
    /// Every credential that authenticates as this gateway: the shared
    /// `api_key` plus any per-tenant keys, derived once from `router_config`.
    /// The serving auth layer and the `/v1/models` BYOK short-circuit both
    /// read this set, so they cannot drift apart.
    pub gateway_auth: AuthConfig,
    pub rate_limiter: Option<Arc<TokenBucket>>,
    pub rate_limit_manager: Option<Arc<RateLimitManager>>,
    pub tokenizer_registry: Arc<TokenizerRegistry>,
    pub multimodal_config_registry: Arc<MultimodalConfigRegistry>,
    pub reasoning_parser_factory: Option<ReasoningParserFactory>,
    pub tool_parser_factory: Option<ToolParserFactory>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub prefill_admission: Option<Arc<PrefillAdmission>>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub gateway: Option<Arc<Gateway>>,
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
    pub worker_monitor: Option<Arc<WorkerMonitor>>,
    pub configured_reasoning_parser: Option<String>,
    pub configured_tool_parser: Option<String>,
    pub worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>,
    pub workflow_engines: Arc<OnceLock<WorkflowEngines>>,
    pub mcp_orchestrator: Arc<OnceLock<Arc<McpOrchestrator>>>,
    pub mcp_format_registry: FormatRegistry,
    pub wasm_manager: Option<Arc<WasmModuleManager>>,
    pub worker_service: Arc<WorkerService>,
    /// Worker-directed HTTP clients, shared across workers with the same
    /// effective connection config.
    pub worker_client_cache: Arc<WorkerHttpClientCache>,
    pub inflight_tracker: Arc<InFlightRequestTracker>,
    pub kv_event_monitor: Option<Arc<KvEventMonitor>>,
    /// RL control plane state; `None` unless `router_config.rl.enabled`.
    pub rl: Option<Arc<smg_rl::RlState>>,
    pub realtime_registry: Arc<RealtimeRegistry>,
    /// Bind address for WebRTC UDP sockets (`None` = `0.0.0.0`, auto-detect).
    pub webrtc_bind_addr: Option<std::net::IpAddr>,
    /// STUN server for ICE candidate gathering. Defaults to `stun.l.google.com:19302`; `"none"` to disable.
    pub webrtc_stun_server: Option<String>,
}

impl std::fmt::Debug for AppContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppContext")
            .field("router_config", &self.router_config)
            .finish_non_exhaustive()
    }
}

fn start_prefill_admission_notifier(
    worker_registry: &WorkerRegistry,
    admission: &Arc<PrefillAdmission>,
) -> Result<(), AppContextBuildError> {
    let mut events = worker_registry.subscribe_events();
    let admission = Arc::downgrade(admission);
    let runtime = tokio::runtime::Handle::try_current().map_err(|error| {
        AppContextBuildError::InvalidConfig(format!(
            "Prefill admission requires a Tokio runtime: {error}"
        ))
    })?;
    runtime.spawn(async move {
        while let Ok(_) | Err(RecvError::Lagged(_)) = events.recv().await {
            let Some(admission) = admission.upgrade() else {
                break;
            };
            admission.notify_capacity_changed();
        }
    });
    Ok(())
}

/// In-flight requests admitted per available core when
/// `max_concurrent_requests` is unset.
pub(crate) const DEFAULT_INFLIGHT_PER_CORE: usize = 1024;

/// Floor of the derived in-flight bound, so a router with few cores still
/// admits a fleet's worth of long-lived streams.
pub(crate) const DEFAULT_INFLIGHT_FLOOR: usize = 4096;

/// The in-flight bound used when `max_concurrent_requests` is unset (-1):
/// 1024 per available core, at least 4096. Far above a healthy router's
/// in-flight count (arrival rate x latency), and the point past which an
/// overloaded router sheds instead of queueing without bound: every parked
/// request holds its body and its encoding until it is served, so growth
/// without a bound only ends at the memory limit.
pub(crate) fn default_max_concurrent_requests() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    (cores * DEFAULT_INFLIGHT_PER_CORE).max(DEFAULT_INFLIGHT_FLOOR)
}

pub struct AppContextBuilder {
    client: Option<Client>,
    router_config: Option<RouterConfig>,
    rate_limiter: Option<Arc<TokenBucket>>,
    rate_limit_manager: Option<Arc<RateLimitManager>>,
    tokenizer_registry: Option<Arc<TokenizerRegistry>>,
    reasoning_parser_factory: Option<ReasoningParserFactory>,
    tool_parser_factory: Option<ToolParserFactory>,
    worker_registry: Option<Arc<WorkerRegistry>>,
    policy_registry: Option<Arc<PolicyRegistry>>,
    gateway: Option<Arc<Gateway>>,
    response_storage: Option<Arc<dyn ResponseStorage>>,
    conversation_storage: Option<Arc<dyn ConversationStorage>>,
    conversation_item_storage: Option<Arc<dyn ConversationItemStorage>>,
    worker_monitor: Option<Arc<WorkerMonitor>>,
    worker_job_queue: Option<Arc<OnceLock<Arc<JobQueue>>>>,
    workflow_engines: Option<Arc<OnceLock<WorkflowEngines>>>,
    mcp_orchestrator: Option<Arc<OnceLock<Arc<McpOrchestrator>>>>,
    mcp_format_registry: Option<FormatRegistry>,
    wasm_manager: Option<Arc<WasmModuleManager>>,
    kv_event_monitor: Option<Arc<KvEventMonitor>>,
    webrtc_bind_addr: Option<std::net::IpAddr>,
    webrtc_stun_server: Option<String>,
}

impl AppContext {
    pub fn builder() -> AppContextBuilder {
        AppContextBuilder::new()
    }

    /// Create AppContext from config with all components initialized
    /// This is the main entry point that replaces ~194 lines of initialization in server.rs
    pub fn from_config(
        router_config: RouterConfig,
        request_timeout_secs: u64,
        webrtc_bind_addr: Option<std::net::IpAddr>,
        webrtc_stun_server: Option<String>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self, String>> + Send>> {
        Box::pin(async move {
            Box::pin(AppContextBuilder::from_config(
                router_config,
                request_timeout_secs,
                webrtc_bind_addr,
                webrtc_stun_server,
            ))
            .await?
            .build()
            .map_err(|e| e.to_string())
        })
    }
}

/// Whether a worker URL given in the configuration needs TLS. Discovered
/// workers are not known yet when the shared client is built.
fn has_https_worker(config: &RouterConfig) -> bool {
    use crate::config::RoutingMode;

    let https = |url: &str| url.starts_with("https://");
    match &config.mode {
        RoutingMode::Regular { worker_urls }
        | RoutingMode::OpenAI { worker_urls }
        | RoutingMode::Anthropic { worker_urls }
        | RoutingMode::Gemini { worker_urls } => worker_urls.iter().any(|url| https(url)),
        RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            ..
        } => {
            prefill_urls.iter().any(|(url, _)| https(url))
                || decode_urls.iter().any(|url| https(url))
        }
        RoutingMode::EncodePrefillDecode {
            encode_urls,
            prefill_urls,
            decode_urls,
            ..
        } => {
            encode_urls
                .iter()
                .chain(prefill_urls)
                .any(|(url, _)| https(url))
                || decode_urls.iter().any(|url| https(url))
        }
    }
}

impl AppContextBuilder {
    pub fn new() -> Self {
        Self {
            client: None,
            router_config: None,
            rate_limiter: None,
            rate_limit_manager: None,
            tokenizer_registry: None,
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            worker_registry: None,
            policy_registry: None,
            gateway: None,
            response_storage: None,
            conversation_storage: None,
            conversation_item_storage: None,
            worker_monitor: None,
            worker_job_queue: None,
            workflow_engines: None,
            mcp_orchestrator: None,
            mcp_format_registry: None,
            wasm_manager: None,
            kv_event_monitor: None,
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        }
    }

    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    pub fn router_config(mut self, router_config: RouterConfig) -> Self {
        self.router_config = Some(router_config);
        self
    }

    pub fn rate_limiter(mut self, rate_limiter: Option<Arc<TokenBucket>>) -> Self {
        self.rate_limiter = rate_limiter;
        self
    }

    /// Set an already-built tenant rate limiter directly, bypassing
    /// `maybe_rate_limit_manager`'s config-loading. For callers (test
    /// harnesses) that build `AppContext` piecemeal rather than through
    /// `from_config` and so can't reach that private, config-driven setter.
    pub fn rate_limit_manager(mut self, rate_limit_manager: Option<Arc<RateLimitManager>>) -> Self {
        self.rate_limit_manager = rate_limit_manager;
        self
    }

    /// Build the tenant rate limiter from config. `Ok(None)` (feature
    /// disabled) is a valid, non-fatal outcome. `Err` (enabled but the
    /// policy YAML failed to load/parse/validate) fails startup — an
    /// operator who explicitly turned rate limiting on must not get a
    /// gateway that silently runs unlimited.
    fn maybe_rate_limit_manager(mut self, config: &RouterConfig) -> Result<Self, String> {
        self.rate_limit_manager = RateLimitManager::from_config(config)?;
        Ok(self)
    }

    pub fn tokenizer_registry(mut self, tokenizer_registry: Arc<TokenizerRegistry>) -> Self {
        self.tokenizer_registry = Some(tokenizer_registry);
        self
    }

    pub fn reasoning_parser_factory(
        mut self,
        reasoning_parser_factory: Option<ReasoningParserFactory>,
    ) -> Self {
        self.reasoning_parser_factory = reasoning_parser_factory;
        self
    }

    pub fn tool_parser_factory(mut self, tool_parser_factory: Option<ToolParserFactory>) -> Self {
        self.tool_parser_factory = tool_parser_factory;
        self
    }

    pub fn worker_registry(mut self, worker_registry: Arc<WorkerRegistry>) -> Self {
        self.worker_registry = Some(worker_registry);
        self
    }

    pub fn policy_registry(mut self, policy_registry: Arc<PolicyRegistry>) -> Self {
        self.policy_registry = Some(policy_registry);
        self
    }

    pub fn gateway(mut self, gateway: Option<Arc<Gateway>>) -> Self {
        self.gateway = gateway;
        self
    }

    pub fn response_storage(mut self, response_storage: Arc<dyn ResponseStorage>) -> Self {
        self.response_storage = Some(response_storage);
        self
    }

    pub fn conversation_storage(
        mut self,
        conversation_storage: Arc<dyn ConversationStorage>,
    ) -> Self {
        self.conversation_storage = Some(conversation_storage);
        self
    }

    pub fn conversation_item_storage(
        mut self,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
    ) -> Self {
        self.conversation_item_storage = Some(conversation_item_storage);
        self
    }

    pub fn worker_monitor(mut self, worker_monitor: Option<Arc<WorkerMonitor>>) -> Self {
        self.worker_monitor = worker_monitor;
        self
    }

    pub fn worker_job_queue(mut self, worker_job_queue: Arc<OnceLock<Arc<JobQueue>>>) -> Self {
        self.worker_job_queue = Some(worker_job_queue);
        self
    }

    pub fn workflow_engines(mut self, workflow_engines: Arc<OnceLock<WorkflowEngines>>) -> Self {
        self.workflow_engines = Some(workflow_engines);
        self
    }

    pub fn mcp_orchestrator(
        mut self,
        mcp_orchestrator: Arc<OnceLock<Arc<McpOrchestrator>>>,
    ) -> Self {
        self.mcp_orchestrator = Some(mcp_orchestrator);
        self
    }

    pub fn mcp_format_registry(mut self, registry: FormatRegistry) -> Self {
        self.mcp_format_registry = Some(registry);
        self
    }

    pub fn wasm_manager(mut self, wasm_manager: Option<Arc<WasmModuleManager>>) -> Self {
        self.wasm_manager = wasm_manager;
        self
    }

    pub fn kv_event_monitor(mut self, kv_event_monitor: Option<Arc<KvEventMonitor>>) -> Self {
        self.kv_event_monitor = kv_event_monitor;
        self
    }

    pub fn webrtc_bind_addr(mut self, addr: Option<std::net::IpAddr>) -> Self {
        self.webrtc_bind_addr = addr;
        self
    }

    pub fn webrtc_stun_server(mut self, server: Option<String>) -> Self {
        self.webrtc_stun_server = server;
        self
    }

    pub fn build(self) -> Result<AppContext, AppContextBuildError> {
        let router_config = self
            .router_config
            .ok_or(AppContextBuildError::MissingField("router_config"))?;
        let configured_reasoning_parser = router_config.reasoning_parser.clone();
        let configured_tool_parser = router_config.tool_call_parser.clone();

        // Validate configured parser names against their registries at startup
        if let (Some(name), Some(factory)) =
            (&configured_reasoning_parser, &self.reasoning_parser_factory)
        {
            if !factory.registry().has_parser(name) {
                tracing::error!(
                    parser = %name,
                    available = %factory.list_parsers().join(", "),
                    "Unknown reasoning parser"
                );
                return Err(AppContextBuildError::InvalidConfig(format!(
                    "unknown reasoning parser '{name}'"
                )));
            }
        }
        if let (Some(name), Some(factory)) = (&configured_tool_parser, &self.tool_parser_factory) {
            if !factory.has_parser(name) {
                tracing::error!(
                    parser = %name,
                    available = %factory.list_parsers().join(", "),
                    "Unknown tool-call parser"
                );
                return Err(AppContextBuildError::InvalidConfig(format!(
                    "unknown tool-call parser '{name}'"
                )));
            }
        }

        let worker_registry = self
            .worker_registry
            .ok_or(AppContextBuildError::MissingField("worker_registry"))?;
        let worker_job_queue = self
            .worker_job_queue
            .ok_or(AppContextBuildError::MissingField("worker_job_queue"))?;

        let prefill_admission =
            usize::try_from(router_config.prefill_max_inflight_requests_per_worker)
                .ok()
                .filter(|max| *max > 0)
                .map(|max| {
                    Arc::new(PrefillAdmission::new(
                        max,
                        router_config.effective_prefill_queue_size(),
                        Duration::from_secs(router_config.effective_prefill_queue_timeout_secs()),
                    ))
                });
        if let Some(admission) = prefill_admission
            .as_ref()
            .filter(|_| router_config.effective_prefill_queue_size() > 0)
        {
            start_prefill_admission_notifier(&worker_registry, admission)?;
        }

        // Create WorkerService from the already-built components
        let worker_service = Arc::new(WorkerService::new(
            worker_registry.clone(),
            worker_job_queue.clone(),
            router_config.clone(),
        ));

        let worker_client_cache = Arc::new(WorkerHttpClientCache::new(&router_config));
        let gateway_auth = AuthConfig::with_tenant_keys(
            router_config.api_key.clone(),
            &router_config.tenant_api_keys,
        );

        let rl = crate::rl_adapter::build_rl_state(&worker_registry, &router_config);

        Ok(AppContext {
            gateway_auth,
            client: self
                .client
                .ok_or(AppContextBuildError::MissingField("client"))?,
            router_config,
            rate_limiter: self.rate_limiter,
            rate_limit_manager: self.rate_limit_manager,
            tokenizer_registry: self
                .tokenizer_registry
                .ok_or(AppContextBuildError::MissingField("tokenizer_registry"))?,
            multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
            reasoning_parser_factory: self.reasoning_parser_factory,
            tool_parser_factory: self.tool_parser_factory,
            worker_registry,
            prefill_admission,
            policy_registry: self
                .policy_registry
                .ok_or(AppContextBuildError::MissingField("policy_registry"))?,
            gateway: self.gateway,
            response_storage: self
                .response_storage
                .ok_or(AppContextBuildError::MissingField("response_storage"))?,
            conversation_storage: self
                .conversation_storage
                .ok_or(AppContextBuildError::MissingField("conversation_storage"))?,
            conversation_item_storage: self.conversation_item_storage.ok_or(
                AppContextBuildError::MissingField("conversation_item_storage"),
            )?,
            worker_monitor: self.worker_monitor,
            configured_reasoning_parser,
            configured_tool_parser,
            worker_job_queue,
            workflow_engines: self
                .workflow_engines
                .ok_or(AppContextBuildError::MissingField("workflow_engines"))?,
            mcp_orchestrator: self
                .mcp_orchestrator
                .ok_or(AppContextBuildError::MissingField("mcp_orchestrator"))?,
            mcp_format_registry: self.mcp_format_registry.unwrap_or_default(),
            wasm_manager: self.wasm_manager,
            worker_service,
            worker_client_cache,
            inflight_tracker: InFlightRequestTracker::new(),
            kv_event_monitor: self.kv_event_monitor,
            rl,
            realtime_registry: Arc::new(RealtimeRegistry::new()),
            webrtc_bind_addr: self.webrtc_bind_addr,
            webrtc_stun_server: self.webrtc_stun_server,
        })
    }

    /// Initialize AppContext from config - creates ALL components
    /// This replaces ~194 lines of initialization logic from server.rs
    pub async fn from_config(
        router_config: RouterConfig,
        request_timeout_secs: u64,
        webrtc_bind_addr: Option<std::net::IpAddr>,
        webrtc_stun_server: Option<String>,
    ) -> Result<Self, String> {
        // A served model aliased under a vendor name keeps that vendor's
        // contract profile after the alias is resolved into the served name.
        ProviderProfile::register_model_aliases(
            router_config
                .model_aliases
                .iter()
                .map(|(alias, canonical)| (alias.as_str(), canonical.as_str())),
        );
        Ok(Self::new()
            .with_client(&router_config, request_timeout_secs)?
            .maybe_rate_limiter(&router_config)
            .maybe_rate_limit_manager(&router_config)?
            .with_tokenizer_registry()
            .with_reasoning_parser_factory()
            .with_tool_parser_factory()
            .with_worker_registry()
            .with_policy_registry(&router_config)
            .with_storage(&router_config)
            .await?
            .with_worker_monitor(&router_config)?
            .with_worker_job_queue()
            .with_workflow_engines()
            .with_mcp_orchestrator(&router_config)
            .await?
            .with_wasm_manager(&router_config)
            .with_kv_event_monitor(&router_config)
            .webrtc_bind_addr(
                webrtc_bind_addr.or_else(|| Some(default_bind_addr(&router_config.host))),
            )
            .webrtc_stun_server(
                webrtc_stun_server.or_else(|| Some("stun.l.google.com:19302".to_string())),
            )
            .router_config(router_config))
    }

    /// Create the shared HTTP client for upstream calls not addressed to a
    /// registered worker (external providers, IGW model discovery, worker
    /// classification). Worker-directed traffic uses each worker's own client
    /// from [`WorkerHttpClientCache`].
    ///
    /// Uses the rustls TLS backend when TLS/mTLS is configured (client cert or
    /// CA certs provided) for PKCS#8 key support; plain HTTP skips TLS setup.
    /// A host without a native CA root store is tolerated unless TLS is
    /// needed (see [`build_client`]).
    fn with_client(mut self, config: &RouterConfig, timeout_secs: u64) -> Result<Self, String> {
        let has_tls_config = config.client_identity.is_some() || !config.ca_certificates.is_empty();
        // With a client identity, a CA bundle or an HTTPS worker URL, TLS is
        // needed: a missing root store is then an error, not worked around.
        let tls_required = has_tls_config || has_https_worker(config);

        // Idle pooled connections must expire before the backend server's
        // keep-alive closes them (the engines' default: 5s), or checkout races
        // the server's FIN and non-idempotent sends fail.
        let pool_idle_timeout = match config.upstream_pool_idle_timeout_secs {
            0 => None,
            secs => Some(Duration::from_secs(secs)),
        };
        // mTLS client identity and CA certificates for verifying worker TLS
        // (both loaded during config creation), parsed once: the builder
        // closure runs again if the first build fails for want of a root store.
        let identity = config
            .client_identity
            .as_deref()
            .map(reqwest::Identity::from_pem)
            .transpose()
            .map_err(|e| format!("Failed to create client identity: {e}"))?;
        let ca_certificates = config
            .ca_certificates
            .iter()
            .map(|pem| reqwest::Certificate::from_pem(pem))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to add CA certificate: {e}"))?;
        if has_tls_config {
            debug!("Using rustls TLS backend for TLS/mTLS connections");
        }
        if identity.is_some() {
            debug!("mTLS client authentication enabled");
        }
        if !ca_certificates.is_empty() {
            debug!(
                "Added {} CA certificate(s) for worker verification",
                ca_certificates.len()
            );
        }

        let make_builder = || {
            let mut client_builder = Client::builder()
                .pool_idle_timeout(pool_idle_timeout)
                .pool_max_idle_per_host(500)
                .timeout(Duration::from_secs(timeout_secs))
                .connect_timeout(Duration::from_secs(10))
                .tcp_nodelay(true)
                .tcp_keepalive(Some(Duration::from_secs(30)));
            // Force rustls backend when TLS is configured
            if has_tls_config {
                client_builder = client_builder.use_rustls_tls();
            }
            if let Some(identity) = &identity {
                client_builder = client_builder.identity(identity.clone());
            }
            for cert in &ca_certificates {
                client_builder = client_builder.add_root_certificate(cert.clone());
            }
            client_builder
        };
        let client = build_client(make_builder, tls_required, "HTTP client")?;

        self.client = Some(client);
        Ok(self)
    }

    /// Create the admission bucket from config: an explicit
    /// `max_concurrent_requests` is the cap, 0 disables admission control,
    /// and the unset value (-1) takes the host-derived default so an
    /// overloaded router sheds instead of queueing without bound.
    fn maybe_rate_limiter(mut self, config: &RouterConfig) -> Self {
        let capacity = match config.max_concurrent_requests {
            0 => {
                info!(
                    "Admission bound disabled (max_concurrent_requests = 0): in-flight requests are unbounded"
                );
                self.rate_limiter = None;
                return self;
            }
            n if n < 0 => {
                let bound = default_max_concurrent_requests();
                info!(
                    "Admission bound: at most {bound} in-flight requests (default: {DEFAULT_INFLIGHT_PER_CORE} per available core, at least {DEFAULT_INFLIGHT_FLOOR}; --max-concurrent-requests overrides it, 0 disables it)"
                );
                bound
            }
            n => {
                let bound = n as usize;
                info!(
                    "Admission bound: at most {bound} in-flight requests (max_concurrent_requests)"
                );
                bound
            }
        };
        // No refill unless explicitly configured: the cap bounds
        // standing concurrency, not admission rate.
        let rate_limit_tokens = config
            .rate_limit_tokens_per_second
            .filter(|&t| t > 0)
            .unwrap_or(0);
        self.rate_limiter = Some(Arc::new(TokenBucket::new(
            capacity,
            rate_limit_tokens as usize,
        )));
        self
    }

    /// Create reasoning parser factory for gRPC mode or IGW mode
    fn with_reasoning_parser_factory(mut self) -> Self {
        // Initialize reasoning parser factory
        self.reasoning_parser_factory = Some(ReasoningParserFactory::new());
        self
    }

    /// Create tool parser factory for gRPC mode or IGW mode
    fn with_tool_parser_factory(mut self) -> Self {
        // Initialize tool parser factory
        self.tool_parser_factory = Some(ToolParserFactory::new());
        self
    }

    /// Create empty tokenizer registry
    ///
    /// Tokenizers are loaded via the tokenizer_registration workflow, which is triggered:
    /// - At startup (if --tokenizer-path or --model-path is provided)
    /// - When workers connect (registers under model_id)
    /// - Via POST /v1/tokenizers API (registers under user-specified name)
    ///
    /// This unified approach ensures consistent behavior (caching, validation) across all paths.
    fn with_tokenizer_registry(mut self) -> Self {
        self.tokenizer_registry = Some(Arc::new(TokenizerRegistry::new()));
        self
    }

    /// Create worker registry
    fn with_worker_registry(mut self) -> Self {
        self.worker_registry = Some(Arc::new(WorkerRegistry::new()));
        self
    }

    /// Create policy registry
    fn with_policy_registry(mut self, config: &RouterConfig) -> Self {
        self.policy_registry = Some(Arc::new(
            PolicyRegistry::with_override(
                config.policy.clone(),
                config.routing_key_override.clone(),
            )
            .with_pd_pairing_mode(config.pd_pairing_mode),
        ));
        self
    }

    /// Create all storage backends using the factory function
    async fn with_storage(mut self, config: &RouterConfig) -> Result<Self, String> {
        let hook: Option<Arc<dyn smg_data_connector::hooks::StorageHook>> =
            match &config.storage_hook_wasm_path {
                Some(path) => {
                    let bytes = tokio::fs::read(path)
                        .await
                        .map_err(|e| format!("failed to read WASM storage hook at {path}: {e}"))?;
                    let wasm_hook =
                        tokio::task::spawn_blocking(move || smg_wasm::WasmStorageHook::new(&bytes))
                            .await
                            .map_err(|e| format!("WASM compilation task panicked: {e}"))?
                            .map_err(|e| {
                                format!("failed to compile WASM storage hook at {path}: {e}")
                            })?;
                    debug!("loaded WASM storage hook from {path}");
                    Some(Arc::new(wasm_hook))
                }
                None => None,
            };

        let storage_config = StorageFactoryConfig {
            backend: &config.history_backend,
            oracle: config.oracle.as_ref(),
            postgres: config.postgres.as_ref(),
            redis: config.redis.as_ref(),
            hook,
        };
        let bundle = create_storage(storage_config).await?;

        self.response_storage = Some(bundle.response_storage);
        self.conversation_storage = Some(bundle.conversation_storage);
        self.conversation_item_storage = Some(bundle.conversation_item_storage);

        Ok(self)
    }

    /// Create load monitor
    fn with_worker_monitor(mut self, config: &RouterConfig) -> Result<Self, String> {
        let policy_registry = self
            .policy_registry
            .as_ref()
            .ok_or_else(|| "policy_registry must be set before load monitor".to_string())?
            .clone();
        let monitor = Arc::new(WorkerMonitor::new(
            self.worker_registry
                .as_ref()
                .ok_or_else(|| "worker_registry must be set before load monitor".to_string())?
                .clone(),
            Arc::clone(&policy_registry),
            config.load_monitor_interval_secs,
            config.engine_metrics,
            config.disable_load_monitoring,
        ));
        // The overload shed advertises the poll interval as Retry-After — the
        // veto cannot clear between polls.
        overload::set_shed_retry_after_secs(config.load_monitor_interval_secs);
        if let Some(registry) = self.worker_registry.as_ref() {
            registry.set_overload_shed(config.worker_overload_shed);
        }
        // Progress-based liveness thresholds (see `worker::liveness`).
        liveness::configure(
            Duration::from_secs(config.worker_stall_secs),
            Duration::from_secs(config.worker_wedge_secs),
            Duration::from_secs(config.worker_stale_secs),
        );
        liveness::configure_warmup(liveness::Warmup {
            secs: Duration::from_secs(config.worker_warmup_secs),
            share: config.worker_warmup_share,
            blocks: config.worker_warmup_blocks,
            thin_ratio: config.worker_warmup_thin_ratio,
            divert_every: config.worker_warmup_divert_every,
        });
        // PD dispatch waits here, not in the decode engine's queue, when the
        // pair's running window is full.
        pd_admission::set_pd_admission_wait_secs(config.pd_admission_wait_secs);
        // Wire the backend load-snapshot feed into every policy that consumes
        // it; the monitor polls every group by default, conditionally under
        // `--disable-load-monitoring`.
        policy_registry.set_load_receiver(Some(monitor.subscribe()));
        self.worker_monitor = Some(monitor);
        Ok(self)
    }

    /// Create worker job queue OnceLock container
    fn with_worker_job_queue(mut self) -> Self {
        self.worker_job_queue = Some(Arc::new(OnceLock::new()));
        self
    }

    /// Create workflow engines OnceLock container
    fn with_workflow_engines(mut self) -> Self {
        self.workflow_engines = Some(Arc::new(OnceLock::new()));
        self
    }

    /// Create and initialize the MCP orchestrator from the operator's MCP
    /// config (`--mcp-config-path`), minus its server list.
    ///
    /// The servers are registered later via the InitializeMcpServers job, so
    /// startup never waits on one. The pool limits, the global proxy, the
    /// inventory settings and the approval policy have to be in place before
    /// that: the orchestrator resolves each server's proxy against its global
    /// proxy and builds its policy engine once, at construction.
    async fn with_mcp_orchestrator(mut self, router_config: &RouterConfig) -> Result<Self, String> {
        // Create OnceLock container
        let mcp_orchestrator_lock = Arc::new(OnceLock::new());

        let config = mcp_bootstrap_config(router_config.mcp_config.as_ref());
        debug!(
            max_connections = config.pool.max_connections,
            proxy = config.proxy.is_some(),
            "Initializing MCP orchestrator; config-file servers register through the job queue"
        );

        let orchestrator = McpOrchestrator::new(config)
            .await
            .map_err(|e| format!("Failed to initialize MCP orchestrator: {e}"))?;

        // Store the initialized orchestrator in the OnceLock
        mcp_orchestrator_lock
            .set(Arc::new(orchestrator))
            .map_err(|_| "Failed to set MCP orchestrator in OnceLock".to_string())?;

        self.mcp_orchestrator = Some(mcp_orchestrator_lock);
        self.mcp_format_registry = Some(FormatRegistry::new());
        Ok(self)
    }

    /// Create KV event monitor for event-driven cache-aware routing.
    ///
    /// The monitor is created when ANY serving policy is cache_aware — the
    /// global default or a PD/EPD role override (a non-cache-aware global with
    /// a cache-aware decode policy still needs event-driven indexers) —
    /// regardless of connection mode. The monitor itself is cheap (empty
    /// DashMaps) and stays dormant until workers are added. The
    /// UpdatePoliciesStep gates subscriptions on `cache_aware && gRPC`, so
    /// HTTP workers are never subscribed.
    fn with_kv_event_monitor(mut self, config: &RouterConfig) -> Self {
        use crate::config::types::{PolicyConfig, RoutingMode};

        let role_is_cache_aware =
            |policy: &Option<PolicyConfig>| matches!(policy, Some(PolicyConfig::CacheAware { .. }));
        let is_cache_aware = matches!(config.policy, PolicyConfig::CacheAware { .. })
            || match &config.mode {
                RoutingMode::PrefillDecode {
                    prefill_policy,
                    decode_policy,
                    ..
                } => role_is_cache_aware(prefill_policy) || role_is_cache_aware(decode_policy),
                RoutingMode::EncodePrefillDecode {
                    encode_policy,
                    prefill_policy,
                    decode_policy,
                    ..
                } => {
                    role_is_cache_aware(encode_policy)
                        || role_is_cache_aware(prefill_policy)
                        || role_is_cache_aware(decode_policy)
                }
                _ => false,
            };

        if is_cache_aware {
            let monitor = Arc::new(KvEventMonitor::with_kind(config.kv_index, None));
            debug!(
                kv_index = config.kv_index.as_str(),
                "Created KV event monitor for event-driven cache-aware routing"
            );
            // The load records on the event streams are polls of the worker.
            if let Some(worker_monitor) = &self.worker_monitor {
                monitor.set_load_sink(worker_monitor);
            }
            if KvIndexKind::deprecated_alias_used() {
                warn!("--kv-index run is the deprecated spelling of --kv-index chain");
            }

            // Optional indexer bounding: prune entries by last-touch TTL and/or
            // capacity ceiling. Both default off (unbounded, prior behavior).
            monitor.start_prune_task(
                config.kv_indexer_ttl_secs.unwrap_or(0),
                config.kv_indexer_max_entries.unwrap_or(0),
            );
            monitor.start_stats_task();

            // Inject monitor into PolicyRegistry — propagates to default_policy
            // and any other existing cache-aware policies.
            if let Some(ref registry) = self.policy_registry {
                registry.set_kv_event_monitor(Some(Arc::clone(&monitor)));
            }

            self.kv_event_monitor = Some(monitor);
        }

        self
    }

    /// Create wasm manager if enabled in config
    fn with_wasm_manager(mut self, config: &RouterConfig) -> Self {
        self.wasm_manager = if config.enable_wasm {
            Some(Arc::new(WasmModuleManager::new(
                WasmRuntimeConfig::default(),
            )))
        } else {
            None
        };
        self
    }
}

impl Default for AppContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The orchestrator's startup configuration: the operator's MCP file with
/// its server list removed (the `InitializeMcpServers` job registers those
/// once the gateway is up), and the global proxy taken from the environment
/// (`MCP_HTTP_PROXY`, `MCP_HTTPS_PROXY`, `MCP_NO_PROXY`, or their unprefixed
/// forms) when the file sets none.
fn mcp_bootstrap_config(file: Option<&smg_mcp::McpConfig>) -> smg_mcp::McpConfig {
    let mut config = file.cloned().unwrap_or_default();
    config.servers.clear();
    config.with_env_proxy()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use openai_protocol::worker::{HealthCheckConfig, WorkerStatus};

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        worker::{BasicWorkerBuilder, PrefillAdmissionAttempt, Worker, WorkerType},
    };

    /// Loopback echo server; axum::serve accepts HTTP/1.1 and prior-knowledge
    /// h2c on the same listener, mirroring a dual-protocol engine.
    async fn spawn_echo_server() -> String {
        let app = axum::Router::new().route("/probe", axum::routing::get(|| async { "ok" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind echo server");
        let addr = listener.local_addr().expect("echo server address");
        #[expect(
            clippy::disallowed_methods,
            reason = "test server lives for the duration of the test process"
        )]
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("echo serve");
        });
        format!("http://{addr}/probe")
    }

    /// Child-process body of the root-store test below (see
    /// `worker::http_client::no_root_store`); a no-op unless the case
    /// variable is set.
    #[test]
    fn child_with_client_without_native_roots() {
        use crate::worker::http_client::no_root_store;

        if std::env::var(no_root_store::CASE).is_err() {
            return;
        }
        AppContextBuilder::new()
            .with_client(&RouterConfig::default(), 5)
            .expect("a gateway without TLS configuration starts without a root store");
    }

    #[test]
    fn the_gateway_client_builds_without_a_native_root_store() {
        use crate::worker::http_client::no_root_store;

        no_root_store::run(
            &no_root_store::test_name(module_path!(), "child_with_client_without_native_roots"),
            "plaintext",
        );
    }

    #[test]
    fn https_worker_urls_make_tls_required() {
        use crate::config::RoutingMode;

        let with_worker = |url: &str| RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![url.to_string()],
            },
            ..RouterConfig::default()
        };
        assert!(!has_https_worker(&with_worker("http://worker:8000")));
        assert!(has_https_worker(&with_worker("https://worker:8443")));
    }

    fn built_client(upstream_http2: bool) -> Client {
        let config = RouterConfig {
            upstream_http2,
            ..RouterConfig::default()
        };
        AppContextBuilder::new()
            .with_client(&config, 5)
            .expect("client builds")
            .client
            .expect("client set")
    }

    /// `--upstream-http2` is a worker-client concern; the shared client keeps
    /// negotiating normally (HTTP/1.1 on cleartext, ALPN on TLS).
    #[tokio::test]
    async fn shared_client_ignores_upstream_http2() {
        let url = spawn_echo_server().await;
        let resp = built_client(true)
            .get(&url)
            .send()
            .await
            .expect("h1 request");
        assert_eq!(resp.version(), http::Version::HTTP_11);
        assert_eq!(resp.text().await.expect("body"), "ok");
    }

    #[tokio::test]
    async fn default_client_stays_http1() {
        let url = spawn_echo_server().await;
        let resp = built_client(false)
            .get(&url)
            .send()
            .await
            .expect("h1 request");
        assert_eq!(resp.version(), http::Version::HTTP_11);
        assert_eq!(resp.text().await.expect("body"), "ok");
    }

    #[tokio::test]
    async fn unset_rate_limit_defaults_to_no_refill() {
        let config = RouterConfig {
            max_concurrent_requests: 10,
            rate_limit_tokens_per_second: None,
            ..RouterConfig::default()
        };
        let bucket = AppContextBuilder::new()
            .maybe_rate_limiter(&config)
            .rate_limiter
            .expect("rate limiter should be enabled");

        assert!(bucket.try_acquire(10.0).is_ok());
        // The old fallback refilled at max_concurrent_requests per second,
        // which would restore a token during this wait.
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(bucket.try_acquire(1.0).is_err());

        bucket.return_tokens_sync(1.0);
        assert!(bucket.try_acquire(1.0).is_ok());
    }

    /// With no explicit cap the admission bucket still exists, sized by the
    /// host-derived default: the request past the bound is rejected instead
    /// of queued without limit, and a completion frees a slot again.
    #[test]
    fn unset_concurrency_cap_gets_a_host_derived_bound() {
        let bucket = AppContextBuilder::new()
            .maybe_rate_limiter(&RouterConfig::default())
            .rate_limiter
            .expect("the unset cap must still bound in-flight requests");
        let bound = default_max_concurrent_requests();
        assert!(bound >= DEFAULT_INFLIGHT_FLOOR);

        for admitted in 0..bound {
            assert!(
                bucket.try_acquire(1.0).is_ok(),
                "request {admitted} is within the bound of {bound}"
            );
        }
        assert!(
            bucket.try_acquire(1.0).is_err(),
            "the request past the bound must be rejected, not admitted"
        );

        bucket.return_tokens_sync(1.0);
        assert!(bucket.try_acquire(1.0).is_ok());
    }

    #[test]
    fn zero_concurrency_cap_disables_admission_control() {
        let config = RouterConfig {
            max_concurrent_requests: 0,
            ..RouterConfig::default()
        };
        assert!(AppContextBuilder::new()
            .maybe_rate_limiter(&config)
            .rate_limiter
            .is_none());
    }

    #[tokio::test]
    async fn explicit_zero_rate_limit_disables_refill() {
        let config = RouterConfig {
            max_concurrent_requests: 10,
            rate_limit_tokens_per_second: Some(0),
            ..RouterConfig::default()
        };
        let bucket = AppContextBuilder::new()
            .maybe_rate_limiter(&config)
            .rate_limiter
            .expect("rate limiter should be enabled");

        assert!(bucket.try_acquire(10.0).is_ok());
        // The previous fallback used max_concurrent_requests as the refill
        // rate, which would add more than one token during this wait.
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(bucket.try_acquire(1.0).is_err());

        bucket.return_tokens_sync(1.0);
        assert!(bucket.try_acquire(1.0).is_ok());
    }

    fn config_with_policy(policy: PolicyConfig) -> RouterConfig {
        RouterConfig {
            policy,
            ..Default::default()
        }
    }

    /// `with_kv_event_monitor` only creates a monitor for the cache-aware policy.
    /// This run of the builder needs no storage or network, so it exercises the
    /// real gating path rather than the predicate in isolation.
    fn kv_monitor_created_for(policy: PolicyConfig) -> bool {
        let config = config_with_policy(policy);
        AppContextBuilder::new()
            .with_policy_registry(&config)
            .with_kv_event_monitor(&config)
            .kv_event_monitor
            .is_some()
    }

    /// The load-snapshot feed must be wired whenever the worker monitor is
    /// built — without a KV-event monitor in the chain — so HTTP-only
    /// cache-aware deployments get waiting-prefill and KV-usage data.
    #[test]
    fn worker_monitor_wires_load_receiver_into_policies() {
        use crate::policies::CacheAwarePolicy;

        let config = config_with_policy(PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 1000,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 1.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
            selection_policy: None,
            selection_accounting_ttl_ms: 0,
        });
        let builder = AppContextBuilder::new()
            .with_client(&config, 5)
            .expect("client builds")
            .with_worker_registry()
            .with_policy_registry(&config)
            .with_worker_monitor(&config)
            .expect("worker monitor builds");

        let policy = builder
            .policy_registry
            .as_ref()
            .expect("policy registry set")
            .get_default_policy();
        let cache_aware = policy
            .as_any()
            .downcast_ref::<CacheAwarePolicy>()
            .expect("default policy is cache-aware");
        assert!(cache_aware.has_load_receiver_for_test());
    }

    /// The #1794-relevant guarantee: passthrough never starts the KV-event
    /// monitor, so single-backend gateways skip the `SubscribeKvEvents` overhead.
    #[test]
    fn test_passthrough_does_not_create_kv_event_monitor() {
        assert!(!kv_monitor_created_for(PolicyConfig::Passthrough));
        // Other non-cache-aware policies are likewise skipped.
        assert!(!kv_monitor_created_for(PolicyConfig::RoundRobin));
        // Control: cache-aware still creates the monitor.
        assert!(kv_monitor_created_for(PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 1000,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
            selection_policy: None,
            selection_accounting_ttl_ms: 0,
        }));
    }

    /// A cache-aware PD/EPD role policy needs the monitor even when the
    /// global policy is not cache-aware.
    #[test]
    fn test_cache_aware_role_policy_creates_kv_event_monitor() {
        use crate::config::types::RoutingMode;

        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 30,
            max_tree_size: 1000,
            block_size: 16,
            balance_token_usage_threshold: 1.0,
            overload_token_usage_threshold: 1.0,
            overlap_decay: 0.0,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
            selection_policy: None,
            selection_accounting_ttl_ms: 0,
        };

        let mut config = config_with_policy(PolicyConfig::Random);
        config.mode = RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: Some(cache_aware.clone()),
        };
        let created = AppContextBuilder::new()
            .with_policy_registry(&config)
            .with_kv_event_monitor(&config)
            .kv_event_monitor
            .is_some();
        assert!(created, "cache-aware decode policy must create the monitor");

        // Non-cache-aware role policies still skip it.
        let mut config = config_with_policy(PolicyConfig::Random);
        config.mode = RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: Some(PolicyConfig::RoundRobin),
            decode_policy: None,
        };
        let created = AppContextBuilder::new()
            .with_policy_registry(&config)
            .with_kv_event_monitor(&config)
            .kv_event_monitor
            .is_some();
        assert!(!created);
    }

    #[test]
    fn maybe_rate_limit_manager_disabled_is_ok_none() {
        let config = RouterConfig::default();
        let result = AppContextBuilder::new().maybe_rate_limit_manager(&config);
        assert!(result.is_ok());
        assert!(result.unwrap().rate_limit_manager.is_none());
    }

    #[test]
    fn maybe_rate_limit_manager_enabled_with_missing_file_fails_startup() {
        let config = RouterConfig::builder()
            .tenant_rate_limit_enabled(true)
            .tenant_rate_limit_config(Some("/nonexistent/rate_limit.yaml".to_string()))
            .build_unchecked();
        assert!(AppContextBuilder::new()
            .maybe_rate_limit_manager(&config)
            .is_err());
    }

    #[test]
    fn maybe_rate_limit_manager_enabled_with_valid_policy_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rate_limit.yaml");
        std::fs::write(
            &path,
            "default_policy:\n  tokens_per_minute: 1000\n  requests_per_minute: 60\n",
        )
        .unwrap();
        let config = RouterConfig::builder()
            .tenant_rate_limit_enabled(true)
            .tenant_rate_limit_config(Some(path.to_str().unwrap().to_string()))
            .build_unchecked();
        let result = AppContextBuilder::new().maybe_rate_limit_manager(&config);
        assert!(result.is_ok());
        assert!(result.unwrap().rate_limit_manager.is_some());
    }

    #[test]
    fn mcp_bootstrap_keeps_everything_but_the_server_list() {
        let file: smg_mcp::McpConfig = serde_yaml::from_str(
            r#"
servers:
  - name: "docs"
    protocol: sse
    url: "https://mcp.example.com/sse"
pool:
  max_connections: 7
proxy:
  https: "http://proxy.example:3128"
policy:
  default: deny
"#,
        )
        .unwrap();
        assert_eq!(file.servers.len(), 1);

        let config = mcp_bootstrap_config(Some(&file));
        assert!(
            config.servers.is_empty(),
            "servers register through the job queue"
        );
        assert_eq!(config.pool.max_connections, 7);
        assert_eq!(
            config
                .proxy
                .as_ref()
                .and_then(|proxy| proxy.https.as_deref()),
            Some("http://proxy.example:3128")
        );
        assert!(matches!(
            config.policy.default,
            smg_mcp::PolicyDecisionConfig::Deny
        ));

        assert!(mcp_bootstrap_config(None).servers.is_empty());
    }

    #[test]
    fn mcp_bootstrap_reads_the_proxy_from_the_environment_when_the_file_has_none() {
        std::env::set_var("MCP_HTTPS_PROXY", "http://env-proxy.example:3128");
        let config = mcp_bootstrap_config(None);
        std::env::remove_var("MCP_HTTPS_PROXY");
        assert_eq!(
            config
                .proxy
                .as_ref()
                .and_then(|proxy| proxy.https.as_deref()),
            Some("http://env-proxy.example:3128")
        );
    }

    fn prefill_worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Prefill)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        )
    }

    #[tokio::test]
    async fn worker_status_change_wakes_prefill_admission() {
        let registry = Arc::new(WorkerRegistry::new());
        let admission = Arc::new(PrefillAdmission::new(1, 1, Duration::from_secs(1)));
        start_prefill_admission_notifier(&registry, &admission).unwrap();

        let first = prefill_worker("http://prefill-1:8000");
        registry.register(Arc::clone(&first)).unwrap();
        let occupied = admission
            .admit(None, {
                let first = Arc::clone(&first);
                move |capacity| capacity.select(Arc::clone(&first), ())
            })
            .await
            .unwrap();

        let attempts = Arc::new(AtomicUsize::new(0));
        #[expect(
            clippy::disallowed_methods,
            reason = "test waiter must run concurrently with worker status changes"
        )]
        let waiting = tokio::spawn({
            let admission = Arc::clone(&admission);
            let registry = Arc::clone(&registry);
            let attempts = Arc::clone(&attempts);
            async move {
                admission
                    .admit(None, |capacity| {
                        attempts.fetch_add(1, Ordering::Relaxed);
                        let workers = registry.get_by_type(WorkerType::Prefill);
                        if workers.is_empty() {
                            return PrefillAdmissionAttempt::Unavailable;
                        }
                        workers
                            .iter()
                            .find(|worker| worker.is_available() && capacity.has_capacity(worker))
                            .map_or(PrefillAdmissionAttempt::AtCapacity, |worker| {
                                capacity.select(Arc::clone(worker), Arc::clone(worker))
                            })
                    })
                    .await
            }
        });

        tokio::time::timeout(Duration::from_secs(1), async {
            while admission.queued_requests() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let second: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://prefill-2:8000")
                .worker_type(WorkerType::Prefill)
                .build(),
        );
        let second_id = registry.register(Arc::clone(&second)).unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while attempts.load(Ordering::Relaxed) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(admission.queued_requests(), 1);

        registry.transition_status(&second_id, WorkerStatus::Ready);
        let selected = tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .unwrap()
            .unwrap()
            .unwrap();

        assert_eq!(selected.selected.url(), second.url());
        drop(selected);
        drop(occupied);
    }
}
