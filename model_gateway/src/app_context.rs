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
use tracing::{debug, info};

use crate::{
    config::{ParserChoice, RouterConfig},
    middleware::{AuthConfig, TokenBucket},
    observability::inflight_tracker::InFlightRequestTracker,
    policies::PolicyRegistry,
    rate_limit::RateLimitManager,
    routers::{
        common::{
            openai_bridge::FormatRegistry, overload, pd_admission, realtime::RealtimeRegistry,
        },
        gateway::Gateway,
        grpc::{
            multimodal::MultimodalConfigRegistry,
            regular::symphony::{check_names, named_table},
            utils::{Names, ALIASES},
        },
    },
    wasm::{config::WasmRuntimeConfig, module_manager::WasmModuleManager},
    worker::{
        http_client::{build_client, connect_timeout_secs_below, DEFAULT_CONNECT_TIMEOUT_SECS},
        liveness, rl_adapter, KvEventMonitor, PrefillAdmission, WorkerHttpClientCache,
        WorkerMonitor, WorkerRegistry, WorkerService,
    },
    workflow::{JobQueue, TokenizerLoadFailures, WorkflowEngines},
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
    /// The served models whose tokenizer load was skipped or failed for good:
    /// written by the tokenizer registration (a worker's autoload, an
    /// operator's request), read where a request finds no tokenizer for its
    /// model.
    pub tokenizer_load_failures: Arc<TokenizerLoadFailures>,
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
fn default_max_concurrent_requests() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    (cores * DEFAULT_INFLIGHT_PER_CORE).max(DEFAULT_INFLIGHT_FLOOR)
}

/// The admission bound `max_concurrent_requests` resolves to: none when
/// admission control is off (0), the host-derived default when unset (-1),
/// the value otherwise. The one place the sentinels are read, for the
/// admission bucket and for the start-up checks that compare with it.
pub(crate) fn effective_max_concurrent_requests(configured: i32) -> Option<usize> {
    match configured {
        0 => None,
        n if n < 0 => Some(default_max_concurrent_requests()),
        n => usize::try_from(n).ok(),
    }
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
        // The configuration's liveness budgets (the stall, wedge and stale
        // thresholds and the warm-up slice, see `worker::liveness`) take
        // effect for every context built from it, the server's and a test's
        // alike, here and nowhere else. They are process-wide and the first
        // call wins, so the first context built in a process sets them.
        liveness::configure_from(&router_config);
        let configured_reasoning_parser = router_config.reasoning_parser.clone();
        let configured_tool_parser = router_config.tool_call_parser.clone();

        // The configured parser names are checked against the registries at startup: the
        // legacy parsers', and Symphony's tables under `--parser symphony`
        // (`regular::symphony::check_names`). A name whose factory is absent is not checked,
        // as before.
        let names = Names {
            tool: configured_tool_parser.clone(),
            reasoning: configured_reasoning_parser.clone(),
        };
        if let Err(why) = check_names(
            router_config.parser,
            &names,
            self.tool_parser_factory.as_ref(),
            self.reasoning_parser_factory.as_ref(),
        ) {
            let listed = |names: Option<Vec<String>>| names.unwrap_or_default().join(", ");
            let legacy_tool_parsers =
                listed(self.tool_parser_factory.as_ref().map(|f| f.list_parsers()));
            let legacy_reasoning_parsers = listed(
                self.reasoning_parser_factory
                    .as_ref()
                    .map(|f| f.list_parsers()),
            );
            let symphony_tables: Vec<&str> = symphony::formats::Table::ALL
                .iter()
                .map(|table| table.name())
                .collect();
            // The other names a table answers to: the legacy names that read a table's format,
            // and the names no other list has, a launcher's for a family whose table has no
            // older parser and an engine's exact name for a table.
            let symphony_aliases: Vec<&str> = ALIASES.iter().map(|(alias, _)| *alias).collect();
            tracing::error!(
                tool_call_parser = ?names.tool,
                reasoning_parser = ?names.reasoning,
                legacy_tool_parsers = %legacy_tool_parsers,
                legacy_reasoning_parsers = %legacy_reasoning_parsers,
                symphony_tables = %symphony_tables.join(", "),
                symphony_aliases = %symphony_aliases.join(", "),
                "Unknown parser name"
            );
            return Err(AppContextBuildError::InvalidConfig(why.into()));
        }
        if router_config.parser == ParserChoice::Symphony
            && names.tool.is_some()
            && named_table(&names).is_none()
        {
            tracing::warn!(
                tool_call_parser = ?names.tool,
                reasoning_parser = ?names.reasoning,
                "The configured tool-call parser name selects no Symphony table, so under \
                 --parser symphony every model the registry does not know is read by the legacy \
                 parsers (the registry's table reads the models it knows, a worker's own labels \
                 aside); a table's name, a legacy name that reads a table's format (qwen, \
                 qwen_xml, ...), a launcher's name for a family whose table has no older \
                 parser (muse), or an engine's exact name for a table (llama4_pythonic), reads \
                 those models with that table"
            );
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

        let rl = rl_adapter::build_rl_state(&worker_registry, &worker_client_cache, &router_config);

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
            tokenizer_load_failures: Arc::new(TokenizerLoadFailures::new()),
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
            .with_kv_event_monitor()
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
            let mut client_builder = crate::clients::http_client_builder()
                .pool_idle_timeout(pool_idle_timeout)
                .pool_max_idle_per_host(500)
                .timeout(Duration::from_secs(timeout_secs))
                // Below the request timeout, as the worker clients keep it
                // (`worker::http_client::connect_timeout_secs_below`): a dark
                // external host fails as a connect timeout, not as the
                // request's own (except under a 1 s request timeout, where the
                // two timers race).
                .connect_timeout(Duration::from_secs(connect_timeout_secs_below(
                    DEFAULT_CONNECT_TIMEOUT_SECS,
                    timeout_secs,
                )))
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
        let capacity = match effective_max_concurrent_requests(config.max_concurrent_requests) {
            None => {
                info!(
                    "Admission bound disabled (max_concurrent_requests = 0): in-flight requests are unbounded"
                );
                self.rate_limiter = None;
                return self;
            }
            Some(bound) if config.max_concurrent_requests < 0 => {
                info!(
                    "Admission bound: at most {bound} in-flight requests (default: {DEFAULT_INFLIGHT_PER_CORE} per available core, at least {DEFAULT_INFLIGHT_FLOOR}; --max-concurrent-requests overrides it, 0 disables it)"
                );
                bound
            }
            Some(bound) => {
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
            memory: config.memory.as_ref(),
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

    /// Create the KV event monitor: the streams of the gRPC workers.
    ///
    /// Every gateway has one, whatever its policies: a worker under a
    /// cache-aware policy streams its KV events into the model's index, every
    /// other gRPC worker streams its load records alone, so load-aware
    /// routing prices a worker by a record at the engine's cadence rather
    /// than by the last poll (the `UpdatePoliciesStep` picks the kind per
    /// worker; HTTP and ZMQ workers are never subscribed). The monitor itself
    /// is cheap (empty maps) and stays dormant until workers are added; a
    /// model without an events subscription has no index and no index
    /// gauges.
    fn with_kv_event_monitor(mut self) -> Self {
        let monitor = Arc::new(KvEventMonitor::new());
        debug!("Created KV event monitor for the workers' KV-event and load streams");
        // The load records on the streams are polls of the worker.
        if let Some(worker_monitor) = &self.worker_monitor {
            monitor.set_load_sink(worker_monitor);
        }
        monitor.start_stats_task();

        // Inject monitor into PolicyRegistry — propagates to default_policy
        // and any cache-aware policy created later from a worker's policy
        // label.
        if let Some(ref registry) = self.policy_registry {
            registry.set_kv_event_monitor(Some(Arc::clone(&monitor)));
        }

        self.kv_event_monitor = Some(monitor);
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
mod tests;

/// The test fixture shared by the modules that drive a workflow step or a
/// discovery pass against an `AppContext` built by hand.
#[cfg(test)]
pub(crate) mod testing;
