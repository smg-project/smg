use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use dashmap::DashSet;
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
use smg_policy_controller::{
    document::{AppliedSpec, PolicySpec},
    engine::{CacheIndexes, Members, PolicyEngine, WorkerData},
    stages::StageRegistry,
};
#[cfg(any(test, feature = "test-util"))]
use smg_worker::load::state::LoadReceiver;
use smg_worker::{
    connection::{
        engine::EngineConnector,
        http_client::{
            build_client, connect_timeout_secs_below, pool_idle_timeout, WorkerHttpClientCache,
            DEFAULT_CONNECT_TIMEOUT_SECS,
        },
    },
    health::liveness,
    settings::RegistrationSettings,
    workflow::WorkflowHost,
};
use tokio::sync::broadcast::error::RecvError;
use tool_parser::ParserFactory as ToolParserFactory;
use tracing::{debug, info};

use crate::{
    admin::worker_service::WorkerService,
    config::{ParserChoice, RouterConfig},
    middleware::{AuthConfig, TokenBucket},
    observability::inflight_tracker::InFlightRequestTracker,
    policies::{kv_index::event_monitor::KvEventMonitor, EngineCaches, PolicyRegistry},
    rate_limit::RateLimitManager,
    rl_adapter,
    routers::{
        common::{
            openai_bridge::FormatRegistry,
            overload, pd_admission,
            prefill_admission::PrefillAdmission,
            realtime::RealtimeRegistry,
            snapshot::{ModelWorkerSnapshot, WorkerMonitor, WorkerRegistry},
        },
        gateway::Gateway,
        grpc::{
            backend_client::{BackendConnector, Fronting},
            multimodal::MultimodalConfigRegistry,
            regular::symphony::{check_names, named_table},
            utils::{Names, ALIASES},
        },
    },
    server::http_client_builder,
    wasm::{config::WasmRuntimeConfig, module_manager::WasmModuleManager},
    workflow::{data::RegistrationNotes, JobQueue, TokenizerLoadFailures, WorkflowEngines},
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
    /// The ids of the tokenizer entries a worker's registration loaded:
    /// written by the registration workflow's loader under the registry's
    /// per-name lock, before the entry is published (a reader that sees the
    /// entry sees the record), read by the removal workflow's release step
    /// and by the second look of a put-back or a worker's load, which take
    /// no other entry (the startup entry and an operator's stay however
    /// their models' workers come and go), and dropped with the entry on
    /// every removal path, the operator's `DELETE /v1/tokenizers/{id}`
    /// included: that ends the worker's ownership, and the model's next
    /// registration loads afresh and records again.
    pub worker_loaded_tokenizers: Arc<DashSet<String>>,
    pub multimodal_config_registry: Arc<MultimodalConfigRegistry>,
    pub reasoning_parser_factory: Option<ReasoningParserFactory>,
    pub tool_parser_factory: Option<ToolParserFactory>,
    pub worker_registry: Arc<WorkerRegistry>,
    engine_connector: Arc<dyn EngineConnector>,
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
    /// What the worker crate's registration steps read of the configuration.
    pub registration_settings: RegistrationSettings,
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
    engine_connector: Arc<dyn EngineConnector>,
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

/// The context hosts the worker crate's workflow steps: they read the
/// registry, the worker clients, the registration's settings and the engine
/// connector through this and nothing else of the gateway.
impl WorkflowHost for AppContext {
    type Generation = ModelWorkerSnapshot;
    type RegistrationNotes = RegistrationNotes;

    fn worker_registry(&self) -> &Arc<WorkerRegistry> {
        &self.worker_registry
    }

    fn worker_client_cache(&self) -> &WorkerHttpClientCache {
        &self.worker_client_cache
    }

    fn registration_settings(&self) -> &RegistrationSettings {
        &self.registration_settings
    }

    fn engine_connector(&self) -> Arc<dyn EngineConnector> {
        Arc::clone(&self.engine_connector)
    }
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
        Self::from_config_with_extensions(
            router_config,
            request_timeout_secs,
            webrtc_bind_addr,
            webrtc_stun_server,
            None,
            None,
        )
    }

    /// [`Self::from_config`] over what a host added: its worker connector,
    /// and the table of policy stages its policy document may name. With no
    /// table the document is checked against the stock one.
    pub(crate) fn from_config_with_extensions(
        router_config: RouterConfig,
        request_timeout_secs: u64,
        webrtc_bind_addr: Option<std::net::IpAddr>,
        webrtc_stun_server: Option<String>,
        connector: Option<Arc<dyn EngineConnector>>,
        policy_stages: Option<StageRegistry>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self, String>> + Send>> {
        Box::pin(async move {
            Box::pin(AppContextBuilder::from_config_with_extensions(
                router_config,
                request_timeout_secs,
                webrtc_bind_addr,
                webrtc_stun_server,
                connector,
                policy_stages,
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

/// The whole seconds the overload shed advertises as Retry-After under a
/// load monitor that polls every `interval`. The header carries no
/// fraction, so an interval with one is raised to the next second.
fn shed_retry_after_secs(interval: Duration) -> u64 {
    interval.as_secs() + u64::from(interval.subsec_nanos() != 0)
}

impl AppContextBuilder {
    pub fn new() -> Self {
        Self::new_with_connector(None)
    }

    fn new_with_connector(connector: Option<Arc<dyn EngineConnector>>) -> Self {
        Self {
            client: None,
            router_config: None,
            rate_limiter: None,
            rate_limit_manager: None,
            tokenizer_registry: None,
            reasoning_parser_factory: None,
            tool_parser_factory: None,
            worker_registry: None,
            // A host's connector is wrapped once: the gateway fronts each
            // gRPC connection it makes.
            engine_connector: connector.map_or_else(BackendConnector::shared, Fronting::around),
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
        // thresholds and the warm-up slice, see `smg_worker::health::liveness`) take
        // effect for every context built from it, the server's and a test's
        // alike. They are process-wide and the first call wins, so a
        // constructor that builds the policy registry itself configures them
        // before that build (`from_config` and the test harness do).
        // This call covers a context assembled without such a constructor.
        liveness::configure_from(router_config.worker_liveness_budgets());
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
        let policy_registry = self
            .policy_registry
            .ok_or(AppContextBuildError::MissingField("policy_registry"))?;
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
        // The hash rings are filled where a membership change is heard, off
        // the request path. Where no runtime is entered nothing warms them,
        // and the first reader of a generation fills its ring.
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            policy_registry.keep_rings_warm(&worker_registry, &runtime);
        }

        // Create WorkerService from the already-built components
        let worker_service = Arc::new(WorkerService::new(
            worker_registry.clone(),
            worker_job_queue.clone(),
            router_config.clone(),
        ));

        let worker_client_cache = Arc::new(WorkerHttpClientCache::new(
            router_config.worker_http_client_settings(),
        ));
        let registration_settings = router_config.registration_settings();
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
            worker_loaded_tokenizers: Arc::new(DashSet::new()),
            reasoning_parser_factory: self.reasoning_parser_factory,
            tool_parser_factory: self.tool_parser_factory,
            worker_registry,
            engine_connector: self.engine_connector,
            prefill_admission,
            policy_registry,
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
            registration_settings,
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
        Box::pin(Self::from_config_with_extensions(
            router_config,
            request_timeout_secs,
            webrtc_bind_addr,
            webrtc_stun_server,
            None,
            None,
        ))
        .await
    }

    async fn from_config_with_extensions(
        router_config: RouterConfig,
        request_timeout_secs: u64,
        webrtc_bind_addr: Option<std::net::IpAddr>,
        webrtc_stun_server: Option<String>,
        connector: Option<Arc<dyn EngineConnector>>,
        policy_stages: Option<StageRegistry>,
    ) -> Result<Self, String> {
        // The policy document comes first, checked against the stages the
        // host registered: ahead of the stores, the storage hooks and every
        // listener, so a document that is refused ends the start in its own
        // words. The engine is started last, from the document that was
        // checked here.
        let policy = Self::policy_in_force(&router_config, policy_stages.as_ref())?;
        // The overload veto's two words are read once below, where the
        // registration settings are made and the registry's shed word is set.
        // A supplied document this gateway refuses for them ends the start
        // here, behind the document's own check and in the same form.
        router_config
            .supplied_overload_words()
            .map_err(|refused| format!("policy controller: {refused}"))?;
        // A served model aliased under a vendor name keeps that vendor's
        // contract profile after the alias is resolved into the served name.
        ProviderProfile::register_model_aliases(
            router_config
                .model_aliases
                .iter()
                .map(|(alias, canonical)| (alias.as_str(), canonical.as_str())),
        );
        // The configuration's liveness budgets take effect before the policy
        // registry is built below (the first call wins; `build` keeps its
        // call for a context assembled without this constructor).
        liveness::configure_from(router_config.worker_liveness_budgets());
        Ok(Self::new_with_connector(connector)
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
            .with_policy_engine(policy)?
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
        let pool_idle_timeout = pool_idle_timeout(config.upstream_pool_idle_timeout_secs);
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
            let mut client_builder = http_client_builder()
                .pool_idle_timeout(pool_idle_timeout)
                .pool_max_idle_per_host(500)
                .timeout(Duration::from_secs(timeout_secs))
                // Below the request timeout, as the worker clients keep it
                // (`smg_worker::connection::http_client::connect_timeout_secs_below`): a dark
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
        self.worker_registry = Some(Arc::new(
            WorkerRegistry::new().with_engine_connector(Arc::clone(&self.engine_connector)),
        ));
        self
    }

    /// Create policy registry: the one that makes no classic policy where
    /// the configuration runs the policy controller
    /// ([`PolicyRegistry::of_config`]).
    fn with_policy_registry(mut self, config: &RouterConfig) -> Self {
        self.policy_registry = Some(Arc::new(PolicyRegistry::of_config(config)));
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
        let registry = self
            .worker_registry
            .as_ref()
            .ok_or_else(|| "worker_registry must be set before load monitor".to_string())?
            .clone();
        let interval = config.load_monitor_interval_secs;
        let (engine_metrics, conditional) = (config.engine_metrics, config.disable_load_monitoring);
        // The monitor's snapshot feeds every policy that consumes it; the
        // monitor polls every group by default, conditionally under
        // `--disable-load-monitoring`.
        let monitor = WorkerMonitor::new(
            registry,
            Arc::clone(&policy_registry),
            interval,
            engine_metrics,
            conditional,
        );
        // An interval stated in milliseconds takes the place of the seconds.
        let monitor = Arc::new(match config.load_monitor_interval_ms {
            Some(millis) => monitor.with_interval(Duration::from_millis(millis)),
            None => monitor,
        });
        policy_registry.set_load_receiver(Some(monitor.subscribe()));
        // The overload shed advertises the poll interval as Retry-After — the
        // veto cannot clear between polls.
        overload::set_shed_retry_after_secs(shed_retry_after_secs(config.load_monitor_interval()));
        // The shed word in force: a supplied document's own where the policy
        // controller runs one, else the configuration's.
        if let Some(registry) = self.worker_registry.as_ref() {
            registry.set_overload_shed(config.overload_words_in_force().shed);
        }
        // PD dispatch waits here, not in the decode engine's queue, when the
        // pair's running window is full.
        pd_admission::set_pd_admission_wait_secs(config.pd_admission_wait_secs);
        self.worker_monitor = Some(monitor);
        Ok(self)
    }

    /// The policy document this process runs under the policy controller
    /// (`policy_controller: on`), with its checked form: the document in
    /// force, which is the one the routing settings mean or a supplied one,
    /// validated against `stages`. A host's table holds the stock stages and
    /// the host's own, so a supplied document may name both. With no table
    /// the document is validated against the stock one. `None` where the
    /// configuration leaves the classic policies in place.
    ///
    /// A document that is refused ends the start with the refusal's own
    /// words, before a store is opened and before the gateway listens.
    ///
    /// The classic policies stay in place with the option off, and with it
    /// on in a gateway whose every router is one provider's
    /// (`RouterConfig::policy_engine_decides`). That gateway places its
    /// requests by no routing policy, so the option moves nothing there,
    /// and this start says so in one log line.
    ///
    /// A document whose `explain.sample` is not zero is named in one warn
    /// line, ahead of its check: the engine this process hosts keeps no
    /// sample (the served controller alone does), and the document is not
    /// refused for it.
    fn policy_in_force(
        config: &RouterConfig,
        stages: Option<&StageRegistry>,
    ) -> Result<Option<(PolicySpec, AppliedSpec)>, String> {
        if !config.policy_controller {
            return Ok(None);
        }
        if !config.policy_engine_decides() {
            info!(
                "Policy controller on and placing no request: mode = {} places its requests by no routing policy",
                config.mode_type()
            );
            return Ok(None);
        }
        let document = config
            .policy_document_in_force()
            .map_err(|refused| format!("policy controller: {refused}"))?;
        // One document may serve this gateway and a served controller that
        // keeps the sample, so the field is named here and not refused.
        if document.explain.sample != 0 {
            tracing::warn!(
                sample = document.explain.sample,
                "The policy document sets explain.sample, and the policy engine this gateway \
                 hosts keeps no sample of its decisions"
            );
        }
        let checked = match stages {
            Some(stages) => config.hosted_policy(&document, stages),
            None => config.hosted_policy(&document, &StageRegistry::builtin()),
        };
        checked
            .map(|spec| Some((document, spec)))
            .map_err(|refused| format!("policy controller: {refused}"))
    }

    /// Starts the policy engine over `policy`, where the configuration runs
    /// the policy controller: the document [`Self::policy_in_force`]
    /// answered, with its checked form. It is the last step of the start,
    /// so the engine is handed where this process keeps its workers' data,
    /// each once: the worker registry, the load monitor's channel, and the
    /// cache-event monitor with the token trees the engine is lent
    /// ([`Self::engine_caches`]).
    fn with_policy_engine(self, policy: Option<(PolicySpec, AppliedSpec)>) -> Result<Self, String> {
        let Some((document, spec)) = policy else {
            return Ok(self);
        };
        let unset = |field: &str| format!("{field} must be set before the policy engine");
        let policies = self
            .policy_registry
            .as_ref()
            .ok_or_else(|| unset("policy_registry"))?;
        let registry = self
            .worker_registry
            .as_ref()
            .ok_or_else(|| unset("worker_registry"))?;
        let monitor = self
            .worker_monitor
            .as_ref()
            .ok_or_else(|| unset("worker_monitor"))?;
        let cache_events = self
            .kv_event_monitor
            .as_ref()
            .ok_or_else(|| unset("kv_event_monitor"))?;
        let events = Some(cache_events);
        let data = WorkerData {
            registry: Arc::clone(registry) as Arc<dyn Members>,
            loads: Some(monitor.subscribe()),
            cache_indexes: Self::engine_caches(&document, &spec, events, policies, registry),
        };
        Self::start_policy_engine(spec, data, policies)?;
        Ok(self)
    }

    /// What the engine is handed of this process's caches, as one object
    /// ([`EngineCaches`]): `events`, the cache-event monitor where the
    /// process keeps one, and the token trees the engine is lent.
    ///
    /// With the engine on no classic policy object is built, so no policy
    /// keeps the trees of the prompts this process sent. The start asks the
    /// registry for them in a policy's place
    /// ([`PolicyRegistry::lend_token_trees`]), and only where `spec`, the
    /// checked form of `document`, reads the prefixes sent to each worker:
    /// the document's own word, whatever the settings name the policy. The
    /// trees take the document's numbers too: its block size, and each
    /// model's `index` entry. They follow `workers`, the registry the engine
    /// is handed: a worker it lost is taken out of them. `None` where the
    /// process keeps neither a monitor nor trees.
    fn engine_caches(
        document: &PolicySpec,
        spec: &AppliedSpec,
        events: Option<&Arc<KvEventMonitor>>,
        policies: &PolicyRegistry,
        workers: &WorkerRegistry,
    ) -> Option<Arc<dyn CacheIndexes>> {
        let trees = spec
            .needs()
            .sent_prefixes
            .then(|| policies.lend_token_trees(document, spec.block_size(), workers));
        let caches = EngineCaches::of(events.cloned(), trees)?;
        Some(Arc::new(caches) as Arc<dyn CacheIndexes>)
    }

    /// Starts the policy engine that places every request of this process:
    /// one engine, by the policy crate's one start call, over the checked
    /// document ([`Self::policy_in_force`]) and `data`, where this process
    /// keeps its workers' data. The engine's task follows the registry's
    /// events and tells the document's stages a worker that joined or was
    /// removed, and it keeps the newest load snapshot at hand. It is
    /// spawned on the runtime this context is built on, and the placement
    /// functions find the engine through `policies`.
    fn start_policy_engine(
        spec: AppliedSpec,
        data: WorkerData,
        policies: &PolicyRegistry,
    ) -> Result<(), String> {
        let runtime = tokio::runtime::Handle::try_current().map_err(|error| {
            format!("policy controller: the engine's task needs a Tokio runtime: {error}")
        })?;
        // One engine for the life of the process: its placements borrow it.
        let (engine, task) = PolicyEngine::start(spec, data);
        // The engine's task keeps nothing a shutdown waits for: its upkeep,
        // its load snapshot and its follow of the registry end with the
        // runtime.
        runtime.spawn(task);
        policies.set_engine(engine);
        Ok(())
    }

    /// The policy engine's start for a context a test builds by hand, field
    /// by field, where no builder step of this file runs: the start's own
    /// two steps, in their order and with nothing added (`policy_in_force`,
    /// then `start_policy_engine`). It is not a second start. `stages` is
    /// the table a host would bring, where the test brings one, and a
    /// configuration that leaves the policy controller off starts nothing.
    /// The engine is handed the test's worker registry and its load
    /// monitor's channel. Such a context keeps no cache-event monitor, so
    /// the engine is handed none, and the token trees it is lent where its
    /// document reads them (`engine_caches`).
    #[cfg(any(test, feature = "test-util"))]
    pub fn start_policy_engine_by_hand(
        config: &RouterConfig,
        stages: Option<&StageRegistry>,
        policies: &PolicyRegistry,
        registry: &Arc<WorkerRegistry>,
        monitor: &WorkerMonitor,
    ) -> Result<(), String> {
        if let Some((document, spec)) = Self::policy_in_force(config, stages)? {
            let data = WorkerData {
                registry: Arc::clone(registry) as Arc<dyn Members>,
                loads: Some(monitor.subscribe()),
                cache_indexes: Self::engine_caches(&document, &spec, None, policies, registry),
            };
            Self::start_policy_engine(spec, data, policies)?;
        }
        Ok(())
    }

    /// The policy engine's start for a bench, which builds no context and
    /// keeps no load monitor: the start's own two steps, in their order and
    /// with nothing added (`policy_in_force` over the stock stages, then
    /// `start_policy_engine`), as [`Self::start_policy_engine_by_hand`] runs
    /// them for a test. The engine is handed the bench's worker registry and
    /// `loads`, the channel of the load feed the bench publishes where a
    /// monitor would. A bench keeps no cache-event monitor, so the engine is
    /// handed none, and the token trees it is lent where its document reads
    /// them (`engine_caches`). A configuration that leaves the policy
    /// controller off starts nothing.
    #[cfg(any(test, feature = "test-util"))]
    pub fn start_policy_engine_for_bench(
        config: &RouterConfig,
        policies: &PolicyRegistry,
        registry: &Arc<WorkerRegistry>,
        loads: Option<LoadReceiver>,
    ) -> Result<(), String> {
        if let Some((document, spec)) = Self::policy_in_force(config, None)? {
            let data = WorkerData {
                registry: Arc::clone(registry) as Arc<dyn Members>,
                loads,
                cache_indexes: Self::engine_caches(&document, &spec, None, policies, registry),
            };
            Self::start_policy_engine(spec, data, policies)?;
        }
        Ok(())
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
    /// than by the last poll (`start_kv_stream` picks the kind per worker,
    /// for a registration and for an update that rebuilt the worker alike;
    /// HTTP and ZMQ workers are never subscribed). The monitor itself
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
        // The add reads the registry before it stops or starts a stream: a
        // worker the registry no longer holds under its URL gets none.
        if let Some(registry) = &self.worker_registry {
            monitor.set_worker_registry(registry);
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
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use openai_protocol::worker::{HealthCheckConfig, WorkerSpec, WorkerStatus};
    use smg_worker::{
        connection::engine::NO_ENGINE_CONNECTOR,
        identity::basic::{ConnectionMode, Worker, WorkerType},
        kv_events::SubscriptionKind,
        test_support::RefusedAddress,
    };
    use tracing_test::traced_test;

    use super::*;
    use crate::{
        config::types::PolicyConfig,
        routers::{
            common::{
                prefill_admission::{test_support::select_worker, PrefillAdmissionAttempt},
                snapshot::test_support::worker_of,
            },
            grpc::backend_client::BasicWorkerBuilder,
        },
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
    /// `smg_worker::connection::http_client::no_root_store`); a no-op unless the case
    /// variable is set.
    #[test]
    fn child_with_client_without_native_roots() {
        use smg_worker::connection::http_client::no_root_store;

        if std::env::var(no_root_store::CASE).is_err() {
            return;
        }
        AppContextBuilder::new()
            .with_client(&RouterConfig::default(), 5)
            .expect("a gateway without TLS configuration starts without a root store");
    }

    #[test]
    #[cfg_attr(
        any(target_vendor = "apple", windows, target_os = "android"),
        ignore = "the native verifier does not use SSL_CERT_FILE/SSL_CERT_DIR"
    )]
    fn the_gateway_client_builds_without_a_native_root_store() {
        use smg_worker::connection::http_client::no_root_store;

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

    /// The admission bucket of a ten-request cap under
    /// `rate_limit_tokens_per_second` hands out its ten tokens and refills
    /// none over 150 ms; a returned token is handed out again.
    async fn ten_tokens_and_no_refill(rate_limit_tokens_per_second: Option<i32>) {
        let config = RouterConfig {
            max_concurrent_requests: 10,
            rate_limit_tokens_per_second,
            ..RouterConfig::default()
        };
        let bucket = AppContextBuilder::new()
            .maybe_rate_limiter(&config)
            .rate_limiter
            .expect("rate limiter should be enabled");

        assert!(bucket.try_acquire(10.0).is_ok());
        // An earlier fallback refilled at max_concurrent_requests per second,
        // which would restore a token during this wait.
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(bucket.try_acquire(1.0).is_err());

        bucket.return_tokens_sync(1.0);
        assert!(bucket.try_acquire(1.0).is_ok());
    }

    #[tokio::test]
    async fn unset_rate_limit_defaults_to_no_refill() {
        ten_tokens_and_no_refill(None).await;
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
        ten_tokens_and_no_refill(Some(0)).await;
    }

    fn config_with_policy(policy: PolicyConfig) -> RouterConfig {
        RouterConfig {
            policy,
            ..Default::default()
        }
    }

    /// Whether `with_kv_event_monitor` creates the monitor under `policy`.
    /// This run of the builder needs no storage or network.
    fn kv_monitor_created_for(policy: PolicyConfig) -> bool {
        let config = config_with_policy(policy);
        AppContextBuilder::new()
            .with_policy_registry(&config)
            .with_kv_event_monitor()
            .kv_event_monitor
            .is_some()
    }

    /// The load-snapshot feed must be wired whenever the worker monitor is
    /// built — without a KV-event monitor in the chain — so HTTP-only
    /// cache-aware deployments get waiting-prefill and KV-usage data.
    #[test]
    fn worker_monitor_wires_load_receiver_into_policies() {
        use crate::policies::as_cache_aware;

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
            balance_prefix_margin_tokens: 0,
            balance_uncached_credit: false,
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
        let cache_aware = as_cache_aware(policy.as_ref()).expect("default policy is cache-aware");
        assert!(cache_aware.has_load_receiver_for_test());
    }

    /// The load monitor is built with the interval the configuration states
    /// in milliseconds, over its seconds, and with the seconds where it
    /// states no milliseconds.
    #[test]
    fn the_worker_monitor_takes_its_interval_in_milliseconds_over_the_seconds() {
        let built_with = |millis: Option<u64>| {
            let config = RouterConfig {
                load_monitor_interval_secs: 5,
                load_monitor_interval_ms: millis,
                ..config_with_policy(PolicyConfig::RoundRobin)
            };
            AppContextBuilder::new()
                .with_client(&config, 5)
                .expect("client builds")
                .with_worker_registry()
                .with_policy_registry(&config)
                .with_worker_monitor(&config)
                .expect("worker monitor builds")
                .worker_monitor
                .expect("the worker monitor")
                .interval()
        };
        assert_eq!(built_with(Some(250)), Duration::from_millis(250));
        assert_eq!(built_with(None), Duration::from_secs(5));
    }

    /// The shed's Retry-After is the poll interval in whole seconds: an
    /// interval with a fraction is raised to the next second, and whole
    /// seconds read as they are.
    #[test]
    fn the_sheds_retry_after_raises_a_fraction_to_the_next_second() {
        for (millis, seconds) in [(100, 1), (250, 1), (1_000, 1), (1_500, 2), (10_000, 10)] {
            assert_eq!(
                shed_retry_after_secs(Duration::from_millis(millis)),
                seconds,
                "{millis} ms"
            );
        }
    }

    /// Every policy gets the monitor: a cache-aware one for its indexes, any
    /// other for the load stream of its gRPC workers, which the policies'
    /// load feed reads in place of the poll.
    #[test]
    fn every_policy_creates_the_kv_event_monitor() {
        assert!(kv_monitor_created_for(PolicyConfig::Passthrough));
        assert!(kv_monitor_created_for(PolicyConfig::RoundRobin));
        assert!(kv_monitor_created_for(PolicyConfig::PowerOfTwo {
            load_metric: Default::default(),
        }));
        assert!(kv_monitor_created_for(PolicyConfig::cache_aware(
            0.5, 32, 1.1, 30, 1000
        )));
    }

    /// The monitor reaches a cache-aware policy created later from a
    /// worker's policy label under a non-cache-aware default: its model's
    /// workers get the event-driven index rather than the approximate tree.
    #[test]
    fn a_cache_aware_policy_from_a_worker_label_gets_the_kv_event_monitor() {
        use crate::policies::as_cache_aware;

        let config = config_with_policy(PolicyConfig::Random);
        let builder = AppContextBuilder::new()
            .with_policy_registry(&config)
            .with_kv_event_monitor();
        assert!(builder.kv_event_monitor.is_some());
        let registry = builder
            .policy_registry
            .as_ref()
            .expect("policy registry set");
        let policy = registry.on_worker_added("affine", Some("cache_aware"));
        let cache_aware =
            as_cache_aware(policy.as_ref()).expect("the label's policy is cache-aware");
        assert!(cache_aware.kv_event_monitor_is_set_for_test());
    }

    /// A dropped context lets its worker monitor go: the monitor holds the
    /// policy registry, and nothing the context builds holds the monitor
    /// back.
    #[tokio::test]
    async fn a_dropped_context_lets_its_worker_monitor_go() {
        let mut config = RouterConfig::builder()
            .round_robin_policy()
            .build_unchecked();
        config.health_check.disable_health_check = true;
        let context = AppContext::from_config(config, 30, None, None)
            .await
            .expect("a context");
        let monitor = Arc::downgrade(context.worker_monitor.as_ref().expect("the worker monitor"));
        drop(context);
        assert!(
            monitor.upgrade().is_none(),
            "the worker monitor outlives its context"
        );
    }

    /// The start lends the policy engine the gateway's own token trees
    /// where the document in force reads the prefixes sent to each worker,
    /// and makes none under a document that reads none. Under `cache_aware`
    /// a conversation's next turns then go back to the worker its first turn
    /// was sent to, and the model's tree holds the conversation for that
    /// worker alone.
    ///
    /// The classic policy keeps such a tree itself and reads it the same
    /// way (`model_gateway/src/policies/cache_aware.rs`,
    /// `select_worker_with_tokens`).
    #[tokio::test]
    async fn the_start_lends_the_engine_the_token_trees_its_document_reads() {
        use std::time::Instant;

        use openai_protocol::worker::RuntimeType;

        use crate::routers::common::{
            placement::{select_single, PlacementInputs, Stated},
            snapshot::{test_support::grpc_worker, RoutingPool},
        };

        let started = |policy: PolicyConfig| async {
            let mut config = config_with_policy(policy);
            config.policy_controller = true;
            config.health_check.disable_health_check = true;
            let context = AppContext::from_config(config, 30, None, None).await;
            context.expect("a context")
        };
        let plain = started(PolicyConfig::RoundRobin).await;
        assert!(plain.policy_registry.engine().is_some());
        assert!(plain.policy_registry.lent_trees().is_none());

        let context = started(PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096)).await;
        let trees = context.policy_registry.lent_trees();
        let trees = trees.expect("the start made the trees");
        for at in 0..4 {
            let url = format!("grpc://w{at}:9000");
            let worker = grpc_worker(&url, "m", RuntimeType::Vllm, &[]);
            context.worker_registry.register(worker).expect("a worker");
        }
        let turn = |said: &[u32]| {
            let inputs = PlacementInputs {
                tokens: Some(said),
                ..PlacementInputs::default()
            };
            let placed = select_single(
                &context.worker_registry,
                &context.policy_registry,
                "m",
                RoutingPool::GrpcPipelineRegular,
                None,
                inputs,
                Stated::nothing,
            );
            placed.expect("a worker").into_worker().url().to_owned()
        };
        let mut said: Vec<u32> = (0..256).collect();
        let first = turn(&said);
        for next in 1..8_u32 {
            said.extend(next * 1_000_000..next * 1_000_000 + 16);
            assert_eq!(turn(&said), first, "turn {next}");
        }
        let tree = trees.token_tree("m");
        let held: Vec<String> = tree.get_tenant_token_counts().into_keys().collect();
        assert_eq!(held, [first.as_str()]);

        // The start made the trees with the task that follows the context's
        // own registry: the worker that leaves it is taken out of them.
        let registry = &context.worker_registry;
        registry.remove_by_url(&first).expect("the removal");
        let bound = Instant::now() + Duration::from_secs(60);
        while !tree.get_tenant_token_counts().is_empty() {
            assert!(Instant::now() < bound, "the trees' task never purged");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // The trees take the document's numbers. The document the settings
        // mean states the settings' block size. A supplied document states
        // its own, whatever policy the settings name.
        let mut paged = PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096);
        if let PolicyConfig::CacheAware { block_size, .. } = &mut paged {
            *block_size = 8;
        }
        let supplied = config_with_policy(paged.clone()).policy_document();
        let by_settings = started(paged).await.policy_registry.lent_trees();
        let by_settings = by_settings.expect("the start made the trees");
        assert_eq!(by_settings.token_tree("m").page_size(), 8);
        let mut config = RouterConfig {
            policy_controller: true,
            policy_spec: Some(supplied.expect("the settings' document")),
            ..RouterConfig::default()
        };
        config.health_check.disable_health_check = true;
        let context = AppContext::from_config(config, 30, None, None).await;
        let by_document = context.expect("a context").policy_registry.lent_trees();
        let by_document = by_document.expect("the supplied document reads sent prefixes");
        assert_eq!(by_document.token_tree("m").page_size(), 8);

        // The start a test makes by hand lends the trees by the same rule.
        let mut config = config_with_policy(PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096));
        config.policy_controller = true;
        let policies = Arc::new(PolicyRegistry::of_config(&config));
        let registry = Arc::new(WorkerRegistry::new());
        let monitor =
            WorkerMonitor::new(Arc::clone(&registry), Arc::clone(&policies), 1, false, true);
        let by_hand = AppContextBuilder::start_policy_engine_by_hand(
            &config, None, &policies, &registry, &monitor,
        );
        by_hand.expect("the settings' own document");
        assert!(policies.lent_trees().is_some());
    }

    /// A context built on a runtime starts the task that fills the hash
    /// rings: the ring of a generation is filled with no request to read it.
    #[tokio::test]
    async fn a_context_built_on_a_runtime_fills_the_hash_rings_itself() {
        use openai_protocol::UNKNOWN_MODEL_ID;

        use crate::routers::common::snapshot::WorkerRegistryExt as _;

        let mut config = RouterConfig::builder()
            .round_robin_policy()
            .build_unchecked();
        config.health_check.disable_health_check = true;
        let context = AppContext::from_config(config, 30, None, None)
            .await
            .expect("a context");
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://ring:8080")
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        context.worker_registry.register(worker).expect("a worker");

        let bound = std::time::Instant::now() + Duration::from_secs(30);
        while !context
            .worker_registry
            .get_routing_snapshot(UNKNOWN_MODEL_ID)
            .hash_ring_is_filled()
        {
            assert!(
                std::time::Instant::now() < bound,
                "no task filled the ring of the generation"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// A gateway whose one router is a provider's, with the policy
    /// controller on.
    fn a_providers_gateway_with_the_option_on() -> RouterConfig {
        let mut config = RouterConfig {
            mode: crate::config::RoutingMode::OpenAI {
                worker_urls: Vec::new(),
            },
            policy_controller: true,
            ..RouterConfig::default()
        };
        config.health_check.disable_health_check = true;
        config
    }

    /// A gateway whose one router is a provider's places its requests by no
    /// routing policy, so the policy controller's option moves nothing
    /// there. The gateway starts, no engine runs, the registry keeps the
    /// classic policies, and the pairing mode is the configuration's own.
    #[tokio::test]
    async fn a_providers_gateway_starts_with_the_option_on_and_runs_no_engine() {
        use crate::config::PdPairingMode;

        let config = RouterConfig {
            pd_pairing_mode: PdPairingMode::Strict,
            ..a_providers_gateway_with_the_option_on()
        };
        let context = AppContext::from_config(config, 30, None, None)
            .await
            .expect("the gateway starts");
        let registry = &context.policy_registry;
        assert!(registry.engine().is_none(), "no engine runs");
        assert!(registry.keeps_classic_policies());
        assert_eq!(registry.pd_pairing_mode(), PdPairingMode::Strict);
    }

    /// A supplied document is never dropped in silence: beside a provider's
    /// mode alone the start refuses it in the policy crate's words, which
    /// name the mode.
    #[tokio::test]
    async fn a_document_supplied_to_a_providers_gateway_is_refused_at_the_start() {
        use smg_policy_controller::document::PolicySpec;

        let config = RouterConfig {
            policy_spec: Some(PolicySpec::preset("round_robin").expect("a preset")),
            ..a_providers_gateway_with_the_option_on()
        };
        let words = config
            .policy_document()
            .expect_err("the mode has no document")
            .to_string();
        assert!(words.contains("openai"), "{words}");

        let refused = AppContext::from_config(config, 30, None, None)
            .await
            .expect_err("the start is refused");
        assert_eq!(refused, format!("policy controller: {words}"));
    }

    /// Under the multi-router setting a provider's mode has the regular
    /// document: the engine starts, and it decides for the gateway's
    /// self-hosted routers.
    #[tokio::test]
    async fn a_providers_mode_beside_other_routers_starts_the_engine() {
        let config = RouterConfig {
            enable_igw: true,
            ..a_providers_gateway_with_the_option_on()
        };
        let regular = RouterConfig {
            mode: RouterConfig::default().mode,
            ..config.clone()
        };
        assert_eq!(
            config.policy_document_in_force(),
            regular.policy_document_in_force(),
            "the regular document"
        );

        let context = AppContext::from_config(config, 30, None, None)
            .await
            .expect("the gateway starts");
        let registry = &context.policy_registry;
        assert!(registry.engine().is_some(), "the engine runs");
        assert!(!registry.keeps_classic_policies());
    }

    /// The start says that the option moves nothing, once. The option's
    /// other readers say nothing.
    #[traced_test]
    #[tokio::test]
    async fn the_start_alone_says_once_that_the_option_moves_nothing() {
        use crate::config::validation::ConfigValidator;

        const LINE: &str = "Policy controller on and placing no request: mode = openai";
        let config = a_providers_gateway_with_the_option_on();
        assert!(PolicyRegistry::of_config(&config).keeps_classic_policies());
        assert_eq!(config.pd_pairing_in_force(), config.pd_pairing_mode);
        ConfigValidator::validate_policy_document(&config).expect("no document to refuse");
        assert!(!logs_contain(LINE), "a reader other than the start said it");

        AppContext::from_config(config, 30, None, None)
            .await
            .expect("the gateway starts");
        logs_assert(|lines: &[&str]| {
            match lines.iter().filter(|line| line.contains(LINE)).count() {
                1 => Ok(()),
                said => Err(format!("the start said the line {said} times")),
            }
        });
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

    #[tokio::test]
    async fn maybe_rate_limit_manager_enabled_with_valid_policy_succeeds() {
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

    /// The registry the gateway builds connects its mesh imports through
    /// the gateway's connector: an imported gRPC worker's connect reaches
    /// for its engine, it is not refused for want of a connector.
    #[tokio::test]
    async fn the_gateways_registry_connects_its_mesh_imports() {
        let refused = RefusedAddress::reserve();
        let registry = AppContextBuilder::new()
            .with_worker_registry()
            .worker_registry
            .expect("the registry");
        let mut spec = WorkerSpec::new(format!("grpc://{}", refused.addr()));
        spec.connection_mode = ConnectionMode::Grpc;
        registry.on_remote_worker_state(&smg_worker::registry::RemoteWorkerState {
            worker_id: "peer-w1".to_string(),
            model_id: "m".to_string(),
            url: spec.url.clone(),
            health: true,
            version: 1,
            spec: serde_json::to_vec(&spec).expect("a spec encodes"),
        });
        let import = registry.get_all().pop().expect("the import");
        assert_eq!(*import.connection_mode(), ConnectionMode::Grpc);

        match import.engine_connection().await {
            Ok(connection) => assert!(connection.is_some()),
            Err(refusal) => assert!(
                !refusal.to_string().contains(NO_ENGINE_CONNECTOR),
                "{refusal}"
            ),
        }
    }

    #[tokio::test]
    async fn worker_status_change_wakes_prefill_admission() {
        let registry = Arc::new(WorkerRegistry::new());
        let admission = Arc::new(PrefillAdmission::new(1, 1, Duration::from_secs(1)));
        start_prefill_admission_notifier(&registry, &admission).unwrap();

        let first = worker_of("http://prefill-1:8000", WorkerType::Prefill);
        registry.register(Arc::clone(&first)).unwrap();
        let occupied = admission.admit(None, select_worker(&first)).await.unwrap();

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

    /// The app wires its worker registry into the monitor: an add for a
    /// gRPC worker the registry does not hold starts no stream, and the
    /// same worker gets one once registered.
    #[traced_test]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_kv_event_monitor_reads_the_apps_worker_registry() {
        let config = config_with_policy(PolicyConfig::Random);
        let builder = AppContextBuilder::new()
            .with_worker_registry()
            .with_policy_registry(&config)
            .with_kv_event_monitor();
        let monitor = builder
            .kv_event_monitor
            .as_ref()
            .expect("the monitor is built");
        let registry = builder
            .worker_registry
            .as_ref()
            .expect("the worker registry is built");
        // A port that refuses connections at once: the stream's task loops
        // on reconnects until `stop`, with no engine involved.
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("grpc://127.0.0.1:1")
                .connection_mode(ConnectionMode::Grpc)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );

        monitor
            .on_worker_added(&worker, SubscriptionKind::Loads)
            .await;
        assert!(logs_contain(
            "no KV event stream for a worker the registry no longer holds"
        ));
        assert!(!logs_contain("Starting KV event subscription"));

        registry
            .register(Arc::clone(&worker))
            .expect("the worker registers");
        monitor
            .on_worker_added(&worker, SubscriptionKind::Loads)
            .await;
        assert!(logs_contain("Starting KV event subscription"));
        monitor.stop().await;
    }

    /// The engine this process hosts keeps no sample of its decisions. A
    /// supplied document that leaves `explain.sample` at zero is taken
    /// without a word, and one that sets it is taken with one warn line.
    /// One the check refuses is named too, ahead of its refusal.
    #[traced_test]
    #[test]
    fn a_documents_explain_sample_is_named_at_the_start() {
        let mut config = RouterConfig {
            policy_controller: true,
            ..Default::default()
        };
        let mut document = config.policy_document().expect("the settings' document");
        config.policy_spec = Some(document.clone());
        let taken = AppContextBuilder::policy_in_force(&config, None);
        assert!(matches!(taken, Ok(Some(_))));
        assert!(!logs_contain("explain.sample"));

        document.explain.sample = 10;
        config.policy_spec = Some(document.clone());
        let taken = AppContextBuilder::policy_in_force(&config, None);
        assert!(matches!(taken, Ok(Some(_))));
        logs_assert(|lines: &[&str]| {
            let named: Vec<_> = lines
                .iter()
                .filter(|line| line.contains("explain.sample"))
                .collect();
            match named.as_slice() {
                [line] if line.contains("WARN") && line.contains("sample=10") => Ok(()),
                _ => Err(format!("the lines that name explain.sample: {named:?}")),
            }
        });

        // Its binding's profile gone, the document is refused by the check.
        document.explain.sample = 11;
        document.profiles.clear();
        config.policy_spec = Some(document);
        let taken = AppContextBuilder::policy_in_force(&config, None);
        assert!(taken.is_err());
        logs_assert(|lines: &[&str]| {
            let named = lines.iter().filter(|line| line.contains("sample=11"));
            match named.count() {
                1 => Ok(()),
                named => Err(format!("the lines of the refused document: {named}")),
            }
        });
    }
}

#[cfg(test)]
mod effective_max_concurrent_requests_tests {
    use super::{default_max_concurrent_requests, effective_max_concurrent_requests};

    /// The three readings of `max_concurrent_requests`, pinned in one place
    /// for the admission bucket and the start-up checks alike.
    #[test]
    fn the_sentinels_resolve_to_off_the_default_and_the_value() {
        assert_eq!(effective_max_concurrent_requests(0), None);
        assert_eq!(
            effective_max_concurrent_requests(-1),
            Some(default_max_concurrent_requests())
        );
        assert_eq!(effective_max_concurrent_requests(7), Some(7));
    }
}

/// The test fixture shared by the modules whose cases need an `AppContext`
/// built by hand.
#[cfg(test)]
pub(crate) mod testing {
    use std::sync::Arc;

    use smg_worker::connection::http_client::WorkerHttpClientCache;

    use super::{AppContext, BackendConnector, ToolParserFactory};
    use crate::{
        admin::worker_service::WorkerService,
        config::RouterConfig,
        middleware::{AuthConfig, TokenBucket},
        observability::inflight_tracker::InFlightRequestTracker,
        policies::PolicyRegistry,
        routers::{
            common::{openai_bridge, realtime::RealtimeRegistry, snapshot::WorkerRegistry},
            grpc::multimodal::MultimodalConfigRegistry,
        },
        server::http_client,
    };

    /// An `AppContext` over `router_config` and `worker_registry`: in-memory
    /// storage, a generous rate limiter, no gateway, worker monitor or WASM
    /// manager, the policy registry the configuration's policy, routing-key
    /// override and pairing mode select, the given tool parsers (a parser override is validated
    /// against them when they are given), and an uninitialized worker job
    /// queue: submitted jobs are accepted and queued but never processed, so a
    /// test observes what a step or a pass submits without spawning workflow
    /// background tasks.
    pub(crate) fn app_context_for_tests(
        router_config: RouterConfig,
        worker_registry: Arc<WorkerRegistry>,
        tool_parser_factory: Option<ToolParserFactory>,
    ) -> Arc<AppContext> {
        let job_queue = Arc::new(std::sync::OnceLock::new());

        Arc::new(AppContext {
            gateway_auth: AuthConfig::new(None),
            client: http_client(),
            router_config: router_config.clone(),
            rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
            rate_limit_manager: None,
            worker_registry: Arc::clone(&worker_registry),
            engine_connector: BackendConnector::shared(),
            prefill_admission: None,
            policy_registry: Arc::new(
                PolicyRegistry::with_override(
                    router_config.policy.clone(),
                    router_config.routing_key_override.clone(),
                )
                .with_pd_pairing_mode(router_config.pd_pairing_mode),
            ),
            reasoning_parser_factory: None,
            tool_parser_factory,
            gateway: None,
            response_storage: Arc::new(smg_data_connector::MemoryResponseStorage::new()),
            conversation_storage: Arc::new(smg_data_connector::MemoryConversationStorage::new()),
            conversation_item_storage: Arc::new(
                smg_data_connector::MemoryConversationItemStorage::new(),
            ),
            worker_monitor: None,
            configured_reasoning_parser: None,
            configured_tool_parser: None,
            worker_job_queue: Arc::clone(&job_queue),
            workflow_engines: Arc::new(std::sync::OnceLock::new()),
            mcp_orchestrator: Arc::new(std::sync::OnceLock::new()),
            mcp_format_registry: openai_bridge::FormatRegistry::new(),
            tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
            multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
            tokenizer_load_failures: Arc::default(),
            worker_loaded_tokenizers: Arc::default(),
            wasm_manager: None,
            worker_client_cache: Arc::new(WorkerHttpClientCache::new(
                router_config.worker_http_client_settings(),
            )),
            registration_settings: router_config.registration_settings(),
            worker_service: Arc::new(WorkerService::new(
                worker_registry,
                job_queue,
                router_config,
            )),
            inflight_tracker: InFlightRequestTracker::new(),
            kv_event_monitor: None,
            rl: None,
            realtime_registry: Arc::new(RealtimeRegistry::new()),
            webrtc_bind_addr: None,
            webrtc_stun_server: None,
        })
    }

    /// An `AppContext` with an uninitialized worker job queue: submitted jobs are
    /// accepted and queued but never processed, so a test observes what is
    /// *submitted* without spawning workflow background tasks.
    /// [`live_app_context`] is the one whose jobs run.
    pub(crate) fn create_test_app_context() -> Arc<AppContext> {
        let router_config = RouterConfig::builder()
            .worker_startup_timeout_secs(1)
            .build_unchecked();

        // Note: Using uninitialized queue for tests to avoid spawning background workers
        // Jobs submitted during tests will queue but not be processed
        app_context_for_tests(router_config, Arc::new(WorkerRegistry::new()), None)
    }

    /// An `AppContext` whose jobs run: a live job queue and workflow engines, so
    /// a test observes what the registry *ends up with* once its jobs run.
    /// Regular routing with round-robin, a five-second startup window for an
    /// engine slow to answer its probes, and no drain wait on removal.
    /// [`create_test_app_context`] is the one whose jobs never run.
    pub(crate) fn live_app_context() -> Arc<AppContext> {
        use std::sync::OnceLock;

        use llm_tokenizer::registry::TokenizerRegistry;
        use smg_data_connector::{
            MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        };

        use crate::{
            config::{HealthCheckConfig, PolicyConfig, RoutingMode},
            workflow::{JobQueue, JobQueueConfig, WorkflowEngines},
        };

        let config = RouterConfig::builder()
            .mode(RoutingMode::Regular {
                worker_urls: vec![],
            })
            .policy(PolicyConfig::RoundRobin)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .health_check_config(HealthCheckConfig {
                drain_settle_secs: 0,
                ..HealthCheckConfig::default()
            })
            .build_unchecked();
        let context = Arc::new(
            AppContext::builder()
                .router_config(config.clone())
                .client(http_client())
                .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
                .worker_registry(Arc::new(WorkerRegistry::new()))
                .policy_registry(Arc::new(PolicyRegistry::new(config.policy.clone())))
                .response_storage(Arc::new(MemoryResponseStorage::new()))
                .conversation_storage(Arc::new(MemoryConversationStorage::new()))
                .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
                .worker_job_queue(Arc::new(OnceLock::new()))
                .workflow_engines(Arc::new(OnceLock::new()))
                .mcp_orchestrator(Arc::new(OnceLock::new()))
                .build()
                .unwrap(),
        );
        context
            .worker_job_queue
            .set(JobQueue::new(
                JobQueueConfig::default(),
                Arc::downgrade(&context),
            ))
            .ok();
        context
            .workflow_engines
            .set(WorkflowEngines::new(&context.router_config))
            .ok();
        context
    }
}
