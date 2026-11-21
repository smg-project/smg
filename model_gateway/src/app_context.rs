use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use dashmap::DashSet;
use llm_tokenizer::registry::TokenizerRegistry;
use openai_protocol::profile::ProviderProfile;
use parking_lot::Mutex;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use reqwest::Client;
use smg_data_connector::{
    create_storage, ConversationItemStorage, ConversationStorage, ResponseStorage,
    StorageFactoryConfig,
};
use smg_external_router::realtime::webrtc::default_bind_addr;
use smg_mcp::McpOrchestrator;
use smg_policy_controller::{stages::Catalog, store::MemoryStore};
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
    policies::{kv_index::event_monitor::KvEventMonitor, PolicyRegistry},
    policy_controller::{
        self,
        engine::PolicyEngine,
        feed::Feed,
        follower::{Follower, Settings},
        operators::Operators,
        InProcess, Started,
    },
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
    /// The policy controller's store, where the controller was picked to
    /// decide for this process; `None` under the classic policies.
    pub policy_store: Option<Arc<MemoryStore>>,
    /// The operators' door over the policy store: the control plane's
    /// policy document routes stand on it. `None` under the classic
    /// policies.
    pub(crate) policy_operators: Option<Arc<Operators>>,
    /// The worker watch, owned by the context until server teardown.
    pub(crate) policy_follower: Option<Arc<Mutex<Option<Follower>>>>,
    /// The policy engine. Routers ask it for placement wherever
    /// the policy store is present.
    pub(crate) policy_engine: Option<Arc<PolicyEngine>>,
    /// The membership feed over [`Self::policy_store`], present wherever the store is.
    pub(crate) policy_feed: Option<Feed>,
    /// What the start left a gateway that decides in its own process (the
    /// setting on and no controller's address): the policy engine the
    /// policies of [`Self::policy_registry`] ask, and the document it
    /// decides under as written. Nothing else of the policy controller is
    /// built there: the store, its door, its feed, the follower and
    /// [`Self::policy_engine`] are a gateway's that follows a controller.
    /// `None` with the setting off and under an address.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the control plane's policy routes read what the start left"
        )
    )]
    pub(crate) in_process: Option<InProcess>,
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
    policy_store: Option<Arc<MemoryStore>>,
    /// The operators' door over the store, built with it.
    policy_operators: Option<Arc<Operators>>,
    /// The membership feed, built with the policy store.
    policy_feed: Option<Feed>,
    /// The policy engine over the feed's holder, built with the feed.
    policy_engine: Option<Arc<PolicyEngine>>,
    policy_follower: Option<Arc<Mutex<Option<Follower>>>>,
    /// What a start in this process left: the policy engine the policy
    /// registry is built with.
    in_process: Option<InProcess>,
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

    /// Whether a policy controller supplies this gateway's workers: the
    /// context was built under a controller's address and holds the
    /// follower of its watch. The start asks it at each step such a
    /// gateway leaves to the controller: the startup registration, the
    /// health checks and the discovery.
    pub(crate) fn follows_controller(&self) -> bool {
        self.policy_follower.is_some()
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

    pub(crate) fn from_config_with_extensions(
        router_config: RouterConfig,
        request_timeout_secs: u64,
        webrtc_bind_addr: Option<std::net::IpAddr>,
        webrtc_stun_server: Option<String>,
        stages: Option<Catalog>,
        connector: Option<Arc<dyn EngineConnector>>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self, String>> + Send>> {
        Box::pin(async move {
            Box::pin(AppContextBuilder::from_config_with_extensions(
                router_config,
                request_timeout_secs,
                webrtc_bind_addr,
                webrtc_stun_server,
                stages,
                connector,
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
            policy_store: None,
            policy_operators: None,
            policy_feed: None,
            policy_engine: None,
            policy_follower: None,
            in_process: None,
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
            policy_store: self.policy_store,
            policy_operators: self.policy_operators,
            policy_follower: self.policy_follower,
            policy_engine: self.policy_engine,
            policy_feed: self.policy_feed,
            in_process: self.in_process,
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
        stages: Option<Catalog>,
        connector: Option<Arc<dyn EngineConnector>>,
    ) -> Result<Self, String> {
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
            .with_policy_controller(&router_config, stages)?
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

    /// Start the policy controller as the configuration picks it.
    ///
    /// With no controller's address the gateway decides in its own
    /// process: the start builds the policy engine of the document in
    /// force and nothing else, and the policy registry is built with it
    /// ([`Self::with_policy_registry`]). The store, its door, its feed, the
    /// follower and the gateway's own engine object stay unset, so every
    /// site that asks for one of them takes its classic arm: the worker
    /// registry, the health checks, the load monitor and the worker
    /// workflows are the classic ones.
    ///
    /// Under a controller's address it starts the policy store with its
    /// operators' door, its membership feed, its policy engine with the
    /// engine's timer and the follower of that controller's workers. A
    /// start the document's check refuses stops this one, by name.
    fn with_policy_controller(
        mut self,
        config: &RouterConfig,
        stages: Option<Catalog>,
    ) -> Result<Self, String> {
        if config.policy_controller_address.is_none() {
            self.in_process = policy_controller::start_in_process(config, stages)
                .map_err(|refusal| refusal.to_string())?;
            return Ok(self);
        }
        let started =
            policy_controller::start_with(config, stages).map_err(|refusal| refusal.to_string())?;
        if let Some(Started { store, door }) = started {
            let registry = self.worker_registry.as_ref().ok_or_else(|| {
                "worker_registry must be set before the policy controller".to_string()
            })?;
            let controller = config
                .policy_controller_address
                .as_ref()
                .map(|address| Settings::new(address.clone(), config))
                .transpose()?;
            let connector = Arc::clone(&self.engine_connector);
            let (feed, follower) =
                policy_controller::feeds(&store, registry, connector, controller);
            let policy_engine = Arc::new(PolicyEngine::new(
                Arc::clone(&store),
                Arc::clone(&feed.objects),
            ));
            // The door's clock runs from here on. The timer ends behind
            // the policy engine, so its handle detaches.
            let _timer = policy_controller::upkeep::start(&policy_engine);
            self.policy_feed = Some(feed);
            self.policy_engine = Some(policy_engine);
            self.policy_follower = follower.map(|follower| Arc::new(Mutex::new(Some(follower))));
            self.policy_operators = Some(Arc::new(Operators::new(Arc::clone(&store), door)));
            self.policy_store = Some(store);
        }
        Ok(self)
    }

    /// Create policy registry: the classic one, whose policies ask the
    /// policy engine where the gateway decides in its own process. The
    /// policy controller's step comes first
    /// ([`Self::with_policy_controller`]): it leaves the engine this
    /// registry is built with, and a registry built ahead of it would be
    /// built without one.
    fn with_policy_registry(mut self, config: &RouterConfig) -> Self {
        let policy = config.policy.clone();
        let routing_key_override = config.routing_key_override.clone();
        let registry = match &self.in_process {
            Some(started) => PolicyRegistry::with_policy_engine(
                policy,
                routing_key_override,
                Arc::clone(&started.engine),
                started.origin,
            ),
            None => PolicyRegistry::with_override(policy, routing_key_override),
        };
        self.policy_registry = Some(Arc::new(
            registry.with_pd_pairing_mode(config.pd_pairing_mode),
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
        if config.policy_controller_address.is_some() {
            info!("Skipping worker load monitoring because --policy-controller-address is set");
            return Ok(self);
        }
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
        // Where the controller was picked the collector says its facts to
        // the feed's sink, which writes the store, and no classic policy
        // reads a load. Else it keeps the classic sink, and its snapshot
        // feeds every policy that consumes it; the monitor polls every group
        // by default, conditionally under `--disable-load-monitoring`.
        let monitor = match &self.policy_feed {
            Some(feed) => Arc::new(WorkerMonitor::with_sink(
                registry,
                feed.load_sink(Duration::from_secs(interval.max(1))),
                interval,
                engine_metrics,
                conditional,
            )),
            None => {
                let monitor = Arc::new(WorkerMonitor::new(
                    registry,
                    Arc::clone(&policy_registry),
                    interval,
                    engine_metrics,
                    conditional,
                ));
                policy_registry.set_load_receiver(Some(monitor.subscribe()));
                monitor
            }
        };
        // The overload shed advertises the poll interval as Retry-After — the
        // veto cannot clear between polls.
        overload::set_shed_retry_after_secs(config.load_monitor_interval_secs);
        if let Some(registry) = self.worker_registry.as_ref() {
            registry.set_overload_shed(config.worker_overload_shed);
        }
        // PD dispatch waits here, not in the decode engine's queue, when the
        // pair's running window is full.
        pd_admission::set_pd_admission_wait_secs(config.pd_admission_wait_secs);
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
    /// A gateway has one whatever its classic policies: a worker under a
    /// cache-aware policy streams its KV events into the model's index, every
    /// other gRPC worker streams its load records alone, so load-aware
    /// routing prices a worker by a record at the engine's cadence rather
    /// than by the last poll (`start_kv_stream` picks the kind per worker,
    /// for a registration and for an update that rebuilt the worker alike;
    /// HTTP and ZMQ workers are never subscribed). The monitor itself
    /// is cheap (empty maps) and stays dormant until workers are added; a
    /// model without an events subscription has no index and no index
    /// gauges.
    ///
    /// Under the policy controller's feed none is built: the feed's own
    /// follow reads the workers' cache events into the store's index and
    /// their load records into the worker monitor, and each reader of this
    /// monitor takes `None`.
    fn with_kv_event_monitor(mut self) -> Self {
        // One collector a gateway: under the feed the store's follow is it.
        if self.policy_feed.is_some() {
            return self;
        }
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
    use smg_policy_controller::document::PolicySpec;
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
        policies::{DocumentOrigin, EnginePolicy},
        routers::{
            common::{
                prefill_admission::{test_support::select_worker, PrefillAdmissionAttempt},
                snapshot::test_support::worker_of,
            },
            grpc::backend_client::BasicWorkerBuilder,
        },
    };

    /// With the setting on and no controller's address the gateway decides
    /// in its own process. The context holds the policy engine and nothing
    /// else of the policy controller: no store, no door, no feed, no
    /// follower and no engine object of the gateway's own. Its policy
    /// registry is the classic one, and the default policy it built is the
    /// object that asks the engine. With the setting off it holds neither.
    #[tokio::test]
    async fn a_context_in_process_holds_the_policy_engine_and_nothing_else_of_the_controller() {
        for policy_controller in [false, true] {
            let config = RouterConfig {
                policy_controller,
                policy: PolicyConfig::RoundRobin,
                ..Default::default()
            };
            let context = AppContext::from_config(config, 10, None, None)
                .await
                .unwrap();
            assert!(context.policy_follower.is_none());
            assert!(context.policy_store.is_none());
            assert!(context.policy_operators.is_none());
            assert!(context.policy_feed.is_none());
            assert!(context.policy_engine.is_none());
            assert!(!context.follows_controller());
            assert_eq!(context.in_process.is_some(), policy_controller);
            let default = context.policy_registry.get_default_policy();
            assert_eq!(default.name(), "round_robin");
            assert_eq!(default.as_any().is::<EnginePolicy>(), policy_controller);
            assert!(context.worker_monitor.is_some());
        }
    }

    /// A policy the engine does not pick by itself yet stays the classic
    /// policy of its setting under the setting, and the start is not
    /// refused: the document the routing settings mean has that policy
    /// behind each place.
    #[tokio::test]
    async fn a_setting_the_engine_does_not_pick_yet_starts_on_its_classic_policy() {
        let config = RouterConfig {
            policy_controller: true,
            policy: PolicyConfig::Random,
            ..Default::default()
        };
        let context = AppContext::from_config(config, 10, None, None)
            .await
            .unwrap();
        let default = context.policy_registry.get_default_policy();
        assert_eq!(default.name(), "random");
        assert!(!default.as_any().is::<EnginePolicy>());

        // What the control plane's read of the document answers from, with
        // no store: the engine's document under the number a gateway's
        // first document takes, its YAML as the start wrote it, the time
        // it took effect and the stages it was validated over.
        let started = context
            .in_process
            .as_ref()
            .expect("the start left its engine");
        assert_eq!(started.origin, DocumentOrigin::Settings);
        assert_eq!(started.engine.spec().generation(), 1);
        let read: PolicySpec = serde_yaml::from_str(&started.written).expect("YAML");
        assert!(read.profiles.contains_key("random"));
        assert!(started.applied <= std::time::SystemTime::now());
        assert!(Arc::strong_count(&started.stages) >= 1);
    }

    /// A document its operator supplied has no setting behind a place. One
    /// whose every place the engine picks by itself starts, and its
    /// bindings place the models. One that holds a place the engine does
    /// not decide in this process yet is refused at the start, by the
    /// profile, the stage of it that asks more, the binding and the phase.
    #[tokio::test]
    async fn a_supplied_document_starts_where_the_engine_picks_each_of_its_places() {
        // A document that names no number: the first of its gateway.
        let supplied = |name: &str| {
            let mut document = PolicySpec::preset(name).expect("a built-in preset");
            document.generation = 0;
            RouterConfig {
                policy_controller: true,
                policy_spec: Some(document),
                ..Default::default()
            }
        };
        let context = AppContext::from_config(supplied("round_robin"), 10, None, None)
            .await
            .unwrap();
        assert!(context.policy_store.is_none());
        let default = context.policy_registry.get_default_policy();
        assert!(default.as_any().is::<EnginePolicy>());
        // It takes the number 1, as at the operators' door.
        let started = context
            .in_process
            .as_ref()
            .expect("the start left its engine");
        assert_eq!(started.origin, DocumentOrigin::Supplied);
        assert_eq!(started.engine.spec().generation(), 1);

        let started = AppContext::from_config(supplied("least_load"), 10, None, None).await;
        let Err(refused) = started else {
            panic!("the start is not refused");
        };
        assert!(
            refused.contains("profile `least_load` is not decided in this process yet"),
            "{refused}"
        );
        assert!(refused.contains("it holds the scorer `"), "{refused}");
        assert!(refused.contains("(binding `*`, phase single)"), "{refused}");
    }

    /// The parts a start builds for a context or leaves out, by name, and
    /// whether this one holds each.
    fn parts(context: &AppContext) -> Vec<(&'static str, bool)> {
        vec![
            ("rate_limiter", context.rate_limiter.is_some()),
            ("rate_limit_manager", context.rate_limit_manager.is_some()),
            (
                "reasoning_parser_factory",
                context.reasoning_parser_factory.is_some(),
            ),
            ("tool_parser_factory", context.tool_parser_factory.is_some()),
            ("prefill_admission", context.prefill_admission.is_some()),
            ("gateway", context.gateway.is_some()),
            ("worker_monitor", context.worker_monitor.is_some()),
            ("wasm_manager", context.wasm_manager.is_some()),
            ("kv_event_monitor", context.kv_event_monitor.is_some()),
            ("policy_store", context.policy_store.is_some()),
            ("policy_operators", context.policy_operators.is_some()),
            ("policy_follower", context.policy_follower.is_some()),
            ("policy_engine", context.policy_engine.is_some()),
            ("policy_feed", context.policy_feed.is_some()),
            ("rl", context.rl.is_some()),
        ]
    }

    /// A gateway that decides in its own process is built as a gateway with
    /// the setting off is: the same storage, and the same objects handed to
    /// the same places. Under one configuration, the setting off against
    /// on: the context holds and leaves out the same parts, but for what
    /// the start left (the policy engine); and a classic policy is handed
    /// the load feed and the cache-event monitor under both. The worker
    /// workflows read nothing of the context: the server builds the one
    /// line of each under both.
    #[tokio::test]
    async fn a_context_in_process_is_built_as_with_the_setting_off() {
        use crate::policies::as_cache_aware;

        // Cache-aware is a policy the engine does not pick by itself yet:
        // the classic policy of the setting decides under both, with what
        // a context hands a classic policy.
        let built = |policy_controller: bool| async move {
            let config = RouterConfig {
                policy_controller,
                ..config_with_policy(PolicyConfig::cache_aware(0.5, 32, 1.1, 30, 1000))
            };
            AppContext::from_config(config, 10, None, None)
                .await
                .expect("a context")
        };
        let (off, on) = (built(false).await, built(true).await);

        assert_eq!(parts(&off), parts(&on));
        assert!(off.in_process.is_none());
        assert!(on.in_process.is_some());

        for context in [&off, &on] {
            let policy = context.policy_registry.get_default_policy();
            let cache_aware =
                as_cache_aware(policy.as_ref()).expect("the classic cache-aware policy");
            assert!(cache_aware.has_load_receiver_for_test());
            assert!(cache_aware.kv_event_monitor_is_set_for_test());
        }
    }

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

    /// Under the policy controller the monitor says its facts to the feed's
    /// sink: a pushed record lands on the store's row, and no classic
    /// policy is handed the monitor's load feed.
    #[tokio::test]
    async fn under_the_feed_the_collectors_sink_is_the_feeds() {
        use smg_policy_controller::{
            engine::{Role, WireModes},
            store::{HealthPolicy, LoadSink, Maker, WorkerFacts},
            worker_host::{Expect, Wrote},
        };
        use smg_worker::common_proto::EngineLoad;

        use crate::policies::as_cache_aware;

        const URL: &str = "http://worker:8000";
        let config = RouterConfig {
            policy_controller: true,
            ..config_with_policy(PolicyConfig::cache_aware(0.5, 32, 1.1, 30, 1000))
        };
        let builder = AppContextBuilder::new()
            .with_client(&config, 5)
            .expect("client builds")
            .with_worker_registry()
            .with_policy_registry(&config)
            .with_policy_controller(&config, None)
            .expect("the policy controller starts")
            .with_worker_monitor(&config)
            .expect("worker monitor builds");
        let feed = builder.policy_feed.as_ref().expect("the store has a feed");
        let store = builder.policy_store.as_ref().expect("the store started");
        let monitor = builder.worker_monitor.as_ref().expect("the monitor");
        let registry = builder.worker_registry.as_ref().expect("the registry");

        let facts = WorkerFacts {
            key: URL.to_owned(),
            engine: URL.to_owned(),
            role: Role::Single,
            wire_modes: WireModes::HTTP,
            models: vec!["model-a".to_owned()],
            dp_size: 1,
            health: HealthPolicy {
                probes: false,
                ..HealthPolicy::default()
            },
            ..WorkerFacts::default()
        };
        let Ok(Wrote::Registered(row)) =
            feed.door
                .found(&Maker::Control, Expect::Nothing, URL, &|_| facts.clone())
        else {
            panic!("the worker registers at the feed's door");
        };
        let worker = registry.get_by_url(URL).expect("the row's worker stands");
        let record = EngineLoad {
            waiting_requests: 6,
            ..EngineLoad::default()
        };
        monitor.apply_pushed_load(&worker, 0, &record, std::time::Instant::now());
        let sample = LoadSink::load(&**store, row, 0).expect("the row holds the pushed sample");
        assert_eq!(sample.row.queued, 6);

        let policy = builder
            .policy_registry
            .as_ref()
            .expect("policy registry set")
            .get_default_policy();
        let cache_aware = as_cache_aware(policy.as_ref()).expect("default policy is cache-aware");
        assert!(!cache_aware.has_load_receiver_for_test());
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

    /// Under the policy controller's feed the classic monitor is not built:
    /// the context holds none, and a cache-aware policy is handed none.
    #[tokio::test]
    async fn under_the_feed_the_classic_cache_event_monitor_is_not_built() {
        use crate::policies::as_cache_aware;

        let config = RouterConfig {
            policy_controller: true,
            ..config_with_policy(PolicyConfig::cache_aware(0.5, 32, 1.1, 30, 1000))
        };
        let builder = AppContextBuilder::new()
            .with_client(&config, 5)
            .expect("client builds")
            .with_worker_registry()
            .with_policy_registry(&config)
            .with_policy_controller(&config, None)
            .expect("the policy controller starts")
            .with_kv_event_monitor();
        assert!(builder.policy_feed.is_some());
        assert!(builder.kv_event_monitor.is_none());
        let policy = builder
            .policy_registry
            .as_ref()
            .expect("policy registry set")
            .get_default_policy();
        let cache_aware = as_cache_aware(policy.as_ref()).expect("default policy is cache-aware");
        assert!(!cache_aware.kv_event_monitor_is_set_for_test());
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
            policy_store: None,
            policy_operators: None,
            policy_follower: None,
            policy_engine: None,
            policy_feed: None,
            in_process: None,
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
