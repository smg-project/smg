use std::sync::{Arc, OnceLock};

use axum::Router;
use llm_tokenizer::registry::TokenizerRegistry;
use reqwest::Client;
use smg::{
    app_context::AppContext,
    config::RouterConfig,
    health,
    middleware::{AuthConfig, TokenBucket},
    policies::PolicyRegistry,
    routers::RouterTrait,
    server::{build_app, AppState},
    worker::{
        BasicWorkerBuilder, ModelCard, RuntimeType, Worker, WorkerMonitor, WorkerRegistry,
        WorkerType,
    },
};
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use smg_mcp::{McpConfig, McpOrchestrator};

/// Create a test Axum application using the actual server's build_app function
#[expect(
    clippy::unwrap_used,
    reason = "test helper - panicking on failure is intentional"
)]
pub fn create_test_app(
    router: Arc<dyn RouterTrait>,
    client: Client,
    router_config: &RouterConfig,
) -> Router {
    // Initialize rate limiter
    let rate_limiter = match router_config.max_concurrent_requests {
        n if n <= 0 => None,
        n => {
            let rate_limit_tokens = router_config
                .rate_limit_tokens_per_second
                .filter(|&t| t > 0)
                .unwrap_or(0);
            Some(Arc::new(TokenBucket::new(
                n as usize,
                rate_limit_tokens as usize,
            )))
        }
    };

    // Initialize registries
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

    // Initialize storage backends
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

    // Initialize the worker monitor with the same interval the
    // production builder uses so tests exercise the real polling
    // cadence.
    let worker_monitor = Some(Arc::new(WorkerMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        router_config.load_monitor_interval_secs,
        router_config.engine_metrics,
        router_config.disable_load_monitoring,
    )));

    // Create empty OnceLock for worker job queue and workflow engines
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());

    // Create AppContext using builder pattern
    let app_context = Arc::new(
        AppContext::builder()
            .router_config(router_config.clone())
            .client(client)
            .rate_limiter(rate_limiter)
            .tokenizer_registry(Arc::new(TokenizerRegistry::new())) // tokenizer
            .reasoning_parser_factory(None) // reasoning_parser_factory
            .tool_parser_factory(None) // tool_parser_factory
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .response_storage(response_storage)
            .conversation_storage(conversation_storage)
            .conversation_item_storage(conversation_item_storage)
            .worker_monitor(worker_monitor)
            .worker_job_queue(worker_job_queue)
            .workflow_engines(workflow_engines)
            .build()
            .unwrap(),
    );

    // Mirror production wiring: start the WorkerMonitor event loop so
    // tests that depend on event-driven group reconciliation exercise
    // the real code path instead of an inert monitor.
    if let Some(monitor) = &app_context.worker_monitor {
        monitor.start_event_loop();
    }

    // Create AppState with the test router and context
    let app_state = Arc::new(AppState {
        router,
        probe_state: start_probe_state(&app_context),
        context: app_context,
        admission_queue: None,
        router_manager: None,
        mesh_handler: None,
        mesh_adapters: None,
    });

    // Configure request ID headers (use defaults if not specified)
    let request_id_headers = router_config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    let serving_auth_config = AuthConfig::with_tenant_keys(
        router_config.api_key.clone(),
        &router_config.tenant_api_keys,
    );
    let admin_auth_config = AuthConfig::new(router_config.api_key.clone());

    // Use the actual server's build_app function
    #[expect(
        clippy::expect_used,
        reason = "test helper assumes router config is already validated"
    )]
    build_app(
        app_state,
        serving_auth_config,
        admin_auth_config,
        None, // No control plane auth for tests
        router_config.max_payload_size,
        request_id_headers,
        router_config.cors_allowed_origins.clone(),
    )
    .expect("valid tenant resolution config")
}

/// Mirror production wiring: cached probe state plus its readiness
/// maintainer, so `/readiness` reflects worker registry state in tests.
/// The initial snapshot is computed synchronously, so workers registered
/// before app creation are visible immediately; later mutations arrive via
/// the registry event subscription. Must be called from a tokio runtime.
fn start_probe_state(app_context: &Arc<AppContext>) -> Arc<health::ProbeState> {
    let probe_state = health::ProbeState::new(app_context.inflight_tracker.clone());
    let _maintainer = health::spawn_readiness_maintainer(
        probe_state.clone(),
        app_context.worker_registry.clone(),
        app_context.tokenizer_registry.clone(),
        app_context.router_config.clone(),
    );
    probe_state
}

/// Create a test Axum application with an existing AppContext
pub fn create_test_app_with_context(
    router: Arc<dyn RouterTrait>,
    app_context: Arc<AppContext>,
) -> Router {
    // Create AppState with the test router and context
    let app_state = Arc::new(AppState {
        router,
        probe_state: start_probe_state(&app_context),
        context: app_context.clone(),
        admission_queue: None,
        router_manager: None,
        mesh_handler: None,
        mesh_adapters: None,
    });

    // Get config from the context
    let router_config = &app_context.router_config;

    // Configure request ID headers (use defaults if not specified)
    let request_id_headers = router_config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    let serving_auth_config = AuthConfig::with_tenant_keys(
        router_config.api_key.clone(),
        &router_config.tenant_api_keys,
    );
    let admin_auth_config = AuthConfig::new(router_config.api_key.clone());

    // Use the actual server's build_app function
    #[expect(
        clippy::expect_used,
        reason = "test helper assumes router config is already validated"
    )]
    build_app(
        app_state,
        serving_auth_config,
        admin_auth_config,
        None, // No control plane auth for tests
        router_config.max_payload_size,
        request_id_headers,
        router_config.cors_allowed_origins.clone(),
    )
    .expect("valid tenant resolution config")
}

/// Create a minimal test AppContext for unit tests
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
pub async fn create_test_app_context() -> Arc<AppContext> {
    let router_config = RouterConfig::default();
    let client = Client::new();

    // Initialize empty OnceLocks
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());

    // Initialize MCP orchestrator with empty config
    let mcp_orchestrator_lock = Arc::new(OnceLock::new());
    let empty_config = McpConfig {
        servers: vec![],
        pool: Default::default(),
        proxy: None,
        warmup: vec![],
        inventory: Default::default(),
        policy: Default::default(),
    };
    let mcp_orchestrator = McpOrchestrator::new(empty_config)
        .await
        .expect("Failed to create MCP orchestrator");
    mcp_orchestrator_lock.set(Arc::new(mcp_orchestrator)).ok();

    // Initialize registries
    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

    // Initialize storage backends
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

    Arc::new(
        AppContext::builder()
            .router_config(router_config)
            .client(client)
            .rate_limiter(None)
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .reasoning_parser_factory(None)
            .tool_parser_factory(None)
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .response_storage(response_storage)
            .conversation_storage(conversation_storage)
            .conversation_item_storage(conversation_item_storage)
            .worker_monitor(None)
            .worker_job_queue(worker_job_queue)
            .workflow_engines(workflow_engines)
            .mcp_orchestrator(mcp_orchestrator_lock)
            .build()
            .unwrap(),
    )
}

/// Register an external worker (OpenAI-compatible API endpoint) in the test AppContext.
///
/// This is used by tests that need to test the OpenAI router, which expects
/// workers to be registered in the WorkerRegistry before routing requests.
///
/// # Arguments
/// * `ctx` - The AppContext to register the worker in
/// * `url` - The base URL of the external API endpoint
/// * `models` - Optional list of model IDs this worker supports. If empty, uses "gpt-3.5-turbo" as default.
pub fn register_external_worker(ctx: &Arc<AppContext>, url: &str, models: Option<Vec<&str>>) {
    let model_list: Vec<ModelCard> = models
        .unwrap_or_else(|| vec!["gpt-3.5-turbo"])
        .into_iter()
        .map(ModelCard::new)
        .collect();

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .runtime_type(RuntimeType::External)
            .models(model_list)
            .health_config(openai_protocol::worker::HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );

    ctx.worker_registry.register(worker);
}

/// Register an external worker with a custom model card that has aliases.
///
/// # Arguments
/// * `ctx` - The AppContext to register the worker in
/// * `url` - The base URL of the external API endpoint
/// * `model_card` - A fully configured ModelCard with aliases, provider, etc.
pub fn register_external_worker_with_card(ctx: &Arc<AppContext>, url: &str, model_card: ModelCard) {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(WorkerType::Regular)
            .runtime_type(RuntimeType::External)
            .model(model_card)
            .health_config(openai_protocol::worker::HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );

    ctx.worker_registry.register(worker);
}
