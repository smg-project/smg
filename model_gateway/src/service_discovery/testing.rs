//! Shared test fixtures for the discovery modules.

use std::sync::Arc;

use crate::{
    app_context::AppContext,
    middleware::AuthConfig,
    routers::{common::openai_bridge, grpc::multimodal::MultimodalConfigRegistry},
};

/// An `AppContext` with an uninitialized worker job queue: submitted jobs are
/// accepted and queued but never processed, so tests observe what discovery
/// *submits* without spawning workflow background tasks. [`live_app_context`]
/// is the one whose jobs run.
pub(crate) fn create_test_app_context() -> Arc<AppContext> {
    use crate::{
        config::RouterConfig, middleware::TokenBucket,
        observability::inflight_tracker::InFlightRequestTracker,
        routers::common::realtime::RealtimeRegistry, worker::WorkerService,
    };

    let router_config = RouterConfig::builder()
        .worker_startup_timeout_secs(1)
        .build_unchecked();

    let worker_registry = Arc::new(crate::worker::WorkerRegistry::new());
    let worker_job_queue = Arc::new(std::sync::OnceLock::new());

    // Note: Using uninitialized queue for tests to avoid spawning background workers
    // Jobs submitted during tests will queue but not be processed
    Arc::new(AppContext {
        gateway_auth: AuthConfig::new(None),
        client: crate::clients::http_client(),
        router_config: router_config.clone(),
        rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
        rate_limit_manager: None,
        worker_registry: worker_registry.clone(),
        prefill_admission: None,
        policy_registry: Arc::new(crate::policies::PolicyRegistry::with_override(
            router_config.policy.clone(),
            router_config.routing_key_override.clone(),
        )),
        reasoning_parser_factory: None,
        tool_parser_factory: None,
        gateway: None,
        response_storage: Arc::new(smg_data_connector::MemoryResponseStorage::new()),
        conversation_storage: Arc::new(smg_data_connector::MemoryConversationStorage::new()),
        conversation_item_storage: Arc::new(
            smg_data_connector::MemoryConversationItemStorage::new(),
        ),
        worker_monitor: None,
        configured_reasoning_parser: None,
        configured_tool_parser: None,
        worker_job_queue: worker_job_queue.clone(),
        workflow_engines: Arc::new(std::sync::OnceLock::new()),
        mcp_orchestrator: Arc::new(std::sync::OnceLock::new()),
        mcp_format_registry: openai_bridge::FormatRegistry::new(),
        tokenizer_registry: Arc::new(llm_tokenizer::registry::TokenizerRegistry::new()),
        multimodal_config_registry: Arc::new(MultimodalConfigRegistry::new()),
        tokenizer_load_failures: Arc::default(),
        wasm_manager: None,
        worker_client_cache: Arc::new(crate::worker::WorkerHttpClientCache::new(&router_config)),
        worker_service: Arc::new(WorkerService::new(
            worker_registry,
            worker_job_queue,
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

/// An `AppContext` whose jobs run: a live job queue and workflow engines, so
/// tests observe what the registry *ends up with* after discovery's passes.
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
        config::{HealthCheckConfig, PolicyConfig, RouterConfig, RoutingMode},
        policies::PolicyRegistry,
        worker::WorkerRegistry,
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
            .client(crate::clients::http_client())
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
