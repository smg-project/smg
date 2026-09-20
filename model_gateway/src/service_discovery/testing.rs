//! Shared test fixtures for the discovery modules.

use std::sync::Arc;

use crate::{
    app_context::AppContext,
    middleware::AuthConfig,
    routers::{common::openai_bridge, grpc::multimodal::MultimodalConfigRegistry},
};

/// An `AppContext` with an uninitialized worker job queue: submitted jobs are
/// accepted and queued but never processed, so tests observe what discovery
/// *submits* without spawning workflow background tasks.
pub(super) fn create_test_app_context() -> Arc<AppContext> {
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
        client: reqwest::Client::new(),
        router_config: router_config.clone(),
        rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
        rate_limit_manager: None,
        worker_registry: worker_registry.clone(),
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
        wasm_manager: None,
        worker_client_cache: Arc::new(crate::worker::WorkerHttpClientCache::new(&router_config)),
        worker_service: Arc::new(WorkerService::new(
            worker_registry,
            worker_job_queue,
            router_config,
        )),
        inflight_tracker: InFlightRequestTracker::new(),
        kv_event_monitor: None,
        remote_index: None,
        unindexed_cache_aware_leg: false,
        rl: None,
        realtime_registry: Arc::new(RealtimeRegistry::new()),
        webrtc_bind_addr: None,
        webrtc_stun_server: None,
    })
}
