use std::sync::Arc;

use super::{AppContext, ToolParserFactory};
use crate::{
    config::RouterConfig,
    middleware::{AuthConfig, TokenBucket},
    observability::inflight_tracker::InFlightRequestTracker,
    routers::{
        common::{openai_bridge, realtime::RealtimeRegistry},
        grpc::multimodal::MultimodalConfigRegistry,
    },
    worker::{WorkerRegistry, WorkerService},
};

/// An `AppContext` over `router_config` and `worker_registry`: in-memory
/// storage, a generous rate limiter, no gateway, worker monitor or WASM
/// manager, the policy registry the configuration's policy and routing-key
/// override select, the given tool parsers (a parser override is validated
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
        client: crate::clients::http_client(),
        router_config: router_config.clone(),
        rate_limiter: Some(Arc::new(TokenBucket::new(1000, 1000))),
        rate_limit_manager: None,
        worker_registry: Arc::clone(&worker_registry),
        prefill_admission: None,
        policy_registry: Arc::new(crate::policies::PolicyRegistry::with_override(
            router_config.policy.clone(),
            router_config.routing_key_override.clone(),
        )),
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
        wasm_manager: None,
        worker_client_cache: Arc::new(crate::worker::WorkerHttpClientCache::new(&router_config)),
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
