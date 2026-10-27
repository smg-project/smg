//! Test support of the gRPC router: the `AppContext` a test constructs its
//! router over, with the components `GrpcRouter::new` needs (the parser
//! factories always; an initialized MCP orchestrator, which the Responses
//! contexts of the regular and the disaggregated modes read).

use std::sync::{Arc, OnceLock};

use llm_tokenizer::registry::TokenizerRegistry;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use smg_data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use smg_mcp::{McpConfig, McpOrchestrator};
use tool_parser::ParserFactory as ToolParserFactory;

use crate::{
    app_context::AppContext,
    config::{ParserChoice, PolicyConfig, RouterConfig, RouterConfigBuilder, RoutingMode},
    policies::PolicyRegistry,
    worker::WorkerRegistry,
};

/// Minimal `AppContext` for constructing a gRPC router in `mode`. Regular and
/// PD both serve /v1/responses, so the MCP orchestrator must be initialized.
pub(crate) async fn grpc_ctx(mode: RoutingMode) -> Arc<AppContext> {
    grpc_ctx_with_parsers(mode, ParserChoice::default(), None).await
}

/// [`grpc_ctx`] with the gateway's `--parser` choice and its
/// `--tool-call-parser` name, which under Symphony pins the table that
/// answers to it for every model the registry does not know.
pub(crate) async fn grpc_ctx_with_parsers(
    mode: RoutingMode,
    parser: ParserChoice,
    tool_parser: Option<&str>,
) -> Arc<AppContext> {
    let mcp_orchestrator = mcp_orchestrator().await;
    let mut builder = grpc_config(mode).parser(parser);
    if let Some(name) = tool_parser {
        builder = builder.tool_call_parser(name);
    }
    grpc_ctx_from(&builder.build_unchecked(), mcp_orchestrator)
}

/// The gRPC router configuration the tests start from: `mode` over the gRPC
/// connection mode, the random policy, the HTTP request bound at 60 s and the
/// gRPC one at its default.
pub(crate) fn grpc_config(mode: RoutingMode) -> RouterConfigBuilder {
    RouterConfig::builder()
        .mode(mode)
        .grpc_connection()
        .policy(PolicyConfig::Random)
        .host("127.0.0.1")
        .port(3001)
        .max_payload_size(1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(10)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
}

/// The initialized MCP orchestrator every test context carries: PD serves
/// /v1/responses, which needs it. Awaited before the configuration is built,
/// so no context future holds a `RouterConfig` across the await.
pub(crate) async fn mcp_orchestrator() -> Arc<OnceLock<Arc<McpOrchestrator>>> {
    let mcp_orchestrator = Arc::new(OnceLock::new());
    mcp_orchestrator
        .set(Arc::new(
            McpOrchestrator::new(McpConfig::default())
                .await
                .expect("mcp orchestrator"),
        ))
        .ok();
    mcp_orchestrator
}

/// Minimal `AppContext` over `config` for constructing a gRPC router.
pub(crate) fn grpc_ctx_from(
    config: &RouterConfig,
    mcp_orchestrator: Arc<OnceLock<Arc<McpOrchestrator>>>,
) -> Arc<AppContext> {
    Arc::new(
        AppContext::builder()
            .router_config(config.clone())
            .client(crate::clients::http_client())
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .reasoning_parser_factory(Some(ReasoningParserFactory::new()))
            .tool_parser_factory(Some(ToolParserFactory::new()))
            .worker_registry(Arc::new(WorkerRegistry::new()))
            .policy_registry(Arc::new(PolicyRegistry::new(config.policy.clone())))
            .response_storage(Arc::new(MemoryResponseStorage::new()))
            .conversation_storage(Arc::new(MemoryConversationStorage::new()))
            .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
            .worker_job_queue(Arc::new(OnceLock::new()))
            .workflow_engines(Arc::new(OnceLock::new()))
            .mcp_orchestrator(mcp_orchestrator)
            .build()
            .expect("app context"),
    )
}
