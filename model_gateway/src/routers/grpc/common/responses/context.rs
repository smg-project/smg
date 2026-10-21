//! Shared context for /v1/responses endpoint handlers
//!
//! This context is used by both regular and harmony response implementations.

use std::sync::Arc;

use smg_data_connector::{
    ConversationItemStorage, ConversationStorage, RequestContext as StorageRequestContext,
    ResponseStorage,
};
use smg_mcp::McpOrchestrator;

use super::BackgroundResponses;
use crate::routers::{
    common::openai_bridge::FormatRegistry,
    grpc::{context::SharedComponents, pipeline::RequestPipeline},
};

/// Context for /v1/responses endpoint
///
/// Used by both regular and harmony implementations.
/// All fields are Arc/shared references, so cloning this context is cheap.
#[derive(Clone)]
pub(crate) struct ResponsesContext {
    /// Chat pipeline for executing requests
    pub pipeline: Arc<RequestPipeline>,

    /// Shared components (tokenizer, parsers)
    pub components: Arc<SharedComponents>,

    /// Response storage backend
    pub response_storage: Arc<dyn ResponseStorage>,

    /// Conversation storage backend
    pub conversation_storage: Arc<dyn ConversationStorage>,

    /// Conversation item storage backend
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,

    /// MCP orchestrator for tool support
    pub mcp_orchestrator: Arc<McpOrchestrator>,

    pub mcp_format_registry: FormatRegistry,

    /// Storage hook request context extracted from HTTP headers by middleware.
    pub request_context: Option<StorageRequestContext>,

    /// The background responses in flight (shared by every context of the
    /// router, so one cap and one cancel registry serve all of them).
    pub background: Arc<BackgroundResponses>,
}

impl ResponsesContext {
    /// Create a new responses context.
    #[expect(
        clippy::too_many_arguments,
        reason = "responses context assembles shared pipeline + storage handles in one place"
    )]
    pub fn new(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        response_storage: Arc<dyn ResponseStorage>,
        conversation_storage: Arc<dyn ConversationStorage>,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
        mcp_orchestrator: Arc<McpOrchestrator>,
        mcp_format_registry: FormatRegistry,
        request_context: Option<StorageRequestContext>,
        background: Arc<BackgroundResponses>,
    ) -> Self {
        Self {
            pipeline,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            mcp_orchestrator,
            mcp_format_registry,
            request_context,
            background,
        }
    }
}
