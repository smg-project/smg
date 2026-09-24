//! Context types for Anthropic router
//!
//! - `RouterContext`: shared infrastructure (HTTP client, worker registry, MCP orchestrator)
//! - `RequestContext`: per-request input (request body, headers, model ID)

use std::{sync::Arc, time::Duration};

use axum::http::HeaderMap;
use openai_protocol::messages::CreateMessageRequest;
use smg_mcp::{McpOrchestrator, McpServerBinding};

use crate::{
    openai_bridge::FormatRegistry,
    tenant::TenantRequestMeta,
    worker::{ExternalWorker, WorkerSource},
};

/// Shared context passed to all Anthropic handler functions.
#[derive(Clone)]
pub(crate) struct RouterContext {
    pub mcp_orchestrator: Arc<McpOrchestrator>,
    pub mcp_format_registry: FormatRegistry,
    pub http_client: reqwest::Client,
    pub workers: Arc<dyn WorkerSource>,
    pub request_timeout: Duration,
}

/// Per-request input that flows through handler functions.
pub(crate) struct RequestContext {
    pub request: CreateMessageRequest,
    pub headers: Option<HeaderMap>,
    pub model_id: String,
    /// Explicit tenant identity resolved once at the HTTP boundary.
    #[expect(
        dead_code,
        reason = "tenant-scoped Anthropic consumers land after the shared serving-path plumbing"
    )]
    pub tenant_request_meta: TenantRequestMeta,
    /// Connected MCP server keys, present when the request includes `mcp_toolset` tools.
    pub mcp_servers: Option<Vec<McpServerBinding>>,
    /// Worker selected once in `route_messages`, reused for all iterations.
    pub worker: Arc<dyn ExternalWorker>,
}
