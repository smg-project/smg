//! Anthropic API router implementation
//!
//! This router handles Anthropic-specific APIs including:
//! - Messages API (/v1/messages) with SSE streaming
//! - Tool use and MCP integration
//! - Extended thinking and prompt caching

use std::{any::Any, collections::HashMap, fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::{http::HeaderMap, response::Response};
use openai_protocol::messages::CreateMessageRequest;
use tracing::{error, info};

use super::{
    context::{RequestContext, RouterContext},
    mcp, non_streaming, streaming,
};
use crate::{
    app_context::AppContext,
    middleware::TenantRequestMeta,
    routers::{
        common::{
            header_utils, mcp_utils,
            worker_selection::{SelectWorkerRequest, WorkerSelector},
        },
        error::bad_gateway,
        RouterTrait,
    },
    worker::ProviderType,
};

/// Router for Anthropic-specific APIs
///
/// Handles Anthropic's Messages API with support for:
/// - Streaming and non-streaming responses
/// - Tool use via MCP
/// - Extended thinking
/// - Prompt caching
/// - Citations
pub struct AnthropicRouter {
    context: Arc<AppContext>,
    router_ctx: RouterContext,
}

impl fmt::Debug for AnthropicRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicRouter")
            .field("context", &"<AppContext>")
            .finish()
    }
}

impl AnthropicRouter {
    pub fn new(context: Arc<AppContext>) -> Result<Self, String> {
        let request_timeout = Duration::from_secs(context.router_config.request_timeout_secs);
        let mcp_orchestrator = context
            .mcp_orchestrator
            .get()
            .ok_or_else(|| "Anthropic router requires MCP orchestrator".to_string())?
            .clone();

        let router_ctx = RouterContext {
            mcp_orchestrator,
            mcp_format_registry: context.mcp_format_registry.clone(),
            http_client: context.client.clone(),
            worker_registry: context.worker_registry.clone(),
            request_timeout,
        };

        Ok(Self {
            context,
            router_ctx,
        })
    }

    pub fn context(&self) -> &Arc<AppContext> {
        &self.context
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.context.client
    }
}

#[async_trait]
impl RouterTrait for AnthropicRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn route_messages(
        &self,
        headers: Option<&HeaderMap>,
        tenant_meta: &TenantRequestMeta,
        body: CreateMessageRequest,
        model_id: &str,
    ) -> Response {
        let request = body;
        let headers_owned = headers.cloned();

        let mcp_servers = if header_utils::is_smg_mcp_enabled(headers) && request.has_mcp_toolset()
        {
            // Build per-server allowed tools from McpToolset entries in tools array.
            let toolset_allowed = mcp::collect_allowed_tools_per_server(request.tools.as_ref());

            let inputs: Vec<mcp_utils::McpServerInput> = request
                .mcp_server_configs()
                .unwrap_or_default()
                .iter()
                .map(|server| mcp_utils::McpServerInput {
                    label: server.name.clone(),
                    url: Some(server.url.clone()),
                    authorization: server.authorization_token.clone(),
                    headers: HashMap::new(),
                    allowed_tools: toolset_allowed.get(&server.name).and_then(|v| v.clone()),
                })
                .collect();

            match mcp_utils::ensure_mcp_servers(
                &self.router_ctx.mcp_orchestrator,
                &self.router_ctx.mcp_format_registry,
                &inputs,
                &[],
            )
            .await
            {
                Some(servers) => {
                    info!(
                        server_count = servers.len(),
                        "MCP: connected to MCP servers"
                    );
                    Some(servers)
                }
                None => {
                    error!("Failed to connect to any MCP servers");
                    return bad_gateway(
                        "mcp_connection_failed",
                        "Failed to connect to MCP servers. Check server URLs and authorization.",
                    );
                }
            }
        } else {
            None
        };

        let is_streaming = request.stream.unwrap_or(false);
        info!(
            model = %model_id,
            streaming = %is_streaming,
            mcp = %mcp_servers.is_some(),
            "Processing Messages API request"
        );

        let selector = WorkerSelector::new(
            &self.router_ctx.worker_registry,
            &self.router_ctx.http_client,
        );
        let selected_worker = match selector
            .select_worker(&SelectWorkerRequest {
                model_id,
                headers,
                provider: Some(ProviderType::Anthropic),
                ..Default::default()
            })
            .await
        {
            Ok(w) => w,
            Err(resp) => return resp,
        };

        let req_ctx = RequestContext {
            request,
            headers: headers_owned,
            model_id: model_id.to_string(),
            tenant_request_meta: tenant_meta.clone(),
            mcp_servers,
            worker: selected_worker,
        };

        if is_streaming {
            streaming::execute(&self.router_ctx, req_ctx).await
        } else {
            non_streaming::execute(&self.router_ctx, req_ctx).await
        }
    }

    fn router_type(&self) -> &'static str {
        "anthropic"
    }
}
