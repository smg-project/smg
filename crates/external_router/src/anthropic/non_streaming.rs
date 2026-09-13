//! Non-streaming processor for Anthropic Messages API
//!
//! Handles both plain (no MCP) and MCP tool loop paths for
//! non-streaming requests, composing worker and mcp primitives.

use std::time::Instant;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use openai_protocol::messages::{ContentBlock, InputContent, InputMessage, Message, Role};
use smg_mcp::McpToolSession;
use tracing::warn;

use super::{
    context::{RequestContext, RouterContext},
    mcp, worker,
};
use crate::{error, mcp_utils::DEFAULT_MAX_ITERATIONS, metrics::Metrics};

/// Execute a non-streaming Messages API request, handling both
/// plain and MCP tool loop paths.
pub(crate) async fn execute(router: &RouterContext, mut req_ctx: RequestContext) -> Response {
    if req_ctx.mcp_servers.is_none() {
        return match send_one_request(router, &req_ctx).await {
            Ok(message) => (StatusCode::OK, Json(message)).into_response(),
            Err(response) => response,
        };
    }

    // MCP tool loop path
    let session_id = format!("msg_{}", uuid::Uuid::now_v7());
    let mcp_servers = req_ctx.mcp_servers.take().unwrap_or_default();
    let session = McpToolSession::new(&router.mcp_orchestrator, mcp_servers, &session_id);

    // Inject MCP tools into the request as regular tools
    mcp::inject_mcp_tools_into_request(&mut req_ctx.request, &session);

    let mut all_mcp_calls: Vec<mcp::McpToolCall> = Vec::new();
    let mut prior_content_blocks: Vec<ContentBlock> = Vec::new();

    let mut message = match send_one_request(router, &req_ctx).await {
        Ok(m) => m,
        Err(response) => return response,
    };

    for _iteration in 0..DEFAULT_MAX_ITERATIONS {
        Metrics::record_mcp_tool_iteration(&req_ctx.model_id);

        let result = mcp::IterationResult::from_message(&message);
        match mcp::process_iteration(&result, &session, &req_ctx.model_id).await {
            mcp::ToolLoopAction::Done => {
                let final_message = mcp::rebuild_response_with_mcp_blocks(
                    message,
                    &all_mcp_calls,
                    &prior_content_blocks,
                );
                return (StatusCode::OK, Json(final_message)).into_response();
            }
            mcp::ToolLoopAction::Error(msg) => {
                return error::bad_gateway("mcp_tool_loop_error", msg);
            }
            mcp::ToolLoopAction::Continue(cont) => {
                // Collect content blocks from this iteration for the final response
                prior_content_blocks.extend(message.content.clone());
                all_mcp_calls.extend(cont.mcp_calls);
                req_ctx.request.messages.push(InputMessage {
                    role: Role::Assistant,
                    content: InputContent::Blocks(cont.assistant_blocks),
                });
                req_ctx.request.messages.push(InputMessage {
                    role: Role::User,
                    content: InputContent::Blocks(cont.tool_result_blocks),
                });
            }
        }

        message = match send_one_request(router, &req_ctx).await {
            Ok(m) => m,
            Err(response) => return response,
        };
    }

    // Max iterations — check if the last response completed naturally
    let result = mcp::IterationResult::from_message(&message);
    if result.tool_use_blocks.is_empty() {
        let final_message =
            mcp::rebuild_response_with_mcp_blocks(message, &all_mcp_calls, &prior_content_blocks);
        return (StatusCode::OK, Json(final_message)).into_response();
    }

    warn!(
        "Non-streaming MCP tool loop exceeded max iterations ({})",
        DEFAULT_MAX_ITERATIONS
    );
    error::bad_gateway(
        "mcp_max_iterations",
        format!("MCP tool loop exceeded maximum iterations ({DEFAULT_MAX_ITERATIONS})"),
    )
}

/// Send a single non-streaming request to a worker and parse the response.
async fn send_one_request(
    router: &RouterContext,
    req_ctx: &RequestContext,
) -> Result<Message, Response> {
    let model_id = &req_ctx.model_id;
    let start_time = Instant::now();

    worker::record_router_request(model_id, false);
    let (url, req_headers) = worker::build_request(&*req_ctx.worker, req_ctx.headers.as_ref());
    let response = worker::send_request(
        &router.http_client,
        &url,
        &req_headers,
        &req_ctx.request,
        router.request_timeout,
    )
    .await?;

    if !response.status().is_success() {
        return Err(worker::handle_error_response(response, model_id, start_time).await);
    }

    worker::parse_response(response, model_id, start_time).await
}
