//! RequestBuilding step.
//!
//! Transition: BuildRequest → NonStreamRequest | StreamRequestWithTool | StreamRequest

use axum::response::Response;
use serde_json::Value;

use crate::{
    error,
    gemini::{
        context::RequestContext,
        state::{RequestState, StepResult},
    },
};

/// Build the upstream request payload and create an MCP session if needed.
///
/// ## Payload transformations
/// 1. Override `model` with `model_id` if set (e.g. from URL path).
/// 2. Set `store: false` — the gateway handles persistence, not the worker.
///
/// ## MCP handling (Phase 2)
/// When the request contains `InteractionsTool::McpServer` tools, this step will:
/// 1. Validate MCP server connectivity via `ensure_request_mcp_client()`.
/// 2. Create an `McpToolSession` for the request lifetime.
/// 3. Convert MCP tool definitions to function tool definitions in the payload
///    (via `prepare_mcp_tools_as_functions`), so the upstream worker sees only
///    function tools.
pub(crate) async fn request_building(ctx: &mut RequestContext) -> Result<StepResult, Response> {
    // Phase 1: interaction persistence is not implemented yet.
    // Reject store=true for model requests (agent requests proxy store to upstream).
    if ctx.input.original_request.agent.is_none() && ctx.input.original_request.store {
        return Err(error::not_implemented(
            "not_implemented",
            "Interaction persistence (store=true) is not yet implemented for model requests",
        ));
    }

    // Serialize the original request as the upstream payload.
    let mut payload = serde_json::to_value(ctx.input.original_request.as_ref()).map_err(|e| {
        tracing::error!(error = %e, "Failed to serialize Gemini interactions request");
        error::internal_error("internal_error", "Failed to build upstream request payload")
    })?;

    // Apply payload transformations for the upstream worker.
    transform_payload(&mut payload, ctx);

    // TODO (Phase 2): Check if tools contains any InteractionsTool::McpServer entries.
    // If MCP tools present:
    //   a. Call ensure_request_mcp_client() to validate servers.
    //   b. Create McpToolSession.
    //   c. Call prepare_mcp_tools_as_functions(&mut payload, &session).
    //   d. Store session and tool_loop_state on ctx.
    let has_mcp_tools = false;

    ctx.processing.payload = Some(payload);

    if ctx.input.original_request.stream {
        if has_mcp_tools {
            ctx.state = RequestState::StreamRequestWithTool;
        } else {
            ctx.state = RequestState::StreamRequest;
        }
    } else {
        ctx.state = RequestState::NonStreamRequest;
    }
    Ok(StepResult::Continue)
}

/// Apply transformations to the upstream payload.
fn transform_payload(payload: &mut Value, ctx: &RequestContext) {
    let Some(obj) = payload.as_object_mut() else {
        return;
    };

    // Override model if this is a model request.
    if let Some(model_id) = &ctx.input.model_id {
        if ctx.input.original_request.agent.is_none() {
            obj.insert("model".to_string(), Value::String(model_id.clone()));
        }
    }

    // For model requests, the gateway handles persistence — tell the worker not to store.
    // For agent requests, the upstream API requires store: true (agent+background mode), so we leave it unchanged.
    if ctx.input.original_request.agent.is_none() {
        obj.insert("store".to_string(), Value::Bool(false));
    }
}
