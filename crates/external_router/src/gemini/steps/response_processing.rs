//! ResponseProcessing step (non-streaming).
//!
//! Terminal step: returns `StepResult::Response` directly.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::Value;

use crate::gemini::{context::RequestContext, state::StepResult};

/// Finalize the non-streaming response and return it to the client.
///
/// This step produces the terminal HTTP `Response` via `StepResult::Response`.
/// The driver returns it directly.
pub(crate) async fn response_processing(ctx: &mut RequestContext) -> Result<StepResult, Response> {
    let Some(mut response_json) = ctx.processing.upstream_response.take() else {
        return Err((StatusCode::BAD_GATEWAY, "no upstream response received").into_response());
    };

    // TODO (Phase 2): If MCP session exists, inject MCP metadata (mcp_list_tools, mcp_call items)
    //                  and restore original tools (convert function tools back to MCP format).

    patch_response_metadata(&mut response_json, ctx);

    // TODO (Phase 3): Persist interaction to interaction_storage if store is true.
    // TODO (Phase 3): Generate interaction ID when upstream doesn't provide one (store=false).

    Ok(StepResult::Response(
        (StatusCode::OK, Json(response_json)).into_response(),
    ))
}

/// Patch response JSON with metadata from the original request.
fn patch_response_metadata(response_json: &mut Value, ctx: &RequestContext) {
    let Some(obj) = response_json.as_object_mut() else {
        return;
    };

    let req = &ctx.input.original_request;

    // Ensure the response includes the model or agent from the original request.
    if let Some(agent) = &req.agent {
        obj.insert("agent".to_string(), Value::String(agent.clone()));
    } else if let Some(model) = ctx.input.model_id.as_deref().or(req.model.as_deref()) {
        obj.insert("model".to_string(), Value::String(model.to_string()));
    }

    // Always set store=false for model requests in request building step, so set the original value back in response
    if req.agent.is_none() && req.store {
        obj.insert("store".to_string(), Value::Bool(req.store));
    }

    // Set previous_interaction_id if missing
    if let Some(prev_id) = &req.previous_interaction_id {
        if is_missing_or_empty(obj.get("previous_interaction_id")) {
            obj.insert(
                "previous_interaction_id".to_string(),
                Value::String(prev_id.clone()),
            );
        }
    }
}

/// Check if a JSON value is missing, null, or an empty string.
fn is_missing_or_empty(value: Option<&Value>) -> bool {
    match value {
        None => true,
        Some(v) => v.is_null() || v.as_str().is_some_and(|s| s.is_empty()),
    }
}
