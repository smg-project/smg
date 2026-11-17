//! RequestBuilding step.
//!
//! Transition: BuildRequest → NonStreamRequest | StreamRequestWithTool | StreamRequest

use axum::response::Response;
use openai_protocol::common::to_value_exact;
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

    // Serialize the original request as the upstream payload, every number
    // as the client wrote it.
    let mut payload = to_value_exact(ctx.input.original_request.as_ref()).map_err(|e| {
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

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use openai_protocol::interactions::InteractionsRequest;
    use serde_json::json;

    use super::request_building;
    use crate::{
        gemini::context::{RequestContext, SharedComponents},
        tenant::{TenantKey, TenantRequestMeta},
        test_support::{http_client, StubWorkers},
    };

    /// The step's upstream payload for a request with this body, routed to
    /// `gemini-routed`.
    async fn payload_of(body: serde_json::Value) -> serde_json::Value {
        let request: InteractionsRequest = serde_json::from_value(body).expect("a request");
        let components = SharedComponents {
            client: http_client(),
            workers: Arc::new(StubWorkers::default()),
            request_timeout: Duration::from_secs(5),
        };
        let mut ctx = RequestContext::new(
            Arc::new(request),
            None,
            Some("gemini-routed".to_string()),
            TenantRequestMeta::new(TenantKey::new("t")),
            Arc::new(components),
        );
        assert!(request_building(&mut ctx).await.is_ok());
        ctx.processing.payload.expect("a payload")
    }

    /// The upstream payload carries the request's fields the gateway does not
    /// model as the client wrote them, and the gateway's own overrides still
    /// apply: the model the route names, and `store: false` for a model
    /// request.
    #[tokio::test]
    async fn the_payload_keeps_the_fields_the_gateway_does_not_model() {
        let payload = payload_of(json!({
            "model": "m", "input": "hi", "store": false,
            "continuation_token": "tok_1",
            "safety_settings": [{"type": "harassment", "threshold": "block_none"}],
            "service_tier": "flex"
        }))
        .await;
        assert_eq!(payload["continuation_token"], "tok_1");
        assert_eq!(payload["safety_settings"][0]["threshold"], "block_none");
        assert_eq!(payload["service_tier"], "flex");
        assert_eq!(payload["model"], "gemini-routed");
        assert_eq!(payload["store"], false);
    }

    /// A generation or agent config written in camelCase, keys and all,
    /// reaches the upstream with each of its values, in snake_case.
    #[tokio::test]
    async fn a_camel_case_config_reaches_the_payload_with_its_values() {
        let payload = payload_of(json!({
            "model": "m", "input": "hi", "store": false,
            "generationConfig": {"maxOutputTokens": 1, "stopSequences": ["END"]}
        }))
        .await;
        assert_eq!(
            payload["generation_config"],
            json!({"max_output_tokens": 1, "stop_sequences": ["END"]})
        );
        assert!(payload.get("generationConfig").is_none(), "{payload}");

        let payload = payload_of(json!({
            "agent": "deep-research", "input": "hi",
            "agentConfig": {"type": "deep-research", "thinkingSummaries": "auto"}
        }))
        .await;
        assert_eq!(
            payload["agent_config"],
            json!({"type": "deep-research", "thinking_summaries": "auto"})
        );
    }
}
