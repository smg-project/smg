//! Non-streaming response handling for OpenAI-compatible responses
//!
//! This module handles non-streaming Responses API requests with MCP tool support.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::Value;
use smg_mcp::McpToolSession;
use tracing::warn;

use super::utils::{patch_response_with_request_metadata, restore_original_tools};
use crate::{
    error,
    header_utils::{extract_forwardable_request_headers, ApiProvider},
    mcp_utils::{ensure_request_mcp_client, request_uses_mcp_routing},
    openai::{
        context::{PayloadState, RequestContext, ResponsesPayloadState},
        mcp::{execute_tool_loop, prepare_mcp_tools_as_functions, ToolLoopExecutionContext},
    },
    openai_bridge,
    persistence_utils::persist_conversation_items,
};

/// Handle a non-streaming responses request
pub async fn handle_non_streaming_response(mut ctx: RequestContext) -> Response {
    let payload_state = match ctx.state.payload.take() {
        Some(ps) => ps,
        None => {
            return error::internal_error("internal_error", "Payload not prepared");
        }
    };

    let PayloadState {
        json: mut payload,
        url,
    } = payload_state;
    let ResponsesPayloadState {
        previous_response_id,
        existing_mcp_list_tools_labels,
    } = ctx.take_responses_payload().unwrap_or_default();

    let original_body = match ctx.responses_request() {
        Some(r) => r,
        None => {
            return error::internal_error("internal_error", "Expected responses request");
        }
    };
    let worker = match ctx.worker() {
        Some(w) => w.clone(),
        None => {
            return error::internal_error("internal_error", "Worker not selected");
        }
    };
    // Only MCP-laden requests need the orchestrator and format registry;
    // without this narrowing, plain non-MCP requests would 500 in
    // deployments that run the gateway without MCP wiring. A registry-less
    // MCP request still hard-fails — silent fallback would mis-route
    // hosted tools.
    let mcp_routing = match original_body.tools.as_deref() {
        Some(tools) if request_uses_mcp_routing(tools) => {
            let Some(mcp_orchestrator) = ctx.components.mcp_orchestrator() else {
                return error::internal_error(
                    "internal_error",
                    "MCP orchestrator required for requests carrying MCP/builtin tools",
                );
            };
            let Some(registry) = ctx.components.mcp_format_registry() else {
                return error::internal_error(
                    "internal_error",
                    "MCP format registry required for requests carrying MCP/builtin tools",
                );
            };
            ensure_request_mcp_client(mcp_orchestrator, registry, tools)
                .await
                .map(|servers| (servers, mcp_orchestrator.clone(), registry.clone()))
        }
        _ => None,
    };

    let mut response_json: Value;

    if let Some((mcp_servers, mcp_orchestrator, mcp_format_registry)) = mcp_routing {
        let session_request_id = original_body
            .request_id
            .clone()
            .unwrap_or_else(|| format!("req_{}", uuid::Uuid::now_v7()));
        let forwarded_headers = extract_forwardable_request_headers(ctx.headers());
        let mut session = McpToolSession::new_with_headers(
            &mcp_orchestrator,
            mcp_servers,
            &session_request_id,
            forwarded_headers,
        );
        if let Some(tools) = original_body.tools.as_deref() {
            openai_bridge::configure_response_tools_approval(&mut session, tools);
        }
        prepare_mcp_tools_as_functions(&mut payload, &session);

        match execute_tool_loop(
            ctx.components.client(),
            &url,
            ctx.headers(),
            worker.api_key(),
            payload,
            ToolLoopExecutionContext {
                original_body,
                existing_mcp_list_tools_labels: &existing_mcp_list_tools_labels,
                session: &session,
                format_registry: &mcp_format_registry,
            },
        )
        .await
        {
            Ok(resp) => {
                worker.record_outcome(200);
                response_json = resp;
            }
            Err(err) => {
                worker.record_outcome(502);
                return error::internal_error("upstream_error", err);
            }
        }

        restore_original_tools(&mut response_json, original_body, Some(&session));
    } else {
        let mut request_builder = ctx.components.client().post(&url).json(&payload);
        let provider = ApiProvider::from_url(&url);
        let auth_header = provider.extract_auth_header(ctx.headers(), worker.api_key());
        request_builder = provider.apply_headers(request_builder, auth_header.as_ref());

        let response = match request_builder.send().await {
            Ok(r) => r,
            Err(e) => {
                worker.record_outcome(502);
                tracing::error!(
                    url = %url,
                    error = %e,
                    "Failed to forward request to OpenAI"
                );
                return error::bad_gateway(
                    "upstream_error",
                    format!("Failed to forward request to OpenAI: {e}"),
                );
            }
        };

        let status = response.status();
        worker.record_outcome(status.as_u16());

        if !status.is_success() {
            let status =
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let body = response.text().await.unwrap_or_default();
            let body = error::sanitize_error_body(&body);
            return (status, body).into_response();
        }

        response_json = match response.json::<Value>().await {
            Ok(r) => r,
            Err(e) => {
                return error::internal_error(
                    "parse_error",
                    format!("Failed to parse upstream response: {e}"),
                );
            }
        };

        restore_original_tools(&mut response_json, original_body, None);
    }
    patch_response_with_request_metadata(
        &mut response_json,
        original_body,
        previous_response_id.as_deref(),
    );

    if let (Some(conv_storage), Some(item_storage), Some(resp_storage)) = (
        ctx.components.conversation_storage(),
        ctx.components.conversation_item_storage(),
        ctx.components.response_storage(),
    ) {
        if let Err(err) = persist_conversation_items(
            conv_storage.clone(),
            item_storage.clone(),
            resp_storage.clone(),
            &response_json,
            original_body,
            ctx.storage_request_context.clone(),
        )
        .await
        {
            warn!("Failed to persist conversation items: {}", err);
        }
    } else {
        warn!("Storage not configured, skipping conversation persistence");
    }

    (StatusCode::OK, Json(response_json)).into_response()
}
