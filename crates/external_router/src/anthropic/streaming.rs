//! Streaming processor for Anthropic Messages API
//!
//! Handles both passthrough (no MCP) and MCP tool loop streaming paths,
//! composing worker, sse, and mcp primitives.

use std::{io, time::Instant};

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::Response,
};
use bytes::Bytes;
use openai_protocol::messages::{InputContent, InputMessage, Role};
use smg_mcp::McpToolSession;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use super::{
    context::{RequestContext, RouterContext},
    mcp, sse, worker,
};
use crate::{
    error as router_error,
    error::extract_error_code_from_response,
    mcp_utils::DEFAULT_MAX_ITERATIONS,
    metrics::{metrics_labels, Metrics},
    sse::SseEncoder,
};

/// Channel buffer size for SSE events sent to the client.
const SSE_CHANNEL_SIZE: usize = 128;

/// Execute a streaming Messages API request, handling both
/// passthrough (no MCP) and MCP tool loop paths.
pub(crate) async fn execute(router: &RouterContext, req_ctx: RequestContext) -> Response {
    if req_ctx.mcp_servers.is_some() {
        return execute_mcp_streaming(router, req_ctx);
    }
    execute_passthrough(router, &req_ctx).await
}

// ============================================================================
// Passthrough streaming (no MCP)
// ============================================================================

/// Direct streaming: send request and wrap in SSE response.
async fn execute_passthrough(router: &RouterContext, req_ctx: &RequestContext) -> Response {
    let model_id = &req_ctx.model_id;
    let start_time = Instant::now();

    worker::record_router_request(model_id, true);
    let (url, req_headers) = worker::build_request(&*req_ctx.worker, req_ctx.headers.as_ref());
    let response = match worker::send_request(
        &router.http_client,
        &url,
        &req_headers,
        &req_ctx.request,
        router.request_timeout,
    )
    .await
    {
        Ok(r) => r,
        Err(resp) => return resp,
    };

    build_streaming_response(response, model_id, start_time).await
}

/// Build a streaming SSE response.
async fn build_streaming_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
) -> Response {
    let status = response.status();

    if !status.is_success() {
        return build_streaming_error_response(response, model_id, start_time).await;
    }

    debug!(model = %model_id, status = %status, "Starting streaming response");

    Metrics::record_router_duration(
        metrics_labels::ROUTER_HTTP,
        metrics_labels::BACKEND_EXTERNAL,
        metrics_labels::CONNECTION_HTTP,
        model_id,
        "messages",
        start_time.elapsed(),
    );

    let headers = response.headers().clone();
    let body = Body::from_stream(response.bytes_stream());

    sse::build_sse_response(status, headers, body)
}

/// Handle a non-success streaming response.
async fn build_streaming_error_response(
    response: reqwest::Response,
    model_id: &str,
    start_time: Instant,
) -> Response {
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    // If it's an SSE error stream, pass it through
    if content_type
        .to_ascii_lowercase()
        .contains("text/event-stream")
    {
        warn!(model = %model_id, status = %status, "Streaming error response (SSE)");

        Metrics::record_router_duration(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            start_time.elapsed(),
        );
        Metrics::record_router_error(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_EXTERNAL,
            metrics_labels::CONNECTION_HTTP,
            model_id,
            "messages",
            "streaming_error",
        );

        let headers = response.headers().clone();
        let body = Body::from_stream(response.bytes_stream());

        return sse::build_sse_response(status, headers, body);
    }

    // Non-SSE error: read the body and return a proper error response
    worker::handle_error_response(response, model_id, start_time).await
}

// ============================================================================
// MCP streaming (tool loop)
// ============================================================================

/// Spawn the MCP tool loop in a background task and return an SSE response.
fn execute_mcp_streaming(router: &RouterContext, req_ctx: RequestContext) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(SSE_CHANNEL_SIZE);

    let router = router.clone();

    #[expect(
        clippy::disallowed_methods,
        reason = "fire-and-forget streaming task; gateway shutdown need not wait for individual MCP tool loops"
    )]
    tokio::spawn(async move {
        if let Err(e) = run_tool_loop(tx.clone(), router, req_ctx).await {
            if e == sse::CLIENT_DISCONNECTED_ERROR {
                debug!(error = %e, "Streaming tool loop ended: client disconnected");
                return;
            }
            warn!(error = %e, "Streaming tool loop failed");
            let mut enc = SseEncoder::new();
            let _ = sse::send_error(&tx, &mut enc, &e).await;
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap_or_else(|e| {
            error!("Failed to build streaming response: {}", e);
            router_error::internal_error("response_build_failed", "Failed to build response")
        })
}

/// Run the MCP tool loop, sending SSE events to the client via `tx`.
async fn run_tool_loop(
    tx: mpsc::Sender<Result<Bytes, io::Error>>,
    router: RouterContext,
    mut req_ctx: RequestContext,
) -> Result<(), String> {
    let session_id = format!("msg_{}", uuid::Uuid::now_v7());
    let mcp_servers = req_ctx.mcp_servers.take().unwrap_or_default();
    let session = McpToolSession::new(&router.mcp_orchestrator, mcp_servers, &session_id);

    // Inject MCP tools into the request as regular tools
    mcp::inject_mcp_tools_into_request(&mut req_ctx.request, &session);

    let mut global_index: u32 = 0;
    let mut total_input_tokens: u32 = 0;
    let mut total_output_tokens: u32 = 0;
    let mut is_first_iteration = true;
    // Reusable SSE encoder shared across every event emitted for this stream.
    let mut encoder = SseEncoder::new();

    for _iteration in 0..DEFAULT_MAX_ITERATIONS {
        Metrics::record_mcp_tool_iteration(&req_ctx.model_id);

        let response = send_streaming_request(&router, &req_ctx).await?;

        if !response.status().is_success() {
            return Err(format!(
                "Worker returned error status: {}",
                response.status()
            ));
        }

        // Consume the upstream SSE stream
        let result = sse::consume_and_forward(
            &tx,
            &mut encoder,
            response,
            &mut global_index,
            is_first_iteration,
            |name| session.resolve_tool_server_label(name),
        )
        .await;

        let consumed = result?;
        is_first_iteration = false;

        // Accumulate usage
        if let Some(ref usage) = consumed.usage {
            total_output_tokens += usage.output_tokens;
            if let Some(input) = usage.input_tokens {
                total_input_tokens += input;
            }
        }

        // Check if we should continue the tool loop
        match mcp::process_iteration(&consumed.iteration, &session, &req_ctx.model_id).await {
            mcp::ToolLoopAction::Done => {
                sse::emit_final(
                    &tx,
                    &mut encoder,
                    consumed.iteration.stop_reason.as_ref(),
                    total_input_tokens,
                    total_output_tokens,
                )
                .await;
                return Ok(());
            }
            mcp::ToolLoopAction::Error(msg) => {
                return Err(msg);
            }
            mcp::ToolLoopAction::Continue(cont) => {
                // Emit mcp_tool_result events for each completed tool call
                for call in &cont.mcp_calls {
                    if !sse::emit_mcp_tool_result(&tx, &mut encoder, call, &mut global_index).await
                    {
                        return Ok(());
                    }
                }

                // Append assistant + tool_result messages for next iteration
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
    }

    // Max iterations exceeded
    warn!(
        "Streaming MCP tool loop exceeded max iterations ({})",
        DEFAULT_MAX_ITERATIONS
    );
    let error_msg = format!("MCP tool loop exceeded maximum iterations ({DEFAULT_MAX_ITERATIONS})");
    let _ = sse::send_error(&tx, &mut encoder, &error_msg).await;
    Ok(())
}

/// Send a streaming request to the pre-selected worker.
async fn send_streaming_request(
    router: &RouterContext,
    req_ctx: &RequestContext,
) -> Result<reqwest::Response, String> {
    worker::record_router_request(&req_ctx.model_id, true);

    let (url, req_headers) = worker::build_request(&*req_ctx.worker, req_ctx.headers.as_ref());
    let response = worker::send_request(
        &router.http_client,
        &url,
        &req_headers,
        &req_ctx.request,
        router.request_timeout,
    )
    .await
    .map_err(|e| {
        format!(
            "Failed to send request to worker ({})",
            extract_error_code_from_response(&e)
        )
    })?;

    Ok(response)
}
