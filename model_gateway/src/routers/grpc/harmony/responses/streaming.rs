//! Streaming Harmony Responses API implementation

use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::Response;
use openai_protocol::responses::ResponsesRequest;
use serde_json::json;
use smg_mcp::{McpServerBinding, McpToolSession};
use tracing::{debug, warn};
use uuid::Uuid;

use super::{
    common::{
        build_next_request_with_tools, load_previous_messages,
        strip_image_generation_from_request_tools, McpCallTracking,
    },
    execution::{convert_mcp_tools_to_response_tools, execute_mcp_tools},
};
use crate::{
    middleware::TenantRequestMeta,
    observability::metrics::Metrics,
    routers::{
        common::{
            mcp_utils::DEFAULT_MAX_ITERATIONS,
            sse::{sse_channel, SseSender},
        },
        grpc::{
            common::responses::{
                await_stream_startup, ensure_mcp_connection, persist_response_if_needed,
                stream_startup_channel,
                streaming::{ResponseStreamEventEmitter, StreamStartupSender},
                ResponsesContext,
            },
            harmony::{processor::ResponsesIterationResult, streaming::HarmonyStreamingProcessor},
        },
    },
};

/// Serve Harmony Responses API with streaming (SSE)
///
/// This is the streaming equivalent of `serve_harmony_responses()`.
/// Emits SSE events for lifecycle, MCP list_tools, and per-iteration streaming.
#[expect(
    clippy::disallowed_methods,
    reason = "streaming task is fire-and-forget; client disconnect terminates it"
)]
pub(crate) async fn serve_harmony_responses_stream(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
    tenant_request_meta: TenantRequestMeta,
) -> Response {
    // Load previous conversation history if previous_response_id is set
    let current_request = match load_previous_messages(ctx, request.clone()).await {
        Ok(req) => req,
        Err(err_response) => return err_response,
    };

    // Check MCP connection BEFORE starting stream and get whether MCP tools are present
    let (has_mcp_tools, mcp_servers) = match ensure_mcp_connection(
        &ctx.mcp_orchestrator,
        &ctx.mcp_format_registry,
        current_request.tools.as_deref(),
    )
    .await
    {
        Ok(result) => result,
        Err(response) => return response,
    };

    // Create SSE channel
    let (tx, rx) = sse_channel();
    let (startup_tx, startup_rx) = stream_startup_channel();

    // Create response event emitter
    let response_id = format!("resp_{}", Uuid::now_v7());
    let model = current_request.model.clone();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut emitter = ResponseStreamEventEmitter::new(response_id.clone(), model, created_at);

    // Set original request for complete response fields
    emitter.set_original_request(current_request.clone());

    // Clone context for spawned task
    let ctx_clone = ctx.clone();

    // Spawn async task to handle streaming
    tokio::spawn(async move {
        let ctx = &ctx_clone;

        // Emit initial response.created and response.in_progress events
        let event = emitter.emit_created();
        if emitter.send_event(&event, &tx).await.is_err() {
            return;
        }
        let event = emitter.emit_in_progress();
        if emitter.send_event(&event, &tx).await.is_err() {
            return;
        }

        if has_mcp_tools {
            execute_mcp_tool_loop_streaming(
                ctx,
                current_request,
                &request,
                tenant_request_meta.clone(),
                mcp_servers,
                &mut emitter,
                &tx,
                Some(startup_tx),
            )
            .await;
        } else {
            execute_without_mcp_streaming(
                ctx,
                &current_request,
                &request,
                tenant_request_meta,
                &mut emitter,
                &tx,
                Some(startup_tx),
            )
            .await;
        }
    });

    await_stream_startup(startup_rx, rx).await
}

/// Execute MCP tool loop with streaming
///
/// Handles the full MCP workflow:
/// - Adds static MCP tools to request
/// - Emits mcp_list_tools events
/// - Loops through tool execution iterations
/// - Emits final response.completed event
/// - Persists response internally
#[expect(
    clippy::too_many_arguments,
    reason = "streaming MCP loop needs request state, emitter, channel and startup barrier"
)]
async fn execute_mcp_tool_loop_streaming(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    tenant_request_meta: TenantRequestMeta,
    mcp_servers: Vec<McpServerBinding>,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &SseSender,
    mut startup: Option<StreamStartupSender>,
) {
    let max_tool_calls = current_request.max_tool_calls.map(|n| n as usize);

    // Note: For streaming, the emitter's original_request (set before spawn) preserves
    // the original tools. MCP tools are only merged into current_request for model calls.

    // Create session once — bundles orchestrator, request_ctx, server_keys, mcp_tools
    let session_request_id = format!("resp_{}", Uuid::now_v7());

    let session = McpToolSession::new(&ctx.mcp_orchestrator, mcp_servers, &session_request_id);

    // Add filtered MCP tools (static + requested dynamic) to the request
    let mcp_tools = session.mcp_tools();
    if !mcp_tools.is_empty() {
        let mcp_response_tools = convert_mcp_tools_to_response_tools(&session);
        let mut all_tools = current_request.tools.take().unwrap_or_default();
        all_tools.extend(mcp_response_tools);
        current_request.tools = Some(all_tools);

        debug!(
            mcp_tool_count = mcp_tools.len(),
            total_tool_count = current_request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            "MCP client available - added static MCP tools to Harmony Responses streaming request"
        );
    }

    // Once the MCP loop has taken ownership of image_generation
    // dispatch, drop the hosted-tool descriptor so the harmony builder
    // advertises only the MCP-exposed function-tool name (which
    // `has_exposed_tool` actually recognizes for dispatch). See the
    // helper doc comment in `common.rs` for the full rationale.
    strip_image_generation_from_request_tools(
        &mut current_request,
        &session,
        &ctx.mcp_format_registry,
    );

    let mut mcp_tracking = McpCallTracking::new();

    // Emit mcp_list_tools on first iteration
    for binding in session.mcp_servers() {
        let tools_for_server = session.list_tools_for_server(&binding.server_key);

        if emitter
            .emit_mcp_list_tools_sequence(&binding.label, &tools_for_server, tx)
            .await
            .is_err()
        {
            return;
        }
    }

    debug!(
        tool_count = mcp_tools.len(),
        "Emitted mcp_list_tools on first iteration"
    );

    // MCP tool loop (max 10 iterations)
    let mut iteration_count = 0;
    loop {
        iteration_count += 1;

        // Record tool loop iteration metric
        Metrics::record_mcp_tool_iteration(&current_request.model);

        // Safety check: prevent infinite loops
        if iteration_count > DEFAULT_MAX_ITERATIONS {
            emitter
                .emit_error(
                    &format!("Maximum tool iterations ({DEFAULT_MAX_ITERATIONS}) exceeded"),
                    Some("max_iterations_exceeded"),
                    tx,
                )
                .await;
            return;
        }

        debug!(
            iteration = iteration_count,
            "Harmony Responses streaming iteration"
        );

        // Execute pipeline and get stream + load guards
        let startup_pending = startup.is_some();
        let (execution_result, _load_guards) = match ctx
            .pipeline
            .execute_harmony_responses_streaming(
                &current_request,
                ctx,
                Some(tenant_request_meta.clone()),
                startup.take(),
            )
            .await
        {
            Ok(result) => result,
            Err(err_response) => {
                if startup_pending {
                    return;
                }
                emitter
                    .emit_error(
                        &format!("Pipeline execution failed: {err_response:?}"),
                        Some("pipeline_error"),
                        tx,
                    )
                    .await;
                return;
            }
        };

        // Process stream with token-level streaming
        let iteration_result = match HarmonyStreamingProcessor::process_responses_iteration_stream(
            execution_result,
            emitter,
            tx,
            Some(&session),
            Some(&ctx.mcp_format_registry),
        )
        .await
        {
            Ok(result) => result,
            Err(err_msg) => {
                emitter
                    .emit_error(&err_msg, Some("processing_error"), tx)
                    .await;
                return;
            }
        };

        // Handle iteration result (tool calls or completion)
        match iteration_result {
            ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis,
                partial_text,
                usage,
                request_id: _,
            } => {
                debug!(
                    tool_call_count = tool_calls.len(),
                    has_analysis = analysis.is_some(),
                    partial_text_len = partial_text.len(),
                    "Tool calls found - separating MCP and function tools"
                );

                // Separate MCP and function tool calls based on session exposure.
                let (mcp_tool_calls, function_tool_calls): (Vec<_>, Vec<_>) = tool_calls
                    .into_iter()
                    .partition(|tc| session.has_exposed_tool(&tc.function.name));

                debug!(
                    mcp_calls = mcp_tool_calls.len(),
                    function_calls = function_tool_calls.len(),
                    "Tool calls separated by type in streaming"
                );

                // Check combined limit (user's max_tool_calls vs safety limit)
                let effective_limit = match max_tool_calls {
                    Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                    None => DEFAULT_MAX_ITERATIONS,
                };

                // Check if we would exceed the limit with these new MCP tool calls
                let total_calls_after = mcp_tracking.total_calls() + mcp_tool_calls.len();
                if total_calls_after > effective_limit {
                    warn!(
                        current_calls = mcp_tracking.total_calls(),
                        new_calls = mcp_tool_calls.len() + function_tool_calls.len(),
                        total_after = total_calls_after,
                        effective_limit = effective_limit,
                        user_max = ?max_tool_calls,
                        "Reached tool call limit in streaming - emitting completion with incomplete_details"
                    );

                    // Emit response.completed with incomplete_details and usage
                    let incomplete_details = json!({ "reason": "max_tool_calls" });
                    let usage_json = json!({
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "incomplete_details": incomplete_details,
                    });
                    let event = emitter.emit_completed(Some(&usage_json));
                    emitter.send_event_best_effort(&event, tx).await;
                    return;
                }

                // Execute MCP tools (if any). `original_request.tools` is the
                // caller-declared tool list (hosted-tool config lives there);
                // `execute_mcp_tools` merges the per-kind overrides into
                // dispatch args and forwards the request-level `user`
                // identifier for hosted tools.
                let mcp_results = if mcp_tool_calls.is_empty() {
                    Vec::new()
                } else {
                    match execute_mcp_tools(
                        &session,
                        &ctx.mcp_format_registry,
                        &mcp_tool_calls,
                        &mut mcp_tracking,
                        &current_request.model,
                        original_request.tools.as_deref().unwrap_or(&[]),
                        original_request.user.as_deref(),
                    )
                    .await
                    {
                        Ok(results) => results,
                        Err(err_response) => {
                            emitter
                                .emit_error(
                                    &format!("MCP tool execution failed: {err_response:?}"),
                                    Some("mcp_tool_error"),
                                    tx,
                                )
                                .await;
                            return;
                        }
                    }
                };

                // Update mcp_call output items with execution results (if any MCP tools were executed)
                if !mcp_results.is_empty() {
                    emitter.update_mcp_call_outputs(&mcp_results);
                }

                // If there are function tools, exit MCP loop and emit completion
                if !function_tool_calls.is_empty() {
                    debug!(
                        "Function tool calls present - exiting MCP loop and emitting completion"
                    );

                    // Function tool calls were already emitted during streaming processing
                    // Just emit response.completed with usage
                    let usage_json = json!({
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    });
                    let event = emitter.emit_completed(Some(&usage_json));
                    emitter.send_event_best_effort(&event, tx).await;
                    return;
                }

                // Only MCP tools - continue loop with their results
                debug!("Only MCP tools - continuing loop with results");

                // Build next request with appended history
                current_request = build_next_request_with_tools(
                    current_request,
                    mcp_tool_calls,
                    mcp_results,
                    analysis,
                    partial_text,
                );

                // Continue loop
            }
            ResponsesIterationResult::Completed { response, usage } => {
                debug!(
                    output_items = response.output.len(),
                    input_tokens = usage.prompt_tokens,
                    output_tokens = usage.completion_tokens,
                    "Harmony Responses streaming completed - no more tool calls"
                );

                // Finalize response from emitter's accumulated data
                let final_response = emitter.finalize(Some(usage.clone()));

                // Persist response to storage if store=true
                persist_response_if_needed(
                    ctx.conversation_storage.clone(),
                    ctx.conversation_item_storage.clone(),
                    ctx.response_storage.clone(),
                    &final_response,
                    original_request,
                    ctx.request_context.clone(),
                )
                .await;

                // Emit response.completed with usage
                let mut usage_json = json!({
                    "input_tokens": usage.prompt_tokens,
                    "output_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                });
                if let Some(details) = &usage.prompt_tokens_details {
                    if details.cached_tokens > 0 {
                        usage_json["input_tokens_details"] =
                            json!({ "cached_tokens": details.cached_tokens });
                    }
                }
                let event = emitter.emit_completed(Some(&usage_json));
                emitter.send_event_best_effort(&event, tx).await;
                return;
            }
        }
    }
}

/// Execute without MCP tool loop (single execution with streaming)
///
/// For function tools or no tools - executes pipeline once and emits completion.
/// The streaming processor handles all output items (reasoning, message, function tool calls).
async fn execute_without_mcp_streaming(
    ctx: &ResponsesContext,
    current_request: &ResponsesRequest,
    original_request: &ResponsesRequest,
    tenant_request_meta: TenantRequestMeta,
    emitter: &mut ResponseStreamEventEmitter,
    tx: &SseSender,
    mut startup: Option<StreamStartupSender>,
) {
    debug!("No MCP tools - executing single iteration");

    // Execute pipeline and get stream + load guards
    let startup_pending = startup.is_some();
    let (execution_result, _load_guards) = match ctx
        .pipeline
        .execute_harmony_responses_streaming(
            current_request,
            ctx,
            Some(tenant_request_meta),
            startup.take(),
        )
        .await
    {
        Ok(result) => result,
        Err(err_response) => {
            if startup_pending {
                return;
            }
            emitter
                .emit_error(
                    &format!("Pipeline execution failed: {err_response:?}"),
                    Some("pipeline_error"),
                    tx,
                )
                .await;
            return;
        }
    };

    // Process stream (no MCP context, all tools treated as function tools)
    let iteration_result = match HarmonyStreamingProcessor::process_responses_iteration_stream(
        execution_result,
        emitter,
        tx,
        None,
        None,
    )
    .await
    {
        Ok(result) => result,
        Err(err_msg) => {
            emitter
                .emit_error(&err_msg, Some("processing_error"), tx)
                .await;
            return;
        }
    };
    // _load_guards dropped here after iteration completes

    // Extract usage from iteration result
    let usage = match iteration_result {
        ResponsesIterationResult::ToolCallsFound { usage, .. } => usage,
        ResponsesIterationResult::Completed { usage, .. } => usage,
    };

    // Finalize response from emitter's accumulated data
    let final_response = emitter.finalize(Some(usage.clone()));

    // Persist response to storage if store=true
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &final_response,
        original_request,
        ctx.request_context.clone(),
    )
    .await;

    // Emit response.completed with usage
    let mut usage_json = json!({
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    });
    if let Some(details) = &usage.prompt_tokens_details {
        if details.cached_tokens > 0 {
            usage_json["input_tokens_details"] = json!({ "cached_tokens": details.cached_tokens });
        }
    }
    let event = emitter.emit_completed(Some(&usage_json));
    emitter.send_event_best_effort(&event, tx).await;
}
