//! Non-streaming execution for Regular Responses API
//!
//! This module handles non-streaming request execution:
//! - `route_responses_internal` - Core execution orchestration
//! - `execute_tool_loop` - MCP tool loop execution
//! - `execute_without_mcp` - Simple pipeline execution without MCP

use std::sync::Arc;

use axum::response::Response;
use openai_protocol::responses::{ResponseStatus, ResponsesRequest, ResponsesResponse};
use serde_json::json;
use smg_mcp::{McpServerBinding, McpToolSession, ToolExecutionInput};
use tracing::{debug, error, trace, warn};

use super::{
    common::{
        build_next_request, convert_mcp_tools_to_chat_tools, extract_all_tool_calls_from_chat,
        load_conversation_history, prepare_chat_tools_and_choice, ExtractedToolCall,
        ResponsesCallContext, ToolLoopState,
    },
    conversions,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            mcp_utils::{prepare_hosted_dispatch_args, DEFAULT_MAX_ITERATIONS},
            openai_bridge::{self, ResponseFormat},
        },
        error,
        grpc::common::responses::{
            collect_user_function_names, ensure_mcp_connection, persist_response_if_needed,
            ResponsesContext,
        },
    },
};

/// Internal implementation for non-streaming responses
///
/// This is the core execution path that:
/// 1. Loads conversation history / response chain
/// 2. Checks for MCP tools
/// 3. Executes with or without MCP tool loop
/// 4. Persists to storage
pub(super) async fn route_responses_internal(
    ctx: &ResponsesContext,
    request: Arc<ResponsesRequest>,
    params: ResponsesCallContext,
) -> Result<ResponsesResponse, Response> {
    // 1. Load conversation history and build modified request
    let modified_request = load_conversation_history(ctx, &request).await?;

    // 2. Check MCP connection and get whether MCP tools are present
    let (has_mcp_tools, mcp_servers) = ensure_mcp_connection(
        &ctx.mcp_orchestrator,
        &ctx.mcp_format_registry,
        request.tools.as_deref(),
    )
    .await?;

    let responses_response = if has_mcp_tools {
        debug!("MCP tools detected, using tool loop");

        // Execute with MCP tool loop
        execute_tool_loop(ctx, modified_request, &request, &params, mcp_servers).await?
    } else {
        // No MCP tools - execute without MCP (may have function tools or no tools)
        execute_without_mcp(ctx, &modified_request, &request, params).await?
    };

    // 5. Persist response to storage if store=true
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &responses_response,
        &request,
        ctx.request_context.clone(),
    )
    .await;

    Ok(responses_response)
}

/// Execute request without MCP tool loop (simple pipeline execution)
pub(super) async fn execute_without_mcp(
    ctx: &ResponsesContext,
    modified_request: &ResponsesRequest,
    original_request: &ResponsesRequest,
    params: ResponsesCallContext,
) -> Result<ResponsesResponse, Response> {
    // Convert ResponsesRequest → ChatCompletionRequest
    let chat_request = conversions::responses_to_chat(modified_request).map_err(|e| {
        error!(
            function = "execute_without_mcp",
            error = %e,
            "Failed to convert ResponsesRequest to ChatCompletionRequest"
        );
        error::bad_request(
            "convert_request_failed",
            format!("Failed to convert request: {e}"),
        )
    })?;

    // Execute chat pipeline (errors already have proper HTTP status codes)
    let chat_response = ctx
        .pipeline
        .execute_chat_for_responses(
            Arc::new(chat_request),
            params.headers,
            params.model_id,
            ctx.components.clone(),
            Some(params.tenant_request_meta),
        )
        .await?; // Preserve the Response error as-is

    // Convert ChatCompletionResponse → ResponsesResponse
    conversions::chat_to_responses(&chat_response, original_request, params.response_id).map_err(
        |e| {
            error!(
                function = "execute_without_mcp",
                error = %e,
                "Failed to convert ChatCompletionResponse to ResponsesResponse"
            );
            error::internal_error(
                "convert_to_responses_format_failed",
                format!("Failed to convert to responses format: {e}"),
            )
        },
    )
}

/// Execute the MCP tool calling loop
///
/// This wraps pipeline.execute_chat_for_responses() in a loop that:
/// 1. Executes the chat pipeline
/// 2. Checks if response has tool calls
/// 3. If yes, executes MCP tools and builds resume request
/// 4. Repeats until no more tool calls or limit reached
pub(super) async fn execute_tool_loop(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    original_request: &ResponsesRequest,
    params: &ResponsesCallContext,
    mcp_servers: Vec<McpServerBinding>,
) -> Result<ResponsesResponse, Response> {
    let mut state = ToolLoopState::new(original_request);

    // Configuration: max iterations as safety limit
    let max_tool_calls = original_request.max_tool_calls.map(|n| n as usize);

    trace!(
        "Starting MCP tool loop: max_tool_calls={:?}, max_iterations={}",
        max_tool_calls,
        DEFAULT_MAX_ITERATIONS
    );

    // Create session once — bundles orchestrator, request_ctx, server_keys, mcp_tools
    let session_request_id = params
        .response_id
        .clone()
        .unwrap_or_else(|| format!("resp_{}", uuid::Uuid::now_v7()));

    let session = McpToolSession::new(&ctx.mcp_orchestrator, mcp_servers, &session_request_id);
    let user_function_names = collect_user_function_names(original_request);

    // Get MCP tools and convert to chat format (do this once before loop)
    let mcp_chat_tools = convert_mcp_tools_to_chat_tools(&session);
    trace!(
        "Converted {} MCP tools to chat format",
        mcp_chat_tools.len()
    );

    loop {
        // Convert to chat request
        let mut chat_request = conversions::responses_to_chat(&current_request).map_err(|e| {
            error!(
                function = "tool_loop",
                iteration = state.iteration,
                error = %e,
                "Failed to convert ResponsesRequest to ChatCompletionRequest in tool loop"
            );
            error::bad_request(
                "convert_request_failed",
                format!("Failed to convert request: {e}"),
            )
        })?;

        // Prepare tools and tool_choice for this iteration
        prepare_chat_tools_and_choice(&mut chat_request, &mcp_chat_tools, state.iteration);

        // Execute chat pipeline (errors already have proper HTTP status codes)
        let chat_response = ctx
            .pipeline
            .execute_chat_for_responses(
                Arc::new(chat_request),
                params.headers.clone(),
                params.model_id.clone(),
                ctx.components.clone(),
                Some(params.tenant_request_meta.clone()),
            )
            .await?;

        // Check for function calls (extract all for parallel execution)
        let tool_calls = extract_all_tool_calls_from_chat(&chat_response);

        if tool_calls.is_empty() {
            // No more tool calls, we're done
            trace!(
                "Tool loop completed: {} iterations, {} total calls",
                state.iteration,
                state.total_calls
            );

            // Convert final chat response to responses format
            let mut responses_response = conversions::chat_to_responses(
                &chat_response,
                original_request,
                params.response_id.clone(),
            )
            .map_err(|e| {
                error!(
                    function = "tool_loop",
                    iteration = state.iteration,
                    error = %e,
                    context = "final_response",
                    "Failed to convert ChatCompletionResponse to ResponsesResponse"
                );
                error::internal_error(
                    "convert_to_responses_format_failed",
                    format!("Failed to convert to responses format: {e}"),
                )
            })?;

            // Inject MCP metadata into output
            if state.total_calls > 0 {
                openai_bridge::inject_client_visible_mcp_output_items(
                    &session,
                    &mut responses_response.output,
                    state.mcp_call_items,
                    &user_function_names,
                );

                trace!(
                    "Injected MCP metadata: {} mcp_list_tools + {} mcp_call items",
                    session.mcp_servers().len(),
                    state.total_calls
                );
            }

            return Ok(responses_response);
        } else {
            state.iteration += 1;

            // Record tool loop iteration metric
            Metrics::record_mcp_tool_iteration(&current_request.model);

            trace!(
                "Tool loop iteration {}: found {} tool call(s)",
                state.iteration,
                tool_calls.len()
            );

            // Separate MCP and function tool calls using session-exposed names.
            let (mcp_tool_calls, function_tool_calls): (Vec<ExtractedToolCall>, Vec<_>) =
                tool_calls
                    .into_iter()
                    .partition(|tc| session.has_exposed_tool(tc.name.as_str()));

            trace!(
                "Separated tool calls: {} MCP, {} function",
                mcp_tool_calls.len(),
                function_tool_calls.len()
            );

            // If ANY tool call is a function tool, return to caller immediately
            if !function_tool_calls.is_empty() {
                // Convert chat response to responses format (includes all tool calls)
                let responses_response = conversions::chat_to_responses(
                    &chat_response,
                    original_request,
                    params.response_id.clone(),
                )
                .map_err(|e| {
                    error!(
                        function = "tool_loop",
                        iteration = state.iteration,
                        error = %e,
                        context = "function_tool_calls",
                        "Failed to convert ChatCompletionResponse to ResponsesResponse"
                    );
                    error::internal_error(
                        "convert_to_responses_format_failed",
                        format!("Failed to convert to responses format: {e}"),
                    )
                })?;

                // Return response with function tool calls to caller
                return Ok(responses_response);
            }

            // All MCP tools - check combined limit BEFORE executing
            let effective_limit = match max_tool_calls {
                Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
                None => DEFAULT_MAX_ITERATIONS,
            };

            if state.total_calls + mcp_tool_calls.len() > effective_limit {
                warn!(
                    "Reached tool call limit: {} + {} > {} (max_tool_calls={:?}, safety_limit={})",
                    state.total_calls,
                    mcp_tool_calls.len(),
                    effective_limit,
                    max_tool_calls,
                    DEFAULT_MAX_ITERATIONS
                );

                // Convert chat response to responses format and mark as incomplete
                let mut responses_response = conversions::chat_to_responses(
                    &chat_response,
                    original_request,
                    params.response_id.clone(),
                )
                .map_err(|e| {
                    error!(
                        function = "tool_loop",
                        iteration = state.iteration,
                        error = %e,
                        context = "max_tool_calls_limit",
                        "Failed to convert ChatCompletionResponse to ResponsesResponse"
                    );
                    error::internal_error(
                        "convert_to_responses_format_failed",
                        format!("Failed to convert to responses format: {e}"),
                    )
                })?;

                // Tool-call limit reached before executing the remaining calls:
                // an aborted run, not a successful answer. Per the design,
                // exhausting `max_tool_calls` is a `failed` status with an
                // `error` payload (truncation `incomplete_details` is reserved
                // for `max_output_tokens` / `content_filter`).
                responses_response.status = ResponseStatus::Failed;
                responses_response.error = Some(json!({
                    "code": "max_tool_calls_exceeded",
                    "message": format!(
                        "Reached the max_tool_calls limit ({effective_limit}) before executing the remaining tool calls."
                    ),
                }));

                return Ok(responses_response);
            }

            // Convert tool calls to execution inputs, merging caller-declared
            // hosted-tool config from `original_request.tools` into dispatch args.
            // Non-object model payloads coerce to `{}` so the merge actually
            // applies instead of silently dropping the caller's config. The
            // request-level `user` is also forwarded into hosted-tool args.
            let request_tools = original_request.tools.as_deref().unwrap_or(&[]);
            let request_user = original_request.user.as_deref();
            // Resolve `response_format` once per call here and zip it through
            // to the output processing pass — looking it up twice (once for
            // arg prep, once for transform) allocates two extra `Arc<str>`s
            // per call. `session.execute_tools` preserves input ordering.
            let prepared: Vec<(ToolExecutionInput, ResponseFormat)> = mcp_tool_calls
                .into_iter()
                .map(|tc| {
                    let mut arguments =
                        match serde_json::from_str::<serde_json::Value>(&tc.arguments) {
                            Ok(serde_json::Value::Object(map)) => serde_json::Value::Object(map),
                            _ => json!({}),
                        };
                    let response_format = openai_bridge::lookup_tool_format(
                        &session,
                        &ctx.mcp_format_registry,
                        &tc.name,
                    );
                    prepare_hosted_dispatch_args(
                        &mut arguments,
                        response_format,
                        request_tools,
                        request_user,
                    );
                    let input = ToolExecutionInput {
                        call_id: tc.call_id,
                        tool_name: tc.name,
                        arguments,
                    };
                    (input, response_format)
                })
                .collect();
            let (inputs, formats): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();

            let results = session.execute_tools(inputs).await;
            // `session.execute_tools` preserves input order and length; assert
            // it so a regression there can't silently truncate via `zip`.
            assert_eq!(
                results.len(),
                formats.len(),
                "session.execute_tools returned {} outputs for {} inputs; \
                 per-call format zip would silently drop entries",
                results.len(),
                formats.len(),
            );

            for (result, response_format) in results.into_iter().zip(formats) {
                trace!(
                    "Tool '{}' (call_id: {}) completed in {:?}, success={}",
                    result.tool_name,
                    result.call_id,
                    result.duration,
                    !result.is_error
                );

                Metrics::record_mcp_tool_duration(
                    &current_request.model,
                    &result.tool_name,
                    result.duration,
                );
                Metrics::record_mcp_tool_call(
                    &current_request.model,
                    &result.tool_name,
                    if result.is_error {
                        metrics_labels::RESULT_ERROR
                    } else {
                        metrics_labels::RESULT_SUCCESS
                    },
                );

                let output_item = openai_bridge::transform_tool_output(&result, response_format);
                let output_str = result.output.to_string();
                state.record_call(
                    result.call_id,
                    result.tool_name,
                    result.arguments_str,
                    output_str,
                    output_item,
                    !result.is_error,
                );

                // Increment total calls counter
                state.total_calls += 1;
            }

            // Build resume request with conversation history
            current_request = build_next_request(&state, current_request);

            // Continue to next iteration
        }
    }
}
