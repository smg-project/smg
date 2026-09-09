//! Non-streaming Harmony Responses API implementation

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::response::Response;
use openai_protocol::{
    common::{ToolCall, Usage},
    responses::{
        InputTokensDetails, OutputTokensDetails, ResponseContentPart, ResponseOutputItem,
        ResponseReasoningContent, ResponseStatus, ResponseUsage, ResponsesRequest,
        ResponsesResponse, ResponsesUsage,
    },
};
use serde_json::{json, to_string};
use smg_mcp::{McpServerBinding, McpToolSession};
use tracing::{debug, error, warn};

use super::{
    common::{
        build_next_request_with_tools, inject_mcp_metadata, load_previous_messages,
        strip_image_generation_from_request_tools, McpCallTracking,
    },
    execution::{convert_mcp_tools_to_response_tools, execute_mcp_tools, ToolResult},
};
use crate::{
    middleware::TenantRequestMeta,
    observability::metrics::Metrics,
    routers::{
        common::mcp_utils::DEFAULT_MAX_ITERATIONS,
        error,
        grpc::{
            common::responses::{
                collect_user_function_names, ensure_mcp_connection, persist_response_if_needed,
                utils::resolve_function_identity, ResponsesContext,
            },
            harmony::processor::ResponsesIterationResult,
        },
    },
};

/// Execute Harmony Responses API request with multi-turn MCP tool support
///
/// This function orchestrates the multi-turn conversation flow:
/// 1. Execute request through full pipeline
/// 2. Check for tool calls in commentary channel
/// 3. If tool calls found:
///    - Execute MCP tools
///    - Build next request with tool results
///    - Repeat from step 1 (full pipeline re-execution)
/// 4. If no tool calls, return final response
pub(crate) async fn serve_harmony_responses(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
    tenant_request_meta: TenantRequestMeta,
) -> Result<ResponsesResponse, Response> {
    // Clone request for persistence
    let original_request = request.clone();

    // Load previous conversation history if previous_response_id is set
    let current_request = load_previous_messages(ctx, request).await?;

    // Check MCP connection and get whether MCP tools are present
    let (has_mcp_tools, mcp_servers) = ensure_mcp_connection(
        &ctx.mcp_orchestrator,
        &ctx.mcp_format_registry,
        current_request.tools.as_deref(),
    )
    .await?;

    let response = if has_mcp_tools {
        execute_with_mcp_loop(
            ctx,
            current_request,
            tenant_request_meta.clone(),
            mcp_servers,
        )
        .await?
    } else {
        // No MCP tools - execute pipeline once (may have function tools or no tools)
        execute_without_mcp_loop(ctx, current_request, tenant_request_meta).await?
    };

    // Persist response to storage if store=true
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &response,
        &original_request,
        ctx.request_context.clone(),
    )
    .await;

    Ok(response)
}

/// Execute Harmony Responses with MCP tool loop
///
/// Automatically executes MCP tools in a loop until no more tool calls or max iterations
async fn execute_with_mcp_loop(
    ctx: &ResponsesContext,
    mut current_request: ResponsesRequest,
    tenant_request_meta: TenantRequestMeta,
    mcp_servers: Vec<McpServerBinding>,
) -> Result<ResponsesResponse, Response> {
    let mut iteration_count = 0;

    let mut mcp_tracking = McpCallTracking::new();

    // Extract user's max_tool_calls limit (if set)
    let max_tool_calls = current_request.max_tool_calls.map(|n| n as usize);

    // Preserve original tools for response (before merging MCP tools)
    // The response should show the user's original request tools, not internal MCP tools
    let mut original_tools = current_request.tools.clone();
    let user_function_names = collect_user_function_names(&current_request);

    // Create session once — bundles orchestrator, request_ctx, server_keys, mcp_tools
    let session_request_id = format!("resp_{}", uuid::Uuid::now_v7());

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
            "MCP client available - added static MCP tools to Harmony Responses request"
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

    loop {
        iteration_count += 1;

        // Record tool loop iteration metric
        Metrics::record_mcp_tool_iteration(&current_request.model);

        // Safety check: prevent infinite loops
        if iteration_count > DEFAULT_MAX_ITERATIONS {
            error!(
                function = "execute_with_mcp_loop",
                iteration_count = iteration_count,
                max_iterations = DEFAULT_MAX_ITERATIONS,
                "Maximum tool iterations exceeded"
            );
            return Err(error::internal_error(
                "tool_iterations_exceeded",
                format!("Maximum tool iterations ({DEFAULT_MAX_ITERATIONS}) exceeded"),
            ));
        }

        debug!(
            iteration = iteration_count,
            "Harmony Responses serving iteration"
        );

        // Execute through full pipeline
        let iteration_result = ctx
            .pipeline
            .execute_harmony_responses(&current_request, ctx, Some(tenant_request_meta.clone()))
            .await?;

        match iteration_result {
            ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis,
                partial_text,
                usage,
                request_id,
            } => {
                debug!(
                    tool_call_count = tool_calls.len(),
                    has_analysis = analysis.is_some(),
                    partial_text_len = partial_text.len(),
                    "Tool calls found - separating MCP and function tools"
                );

                // Separate MCP and function tool calls based on session exposure.
                // MCP tools are exposed to the model as function tools, so the only reliable
                // discriminator is whether the name belongs to the MCP session.
                let (mcp_tool_calls, function_tool_calls): (Vec<_>, Vec<_>) = tool_calls
                    .into_iter()
                    .partition(|tc| session.has_exposed_tool(tc.function.name.as_str()));

                debug!(
                    mcp_calls = mcp_tool_calls.len(),
                    function_calls = function_tool_calls.len(),
                    "Tool calls separated by type"
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
                        "Reached tool call limit - returning incomplete response"
                    );

                    // Combine back for response
                    let all_tool_calls: Vec<_> = mcp_tool_calls
                        .into_iter()
                        .chain(function_tool_calls)
                        .collect();

                    // Build response with incomplete status - no tools executed due to limit
                    // Use original_tools for response (hide internal MCP tools)
                    let mut response_request = current_request.clone();
                    response_request.tools.clone_from(&original_tools);
                    let mut response = build_tool_response(
                        vec![],         // No MCP tools executed
                        vec![],         // No MCP results
                        all_tool_calls, // All tools returned as function calls (not executed)
                        analysis,
                        partial_text,
                        usage,
                        request_id,
                        Arc::new(response_request),
                    );

                    // Tool-call limit reached before executing the remaining
                    // calls: this is an aborted run, not a successful answer.
                    // Per the design, exhausting `max_tool_calls` is a `failed`
                    // status with an `error` payload (truncation
                    // `incomplete_details` is reserved for `max_output_tokens` /
                    // `content_filter`).
                    response.status = ResponseStatus::Failed;
                    response.error = Some(json!({
                        "code": "max_tool_calls_exceeded",
                        "message": format!(
                            "Reached the max_tool_calls limit ({effective_limit}) before executing the remaining tool calls."
                        ),
                    }));

                    // Inject MCP metadata if any calls were executed
                    if mcp_tracking.total_calls() > 0 {
                        inject_mcp_metadata(
                            &mut response,
                            &mcp_tracking,
                            &session,
                            &user_function_names,
                        );
                    }

                    return Ok(response);
                }

                // Execute MCP tools (if any). Caller-declared hosted-tool overrides
                // live on `original_tools` (pre-MCP-injection), so we thread those
                // into dispatch — `execute_mcp_tools` merges per-kind. The
                // request-level `user` identifier is also forwarded into
                // hosted-tool dispatch args.
                let mcp_results = if mcp_tool_calls.is_empty() {
                    Vec::new()
                } else {
                    execute_mcp_tools(
                        &session,
                        &ctx.mcp_format_registry,
                        &mcp_tool_calls,
                        &mut mcp_tracking,
                        &current_request.model,
                        original_tools.as_deref().unwrap_or(&[]),
                        current_request.user.as_deref(),
                    )
                    .await?
                };

                // If there are function tools, exit MCP loop and return response
                if !function_tool_calls.is_empty() {
                    debug!(
                        "Function tool calls present - exiting MCP loop and returning to caller"
                    );

                    // Build response that includes:
                    // 1. Reasoning/message from this iteration
                    // 2. MCP tools as completed (with output) - these were executed
                    // 3. Function tools as completed (without output) - need caller execution
                    // Use original_tools for response (hide internal MCP tools)
                    let mut response_request = current_request.clone();
                    response_request.tools.clone_from(&original_tools);
                    let mut response = build_tool_response(
                        mcp_tool_calls,
                        mcp_results,
                        function_tool_calls,
                        analysis,
                        partial_text,
                        usage,
                        request_id,
                        Arc::new(response_request),
                    );

                    // Inject MCP metadata for all executed calls
                    if mcp_tracking.total_calls() > 0 {
                        inject_mcp_metadata(
                            &mut response,
                            &mcp_tracking,
                            &session,
                            &user_function_names,
                        );
                    }

                    return Ok(response);
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

                // Continue loop - next iteration will select workers and execute
            }
            ResponsesIterationResult::Completed {
                mut response,
                usage,
            } => {
                debug!(
                    output_items = response.output.len(),
                    input_tokens = usage.prompt_tokens,
                    output_tokens = usage.completion_tokens,
                    "MCP loop completed - no more tool calls"
                );

                // Inject MCP metadata into final response
                inject_mcp_metadata(&mut response, &mcp_tracking, &session, &user_function_names);

                // Restore original tools (hide internal MCP tools from response)
                response.tools = original_tools.take().unwrap_or_default();

                debug!(
                    mcp_calls = mcp_tracking.total_calls(),
                    output_items_after = response.output.len(),
                    "Injected MCP metadata into final response"
                );

                // No tool calls - this is the final response
                return Ok(*response);
            }
        }
    }
}

/// Execute Harmony Responses without MCP loop (single execution)
///
/// For function tools or no tools - executes pipeline once and returns
async fn execute_without_mcp_loop(
    ctx: &ResponsesContext,
    current_request: ResponsesRequest,
    tenant_request_meta: TenantRequestMeta,
) -> Result<ResponsesResponse, Response> {
    debug!("Executing Harmony Responses without MCP loop");

    // Execute pipeline once
    let iteration_result = ctx
        .pipeline
        .execute_harmony_responses(&current_request, ctx, Some(tenant_request_meta))
        .await?;

    match iteration_result {
        ResponsesIterationResult::ToolCallsFound {
            tool_calls,
            analysis,
            partial_text,
            usage,
            request_id,
        } => {
            // Function tool calls found - return to caller for execution
            debug!(
                tool_call_count = tool_calls.len(),
                "Function tool calls found - returning to caller"
            );

            Ok(build_tool_response(
                vec![],
                vec![],
                tool_calls,
                analysis,
                partial_text,
                usage,
                request_id,
                Arc::new(current_request),
            ))
        }
        ResponsesIterationResult::Completed { response, usage: _ } => {
            // No tool calls - return completed response
            debug!("No tool calls - returning completed response");
            Ok(*response)
        }
    }
}

/// Build ResponsesResponse with tool calls (MCP and/or function tools)
#[expect(clippy::too_many_arguments)]
fn build_tool_response(
    mcp_tool_calls: Vec<ToolCall>,
    mcp_results: Vec<ToolResult>,
    function_tool_calls: Vec<ToolCall>,
    analysis: Option<String>, // Analysis channel content (reasoning)
    partial_text: String,     // Final channel content (message)
    usage: Usage,
    request_id: String,
    responses_request: Arc<ResponsesRequest>,
) -> ResponsesResponse {
    let mut output: Vec<ResponseOutputItem> = Vec::new();

    // Add reasoning output item if analysis exists
    if let Some(analysis_text) = analysis {
        output.push(ResponseOutputItem::new_reasoning(
            format!("reasoning_{request_id}"),
            vec![],
            vec![ResponseReasoningContent::ReasoningText {
                text: analysis_text,
            }],
            Some("completed".to_string()),
        ));
    }

    // Add message output item if partial text exists
    if !partial_text.is_empty() {
        output.push(ResponseOutputItem::Message {
            id: format!("msg_{request_id}"),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: partial_text,
                annotations: vec![],
                logprobs: None,
            }],
            status: "completed".to_string(),
            phase: None,
        });
    }

    // Add MCP tool calls WITH output (these were executed)
    for (tool_call, result) in mcp_tool_calls.iter().zip(mcp_results.iter()) {
        let output_str = to_string(&result.output)
            .unwrap_or_else(|e| format!("{{\"error\": \"Failed to serialize tool output: {e}\"}}"));

        let (name, namespace) =
            resolve_function_identity(responses_request.tools.as_deref(), &tool_call.function.name);
        output.push(ResponseOutputItem::FunctionToolCall {
            id: Some(tool_call.id.clone()),
            call_id: tool_call.id.clone(),
            name,
            namespace,
            arguments: tool_call.function.arguments.clone().unwrap_or_default(),
            output: Some(output_str),
            status: if result.is_error {
                "failed"
            } else {
                "completed"
            }
            .to_string(),
        });
    }

    // Add function tool calls WITHOUT output (need caller execution)
    for tool_call in function_tool_calls {
        let call_id = tool_call.id.clone();
        let arguments = tool_call.function.arguments.unwrap_or_default();
        let (name, namespace) =
            resolve_function_identity(responses_request.tools.as_deref(), &tool_call.function.name);
        output.push(ResponseOutputItem::FunctionToolCall {
            id: Some(tool_call.id),
            call_id,
            name,
            namespace,
            arguments,
            output: None, // No output = needs execution
            status: "completed".to_string(),
        });
    }

    // Build ResponsesResponse with Completed status
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    ResponsesResponse::builder(&request_id, &responses_request.model)
        .copy_from_request(&responses_request)
        .created_at(created_at)
        .status(ResponseStatus::Completed)
        .output(output)
        .usage(ResponsesUsage::Modern(ResponseUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            input_tokens_details: usage
                .prompt_tokens_details
                .as_ref()
                .map(InputTokensDetails::from),
            output_tokens_details: usage.completion_tokens_details.as_ref().and_then(|d| {
                d.reasoning_tokens.map(|tokens| OutputTokensDetails {
                    reasoning_tokens: tokens,
                })
            }),
        }))
        .build()
}

#[cfg(test)]
mod namespace_tests {
    use super::*;
    use crate::routers::grpc::common::responses::utils::namespace_test_request;

    #[test]
    fn namespace_function_response_preserves_identity() {
        let request = namespace_test_request();
        let call = serde_json::from_value(json!({
            "id": "call_test", "type": "function",
            "function": {"name": "weather.lookup", "arguments": "{}"}
        }))
        .unwrap();
        let response = build_tool_response(
            vec![],
            vec![],
            vec![call],
            None,
            String::new(),
            Usage::from_counts(1, 1),
            "resp_test".into(),
            Arc::new(request),
        );
        let wire = serde_json::to_value(response).unwrap();
        assert_eq!(wire["output"][0]["name"], "lookup");
        assert_eq!(wire["output"][0]["namespace"], "weather");
    }
}
