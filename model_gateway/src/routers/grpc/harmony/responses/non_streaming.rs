//! Non-streaming Harmony Responses API implementation

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::response::Response;
use openai_protocol::{
    common::{ToolCall, Usage},
    responses::{
        ResponseContentPart, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
        ResponseUsage, ResponsesRequest, ResponsesResponse, ResponsesUsage,
    },
};
use serde_json::{json, to_string};
use smg_mcp::McpServerBinding;
use tracing::{debug, error, warn};

use super::{
    common::{
        build_next_request_with_tools, inject_mcp_metadata, open_mcp_session,
        strip_image_generation_from_request_tools, McpCallTracking,
    },
    execution::{execute_mcp_tools, ToolResult},
};
use crate::{
    middleware::TenantRequestMeta,
    observability::metrics::Metrics,
    routers::{
        common::mcp_utils::DEFAULT_MAX_ITERATIONS,
        error,
        grpc::{
            common::responses::{
                collect_user_function_names, ensure_mcp_connection, load_history,
                persist_response_if_needed, tool_call_limit,
                utils::{
                    function_call_status, generation_failure_error, resolve_function_identity,
                    response_status,
                },
                NotStored, ReplayInto, ResponsesContext,
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
///
/// `response_id` is the id a background response was answered and stored
/// under: its terminal object is persisted under that id.
pub(crate) async fn serve_harmony_responses(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
    tenant_request_meta: TenantRequestMeta,
    response_id: Option<String>,
) -> Result<ResponsesResponse, Response> {
    // Load the history the request continues, as the caller's tenant may see
    // it; `request` itself stays as it came, for the persist.
    let tenant = crate::tenant::storage_tenant(&tenant_request_meta);
    let current_request = load_history(
        ctx.history_stores(),
        &request,
        tenant.as_deref(),
        ReplayInto::Harmony,
    )
    .await?;

    // Check MCP connection and get whether MCP tools are present
    let (has_mcp_tools, mcp_servers) = ensure_mcp_connection(
        &ctx.mcp_orchestrator,
        &ctx.mcp_format_registry,
        current_request.tools.as_deref(),
    )
    .await?;

    let mut response = if has_mcp_tools {
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
    if let Some(response_id) = response_id {
        response.id = response_id;
    }

    // Persist response to storage if store=true; a foreground response the
    // client expects stored is answered as stored or not at all.
    persist_response_if_needed(
        ctx.conversation_storage.clone(),
        ctx.conversation_item_storage.clone(),
        ctx.response_storage.clone(),
        &response,
        &request,
        tenant.as_deref(),
        ctx.request_context.clone(),
    )
    .await
    .map_err(NotStored::into_response)?;

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

    let session = open_mcp_session(ctx, mcp_servers, &mut current_request);

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
                finish_reason,
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
                let effective_limit = tool_call_limit(max_tool_calls);

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
                        finish_reason.as_deref(),
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
                    // `content_filter`); a truncation the builder read from the
                    // finish is cleared, as the streamed exit clears it.
                    response.status = ResponseStatus::Failed;
                    response.incomplete_details = None;
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
                        finish_reason.as_deref(),
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
            finish_reason,
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
                finish_reason.as_deref(),
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

/// Build ResponsesResponse with tool calls (MCP and/or function tools). The
/// function calls' status and the response's are read under `finish_reason`,
/// the turn's as the parser reads it.
#[expect(clippy::too_many_arguments)]
fn build_tool_response(
    mcp_tool_calls: Vec<ToolCall>,
    mcp_results: Vec<ToolResult>,
    function_tool_calls: Vec<ToolCall>,
    finish_reason: Option<&str>,
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

    // Add function tool calls WITHOUT output (need caller execution): each
    // call's status under the turn's finish, which a cut call reads as
    // incomplete, and the response as cut.
    let mut cut_call = false;
    for tool_call in function_tool_calls {
        let call_id = tool_call.id.clone();
        let arguments = tool_call.function.arguments.unwrap_or_default();
        let (name, namespace) =
            resolve_function_identity(responses_request.tools.as_deref(), &tool_call.function.name);
        let status = function_call_status(finish_reason, &arguments);
        cut_call |= status == "incomplete";
        output.push(ResponseOutputItem::FunctionToolCall {
            id: Some(tool_call.id),
            call_id,
            name,
            namespace,
            arguments,
            output: None, // No output = needs execution
            status: status.to_string(),
        });
    }

    // The status is read where every Responses object reads it: a `length`
    // finish is incomplete with `max_output_tokens`, a cut call incomplete
    // without a reason, a failed generation failed with its error.
    let (status, incomplete_details) = response_status(finish_reason, cut_call);
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let mut builder = ResponsesResponse::builder(&request_id, &responses_request.model)
        .copy_from_request(&responses_request)
        .created_at(created_at)
        .status(status)
        .output(output)
        .usage(ResponsesUsage::Modern(ResponseUsage::from(&usage)));
    if let Some(details) = incomplete_details {
        builder = builder.incomplete_details(details);
    }
    if let Some(error) = generation_failure_error(finish_reason) {
        builder = builder.error(error);
    }
    builder.build()
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
            Some("tool_calls"),
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

/// The status of the non-streamed response of a turn that returns its calls
/// to the caller is read from the turn's finish, as every Responses object's
/// is: a `length` finish is incomplete with `max_output_tokens`, a call cut
/// by anything else leaves it incomplete without a reason, and a turn that
/// ended on its closed calls is completed.
#[cfg(test)]
mod status_tests {
    use serde_json::Value;

    use super::*;

    fn call(arguments: &str) -> ToolCall {
        serde_json::from_value(json!({
            "id": "call_time", "type": "function",
            "function": {"name": "time", "arguments": arguments}
        }))
        .unwrap()
    }

    fn wire(finish_reason: Option<&str>, arguments: &str) -> Value {
        let response = build_tool_response(
            vec![],
            vec![],
            vec![call(arguments)],
            finish_reason,
            None,
            String::new(),
            Usage::from_counts(1, 1),
            "resp_test".into(),
            Arc::new(ResponsesRequest::default()),
        );
        serde_json::to_value(response).unwrap()
    }

    #[test]
    fn a_turn_cut_by_max_output_tokens_reads_incomplete_with_the_reason() {
        let wire = wire(Some("length"), "{\"tz\":");
        assert_eq!(wire["status"], "incomplete", "{wire}");
        assert_eq!(wire["incomplete_details"]["reason"], "max_output_tokens");
        assert_eq!(wire["output"][0]["status"], "incomplete");
        assert_eq!(wire["output"][0]["arguments"], "{\"tz\":");
    }

    #[test]
    fn a_cut_call_under_another_finish_reads_incomplete_without_a_reason() {
        for finish_reason in [Some("abort"), Some("stop"), None] {
            let wire = wire(finish_reason, "{\"tz\":");
            assert_eq!(wire["status"], "incomplete", "{finish_reason:?}: {wire}");
            assert!(wire["incomplete_details"].is_null(), "{finish_reason:?}");
            assert_eq!(
                wire["output"][0]["status"], "incomplete",
                "{finish_reason:?}"
            );
        }
    }

    #[test]
    fn a_turn_that_ended_on_its_closed_calls_reads_completed() {
        let wire = wire(Some("tool_calls"), "{\"tz\": \"UTC\"}");
        assert_eq!(wire["status"], "completed", "{wire}");
        assert!(wire["incomplete_details"].is_null());
        assert!(wire["error"].is_null());
        assert_eq!(wire["output"][0]["status"], "completed");
    }
}
