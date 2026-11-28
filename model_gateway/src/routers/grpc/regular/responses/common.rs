//! Shared helpers and state tracking for Regular Responses
//!
//! This module contains common utilities used by both streaming and non-streaming paths:
//! - ToolLoopState for tracking multi-turn tool calling
//! - Helper functions for tool preparation and extraction
//! - MCP metadata builders
//!
//! A request's history is loaded by the shared
//! [`crate::routers::grpc::common::responses::load_history`].

use axum::http;
use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{Tool, ToolChoice, ToolChoiceValue, ToolReference, Usage},
    responses::{
        self, ResponseInput, ResponseInputOutputItem, ResponseOutputItem, ResponseUsage,
        ResponsesRequest,
    },
};
use serde_json::{json, Value};
use smg_mcp::{McpToolSession, ToolExecutionOutput};
use tracing::{trace, warn};

pub(super) use crate::routers::grpc::common::responses::utils::McpToolCallLimit;
use crate::{
    middleware::TenantRequestMeta,
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            mcp_utils::prepare_hosted_dispatch_args,
            openai_bridge::{self, ResponseFormat},
        },
        grpc::common::responses::{
            history::current_input_items,
            tool_call_limit,
            utils::{add_model_call_usage, resolve_function_identity, tool_call_overflow},
        },
    },
};

// ============================================================================
// Tool Loop State
// ============================================================================

/// State for tracking multi-turn tool calling loop
///
/// The state holds no copy of the request: the loop owns its request and
/// rebuilds it in place each iteration ([`build_next_request`]), so the input
/// the loop started from (the request's history loaded, a stored turn's tool
/// calls and outputs in front of the calls this loop executes) is carried by
/// the request itself and copied by no iteration.
#[derive(Default)]
pub(super) struct ToolLoopState {
    pub iteration: usize,
    pub total_calls: usize,
    pub usage: Option<ResponseUsage>,
    /// The calls this loop executed since its request was last built, each
    /// with its output, in order: the next build moves them into the request
    /// behind the items it already carries.
    pub calls_to_carry: Vec<ResponseInputOutputItem>,
    pub mcp_call_items: Vec<ResponseOutputItem>,
    /// The request's `max_tool_calls` is spent: the next generation is the
    /// response's final turn, made with the server tools withheld so the
    /// model answers on the results it has, and its answer ends the loop
    /// whatever it holds (a server call made anyway stays hidden and does
    /// not run).
    pub final_turn: bool,
}

/// Per-request parameters for chat pipeline execution.
/// Bundles values that are always threaded together through the regular responses call chain.
pub(super) struct ResponsesCallContext {
    pub headers: Option<http::HeaderMap>,
    pub model_id: String,
    pub response_id: Option<String>,
    pub tenant_request_meta: TenantRequestMeta,
}

impl ToolLoopState {
    /// Usage is per model request; missing reports must not erase prior totals.
    pub fn record_usage(&mut self, usage: Option<&Usage>) {
        add_model_call_usage(&mut self.usage, usage);
    }

    /// `tools` are the request's own tool definitions, read for the call's
    /// namespace; the loop's request carries them, so the state keeps no copy.
    pub fn record_call(
        &mut self,
        tools: Option<&[responses::ResponseTool]>,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
        output_item: ResponseOutputItem,
    ) {
        // Add function_tool_call item with both arguments and output
        let id = call_id.clone();
        let (name, namespace) = resolve_function_identity(tools, &tool_name);
        self.calls_to_carry
            .push(ResponseInputOutputItem::FunctionToolCall {
                id: Some(id),
                call_id,
                caller: None,
                name,
                namespace,
                arguments: args_json_str,
                output: Some(output_str),
                status: Some("completed".to_string()),
            });

        // Add transformed output item (respects tool's response_format)
        self.mcp_call_items.push(output_item);
    }
}

// ============================================================================
// Tool Preparation and Extraction
// ============================================================================

/// Set tool_choice based on iteration (the MCP tools join the request's tools
/// where the chat request is built: `conversions::responses_to_chat_with_mcp_tools`).
/// The final turn allows the request's own tools only, the ones
/// `is_server_tool` does not name: the prompt renders the allowed tools, so
/// the model is offered no server tool, while the response is read against
/// the whole list as on every turn. A legacy streaming parser takes a call to
/// a tool outside that list for text, so a list without the server tools
/// would leave a repeated server call's markup in the message.
pub(super) fn prepare_tool_choice(
    chat_request: &mut ChatCompletionRequest,
    state: &ToolLoopState,
    is_server_tool: impl Fn(&str) -> bool,
) {
    // Set tool_choice based on iteration
    // - Iteration 0: Use user's tool_choice or default to auto
    // - Iteration 1+: Always use auto to avoid infinite loops
    chat_request.tool_choice = if state.final_turn {
        Some(ToolChoice::AllowedTools {
            tool_type: "allowed_tools".to_string(),
            mode: "auto".to_string(),
            tools: chat_request
                .tools
                .as_deref()
                .unwrap_or_default()
                .iter()
                .filter(|tool| !is_server_tool(&tool.function.name))
                .map(|tool| ToolReference::Function {
                    name: tool.function.name.clone(),
                })
                .collect(),
        })
    } else if state.iteration == 0 {
        chat_request
            .tool_choice
            .take()
            .or(Some(ToolChoice::Value(ToolChoiceValue::Auto)))
    } else {
        Some(ToolChoice::Value(ToolChoiceValue::Auto))
    };
}

/// Tool call extracted from a ChatCompletionResponse
#[derive(Debug, Clone)]
pub(super) struct ExtractedToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

/// One iteration's extracted tool calls, traced and split into the MCP
/// calls (those the session exposes) and the function calls, the MCP calls
/// capped to the request's remaining allowance (see
/// [`apply_mcp_tool_call_limit`], whose verdict comes with them). The traces
/// log under this module's target; `streaming` tells the two loops' lines
/// apart.
pub(super) fn partition_tool_calls(
    tool_calls: Vec<ExtractedToolCall>,
    session: &McpToolSession,
    state: &ToolLoopState,
    max_tool_calls: Option<usize>,
    streaming: bool,
) -> (
    Vec<ExtractedToolCall>,
    Vec<ExtractedToolCall>,
    Option<McpToolCallLimit>,
) {
    trace!(
        streaming,
        "Tool loop iteration {}: found {} tool call(s)",
        state.iteration,
        tool_calls.len()
    );

    // Separate MCP and function tool calls using session-exposed names.
    let (mut mcp_tool_calls, function_tool_calls): (Vec<ExtractedToolCall>, Vec<_>) = tool_calls
        .into_iter()
        .partition(|tc| session.has_exposed_tool(tc.name.as_str()));

    trace!(
        streaming,
        "Separated tool calls: {} MCP, {} function",
        mcp_tool_calls.len(),
        function_tool_calls.len()
    );

    // Only execute calls that fit; preserve their results before ending
    // the response when this batch exceeds the remaining allowance.
    let tool_call_limit =
        apply_mcp_tool_call_limit(&mut mcp_tool_calls, state.total_calls, max_tool_calls);
    (mcp_tool_calls, function_tool_calls, tool_call_limit)
}

/// Keep the prefix of MCP calls that fits the remaining request budget.
///
/// A batch that fits may continue to a final model answer. An overflowing
/// batch executes its permitted prefix, then stops normally for a user cap
/// or fails for the internal safety cap. A user cap wins when both are equal.
fn apply_mcp_tool_call_limit(
    calls: &mut Vec<ExtractedToolCall>,
    total_calls: usize,
    max_tool_calls: Option<usize>,
) -> Option<McpToolCallLimit> {
    let (limit, remaining) = tool_call_overflow(max_tool_calls, total_calls, calls.len())?;
    if limit == McpToolCallLimit::Safety {
        warn!(
            total_calls,
            batch_size = calls.len(),
            effective_limit = tool_call_limit(max_tool_calls),
            "Internal MCP tool call safety limit reached"
        );
    }
    calls.truncate(remaining);
    Some(limit)
}

/// Whether a server call among `calls` has arguments that are not JSON. A
/// parser can report a call's name before its arguments, so a call whose
/// output was cut short or failed to parse arrives without them. Like the
/// calls of a truncated or failed generation, it must not be dispatched.
pub(super) fn has_unfinished_mcp_call(
    calls: &[ExtractedToolCall],
    session: &McpToolSession<'_>,
) -> bool {
    calls.iter().any(|call| {
        session.has_exposed_tool(&call.name)
            && serde_json::from_str::<Value>(&call.arguments).is_err()
    })
}

/// Extract all tool calls from chat response (for parallel tool call support)
pub(super) fn extract_all_tool_calls_from_chat(
    response: &openai_protocol::chat::ChatCompletionResponse,
) -> Vec<ExtractedToolCall> {
    // Check if response has choices with tool calls
    let Some(choice) = response.choices.first() else {
        return Vec::new();
    };
    let message = &choice.message;

    // Look for tool_calls in the message
    if let Some(tool_calls) = &message.tool_calls {
        tool_calls
            .iter()
            .map(|tool_call| ExtractedToolCall {
                call_id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call
                    .function
                    .arguments
                    .clone()
                    .unwrap_or_else(|| "{}".to_string()),
            })
            .collect()
    } else {
        Vec::new()
    }
}

/// The model's argument payload as the object the hosted-tool merge needs,
/// with the request's hosted-tool config and user merged in. A scalar, an
/// array or null becomes an empty object first, so caller-declared config is
/// applied instead of silently dropped (`apply_hosted_tool_overrides` is a
/// no-op on anything but an object).
pub(super) fn dispatch_arguments(
    response_format: ResponseFormat,
    request: &ResponsesRequest,
    arguments: &str,
) -> Value {
    let mut arguments = match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => Value::Object(map),
        _ => json!({}),
    };
    prepare_hosted_dispatch_args(
        &mut arguments,
        response_format,
        request.tools.as_deref().unwrap_or(&[]),
        request.user.as_deref(),
    );
    arguments
}

/// An executed MCP call's metrics and its place in the loop's history;
/// `request` is the loop's current request (its model labels the metrics, its
/// tools name the call's namespace).
pub(super) fn record_mcp_call(
    state: &mut ToolLoopState,
    request: &ResponsesRequest,
    output: ToolExecutionOutput,
    output_str: String,
    output_item: ResponseOutputItem,
) {
    let success = !output.is_error;
    let model = request.model.as_str();
    Metrics::record_mcp_tool_duration(model, &output.tool_name, output.duration);
    Metrics::record_mcp_tool_call(
        model,
        &output.tool_name,
        if success {
            metrics_labels::RESULT_SUCCESS
        } else {
            metrics_labels::RESULT_ERROR
        },
    );
    state.record_call(
        request.tools.as_deref(),
        output.call_id,
        output.tool_name,
        output.arguments_str,
        output_str,
        output_item,
    );
}

pub(super) fn convert_mcp_tools_to_chat_tools(session: &McpToolSession<'_>) -> Vec<Tool> {
    openai_bridge::chat_function_tools(session)
}

/// Build next request with updated conversation history: the input the loop
/// started from, then every call this loop executed with its output. The
/// items the request already carries are moved into the next request as they
/// are (the loaded input is turned into items once, on the first build) and
/// the calls recorded since the last build are moved in behind them: no
/// iteration copies the base input or the calls before it.
pub(super) fn build_next_request(
    state: &mut ToolLoopState,
    current_request: ResponsesRequest,
) -> ResponsesRequest {
    let mut input_items = current_input_items(current_request.input);
    input_items.append(&mut state.calls_to_carry);

    // Build new request for next iteration, moving fields from old request to avoid cloning
    ResponsesRequest {
        input: ResponseInput::Items(input_items),
        model: current_request.model,
        instructions: current_request.instructions,
        tools: current_request.tools,
        max_output_tokens: current_request.max_output_tokens,
        temperature: current_request.temperature,
        top_p: current_request.top_p,
        stream: current_request.stream,
        store: Some(false), // Don't store intermediate responses
        max_tool_calls: current_request.max_tool_calls,
        tool_choice: current_request.tool_choice,
        parallel_tool_calls: current_request.parallel_tool_calls,
        previous_response_id: None,
        conversation: None,
        user: current_request.user,
        metadata: current_request.metadata,
        include: current_request.include,
        background: current_request.background,
        reasoning: current_request.reasoning,
        service_tier: current_request.service_tier,
        top_logprobs: current_request.top_logprobs,
        truncation: current_request.truncation,
        text: current_request.text,
        request_id: None,
        priority: current_request.priority,
        frequency_penalty: current_request.frequency_penalty,
        presence_penalty: current_request.presence_penalty,
        stop: current_request.stop,
        // Responses API top-level fields (P2): propagate per-request knobs so
        // multi-turn tool-loop continuations keep the same prompt template,
        // cache key, safety identifier, streaming options, and context-
        // management config as the original request.
        prompt: current_request.prompt,
        prompt_cache_key: current_request.prompt_cache_key,
        prompt_cache_retention: current_request.prompt_cache_retention,
        prompt_cache_options: current_request.prompt_cache_options,
        background_options: current_request.background_options,
        moderation: current_request.moderation,
        image: current_request.image,
        access_programs: current_request.access_programs,
        safety_identifier: current_request.safety_identifier,
        stream_options: current_request.stream_options,
        context_management: current_request.context_management,
        client_metadata: current_request.client_metadata,
        top_k: current_request.top_k,
        min_p: current_request.min_p,
        repetition_penalty: current_request.repetition_penalty,
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/regular/responses/common/namespace_tests.rs"]
mod namespace_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/regular/responses/common/final_turn_tests.rs"]
mod final_turn_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/regular/responses/common/usage_tests.rs"]
mod usage_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/regular/responses/common/history_tests.rs"]
mod history_tests;
