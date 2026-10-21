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
    common::{Tool, ToolChoice, ToolChoiceValue, Usage},
    responses::{
        self, InputTokensDetails, OutputTokensDetails, ResponseInput, ResponseInputOutputItem,
        ResponseOutputItem, ResponseUsage, ResponsesRequest,
    },
};
use serde_json::{json, Value};
use smg_mcp::{McpToolSession, ToolExecutionOutput};
use tracing::{trace, warn};

use crate::{
    middleware::TenantRequestMeta,
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            mcp_utils::{prepare_hosted_dispatch_args, DEFAULT_MAX_ITERATIONS},
            openai_bridge::{self, ResponseFormat},
        },
        grpc::common::responses::{history::current_input_items, utils::resolve_function_identity},
    },
};

// ============================================================================
// Tool Loop State
// ============================================================================

/// State for tracking multi-turn tool calling loop
pub(super) struct ToolLoopState {
    tools: Option<Vec<responses::ResponseTool>>,
    pub iteration: usize,
    pub total_calls: usize,
    pub usage: Option<ResponseUsage>,
    /// The calls this loop executed, each with its output, in order.
    pub conversation_history: Vec<ResponseInputOutputItem>,
    /// The input the loop started from, the request's history loaded: every
    /// later iteration is rebuilt on it, so a stored turn's tool calls and
    /// their outputs stay in front of the calls this loop executes.
    pub base_input: ResponseInput,
    pub mcp_call_items: Vec<ResponseOutputItem>,
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
    /// `request` is the request the loop starts from, its history loaded.
    pub fn new(request: &ResponsesRequest) -> Self {
        Self {
            tools: request.tools.clone(),
            iteration: 0,
            total_calls: 0,
            usage: None,
            conversation_history: Vec::new(),
            base_input: request.input.clone(),
            mcp_call_items: Vec::new(),
        }
    }

    /// Usage is per model request; missing reports must not erase prior totals.
    pub fn record_usage(&mut self, usage: Option<&Usage>) {
        let Some(usage) = usage else {
            return;
        };
        let total = self.usage.get_or_insert_with(|| ResponseUsage {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            input_tokens_details: InputTokensDetails::default(),
            output_tokens_details: OutputTokensDetails::default(),
        });
        total.input_tokens = total.input_tokens.saturating_add(usage.prompt_tokens);
        total.output_tokens = total.output_tokens.saturating_add(usage.completion_tokens);
        total.total_tokens = total.total_tokens.saturating_add(usage.total_tokens);
        if let Some(details) = &usage.prompt_tokens_details {
            total.input_tokens_details.cached_tokens = total
                .input_tokens_details
                .cached_tokens
                .saturating_add(details.cached_tokens);
        }
        if let Some(tokens) = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens)
        {
            total.output_tokens_details.reasoning_tokens = total
                .output_tokens_details
                .reasoning_tokens
                .saturating_add(tokens);
        }
    }

    pub fn record_call(
        &mut self,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
        output_item: ResponseOutputItem,
        _success: bool,
    ) {
        // Add function_tool_call item with both arguments and output
        let id = call_id.clone();
        let (name, namespace) = resolve_function_identity(self.tools.as_deref(), &tool_name);
        self.conversation_history
            .push(ResponseInputOutputItem::FunctionToolCall {
                id: Some(id),
                call_id,
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
/// where the chat request is built: `conversions::responses_to_chat_with_mcp_tools`)
pub(super) fn prepare_tool_choice(chat_request: &mut ChatCompletionRequest, iteration: usize) {
    // Set tool_choice based on iteration
    // - Iteration 0: Use user's tool_choice or default to auto
    // - Iteration 1+: Always use auto to avoid infinite loops
    chat_request.tool_choice = if iteration == 0 {
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

/// Why a generated MCP batch must stop after its permitted calls execute.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum McpToolCallLimit {
    User,
    Safety,
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
pub(super) fn apply_mcp_tool_call_limit(
    calls: &mut Vec<ExtractedToolCall>,
    total_calls: usize,
    max_tool_calls: Option<usize>,
) -> Option<McpToolCallLimit> {
    let effective_limit = max_tool_calls
        .unwrap_or(DEFAULT_MAX_ITERATIONS)
        .min(DEFAULT_MAX_ITERATIONS);
    let remaining = effective_limit.saturating_sub(total_calls);
    if calls.len() <= remaining {
        return None;
    }

    if max_tool_calls.is_none_or(|limit| limit > DEFAULT_MAX_ITERATIONS) {
        warn!(
            total_calls,
            batch_size = calls.len(),
            effective_limit,
            "Internal MCP tool call safety limit reached"
        );
    }
    calls.truncate(remaining);
    Some(
        if max_tool_calls.is_some_and(|limit| limit <= DEFAULT_MAX_ITERATIONS) {
            McpToolCallLimit::User
        } else {
            McpToolCallLimit::Safety
        },
    )
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
    original_request: &ResponsesRequest,
    arguments: &str,
) -> Value {
    let mut arguments = match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => Value::Object(map),
        _ => json!({}),
    };
    prepare_hosted_dispatch_args(
        &mut arguments,
        response_format,
        original_request.tools.as_deref().unwrap_or(&[]),
        original_request.user.as_deref(),
    );
    arguments
}

/// An executed MCP call's metrics and its place in the loop's history.
pub(super) fn record_mcp_call(
    state: &mut ToolLoopState,
    model: &str,
    output: ToolExecutionOutput,
    output_str: String,
    output_item: ResponseOutputItem,
) {
    let success = !output.is_error;
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
        output.call_id,
        output.tool_name,
        output.arguments_str,
        output_str,
        output_item,
        success,
    );
}

pub(super) fn convert_mcp_tools_to_chat_tools(session: &McpToolSession<'_>) -> Vec<Tool> {
    openai_bridge::chat_function_tools(session)
}

/// Build next request with updated conversation history: the input the loop
/// started from, then every call this loop executed with its output.
pub(super) fn build_next_request(
    state: &ToolLoopState,
    current_request: ResponsesRequest,
) -> ResponsesRequest {
    let mut input_items = current_input_items(&state.base_input);
    input_items.extend_from_slice(&state.conversation_history);

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
mod namespace_tests {
    use super::*;
    use crate::routers::grpc::common::responses::utils::namespace_test_request;
    #[test]
    fn namespace_replay_preserves_structured_identity() {
        let request: ResponsesRequest = namespace_test_request();
        let mut state = ToolLoopState::new(&request);
        state.record_call(
            "call_test".into(),
            "weather.lookup".into(),
            "{}".into(),
            "sunny".into(),
            ResponseOutputItem::new_function_tool_call(
                "fc_test".into(),
                "call_test".into(),
                "weather.lookup".into(),
                "{}".into(),
                None,
                "completed".into(),
            ),
            true,
        );
        let wire = serde_json::to_value(&state.conversation_history[0]).unwrap();

        assert_eq!(wire["name"], "lookup");
        assert_eq!(wire["namespace"], "weather");
        assert_eq!(wire["call_id"], "call_test");
    }
}

#[cfg(test)]
mod usage_tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn dispatch_arguments_keep_an_object_and_make_an_empty_one_of_anything_else() {
        let request = ResponsesRequest::default();
        assert_eq!(
            dispatch_arguments(ResponseFormat::Passthrough, &request, r#"{"q": "x"}"#),
            json!({"q": "x"})
        );
        for payload in ["\"text\"", "[1, 2]", "null", "not json"] {
            assert_eq!(
                dispatch_arguments(ResponseFormat::Passthrough, &request, payload),
                json!({}),
                "{payload}"
            );
        }
    }

    #[test]
    fn dispatch_arguments_merge_the_requests_hosted_tool_config_and_user_into_a_coerced_payload() {
        // The reason for the coercion: a scalar payload under a hosted format
        // still takes the caller-declared config and the request's user.
        let request = ResponsesRequest {
            tools: Some(vec![responses::ResponseTool::WebSearchPreview(
                responses::WebSearchPreviewTool {
                    search_context_size: Some("low".to_string()),
                    ..Default::default()
                },
            )]),
            user: Some("u-1".to_string()),
            ..Default::default()
        };
        let merged = dispatch_arguments(ResponseFormat::WebSearchCall, &request, "\"text\"");
        assert_eq!(merged["search_context_size"], json!("low"));
        assert_eq!(merged["user"], json!("u-1"));
    }

    #[test]
    fn mcp_usage_accumulates_reasoning_and_cache_and_survives_missing_usage() {
        let mut state = ToolLoopState::new(&ResponsesRequest::default());
        state.record_usage(None);
        assert!(state.usage.is_none());
        state.record_usage(Some(
            &Usage::from_counts(12, 7)
                .with_cached_tokens(3)
                .with_reasoning_tokens(2),
        ));
        state.record_usage(Some(
            &Usage::from_counts(5, 4)
                .with_cached_tokens(0)
                .with_reasoning_tokens(1),
        ));
        state.record_usage(None);
        assert_eq!(
            serde_json::to_value(state.usage).unwrap(),
            json!({
                "input_tokens":17, "output_tokens":11, "total_tokens":28,
                "input_tokens_details":{"cached_tokens":3},
                "output_tokens_details":{"reasoning_tokens":3},
            })
        );
    }
}

#[cfg(test)]
mod history_tests {
    //! The tool loop is rebuilt on the loaded request: a stored turn's tool
    //! calls and outputs stay in front of the calls this loop executes.

    use openai_protocol::{chat::ChatMessage, common::ConversationRef};

    use super::{super::conversions, *};
    use crate::routers::grpc::common::responses::{
        history::tests::{function_call_output, function_call_output_input, shape, Stores},
        load_history,
    };

    /// Turn 2 answers the call a stored turn issued: the engine sees the
    /// question, the call and its answer on the first iteration, and the
    /// request rebuilt after a call this loop executed still carries them,
    /// in front of that call.
    #[tokio::test]
    async fn the_tool_loop_rebuilds_on_the_loaded_history() {
        let stores = Stores::with_conversation().await;
        let turn_1 = ResponsesRequest {
            input: ResponseInput::Text("What is 6*7?".to_string()),
            conversation: Some(ConversationRef::Id("conv_1".to_string())),
            ..Default::default()
        };
        stores
            .store_turn("resp_1", &turn_1, function_call_output())
            .await;
        let turn_2 = ResponsesRequest {
            input: function_call_output_input(),
            conversation: Some(ConversationRef::Id("conv_1".to_string())),
            ..Default::default()
        };
        let loaded = load_history(stores.history(), &turn_2, None)
            .await
            .expect("the conversation loads");

        let chat = conversions::responses_to_chat(&loaded).expect("the loaded history converts");
        let turns: Vec<&str> = chat
            .messages
            .iter()
            .map(|message| match message {
                ChatMessage::User { .. } => "user",
                ChatMessage::Assistant {
                    tool_calls: Some(_),
                    ..
                } => "assistant+tool_calls",
                ChatMessage::Tool { .. } => "tool",
                _ => "other",
            })
            .collect();
        assert_eq!(turns, ["user", "assistant+tool_calls", "tool"]);

        let mut state = ToolLoopState::new(&loaded);
        state.record_call(
            "call_B".to_string(),
            "search".to_string(),
            "{}".to_string(),
            "found".to_string(),
            ResponseOutputItem::new_function_tool_call(
                "fc_B".to_string(),
                "call_B".to_string(),
                "search".to_string(),
                "{}".to_string(),
                None,
                "completed".to_string(),
            ),
            true,
        );
        let next = build_next_request(&state, loaded);

        let ResponseInput::Items(items) = &next.input else {
            panic!("the next request's input is an item list");
        };
        assert_eq!(
            shape(items),
            [
                "message:user",
                "function_call:call_A",
                "function_call_output:call_A",
                "function_call:call_B+output",
            ]
        );
        assert_eq!(next.conversation, None);
        assert_eq!(next.store, Some(false));
    }

    /// A chain longer than the store keeps reloads as its newest turns, and
    /// the oldest turn kept opens with the result of a call the dropped turn
    /// issued: the loader leaves that result out, so a long tool loop keeps
    /// converting under the contract profiles instead of being refused, from
    /// then on, for a call the client never sent in the request.
    #[tokio::test]
    async fn a_chain_cut_at_the_stores_depth_cap_converts_without_the_dropped_calls_result() {
        use openai_protocol::responses::StringOrContentParts;
        use serde_json::json;

        let stores = Stores::with_conversation().await;
        let call = |k: usize| {
            json!([{
                "type": "function_call", "id": format!("fc_{k}"), "call_id": format!("call_{k}"),
                "name": "calc", "arguments": "{}", "status": "completed"
            }])
        };
        let result = |k: usize| {
            ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
                id: None,
                call_id: format!("call_{k}"),
                output: StringOrContentParts::String("ok".to_string()),
                status: None,
            }])
        };
        // Turn 1 asks and gets call_1; turn k answers call_{k-1} and gets
        // call_k: 101 turns, one more than the memory store's chain keeps.
        stores
            .store_turn(
                "resp_1",
                &ResponsesRequest {
                    model: "deepseek-flash".to_string(),
                    input: ResponseInput::Text("Start.".to_string()),
                    ..Default::default()
                },
                call(1),
            )
            .await;
        for k in 2..=101 {
            stores
                .store_turn(
                    &format!("resp_{k}"),
                    &ResponsesRequest {
                        model: "deepseek-flash".to_string(),
                        input: result(k - 1),
                        previous_response_id: Some(format!("resp_{}", k - 1)),
                        ..Default::default()
                    },
                    call(k),
                )
                .await;
        }
        let turn_102 = ResponsesRequest {
            model: "deepseek-flash".to_string(),
            input: result(101),
            previous_response_id: Some("resp_101".to_string()),
            ..Default::default()
        };

        let loaded = load_history(stores.history(), &turn_102, None)
            .await
            .expect("the chain loads");

        let ResponseInput::Items(items) = &loaded.input else {
            panic!("the loaded input is an item list");
        };
        let shapes = shape(items);
        // resp_2 to resp_101 were kept; resp_2's input, the result of call_1,
        // answers a call that is gone and is left out.
        assert_eq!(
            shapes.first().map(String::as_str),
            Some("function_call:call_2")
        );
        assert!(!shapes.iter().any(|s| s == "function_call_output:call_1"));
        assert_eq!(shapes.len(), 200);
        assert_eq!(
            shapes.last().map(String::as_str),
            Some("function_call_output:call_101")
        );
        let chat = conversions::responses_to_chat(&loaded)
            .expect("the reloaded chain converts under the contract");
        assert!(matches!(
            chat.messages.last(),
            Some(ChatMessage::Tool { tool_call_id, .. }) if tool_call_id == "call_101"
        ));
    }

    /// A client's own tool result with no call before it, sent with a
    /// `previous_response_id`, is not the loader's to drop: it stays after the
    /// chain's items in the loaded input, and the contract profile refuses it
    /// as the public API does. (The loader judges the chain's items alone.)
    #[tokio::test]
    async fn a_clients_own_result_after_a_chain_keeps_its_refusal() {
        use openai_protocol::responses::StringOrContentParts;
        use serde_json::json;

        let stores = Stores::with_conversation().await;
        stores
            .store_turn(
                "resp_1",
                &ResponsesRequest {
                    model: "deepseek-flash".to_string(),
                    input: ResponseInput::Text("Start.".to_string()),
                    ..Default::default()
                },
                json!([{
                    "type": "function_call", "id": "fc_1", "call_id": "call_1",
                    "name": "calc", "arguments": "{}", "status": "completed"
                }]),
            )
            .await;
        let turn_2 = ResponsesRequest {
            model: "deepseek-flash".to_string(),
            input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionCallOutput {
                id: None,
                call_id: "call_unknown".to_string(),
                output: StringOrContentParts::String("ok".to_string()),
                status: None,
            }]),
            previous_response_id: Some("resp_1".to_string()),
            ..Default::default()
        };

        let loaded = load_history(stores.history(), &turn_2, None)
            .await
            .expect("the chain loads");

        let ResponseInput::Items(items) = &loaded.input else {
            panic!("the loaded input is an item list");
        };
        let shapes = shape(items);
        assert_eq!(
            shapes.last().map(String::as_str),
            Some("function_call_output:call_unknown"),
            "{shapes:?}"
        );
        let refused = conversions::responses_to_chat(&loaded)
            .expect_err("a result no call before it issued is a malformed history");
        assert!(refused.contains("call_unknown"), "{refused}");
    }

    /// A stored turn keeps its `item_reference`s as the client sent them, so
    /// the call a stored result answers may sit behind a reference: the
    /// loader resolves the references first and judges the results after,
    /// and the pair reaches the engine intact.
    #[tokio::test]
    async fn a_stored_result_whose_call_sits_behind_a_reference_keeps_its_call() {
        use openai_protocol::responses::StringOrContentParts;
        use serde_json::json;

        let stores = Stores::with_conversation().await;
        // resp_1: the question, answered by call_1 (item fc_1).
        stores
            .store_turn(
                "resp_1",
                &ResponsesRequest {
                    model: "deepseek-flash".to_string(),
                    input: ResponseInput::Text("Start.".to_string()),
                    ..Default::default()
                },
                json!([{
                    "type": "function_call", "id": "fc_1", "call_id": "call_1",
                    "name": "calc", "arguments": "{}", "status": "completed"
                }]),
            )
            .await;
        // resp_2, no chain: the client references the call and sends its
        // result; the turn is stored with the reference unresolved.
        let reference: ResponseInputOutputItem =
            serde_json::from_value(json!({"type": "item_reference", "id": "fc_1"}))
                .expect("an item reference");
        stores
            .store_turn(
                "resp_2",
                &ResponsesRequest {
                    model: "deepseek-flash".to_string(),
                    input: ResponseInput::Items(vec![
                        reference,
                        ResponseInputOutputItem::FunctionCallOutput {
                            id: None,
                            call_id: "call_1".to_string(),
                            output: StringOrContentParts::String("42".to_string()),
                            status: None,
                        },
                    ]),
                    ..Default::default()
                },
                json!([{
                    "type": "message", "id": "msg_2", "role": "assistant",
                    "content": [{"type": "output_text", "text": "42 it is.", "annotations": []}]
                }]),
            )
            .await;
        let turn_3 = ResponsesRequest {
            model: "deepseek-flash".to_string(),
            input: ResponseInput::Text("Thanks.".to_string()),
            previous_response_id: Some("resp_2".to_string()),
            ..Default::default()
        };

        let loaded = load_history(stores.history(), &turn_3, None)
            .await
            .expect("the chain loads");

        let ResponseInput::Items(items) = &loaded.input else {
            panic!("the loaded input is an item list");
        };
        assert_eq!(
            shape(items),
            [
                "function_call:call_1",
                "function_call_output:call_1",
                "message:assistant",
                "message:user"
            ]
        );
        let chat =
            conversions::responses_to_chat(&loaded).expect("the pair converts under the contract");
        assert!(chat.messages.iter().any(|message| matches!(
            message,
            ChatMessage::Tool { tool_call_id, .. } if tool_call_id == "call_1"
        )));
    }
}
