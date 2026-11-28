//! MCP (Model Context Protocol) Integration Module
//!
//! This module contains all MCP-related functionality for the OpenAI router:
//! - Tool loop state management for multi-turn tool calling
//! - MCP tool execution and result handling
//! - Output item builders for MCP-specific response formats
//! - SSE event generation for streaming MCP operations
//! - Payload transformation for MCP tool interception
//! - Metadata injection for MCP operations

use std::collections::HashSet;

use axum::http::{HeaderMap, HeaderValue};
use bytes::Bytes;
use openai_protocol::{
    event_types::{is_function_call_type, ItemType, McpEvent, OutputItemEvent},
    responses::{generate_id, McpToolCallError, ResponseTool, ResponsesRequest},
};
use serde::Serialize;
use serde_json::{json, to_value, Value};
use smg_mcp::{McpServerBinding, McpToolSession, ToolExecutionInput, ToolExecutionResult};
use tracing::{debug, info, warn};

use super::tool_handler::FunctionCallInProgress;
use crate::{
    header_utils::ApiProvider,
    mcp_utils::{prepare_hosted_dispatch_args, DEFAULT_MAX_ITERATIONS},
    metrics::{metrics_labels, Metrics},
    openai::responses::utils,
    openai_bridge,
    openai_bridge::{
        extract_embedded_openai_responses, mcp_response_item_id, FormatRegistry, ResponseFormat,
        ResponseTransformer,
    },
    sse::{SseEncoder, SseSender},
    upstream_body::error_body_text,
};

/// State for tracking multi-turn tool calling loop
pub(crate) struct ToolLoopState {
    /// Current iteration number (starts at 0, increments with each tool call)
    pub iteration: usize,
    /// Total number of tool calls executed
    pub total_calls: usize,
    /// The calls this loop executed, each with its output, in order
    pub conversation_history: Vec<Value>,
    /// MCP bindings already represented by historical `mcp_list_tools` items.
    pub existing_mcp_list_tools_labels: HashSet<String>,
    /// Transformed output items (mcp_call, web_search_call, etc.) - stored to avoid reconstruction
    pub mcp_call_items: Vec<Value>,
    /// The output index the client read for each of `mcp_call_items`, kept
    /// by the streamed loop alone (see [`Self::retain_streamed_item`]).
    streamed_item_indices: Vec<usize>,
    /// The `mcp_list_tools` items this loop renders, built at their first use
    /// and kept (see [`Self::mcp_list_tools_items`]).
    mcp_list_tools_items: Option<Vec<Value>>,
    /// How many of `conversation_history`'s items the loop's payload carries
    /// already; a resume appends the rest.
    history_in_payload: usize,
}

impl ToolLoopState {
    pub fn new(prior_mcp_list_tools_labels: Vec<String>) -> Self {
        let known_labels = prior_mcp_list_tools_labels
            .into_iter()
            .collect::<HashSet<_>>();

        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            existing_mcp_list_tools_labels: known_labels,
            mcp_call_items: Vec::new(),
            streamed_item_indices: Vec::new(),
            mcp_list_tools_items: None,
            history_in_payload: 0,
        }
    }

    /// The `mcp_list_tools` items of the bindings this loop announces: every
    /// server of the session without an item in the history the request
    /// continues, the internal servers' left out (the stream announces what
    /// the final response and the stored record carry). Built once, each
    /// tool's schema copied into them once: the stream's start and the final
    /// response render the same items.
    pub fn mcp_list_tools_items(&mut self, session: &McpToolSession<'_>) -> &[Value] {
        self.mcp_list_tools_items.get_or_insert_with(|| {
            mcp_list_tools_bindings_to_emit(
                &self.existing_mcp_list_tools_labels,
                session.mcp_servers(),
            )
            .iter()
            .filter(|(server_label, _)| !session.is_internal_non_builtin_server_label(server_label))
            .map(|(server_label, server_key)| {
                openai_bridge::mcp_list_tools_json(session, server_label, server_key)
            })
            .collect()
        })
    }

    /// The kept `mcp_list_tools` items, moved out for a response's last
    /// render: the stored record on the streaming path; the answered, the
    /// approval or the tool-call-limit response on the non-streaming path.
    /// Nothing reads them afterwards, so that render owns them instead of
    /// copying every schema once more.
    pub fn take_mcp_list_tools_items(&mut self, session: &McpToolSession<'_>) -> Vec<Value> {
        self.mcp_list_tools_items(session);
        self.mcp_list_tools_items.take().unwrap_or_default()
    }

    /// The calls recorded since the loop's payload last took them, each with
    /// its output: the next resume payload appends them.
    pub fn history_to_append(&mut self) -> &[Value] {
        let from = self.history_in_payload.min(self.conversation_history.len());
        self.history_in_payload = self.conversation_history.len();
        &self.conversation_history[from..]
    }

    /// Record a tool call in the loop state
    ///
    /// Stores both the conversation history (for resume payloads) and the
    /// transformed output item (to avoid re-transformation later).
    pub fn record_call(
        &mut self,
        is_builtin_tool: bool,
        call_id: String,
        tool_name: String,
        args_json_str: String,
        output_str: String,
        transformed_item: Value,
    ) {
        let func_item = json!({
            "type": ItemType::FUNCTION_CALL,
            "call_id": call_id,
            "name": tool_name,
            "arguments": args_json_str
        });
        self.conversation_history.push(func_item);

        let output_item = json!({
            "type": ItemType::FUNCTION_CALL_OUTPUT,
            "call_id": call_id,
            "output": output_str
        });
        self.conversation_history.push(output_item);

        self.mcp_call_items.push(transformed_item);

        if is_builtin_tool {
            let openai_output_items = extract_openai_response_output_items(&output_str);
            if !openai_output_items.is_empty() {
                debug!(
                    call_id = %call_id,
                    extracted_items = openai_output_items.len(),
                    "Extracted intermediary OpenAI response items from MCP tool output"
                );
                self.mcp_call_items.extend(openai_output_items);
            }
        }
    }

    /// Says that the items recorded since the last one placed are those of
    /// the streamed call the client read at `output_index`: the call's own
    /// item and the items its output carried.
    fn streamed_call_recorded(&mut self, output_index: usize) {
        self.streamed_item_indices
            .resize(self.mcp_call_items.len(), output_index);
    }

    /// Keeps an item of a streamed turn that the last turn's own output
    /// does not carry (a call that did not run, an item a cut turn or the
    /// tool-call limit left half-open, an item the upstream closed behind a
    /// call of an earlier turn or of the turn that limit ended) for the
    /// final response, at
    /// the place of `output_index`, the index the client read for it: before
    /// the first kept item of a higher index. So the response lists a turn's
    /// items in the order the stream numbered them, whatever the order they
    /// were closed in.
    pub fn retain_streamed_item(&mut self, output_index: usize, item: Value) {
        let at = self
            .streamed_item_indices
            .iter()
            .position(|&index| index > output_index)
            .unwrap_or(self.streamed_item_indices.len());
        self.streamed_item_indices.insert(at, output_index);
        self.mcp_call_items.insert(at, item);
    }
}

fn extract_openai_response_output_items(output_str: &str) -> Vec<Value> {
    let result = match serde_json::from_str::<Value>(output_str) {
        Ok(value) => value,
        _ => return Vec::new(),
    };

    extract_embedded_openai_responses(&result)
        .into_iter()
        .filter_map(build_message_from_openai_response)
        .collect()
}

fn build_message_from_openai_response(openai_response: Value) -> Option<Value> {
    let obj = openai_response.as_object()?;

    let content_value = obj.get("content")?;

    let content = match content_value {
        Value::Array(items) => items.clone(),
        Value::Object(_) => vec![content_value.clone()],
        _ => return None,
    };

    if content.is_empty() {
        return None;
    }

    Some(json!({
        "id": generate_id("msg"),
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": content
    }))
}

/// Execute detected tool calls and send completion events to client.
///
/// `request_tools` carries the caller-declared `tools` list from the original
/// request so per-kind hosted-tool overrides can be merged into dispatch args
/// before [`McpToolSession::execute_tool`].
///
/// `request_user` is the request-level `user` identifier (OpenAI Responses API
/// `user` field), forwarded into hosted-tool dispatch args so the MCP server
/// can attribute usage. Plain MCP function tools (Passthrough format) are
/// not affected.
///
/// Returns false if client disconnected during execution
#[expect(
    clippy::too_many_arguments,
    reason = "The entry point keeps its flat signature for the streaming handler; the \
              per-request inputs are bundled into a `StreamingDispatch` once per batch \
              for the per-call step."
)]
pub(crate) async fn execute_streaming_tool_calls(
    pending_calls: Vec<FunctionCallInProgress>,
    session: &McpToolSession<'_>,
    format_registry: &FormatRegistry,
    tx: &SseSender,
    state: &mut ToolLoopState,
    sequence_number: &mut u64,
    model_id: &str,
    request_tools: &[ResponseTool],
    request_user: Option<&str>,
) -> bool {
    let dispatch = StreamingDispatch {
        session,
        format_registry,
        tx,
        model_id,
        request_tools,
        request_user,
    };
    for call in pending_calls {
        if call.name.is_empty() {
            warn!(
                "Skipping incomplete tool call: name is empty, args_len={}",
                call.arguments_buffer.len()
            );
            continue;
        }
        if !dispatch_streaming_call(call, &dispatch, state, sequence_number).await {
            return false;
        }
    }
    true
}

/// What every call of a streamed batch is dispatched against: the session
/// and the format registry that resolve the tool, the channel the call's
/// events go down, and the request's model, declared tools and user, which
/// the prepared arguments and the metrics read.
struct StreamingDispatch<'a> {
    session: &'a McpToolSession<'a>,
    format_registry: &'a FormatRegistry,
    tx: &'a SseSender,
    model_id: &'a str,
    request_tools: &'a [ResponseTool],
    request_user: Option<&'a str>,
}

/// One call of a streamed batch dispatched and recorded, the shape of
/// [`dispatch_call`] with the stream's events around it: arguments the model
/// wrote that are not JSON close the call on the wire with an error item and
/// record it; otherwise the format's intermediate event goes out, the
/// prepared arguments go to the session, the result is recorded with its
/// metrics, and the call is closed on the wire with its item. Returns false
/// when the client left.
async fn dispatch_streaming_call(
    call: FunctionCallInProgress,
    dispatch: &StreamingDispatch<'_>,
    state: &mut ToolLoopState,
    sequence_number: &mut u64,
) -> bool {
    let StreamingDispatch {
        session,
        format_registry,
        tx,
        model_id,
        request_tools,
        request_user,
    } = *dispatch;

    info!(
        "Executing tool call during streaming: {} ({})",
        call.name, call.call_id
    );

    let args_str = if call.arguments_buffer.is_empty() {
        "{}"
    } else {
        &call.arguments_buffer
    };

    let response_format = openai_bridge::lookup_tool_format(session, format_registry, &call.name);
    let server_label = session.resolve_tool_server_label(&call.name);

    let arguments = match prepared_arguments(args_str, response_format, request_tools, request_user)
    {
        Ok(v) => v,
        Err(e) => {
            warn!(tool = %call.name, error = %e, "Failed to parse tool arguments as JSON");
            let err_str = invalid_tool_arguments(&e);
            record_parse_failure(model_id, &call.name);
            let error_output = json!({ "error": &err_str });
            let mcp_call_item = streaming_call_item(
                build_transformed_mcp_call_item(
                    &error_output,
                    response_format,
                    &call.call_id,
                    &server_label,
                    &call.name,
                    &call.arguments_buffer,
                ),
                &call,
                response_format,
            );
            return close_streaming_call(
                call,
                mcp_call_item,
                error_output.to_string(),
                response_format,
                dispatch,
                state,
                sequence_number,
            )
            .await;
        }
    };

    if !send_tool_call_intermediate_event(tx, &call, response_format, sequence_number).await {
        return false;
    }

    // Log the effective (post-merge) args so the log reflects what the
    // MCP server actually receives, not the pre-merge string from the model.
    debug!("Calling MCP tool '{}' with args: {}", call.name, arguments);
    let tool_output = session
        .execute_tool(ToolExecutionInput {
            call_id: call.call_id.clone(),
            tool_name: call.name.clone(),
            arguments,
        })
        .await;

    record_tool_metrics(model_id, &tool_output);

    let output_str = tool_output.output.to_string();
    let mcp_call_item = streaming_call_item(
        to_value(openai_bridge::transform_tool_output(
            &tool_output,
            response_format,
        ))
        .unwrap_or_else(|e| {
            warn!(tool = %call.name, error = %e, "Failed to convert item to Value");
            json!({})
        }),
        &call,
        response_format,
    );
    close_streaming_call(
        call,
        mcp_call_item,
        output_str,
        response_format,
        dispatch,
        state,
        sequence_number,
    )
    .await
}

/// A streamed call's item under the id the stream announced for it.
fn streaming_call_item(
    mut item: Value,
    call: &FunctionCallInProgress,
    response_format: ResponseFormat,
) -> Value {
    if let Some(obj) = item.as_object_mut() {
        obj.insert(
            "id".to_string(),
            Value::String(stable_streaming_tool_item_id(call, response_format)),
        );
    }
    item
}

/// A streamed call closed on the wire with its item (the format's completed
/// event, or `response.mcp_call.failed` for an `mcp_call` that failed, then
/// the item's `response.output_item.done`) and recorded in the loop's state,
/// as the non-streamed loop records a call. Returns false when the client
/// left.
async fn close_streaming_call(
    call: FunctionCallInProgress,
    mcp_call_item: Value,
    output_str: String,
    response_format: ResponseFormat,
    dispatch: &StreamingDispatch<'_>,
    state: &mut ToolLoopState,
    sequence_number: &mut u64,
) -> bool {
    if !send_tool_call_completion_events(
        dispatch.tx,
        &call,
        &mcp_call_item,
        response_format,
        sequence_number,
    )
    .await
    {
        return false;
    }
    let output_index = call.effective_output_index();
    state.record_call(
        dispatch.session.is_builtin_tool(&call.name),
        call.call_id,
        call.name,
        call.arguments_buffer,
        output_str,
        mcp_call_item,
    );
    state.streamed_call_recorded(output_index);
    true
}

/// The calls the streamed loop does not run: those of a batch beyond the
/// request's tool-call allowance ([`NOT_EXECUTED_AT_TOOL_CALL_LIMIT`]), or a
/// call an upstream turn left half-open when its stream broke
/// ([`NOT_EXECUTED_AT_STREAM_BREAK`]); `error` says which. None runs. Each
/// is closed on the wire under the type the stream announced for it, with
/// the item of [`not_executed_tool_call_item`] (the item the non-streamed
/// loop renders for a call past the allowance): an `mcp_call` gets
/// `response.mcp_call.failed` and then its `response.output_item.done`; a
/// call in a hosted format gets its `response.output_item.done` alone (the
/// API has no failed sub-event for those item types; the item's status says
/// it). So no item the stream announced stays open, and each takes the place
/// of its output index among the final response's items (see
/// [`ToolLoopState::retain_streamed_item`]). Returns false when the client
/// left.
pub(crate) async fn skip_streaming_tool_calls(
    skipped: Vec<FunctionCallInProgress>,
    error: &str,
    session: &McpToolSession<'_>,
    format_registry: &FormatRegistry,
    tx: &SseSender,
    state: &mut ToolLoopState,
    sequence_number: &mut u64,
) -> bool {
    for call in skipped {
        let response_format =
            openai_bridge::lookup_tool_format(session, format_registry, &call.name);
        let server_label = session.resolve_tool_server_label(&call.name);
        let arguments = if call.arguments_buffer.is_empty() {
            "{}"
        } else {
            &call.arguments_buffer
        };
        let item_id = stable_streaming_tool_item_id(&call, response_format);
        let mut item = not_executed_tool_call_item(
            response_format,
            call.item_id.as_deref().unwrap_or(call.call_id.as_str()),
            &server_label,
            &call.name,
            arguments,
            error,
        );
        if let Some(obj) = item.as_object_mut() {
            obj.insert("id".to_string(), Value::String(item_id.clone()));
        }
        let output_index = call.effective_output_index();

        if response_format == ResponseFormat::Passthrough {
            let failed_payload = json!({
                "type": McpEvent::CALL_FAILED,
                "sequence_number": *sequence_number,
                "output_index": output_index,
                "item_id": item_id
            });
            *sequence_number += 1;
            let failed_event = format!(
                "event: {}\ndata: {}\n\n",
                McpEvent::CALL_FAILED,
                failed_payload
            );
            if tx.send(Ok(Bytes::from(failed_event))).await.is_err() {
                return false;
            }
        }

        let done_payload = json!({
            "type": OutputItemEvent::DONE,
            "sequence_number": *sequence_number,
            "output_index": output_index,
            "item": item
        });
        *sequence_number += 1;
        let done_event = format!(
            "event: {}\ndata: {}\n\n",
            OutputItemEvent::DONE,
            done_payload
        );
        if tx.send(Ok(Bytes::from(done_event))).await.is_err() {
            return false;
        }

        state.retain_streamed_item(output_index, item);
    }
    true
}

/// The items other than function calls that an upstream turn left half-open
/// when it was cut behind a call that ran or stopped there at the tool-call
/// limit, as
/// [`super::StreamingToolHandler::take_open_items`] hands them over: each gets the
/// `response.output_item.done` the upstream did not write, with the item as
/// closed there, and takes the place of its output index among the final
/// response's items (see [`ToolLoopState::retain_streamed_item`]). Returns
/// false when the client left.
pub(crate) async fn close_streaming_open_items(
    open_items: Vec<(usize, Value)>,
    tx: &SseSender,
    state: &mut ToolLoopState,
    sequence_number: &mut u64,
) -> bool {
    for (output_index, item) in open_items {
        let done_payload = json!({
            "type": OutputItemEvent::DONE,
            "sequence_number": *sequence_number,
            "output_index": output_index,
            "item": item
        });
        *sequence_number += 1;
        let done_event = format!(
            "event: {}\ndata: {}\n\n",
            OutputItemEvent::DONE,
            done_payload
        );
        if tx.send(Ok(Bytes::from(done_event))).await.is_err() {
            return false;
        }

        state.retain_streamed_item(output_index, item);
    }
    true
}

/// Transform payload to replace MCP/builtin tools with function tools.
///
/// Retains existing function tools from the request, removes non-function tools
/// (MCP, builtin), and appends function tools for discovered MCP server tools.
pub(crate) fn prepare_mcp_tools_as_functions(payload: &mut Value, session: &McpToolSession<'_>) {
    let Some(obj) = payload.as_object_mut() else {
        return;
    };

    let mut retained_tools: Vec<Value> = Vec::new();
    if let Some(v) = obj.get_mut("tools") {
        if let Some(arr) = v.as_array_mut() {
            retained_tools = arr
                .drain(..)
                .filter(|item| {
                    item.get("type")
                        .and_then(|v| v.as_str())
                        .map(|s| s == ItemType::FUNCTION)
                        .unwrap_or(false)
                })
                .collect();
        }
    }

    let session_tools = openai_bridge::function_tools_json(session);
    let mut tools_json = Vec::with_capacity(retained_tools.len() + session_tools.len());
    tools_json.append(&mut retained_tools);
    tools_json.extend(session_tools);

    if !tools_json.is_empty() {
        let tool_choice = remap_tool_choice_for_functions(obj.get("tool_choice"), &tools_json);
        obj.insert("tools".to_string(), Value::Array(tools_json));
        obj.insert("tool_choice".to_string(), tool_choice);
    }
}

/// Remap an explicit `tool_choice` through the MCP/hosted→function rewrite for
/// the initial request; resume payloads switch to "auto" so the loop can end.
///
/// String options pass through. Object choices resolve by name against the
/// rewritten function tools (hosted types like `image_generation` match the
/// function of the same name); anything unmappable falls back to "auto", since
/// the referenced tool no longer exists in `tools`.
fn remap_tool_choice_for_functions(tool_choice: Option<&Value>, tools: &[Value]) -> Value {
    let selected = match tool_choice {
        Some(Value::String(choice)) => return Value::String(choice.clone()),
        Some(choice) if choice.is_object() => {
            choice.get("name").and_then(Value::as_str).or_else(|| {
                choice
                    .get("type")
                    .and_then(Value::as_str)
                    .filter(|choice_type| *choice_type != ItemType::FUNCTION)
            })
        }
        _ => None,
    };

    match selected {
        Some(name)
            if tools
                .iter()
                .any(|tool| utils::function_tool_name(tool) == Some(name)) =>
        {
            json!({ "type": ItemType::FUNCTION, "name": name })
        }
        _ => Value::String("auto".to_string()),
    }
}

/// Turn the payload the loop just sent into the payload of its next turn, in
/// place: the calls executed since the last turn, each with its output, go
/// behind its input (the request's own, with the history it continues loaded
/// in front), the model is left free to answer, and the loop's own `stream`
/// and `store` are set. One payload serves every turn; its input items and
/// its tools are never copied.
pub(crate) fn resume_payload_in_place(
    payload: &mut Value,
    new_history: &[Value],
    is_streaming: bool,
) -> Result<(), String> {
    let obj = payload
        .as_object_mut()
        .ok_or_else(|| "payload not an object".to_string())?;

    // The input is taken out in place (a removal would move the last key into
    // its slot under the crate's ordered maps) and its items are moved into
    // the resumed list; the insert below fills the slot again.
    let mut input_array = match obj.get_mut("input").map(Value::take) {
        Some(Value::Array(items)) => items,
        Some(Value::String(text)) => vec![json!({
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": text }]
        })],
        _ => return Err("payload input is neither a string nor an item list".to_string()),
    };

    input_array.extend_from_slice(new_history);
    obj.insert("input".to_string(), Value::Array(input_array));

    if obj
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
    {
        // The tool call already happened; "auto" lets the model answer
        // instead of looping a preserved "required" until the call limit.
        obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
    }

    obj.insert("stream".to_string(), Value::Bool(is_streaming));
    obj.insert("store".to_string(), Value::Bool(false));

    Ok(())
}

/// Send the `mcp_list_tools` events of one item to the client at the start of
/// streaming: the item added empty, the listing in progress and completed
/// (or failed, when the item carries the listing's `error`), the item done
/// with its tools. The item is written to the wire where it stands (the loop
/// keeps it for the final response), not copied into the events. Returns
/// false if the client disconnected.
pub(crate) async fn send_mcp_list_tools_events(
    tx: &SseSender,
    enc: &mut SseEncoder,
    item: &Value,
    output_index: usize,
    sequence_number: &mut u64,
) -> bool {
    let item_id = item.get("id").and_then(Value::as_str).unwrap_or("");
    let server_label = item
        .get("server_label")
        .and_then(Value::as_str)
        .unwrap_or("");

    // Event 1: response.output_item.added with empty tools
    let event1_payload = json!({
        "type": OutputItemEvent::ADDED,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item": {
            "type": ItemType::MCP_LIST_TOOLS,
            "id": item_id,
            "server_label": server_label,
            "tools": []
        }
    });
    *sequence_number += 1;
    let event1 = format!(
        "event: {}\ndata: {}\n\n",
        OutputItemEvent::ADDED,
        event1_payload
    );
    if tx.send(Ok(Bytes::from(event1))).await.is_err() {
        return false; // Client disconnected
    }

    // Event 2: response.mcp_list_tools.in_progress
    let event2_payload = json!({
        "type": McpEvent::LIST_TOOLS_IN_PROGRESS,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item_id": item_id
    });
    *sequence_number += 1;
    let event2 = format!(
        "event: {}\ndata: {}\n\n",
        McpEvent::LIST_TOOLS_IN_PROGRESS,
        event2_payload
    );
    if tx.send(Ok(Bytes::from(event2))).await.is_err() {
        return false;
    }

    // Event 3: response.mcp_list_tools.completed, or .failed when the item
    // carries the listing's error (the done item below carries it to the
    // client; the event itself names only the outcome)
    let event3_type = if item.get("error").is_some_and(|error| !error.is_null()) {
        McpEvent::LIST_TOOLS_FAILED
    } else {
        McpEvent::LIST_TOOLS_COMPLETED
    };
    let event3_payload = json!({
        "type": event3_type,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item_id": item_id
    });
    *sequence_number += 1;
    let event3 = format!("event: {event3_type}\ndata: {event3_payload}\n\n");
    if tx.send(Ok(Bytes::from(event3))).await.is_err() {
        return false;
    }

    // Event 4: response.output_item.done with the full tools list, the item
    // serialised from where it stands
    let event4_payload = OutputItemEventPayload {
        event_type: OutputItemEvent::DONE,
        sequence_number: *sequence_number,
        output_index,
        item,
    };
    *sequence_number += 1;
    let event4 = match enc.encode_event(OutputItemEvent::DONE, &event4_payload) {
        Ok(bytes) => bytes,
        Err(error) => {
            warn!(
                "failed to encode the list-tools done event via SseEncoder, falling back: {error}"
            );
            Bytes::from(format!(
                "event: {}\ndata: {}\n\n",
                OutputItemEvent::DONE,
                json!({
                    "type": OutputItemEvent::DONE,
                    "sequence_number": event4_payload.sequence_number,
                    "output_index": output_index,
                    "item": item
                })
            ))
        }
    };
    tx.send(Ok(event4)).await.is_ok()
}

/// An `output_item` event's payload around a borrowed item: the item is
/// serialised onto the wire from where it stands, where `json!` would copy
/// it into the event first.
#[derive(Serialize)]
struct OutputItemEventPayload<'a> {
    #[serde(rename = "type")]
    event_type: &'static str,
    sequence_number: u64,
    output_index: usize,
    item: &'a Value,
}

/// Send intermediate event during tool execution (searching/interpreting).
/// Returns false if client disconnected.
async fn send_tool_call_intermediate_event(
    tx: &SseSender,
    call: &FunctionCallInProgress,
    response_format: ResponseFormat,
    sequence_number: &mut u64,
) -> bool {
    // mcp_call has no intermediate event (descriptor.searching_event = None).
    let Some(event_type) = openai_bridge::descriptor(response_format).searching_event else {
        return true;
    };

    let effective_output_index = call.effective_output_index();

    let item_id = stable_streaming_tool_item_id(call, response_format);

    let event_payload = json!({
        "type": event_type,
        "sequence_number": *sequence_number,
        "output_index": effective_output_index,
        "item_id": item_id
    });
    *sequence_number += 1;

    let event = format!("event: {event_type}\ndata: {event_payload}\n\n");
    tx.send(Ok(Bytes::from(event))).await.is_ok()
}

/// Send tool call completion events after tool execution.
/// Handles mcp_call, web_search_call, code_interpreter_call, file_search_call,
/// and image_generation_call items. An `mcp_call` whose item reads `failed`
/// closes under `response.mcp_call.failed`, as a call past the allowance
/// does ([`skip_streaming_tool_calls`]); the hosted formats have no failed
/// sub-event, so their item's status says it.
/// Returns false if client disconnected.
async fn send_tool_call_completion_events(
    tx: &SseSender,
    call: &FunctionCallInProgress,
    tool_call_item: &Value,
    response_format: ResponseFormat,
    sequence_number: &mut u64,
) -> bool {
    let effective_output_index = call.effective_output_index();
    let item_id = stable_streaming_tool_item_id(call, response_format);

    // Resolve the closing event from the item's `type` (the typed
    // `tool_call_item` may have been re-tagged after dispatch); fall back
    // to the format-derived event for unknown types. A failed `mcp_call`
    // has a sub-event of its own.
    let item_type = tool_call_item
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let failed_mcp_call = item_type == ItemType::MCP_CALL
        && tool_call_item.get("status").and_then(|v| v.as_str()) == Some("failed");
    let close_event_type: &str = if failed_mcp_call {
        McpEvent::CALL_FAILED
    } else {
        openai_bridge::format_from_type_str(item_type)
            .map(|f| openai_bridge::descriptor(f).completed_event)
            .unwrap_or_else(|| openai_bridge::descriptor(response_format).completed_event)
    };

    // Event 1: response.<type>.completed, or response.mcp_call.failed
    let close_payload = json!({
        "type": close_event_type,
        "sequence_number": *sequence_number,
        "output_index": effective_output_index,
        "item_id": item_id
    });
    *sequence_number += 1;

    let close_event = format!("event: {close_event_type}\ndata: {close_payload}\n\n");
    if tx.send(Ok(Bytes::from(close_event))).await.is_err() {
        return false;
    }

    // Event 2: response.output_item.done (with the call's final item)
    let done_payload = json!({
        "type": OutputItemEvent::DONE,
        "sequence_number": *sequence_number,
        "output_index": effective_output_index,
        "item": tool_call_item
    });
    *sequence_number += 1;

    let done_event = format!(
        "event: {}\ndata: {}\n\n",
        OutputItemEvent::DONE,
        done_payload
    );
    tx.send(Ok(Bytes::from(done_event))).await.is_ok()
}

fn stable_streaming_tool_item_id(
    call: &FunctionCallInProgress,
    response_format: ResponseFormat,
) -> String {
    let source_id = call.item_id.as_deref().unwrap_or(call.call_id.as_str());

    if response_format == ResponseFormat::Passthrough {
        // mcp_response_item_id encodes the `mcp_*` rewrite rules
        // (preserving an existing `mcp_` prefix instead of double-prefixing).
        mcp_response_item_id(source_id)
    } else {
        let prefix = openai_bridge::descriptor(response_format).id_prefix;
        normalize_tool_item_id_with_prefix(source_id, prefix)
    }
}

fn normalize_tool_item_id_with_prefix(source_id: &str, prefix: &str) -> String {
    let prefix_with_underscore = format!("{prefix}_");
    if source_id.starts_with(&prefix_with_underscore) {
        return source_id.to_string();
    }

    source_id
        .strip_prefix("fc_")
        .or_else(|| source_id.strip_prefix("call_"))
        .map(|stripped| format!("{prefix_with_underscore}{stripped}"))
        .unwrap_or_else(|| format!("{prefix_with_underscore}{source_id}"))
}

fn non_streaming_tool_item_id_source(item_id: &str, response_format: ResponseFormat) -> String {
    if response_format == ResponseFormat::Passthrough {
        item_id.to_string()
    } else {
        // Hosted-builtin formats strip the upstream function-call prefix; the
        // bridge's success builders re-add the format-specific prefix.
        item_id
            .strip_prefix("fc_")
            .or_else(|| item_id.strip_prefix("call_"))
            .unwrap_or(item_id)
            .to_string()
    }
}

fn approval_request_item_id_source(item_id: &str) -> String {
    normalize_tool_item_id_with_prefix(item_id, "mcpr")
}

pub(crate) fn mcp_list_tools_bindings_to_emit(
    existing_labels: &HashSet<String>,
    bindings: &[McpServerBinding],
) -> Vec<(String, String)> {
    bindings
        .iter()
        .filter(|binding| !existing_labels.contains(&binding.label))
        .map(|binding| (binding.label.clone(), binding.server_key.clone()))
        .collect()
}

/// Inject MCP metadata into a response's last render: the streamed loop's
/// final response (the wire and the record share it), the answered response
/// on the non-streaming path. The kept `mcp_list_tools` items move into it;
/// nothing reads them afterwards.
pub(crate) fn inject_mcp_metadata_last(
    response: &mut Value,
    state: &mut ToolLoopState,
    session: &McpToolSession<'_>,
) {
    let prefix = mcp_output_prefix_last(state, session, 0);
    inject_mcp_output_prefix(response, prefix);
}

fn inject_mcp_output_prefix(response: &mut Value, prefix: Vec<Value>) {
    if let Some(output_array) = response.get_mut("output").and_then(|v| v.as_array_mut()) {
        output_array.retain(|item| {
            item.get("type").and_then(|t| t.as_str()) != Some(ItemType::MCP_LIST_TOOLS)
        });
        output_array.splice(0..0, prefix);
    } else if let Some(obj) = response.as_object_mut() {
        obj.insert("output".to_string(), Value::Array(prefix));
    }
}

/// The prefix of a response's last render: the kept `mcp_list_tools` items
/// moved, not copied (the answered, the approval and the tool-call-limit
/// responses of the non-streaming path, the streamed loop's final response
/// that the wire and the record share), then the calls the loop executed
/// (the stored transformed items, no reconstruction), the internal servers'
/// left out; with room for `extra` items the caller appends. Nothing reads
/// the moved items afterwards.
fn mcp_output_prefix_last(
    state: &mut ToolLoopState,
    session: &McpToolSession<'_>,
    extra: usize,
) -> Vec<Value> {
    let list_tools_items = state.take_mcp_list_tools_items(session);
    mcp_output_prefix_with(list_tools_items, state, session, extra)
}

/// The prefix around the `mcp_list_tools` items the caller took, then the
/// executed calls.
fn mcp_output_prefix_with(
    list_tools_items: Vec<Value>,
    state: &ToolLoopState,
    session: &McpToolSession<'_>,
    extra: usize,
) -> Vec<Value> {
    let mut prefix =
        Vec::with_capacity(list_tools_items.len() + state.mcp_call_items.len() + extra);
    prefix.extend(list_tools_items);
    prefix.extend(
        state
            .mcp_call_items
            .iter()
            .filter(|item| !is_internal_mcp_response_item(item, session))
            .cloned(),
    );
    prefix
}

fn build_approval_response(
    mut response: Value,
    mut state: ToolLoopState,
    session: &McpToolSession<'_>,
    original_body: &ResponsesRequest,
    approval_item: Value,
) -> Result<Value, String> {
    let obj = response
        .as_object_mut()
        .ok_or_else(|| "response not an object".to_string())?;
    obj.insert("status".to_string(), Value::String("completed".to_string()));

    match obj.get_mut("output").and_then(|v| v.as_array_mut()) {
        Some(output_array) => {
            let retained_items = retained_output_items(output_array, original_body);
            let prefix = approval_prefix_items(&mut state, session, approval_item);

            output_array.clear();
            output_array.extend(prefix);
            output_array.extend(retained_items);
        }
        None => {
            let output_items = approval_prefix_items(&mut state, session, approval_item);
            obj.insert("output".to_string(), Value::Array(output_items));
        }
    }

    Ok(response)
}

/// The upstream's output items a response keeps when the loop ends short of
/// an answer: everything but the model's asks for server calls (the
/// `function_call` items bound to the session's tools), which the loop
/// renders as its own `mcp_call` items; a client function's call stays.
pub(crate) fn retained_output_items(
    output_array: &[Value],
    original_body: &ResponsesRequest,
) -> Vec<Value> {
    let user_function_names: HashSet<&str> = original_body
        .tools
        .as_deref()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| match tool {
                    ResponseTool::Function(function_tool) => {
                        Some(function_tool.function.name.as_str())
                    }
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    output_array
        .iter()
        .filter(|item| {
            let item_type = item.get("type").and_then(|value| value.as_str());
            if !item_type.is_some_and(is_function_call_type) {
                return true;
            }

            item.get("name")
                .and_then(|value| value.as_str())
                .is_some_and(|name| user_function_names.contains(name))
        })
        .cloned()
        .collect()
}

fn approval_prefix_items(
    state: &mut ToolLoopState,
    session: &McpToolSession<'_>,
    approval_item: Value,
) -> Vec<Value> {
    let mut prefix = mcp_output_prefix_last(state, session, 1);
    if !is_internal_mcp_response_item(&approval_item, session) {
        prefix.push(approval_item);
    }
    prefix
}

pub(crate) struct ToolLoopExecutionContext<'a> {
    pub original_body: &'a ResponsesRequest,
    pub existing_mcp_list_tools_labels: &'a [String],
    pub session: &'a McpToolSession<'a>,
    pub format_registry: &'a FormatRegistry,
}

/// Execute the tool calling loop
pub(crate) async fn execute_tool_loop(
    client: &reqwest::Client,
    url: &str,
    headers: Option<&HeaderMap>,
    worker_api_key: Option<&String>,
    initial_payload: Value,
    tool_loop_ctx: ToolLoopExecutionContext<'_>,
) -> Result<Value, String> {
    let ToolLoopExecutionContext {
        original_body,
        existing_mcp_list_tools_labels,
        session,
        format_registry,
    } = tool_loop_ctx;

    let mut state = ToolLoopState::new(existing_mcp_list_tools_labels.to_vec());
    let max_tool_calls = original_body.max_tool_calls.map(|n| n as usize);
    let mut current_payload = initial_payload;

    info!(
        "Starting tool loop: max_tool_calls={:?}, max_iterations={}",
        max_tool_calls, DEFAULT_MAX_ITERATIONS
    );
    let provider = ApiProvider::from_url(url);
    let auth_header = provider.extract_auth_header(headers, worker_api_key);

    loop {
        let (response_json, function_calls) = match upstream_round(
            client,
            url,
            &provider,
            auth_header.as_ref(),
            &current_payload,
        )
        .await?
        {
            Round::Done(mut response_json) => {
                info!(
                    "Tool loop completed: {} iterations, {} total calls",
                    state.iteration, state.total_calls
                );
                if state.total_calls > 0 {
                    inject_mcp_metadata_last(&mut response_json, &mut state, session);
                }
                return Ok(response_json);
            }
            Round::Calls(response_json, function_calls) => (response_json, function_calls),
        };

        state.iteration += 1;
        Metrics::record_mcp_tool_iteration(&original_body.model);

        info!(
            "Tool loop iteration {}: {} function call(s) detected",
            state.iteration,
            function_calls.len()
        );

        let effective_limit = match max_tool_calls {
            Some(user_max) => user_max.min(DEFAULT_MAX_ITERATIONS),
            None => DEFAULT_MAX_ITERATIONS,
        };

        for call in function_calls {
            state.total_calls += 1;

            if state.total_calls > effective_limit {
                warn!(
                    "Reached tool call limit ({}) after {} calls",
                    effective_limit, state.total_calls
                );
                return build_incomplete_response(
                    response_json,
                    state,
                    TOOL_CALL_LIMIT_REASON,
                    session,
                    format_registry,
                    original_body,
                );
            }
            if let Some(approval_item) =
                dispatch_call(call, &mut state, session, format_registry, original_body).await
            {
                return build_approval_response(
                    response_json,
                    state,
                    session,
                    original_body,
                    approval_item,
                );
            }
        }

        resume_payload_in_place(&mut current_payload, state.history_to_append(), false)?;
    }
}

/// The `incomplete_details.reason` of a tool loop stopped at the request's
/// `max_tool_calls` before the model's final answer, in both modes.
pub(crate) const TOOL_CALL_LIMIT_REASON: &str = "max_tool_calls";
/// The error of a call the loop did not run because the request's tool-call
/// allowance was spent before it, in both modes.
pub(crate) const NOT_EXECUTED_AT_TOOL_CALL_LIMIT: &str =
    "Not executed - response stopped due to limit";
/// The error of a call the streamed loop did not run because the upstream
/// turn's stream broke before the call was complete.
pub(crate) const NOT_EXECUTED_AT_STREAM_BREAK: &str =
    "Not executed - the upstream stream ended before this call was complete";

/// The item of a call the loop did not run, in the call's own format, in
/// both modes: an `mcp_call` that `failed` with `error`
/// ([`NOT_EXECUTED_AT_TOOL_CALL_LIMIT`] where the request's tool-call
/// allowance was spent before the call, [`NOT_EXECUTED_AT_STREAM_BREAK`]
/// where the turn's stream broke before the call was complete), or, for a
/// tool answered in a hosted format, the item type the stream announces for
/// the call (`web_search_call`, ...) with the status `failed`. The id is
/// derived from the call's, as an executed call's is.
fn not_executed_tool_call_item(
    response_format: ResponseFormat,
    item_id_source: &str,
    server_label: &str,
    tool_name: &str,
    arguments: &str,
    error: &str,
) -> Value {
    if response_format == ResponseFormat::Passthrough {
        let mut item =
            build_mcp_call_item(tool_name, arguments, "", server_label, false, Some(error));
        if let Some(obj) = item.as_object_mut() {
            obj.insert(
                "id".to_string(),
                Value::String(mcp_response_item_id(item_id_source)),
            );
        }
        return item;
    }
    let mut item = build_transformed_mcp_call_item(
        &json!({ "error": error }),
        response_format,
        item_id_source,
        server_label,
        tool_name,
        arguments,
    );
    if let Some(obj) = item.as_object_mut() {
        obj.insert("status".to_string(), Value::String("failed".to_string()));
    }
    item
}

/// What one upstream round of the loop answered: the final response, or the
/// response with the function calls the model asked for.
enum Round {
    Done(Value),
    Calls(Value, Vec<ExtractedFunctionCall>),
}

/// One upstream round: the payload posted under the provider's headers, an
/// error status answered as the loop's failure with its body sanitized, the
/// JSON read and its function calls extracted.
async fn upstream_round(
    client: &reqwest::Client,
    url: &str,
    provider: &ApiProvider,
    auth_header: Option<&HeaderValue>,
    payload: &Value,
) -> Result<Round, String> {
    let request_builder = client.post(url).json(payload);
    let request_builder = provider.apply_headers(request_builder, auth_header);

    let response = request_builder
        .send()
        .await
        .map_err(|e| format!("upstream request failed: {}", e.without_url()))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = error_body_text(response).await;
        return Err(format!("upstream error {status}: {body}"));
    }

    let response_json = response
        .json::<Value>()
        .await
        .map_err(|e| format!("parse response: {}", e.without_url()))?;

    let function_calls = extract_function_calls(&response_json);
    if function_calls.is_empty() {
        return Ok(Round::Done(response_json));
    }
    Ok(Round::Calls(response_json, function_calls))
}

/// One call of a round dispatched and recorded: arguments the model wrote
/// that are not JSON record an error item; otherwise the prepared arguments
/// go to the session, and the result is recorded with its metrics. A call the
/// session holds for approval returns the approval item, which ends the
/// loop with the approval response.
async fn dispatch_call(
    call: ExtractedFunctionCall,
    state: &mut ToolLoopState,
    session: &McpToolSession<'_>,
    format_registry: &FormatRegistry,
    original_body: &ResponsesRequest,
) -> Option<Value> {
    let response_format = openai_bridge::lookup_tool_format(session, format_registry, &call.name);
    let server_label = session.resolve_tool_server_label(&call.name);
    let tool_item_id = non_streaming_tool_item_id_source(&call.item_id, response_format);
    let arguments = match prepared_arguments(
        &call.arguments,
        response_format,
        original_body.tools.as_deref().unwrap_or(&[]),
        original_body.user.as_deref(),
    ) {
        Ok(v) => v,
        Err(e) => {
            warn!(tool = %call.name, error = %e, "Failed to parse tool arguments as JSON");
            let error_output = invalid_tool_arguments(&e);
            let error_json = json!({ "error": &error_output });
            let transformed_item = build_transformed_mcp_call_item(
                &error_json,
                response_format,
                &tool_item_id,
                &server_label,
                &call.name,
                &call.arguments,
            );

            record_parse_failure(&original_body.model, &call.name);

            state.record_call(
                session.is_builtin_tool(&call.name),
                call.call_id,
                call.name,
                call.arguments,
                error_json.to_string(),
                transformed_item,
            );
            return None;
        }
    };

    // Serialize the post-merge args once so downstream logging + the
    // approval payload show the effective (dispatched) payload rather
    // than the pre-merge string the model emitted.
    let effective_arguments =
        serde_json::to_string(&arguments).unwrap_or_else(|_| call.arguments.clone());

    debug!(
        "Calling MCP tool '{}' with args: {}",
        call.name, effective_arguments
    );
    let tool_result = session
        .execute_tool_result(ToolExecutionInput {
            call_id: call.call_id.clone(),
            tool_name: call.name.clone(),
            arguments,
        })
        .await;

    let approval_request_id = approval_request_item_id_source(&call.item_id);

    let tool_output = match tool_result {
        ToolExecutionResult::Executed(tool_output) => tool_output,
        ToolExecutionResult::PendingApproval(pending) => {
            return Some(build_mcp_approval_request_item(
                &approval_request_id,
                &pending.tool_name,
                &effective_arguments,
                &server_label,
            ));
        }
    };

    record_tool_metrics(&original_body.model, &tool_output);

    let output_str = tool_output.output.to_string();
    let transformed_item = build_transformed_mcp_call_item(
        &tool_output.output,
        response_format,
        &tool_item_id,
        &server_label,
        &call.name,
        // Use the post-merge string so the client-visible item describes
        // what the router actually dispatched (e.g. hosted-tool overrides
        // like image-generation `size`/`quality` merged in above), not
        // the pre-merge arguments the model emitted.
        &effective_arguments,
    );

    state.record_call(
        session.is_builtin_tool(&call.name),
        call.call_id,
        call.name,
        call.arguments,
        output_str,
        transformed_item,
    );
    None
}

/// The arguments a call dispatches with: the model's string parsed as JSON,
/// a payload that is not an object coerced to `{}` before the merge (the
/// merge is a no-op on scalars and arrays, and caller-declared hosted-tool
/// config must not be dropped silently), the caller-declared hosted-tool
/// configuration (e.g. `size`, `quality` on image_generation) merged in, and
/// the request-level `user` forwarded so a downstream MCP server can
/// attribute per-user usage. The merge and the forwarding are no-ops for
/// plain MCP function tools (Passthrough format).
fn prepared_arguments(
    raw: &str,
    response_format: ResponseFormat,
    request_tools: &[ResponseTool],
    request_user: Option<&str>,
) -> Result<Value, serde_json::Error> {
    let mut arguments: Value = serde_json::from_str(raw)?;
    if !matches!(arguments, Value::Object(_)) {
        arguments = json!({});
    }
    prepare_hosted_dispatch_args(&mut arguments, response_format, request_tools, request_user);
    Ok(arguments)
}

/// The error a call records when the model's arguments are not JSON: the
/// same text on the call's item in both loops.
fn invalid_tool_arguments(e: &serde_json::Error) -> String {
    format!("Invalid tool arguments: {e}")
}

/// A call that never ran because its arguments did not parse, counted as an
/// errored call by model and tool, in both loops.
fn record_parse_failure(model_id: &str, tool_name: &str) {
    Metrics::record_mcp_tool_call(model_id, tool_name, metrics_labels::RESULT_ERROR);
}

/// A dispatched call's duration and result, by model and tool.
fn record_tool_metrics(model_id: &str, tool_output: &smg_mcp::ToolExecutionOutput) {
    Metrics::record_mcp_tool_duration(model_id, &tool_output.tool_name, tool_output.duration);
    Metrics::record_mcp_tool_call(
        model_id,
        &tool_output.tool_name,
        if tool_output.is_error {
            metrics_labels::RESULT_ERROR
        } else {
            metrics_labels::RESULT_SUCCESS
        },
    );
}

/// Build an incomplete response when limits are exceeded
fn build_incomplete_response(
    mut response: Value,
    mut state: ToolLoopState,
    reason: &str,
    session: &McpToolSession<'_>,
    format_registry: &FormatRegistry,
    original_body: &ResponsesRequest,
) -> Result<Value, String> {
    let obj = response
        .as_object_mut()
        .ok_or_else(|| "response not an object".to_string())?;

    // Set status to completed (not failed - partial success)
    obj.insert("status".to_string(), Value::String("completed".to_string()));

    obj.insert(
        "incomplete_details".to_string(),
        json!({ "reason": reason }),
    );

    let user_function_names: HashSet<&str> = original_body
        .tools
        .as_deref()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| match tool {
                    ResponseTool::Function(function_tool) => {
                        Some(function_tool.function.name.as_str())
                    }
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    // The calls this loop ran, each already among the executed items: a turn
    // the allowance cut in two renders them once.
    let executed_call_ids: HashSet<&str> = state
        .conversation_history
        .iter()
        .filter(|item| {
            item.get("type").and_then(Value::as_str) == Some(ItemType::FUNCTION_CALL_OUTPUT)
        })
        .filter_map(|item| item.get("call_id").and_then(Value::as_str))
        .collect();

    // Convert MCP function_call items in output to mcp_call format.
    if let Some(output_array) = obj.get_mut("output").and_then(|v| v.as_array_mut()) {
        // Find any function_call items and convert them to mcp_call (incomplete)
        let mut incomplete_items = Vec::new();
        for item in output_array.iter() {
            let item_type = item.get("type").and_then(|t| t.as_str());
            if item_type.is_some_and(is_function_call_type) {
                let tool_name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                if user_function_names.contains(tool_name) {
                    // User function calls must remain as function_call output items.
                    continue;
                }
                let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                if executed_call_ids.contains(call_id) {
                    // The call ran before the allowance was spent: its item
                    // is among the executed ones.
                    continue;
                }
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");

                // Not executed: the item in the call's own format, as the
                // streamed loop closes the same call.
                let response_format =
                    openai_bridge::lookup_tool_format(session, format_registry, tool_name);
                let item_id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("call_id").and_then(|v| v.as_str()))
                    .unwrap_or("");
                let resolved_label = session.resolve_tool_server_label(tool_name);
                incomplete_items.push(not_executed_tool_call_item(
                    response_format,
                    &non_streaming_tool_item_id_source(item_id, response_format),
                    &resolved_label,
                    tool_name,
                    args,
                    NOT_EXECUTED_AT_TOOL_CALL_LIMIT,
                ));
            }
        }

        // The asks themselves leave: each is rendered by its call's item (the
        // executed one, or the not-executed one in the call's own format), as
        // the approval exit keeps only the client's own function calls.
        let retained = retained_output_items(output_array, original_body);
        output_array.clear();
        output_array.extend(retained);

        // Add mcp_list_tools and executed mcp_call items at the beginning
        if state.total_calls > 0 || !incomplete_items.is_empty() {
            let mut prefix = mcp_output_prefix_last(&mut state, session, incomplete_items.len());
            prefix.extend(
                incomplete_items
                    .into_iter()
                    .filter(|item| !is_internal_mcp_response_item(item, session)),
            );
            output_array.splice(0..0, prefix);
        }
    }

    if let Some(metadata_val) = obj.get_mut("metadata") {
        if let Some(metadata_obj) = metadata_val.as_object_mut() {
            if let Some(mcp_val) = metadata_obj.get_mut("mcp") {
                if let Some(mcp_obj) = mcp_val.as_object_mut() {
                    mcp_obj.insert(
                        "truncation_warning".to_string(),
                        Value::String(format!(
                            "Loop terminated at {} iterations, {} total calls (reason: {})",
                            state.iteration, state.total_calls, reason
                        )),
                    );
                }
            }
        }
    }

    Ok(response)
}

fn is_internal_mcp_response_item(item: &Value, session: &McpToolSession<'_>) -> bool {
    let matches_internal_server = item
        .get("server_label")
        .and_then(|value| value.as_str())
        .is_some_and(|server_label| session.is_internal_non_builtin_server_label(server_label));

    match item.get("name").and_then(|value| value.as_str()) {
        Some(name) if session.has_exposed_tool(name) => session.is_internal_non_builtin_tool(name),
        _ => matches_internal_server,
    }
}

/// Build a mcp_call output item
fn build_mcp_call_item(
    tool_name: &str,
    arguments: &str,
    output: &str,
    server_label: &str,
    success: bool,
    error: Option<&str>,
) -> Value {
    json!({
        "id": generate_id("mcp"),
        "type": ItemType::MCP_CALL,
        "status": if success { "completed" } else { "failed" },
        "approval_request_id": Value::Null,
        "arguments": arguments,
        "error": error.map(McpToolCallError::execution),
        "name": tool_name,
        "output": output,
        "server_label": server_label
    })
}

fn build_mcp_approval_request_item(
    approval_request_id: &str,
    tool_name: &str,
    arguments: &str,
    server_label: &str,
) -> Value {
    json!({
        "id": approval_request_id,
        "type": "mcp_approval_request",
        "arguments": arguments,
        "name": tool_name,
        "server_label": server_label,
    })
}

/// Build a transformed output item using ResponseTransformer
///
/// Converts the output using the tool's response_format to the correctly-typed
/// output item (mcp_call, web_search_call, code_interpreter_call, file_search_call,
/// image_generation_call).
/// Returns the result as a JSON Value for SSE event streaming.
fn build_transformed_mcp_call_item(
    output: &Value,
    response_format: ResponseFormat,
    tool_item_id: &str,
    server_label: &str,
    tool_name: &str,
    arguments: &str,
) -> Value {
    let output_item = ResponseTransformer::transform(
        output,
        response_format,
        tool_item_id,
        server_label,
        tool_name,
        arguments,
    );
    to_value(&output_item).unwrap_or_else(|e| {
        warn!(tool = %tool_name, error = %e, "Failed to serialize transformed output item");
        json!({})
    })
}

/// A function call extracted from a non-streaming response
struct ExtractedFunctionCall {
    pub call_id: String,
    pub item_id: String,
    pub name: String,
    pub arguments: String,
}

/// Extract all function calls from a response
fn extract_function_calls(resp: &Value) -> Vec<ExtractedFunctionCall> {
    let Some(output) = resp.get("output").and_then(|v| v.as_array()) else {
        return Vec::new();
    };

    let mut calls = Vec::with_capacity(4);
    for item in output {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let Some(t) = obj.get("type").and_then(|v| v.as_str()) else {
            continue;
        };
        if !is_function_call_type(t) {
            continue;
        }

        let call_id = obj.get("call_id").and_then(|v| v.as_str());
        let item_id = obj.get("id").and_then(|v| v.as_str()).or(call_id);
        let name = obj.get("name").and_then(|v| v.as_str());
        let arguments = obj.get("arguments").and_then(|v| v.as_str());

        if let (Some(call_id), Some(item_id), Some(name), Some(arguments)) =
            (call_id, item_id, name, arguments)
        {
            calls.push(ExtractedFunctionCall {
                call_id: call_id.to_string(),
                item_id: item_id.to_string(),
                name: name.to_string(),
                arguments: arguments.to_string(),
            });
        }
    }

    calls
}

#[cfg(test)]
#[path = "../../../tests/unit/openai/mcp/tool_loop/tests.rs"]
mod tests;
