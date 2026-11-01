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
use serde_json::{json, to_value, Value};
use smg_mcp::{McpServerBinding, McpToolSession, ToolExecutionInput, ToolExecutionResult};
use tracing::{debug, info, warn};

use super::tool_handler::FunctionCallInProgress;
use crate::{
    error,
    header_utils::ApiProvider,
    mcp_utils::{prepare_hosted_dispatch_args, DEFAULT_MAX_ITERATIONS},
    metrics::{metrics_labels, Metrics},
    openai::responses::utils,
    openai_bridge,
    openai_bridge::{
        extract_embedded_openai_responses, mcp_response_item_id, FormatRegistry, ResponseFormat,
        ResponseTransformer,
    },
    sse::SseSender,
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
        }
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
/// event, then the item's `response.output_item.done`) and recorded in the
/// loop's state, as the non-streamed loop records a call. Returns false when
/// the client left.
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
    state.record_call(
        dispatch.session.is_builtin_tool(&call.name),
        call.call_id,
        call.name,
        call.arguments_buffer,
        output_str,
        mcp_call_item,
    );
    true
}

/// The calls of a batch beyond the request's tool-call allowance: none runs.
/// Each is closed on the wire under the type the stream announced for it,
/// with the item the non-streamed loop renders for the same call
/// ([`not_executed_tool_call_item`]): an `mcp_call` gets
/// `response.mcp_call.failed` and then its `response.output_item.done`; a
/// call in a hosted format gets its `response.output_item.done` alone (the
/// API has no failed sub-event for those item types; the item's status says
/// it). So no item the stream announced stays open, and each takes its place
/// among the final response's items after the calls that ran. Returns false
/// when the client left.
pub(crate) async fn skip_streaming_tool_calls(
    skipped: Vec<FunctionCallInProgress>,
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

        state.mcp_call_items.push(item);
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

/// Build a resume payload: the payload the loop started from, its input (the
/// request's own, with the history it continues loaded in front) followed by
/// every call this loop executed with its output.
pub(crate) fn build_resume_payload(
    base_payload: &Value,
    conversation_history: &[Value],
    tools_json: &Value,
    is_streaming: bool,
) -> Result<Value, String> {
    let mut payload = base_payload.clone();

    let obj = payload
        .as_object_mut()
        .ok_or_else(|| "payload not an object".to_string())?;

    let mut input_array = match obj.get("input") {
        Some(Value::Array(items)) => items.clone(),
        Some(Value::String(text)) => vec![json!({
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": text }]
        })],
        _ => return Err("payload input is neither a string nor an item list".to_string()),
    };

    input_array.extend_from_slice(conversation_history);
    obj.insert("input".to_string(), Value::Array(input_array));

    if let Some(tools_arr) = tools_json.as_array() {
        if !tools_arr.is_empty() {
            obj.insert("tools".to_string(), tools_json.clone());
            // The tool call already happened; "auto" lets the model answer
            // instead of looping a preserved "required" until the call limit.
            obj.insert("tool_choice".to_string(), Value::String("auto".to_string()));
        }
    }

    obj.insert("stream".to_string(), Value::Bool(is_streaming));
    obj.insert("store".to_string(), Value::Bool(false));

    Ok(payload)
}

/// Send mcp_list_tools events to client at the start of streaming
/// Returns false if client disconnected
pub(crate) async fn send_mcp_list_tools_events(
    tx: &SseSender,
    session: &McpToolSession<'_>,
    server_label: &str,
    output_index: usize,
    sequence_number: &mut u64,
    server_key: &str,
) -> bool {
    let tools_item_full = openai_bridge::mcp_list_tools_json(session, server_label, server_key);
    let item_id = tools_item_full
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Create empty tools version for the initial added event
    let mut tools_item_empty = tools_item_full.clone();
    if let Some(obj) = tools_item_empty.as_object_mut() {
        obj.insert("tools".to_string(), json!([]));
    }

    // Event 1: response.output_item.added with empty tools
    let event1_payload = json!({
        "type": OutputItemEvent::ADDED,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item": tools_item_empty
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

    // Event 3: response.mcp_list_tools.completed
    let event3_payload = json!({
        "type": McpEvent::LIST_TOOLS_COMPLETED,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item_id": item_id
    });
    *sequence_number += 1;
    let event3 = format!(
        "event: {}\ndata: {}\n\n",
        McpEvent::LIST_TOOLS_COMPLETED,
        event3_payload
    );
    if tx.send(Ok(Bytes::from(event3))).await.is_err() {
        return false;
    }

    // Event 4: response.output_item.done with full tools list
    let event4_payload = json!({
        "type": OutputItemEvent::DONE,
        "sequence_number": *sequence_number,
        "output_index": output_index,
        "item": tools_item_full
    });
    *sequence_number += 1;
    let event4 = format!(
        "event: {}\ndata: {}\n\n",
        OutputItemEvent::DONE,
        event4_payload
    );
    tx.send(Ok(Bytes::from(event4))).await.is_ok()
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
/// and image_generation_call items.
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

    // Resolve the completion event from the item's `type` (the typed
    // `tool_call_item` may have been re-tagged after dispatch); fall back
    // to the format-derived event for unknown types.
    let item_type = tool_call_item
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let completed_event_type: &str = openai_bridge::format_from_type_str(item_type)
        .map(|f| openai_bridge::descriptor(f).completed_event)
        .unwrap_or_else(|| openai_bridge::descriptor(response_format).completed_event);

    // Event 1: response.<type>.completed
    let completed_payload = json!({
        "type": completed_event_type,
        "sequence_number": *sequence_number,
        "output_index": effective_output_index,
        "item_id": item_id
    });
    *sequence_number += 1;

    let completed_event = format!("event: {completed_event_type}\ndata: {completed_payload}\n\n");
    if tx.send(Ok(Bytes::from(completed_event))).await.is_err() {
        return false;
    }

    // Event 2: response.output_item.done (with completed tool call)
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

/// Inject MCP metadata into a streaming response
pub(crate) fn inject_mcp_metadata_streaming(
    response: &mut Value,
    state: &ToolLoopState,
    session: &McpToolSession<'_>,
) {
    let list_tools_bindings = mcp_list_tools_bindings_to_emit(
        &state.existing_mcp_list_tools_labels,
        session.mcp_servers(),
    );

    if let Some(output_array) = response.get_mut("output").and_then(|v| v.as_array_mut()) {
        output_array.retain(|item| {
            item.get("type").and_then(|t| t.as_str()) != Some(ItemType::MCP_LIST_TOOLS)
        });

        let mut prefix = Vec::with_capacity(list_tools_bindings.len() + state.mcp_call_items.len());
        for (server_label, server_key) in &list_tools_bindings {
            if !session.is_internal_server_label(server_label) {
                prefix.push(openai_bridge::mcp_list_tools_json(
                    session,
                    server_label,
                    server_key,
                ));
            }
        }
        prefix.extend(
            state
                .mcp_call_items
                .iter()
                .filter(|item| !is_internal_mcp_response_item(item, session))
                .cloned(),
        );
        output_array.splice(0..0, prefix);
    } else if let Some(obj) = response.as_object_mut() {
        let mut output_items = Vec::new();
        for (server_label, server_key) in &list_tools_bindings {
            if !session.is_internal_server_label(server_label) {
                output_items.push(openai_bridge::mcp_list_tools_json(
                    session,
                    server_label,
                    server_key,
                ));
            }
        }
        // Use stored transformed items (no reconstruction needed)
        output_items.extend(
            state
                .mcp_call_items
                .iter()
                .filter(|item| !is_internal_mcp_response_item(item, session))
                .cloned(),
        );
        obj.insert("output".to_string(), Value::Array(output_items));
    }
}

fn build_approval_response(
    mut response: Value,
    state: ToolLoopState,
    session: &McpToolSession<'_>,
    original_body: &ResponsesRequest,
    approval_item: Value,
) -> Result<Value, String> {
    let obj = response
        .as_object_mut()
        .ok_or_else(|| "response not an object".to_string())?;
    obj.insert("status".to_string(), Value::String("completed".to_string()));

    let list_tools_bindings = mcp_list_tools_bindings_to_emit(
        &state.existing_mcp_list_tools_labels,
        session.mcp_servers(),
    );

    match obj.get_mut("output").and_then(|v| v.as_array_mut()) {
        Some(output_array) => {
            let retained_items = retained_output_items(output_array, original_body);
            let prefix =
                approval_prefix_items(&state, session, &list_tools_bindings, approval_item);

            output_array.clear();
            output_array.extend(prefix);
            output_array.extend(retained_items);
        }
        None => {
            let output_items =
                approval_prefix_items(&state, session, &list_tools_bindings, approval_item);
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
    state: &ToolLoopState,
    session: &McpToolSession<'_>,
    list_tools_bindings: &[(String, String)],
    approval_item: Value,
) -> Vec<Value> {
    let mut prefix = Vec::with_capacity(list_tools_bindings.len() + state.mcp_call_items.len() + 1);
    for (list_server_label, server_key) in list_tools_bindings {
        if !session.is_internal_server_label(list_server_label) {
            prefix.push(openai_bridge::mcp_list_tools_json(
                session,
                list_server_label,
                server_key,
            ));
        }
    }
    prefix.extend(
        state
            .mcp_call_items
            .iter()
            .filter(|item| !is_internal_mcp_response_item(item, session))
            .cloned(),
    );
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
    let base_payload = initial_payload.clone();
    let tools_json = base_payload.get("tools").cloned().unwrap_or(json!([]));
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
                    inject_mcp_metadata_streaming(&mut response_json, &state, session);
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

        current_payload = build_resume_payload(
            &base_payload,
            &state.conversation_history,
            &tools_json,
            false,
        )?;
    }
}

/// The `incomplete_details.reason` of a tool loop stopped at the request's
/// `max_tool_calls` before the model's final answer, in both modes.
pub(crate) const TOOL_CALL_LIMIT_REASON: &str = "max_tool_calls";
/// The error of a call the loop did not run because the request's tool-call
/// allowance was spent before it, in both modes.
pub(crate) const NOT_EXECUTED_AT_TOOL_CALL_LIMIT: &str =
    "Not executed - response stopped due to limit";

/// The item of a call the loop did not run because the request's tool-call
/// allowance was spent before it, in the call's own format, in both modes:
/// an `mcp_call` that `failed` with [`NOT_EXECUTED_AT_TOOL_CALL_LIMIT`], or,
/// for a tool answered in a hosted format, the item type the stream announces
/// for the call (`web_search_call`, ...) with the status `failed`. The id is
/// derived from the call's, as an executed call's is.
fn not_executed_tool_call_item(
    response_format: ResponseFormat,
    item_id_source: &str,
    server_label: &str,
    tool_name: &str,
    arguments: &str,
) -> Value {
    if response_format == ResponseFormat::Passthrough {
        let mut item = build_mcp_call_item(
            tool_name,
            arguments,
            "",
            server_label,
            false,
            Some(NOT_EXECUTED_AT_TOOL_CALL_LIMIT),
        );
        if let Some(obj) = item.as_object_mut() {
            obj.insert(
                "id".to_string(),
                Value::String(mcp_response_item_id(item_id_source)),
            );
        }
        return item;
    }
    let mut item = build_transformed_mcp_call_item(
        &json!({ "error": NOT_EXECUTED_AT_TOOL_CALL_LIMIT }),
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
        .map_err(|e| format!("upstream request failed: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let body = error::sanitize_error_body(&body);
        return Err(format!("upstream error {status}: {body}"));
    }

    let response_json = response
        .json::<Value>()
        .await
        .map_err(|e| format!("parse response: {e}"))?;

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
    state: ToolLoopState,
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

    let list_tools_bindings = mcp_list_tools_bindings_to_emit(
        &state.existing_mcp_list_tools_labels,
        session.mcp_servers(),
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
            let mut prefix = Vec::with_capacity(
                list_tools_bindings.len() + state.mcp_call_items.len() + incomplete_items.len(),
            );
            for (server_label, server_key) in &list_tools_bindings {
                if !session.is_internal_server_label(server_label) {
                    prefix.push(openai_bridge::mcp_list_tools_json(
                        session,
                        server_label,
                        server_key,
                    ));
                }
            }
            prefix.extend(
                state
                    .mcp_call_items
                    .iter()
                    .filter(|item| !is_internal_mcp_response_item(item, session))
                    .cloned(),
            );
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
mod tests {
    use std::collections::HashSet;

    use metrics_exporter_prometheus::PrometheusBuilder;
    use openai_protocol::responses::ResponsesRequest;
    use serde_json::json;
    use smg_mcp::{
        BuiltinToolType, McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig,
        McpToolSession, McpTransport, Tool, ToolEntry,
    };
    use tokio::sync::mpsc;

    use super::{
        build_transformed_mcp_call_item, extract_openai_response_output_items,
        is_internal_mcp_response_item, mcp_list_tools_bindings_to_emit,
        not_executed_tool_call_item, ToolLoopState, NOT_EXECUTED_AT_TOOL_CALL_LIMIT,
    };
    use crate::{
        openai::mcp::test_support,
        openai_bridge::{FormatRegistry, ResponseFormat},
    };

    fn test_tool(name: &str) -> Tool {
        let mut schema = serde_json::Map::new();
        schema.insert("type".to_string(), json!("object"));
        schema.insert("properties".to_string(), json!({}));

        Tool::new(name.to_string(), "internal", schema)
    }

    /// A call left over the tool-call allowance is rendered in its own
    /// format: an `mcp_call` that failed with the not-executed error and the
    /// id an executed call would have had, or the hosted item type the stream
    /// announces for the tool, `failed`, with that format's id.
    #[test]
    fn a_not_executed_call_item_keeps_the_format_the_stream_announced() {
        let plain = not_executed_tool_call_item(
            ResponseFormat::Passthrough,
            "fc_1",
            "mock",
            "brave_web_search",
            "{}",
        );
        assert_eq!(plain["type"], "mcp_call");
        assert_eq!(plain["status"], "failed");
        assert_eq!(plain["id"], "mcp_1");
        assert_eq!(plain["server_label"], "mock");
        assert!(
            plain["error"]
                .to_string()
                .contains(NOT_EXECUTED_AT_TOOL_CALL_LIMIT),
            "{plain}"
        );

        let hosted = not_executed_tool_call_item(
            ResponseFormat::WebSearchCall,
            "1",
            "mock",
            "brave_web_search",
            r#"{"query":"q"}"#,
        );
        assert_eq!(hosted["type"], "web_search_call");
        assert_eq!(hosted["status"], "failed");
        assert_eq!(hosted["id"], "ws_1");
        assert!(hosted.get("server_label").is_none());
    }

    #[test]
    fn build_transformed_mcp_call_item_does_not_add_server_label_for_builtin_formats() {
        let item = build_transformed_mcp_call_item(
            &json!({
                "queries": ["private query"],
                "results": [
                    { "url": "https://example.com" }
                ]
            }),
            ResponseFormat::WebSearchCall,
            "call_123",
            "internal-label",
            "brave_web_search",
            r#"{"query":"private query"}"#,
        );

        assert_eq!(
            item.get("type").and_then(|value| value.as_str()),
            Some("web_search_call")
        );
        assert!(item.get("server_label").is_none());
    }

    #[tokio::test]
    async fn internal_filter_keeps_builtin_passthrough_mcp_call_items() {
        let orchestrator = test_support::internal_orchestrator_serving(
            Some((BuiltinToolType::WebSearchPreview, "brave_web_search")),
            vec![test_tool("brave_web_search")],
        )
        .await;
        let session = test_support::internal_session(&orchestrator);

        let item = json!({
            "type": "mcp_call",
            "name": "brave_web_search",
            "server_label": "internal-label"
        });

        assert!(!is_internal_mcp_response_item(&item, &session));
    }

    #[tokio::test]
    async fn internal_filter_keeps_builtin_passthrough_mcp_call_items_with_mixed_tools() {
        let orchestrator = test_support::internal_orchestrator_serving(
            Some((BuiltinToolType::WebSearchPreview, "brave_web_search")),
            vec![
                test_tool("brave_web_search"),
                test_tool("internal_non_builtin_tool"),
            ],
        )
        .await;
        let session = test_support::internal_session(&orchestrator);

        let builtin_item = json!({
            "type": "mcp_call",
            "name": "brave_web_search",
            "server_label": "internal-label"
        });
        let internal_non_builtin_item = json!({
            "type": "mcp_call",
            "name": "internal_non_builtin_tool",
            "server_label": "internal-label"
        });

        assert!(!is_internal_mcp_response_item(&builtin_item, &session));
        assert!(is_internal_mcp_response_item(
            &internal_non_builtin_item,
            &session
        ));
    }

    #[test]
    fn emits_only_new_binding_when_resume_adds_second_tool_block() {
        let existing_labels = HashSet::from(["deepwiki_ask".to_string()]);
        let bindings = vec![
            McpServerBinding {
                label: "deepwiki_ask".to_string(),
                server_key: "server-ask".to_string(),
                allowed_tools: Some(vec!["ask_question".to_string()]),
            },
            McpServerBinding {
                label: "deepwiki_read".to_string(),
                server_key: "server-read".to_string(),
                allowed_tools: Some(vec!["read_wiki_structure".to_string()]),
            },
        ];

        let bindings_to_emit = mcp_list_tools_bindings_to_emit(&existing_labels, &bindings);

        assert_eq!(
            bindings_to_emit,
            vec![("deepwiki_read".to_string(), "server-read".to_string())]
        );
    }

    #[test]
    fn emits_all_bindings_when_no_prior_mcp_list_tools_exist() {
        let existing_labels = HashSet::new();
        let bindings = vec![McpServerBinding {
            label: "deepwiki_ask".to_string(),
            server_key: "server-ask".to_string(),
            allowed_tools: Some(vec!["ask_question".to_string()]),
        }];

        let bindings_to_emit = mcp_list_tools_bindings_to_emit(&existing_labels, &bindings);

        assert_eq!(
            bindings_to_emit,
            vec![("deepwiki_ask".to_string(), "server-ask".to_string())]
        );
    }

    #[test]
    fn extract_openai_response_output_items_from_embedded_text_json() {
        let output = r#"[{"type":"text","text":"{\"execution_id\":\"abc\",\"openai_response\":{\"content\":{\"type\":\"output_text\",\"annotations\":[{\"type\":\"url_citation\",\"title\":\"Example citation\",\"url\":\"https://example.com/openai-result\",\"start_index\":0,\"end_index\":10}],\"logprobs\":[],\"text\":\"intermediate summary\"}}}"}]"#;

        let extracted = extract_openai_response_output_items(output);
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0]["type"], "message");
        assert_eq!(extracted[0]["role"], "assistant");
        assert_eq!(extracted[0]["content"][0]["type"], "output_text");
        assert_eq!(extracted[0]["content"][0]["text"], "intermediate summary");
        assert_eq!(
            extracted[0]["content"][0]["annotations"][0]["type"],
            "url_citation"
        );
        assert_eq!(
            extracted[0]["content"][0]["annotations"][0]["title"],
            "Example citation"
        );
        assert_eq!(
            extracted[0]["content"][0]["annotations"][0]["url"],
            "https://example.com/openai-result"
        );
        assert_eq!(
            extracted[0]["content"][0]["annotations"][0]["start_index"],
            0
        );
        assert_eq!(
            extracted[0]["content"][0]["annotations"][0]["end_index"],
            10
        );
    }

    #[test]
    fn record_call_appends_openai_response_output_after_tool_item() {
        let mut state = ToolLoopState::new(Vec::new());
        let transformed = json!({
            "type": "web_search_call",
            "id": "ws_test",
            "status": "completed",
            "action": {"type": "search"}
        });
        let output = r#"[{"type":"text","text":"{\"openai_response\":{\"content\":{\"type\":\"output_text\",\"annotations\":[{\"type\":\"url_citation\",\"title\":\"Example citation\",\"url\":\"https://example.com/openai-result\",\"start_index\":0,\"end_index\":10}],\"logprobs\":[],\"text\":\"intermediate\"}}}"}]"#;

        state.record_call(
            true,
            "call_123".to_string(),
            "search_web".to_string(),
            "{\"query\":\"x\"}".to_string(),
            output.to_string(),
            transformed,
        );

        assert_eq!(state.mcp_call_items.len(), 2);
        assert_eq!(state.mcp_call_items[0]["type"], "web_search_call");
        assert_eq!(state.mcp_call_items[1]["type"], "message");
        assert_eq!(
            state.mcp_call_items[1]["content"][0]["text"],
            "intermediate"
        );
        assert_eq!(
            state.mcp_call_items[1]["content"][0]["annotations"][0]["type"],
            "url_citation"
        );
        assert_eq!(
            state.mcp_call_items[1]["content"][0]["annotations"][0]["title"],
            "Example citation"
        );
        assert_eq!(
            state.mcp_call_items[1]["content"][0]["annotations"][0]["url"],
            "https://example.com/openai-result"
        );
        assert_eq!(
            state.mcp_call_items[1]["content"][0]["annotations"][0]["start_index"],
            0
        );
        assert_eq!(
            state.mcp_call_items[1]["content"][0]["annotations"][0]["end_index"],
            10
        );
    }

    #[test]
    fn record_call_does_not_append_openai_response_output_for_non_builtin_tools() {
        let mut state = ToolLoopState::new(Vec::new());
        let transformed = json!({
            "type": "web_search_call",
            "id": "ws_test",
            "status": "completed",
            "action": {"type": "search"}
        });
        let output = r#"[{"type":"text","text":"{\"openai_response\":{\"content\":{\"type\":\"output_text\",\"annotations\":[],\"logprobs\":[],\"text\":\"intermediate\"}}}"}]"#;

        state.record_call(
            false,
            "call_123".to_string(),
            "internal_search_web".to_string(),
            "{\"query\":\"x\"}".to_string(),
            output.to_string(),
            transformed,
        );

        assert_eq!(state.mcp_call_items.len(), 1);
        assert_eq!(state.mcp_call_items[0]["type"], "web_search_call");
    }

    #[test]
    fn extract_openai_response_output_items_ignores_null_openai_response() {
        let output = r#"[{"type":"text","text":"{\"openai_response\":null}"}]"#;
        let extracted = extract_openai_response_output_items(output);
        assert!(extracted.is_empty());
    }

    // ========================================================================
    // Streaming emission ordering invariant.
    //
    // `send_tool_call_completion_events` MUST emit
    // `response.<tool>.completed` BEFORE `response.output_item.done` so the
    // umbrella event terminates the item's sub-events per spec (see
    // `.claude/_audit/openai-responses-api-spec.md` §streaming, events
    // L1054-L1072).
    // ========================================================================

    fn drain_channel(rx: &mut mpsc::Receiver<Result<bytes::Bytes, std::io::Error>>) -> Vec<String> {
        let mut events = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            let bytes = chunk.expect("no io errors in unit-test channel");
            events.push(String::from_utf8(bytes.to_vec()).expect("utf-8 sse block"));
        }
        events
    }

    fn event_type_from_sse_block(block: &str) -> String {
        // SSE block shape: `event: <name>\ndata: {...}\n\n`. We parse only
        // the leading `event: …` line to get the wire type without pulling
        // serde_json into this tiny assertion helper.
        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("event: ") {
                return rest.trim().to_string();
            }
        }
        block.to_string()
    }

    #[tokio::test]
    async fn image_generation_completion_events_fire_before_output_item_done() {
        // Gap B lock test: after the tool executes, the completion
        // emitter must push `response.image_generation_call.completed`
        // onto the wire BEFORE `response.output_item.done`. If a future
        // refactor reverses the order this asserts and fails loudly.
        let call = super::FunctionCallInProgress {
            call_id: "call_img".to_string(),
            name: "image_generation".to_string(),
            arguments_buffer: "{}".to_string(),
            item_id: Some("fc_img".to_string()),
            output_index: 0,
            last_obfuscation: None,
            assigned_output_index: Some(0),
        };

        let tool_call_item = json!({
            "type": "image_generation_call",
            "id": "ig_img",
            "status": "completed",
            "result": "BASE64",
        });

        let (tx, mut rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
        let mut sequence_number: u64 = 0;

        let ok = super::send_tool_call_completion_events(
            &tx,
            &call,
            &tool_call_item,
            ResponseFormat::ImageGenerationCall,
            &mut sequence_number,
        )
        .await;
        assert!(ok, "send_tool_call_completion_events should not disconnect");
        drop(tx);

        let events = drain_channel(&mut rx);
        let types: Vec<String> = events
            .iter()
            .map(|b| event_type_from_sse_block(b))
            .collect();

        let completed_idx = types
            .iter()
            .position(|t| t == "response.image_generation_call.completed")
            .expect("response.image_generation_call.completed must be emitted");
        let done_idx = types
            .iter()
            .position(|t| t == "response.output_item.done")
            .expect("response.output_item.done must be emitted");

        assert!(
            completed_idx < done_idx,
            "`response.image_generation_call.completed` (index {completed_idx}) must come \
             before `response.output_item.done` (index {done_idx}); full sequence: {types:?}"
        );
    }

    #[tokio::test]
    async fn web_search_completion_events_fire_before_output_item_done() {
        // Same ordering contract for the pre-existing web_search_call path,
        // so the invariant applies uniformly across every hosted-tool
        // ResponseFormat we emit.
        let call = super::FunctionCallInProgress {
            call_id: "call_ws".to_string(),
            name: "web_search".to_string(),
            arguments_buffer: "{}".to_string(),
            item_id: Some("fc_ws".to_string()),
            output_index: 0,
            last_obfuscation: None,
            assigned_output_index: Some(0),
        };

        let tool_call_item = json!({
            "type": "web_search_call",
            "id": "ws_ws",
            "status": "completed",
            "action": {"type": "search"}
        });

        let (tx, mut rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
        let mut sequence_number: u64 = 0;

        let ok = super::send_tool_call_completion_events(
            &tx,
            &call,
            &tool_call_item,
            ResponseFormat::WebSearchCall,
            &mut sequence_number,
        )
        .await;
        assert!(ok);
        drop(tx);

        let events = drain_channel(&mut rx);
        let types: Vec<String> = events
            .iter()
            .map(|b| event_type_from_sse_block(b))
            .collect();

        let completed_idx = types
            .iter()
            .position(|t| t == "response.web_search_call.completed")
            .expect("web_search_call.completed must be emitted");
        let done_idx = types
            .iter()
            .position(|t| t == "response.output_item.done")
            .expect("output_item.done must be emitted");
        assert!(completed_idx < done_idx);
    }

    #[tokio::test]
    async fn code_interpreter_completion_events_fire_before_output_item_done() {
        // The suppression gate in `tool_handler.rs` drops the upstream
        // umbrella `output_item.done` for every tool-call item type. This
        // test locks the downstream half of the contract for
        // `code_interpreter_call` — the tool loop's completion emitter
        // must push `response.code_interpreter_call.completed` BEFORE
        // `response.output_item.done` so the gate's suppression lines up
        // with a correctly-ordered wire sequence.
        let call = super::FunctionCallInProgress {
            call_id: "call_ci".to_string(),
            name: "code_interpreter".to_string(),
            arguments_buffer: "{}".to_string(),
            item_id: Some("fc_ci".to_string()),
            output_index: 0,
            last_obfuscation: None,
            assigned_output_index: Some(0),
        };

        let tool_call_item = json!({
            "type": "code_interpreter_call",
            "id": "ci_ci",
            "status": "completed",
            "code": "print('hi')",
        });

        let (tx, mut rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
        let mut sequence_number: u64 = 0;

        let ok = super::send_tool_call_completion_events(
            &tx,
            &call,
            &tool_call_item,
            ResponseFormat::CodeInterpreterCall,
            &mut sequence_number,
        )
        .await;
        assert!(ok);
        drop(tx);

        let events = drain_channel(&mut rx);
        let types: Vec<String> = events
            .iter()
            .map(|b| event_type_from_sse_block(b))
            .collect();

        let completed_idx = types
            .iter()
            .position(|t| t == "response.code_interpreter_call.completed")
            .expect("code_interpreter_call.completed must be emitted");
        let done_idx = types
            .iter()
            .position(|t| t == "response.output_item.done")
            .expect("output_item.done must be emitted");
        assert!(
            completed_idx < done_idx,
            "`response.code_interpreter_call.completed` (index {completed_idx}) must come \
             before `response.output_item.done` (index {done_idx}); full sequence: {types:?}"
        );

        // No duplicate `output_item.done` — the tool loop emits exactly
        // one umbrella (the upstream copy is dropped by the suppression
        // gate in `tool_handler.rs`).
        let done_count = types
            .iter()
            .filter(|t| *t == "response.output_item.done")
            .count();
        assert_eq!(
            done_count, 1,
            "exactly one `output_item.done` expected, got {done_count}: {types:?}"
        );
    }

    #[tokio::test]
    async fn file_search_completion_events_fire_before_output_item_done() {
        // Lock the ordering contract for the hosted `file_search_call`
        // format so the gate's suppression covers every tool-call item
        // type listed in `TOOL_CALL_ITEM_TYPES`.
        let call = super::FunctionCallInProgress {
            call_id: "call_fs".to_string(),
            name: "file_search".to_string(),
            arguments_buffer: "{}".to_string(),
            item_id: Some("fc_fs".to_string()),
            output_index: 0,
            last_obfuscation: None,
            assigned_output_index: Some(0),
        };

        let tool_call_item = json!({
            "type": "file_search_call",
            "id": "fs_fs",
            "status": "completed",
            "queries": ["needle"],
        });

        let (tx, mut rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(4);
        let mut sequence_number: u64 = 0;

        let ok = super::send_tool_call_completion_events(
            &tx,
            &call,
            &tool_call_item,
            ResponseFormat::FileSearchCall,
            &mut sequence_number,
        )
        .await;
        assert!(ok);
        drop(tx);

        let events = drain_channel(&mut rx);
        let types: Vec<String> = events
            .iter()
            .map(|b| event_type_from_sse_block(b))
            .collect();

        let completed_idx = types
            .iter()
            .position(|t| t == "response.file_search_call.completed")
            .expect("file_search_call.completed must be emitted");
        let done_idx = types
            .iter()
            .position(|t| t == "response.output_item.done")
            .expect("output_item.done must be emitted");
        assert!(
            completed_idx < done_idx,
            "`response.file_search_call.completed` (index {completed_idx}) must come \
             before `response.output_item.done` (index {done_idx}); full sequence: {types:?}"
        );

        // No duplicate `output_item.done` — the tool loop emits exactly
        // one umbrella (the upstream copy is dropped by the suppression
        // gate in `tool_handler.rs`).
        let done_count = types
            .iter()
            .filter(|t| *t == "response.output_item.done")
            .count();
        assert_eq!(
            done_count, 1,
            "exactly one `output_item.done` expected, got {done_count}: {types:?}"
        );
    }

    #[test]
    fn approval_request_item_id_source_uses_single_underscore() {
        // Regression: `normalize_tool_item_id_with_prefix` appends `_`
        // internally, so the prefix argument must be the bare token. A
        // prior version passed `"mcpr_"` here and emitted `mcpr__...` ids,
        // breaking the approval wire format.
        assert_eq!(super::approval_request_item_id_source("fc_abc"), "mcpr_abc");
        assert_eq!(
            super::approval_request_item_id_source("call_xyz"),
            "mcpr_xyz"
        );
        assert_eq!(
            super::approval_request_item_id_source("mcpr_already"),
            "mcpr_already"
        );
        assert_eq!(
            super::approval_request_item_id_source("raw_id"),
            "mcpr_raw_id"
        );
    }

    async fn function_tool_session() -> (McpOrchestrator, Vec<McpServerBinding>) {
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "wiki-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: None,
                builtin_tool_name: None,
                internal: false,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "wiki-server",
                test_tool("ask_wiki"),
            ));
        let bindings = vec![McpServerBinding {
            label: "wiki".to_string(),
            server_key: "wiki-server".to_string(),
            allowed_tools: None,
        }];
        (orchestrator, bindings)
    }

    #[tokio::test]
    async fn prepare_preserves_explicit_tool_choice() {
        let (orchestrator, bindings) = function_tool_session().await;
        let session = McpToolSession::new(&orchestrator, bindings, "test-request");

        let mut payload = json!({
            "model": "m",
            "input": "q",
            "tools": [{"type": "mcp", "server_label": "wiki"}],
            "tool_choice": "required"
        });
        super::prepare_mcp_tools_as_functions(&mut payload, &session);

        assert_eq!(payload["tool_choice"], json!("required"));
        assert_eq!(payload["tools"][0]["type"], json!("function"));
    }

    #[tokio::test]
    async fn prepare_remaps_hosted_tool_choice_to_function() {
        let (orchestrator, bindings) = function_tool_session().await;
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "wiki-server",
                test_tool("image_generation"),
            ));
        let session = McpToolSession::new(&orchestrator, bindings, "test-request");

        let mut payload = json!({
            "model": "m",
            "input": "q",
            "tools": [{"type": "image_generation"}],
            "tool_choice": {"type": "image_generation"}
        });
        super::prepare_mcp_tools_as_functions(&mut payload, &session);

        assert_eq!(
            payload["tool_choice"],
            json!({"type": "function", "name": "image_generation"})
        );
    }

    #[tokio::test]
    async fn prepare_downgrades_unmappable_tool_choice_to_auto() {
        let (orchestrator, bindings) = function_tool_session().await;
        let session = McpToolSession::new(&orchestrator, bindings, "test-request");

        let mut payload = json!({
            "model": "m",
            "input": "q",
            "tools": [{"type": "mcp", "server_label": "wiki"}],
            "tool_choice": {"type": "web_search_preview"}
        });
        super::prepare_mcp_tools_as_functions(&mut payload, &session);

        assert_eq!(payload["tool_choice"], json!("auto"));
    }

    #[tokio::test]
    async fn prepare_defaults_missing_tool_choice_to_auto() {
        let (orchestrator, bindings) = function_tool_session().await;
        let session = McpToolSession::new(&orchestrator, bindings, "test-request");

        let mut payload = json!({
            "model": "m",
            "input": "q",
            "tools": [{"type": "mcp", "server_label": "wiki"}]
        });
        super::prepare_mcp_tools_as_functions(&mut payload, &session);

        assert_eq!(payload["tool_choice"], json!("auto"));
    }

    #[test]
    fn resume_payload_forces_auto_tool_choice() {
        let base = json!({
            "model": "m",
            "input": "q",
            "tools": [{"type": "function", "name": "ask_wiki"}],
            "tool_choice": "required"
        });
        let resumed = super::build_resume_payload(
            &base,
            &[],
            &json!([{"type": "function", "name": "ask_wiki"}]),
            false,
        )
        .expect("resume payload");

        assert_eq!(resumed["tool_choice"], json!("auto"));
        assert_eq!(
            resumed["input"],
            json!([{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]}])
        );
    }

    /// The payload the loop started from carries the history the request
    /// continues (a stored turn's function call and the client's answer to
    /// it); the resume payload keeps it in front of the calls this loop ran.
    #[test]
    fn resume_payload_keeps_the_loaded_history_in_front_of_the_loops_calls() {
        let loaded_input = json!([
            {"type": "message", "id": "msg_1", "role": "user", "content": [{"type": "input_text", "text": "What is 6*7?"}]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_A", "name": "calc", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_A", "output": "42"}
        ]);
        let base = json!({
            "model": "m",
            "input": loaded_input,
            "tools": [{"type": "function", "name": "search"}]
        });
        let executed = [
            json!({"type": "function_call", "call_id": "call_B", "name": "search", "arguments": "{}"}),
            json!({"type": "function_call_output", "call_id": "call_B", "output": "found"}),
        ];

        let resumed = super::build_resume_payload(
            &base,
            &executed,
            &json!([{"type": "function", "name": "search"}]),
            false,
        )
        .expect("resume payload");

        let kinds: Vec<(&str, &str)> = resumed["input"]
            .as_array()
            .expect("an item list")
            .iter()
            .map(|item| {
                (
                    item["type"].as_str().unwrap_or("?"),
                    item["call_id"].as_str().unwrap_or(""),
                )
            })
            .collect();
        assert_eq!(
            kinds,
            [
                ("message", ""),
                ("function_call", "call_A"),
                ("function_call_output", "call_A"),
                ("function_call", "call_B"),
                ("function_call_output", "call_B"),
            ]
        );
        assert_eq!(resumed["store"], json!(false));
    }

    /// The argument preparation both loops share: a payload that is not an
    /// object becomes `{}` before the hosted-tool merge, the request's `user`
    /// is forwarded for a hosted format and left out for a plain MCP tool,
    /// and an object payload under a plain MCP tool goes through untouched.
    #[test]
    fn prepared_arguments_coerce_merge_and_forward_the_user() {
        let hosted =
            super::prepared_arguments("42", ResponseFormat::WebSearchCall, &[], Some("u-1"))
                .expect("a scalar payload parses");
        assert_eq!(hosted, json!({ "user": "u-1" }));
        let plain = super::prepared_arguments("42", ResponseFormat::Passthrough, &[], Some("u-1"))
            .expect("a scalar payload parses");
        assert_eq!(plain, json!({}));
        let object = super::prepared_arguments(
            r#"{"query": "weather", "user": null}"#,
            ResponseFormat::Passthrough,
            &[],
            Some("u-1"),
        )
        .expect("an object payload parses");
        assert_eq!(object, json!({ "query": "weather", "user": null }));
        let overwritten = super::prepared_arguments(
            r#"{"query": "weather", "user": null}"#,
            ResponseFormat::WebSearchCall,
            &[],
            Some("u-1"),
        )
        .expect("an object payload parses");
        assert_eq!(overwritten, json!({ "query": "weather", "user": "u-1" }));
        assert!(
            super::prepared_arguments("not json", ResponseFormat::Passthrough, &[], None).is_err()
        );
    }

    /// The `output` of the one `function_call_output` item a loop recorded
    /// for the model's next round.
    fn function_call_output(state: &ToolLoopState) -> String {
        let item = state
            .conversation_history
            .iter()
            .find(|item| item["type"] == "function_call_output")
            .expect("a function_call_output item");
        item["output"]
            .as_str()
            .expect("the output is a string")
            .to_string()
    }

    /// A call whose arguments are not JSON is counted as an errored call in
    /// both loops (the streaming one counted nothing before), and both loops
    /// word the error on the call's item the same way.
    #[tokio::test]
    async fn a_call_with_non_json_arguments_is_counted_and_worded_alike_in_both_loops() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let (orchestrator, bindings) = function_tool_session().await;
        let session = McpToolSession::new(&orchestrator, bindings, "test-request");
        let registry = FormatRegistry::new();

        let (tx, mut rx) = mpsc::channel(16);
        let mut state = ToolLoopState::new(Vec::new());
        let mut sequence_number = 0u64;
        let mut call = super::FunctionCallInProgress::new("call_1".to_string(), 0);
        call.name = "wiki_search".to_string();
        call.arguments_buffer = "not json".to_string();
        let client_stayed = super::execute_streaming_tool_calls(
            vec![call],
            &session,
            &registry,
            &tx,
            &mut state,
            &mut sequence_number,
            "m",
            &[],
            None,
        )
        .await;
        assert!(client_stayed);
        let streamed = state.mcp_call_items[0].to_string();
        assert!(
            streamed.contains("Invalid tool arguments: "),
            "the streaming loop's item: {streamed}"
        );
        let streamed_output = function_call_output(&state);
        drop(tx);
        while rx.recv().await.is_some() {}

        let mut state = ToolLoopState::new(Vec::new());
        let body: ResponsesRequest =
            serde_json::from_value(json!({ "model": "m", "input": "q" })).expect("a request");
        let approval = super::dispatch_call(
            super::ExtractedFunctionCall {
                call_id: "call_2".to_string(),
                item_id: "fc_2".to_string(),
                name: "wiki_search".to_string(),
                arguments: "not json".to_string(),
            },
            &mut state,
            &session,
            &registry,
            &body,
        )
        .await;
        assert!(approval.is_none());
        let direct = state.mcp_call_items[0].to_string();
        assert!(
            direct.contains("Invalid tool arguments: "),
            "the non-streaming loop's item: {direct}"
        );
        let direct_output = function_call_output(&state);
        assert!(
            direct_output.starts_with("{\"error\":\"Invalid tool arguments: ") && direct_output == streamed_output,
            "the function_call_output the model reads back, both loops: {direct_output} vs {streamed_output}"
        );

        let series: Vec<String> = handle
            .render()
            .lines()
            .filter(|line| {
                line.contains("smg_mcp_tool_calls_total") && line.contains("wiki_search")
            })
            .map(str::to_owned)
            .collect();
        assert_eq!(series.len(), 1, "one errored-call series: {series:?}");
        assert!(
            series[0].contains("result=\"error\"") && series[0].ends_with(" 2"),
            "both loops counted: {}",
            series[0]
        );
    }

    /// A streamed call the session cannot run is closed on the wire the way
    /// an executed one is: the format's completed event, then the item's
    /// `response.output_item.done`, in sequence, under the id the stream
    /// announced for the call; the loop's state records the call with its
    /// error output and its errored-call metric counts it once.
    #[tokio::test]
    async fn a_streamed_call_is_closed_on_the_wire_under_its_announced_id_and_recorded() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _guard = metrics::set_default_local_recorder(&recorder);
        let (orchestrator, bindings) = function_tool_session().await;
        let session = McpToolSession::new(&orchestrator, bindings, "test-request");
        let registry = FormatRegistry::new();

        let (tx, mut rx) = mpsc::channel(16);
        let mut state = ToolLoopState::new(Vec::new());
        let mut sequence_number = 7u64;
        let mut call = super::FunctionCallInProgress::new("call_9".to_string(), 3);
        call.name = "not_exposed".to_string();
        call.arguments_buffer = "{\"q\":1}".to_string();
        let client_stayed = super::execute_streaming_tool_calls(
            vec![call],
            &session,
            &registry,
            &tx,
            &mut state,
            &mut sequence_number,
            "m",
            &[],
            None,
        )
        .await;
        assert!(client_stayed);
        drop(tx);

        let events = drain_channel(&mut rx);
        let types: Vec<String> = events
            .iter()
            .map(|e| event_type_from_sse_block(e))
            .collect();
        assert_eq!(
            types,
            ["response.mcp_call.completed", "response.output_item.done"],
            "the call's events, in order"
        );
        assert_eq!(sequence_number, 9, "one sequence number per event");
        let done: serde_json::Value = events[1]
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .and_then(|data| serde_json::from_str(data).ok())
            .expect("the done event's data");
        assert_eq!(done["sequence_number"], 8);
        assert_eq!(done["output_index"], 3);
        assert_eq!(done["item"]["id"], "mcp_9", "the id the stream announced");
        assert_eq!(done["item"]["type"], "mcp_call");
        assert_eq!(done["item"]["status"], "failed");

        assert_eq!(state.mcp_call_items.len(), 1);
        assert_eq!(
            state.mcp_call_items[0], done["item"],
            "the recorded item is the streamed one"
        );
        assert!(
            function_call_output(&state).contains("is not in this session's exposed tool map"),
            "the output the model reads back: {}",
            function_call_output(&state)
        );
        let series: Vec<String> = handle
            .render()
            .lines()
            .filter(|line| {
                line.contains("smg_mcp_tool_calls_total") && line.contains("not_exposed")
            })
            .map(str::to_owned)
            .collect();
        assert_eq!(series.len(), 1, "one errored-call series: {series:?}");
        assert!(
            series[0].contains("result=\"error\"") && series[0].ends_with(" 1"),
            "counted once: {}",
            series[0]
        );
    }
}
