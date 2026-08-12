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

use axum::http::HeaderMap;
use bytes::Bytes;
use openai_protocol::{
    event_types::{is_function_call_type, ItemType, McpEvent, OutputItemEvent},
    responses::{generate_id, ResponseInput, ResponseTool, ResponsesRequest},
};
use serde_json::{json, to_value, Value};
use smg_mcp::{McpServerBinding, McpToolSession, ToolExecutionInput, ToolExecutionResult};
use tracing::{debug, info, warn};

use super::tool_handler::FunctionCallInProgress;
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::{
        common::{
            header_utils::ApiProvider,
            mcp_utils::{prepare_hosted_dispatch_args, DEFAULT_MAX_ITERATIONS},
            openai_bridge::{
                self, extract_embedded_openai_responses, mcp_response_item_id, FormatRegistry,
                ResponseFormat, ResponseTransformer,
            },
            sse::SseSender,
        },
        error,
        openai::responses::utils,
    },
};

/// State for tracking multi-turn tool calling loop
pub(crate) struct ToolLoopState {
    /// Current iteration number (starts at 0, increments with each tool call)
    pub iteration: usize,
    /// Total number of tool calls executed
    pub total_calls: usize,
    /// Conversation history (function_call and function_call_output items)
    pub conversation_history: Vec<Value>,
    /// Original user input (preserved for building resume payloads)
    pub original_input: ResponseInput,
    /// MCP bindings already represented by historical `mcp_list_tools` items.
    pub existing_mcp_list_tools_labels: HashSet<String>,
    /// Transformed output items (mcp_call, web_search_call, etc.) - stored to avoid reconstruction
    pub mcp_call_items: Vec<Value>,
}

impl ToolLoopState {
    pub fn new(original_input: ResponseInput, prior_mcp_list_tools_labels: Vec<String>) -> Self {
        let known_labels = prior_mcp_list_tools_labels
            .into_iter()
            .collect::<HashSet<_>>();

        Self {
            iteration: 0,
            total_calls: 0,
            conversation_history: Vec::new(),
            original_input,
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
    reason = "Streaming tool dispatch threads channel + state + per-request inputs \
              (tools, user) directly to the loop without an intermediate context struct."
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
    for call in pending_calls {
        if call.name.is_empty() {
            warn!(
                "Skipping incomplete tool call: name is empty, args_len={}",
                call.arguments_buffer.len()
            );
            continue;
        }

        info!(
            "Executing tool call during streaming: {} ({})",
            call.name, call.call_id
        );

        let args_str = if call.arguments_buffer.is_empty() {
            "{}"
        } else {
            &call.arguments_buffer
        };

        let response_format =
            openai_bridge::lookup_tool_format(session, format_registry, &call.name);
        let server_label = session.resolve_tool_server_label(&call.name);

        let mut arguments: Value = match serde_json::from_str(args_str) {
            Ok(v) => v,
            Err(e) => {
                let err_str = format!("Failed to parse tool arguments: {e}");
                warn!("{}", err_str);
                let error_output = json!({ "error": &err_str });
                let mut mcp_call_item = build_transformed_mcp_call_item(
                    &error_output,
                    response_format,
                    &call.call_id,
                    &server_label,
                    &call.name,
                    &call.arguments_buffer,
                );
                if let Some(obj) = mcp_call_item.as_object_mut() {
                    obj.insert(
                        "id".to_string(),
                        Value::String(stable_streaming_tool_item_id(&call, response_format)),
                    );
                }
                if !send_tool_call_completion_events(
                    tx,
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
                    session.is_builtin_tool(&call.name),
                    call.call_id,
                    call.name,
                    call.arguments_buffer,
                    error_output.to_string(),
                    mcp_call_item,
                );
                continue;
            }
        };

        if !send_tool_call_intermediate_event(tx, &call, response_format, sequence_number).await {
            return false;
        }

        // Coerce non-object payloads to `{}` before merging overrides — the
        // merge is a no-op on scalars/arrays and we don't want to silently
        // drop caller-declared hosted-tool config.
        if !matches!(arguments, Value::Object(_)) {
            arguments = json!({});
        }
        // Merge caller-declared hosted-tool configuration (e.g. `size`, `quality`
        // on image_generation) into dispatch args, then forward the request-
        // level `user` so a downstream MCP server can attribute per-user usage.
        // Both steps are no-ops for plain MCP function tools.
        prepare_hosted_dispatch_args(&mut arguments, response_format, request_tools, request_user);

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

        let output_str = tool_output.output.to_string();
        let mut mcp_call_item = to_value(openai_bridge::transform_tool_output(
            &tool_output,
            response_format,
        ))
        .unwrap_or_else(|e| {
            warn!(tool = %call.name, error = %e, "Failed to convert item to Value");
            json!({})
        });
        if let Some(obj) = mcp_call_item.as_object_mut() {
            obj.insert(
                "id".to_string(),
                Value::String(stable_streaming_tool_item_id(&call, response_format)),
            );
        }

        if !send_tool_call_completion_events(
            tx,
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
            session.is_builtin_tool(&call.name),
            call.call_id,
            call.name,
            call.arguments_buffer,
            output_str,
            mcp_call_item,
        );
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

/// Build a resume payload with conversation history
pub(crate) fn build_resume_payload(
    base_payload: &Value,
    conversation_history: &[Value],
    original_input: &ResponseInput,
    tools_json: &Value,
    is_streaming: bool,
) -> Result<Value, String> {
    let mut payload = base_payload.clone();

    let obj = payload
        .as_object_mut()
        .ok_or_else(|| "payload not an object".to_string())?;

    let mut input_array = Vec::with_capacity(1 + conversation_history.len());

    match original_input {
        ResponseInput::Text(text) => {
            let user_item = json!({
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": text }]
            });
            input_array.push(user_item);
        }
        ResponseInput::Items(items) => {
            let items_value =
                to_value(items).map_err(|e| format!("Failed to serialize input items: {e}"))?;
            if let Some(items_arr) = items_value.as_array() {
                input_array.extend_from_slice(items_arr);
            }
        }
    }

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

fn retained_output_items(output_array: &[Value], original_body: &ResponsesRequest) -> Vec<Value> {
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

    let mut state = ToolLoopState::new(
        original_body.input.clone(),
        existing_mcp_list_tools_labels.to_vec(),
    );
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
        let request_builder = client.post(url).json(&current_payload);
        let request_builder = provider.apply_headers(request_builder, auth_header.as_ref());

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

        let mut response_json = response
            .json::<Value>()
            .await
            .map_err(|e| format!("parse response: {e}"))?;

        let function_calls = extract_function_calls(&response_json);
        if function_calls.is_empty() {
            info!(
                "Tool loop completed: {} iterations, {} total calls",
                state.iteration, state.total_calls
            );
            if state.total_calls > 0 {
                inject_mcp_metadata_streaming(&mut response_json, &state, session);
            }
            return Ok(response_json);
        }

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
                    "max_tool_calls",
                    session,
                    original_body,
                );
            }
            let mut arguments: Value = match serde_json::from_str(&call.arguments) {
                Ok(v) => v,
                Err(e) => {
                    warn!(tool = %call.name, error = %e, "Failed to parse tool arguments as JSON");
                    let error_output = format!("Invalid tool arguments: {e}");
                    let response_format =
                        openai_bridge::lookup_tool_format(session, format_registry, &call.name);
                    let server_label = session.resolve_tool_server_label(&call.name);
                    let tool_item_id =
                        non_streaming_tool_item_id_source(&call.item_id, response_format);
                    let error_json = json!({ "error": &error_output });
                    let transformed_item = build_transformed_mcp_call_item(
                        &error_json,
                        response_format,
                        &tool_item_id,
                        &server_label,
                        &call.name,
                        &call.arguments,
                    );

                    Metrics::record_mcp_tool_call(
                        &original_body.model,
                        &call.name,
                        metrics_labels::RESULT_ERROR,
                    );

                    state.record_call(
                        session.is_builtin_tool(&call.name),
                        call.call_id,
                        call.name,
                        call.arguments,
                        error_output,
                        transformed_item,
                    );
                    continue;
                }
            };

            // Coerce non-object payloads to `{}` before merging overrides —
            // the merge is a no-op on scalars/arrays and we don't want to
            // silently drop caller-declared hosted-tool config.
            if !matches!(arguments, Value::Object(_)) {
                arguments = json!({});
            }
            // Merge caller-declared hosted-tool configuration into dispatch args
            // and forward the request-level `user` so a downstream MCP server
            // can attribute per-user usage. Both steps are no-ops for plain
            // MCP function tools (Passthrough format).
            let response_format =
                openai_bridge::lookup_tool_format(session, format_registry, &call.name);
            prepare_hosted_dispatch_args(
                &mut arguments,
                response_format,
                original_body.tools.as_deref().unwrap_or(&[]),
                original_body.user.as_deref(),
            );

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

            let server_label = session.resolve_tool_server_label(&call.name);
            let tool_item_id = non_streaming_tool_item_id_source(&call.item_id, response_format);
            let approval_request_id = approval_request_item_id_source(&call.item_id);

            let tool_output = match tool_result {
                ToolExecutionResult::Executed(tool_output) => tool_output,
                ToolExecutionResult::PendingApproval(pending) => {
                    let approval_item = build_mcp_approval_request_item(
                        &approval_request_id,
                        &pending.tool_name,
                        &effective_arguments,
                        &server_label,
                    );
                    return build_approval_response(
                        response_json,
                        state,
                        session,
                        original_body,
                        approval_item,
                    );
                }
            };

            Metrics::record_mcp_tool_duration(
                &original_body.model,
                &tool_output.tool_name,
                tool_output.duration,
            );
            Metrics::record_mcp_tool_call(
                &original_body.model,
                &tool_output.tool_name,
                if tool_output.is_error {
                    metrics_labels::RESULT_ERROR
                } else {
                    metrics_labels::RESULT_SUCCESS
                },
            );

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
        }

        current_payload = build_resume_payload(
            &base_payload,
            &state.conversation_history,
            &state.original_input,
            &tools_json,
            false,
        )?;
    }
}

/// Build an incomplete response when limits are exceeded
fn build_incomplete_response(
    mut response: Value,
    state: ToolLoopState,
    reason: &str,
    session: &McpToolSession<'_>,
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
                let args = item
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");

                // Mark as incomplete - not executed
                let resolved_label = session.resolve_tool_server_label(tool_name);
                let mcp_call_item = build_mcp_call_item(
                    tool_name,
                    args,
                    "", // No output - wasn't executed
                    &resolved_label,
                    false, // Not successful
                    Some("Not executed - response stopped due to limit"),
                );
                incomplete_items.push(mcp_call_item);
            }
        }

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
        "error": error,
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

    use serde_json::json;
    use smg_mcp::{
        BuiltinToolType, McpConfig, McpOrchestrator, McpServerBinding, McpServerConfig,
        McpToolSession, McpTransport, Tool, ToolEntry,
    };
    use tokio::sync::mpsc;

    use super::{
        build_transformed_mcp_call_item, extract_openai_response_output_items,
        is_internal_mcp_response_item, mcp_list_tools_bindings_to_emit, ResponseInput,
        ToolLoopState,
    };
    use crate::routers::common::openai_bridge::ResponseFormat;

    fn test_tool(name: &str) -> Tool {
        let mut schema = serde_json::Map::new();
        schema.insert("type".to_string(), json!("object"));
        schema.insert("properties".to_string(), json!({}));

        Tool::new(name.to_string(), "internal", schema)
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
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");

        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("brave_web_search"),
            ));

        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );

        let item = json!({
            "type": "mcp_call",
            "name": "brave_web_search",
            "server_label": "internal-label"
        });

        assert!(!is_internal_mcp_response_item(&item, &session));
    }

    #[tokio::test]
    async fn internal_filter_keeps_builtin_passthrough_mcp_call_items_with_mixed_tools() {
        let orchestrator = McpOrchestrator::new(McpConfig {
            servers: vec![McpServerConfig {
                name: "internal-server".to_string(),
                transport: McpTransport::Sse {
                    url: "http://localhost:3000/sse".to_string(),
                    token: None,
                    headers: Default::default(),
                },
                proxy: None,
                required: false,
                tools: None,
                builtin_type: Some(BuiltinToolType::WebSearchPreview),
                builtin_tool_name: Some("brave_web_search".to_string()),
                internal: true,
            }],
            ..Default::default()
        })
        .await
        .expect("orchestrator");

        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("brave_web_search"),
            ));
        orchestrator
            .tool_inventory()
            .insert_entry(ToolEntry::from_server_tool(
                "internal-server",
                test_tool("internal_non_builtin_tool"),
            ));

        let session = McpToolSession::new(
            &orchestrator,
            vec![McpServerBinding {
                label: "internal-label".to_string(),
                server_key: "internal-server".to_string(),
                allowed_tools: None,
            }],
            "test-request",
        );

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
        let mut state = ToolLoopState::new(ResponseInput::Text("hello".to_string()), Vec::new());
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
        let mut state = ToolLoopState::new(ResponseInput::Text("hello".to_string()), Vec::new());
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
            &ResponseInput::Text("q".to_string()),
            &json!([{"type": "function", "name": "ask_wiki"}]),
            false,
        )
        .expect("resume payload");

        assert_eq!(resumed["tool_choice"], json!("auto"));
    }
}
