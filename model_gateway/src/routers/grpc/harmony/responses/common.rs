//! Shared helpers and state tracking for Harmony Responses

use std::collections::HashSet;

use axum::response::Response;
use openai_protocol::{
    common::ToolCall,
    responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
        ResponseReasoningContent, ResponseTool, ResponsesRequest, ResponsesResponse,
        ResponsesToolChoice, StringOrContentParts, ToolChoiceOptions,
    },
};
use serde_json::{from_value, to_string, Value};
use smg_data_connector::{ResponseId, ResponseStorageError};
use smg_mcp::McpToolSession;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::execution::ToolResult;
use crate::routers::{
    common::openai_bridge::{self, FormatRegistry, ResponseFormat},
    error,
    grpc::common::responses::{utils::resolve_function_identity, ResponsesContext},
};

/// Record of a single MCP tool call execution
///
/// Stores the transformed output item for Responses API format.
/// The output_item respects the tool's response_format configuration.
#[derive(Debug, Clone)]
pub(super) struct McpCallRecord {
    /// Transformed output item (McpCall, WebSearchCall, etc.)
    pub output_item: ResponseOutputItem,
}

/// Tracking structure for MCP tool calls across iterations
///
/// Accumulates all MCP tool call metadata during multi-turn conversation
/// so we can build proper mcp_list_tools and mcp_call output items.
#[derive(Debug, Clone)]
pub(super) struct McpCallTracking {
    /// All tool call records across all iterations
    pub tool_calls: Vec<McpCallRecord>,
}

impl McpCallTracking {
    pub fn new() -> Self {
        Self {
            tool_calls: Vec::new(),
        }
    }

    pub fn record_call(&mut self, output_item: ResponseOutputItem) {
        self.tool_calls.push(McpCallRecord { output_item });
    }

    pub fn total_calls(&self) -> usize {
        self.tool_calls.len()
    }
}

/// Build next request with tool results appended to history
///
/// Constructs a new ResponsesRequest with:
/// 1. Original input items (preserved)
/// 2. Assistant message with analysis (reasoning) + partial_text + tool_calls
/// 3. Tool result messages for each tool execution
pub(super) fn build_next_request_with_tools(
    mut request: ResponsesRequest,
    tool_calls: Vec<ToolCall>,
    tool_results: Vec<ToolResult>,
    analysis: Option<String>, // Analysis channel content (becomes reasoning content)
    partial_text: String,     // Final channel content (becomes message content)
) -> ResponsesRequest {
    // Get current input items (or empty vec if Text variant)
    let mut items = match request.input {
        ResponseInput::Items(items) => items,
        ResponseInput::Text(text) => {
            // Convert text to items format
            vec![ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String(text),
                role: "user".to_string(),
                r#type: None,
                phase: None,
            }]
        }
    };

    // Build assistant response item with reasoning + content + tool calls
    // This represents what the model generated in this iteration
    let assistant_id = format!("msg_{}", Uuid::now_v7());

    // Add reasoning if present (from analysis channel)
    if let Some(analysis_text) = analysis {
        items.push(ResponseInputOutputItem::new_reasoning(
            format!("reasoning_{assistant_id}"),
            vec![],
            vec![ResponseReasoningContent::ReasoningText {
                text: analysis_text,
            }],
            Some("completed".to_string()),
        ));
    }

    // Add message content if present (from final channel)
    if !partial_text.is_empty() {
        items.push(ResponseInputOutputItem::Message {
            id: assistant_id.clone(),
            role: "assistant".to_string(),
            content: vec![ResponseContentPart::OutputText {
                text: partial_text,
                annotations: vec![],
                logprobs: None,
            }],
            status: Some("completed".to_string()),
            phase: None,
        });
    }

    // Add function tool calls (from commentary channel)
    for tool_call in tool_calls {
        let (name, namespace) =
            resolve_function_identity(request.tools.as_deref(), &tool_call.function.name);
        items.push(ResponseInputOutputItem::FunctionToolCall {
            id: Some(tool_call.id.clone()),
            call_id: tool_call.id.clone(),
            name,
            namespace,
            arguments: tool_call
                .function
                .arguments
                .unwrap_or_else(|| "{}".to_string()),
            output: None, // Output will be added next
            status: Some("in_progress".to_string()),
        });
    }

    // Add tool results
    for tool_result in tool_results {
        // Serialize tool output to string
        let output_str = to_string(&tool_result.output)
            .unwrap_or_else(|e| format!("{{\"error\": \"Failed to serialize tool output: {e}\"}}"));

        // Update the corresponding tool call with output and completed status
        // Find and update the matching FunctionToolCall
        if let Some(ResponseInputOutputItem::FunctionToolCall {
            output,
            status,
            ..
        }) = items
            .iter_mut()
            .find(|item| matches!(item, ResponseInputOutputItem::FunctionToolCall { call_id, .. } if call_id == &tool_result.call_id))
        {
            *output = Some(output_str);
            *status = if tool_result.is_error {
                Some("failed".to_string())
            } else {
                Some("completed".to_string())
            };
        }
    }

    // Update request with new items
    request.input = ResponseInput::Items(items);

    // Switch tool_choice to "auto" for subsequent iterations
    // This prevents infinite loops when original tool_choice was "required" or specific function
    // After receiving tool results, the model should be free to decide whether to call more tools or finish
    request.tool_choice = Some(ResponsesToolChoice::Options(ToolChoiceOptions::Auto));

    request
}

pub(super) fn inject_mcp_metadata(
    response: &mut ResponsesResponse,
    tracking: &McpCallTracking,
    session: &McpToolSession<'_>,
    user_function_names: &HashSet<String>,
) {
    let tool_output_items: Vec<ResponseOutputItem> = tracking
        .tool_calls
        .iter()
        .map(|record| record.output_item.clone())
        .collect();

    openai_bridge::inject_client_visible_mcp_output_items(
        session,
        &mut response.output,
        tool_output_items,
        user_function_names,
    );
}

/// Load previous conversation messages from storage
///
/// If the request has `previous_response_id`, loads the response chain from storage
/// and prepends the conversation history to the request input items.
pub(super) async fn load_previous_messages(
    ctx: &ResponsesContext,
    request: ResponsesRequest,
) -> Result<ResponsesRequest, Response> {
    let Some(ref prev_id_str) = request.previous_response_id else {
        // No previous_response_id, return request as-is
        return Ok(request);
    };

    let prev_id = ResponseId::from(prev_id_str.as_str());

    // Load response chain from storage
    let chain = match ctx
        .response_storage
        .get_response_chain(&prev_id, None)
        .await
    {
        Ok(chain) => chain,
        Err(ResponseStorageError::ResponseNotFound(_)) => {
            return Err(error::bad_request(
                "previous_response_not_found",
                format!("Previous response with id '{prev_id_str}' not found."),
            ));
        }
        Err(e) => {
            error!(
                function = "load_previous_messages",
                prev_id = %prev_id_str,
                error = %e,
                "Failed to load previous response chain from storage"
            );
            return Err(error::internal_error(
                "load_previous_response_chain_failed",
                format!("Failed to load previous response chain for {prev_id_str}: {e}"),
            ));
        }
    };

    if chain.responses.is_empty() {
        return Err(error::bad_request(
            "previous_response_not_found",
            format!("Previous response with id '{prev_id_str}' not found."),
        ));
    }

    // Build conversation history from stored responses
    let mut history_items = Vec::new();

    // Helper to deserialize and collect items from a JSON array
    let deserialize_items = |arr: &Value, item_type: &str| -> Vec<ResponseInputOutputItem> {
        arr.as_array()
            .into_iter()
            .flat_map(|items| items.iter())
            .filter_map(|item| {
                from_value::<ResponseInputOutputItem>(item.clone())
                    .map_err(|e| {
                        warn!(
                            "Failed to deserialize stored {} item: {}. Item: {}",
                            item_type, e, item
                        );
                    })
                    .ok()
            })
            .collect()
    };

    for stored in &chain.responses {
        history_items.extend(deserialize_items(&stored.input, "input"));
        history_items.extend(deserialize_items(
            stored
                .raw_response
                .get("output")
                .unwrap_or(&Value::Array(vec![])),
            "output",
        ));
    }

    debug!(
        previous_response_id = %prev_id_str,
        history_items_count = history_items.len(),
        "Loaded conversation history from previous response"
    );

    // Build modified request with history prepended
    let mut modified_request = request;

    // Convert current input to items format
    let all_items = match modified_request.input {
        ResponseInput::Items(items) => {
            // Prepend history to existing items
            let mut combined = history_items;
            combined.extend(items);
            combined
        }
        ResponseInput::Text(text) => {
            // Convert text to item and prepend history
            history_items.push(ResponseInputOutputItem::SimpleInputMessage {
                content: StringOrContentParts::String(text),
                role: "user".to_string(),
                r#type: None,
                phase: None,
            });
            history_items
        }
    };

    // Update request with combined items and clear previous_response_id
    modified_request.input = ResponseInput::Items(all_items);
    modified_request.previous_response_id = None;

    Ok(modified_request)
}

/// Strip `ResponseTool::ImageGeneration` from a request's tools list once
/// the MCP session has exposed an MCP-routed replacement.
///
/// The harmony builder synthesizes a function-tool description named
/// `image_generation` from
/// [`openai_protocol::responses::ResponseTool::ImageGeneration`]. That is
/// the right advertisement when no MCP server is configured, but once the
/// MCP loop injects a `ResponseTool::Function` for the MCP-exposed
/// image-generation tool (via `convert_mcp_tools_to_response_tools`), two
/// function-tool entries can end up in the developer-message `namespace
/// functions { … }`. Worse, if the MCP server is configured with a
/// non-canonical `builtin_tool_name` (e.g. `generate_image`), the
/// advertised `image_generation` synthesis will not be recognized as
/// MCP-exposed by `McpToolSession::has_exposed_tool`, so any call the
/// model makes against the synthesized name falls through to the
/// unresolved function-call path instead of dispatching.
///
/// Removing the `ResponseTool::ImageGeneration` tag once the MCP loop has
/// taken ownership keeps the advertisement single-source (the MCP-exposed
/// Function tool) and guarantees the model only sees a name the session
/// can dispatch against.
pub(super) fn strip_image_generation_from_request_tools(
    request: &mut ResponsesRequest,
    session: &McpToolSession<'_>,
    format_registry: &FormatRegistry,
) {
    // Strip only when the session exposes the literal `image_generation`
    // name routed to an `ImageGenerationCall` format. Checking every MCP
    // tool's format would also fire for unrelated custom tools that happen
    // to share the image-generation output shape (e.g. a `thumbnailer`),
    // and would then drop the real hosted `image_generation` tag even
    // though the session has no dispatcher for it.
    let mcp_has_image_generation = session
        .qualified_name_for_exposed("image_generation")
        .is_some_and(|qn| {
            matches!(
                format_registry.lookup(&qn),
                ResponseFormat::ImageGenerationCall
            )
        });

    if !mcp_has_image_generation {
        return;
    }

    if let Some(tools) = request.tools.as_mut() {
        let before = tools.len();
        tools.retain(|t| !matches!(t, ResponseTool::ImageGeneration(_)));
        let after = tools.len();
        if before != after {
            debug!(
                removed = before - after,
                "Stripped ResponseTool::ImageGeneration from request.tools because the \
                 MCP session exposes an image_generation-routed dispatcher",
            );
        }
    }
}

#[cfg(test)]
mod namespace_tests {
    use super::*;
    use crate::routers::grpc::common::responses::utils::namespace_test_request;
    #[test]
    fn namespace_replay_preserves_structured_identity() {
        let request: ResponsesRequest = namespace_test_request();
        let call = from_value(serde_json::json!({"id":"call_test","type":"function","function":{"name":"weather.lookup","arguments":"{}"}})).unwrap();
        let replay =
            build_next_request_with_tools(request, vec![call], vec![], None, String::new());
        let ResponseInput::Items(items) = replay.input else {
            panic!("expected replay items");
        };
        let wire = serde_json::to_value(items.last().unwrap()).unwrap();

        assert_eq!(wire["name"], "lookup");
        assert_eq!(wire["namespace"], "weather");
        assert_eq!(wire["call_id"], "call_test");
    }
}
