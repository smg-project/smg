//! Shared helpers and state tracking for Harmony Responses

use std::collections::HashSet;

use openai_protocol::{
    common::ToolCall,
    responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
        ResponseReasoningContent, ResponseTool, ResponsesRequest, ResponsesResponse,
        ResponsesToolChoice, StringOrContentParts, ToolChoiceOptions,
    },
};
use serde_json::to_string;
use smg_mcp::McpToolSession;
use tracing::debug;
use uuid::Uuid;

use super::execution::ToolResult;
use crate::routers::{
    common::openai_bridge::{self, FormatRegistry, ResponseFormat},
    grpc::common::responses::utils::resolve_function_identity,
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
    use serde_json::from_value;

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
