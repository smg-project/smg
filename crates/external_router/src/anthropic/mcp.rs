//! MCP tool processing layer for Anthropic Messages API
//!
//! Provides tool injection, tool execution, and the standard
//! `process_iteration` interface used by both streaming and
//! non-streaming processors.

use std::collections::{hash_map::Entry, HashMap};

use openai_protocol::messages::{
    ContentBlock, CreateMessageRequest, CustomTool, InputContentBlock, InputSchema, Message,
    RedactedThinkingBlock, ServerToolUseBlock, StopReason, TextBlock, ThinkingBlock, Tool,
    ToolChoice, ToolReferenceBlock, ToolResultBlock, ToolResultContent, ToolResultContentBlock,
    ToolSearchToolResultBlock, ToolUseBlock, WebSearchToolResultBlock,
};
use serde_json::Value;
use smg_mcp::{McpToolSession, ToolEntry, ToolExecutionInput};
use tracing::{debug, info, warn};

use crate::metrics::{metrics_labels, Metrics};

// ============================================================================
// Standard I/O types for processor ↔ MCP layer communication
// ============================================================================

/// What processors extract from an LLM response (streaming or non-streaming).
pub(crate) struct IterationResult {
    pub content_blocks: Vec<ContentBlock>,
    pub tool_use_blocks: Vec<ToolUseBlock>,
    pub stop_reason: Option<StopReason>,
}

impl IterationResult {
    /// Construct from a non-streaming `Message`.
    pub(crate) fn from_message(message: &Message) -> Self {
        let tool_use_blocks = extract_tool_calls(&message.content);
        Self {
            content_blocks: message.content.clone(),
            tool_use_blocks,
            stop_reason: message.stop_reason.clone(),
        }
    }
}

/// Continuation data when the tool loop should proceed.
pub(crate) struct ToolLoopContinuation {
    pub mcp_calls: Vec<McpToolCall>,
    pub assistant_blocks: Vec<InputContentBlock>,
    pub tool_result_blocks: Vec<InputContentBlock>,
}

/// Decision from `process_iteration`: stop, continue, or error.
pub(crate) enum ToolLoopAction {
    Done,
    Continue(ToolLoopContinuation),
    Error(String),
}

// ============================================================================
// Tracked MCP tool call
// ============================================================================

/// Tracked MCP tool call for response reconstruction and SSE emission.
pub(crate) struct McpToolCall {
    pub original_id: String,
    pub mcp_id: String,
    pub name: String,
    pub server_name: String,
    pub input: Value,
    pub result_content: String,
    pub is_error: bool,
}

// ============================================================================
// Standard process_iteration interface
// ============================================================================

/// Evaluate one iteration of the MCP tool loop.
///
/// Returns `Done` if there are no tool calls or the stop reason is not `ToolUse`.
/// Returns `Continue` with executed tool results when the loop should proceed.
pub(crate) async fn process_iteration(
    result: &IterationResult,
    session: &McpToolSession<'_>,
    model_id: &str,
) -> ToolLoopAction {
    if result.tool_use_blocks.is_empty() {
        return ToolLoopAction::Done;
    }

    if result.stop_reason != Some(StopReason::ToolUse) {
        let msg = format!(
            "Model returned {} tool_use block(s) but stop_reason is {:?}; tool calls will not be executed",
            result.tool_use_blocks.len(),
            result.stop_reason
        );
        warn!(
            tool_count = result.tool_use_blocks.len(),
            stop_reason = ?result.stop_reason,
            "{}", &msg
        );
        return ToolLoopAction::Error(msg);
    }

    info!(
        tool_count = result.tool_use_blocks.len(),
        "Executing MCP tool calls"
    );

    let (mcp_calls, assistant_blocks, tool_result_blocks) = execute_mcp_tool_calls(
        &result.content_blocks,
        &result.tool_use_blocks,
        session,
        model_id,
    )
    .await;

    ToolLoopAction::Continue(ToolLoopContinuation {
        mcp_calls,
        assistant_blocks,
        tool_result_blocks,
    })
}

// ============================================================================
// MCP connection and tool injection
// ============================================================================

pub(crate) fn extract_tool_calls(content: &[ContentBlock]) -> Vec<ToolUseBlock> {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, input } => Some(ToolUseBlock {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
                cache_control: None,
            }),
            _ => None,
        })
        .collect()
}

/// Strip `mcp_servers` from request and inject MCP tools as regular tools.
///
/// Tool filtering is already applied by `McpServerBinding.allowed_tools` during
/// session creation — this function just converts and injects.
/// Respects `defer_loading` settings from `McpToolset` entries.
pub(crate) fn inject_mcp_tools_into_request(
    request: &mut CreateMessageRequest,
    session: &McpToolSession<'_>,
) {
    let defer_configs = collect_defer_loading_per_server(request.tools.as_ref());

    request.mcp_servers = None;

    let mut tools: Vec<Tool> = request
        .tools
        .take()
        .unwrap_or_default()
        .into_iter()
        .filter(|t| !matches!(t, Tool::McpToolset(_)))
        .collect();

    for entry in session.mcp_tools() {
        let server_label = session.resolve_tool_server_label(entry.tool.name.as_ref());
        let defer_loading = defer_configs
            .get(&server_label)
            .map(|cfg| {
                cfg.per_tool_overrides
                    .get(entry.tool.name.as_ref())
                    .copied()
                    .unwrap_or(cfg.default_defer)
            })
            .unwrap_or(false);

        tools.push(Tool::Custom(convert_tool_entry_to_anthropic_tool(
            entry,
            defer_loading,
        )));
    }

    if !tools.is_empty() {
        request.tools = Some(tools);
        if request.tool_choice.is_none() {
            request.tool_choice = Some(ToolChoice::Auto {
                disable_parallel_tool_use: None,
            });
        }
    }
}

// ============================================================================
// Tool execution
// ============================================================================

/// Execute MCP tool calls and return `(mcp_calls, assistant_blocks, tool_result_blocks)`.
async fn execute_mcp_tool_calls(
    content: &[ContentBlock],
    tool_calls: &[ToolUseBlock],
    session: &McpToolSession<'_>,
    model_id: &str,
) -> (
    Vec<McpToolCall>,
    Vec<InputContentBlock>,
    Vec<InputContentBlock>,
) {
    let assistant_content_blocks = build_assistant_content_blocks(content);

    let mut mcp_calls = Vec::new();
    let mut tool_result_blocks: Vec<InputContentBlock> = Vec::new();

    for tool_call in tool_calls {
        let server_name = session.resolve_tool_server_label(&tool_call.name);

        debug!(
            tool = %tool_call.name,
            server = %server_name,
            "Executing MCP tool call"
        );

        let input = ToolExecutionInput {
            call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            arguments: tool_call.input.clone(),
        };

        let output = session.execute_tool(input).await;

        Metrics::record_mcp_tool_duration(model_id, &output.tool_name, output.duration);
        Metrics::record_mcp_tool_call(
            model_id,
            &output.tool_name,
            if output.is_error {
                metrics_labels::RESULT_ERROR
            } else {
                metrics_labels::RESULT_SUCCESS
            },
        );

        let result_content = extract_output_from_value(&output.output);
        let is_error = output.is_error;

        if is_error {
            let err_msg = output.error_message.as_deref().unwrap_or("unknown error");
            warn!(tool = %tool_call.name, error = %err_msg, "MCP tool execution failed");
        }

        let mcp_id = format!("mcptoolu_{}", tool_call.id.trim_start_matches("toolu_"));

        mcp_calls.push(McpToolCall {
            original_id: tool_call.id.clone(),
            mcp_id: mcp_id.clone(),
            name: tool_call.name.clone(),
            server_name,
            input: tool_call.input.clone(),
            result_content: result_content.clone(),
            is_error,
        });

        tool_result_blocks.push(InputContentBlock::ToolResult(ToolResultBlock {
            tool_use_id: tool_call.id.clone(),
            content: Some(ToolResultContent::String(result_content)),
            is_error: is_error.then_some(true),
            cache_control: None,
        }));
    }

    (mcp_calls, assistant_content_blocks, tool_result_blocks)
}

// ============================================================================
// Response reconstruction
// ============================================================================

/// Replace tool_use blocks with mcp_tool_use/mcp_tool_result pairs in the response.
///
/// Builds the final response by combining:
/// 1. Prior iteration content blocks (e.g., server_tool_use, tool_search_tool_result, text)
///    with their ToolUse blocks replaced by mcp_tool_use/mcp_tool_result pairs
/// 2. MCP tool_use/tool_result pairs for the current iteration's tool calls
/// 3. The final message's content blocks (e.g., the end_turn text) with matched
///    ToolUse blocks skipped
pub(crate) fn rebuild_response_with_mcp_blocks(
    mut message: Message,
    mcp_calls: &[McpToolCall],
    prior_content_blocks: &[ContentBlock],
) -> Message {
    let call_lookup: HashMap<&str, &McpToolCall> = mcp_calls
        .iter()
        .map(|c| (c.original_id.as_str(), c))
        .collect();

    let mut new_content: Vec<ContentBlock> = Vec::new();

    // First: emit prior iteration content, replacing ToolUse with MCP pairs
    for block in prior_content_blocks {
        match block {
            ContentBlock::ToolUse { id, .. } if call_lookup.contains_key(id.as_str()) => {
                let call = call_lookup[id.as_str()];
                push_mcp_blocks(&mut new_content, call);
            }
            _ => new_content.push(block.clone()),
        }
    }

    // Then: emit MCP pairs for any calls NOT already covered by prior blocks
    for call in mcp_calls {
        if !prior_content_blocks
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { id, .. } if id == &call.original_id))
        {
            push_mcp_blocks(&mut new_content, call);
        }
    }

    // Finally: append the final message's content (e.g., end_turn text),
    // skipping ToolUse blocks already covered by MCP pairs
    for block in std::mem::take(&mut message.content) {
        match &block {
            ContentBlock::ToolUse { id, .. } if call_lookup.contains_key(id.as_str()) => {
                continue;
            }
            _ => new_content.push(block),
        }
    }

    message.content = new_content;
    message
}

/// Extract per-server allowed tool names from `McpToolset` entries.
///
/// Returns a map of `mcp_server_name` → `Option<Vec<String>>`:
/// - `None` → all tools allowed for that server
/// - `Some(names)` → only these tools allowed
pub(crate) fn collect_allowed_tools_per_server(
    tools: Option<&Vec<Tool>>,
) -> HashMap<String, Option<Vec<String>>> {
    let Some(tools) = tools else {
        return HashMap::new();
    };

    let mut result: HashMap<String, Option<Vec<String>>> = HashMap::new();

    for tool in tools {
        let Tool::McpToolset(toolset) = tool else {
            continue;
        };

        let default_enabled = toolset
            .default_config
            .as_ref()
            .and_then(|c| c.enabled)
            .unwrap_or(true);

        let allowed = match &toolset.configs {
            Some(configs) => {
                let names: Vec<String> = configs
                    .iter()
                    .filter(|(_, config)| config.enabled.unwrap_or(default_enabled))
                    .map(|(name, _)| name.clone())
                    .collect();
                Some(names)
            }
            None if default_enabled => None,
            None => Some(Vec::new()),
        };

        if result.contains_key(&toolset.mcp_server_name) {
            warn!(
                server_name = %toolset.mcp_server_name,
                "Duplicate mcp_toolset for the same server; keeping first entry"
            );
            continue;
        }
        result.insert(toolset.mcp_server_name.clone(), allowed);
    }

    result
}

/// Per-server defer loading configuration.
pub(crate) struct DeferConfig {
    /// Default defer_loading for all tools from this server.
    pub default_defer: bool,
    /// Per-tool overrides: tool name → defer_loading.
    pub per_tool_overrides: HashMap<String, bool>,
}

/// Extract per-server defer loading configuration from `McpToolset` entries.
///
/// Follows the same `default_config` / per-tool `configs` pattern as
/// `collect_allowed_tools_per_server()`.
pub(crate) fn collect_defer_loading_per_server(
    tools: Option<&Vec<Tool>>,
) -> HashMap<String, DeferConfig> {
    let Some(tools) = tools else {
        return HashMap::new();
    };

    let mut result: HashMap<String, DeferConfig> = HashMap::new();

    for tool in tools {
        let Tool::McpToolset(toolset) = tool else {
            continue;
        };

        let default_defer = toolset
            .default_config
            .as_ref()
            .and_then(|c| c.defer_loading)
            .unwrap_or(false);

        let per_tool_overrides = toolset
            .configs
            .as_ref()
            .map(|configs| {
                configs
                    .iter()
                    .filter_map(|(name, config)| config.defer_loading.map(|dl| (name.clone(), dl)))
                    .collect()
            })
            .unwrap_or_default();

        // First entry wins (same as collect_allowed_tools_per_server)
        if let Entry::Vacant(entry) = result.entry(toolset.mcp_server_name.clone()) {
            entry.insert(DeferConfig {
                default_defer,
                per_tool_overrides,
            });
        }
    }

    result
}

// ============================================================================
// Private helpers
// ============================================================================

/// Convert an MCP `ToolEntry` to an Anthropic `CustomTool`.
fn convert_tool_entry_to_anthropic_tool(entry: &ToolEntry, defer_loading: bool) -> CustomTool {
    let schema_map = (*entry.tool.input_schema).clone();
    let schema_type = schema_map
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("object")
        .to_string();

    let properties = schema_map
        .get("properties")
        .and_then(|v| v.as_object())
        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect());

    let required = schema_map
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        });

    let additional: HashMap<String, Value> = schema_map
        .into_iter()
        .filter(|(k, _)| k != "type" && k != "properties" && k != "required")
        .collect();

    let input_schema = InputSchema {
        schema_type,
        properties,
        required,
        additional,
    };

    CustomTool {
        name: entry.tool.name.to_string(),
        tool_type: None,
        description: entry.tool.description.as_ref().map(|d| d.to_string()),
        input_schema,
        defer_loading: defer_loading.then_some(true),
        cache_control: None,
    }
}

fn build_assistant_content_blocks(content: &[ContentBlock]) -> Vec<InputContentBlock> {
    let mut blocks = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text { text, citations } => {
                blocks.push(InputContentBlock::Text(TextBlock {
                    text: text.clone(),
                    cache_control: None,
                    citations: citations.clone(),
                }));
            }
            ContentBlock::ToolUse { id, name, input } => {
                blocks.push(InputContentBlock::ToolUse(ToolUseBlock {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    cache_control: None,
                }));
            }
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                blocks.push(InputContentBlock::Thinking(ThinkingBlock {
                    thinking: thinking.clone(),
                    signature: signature.clone(),
                }));
            }
            ContentBlock::RedactedThinking { data } => {
                blocks.push(InputContentBlock::RedactedThinking(RedactedThinkingBlock {
                    data: data.clone(),
                }));
            }
            ContentBlock::ServerToolUse { id, name, input } => {
                blocks.push(InputContentBlock::ServerToolUse(ServerToolUseBlock {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    cache_control: None,
                }));
            }
            ContentBlock::WebSearchToolResult {
                tool_use_id,
                content,
            } => {
                blocks.push(InputContentBlock::WebSearchToolResult(
                    WebSearchToolResultBlock {
                        tool_use_id: tool_use_id.clone(),
                        content: content.clone(),
                        cache_control: None,
                    },
                ));
            }
            ContentBlock::ToolSearchToolResult {
                tool_use_id,
                content,
            } => {
                blocks.push(InputContentBlock::ToolSearchToolResult(
                    ToolSearchToolResultBlock {
                        tool_use_id: tool_use_id.clone(),
                        content: content.clone(),
                        cache_control: None,
                    },
                ));
            }
            ContentBlock::ToolReference {
                tool_name,
                description,
            } => {
                blocks.push(InputContentBlock::ToolReference(ToolReferenceBlock {
                    block_type: "tool_reference".to_string(),
                    tool_name: tool_name.clone(),
                    description: description.clone(),
                }));
            }
            // MCP blocks are handled separately by rebuild_response_with_mcp_blocks
            ContentBlock::McpToolUse { .. } | ContentBlock::McpToolResult { .. } => {}
        }
    }
    blocks
}

fn push_mcp_blocks(content: &mut Vec<ContentBlock>, call: &McpToolCall) {
    content.push(ContentBlock::McpToolUse {
        id: call.mcp_id.clone(),
        name: call.name.clone(),
        server_name: call.server_name.clone(),
        input: call.input.clone(),
    });
    content.push(ContentBlock::McpToolResult {
        tool_use_id: call.mcp_id.clone(),
        content: Some(ToolResultContent::Blocks(vec![
            ToolResultContentBlock::Text(TextBlock {
                text: call.result_content.clone(),
                cache_control: None,
                citations: None,
            }),
        ])),
        is_error: call.is_error.then_some(true),
    });
}

/// Extract output content string from a `ToolExecutionOutput` value.
fn extract_output_from_value(output: &Value) -> String {
    if let Some(text) = output.as_str() {
        text.to_string()
    } else if let Some(arr) = output.as_array() {
        arr.iter()
            .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        output.to_string()
    }
}
