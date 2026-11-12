//! MCP tool processing layer for Anthropic Messages API
//!
//! Provides tool injection, tool execution, and the standard
//! `process_iteration` interface used by both streaming and
//! non-streaming processors.

use std::collections::{hash_map::Entry, HashMap};

use openai_protocol::messages::{
    ContentBlock, CreateMessageRequest, CustomTool, InputContent, InputContentBlock, InputMessage,
    InputSchema, Message, RedactedThinkingBlock, Role, ServerToolUseBlock, StopReason, TextBlock,
    ThinkingBlock, Tool, ToolChoice, ToolReferenceBlock, ToolResultBlock, ToolResultContent,
    ToolResultContentBlock, ToolSearchToolResultBlock, ToolUseBlock, WebSearchToolResultBlock,
};
use serde_json::{Map, Value};
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
    /// Construct from a non-streaming `Message`, taking its content: the
    /// loop that reads the result owns the blocks from here and moves them
    /// on (into the next round's prior blocks, or back into the message
    /// for the final response), so the message is left with none. A round
    /// copied them twice before (into the result, then again into the
    /// prior blocks); what is still copied is the conversion into the next
    /// request's input blocks and the tool calls `extract_tool_calls` reads
    /// out of the blocks (their id, name and input), which the executor
    /// copies once more into its inputs.
    pub(crate) fn take_from(message: &mut Message) -> Self {
        let content_blocks = std::mem::take(&mut message.content);
        let tool_use_blocks = extract_tool_calls(&content_blocks);
        Self {
            content_blocks,
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
            ContentBlock::ToolUse {
                id, name, input, ..
            } => Some(ToolUseBlock {
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
    plain_tool_pairs(&mut request.messages);

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

/// A replayed MCP exchange, the `mcp_tool_use` and `mcp_tool_result` blocks
/// a client sends back from an earlier answer, as this loop sent it upstream
/// the first time: the plain `tool_use` in the assistant turn and its
/// `tool_result` in the user turn right after, under the ids the answer
/// carries. The upstream runs no MCP of its own and reads the plain pair
/// alone, and it refuses a result inside an assistant turn, as the gateway's
/// own pairing rule does. A replayed assistant message is split at its
/// results: the calls before them stay the assistant's, the results open a
/// user turn, and a block of another kind after them opens the next assistant
/// turn (the public API combines consecutive turns of one role). A call right
/// after a result joins the turn's calls: a non-streamed answer writes
/// parallel calls as sequential pairs, and the first pass sent them as one
/// turn. A request without an MCP block is left as it came, without an
/// allocation.
fn plain_tool_pairs(messages: &mut Vec<InputMessage>) {
    if !messages.iter().any(carries_mcp_block) {
        return;
    }
    let mut plain = Vec::with_capacity(messages.len() + 2);
    for message in messages.drain(..) {
        match message.content {
            InputContent::Blocks(blocks) if blocks.iter().any(is_mcp_block) => {
                if matches!(message.role, Role::Assistant) {
                    split_assistant_turn(blocks, &mut plain);
                } else {
                    plain.push(InputMessage {
                        role: message.role,
                        content: InputContent::Blocks(
                            blocks.into_iter().map(plain_block).collect(),
                        ),
                    });
                }
            }
            content => plain.push(InputMessage {
                role: message.role,
                content,
            }),
        }
    }
    *messages = plain;
}

/// Whether a block is one of a replayed MCP exchange.
fn is_mcp_block(block: &InputContentBlock) -> bool {
    matches!(
        block,
        InputContentBlock::McpToolUse(_) | InputContentBlock::McpToolResult(_)
    )
}

/// Whether a message carries a block of a replayed MCP exchange.
fn carries_mcp_block(message: &InputMessage) -> bool {
    match &message.content {
        InputContent::Blocks(blocks) => blocks.iter().any(is_mcp_block),
        InputContent::String(_) => false,
    }
}

/// The plain block of a replayed MCP block; any other block as it came.
fn plain_block(block: InputContentBlock) -> InputContentBlock {
    match block {
        InputContentBlock::McpToolUse(mcp) => InputContentBlock::ToolUse(ToolUseBlock {
            id: mcp.id,
            name: mcp.name,
            input: mcp.input,
            cache_control: mcp.cache_control,
        }),
        InputContentBlock::McpToolResult(mcp) => InputContentBlock::ToolResult(ToolResultBlock {
            tool_use_id: mcp.tool_use_id,
            content: mcp.content,
            is_error: mcp.is_error,
            cache_control: mcp.cache_control,
        }),
        block => block,
    }
}

/// One replayed assistant turn, split so that every result sits in a user
/// turn right after the calls it answers, parallel calls kept in one turn.
fn split_assistant_turn(blocks: Vec<InputContentBlock>, out: &mut Vec<InputMessage>) {
    let turn = |role: Role, blocks: Vec<InputContentBlock>| InputMessage {
        role,
        content: InputContent::Blocks(blocks),
    };
    let mut assistant = Vec::new();
    let mut results = Vec::new();
    for block in blocks.into_iter().map(plain_block) {
        match block {
            InputContentBlock::ToolResult(_) => results.push(block),
            InputContentBlock::ToolUse(_) => assistant.push(block),
            _ => {
                if !results.is_empty() {
                    if !assistant.is_empty() {
                        out.push(turn(Role::Assistant, std::mem::take(&mut assistant)));
                    }
                    out.push(turn(Role::User, std::mem::take(&mut results)));
                }
                assistant.push(block);
            }
        }
    }
    if !assistant.is_empty() {
        out.push(turn(Role::Assistant, assistant));
    }
    if !results.is_empty() {
        out.push(turn(Role::User, results));
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

    let additional: Map<String, Value> = schema_map
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
        extra: Default::default(),
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
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
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

#[cfg(test)]
mod tests {
    use serde_json::json;
    use smg_mcp::Tool as McpTool;

    use super::*;
    use crate::test_support::counting_alloc;

    #[test]
    fn mcp_tool_schema_keeps_the_servers_key_order() {
        // More keys than a hash map would keep in order by chance.
        let names = [
            "zeta", "alpha", "mu", "beta", "omega", "delta", "kappa", "gamma",
        ];
        let properties: Map<String, Value> = names
            .iter()
            .map(|name| (name.to_string(), json!({"type": "string"})))
            .collect();
        let schema = json!({
            "type": "object",
            "properties": properties,
            "required": ["mu", "alpha"],
            "title": "Lookup",
            "additionalProperties": false,
            "$defs": {}
        });
        let tool = McpTool::new(
            "lookup".to_string(),
            "look a value up",
            schema.as_object().unwrap().clone(),
        );
        let entry = ToolEntry::from_server_tool("server", tool);
        let custom = convert_tool_entry_to_anthropic_tool(&entry, false);
        assert_eq!(
            serde_json::to_string(&custom.input_schema).unwrap(),
            serde_json::to_string(&schema).unwrap()
        );
    }

    /// Every message as one line, `role: text` or `role: [block, ...]`, a
    /// block named by its kind and id or text.
    fn turns(request: &CreateMessageRequest) -> Vec<String> {
        let label = |block: &InputContentBlock| match block {
            InputContentBlock::ToolUse(call) => format!("tool_use {}", call.id),
            InputContentBlock::ToolResult(result) => format!("tool_result {}", result.tool_use_id),
            InputContentBlock::Text(text) => format!("text {}", text.text),
            other => format!("{other:?}"),
        };
        request
            .messages
            .iter()
            .map(|message| {
                let role = format!("{:?}", message.role).to_lowercase();
                match &message.content {
                    InputContent::String(text) => format!("{role}: {text}"),
                    InputContent::Blocks(blocks) => {
                        let blocks: Vec<String> = blocks.iter().map(label).collect();
                        format!("{role}: [{}]", blocks.join(", "))
                    }
                }
            })
            .collect()
    }

    /// The content blocks of `messages[index]`.
    fn blocks_of(request: &CreateMessageRequest, index: usize) -> &[InputContentBlock] {
        match &request.messages[index].content {
            InputContent::Blocks(blocks) => blocks,
            InputContent::String(text) => panic!("messages.{index} is the text {text:?}"),
        }
    }

    /// A client replaying an earlier MCP answer sends its `mcp_tool_use` and
    /// `mcp_tool_result` blocks back; the upstream gets the plain pair this
    /// loop sent the first time, the call in the assistant turn and the
    /// result in a user turn right after it, and every other block as it came.
    #[test]
    fn a_replayed_mcp_exchange_goes_upstream_as_the_plain_pair() {
        let mut request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "look it up"},
                {"role": "assistant", "content": [
                    {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
                    {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}], "is_error": false},
                    {"type": "text", "text": "it is 42"}
                ]},
                {"role": "user", "content": "thanks"}
            ]
        }))
        .expect("a request");
        plain_tool_pairs(&mut request.messages);
        assert_eq!(
            turns(&request),
            [
                "user: look it up",
                "assistant: [tool_use mcptoolu_1]",
                "user: [tool_result mcptoolu_1]",
                "assistant: [text it is 42]",
                "user: thanks",
            ]
        );
        match blocks_of(&request, 1) {
            [InputContentBlock::ToolUse(call)] => {
                assert_eq!(call.name, "lookup");
                assert_eq!(call.input, json!({"q": "x"}));
            }
            other => panic!("the plain call alone, not {other:?}"),
        }
        match blocks_of(&request, 2) {
            [InputContentBlock::ToolResult(result)] => {
                assert_eq!(result.is_error, Some(false));
                assert_eq!(
                    serde_json::to_value(&result.content).unwrap(),
                    json!([{"type": "text", "text": "42"}])
                );
            }
            other => panic!("the plain result alone, not {other:?}"),
        }
    }

    /// Two calls in one replayed answer, each with its result and a text
    /// after it: a user turn per result, and the text between them opens the
    /// assistant's next turn with the second call.
    #[test]
    fn a_replayed_exchange_of_two_calls_opens_a_user_turn_per_result() {
        let mut request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "look both up"},
                {"role": "assistant", "content": [
                    {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
                    {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}], "is_error": false},
                    {"type": "text", "text": "x is 42"},
                    {"type": "mcp_tool_use", "id": "mcptoolu_2", "name": "lookup", "server_name": "s", "input": {"q": "y"}},
                    {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_2", "content": [{"type": "text", "text": "7"}], "is_error": false},
                    {"type": "text", "text": "y is 7"}
                ]}
            ]
        }))
        .expect("a request");
        plain_tool_pairs(&mut request.messages);
        assert_eq!(
            turns(&request),
            [
                "user: look both up",
                "assistant: [tool_use mcptoolu_1]",
                "user: [tool_result mcptoolu_1]",
                "assistant: [text x is 42, tool_use mcptoolu_2]",
                "user: [tool_result mcptoolu_2]",
                "assistant: [text y is 7]",
            ]
        );
    }

    /// A non-streamed answer writes two parallel calls as sequential pairs;
    /// the replay keeps them as one assistant turn with one user turn of
    /// results, the shape the loop sent the first time.
    #[test]
    fn a_replayed_answer_of_sequential_pairs_keeps_its_parallel_calls_in_one_turn() {
        let mut request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "look both up"},
                {"role": "assistant", "content": [
                    {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
                    {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}], "is_error": false},
                    {"type": "mcp_tool_use", "id": "mcptoolu_2", "name": "lookup", "server_name": "s", "input": {"q": "y"}},
                    {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_2", "content": [{"type": "text", "text": "7"}], "is_error": false},
                    {"type": "text", "text": "x is 42, y is 7"}
                ]}
            ]
        }))
        .expect("a request");
        plain_tool_pairs(&mut request.messages);
        assert_eq!(
            turns(&request),
            [
                "user: look both up",
                "assistant: [tool_use mcptoolu_1, tool_use mcptoolu_2]",
                "user: [tool_result mcptoolu_1, tool_result mcptoolu_2]",
                "assistant: [text x is 42, y is 7]",
            ]
        );
    }

    /// A replayed answer that ends with its result closes on the user turn
    /// that carries it, nothing after.
    #[test]
    fn a_replayed_answer_ending_with_its_result_closes_on_the_user_turn() {
        let mut request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "look it up"},
                {"role": "assistant", "content": [
                    {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "lookup", "server_name": "s", "input": {"q": "x"}},
                    {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": [{"type": "text", "text": "42"}], "is_error": false}
                ]}
            ]
        }))
        .expect("a request");
        plain_tool_pairs(&mut request.messages);
        assert_eq!(
            turns(&request),
            [
                "user: look it up",
                "assistant: [tool_use mcptoolu_1]",
                "user: [tool_result mcptoolu_1]",
            ]
        );
    }

    /// A request without an MCP block is left as it came, a plain tool pair
    /// included.
    #[test]
    fn a_request_without_mcp_blocks_is_left_as_it_came() {
        let messages = json!([
            {"role": "user", "content": "look it up"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}
            ]}
        ]);
        let mut request: CreateMessageRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 8, "messages": messages
        }))
        .expect("a request");
        let before: Vec<InputMessage> = serde_json::from_value(messages).expect("messages");
        plain_tool_pairs(&mut request.messages);
        assert_eq!(
            serde_json::to_value(&request.messages).unwrap(),
            serde_json::to_value(&before).unwrap()
        );
    }

    /// A round's message as the model answers it: text, tool calls and a
    /// thought, read from the wire shape.
    fn round_message(blocks: usize) -> Message {
        let content: Vec<Value> = (0..blocks)
            .map(|i| match i % 3 {
                0 => json!({"type": "text", "text": format!("paragraph {i} of the answer, long enough to live on the heap")}),
                1 => json!({"type": "tool_use", "id": format!("toolu_{i}"), "name": "lookup",
                    "input": {"query": format!("q{i}"), "limit": 10, "fields": ["a", "b", "c"]}}),
                _ => json!({"type": "thinking", "thinking": format!("step {i}"), "signature": "sig"}),
            })
            .collect();
        serde_json::from_value(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "content": content,
            "model": "m", "stop_reason": "tool_use", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .expect("a message")
    }

    /// What a round does with the message's content: the iteration result,
    /// the conversion into the next request's input blocks, the blocks kept
    /// for the final response. Taken and moved, the blocks are copied once
    /// (the conversion); the round that copied them into the result and again
    /// into the prior blocks allocated the two copies on top. The counts are
    /// printed for the Numbers table and the moved shape is pinned.
    #[test]
    #[expect(
        clippy::print_stderr,
        reason = "the count a pin reads is printed for the Numbers table"
    )]
    fn a_round_takes_the_message_content_and_copies_it_once() {
        let blocks = 6;
        let copied = round_message(blocks);
        let mut moved = round_message(blocks);
        let expected = serde_json::to_value(&copied.content).expect("blocks as JSON");

        // The shape before: cloned into the result, converted, cloned again.
        let before = counting_alloc::calls();
        let content_blocks = copied.content.clone();
        let tool_use_blocks = extract_tool_calls(&content_blocks);
        let assistant = build_assistant_content_blocks(&content_blocks);
        let mut prior: Vec<ContentBlock> = Vec::new();
        prior.extend(copied.content.clone());
        let copied_calls = counting_alloc::calls() - before;
        drop((content_blocks, tool_use_blocks, assistant, prior));

        // The shape now: taken, converted, moved.
        let before = counting_alloc::calls();
        let result = IterationResult::take_from(&mut moved);
        let assistant = build_assistant_content_blocks(&result.content_blocks);
        let mut prior: Vec<ContentBlock> = Vec::new();
        prior.extend(result.content_blocks);
        let moved_calls = counting_alloc::calls() - before;

        eprintln!(
            "a round's content handling, {blocks} blocks: copied shape {copied_calls} allocation calls, moved shape {moved_calls}"
        );
        assert!(moved.content.is_empty(), "the message's content is taken");
        assert_eq!(
            serde_json::to_value(&prior).expect("blocks as JSON"),
            expected
        );
        assert_eq!(assistant.len(), blocks);
        assert_eq!(result.tool_use_blocks.len(), blocks / 3);
        assert!(moved_calls < copied_calls, "{moved_calls} < {copied_calls}");
        // Pinned at today's count: the conversion of the blocks (about 32
        // calls for these six), the tool-call extraction's copies (about
        // 25) and the prior vector's buffer (1).
        assert_eq!(
            moved_calls, PINNED_ROUND_CALLS,
            "the moved shape's allocation calls changed"
        );
    }

    /// Allocation calls of the moved shape for six blocks, today.
    const PINNED_ROUND_CALLS: usize = 58;
}
