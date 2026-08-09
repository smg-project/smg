// Ported from https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/encoding_dsv4.py

use std::fmt::Write as _;

use serde_json::{json, Value};
use thiserror::Error;

// Reuse the public ThinkingMode enum from the V3.2 module to keep the
// "thinking" / "chat" mode invariant identical across DeepSeek versions.
pub use super::deepseek_v32::ThinkingMode;

/// Reasoning effort for the V4 prompt prefix.
///
/// Mirrors the Python `reasoning_effort` parameter, which only accepts
/// `None`, `"high"`, or `"max"`. Only `Max` actually emits a prefix today;
/// `High` is accepted for parity with the Python signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    High,
    Max,
}

/// Parameters for [`encode_messages`].
///
/// `context` is intentionally omitted: SMG always renders from scratch, so
/// the Python default of `context=None` always applies.
#[derive(Debug, Clone, Copy)]
pub struct EncodeParams {
    pub add_default_bos_token: bool,
    pub drop_thinking: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
}
impl Default for EncodeParams {
    fn default() -> Self {
        Self {
            add_default_bos_token: true,
            drop_thinking: true,
            reasoning_effort: None,
        }
    }
}

/// Errors raised when a message list is malformed.
///
/// V4-local error type. The variants overlap with V3.2 but are kept
/// independent so each encoder file is a standalone translation of its
/// Python source.
#[derive(Debug, Error)]
pub enum DsEncodingError {
    #[error("Index {index} out of range for messages list of length {len}")]
    IndexOutOfRange { index: usize, len: usize },
    #[error("Invalid message for role `{role}`: {msg}")]
    InvalidMessage { role: String, msg: String },
    #[error("Unknown role: {0}")]
    UnknownRole(String),
    #[error("DeepSeek V4 merges tool messages into user; preprocess via merge_tool_messages first (got tool message at index {0})")]
    UnmergedToolRole(usize),
    #[error(
        "Invalid task `{0}`. Valid tasks are: action, query, authority, domain, title, read_url"
    )]
    InvalidTask(String),
}

// ---------------------------------------------------------------------------
// Special-token constants — copied verbatim from the Python source.
// ---------------------------------------------------------------------------
pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub const THINKING_START_TOKEN: &str = "<think>";
pub const THINKING_END_TOKEN: &str = "</think>";
pub const DSML_TOKEN: &str = "｜DSML｜";
const USER_SP_TOKEN: &str = "<｜User｜>";
const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
const LATEST_REMINDER_SP_TOKEN: &str = "<｜latest_reminder｜>";
const TOOL_CALLS_BLOCK_NAME: &str = "tool_calls";
// Quick-instruction "task" tokens (`<｜action｜>`, `<｜query｜>`, etc.)
const TASK_ACTION: &str = "<｜action｜>";
const TASK_QUERY: &str = "<｜query｜>";
const TASK_AUTHORITY: &str = "<｜authority｜>";
const TASK_DOMAIN: &str = "<｜domain｜>";
const TASK_TITLE: &str = "<｜title｜>";
const TASK_READ_URL: &str = "<｜read_url｜>";
fn task_sp_token(task: &str) -> Option<&'static str> {
    match task {
        "action" => Some(TASK_ACTION),
        "query" => Some(TASK_QUERY),
        "authority" => Some(TASK_AUTHORITY),
        "domain" => Some(TASK_DOMAIN),
        "title" => Some(TASK_TITLE),
        "read_url" => Some(TASK_READ_URL),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------
const REASONING_EFFORT_MAX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/// Mirrors V4's `TOOLS_TEMPLATE`. The block name is `tool_calls` (not
/// `function_calls` like V3.2) and the wording is updated.
fn render_tools_template(tool_schemas: &str) -> String {
    let dsml = DSML_TOKEN;
    let tcb = TOOL_CALLS_BLOCK_NAME;
    let tstart = THINKING_START_TOKEN;
    let tend = THINKING_END_TOKEN;
    format!(
"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{dsml}{tcb}>\" block like the following:

<{dsml}{tcb}>
<{dsml}invoke name=\"$TOOL_NAME\">
<{dsml}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name=\"$TOOL_NAME2\">
...
</{dsml}invoke>
</{dsml}{tcb}>

String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.

If thinking_mode is enabled (triggered by {tstart}), you MUST output your complete reasoning inside {tstart}...{tend} BEFORE any tool calls or final response.

Otherwise, output directly after {tend} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"
    )
}

// ---------------------------------------------------------------------------
// JSON helpers (mirror V3.2)
// ---------------------------------------------------------------------------
// Python's `to_json` is `json.dumps(value, ensure_ascii=False)`: spaced
// separators, raw UTF-8. Compact `serde_json::to_string` would change the
// prompt bytes vLLM trained on.
fn to_json(value: &Value) -> String {
    crate::json_dumps::to_string(value)
}
fn tools_from_openai_format(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|t| t.get("function").cloned())
        .collect()
}
fn tool_calls_from_openai_format(tool_calls: &[Value]) -> Vec<Value> {
    tool_calls
        .iter()
        .filter_map(|tc| {
            let f = tc.get("function")?;
            Some(json!({
                "name": f.get("name").cloned().unwrap_or(Value::Null),
                "arguments": f.get("arguments").cloned().unwrap_or(Value::Null),
            }))
        })
        .collect()
}

/// V4 differs from V3.2: when `arguments` fails to JSON-parse, the upstream
/// wraps the raw string in `{"arguments": <raw>}` instead of erroring.
///
/// `arguments` may arrive as a JSON *string* (raw OpenAI tool_calls) or as an
/// already-parsed object — `model_gateway`'s `process_tool_call_arguments`
/// converts the string into a dict before the chat template runs. Reading only
/// `as_str()` silently dropped object-form args, which erased every historical
/// tool call's parameters in multi-turn prompts.
fn encode_arguments_to_dsml(tool_call: &Value) -> String {
    let arguments: Value = match tool_call.get("arguments") {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| json!({ "arguments": s }))
        }
        Some(v) if v.is_object() => v.clone(),
        _ => json!({}),
    };
    let obj = match arguments.as_object() {
        Some(obj) => obj,
        None => return String::new(),
    };
    let mut parts = Vec::with_capacity(obj.len());
    for (k, v) in obj {
        let (is_str, value_str) = match v {
            Value::String(s) => ("true", s.clone()),
            other => ("false", to_json(other)),
        };
        parts.push(format!(
            "<{DSML_TOKEN}parameter name=\"{k}\" string=\"{is_str}\">{value_str}</{DSML_TOKEN}parameter>",
        ));
    }
    parts.join("\n")
}

fn render_tools(tools: &[Value]) -> String {
    let schemas: Vec<String> = tools.iter().map(to_json).collect();
    render_tools_template(&schemas.join("\n"))
}
fn find_last_user_index(messages: &[Value]) -> Option<usize> {
    for idx in (0..messages.len()).rev() {
        let role = messages[idx].get("role").and_then(|v| v.as_str());
        if matches!(role, Some("user") | Some("developer")) {
            return Some(idx);
        }
    }
    None
}
fn at_or_after_last_user(index: usize, last_user_idx: Option<usize>) -> bool {
    match last_user_idx {
        Some(idx) => index >= idx,
        None => true,
    }
}
fn after_last_user(index: usize, last_user_idx: Option<usize>) -> bool {
    match last_user_idx {
        Some(idx) => index > idx,
        None => true,
    }
}

// ---------------------------------------------------------------------------
// render_message — direct port of the V4 Python function with the same name.
// ---------------------------------------------------------------------------
#[expect(
    clippy::too_many_lines,
    reason = "mirrors the Python render_message function 1:1 for sync-ability"
)]
fn render_message(
    index: usize,
    messages: &[Value],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    reasoning_effort: Option<ReasoningEffort>,
) -> Result<String, DsEncodingError> {
    if index >= messages.len() {
        return Err(DsEncodingError::IndexOutOfRange {
            index,
            len: messages.len(),
        });
    }
    let mut prompt = String::new();
    let msg = &messages[index];
    let last_user_idx = find_last_user_index(messages);

    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
    let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let tools_raw = msg.get("tools").and_then(|v| v.as_array());
    let response_format = msg.get("response_format");
    let tool_calls_raw = msg.get("tool_calls").and_then(|v| v.as_array());
    let reasoning_content = msg
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let wo_eos = msg.get("wo_eos").and_then(|v| v.as_bool()).unwrap_or(false);
    let tools_owned = tools_raw.map(|t| tools_from_openai_format(t));
    let tools = tools_owned.as_deref();
    let tool_calls_owned = tool_calls_raw.map(|tc| tool_calls_from_openai_format(tc));
    let tool_calls = tool_calls_owned.as_deref();

    // Reasoning effort prefix (only at index 0 in thinking mode with max effort)
    if index == 0
        && thinking_mode == ThinkingMode::Thinking
        && reasoning_effort == Some(ReasoningEffort::Max)
    {
        prompt.push_str(REASONING_EFFORT_MAX);
    }

    match role {
        "system" => {
            prompt.push_str(content);
            if let Some(tools) = tools.filter(|t| !t.is_empty()) {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(tools));
            }
            if let Some(rf) = response_format {
                prompt.push_str("\n\n");
                prompt.push_str(&format!(
                    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{}",
                    to_json(rf)
                ));
            }
        }

        "developer" => {
            if content.is_empty() {
                return Err(DsEncodingError::InvalidMessage {
                    role: role.to_string(),
                    msg: msg.to_string(),
                });
            }
            let mut content_developer = String::new();
            content_developer.push_str(USER_SP_TOKEN);
            content_developer.push_str(content);
            if let Some(tools) = tools.filter(|t| !t.is_empty()) {
                content_developer.push_str("\n\n");
                content_developer.push_str(&render_tools(tools));
            }
            if let Some(rf) = response_format {
                content_developer.push_str("\n\n");
                let _ = write!(
                    content_developer,
                    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{}",
                    to_json(rf)
                );
            }
            prompt.push_str(&content_developer);
        }

        "user" => {
            prompt.push_str(USER_SP_TOKEN);
            // Handle content blocks (tool results mixed with text)
            if let Some(content_blocks) = msg.get("content_blocks").and_then(|v| v.as_array()) {
                let mut parts: Vec<String> = Vec::with_capacity(content_blocks.len());
                for block in content_blocks {
                    let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match block_type {
                        "text" => {
                            let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                            parts.push(text.to_string());
                        }
                        "tool_result" => {
                            let tc = block.get("content");
                            let tool_content = match tc {
                                Some(Value::Array(items)) => {
                                    let mut text_parts: Vec<String> =
                                        Vec::with_capacity(items.len());
                                    for b in items {
                                        let bt =
                                            b.get("type").and_then(|v| v.as_str()).unwrap_or("");
                                        if bt == "text" {
                                            text_parts.push(
                                                b.get("text")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("")
                                                    .to_string(),
                                            );
                                        } else {
                                            text_parts.push(format!("[Unsupported {bt}]"));
                                        }
                                    }
                                    text_parts.join("\n\n")
                                }
                                Some(Value::String(s)) => s.clone(),
                                Some(other) => to_json(other),
                                None => String::new(),
                            };
                            parts.push(format!("<tool_result>{tool_content}</tool_result>"));
                        }
                        other => parts.push(format!("[Unsupported {other}]")),
                    }
                }
                prompt.push_str(&parts.join("\n\n"));
            } else {
                prompt.push_str(content);
            }
        }

        "latest_reminder" => {
            prompt.push_str(LATEST_REMINDER_SP_TOKEN);
            prompt.push_str(content);
        }
        "tool" => {
            return Err(DsEncodingError::UnmergedToolRole(index));
        }

        "assistant" => {
            let mut thinking_part = String::new();
            let mut tc_content = String::new();
            if let Some(tcs) = tool_calls.filter(|t| !t.is_empty()) {
                let mut tc_list = Vec::with_capacity(tcs.len());
                for tc in tcs {
                    let name = tc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args = encode_arguments_to_dsml(tc);
                    tc_list.push(format!(
                        "<{DSML_TOKEN}invoke name=\"{name}\">\n{args}\n</{DSML_TOKEN}invoke>"
                    ));
                }
                let joined = tc_list.join("\n");
                let _ = write!(
                    tc_content,
                    "\n\n<{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>\n{joined}\n</{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>"
                );
            }
            // prev_has_task: if previous message had a task, this is a task
            // output (no thinking).
            let prev_has_task = if index >= 1 {
                messages[index - 1].get("task").is_some()
                    && !messages[index - 1]
                        .get("task")
                        .map(Value::is_null)
                        .unwrap_or(true)
            } else {
                false
            };
            if thinking_mode == ThinkingMode::Thinking && !prev_has_task {
                let emit = !drop_thinking || after_last_user(index, last_user_idx);
                if emit {
                    thinking_part.push_str(reasoning_content);
                    thinking_part.push_str(THINKING_END_TOKEN);
                }
            }
            prompt.push_str(&thinking_part);
            prompt.push_str(content);
            prompt.push_str(&tc_content);
            if !wo_eos {
                prompt.push_str(EOS_TOKEN);
            }
        }
        other => return Err(DsEncodingError::UnknownRole(other.to_string())),
    }

    // Append transition tokens based on what follows.
    if let Some(next) = messages.get(index + 1) {
        let next_role = next.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if !matches!(next_role, "assistant" | "latest_reminder") {
            return Ok(prompt);
        }
    }

    let task = messages[index]
        .get("task")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    if let Some(task) = task {
        let sp_token =
            task_sp_token(task).ok_or_else(|| DsEncodingError::InvalidTask(task.to_string()))?;
        if task == "action" {
            // Action task: append Assistant + thinking token + action sp token.
            prompt.push_str(ASSISTANT_SP_TOKEN);
            prompt.push_str(if thinking_mode == ThinkingMode::Thinking {
                THINKING_START_TOKEN
            } else {
                THINKING_END_TOKEN
            });
            prompt.push_str(sp_token);
        } else {
            // Non-action tasks: append task sp token directly after the message.
            prompt.push_str(sp_token);
        }
    } else if matches!(role, "user" | "developer") {
        // Normal generation: append Assistant + thinking token.
        prompt.push_str(ASSISTANT_SP_TOKEN);
        let opens_thinking = thinking_mode == ThinkingMode::Thinking
            && (!drop_thinking || at_or_after_last_user(index, last_user_idx));
        if opens_thinking {
            prompt.push_str(THINKING_START_TOKEN);
        } else {
            prompt.push_str(THINKING_END_TOKEN);
        }
    }
    Ok(prompt)
}

// ---------------------------------------------------------------------------
// Preprocessing: merge tool messages and sort tool results.
// ---------------------------------------------------------------------------
fn merge_tool_messages(messages: &[Value]) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());
    for msg in messages {
        let msg = msg.clone();
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role == "tool" {
            let tool_block = json!({
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id").cloned().unwrap_or(Value::String(String::new())),
                "content": msg.get("content").cloned().unwrap_or(Value::String(String::new())),
            });
            // Append to a previous user message that already has content_blocks.
            let appended = if let Some(prev) = merged.last_mut() {
                let prev_role = prev.get("role").and_then(|v| v.as_str()).unwrap_or("");
                if prev_role == "user" && prev.get("content_blocks").is_some() {
                    if let Some(blocks) = prev
                        .get_mut("content_blocks")
                        .and_then(|v| v.as_array_mut())
                    {
                        blocks.push(tool_block.clone());
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            if !appended {
                merged.push(json!({
                    "role": "user",
                    "content_blocks": [tool_block],
                }));
            }
        } else if role == "user" {
            let text_block = json!({
                "type": "text",
                "text": msg.get("content").cloned().unwrap_or(Value::String(String::new())),
            });
            let merged_into_prev = if let Some(prev) = merged.last_mut() {
                let prev_role = prev.get("role").and_then(|v| v.as_str()).unwrap_or("");
                let prev_has_blocks = prev.get("content_blocks").is_some();
                let prev_task_none = prev.get("task").map(Value::is_null).unwrap_or(true);
                if prev_role == "user" && prev_has_blocks && prev_task_none {
                    if let Some(blocks) = prev
                        .get_mut("content_blocks")
                        .and_then(|v| v.as_array_mut())
                    {
                        blocks.push(text_block.clone());
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            if !merged_into_prev {
                let mut new_msg = json!({
                    "role": "user",
                    "content": msg.get("content").cloned().unwrap_or(Value::String(String::new())),
                    "content_blocks": [text_block],
                });
                // Preserve extra fields (task, wo_eos, mask, etc.).
                if let Some(obj) = new_msg.as_object_mut() {
                    for key in ["task", "wo_eos", "mask"] {
                        if let Some(v) = msg.get(key) {
                            obj.insert(key.to_string(), v.clone());
                        }
                    }
                }
                merged.push(new_msg);
            }
        } else {
            merged.push(msg);
        }
    }
    merged
}

/// Sort `tool_result` blocks within user messages by the tool-call order
/// of the *preceding* assistant turn.
fn sort_tool_results_by_call_order(messages: Vec<Value>) -> Vec<Value> {
    let mut out = messages;
    let mut last_tool_call_order: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for msg in &mut out {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if role == "assistant" {
            if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                last_tool_call_order.clear();
                for (idx, tc) in tcs.iter().enumerate() {
                    let tc_id = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .or_else(|| {
                            tc.get("function")
                                .and_then(|f| f.get("id"))
                                .and_then(|v| v.as_str())
                                .map(str::to_string)
                        });
                    if let Some(id) = tc_id {
                        last_tool_call_order.insert(id, idx);
                    }
                }
            }
        } else if role == "user" {
            if let Some(blocks) = msg.get("content_blocks").and_then(|v| v.as_array()) {
                let tool_blocks: Vec<&Value> = blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(|v| v.as_str()) == Some("tool_result"))
                    .collect();
                if tool_blocks.len() > 1 && !last_tool_call_order.is_empty() {
                    let mut sorted: Vec<Value> = tool_blocks.iter().map(|b| (*b).clone()).collect();
                    sorted.sort_by_key(|b| {
                        b.get("tool_use_id")
                            .and_then(|v| v.as_str())
                            .and_then(|id| last_tool_call_order.get(id).copied())
                            .unwrap_or(0)
                    });
                    let mut sorted_idx = 0;
                    let mut new_blocks: Vec<Value> = Vec::with_capacity(blocks.len());
                    for block in blocks {
                        if block.get("type").and_then(|v| v.as_str()) == Some("tool_result") {
                            new_blocks.push(sorted[sorted_idx].clone());
                            sorted_idx += 1;
                        } else {
                            new_blocks.push(block.clone());
                        }
                    }
                    if let Some(obj) = msg.as_object_mut() {
                        obj.insert("content_blocks".to_string(), Value::Array(new_blocks));
                    }
                }
            }
        }
    }
    out
}

/// Drop reasoning_content from earlier assistant turns and remove non-essential
/// developer messages before the last user.
fn drop_thinking_messages(messages: &[Value]) -> Vec<Value> {
    let last_user_idx = find_last_user_index(messages);
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for (idx, msg) in messages.iter().enumerate() {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let always_keep = matches!(
            role,
            "user" | "system" | "tool" | "latest_reminder" | "direct_search_results"
        ) || at_or_after_last_user(idx, last_user_idx);
        if always_keep {
            out.push(msg.clone());
            continue;
        }
        if role == "assistant" {
            let mut cloned = msg.clone();
            if let Some(obj) = cloned.as_object_mut() {
                obj.remove("reasoning_content");
            }
            out.push(cloned);
        }
        // developer + other roles before last_user_idx are dropped.
    }
    out
}

// ---------------------------------------------------------------------------
// encode_messages — public entry point
// ---------------------------------------------------------------------------
/// Encode a list of OpenAI-style messages into a DeepSeek V4 prompt string.
///
/// The signature mirrors the Python `encode_messages` function;
/// `context` is omitted because SMG always renders from scratch.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "public API mirrors the documented Rust signature with a borrow"
)]
pub fn encode_messages(
    messages: &[Value],
    thinking_mode: ThinkingMode,
    params: &EncodeParams,
) -> Result<String, DsEncodingError> {
    // Preprocess: merge tool messages and sort tool results.
    let merged = merge_tool_messages(messages);
    let mut full_messages = sort_tool_results_by_call_order(merged);
    let mut prompt = if params.add_default_bos_token {
        BOS_TOKEN.to_string()
    } else {
        String::new()
    };
    // Resolve drop_thinking: if any message has tools defined, never drop.
    let mut effective_drop_thinking = params.drop_thinking;
    if full_messages
        .iter()
        .any(|m| m.get("tools").is_some_and(|v| !v.is_null()))
    {
        effective_drop_thinking = false;
    }
    if thinking_mode == ThinkingMode::Thinking && effective_drop_thinking {
        full_messages = drop_thinking_messages(&full_messages);
    }
    for idx in 0..full_messages.len() {
        prompt.push_str(&render_message(
            idx,
            &full_messages,
            thinking_mode,
            effective_drop_thinking,
            params.reasoning_effort,
        )?);
    }
    Ok(prompt)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    fn user(text: &str) -> Value {
        json!({ "role": "user", "content": text })
    }
    #[test]
    fn one_turn_user_chat_mode() {
        let msgs = [user("Hello")];
        let out = encode_messages(&msgs, ThinkingMode::Chat, &EncodeParams::default()).unwrap();
        let expected =
            format!("{BOS_TOKEN}{USER_SP_TOKEN}Hello{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}");
        assert_eq!(out, expected);
    }
    #[test]
    fn one_turn_user_thinking_mode() {
        let msgs = [user("Hello")];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();
        let expected =
            format!("{BOS_TOKEN}{USER_SP_TOKEN}Hello{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}");
        assert_eq!(out, expected);
    }

    #[test]
    fn reasoning_effort_max_prepends_prefix() {
        let msgs = [user("Hello")];
        let params = EncodeParams {
            reasoning_effort: Some(ReasoningEffort::Max),
            ..EncodeParams::default()
        };
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &params).unwrap();
        // The prefix appears immediately after BOS, before the user message.
        let expected_start = format!("{BOS_TOKEN}{REASONING_EFFORT_MAX}");
        assert!(
            out.starts_with(&expected_start),
            "expected prompt to start with BOS+REASONING_EFFORT_MAX, got: {:?}",
            &out[..120.min(out.len())]
        );
        // Without max effort, the prefix is absent.
        let out_chat = encode_messages(&msgs, ThinkingMode::Chat, &params).unwrap();
        assert!(!out_chat.contains("Reasoning Effort"));
    }

    #[test]
    fn quick_instruction_action_token() {
        // A user message tagged with `task: "action"` triggers the action
        // quick-instruction sequence: ASSISTANT_SP + thinking-end + ACTION token.
        let msgs = [json!({
            "role": "user",
            "content": "Take some action",
            "task": "action",
        })];
        let out = encode_messages(&msgs, ThinkingMode::Chat, &EncodeParams::default()).unwrap();
        let expected = format!(
            "{BOS_TOKEN}{USER_SP_TOKEN}Take some action{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}{TASK_ACTION}"
        );
        assert_eq!(out, expected);
        // Same in thinking mode but uses thinking-start.
        let out_t =
            encode_messages(&msgs, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();
        assert!(out_t.contains(&format!(
            "{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}{TASK_ACTION}"
        )));
    }
    #[test]
    fn quick_instruction_query_token() {
        // Non-action quick-instruction tasks just append the task token.
        let msgs = [json!({
            "role": "user",
            "content": "What is X?",
            "task": "query",
        })];
        let out = encode_messages(&msgs, ThinkingMode::Chat, &EncodeParams::default()).unwrap();
        let expected = format!("{BOS_TOKEN}{USER_SP_TOKEN}What is X?{TASK_QUERY}");
        assert_eq!(out, expected);
    }

    #[test]
    fn assistant_tool_call_renders_dsml() {
        let msgs = [
            user("call my tool"),
            json!({
                "role": "assistant",
                "reasoning_content": "thinking about tool",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{\"query\": \"deepseek\", \"limit\": 5}"
                        }
                    }
                ]
            }),
        ];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();
        // V4 wraps in `<｜DSML｜tool_calls>` (not `function_calls` like V3.2).
        assert!(out.contains(&format!("<{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>")));
        assert!(out.contains(&format!("<{DSML_TOKEN}invoke name=\"search\">")));
        assert!(out.contains(&format!(
            "<{DSML_TOKEN}parameter name=\"query\" string=\"true\">deepseek</{DSML_TOKEN}parameter>"
        )));
        assert!(out.contains(&format!(
            "<{DSML_TOKEN}parameter name=\"limit\" string=\"false\">5</{DSML_TOKEN}parameter>"
        )));
        assert!(out.contains(&format!("</{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>")));
        assert!(out.ends_with(EOS_TOKEN));
    }
    #[test]
    fn object_form_tool_call_arguments_render_params() {
        // `model_gateway`'s `process_tool_call_arguments` converts the OpenAI
        // arguments *string* into a dict before the chat template runs, so the
        // encoder receives object-form arguments. It must render them, not drop
        // them — dropping erased every historical tool call's parameters in
        // multi-turn prompts and made the model loop (re-calling tools forever).
        let msgs = [
            user("call my tool"),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": { "query": "deepseek", "limit": 5 }
                        }
                    }
                ]
            }),
        ];
        let out = encode_messages(&msgs, ThinkingMode::Chat, &EncodeParams::default()).unwrap();
        assert!(out.contains(&format!(
            "<{DSML_TOKEN}parameter name=\"query\" string=\"true\">deepseek</{DSML_TOKEN}parameter>"
        )));
        assert!(out.contains(&format!(
            "<{DSML_TOKEN}parameter name=\"limit\" string=\"false\">5</{DSML_TOKEN}parameter>"
        )));
    }
    #[test]
    fn unknown_role_errors() {
        let msgs = [json!({ "role": "moderator", "content": "hi" })];
        let err = encode_messages(&msgs, ThinkingMode::Chat, &EncodeParams::default()).unwrap_err();
        assert!(matches!(err, DsEncodingError::UnknownRole(ref r) if r == "moderator"));
    }
}
