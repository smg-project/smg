// Ported from vLLM's `vllm/tokenizers/deepseek_v41_encoding.py` (itself a port
// of the `encoding/encoding.py` shipped with deepseek-ai/DeepSeek-V4.1-Flash),
// function by function and in the same order, plus the message normalisation
// that vLLM keeps in `vllm/tokenizers/deepseek_v41.py::_normalize_messages`.
//
// V4.1 differs from V4 (`deepseek_v4.rs`) in four places: the DSML tag strings
// carry a leading space (` invoke`, ` parameter`, ` calls`), the reasoning
// effort is a number in a `<｜System｜>`-led prefix instead of a prose block,
// mid-conversation system messages count as user turns, and image content
// parts collapse to [`IMAGE_PLACEHOLDER`].

use std::fmt::Write as _;

use serde_json::{json, Value};

// `DsEncodingError` lives in `deepseek_common` (shared with V4); re-exported
// here because it is part of this module's public API, and `pub(super)` items
// are not reachable from outside `encoders` on their own.
pub use super::deepseek_common::DsEncodingError;
// Message-shape preprocessing, drop-thinking and DSML argument rendering are
// identical to V4 except for the tag strings and the last-user definition; see
// `deepseek_common` for the shared implementation and its own doc comments.
use super::deepseek_common::{
    at_or_after_last_user, drop_thinking_messages, encode_arguments_to_dsml, merge_tool_messages,
    sort_tool_results_by_call_order, DsmlTags,
};
// Reuse the public ThinkingMode enum from the V3.2 module to keep the
// "thinking" / "chat" mode invariant identical across DeepSeek versions.
pub use super::deepseek_v32::ThinkingMode;

// ---------------------------------------------------------------------------
// Special-token constants — copied verbatim from the Python source.
// ---------------------------------------------------------------------------
pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub const THINKING_START_TOKEN: &str = "<think>";
pub const THINKING_END_TOKEN: &str = "</think>";
pub const DSML_TOKEN: &str = "｜DSML｜";
/// What every image content part renders as; the image payload itself is the
/// serving layer's business, the prompt only carries this marker.
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";
const USER_SP_TOKEN: &str = "<｜User｜>";
const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
const SYSTEM_SP_TOKEN: &str = "<｜System｜>";
const LATEST_REMINDER_SP_TOKEN: &str = "<｜latest_reminder｜>";
/// V4.1's DSML dialect: a leading space on the block name and on both tags
/// (V4 uses bare `tool_calls`/`invoke`/`parameter`).
const TOOL_CALLS_BLOCK_NAME: &str = " calls";
const DSML_TAGS: DsmlTags = DsmlTags {
    invoke: " invoke",
    parameter: " parameter",
};
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
// Reasoning effort
// ---------------------------------------------------------------------------
/// The effort names the Python encoder accepts (`REASONING_EFFORT_MAPPINGS`).
pub const NATIVE_EFFORT_VALUES: &[&str] = &["low", "high", "xhigh", "max"];
pub const REASONING_EFFORT_TEMPLATE: &str =
    "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
const DEFAULT_REASONING_EFFORT: ReasoningEffort = ReasoningEffort::High;

/// Reasoning effort for the V4.1 numeric prompt prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    High,
    XHigh,
    Max,
    /// A raw budget, as accepted by the API in `1..=100`.
    Budget(u8),
}

impl ReasoningEffort {
    /// Spec D1: engine table. Flip these four numbers if vLLM/SGLang adopt
    /// DeepSeek's 50/75/100.
    pub const fn budget(self) -> u8 {
        match self {
            Self::Low => 25,
            Self::High => 50,
            Self::XHigh => 75,
            Self::Max => 100,
            Self::Budget(n) => n,
        }
    }

    /// Parse one of [`NATIVE_EFFORT_VALUES`].
    fn from_native(value: &str) -> Option<Self> {
        match value {
            "low" => Some(Self::Low),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::XHigh),
            "max" => Some(Self::Max),
            _ => None,
        }
    }
}

/// Parse the request's `reasoning_effort` field.
///
/// `None` = not present. Strings low/high/xhigh/max, integers 1..=100. The
/// thinking switch (`"none"`/`"minimal"`, see
/// `openai_protocol::chat::thinking_from_reasoning_effort`) is handled by the
/// caller — those mean "thinking off", not an effort level — so they land here
/// only by mistake. Anything else — floats, booleans, unknown names,
/// out-of-range integers — is [`DsEncodingError::InvalidReasoningEffort`].
pub fn parse_reasoning_effort(value: &Value) -> Result<Option<ReasoningEffort>, DsEncodingError> {
    match value {
        Value::Null => Ok(None),
        Value::String(name) => ReasoningEffort::from_native(name)
            .map(Some)
            .ok_or_else(|| DsEncodingError::InvalidReasoningEffort(name.clone())),
        // Python's `type(effort) is int` rejects floats and booleans alike.
        Value::Number(number) => number
            .as_u64()
            .and_then(|budget| u8::try_from(budget).ok())
            .filter(|budget| (1..=100).contains(budget))
            .map(|budget| Some(ReasoningEffort::Budget(budget)))
            .ok_or_else(|| DsEncodingError::InvalidReasoningEffort(number.to_string())),
        other => Err(DsEncodingError::InvalidReasoningEffort(other.to_string())),
    }
}

/// Mirrors `render_reasoning_effort`: the prefix exists in thinking mode, at
/// index 0, only.
fn render_reasoning_effort(
    index: usize,
    thinking_mode: ThinkingMode,
    effort: Option<ReasoningEffort>,
) -> String {
    if index != 0 || thinking_mode != ThinkingMode::Thinking {
        return String::new();
    }
    let budget = effort.unwrap_or(DEFAULT_REASONING_EFFORT).budget();
    REASONING_EFFORT_TEMPLATE.replace("{budget}", &budget.to_string())
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------
/// Mirrors V4.1's `TOOLS_TEMPLATE`.
fn render_tools_template(tool_schemas: &str) -> String {
    let dsml = DSML_TOKEN;
    let tcb = TOOL_CALLS_BLOCK_NAME;
    let invoke = DSML_TAGS.invoke;
    let parameter = DSML_TAGS.parameter;
    let tstart = THINKING_START_TOKEN;
    let tend = THINKING_END_TOKEN;
    format!(
"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{dsml}{tcb}>\" block like the following:

<{dsml}{tcb}>
<{dsml}{invoke} name=\"$TOOL_NAME\">
<{dsml}{parameter} name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{dsml}{parameter}>
...
</{dsml}{invoke}>
<{dsml}{invoke} name=\"$TOOL_NAME2\">
...
</{dsml}{invoke}>
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

fn render_response_format(schema: &Value) -> String {
    format!(
        "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{}",
        to_json(schema)
    )
}

// ---------------------------------------------------------------------------
// JSON helpers (mirror V4)
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
        .filter_map(|tool| tool.get("function").cloned())
        .collect()
}
fn tool_calls_from_openai_format(tool_calls: &[Value]) -> Vec<Value> {
    tool_calls
        .iter()
        .filter_map(|tool_call| {
            let function = tool_call.get("function")?;
            Some(json!({
                "name": function.get("name").cloned().unwrap_or(Value::Null),
                "arguments": function.get("arguments").cloned().unwrap_or(Value::Null),
            }))
        })
        .collect()
}
fn render_tools(tools: &[Value]) -> String {
    let schemas: Vec<String> = tools.iter().map(to_json).collect();
    render_tools_template(&schemas.join("\n"))
}
/// Python's `if msg.get("tools"):` — absent, null and `[]` are all falsy.
fn has_tools(msg: &Value) -> bool {
    msg.get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|tools| !tools.is_empty())
}

// ---------------------------------------------------------------------------
// Image/content-part normalisation
// ---------------------------------------------------------------------------
// vLLM's `_normalize_messages` plus the reference encoder's
// `process_image_messages`: list content collapses to a single string in which
// every image part became `IMAGE_PLACEHOLDER`, parts joined by a blank line.
// The image payloads themselves are not collected here — SMG's multimodal path
// carries them separately, the prompt only needs the marker.

/// Mirrors vLLM's `_normalize_messages` flattening plus the HF
/// `_validate_no_image_sp_tokens` check (see the header comment above): list
/// content collapses to the text V4.1 encodes, and no text may carry the image
/// placeholder.
fn process_image_messages(messages: &[Value]) -> Result<Vec<Value>, DsEncodingError> {
    let mut processed: Vec<Value> = Vec::with_capacity(messages.len());
    for msg in messages {
        let mut msg = msg.clone();
        validate_no_image_sp_tokens(&msg)?;
        let flattened = match msg.get("content") {
            Some(Value::Array(parts)) => Some(flatten_content_parts(parts)?),
            _ => None,
        };
        if let Some(text) = flattened {
            if let Some(obj) = msg.as_object_mut() {
                obj.insert("content".to_string(), Value::String(text));
            }
        }
        processed.push(msg);
    }
    Ok(processed)
}

/// Mirrors `_validate_no_image_sp_tokens`: user text may not smuggle in the
/// image special token. List content is checked part by part below.
fn validate_no_image_sp_tokens(msg: &Value) -> Result<(), DsEncodingError> {
    for field in ["content", "reasoning_content"] {
        let text = msg.get(field).and_then(|v| v.as_str()).unwrap_or("");
        if text.contains(IMAGE_PLACEHOLDER) {
            return Err(DsEncodingError::PlaceholderInText);
        }
    }
    Ok(())
}

/// Join one message's content parts into the text V4.1 encodes.
///
/// Accepts the OpenAI (`text`/`image_url`), Responses (`input_text`,
/// `output_text`, `input_image`) and Anthropic-style (`image`) spellings.
/// Image parts need no URL: SMG's `chat_utils::transform_content_field` strips
/// the payload before the renderer runs, leaving a bare `{"type": "image"}`.
fn flatten_content_parts(parts: &[Value]) -> Result<String, DsEncodingError> {
    let mut texts: Vec<&str> = Vec::with_capacity(parts.len());
    for part in parts {
        let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match part_type {
            "text" | "input_text" | "output_text" => {
                let text = part.get("text").and_then(|v| v.as_str()).unwrap_or("");
                if text.contains(IMAGE_PLACEHOLDER) {
                    return Err(DsEncodingError::PlaceholderInText);
                }
                texts.push(text);
            }
            "image" | "image_url" | "input_image" | "image_pil" => texts.push(IMAGE_PLACEHOLDER),
            other => return Err(DsEncodingError::UnsupportedContentPart(other.to_string())),
        }
    }
    Ok(texts.join("\n\n"))
}

// ---------------------------------------------------------------------------
// render_message — direct port of the V4.1 Python function with the same name.
// ---------------------------------------------------------------------------
/// Mirrors V4.1's `find_last_user_index`: unlike V4, a mid-conversation system
/// message counts as a user turn. Returns `None` where Python returns -1.
fn find_last_user_index(messages: &[Value]) -> Option<usize> {
    for index in (0..messages.len()).rev() {
        let role = messages[index].get("role").and_then(|v| v.as_str());
        let counts = matches!(role, Some("user") | Some("developer"))
            || (role == Some("system") && index > 0);
        if counts {
            return Some(index);
        }
    }
    None
}

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
    let msg = &messages[index];
    let last_user_idx = find_last_user_index(messages);

    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
    let content = msg.get("content").and_then(|v| v.as_str()).unwrap_or("");
    let tools_raw = msg.get("tools").and_then(|v| v.as_array());
    // Python `if response_format:` — absent, null and `{}` are all falsy.
    let response_format = msg
        .get("response_format")
        .filter(|v| !v.is_null() && v.as_object().is_none_or(|map| !map.is_empty()));
    let tool_calls_raw = msg.get("tool_calls").and_then(|v| v.as_array());
    let reasoning_content = msg
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let wo_eos = msg.get("wo_eos").and_then(|v| v.as_bool()).unwrap_or(false);
    let tools_owned = tools_raw.map(|tools| tools_from_openai_format(tools));
    let tools = tools_owned.as_deref().filter(|tools| !tools.is_empty());
    let tool_calls_owned =
        tool_calls_raw.map(|tool_calls| tool_calls_from_openai_format(tool_calls));
    let tool_calls = tool_calls_owned
        .as_deref()
        .filter(|tool_calls| !tool_calls.is_empty());

    // Reasoning effort prefix: index 0, thinking mode only.
    let reasoning_effort_prompt = render_reasoning_effort(index, thinking_mode, reasoning_effort);
    // The system token leads the conversation when there is a reasoning effort
    // prompt or the first message is a system message.
    let mut prompt = String::new();
    if index == 0 && (!reasoning_effort_prompt.is_empty() || role == "system") {
        prompt.push_str(SYSTEM_SP_TOKEN);
    }
    prompt.push_str(&reasoning_effort_prompt);

    match role {
        "system" => {
            if index > 0 {
                // Mid-conversation system message.
                prompt.push_str(SYSTEM_SP_TOKEN);
            }
            prompt.push_str(content);
            if let Some(tools) = tools {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(tools));
            }
            if let Some(schema) = response_format {
                prompt.push_str("\n\n");
                prompt.push_str(&render_response_format(schema));
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
            if let Some(tools) = tools {
                content_developer.push_str("\n\n");
                content_developer.push_str(&render_tools(tools));
            }
            if let Some(schema) = response_format {
                content_developer.push_str("\n\n");
                content_developer.push_str(&render_response_format(schema));
            }
            prompt.push_str(&content_developer);
        }

        "user" => {
            prompt.push_str(USER_SP_TOKEN);
            // Handle content blocks (tool results mixed with text).
            let content_blocks = msg
                .get("content_blocks")
                .and_then(|v| v.as_array())
                .filter(|blocks| !blocks.is_empty());
            if let Some(content_blocks) = content_blocks {
                let mut parts: Vec<String> = Vec::with_capacity(content_blocks.len());
                for block in content_blocks {
                    let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match block_type {
                        "text" => {
                            let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                            parts.push(text.to_string());
                        }
                        "tool_result" => {
                            let tool_content = match block.get("content") {
                                Some(Value::Array(items)) => {
                                    let mut text_parts: Vec<String> =
                                        Vec::with_capacity(items.len());
                                    for item in items {
                                        let item_type =
                                            item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                                        if item_type == "text" {
                                            text_parts.push(
                                                item.get("text")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("")
                                                    .to_string(),
                                            );
                                        } else {
                                            text_parts.push(format!("[Unsupported {item_type}]"));
                                        }
                                    }
                                    text_parts.join("\n\n")
                                }
                                Some(Value::String(text)) => text.clone(),
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
            if let Some(tool_calls) = tool_calls {
                let mut tc_list = Vec::with_capacity(tool_calls.len());
                for tool_call in tool_calls {
                    let name = tool_call.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let arguments = tool_call.get("arguments").unwrap_or(&Value::Null);
                    let args = encode_arguments_to_dsml(arguments, &DSML_TAGS);
                    tc_list.push(format!(
                        "<{DSML_TOKEN}{invoke} name=\"{name}\">\n{args}\n</{DSML_TOKEN}{invoke}>",
                        invoke = DSML_TAGS.invoke,
                    ));
                }
                let joined = tc_list.join("\n");
                let _ = write!(
                    tc_content,
                    "\n\n<{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>\n{joined}\n</{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>"
                );
            }
            // prev_has_task: if the previous message had a task, this is a task
            // output (no thinking).
            let prev_has_task = index >= 1
                && messages[index - 1]
                    .get("task")
                    .is_some_and(|task| !task.is_null());
            if thinking_mode == ThinkingMode::Thinking && !prev_has_task {
                // Python: `not drop_thinking or index > last_user_idx`, where a
                // conversation without a user turn has last_user_idx == -1.
                let after_last_user = match last_user_idx {
                    Some(last) => index > last,
                    None => true,
                };
                if !drop_thinking || after_last_user {
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

    let task = msg.get("task").filter(|task| !task.is_null());
    if let Some(task) = task {
        let task = task.as_str().unwrap_or("");
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
            // Non-action tasks: append the task sp token directly after the message.
            prompt.push_str(sp_token);
        }
    } else if matches!(role, "user" | "developer") || (role == "system" && index > 0) {
        // Normal generation: append Assistant + thinking token
        // (mid-conversation system messages also trigger the assistant header).
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
// encode_messages — public entry point
// ---------------------------------------------------------------------------
/// Parameters for [`encode_messages`].
///
/// `context` is intentionally omitted: SMG always renders from scratch, so the
/// Python default of `context=None` always applies.
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

/// Encode a list of OpenAI-style messages into a DeepSeek V4.1 prompt string.
///
/// The signature mirrors the Python `encode_messages` function; `context` is
/// omitted because SMG always renders from scratch.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "public API mirrors the documented Rust signature with a borrow"
)]
pub fn encode_messages(
    messages: &[Value],
    thinking_mode: ThinkingMode,
    params: &EncodeParams,
) -> Result<String, DsEncodingError> {
    // Normalise multimodal content parts into text before anything else.
    let normalized = process_image_messages(messages)?;
    // Preprocess: merge tool messages and sort tool results.
    let merged = merge_tool_messages(&normalized);
    let mut full_messages = sort_tool_results_by_call_order(merged);
    let mut prompt = if params.add_default_bos_token {
        BOS_TOKEN.to_string()
    } else {
        String::new()
    };
    // Resolve drop_thinking: if any message has tools defined, never drop.
    let mut effective_drop_thinking = params.drop_thinking;
    if full_messages.iter().any(has_tools) {
        effective_drop_thinking = false;
    }
    if thinking_mode == ThinkingMode::Thinking && effective_drop_thinking {
        let last_user_idx = find_last_user_index(&full_messages);
        full_messages = drop_thinking_messages(&full_messages, last_user_idx);
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
    use serde_json::{json, Value};

    use super::*;

    fn user(q: &str) -> Value {
        json!({ "role": "user", "content": q })
    }
    fn params(effort: Option<ReasoningEffort>) -> EncodeParams {
        EncodeParams {
            add_default_bos_token: true,
            drop_thinking: true,
            reasoning_effort: effort,
        }
    }

    #[test]
    fn chat_mode_no_system_has_no_system_token() {
        let out = encode_messages(&[user("q")], ThinkingMode::Chat, &params(None)).unwrap();
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>q<｜Assistant｜></think>"
        );
    }

    #[test]
    fn thinking_mode_prefixes_effort_line_with_default_50() {
        let out = encode_messages(&[user("q")], ThinkingMode::Thinking, &params(None)).unwrap();
        assert_eq!(out, "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>q<｜Assistant｜><think>");
    }

    #[test]
    fn effort_tiers_follow_engine_table() {
        for (tier, n) in [
            (ReasoningEffort::Low, 25),
            (ReasoningEffort::High, 50),
            (ReasoningEffort::XHigh, 75),
            (ReasoningEffort::Max, 100),
            (ReasoningEffort::Budget(42), 42),
        ] {
            let out =
                encode_messages(&[user("q")], ThinkingMode::Thinking, &params(Some(tier))).unwrap();
            assert!(
                out.contains(&format!("Reasoning Effort: {n} (range")),
                "{tier:?}"
            );
        }
    }

    #[test]
    fn parse_effort_rejects_minimal_medium_floats_and_out_of_range() {
        for bad in [
            json!("minimal"),
            json!("medium"),
            json!("none"),
            json!(0),
            json!(101),
            json!(1.5),
            json!(true),
            json!([]),
        ] {
            assert!(parse_reasoning_effort(&bad).is_err(), "{bad}");
        }
        assert_eq!(
            parse_reasoning_effort(&json!("xhigh")).unwrap(),
            Some(ReasoningEffort::XHigh)
        );
        assert_eq!(
            parse_reasoning_effort(&json!(7)).unwrap(),
            Some(ReasoningEffort::Budget(7))
        );
        assert_eq!(parse_reasoning_effort(&Value::Null).unwrap(), None);
        // Every advertised native name parses.
        for name in NATIVE_EFFORT_VALUES {
            assert!(
                parse_reasoning_effort(&json!(name)).unwrap().is_some(),
                "{name}"
            );
        }
    }

    #[test]
    fn tool_call_turn_uses_spaced_dsml_tags() {
        let msgs = [
            json!({"role": "system", "content": "S", "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]}),
            user("What's the weather like in Beijing?"),
            json!({"role": "assistant", "reasoning_content": "R", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Beijing\", \"unit\": \"celsius\"}"}}]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "{\"temperature\": 22}"}),
        ];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &params(None)).unwrap();
        assert!(out.contains("<｜Assistant｜><think>R</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"location\" string=\"true\">Beijing</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"unit\" string=\"true\">celsius</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜><｜User｜><tool_result>{\"temperature\": 22}</tool_result><｜Assistant｜><think>"), "{out}");
        assert!(out.contains("\n\n## Tools\n\nYou have access to a set of tools"));
    }

    #[test]
    fn images_render_as_placeholder_joined_by_blank_lines() {
        let msg = json!({"role": "user", "content": [
            {"type": "text", "text": "first"},
            {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
            {"type": "text", "text": "second"}]});
        let out = encode_messages(&[msg], ThinkingMode::Chat, &params(None)).unwrap();
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>first\n\n<｜deepseek_image｜>\n\nsecond<｜Assistant｜></think>"
        );
    }

    #[test]
    fn image_part_without_a_source_still_renders_the_placeholder() {
        // SMG's `chat_utils::transform_content_field` strips image URLs before
        // the renderer sees them, leaving a bare `{"type": "image"}` part.
        let msg = json!({"role": "user", "content": [
            {"type": "text", "text": "a"},
            {"type": "image"},
            {"type": "text", "text": "b"}]});
        let out = encode_messages(&[msg], ThinkingMode::Chat, &params(None)).unwrap();
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>a\n\n<｜deepseek_image｜>\n\nb<｜Assistant｜></think>"
        );
    }

    #[test]
    fn unsupported_content_part_type_errors() {
        let msg = json!({"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA=="}}]});
        let err = encode_messages(&[msg], ThinkingMode::Chat, &params(None)).unwrap_err();
        assert!(
            matches!(err, DsEncodingError::UnsupportedContentPart(ref t) if t == "input_audio"),
            "{err}"
        );
    }

    #[test]
    fn history_reasoning_dropped_without_tools_and_kept_with_tools() {
        let history = |tools: bool| {
            let mut sys = json!({"role": "system", "content": "S"});
            if tools {
                sys["tools"] =
                    json!([{"type": "function", "function": {"name": "f", "parameters": {}}}]);
            }
            vec![
                sys,
                user("q1"),
                json!({"role": "assistant", "reasoning_content": "r1", "content": "a1"}),
                user("q2"),
            ]
        };
        let dropped =
            encode_messages(&history(false), ThinkingMode::Thinking, &params(None)).unwrap();
        assert!(dropped
            .contains("<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜><｜User｜>q2<｜Assistant｜><think>"));
        let kept = encode_messages(&history(true), ThinkingMode::Thinking, &params(None)).unwrap();
        assert!(kept.contains(
            "<｜User｜>q1<｜Assistant｜><think>r1</think>a1<｜end▁of▁sentence｜><｜User｜>q2<｜Assistant｜><think>"
        ));
    }

    #[test]
    fn mid_conversation_system_counts_as_user_for_the_header() {
        let msgs = [
            user("q1"),
            json!({"role": "assistant", "content": "a1"}),
            json!({"role": "system", "content": "MID"}),
            user("q2"),
        ];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &params(None)).unwrap();
        assert!(out
            .ends_with("a1<｜end▁of▁sentence｜><｜System｜>MID<｜User｜>q2<｜Assistant｜><think>"));
    }

    #[test]
    fn placeholder_inside_user_text_is_rejected() {
        let out = encode_messages(
            &[user("<｜deepseek_image｜>")],
            ThinkingMode::Chat,
            &params(None),
        );
        assert!(matches!(out, Err(DsEncodingError::PlaceholderInText)));
        // ... and inside a text part, and inside reasoning_content.
        let part =
            json!({"role": "user", "content": [{"type": "text", "text": "x<｜deepseek_image｜>"}]});
        assert!(matches!(
            encode_messages(&[part], ThinkingMode::Chat, &params(None)),
            Err(DsEncodingError::PlaceholderInText)
        ));
        let reasoning = json!({"role": "assistant", "reasoning_content": "<｜deepseek_image｜>", "content": "a"});
        assert!(matches!(
            encode_messages(
                &[user("q"), reasoning],
                ThinkingMode::Thinking,
                &params(None)
            ),
            Err(DsEncodingError::PlaceholderInText)
        ));
    }

    #[test]
    fn continue_final_message_omits_the_eos_token() {
        let msgs = [
            user("q"),
            json!({"role": "assistant", "reasoning_content": "r", "content": "Sure,", "wo_eos": true}),
        ];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &params(None)).unwrap();
        assert_eq!(out, "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>q<｜Assistant｜><think>r</think>Sure,");
    }

    #[test]
    fn developer_role_renders_as_a_user_turn() {
        let msgs = [json!({"role": "developer", "content": "D"}), user("q")];
        let out = encode_messages(&msgs, ThinkingMode::Chat, &params(None)).unwrap();
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>D<｜User｜>q<｜Assistant｜></think>"
        );
        // Empty developer content is rejected, as in the Python.
        let empty = [json!({"role": "developer", "content": ""})];
        let err = encode_messages(&empty, ThinkingMode::Chat, &params(None)).unwrap_err();
        assert!(
            matches!(err, DsEncodingError::InvalidMessage { .. }),
            "{err}"
        );
    }

    #[test]
    fn developer_message_is_dropped_before_the_last_user_turn() {
        let msgs = [
            json!({"role": "developer", "content": "DEV"}),
            user("q1"),
            json!({"role": "assistant", "content": "a1"}),
            user("q2"),
        ];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &params(None)).unwrap();
        assert!(!out.contains("DEV"), "{out}");
        assert!(out.ends_with("<｜User｜>q2<｜Assistant｜><think>"), "{out}");
    }

    #[test]
    fn bare_tool_role_reaching_render_message_errors() {
        let msgs = [json!({"role": "tool", "content": "x"})];
        let err = render_message(0, &msgs, ThinkingMode::Chat, true, None).unwrap_err();
        assert!(matches!(err, DsEncodingError::UnmergedToolRole(0)), "{err}");
    }

    #[test]
    fn unknown_role_errors() {
        let msgs = [json!({ "role": "moderator", "content": "hi" })];
        let err = encode_messages(&msgs, ThinkingMode::Chat, &params(None)).unwrap_err();
        assert!(matches!(err, DsEncodingError::UnknownRole(ref r) if r == "moderator"));
    }

    /// Acceptance oracle: every prompt recorded from the reference encoder.
    #[test]
    fn render_fixtures_match_recorded_text() {
        const FIXTURES: &str =
            include_str!("../../tests/fixtures/deepseek_v41/render_fixtures.json");

        let doc: Value = serde_json::from_str(FIXTURES).unwrap();
        let cases = doc["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 29, "fixture case count");

        for case in cases {
            let name = case["name"].as_str().unwrap();
            let mut messages: Vec<Value> = case["messages"].as_array().unwrap().clone();

            // Rule D4, as the generator and the shim's
            // `inject_tools_into_first_system_message` apply it: the request's
            // tools land on the first system message wherever it sits; without
            // one, an empty system message is inserted up front and carries
            // them.
            if let Some(tools) = case.get("tools").filter(|t| !t.is_null()) {
                let system_index = messages.iter().position(|m| m["role"] == json!("system"));
                let index = system_index.unwrap_or_else(|| {
                    messages.insert(0, json!({ "role": "system", "content": "" }));
                    0
                });
                messages[index]["tools"] = tools.clone();
            }
            // `continue_final_message` was recorded as `wo_eos` on the last message.
            if case["continue_final_message"].as_bool().unwrap_or(false) {
                if let Some(last) = messages.last_mut() {
                    last["wo_eos"] = json!(true);
                }
            }

            let thinking_mode = match case["thinking_mode"].as_str().unwrap() {
                "thinking" => ThinkingMode::Thinking,
                "chat" => ThinkingMode::Chat,
                other => panic!("case {name}: unknown thinking_mode {other}"),
            };
            let encode_params = EncodeParams {
                add_default_bos_token: true,
                drop_thinking: case["drop_thinking"].as_bool().unwrap_or(true),
                reasoning_effort: parse_reasoning_effort(&case["reasoning_effort"]).unwrap(),
            };

            let out = encode_messages(&messages, thinking_mode, &encode_params).unwrap();
            let expected = case["text"].as_str().unwrap();
            assert_eq!(
                out,
                expected,
                "case {name}\n{}",
                diff_excerpt(expected, &out)
            );
        }
    }

    /// Unified-diff-style excerpt around the first diverging character.
    fn diff_excerpt(expected: &str, actual: &str) -> String {
        let expected_chars: Vec<char> = expected.chars().collect();
        let actual_chars: Vec<char> = actual.chars().collect();
        let common = expected_chars
            .iter()
            .zip(actual_chars.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let start = common.saturating_sub(40);
        let window = |chars: &[char]| -> String { chars.iter().skip(start).take(120).collect() };
        format!(
            "first difference at char {common}:\n- expected: {:?}\n+ actual:   {:?}",
            window(&expected_chars),
            window(&actual_chars)
        )
    }
}
