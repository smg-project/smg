// Ported from the serving engine's `tokenizers/deepseek_v41_encoding.py` (itself a port
// of the `encoding/encoding.py` shipped with deepseek-ai/DeepSeek-V4.1-Flash),
// function by function and in the same order, plus the message normalisation
// that the engine keeps in `tokenizers/deepseek_v41.py::_normalize_messages`.
//
// V4.1 differs from V4 (`deepseek_v4.rs`) in four places: the DSML tag strings
// carry a leading space (` invoke`, ` parameter`, ` calls`), the reasoning
// effort is a number in a `<｜System｜>`-led prefix instead of a prose block,
// mid-conversation system messages count as user turns, and image content
// parts collapse to [`IMAGE_PLACEHOLDER`].

use serde_json::Value;

// `DsEncodingError` lives in `deepseek_common` (shared with V4); re-exported
// here because it is part of this module's public API, and `pub(super)` items
// are not reachable from outside `encoders` on their own.
pub use super::deepseek_common::DsEncodingError;
// Reuse the public ThinkingMode enum from the V3.2 module to keep the
// "thinking" / "chat" mode invariant identical across DeepSeek versions.
pub use super::deepseek_v32::ThinkingMode;
// Message-shape preprocessing, drop-thinking, DSML argument rendering and the
// tools template are V4's too, apart from the tag strings and the last-turn
// definition; see `deepseek_common` for the shared implementation and its
// own doc comments.
use super::{
    deepseek_common::{
        after_last_user, at_or_after_last_user, drop_thinking_messages, encode_arguments_to_dsml,
        has_tools, merge_tool_messages, render_v4_tools_template, sort_tool_results_by_call_order,
        task_sp_token, to_json, tool_calls_from_openai_format, tool_schemas,
        tools_from_openai_format, DsmlTags,
    },
    RendererProfile,
};
use crate::{
    chat_template::{ChatTemplateContentFormat, ThinkingKeyName, ThinkingToggle},
    traits::RendererCapabilities,
};

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

// ---------------------------------------------------------------------------
// Reasoning effort
// ---------------------------------------------------------------------------
/// The effort names the Python encoder accepts (`REASONING_EFFORT_MAPPINGS`).
pub const NATIVE_EFFORT_VALUES: &[&str] = &["low", "high", "xhigh", "max"];

/// What this renderer answers around a render (see [`RendererProfile`]).
pub(crate) const PROFILE: RendererProfile = RendererProfile {
    // Message parts are preserved (OpenAI wire format) so the gateway passes
    // `image_url` / `image` parts through as a list instead of flattening to
    // a string; the encoder turns each part into a placeholder at its
    // position. Part *order* is still the gateway's media-part order (the
    // multimodal spec declares `Authored`; without a spec the gateway moves
    // media first).
    content_format: Some(ChatTemplateContentFormat::OpenAI),
    // Thinking defaults ON: `reasoning_effort: "none"` or an explicit
    // `thinking: false` (or the engine's `enable_thinking` alias, see the
    // capabilities) turns it off.
    thinking_toggle: ThinkingToggle::DefaultOn,
    thinking_key_name: ThinkingKeyName::Thinking,
    native_effort_values: NATIVE_EFFORT_VALUES,
    // Switches off on the protocol's `none` / `minimal`: no words of its own.
    native_effort_off_values: &[],
    // `<｜Assistant｜><think>` ends the prompt when thinking is on.
    think_in_prefill: true,
    // The shim honours the engine's `enable_thinking` alias, renders a
    // trailing assistant message itself when `add_generation_prompt` is
    // false, and parses tool-call `arguments` strings with the reference's
    // tolerance.
    capabilities: RendererCapabilities {
        enable_thinking_alias: true,
        native_assistant_continuation: true,
        raw_tool_call_arguments: true,
        declares_tools_under_tool_choice_none: false,
    },
    thinking_mode: None,
};
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
    /// Spec D1: engine table. Flip these four numbers if the serving engines adopt
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
/// Mirrors V4.1's `TOOLS_TEMPLATE`: the V4 family's text with this dialect's
/// block name and spaced tags.
fn render_tools(tools: &[Value]) -> String {
    render_v4_tools_template(TOOL_CALLS_BLOCK_NAME, &DSML_TAGS, &tool_schemas(tools))
}

fn render_response_format(schema: &Value) -> String {
    format!(
        "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{}",
        to_json(schema)
    )
}

// ---------------------------------------------------------------------------
// Image/content-part normalisation
// ---------------------------------------------------------------------------
// The engine's `_normalize_messages` plus the reference encoder's
// `process_image_messages`: list content collapses to a single string in which
// every image part became `IMAGE_PLACEHOLDER`, parts joined by a blank line.
// The image payloads themselves are not collected here — SMG's multimodal path
// carries them separately, the prompt only needs the marker.

/// Mirrors the engine's `_normalize_messages` flattening plus the HF
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
/// Mirrors V4.1's `find_last_user_index` under a name of its own, since the
/// rule differs from [`super::deepseek_common::find_last_user_index`]: a
/// mid-conversation system message counts as a user turn here. Returns
/// `None` where Python returns -1.
fn find_last_turn_index(messages: &[Value]) -> Option<usize> {
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

/// One message as the renderer reads it: the fields of the OpenAI-shape
/// value, read once, with the tools and tool calls in the encoder's shape
/// and empty lists dropped the way Python's truthiness drops them.
struct Turn<'m> {
    index: usize,
    msg: &'m Value,
    role: &'m str,
    content: &'m str,
    tools: Option<Vec<Value>>,
    /// Python `if response_format:` — absent, null and `{}` are all falsy.
    response_format: Option<&'m Value>,
    tool_calls: Option<Vec<Value>>,
    reasoning_content: &'m str,
    wo_eos: bool,
}

impl<'m> Turn<'m> {
    fn read(index: usize, messages: &'m [Value]) -> Result<Self, DsEncodingError> {
        let Some(msg) = messages.get(index) else {
            return Err(DsEncodingError::IndexOutOfRange {
                index,
                len: messages.len(),
            });
        };
        let tools = msg
            .get("tools")
            .and_then(|v| v.as_array())
            .map(|tools| tools_from_openai_format(tools))
            .filter(|tools| !tools.is_empty());
        let tool_calls = msg
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|tool_calls| tool_calls_from_openai_format(tool_calls))
            .filter(|tool_calls| !tool_calls.is_empty());
        Ok(Self {
            index,
            msg,
            role: text(msg, "role"),
            content: text(msg, "content"),
            tools,
            response_format: msg
                .get("response_format")
                .filter(|v| !v.is_null() && v.as_object().is_none_or(|map| !map.is_empty())),
            tool_calls,
            reasoning_content: text(msg, "reasoning_content"),
            wo_eos: msg.get("wo_eos").and_then(|v| v.as_bool()).unwrap_or(false),
        })
    }
}

/// A string field of a message, empty when absent or not a string, as the
/// reference's `msg.get(key, "")` reads it.
fn text<'m>(msg: &'m Value, key: &str) -> &'m str {
    msg.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

/// Mirrors the reference's `render_message`: one turn's text, by role in the
/// reference's order, then the transition to the next turn.
fn render_message(
    index: usize,
    messages: &[Value],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    reasoning_effort: Option<ReasoningEffort>,
) -> Result<String, DsEncodingError> {
    let turn = Turn::read(index, messages)?;
    let last_user_idx = find_last_turn_index(messages);
    let mut prompt = render_prefix(&turn, thinking_mode, reasoning_effort);
    match turn.role {
        "system" => render_system(&turn, &mut prompt),
        "developer" => render_developer(&turn, &mut prompt)?,
        "user" => render_user(&turn, &mut prompt),
        "latest_reminder" => {
            prompt.push_str(LATEST_REMINDER_SP_TOKEN);
            prompt.push_str(turn.content);
        }
        "tool" => return Err(DsEncodingError::UnmergedToolRole(index)),
        "assistant" => render_assistant(
            &turn,
            messages,
            thinking_mode,
            drop_thinking,
            last_user_idx,
            &mut prompt,
        ),
        other => return Err(DsEncodingError::UnknownRole(other.to_string())),
    }
    push_transition(
        &turn,
        messages,
        thinking_mode,
        drop_thinking,
        last_user_idx,
        &mut prompt,
    )?;
    Ok(prompt)
}

/// The system token when the conversation opens with a reasoning-effort
/// prompt or a system message, then the reasoning-effort prompt (index 0,
/// thinking mode only).
fn render_prefix(
    turn: &Turn<'_>,
    thinking_mode: ThinkingMode,
    reasoning_effort: Option<ReasoningEffort>,
) -> String {
    let reasoning_effort_prompt =
        render_reasoning_effort(turn.index, thinking_mode, reasoning_effort);
    let mut prompt = String::new();
    if turn.index == 0 && (!reasoning_effort_prompt.is_empty() || turn.role == "system") {
        prompt.push_str(SYSTEM_SP_TOKEN);
    }
    prompt.push_str(&reasoning_effort_prompt);
    prompt
}

/// A system turn: a mid-conversation one carries its own system token; the
/// tools and the response format follow the content.
fn render_system(turn: &Turn<'_>, prompt: &mut String) {
    if turn.index > 0 {
        prompt.push_str(SYSTEM_SP_TOKEN);
    }
    prompt.push_str(turn.content);
    push_tools_and_response_format(turn, prompt);
}

/// A developer turn renders as a user turn; an empty one is an error.
fn render_developer(turn: &Turn<'_>, prompt: &mut String) -> Result<(), DsEncodingError> {
    if turn.content.is_empty() {
        return Err(DsEncodingError::InvalidMessage {
            role: turn.role.to_string(),
            msg: turn.msg.to_string(),
        });
    }
    prompt.push_str(USER_SP_TOKEN);
    prompt.push_str(turn.content);
    push_tools_and_response_format(turn, prompt);
    Ok(())
}

/// The tools template and the response format of a system or developer
/// turn, each after a blank line, when the turn has them.
fn push_tools_and_response_format(turn: &Turn<'_>, prompt: &mut String) {
    if let Some(tools) = &turn.tools {
        prompt.push_str("\n\n");
        prompt.push_str(&render_tools(tools));
    }
    if let Some(schema) = turn.response_format {
        prompt.push_str("\n\n");
        prompt.push_str(&render_response_format(schema));
    }
}

/// A user turn: its content blocks (text and the tool results
/// `merge_tool_messages` folded in) joined by blank lines, else its content.
fn render_user(turn: &Turn<'_>, prompt: &mut String) {
    prompt.push_str(USER_SP_TOKEN);
    let blocks = turn
        .msg
        .get("content_blocks")
        .and_then(|v| v.as_array())
        .filter(|blocks| !blocks.is_empty());
    match blocks {
        Some(blocks) => {
            let parts: Vec<String> = blocks.iter().map(render_content_block).collect();
            prompt.push_str(&parts.join("\n\n"));
        }
        None => prompt.push_str(turn.content),
    }
}

/// One content block: its text, a tool result in its tags, or an
/// `[Unsupported ...]` marker, as the reference renders them.
fn render_content_block(block: &Value) -> String {
    match text(block, "type") {
        "text" => text(block, "text").to_string(),
        "tool_result" => format!(
            "<tool_result>{}</tool_result>",
            tool_result_text(block.get("content"))
        ),
        other => format!("[Unsupported {other}]"),
    }
}

/// A tool result's content as text: a list of parts joined by blank lines
/// (text parts as they are, any other as an `[Unsupported ...]` marker), a
/// string as it is, any other value as JSON, nothing when absent.
fn tool_result_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::Array(items)) => {
            let parts: Vec<String> = items
                .iter()
                .map(|item| match text(item, "type") {
                    "text" => text(item, "text").to_string(),
                    other => format!("[Unsupported {other}]"),
                })
                .collect();
            parts.join("\n\n")
        }
        Some(Value::String(content)) => content.clone(),
        Some(other) => to_json(other),
        None => String::new(),
    }
}

/// An assistant turn: its reasoning closed by the think-end token (in
/// thinking mode, kept after the last user turn or everywhere when thinking
/// is not dropped, never for a task output), its content, its tool calls
/// block, and the end token unless `wo_eos`.
fn render_assistant(
    turn: &Turn<'_>,
    messages: &[Value],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    last_user_idx: Option<usize>,
    prompt: &mut String,
) {
    // A task output: the previous message named a task.
    let prev_has_task = turn.index >= 1
        && messages[turn.index - 1]
            .get("task")
            .is_some_and(|task| !task.is_null());
    if thinking_mode == ThinkingMode::Thinking
        && !prev_has_task
        && (!drop_thinking || after_last_user(turn.index, last_user_idx))
    {
        prompt.push_str(turn.reasoning_content);
        prompt.push_str(THINKING_END_TOKEN);
    }
    prompt.push_str(turn.content);
    if let Some(tool_calls) = &turn.tool_calls {
        prompt.push_str(&tool_calls_block(tool_calls));
    }
    if !turn.wo_eos {
        prompt.push_str(EOS_TOKEN);
    }
}

/// The tool calls block after a blank line: each call's invoke tags around
/// its DSML arguments, inside the dialect's block tags.
fn tool_calls_block(tool_calls: &[Value]) -> String {
    let calls: Vec<String> = tool_calls
        .iter()
        .map(|tool_call| {
            let name = text(tool_call, "name");
            let arguments = tool_call.get("arguments").unwrap_or(&Value::Null);
            let args = encode_arguments_to_dsml(arguments, &DSML_TAGS);
            format!(
                "<{DSML_TOKEN}{invoke} name=\"{name}\">\n{args}\n</{DSML_TOKEN}{invoke}>",
                invoke = DSML_TAGS.invoke,
            )
        })
        .collect();
    let joined = calls.join("\n");
    format!(
        "\n\n<{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>\n{joined}\n</{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>"
    )
}

/// What follows the turn: nothing when a next message exists and is not an
/// assistant or latest-reminder turn; otherwise (the last turn included: the
/// generation prompt) a task's token (the action task behind the assistant
/// header and a think token); else, after a user, developer or
/// mid-conversation system turn, the assistant header and the think token
/// that opens the reasoning (thinking mode, kept at or after the last user
/// turn or everywhere when thinking is not dropped) or closes it.
fn push_transition(
    turn: &Turn<'_>,
    messages: &[Value],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    last_user_idx: Option<usize>,
    prompt: &mut String,
) -> Result<(), DsEncodingError> {
    if let Some(next) = messages.get(turn.index + 1) {
        if !matches!(text(next, "role"), "assistant" | "latest_reminder") {
            return Ok(());
        }
    }
    let task = turn.msg.get("task").filter(|task| !task.is_null());
    if let Some(task) = task {
        let task = task.as_str().unwrap_or("");
        let sp_token =
            task_sp_token(task).ok_or_else(|| DsEncodingError::InvalidTask(task.to_string()))?;
        if task == "action" {
            prompt.push_str(ASSISTANT_SP_TOKEN);
            prompt.push_str(if thinking_mode == ThinkingMode::Thinking {
                THINKING_START_TOKEN
            } else {
                THINKING_END_TOKEN
            });
        }
        prompt.push_str(sp_token);
    } else if matches!(turn.role, "user" | "developer") || (turn.role == "system" && turn.index > 0)
    {
        prompt.push_str(ASSISTANT_SP_TOKEN);
        let opens_thinking = thinking_mode == ThinkingMode::Thinking
            && (!drop_thinking || at_or_after_last_user(turn.index, last_user_idx));
        prompt.push_str(if opens_thinking {
            THINKING_START_TOKEN
        } else {
            THINKING_END_TOKEN
        });
    }
    Ok(())
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
    let normalized = process_image_messages(messages)?;
    let merged = merge_tool_messages(&normalized);
    let mut full_messages = sort_tool_results_by_call_order(merged);
    let mut prompt = if params.add_default_bos_token {
        BOS_TOKEN.to_string()
    } else {
        String::new()
    };
    // Thinking is never dropped from a conversation that defines tools.
    let mut effective_drop_thinking = params.drop_thinking;
    if full_messages.iter().any(has_tools) {
        effective_drop_thinking = false;
    }
    if thinking_mode == ThinkingMode::Thinking && effective_drop_thinking {
        let last_user_idx = find_last_turn_index(&full_messages);
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
