// Ported from https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py

use std::fmt::Write as _;

use serde_json::Value;
use thiserror::Error;

// The index predicates, the JSON spelling and the OpenAI-shape readers are
// the V4 family's too and come from `deepseek_common`. V3.2's tool-message
// handling, drop-thinking role set, DSML argument rendering and tools
// template are different algorithms and texts and stay here.
use super::{
    deepseek_common::{
        after_last_user, at_or_after_last_user, find_last_user_index, to_json,
        tool_calls_from_openai_format, tool_schemas, tools_from_openai_format, ASSISTANT, BOS,
        DSML, EOS, THINKING_END, THINKING_START, USER,
    },
    marks, RendererProfile,
};
use crate::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::RendererCapabilities,
};

/// Mode for thinking/reasoning rendering.
///
/// Mirrors the Python `thinking_mode` parameter, which only accepts `"chat"`
/// and `"thinking"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
    fn is_thinking(self) -> bool {
        matches!(self, ThinkingMode::Thinking)
    }
}

/// Parameters for [`encode_messages`].
///
/// `context` is intentionally omitted: SMG always renders from scratch, so
/// the Python default of `context=None` always applies.
#[derive(Debug, Clone, Copy)]
pub struct EncodeParams {
    pub add_default_bos_token: bool,
    pub drop_thinking: bool,
}

impl Default for EncodeParams {
    fn default() -> Self {
        Self {
            add_default_bos_token: true,
            drop_thinking: true,
        }
    }
}

/// Errors raised when a message list is malformed.
///
/// Mirrors the Python `DS32EncodingError`.
#[derive(Debug, Error)]
pub enum DsEncodingError {
    #[error("Index {index} out of range for messages list of length {len}")]
    IndexOutOfRange { index: usize, len: usize },

    #[error("Invalid message for role `{role}`: {msg}")]
    InvalidMessage { role: String, msg: String },

    #[error("Invalid messages at {index}: {context}")]
    InvalidToolMessages { index: usize, context: String },

    #[error("No tool calls but found tool output")]
    NoToolCalls,

    #[error("Unknown role: {0}")]
    UnknownRole(String),

    #[error("thinking mode: invalid message without reasoning_content/tool_calls after last user message: {0}")]
    MissingReasoningOrToolCalls(String),

    #[error("Failed to parse tool-call arguments as JSON: {0}")]
    InvalidToolArgumentsJson(#[source] serde_json::Error),
}

// ---------------------------------------------------------------------------
// Special-token constants: the family's shared ones come from `deepseek_common`
// (re-exported under this renderer's path). Copied verbatim from the Python
// source.
// ---------------------------------------------------------------------------

pub use super::deepseek_common::{
    BOS_TOKEN, DSML_TOKEN, EOS_TOKEN, THINKING_END_TOKEN, THINKING_START_TOKEN,
};

/// What this renderer answers around a render (see [`RendererProfile`]).
pub(crate) const PROFILE: RendererProfile = RendererProfile {
    // No opinion on content parts: the (usually absent) template's format.
    content_format: None,
    // Thinking is gated on the `thinking` kwarg, default off.
    thinking_toggle: ThinkingToggle::DefaultOff,
    thinking_key_name: ThinkingKeyName::Thinking,
    // Switches off on the protocol's `none` / `minimal`: no words of its own.
    native_effort_values: &[],
    native_effort_off_values: &[],
    // `<｜Assistant｜><think>` ends the prompt when thinking is on.
    think_in_prefill: true,
    // The engine's tokenizer wrapper switches thinking on from `thinking` or
    // its `enable_thinking` alias; the shim reads both, so the gateway arms
    // the reasoning parser from the alias as the prompt is rendered from it.
    capabilities: RendererCapabilities {
        enable_thinking_alias: true,
        ..RendererProfile::NO_CAPABILITIES
    },
    thinking_mode: None,
};

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

/// Mirrors `TOOLS_SYSTEM_TEMPLATE` from the Python source, including the
/// blank lines between paragraphs that the upstream template preserves.
fn render_tools_template(tool_schemas: &str) -> String {
    let dsml = DSML;
    let tstart = THINKING_START;
    let tend = THINKING_END;
    format!(
"## Tools

You have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a \"<{dsml}function_calls>\" block like the following as part of your reply to the user:
<{dsml}function_calls>
<{dsml}invoke name=\"$FUNCTION_NAME\">
<{dsml}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name=\"$FUNCTION_NAME2\">
...
</{dsml}invoke>
</{dsml}function_calls>

String and scalar parameters should be specified as is without any escaping or quotes, while lists and objects should use JSON format. The \"string\" attribute should be set to \"true\" for string type parameters and \"false\" for other types (numbers, booleans, arrays, objects).

If the thinking_mode is enabled, then after function results you should strongly consider outputting a thinking block. Here is an example:

<{dsml}function_calls>
...
</{dsml}function_calls>

<function_results>
...
</function_results>

{tstart}...thinking about results{tend}

Here are the functions available in JSONSchema format:
<functions>
{tool_schemas}
</functions>
"
    )
}

fn response_format_block(schema: &str) -> String {
    format!(
        "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}",
    )
}

fn user_msg(content: &str) -> String {
    format!("{USER}{content}{ASSISTANT}")
}

/// Mirrors `encode_arguments_to_dsml`. `tool_call["arguments"]` is a JSON
/// *string* in OpenAI schema; the Python code does `json.loads(...)` and
/// iterates over the resulting dict.
///
/// It may also arrive as an already-parsed object: `model_gateway`'s
/// `process_tool_call_arguments` converts the string into a dict before the
/// chat template runs. Reading only `as_str()` silently dropped object-form
/// args, which erased every historical tool call's parameters in multi-turn
/// prompts.
fn encode_arguments_to_dsml(tool_call: &Value) -> Result<String, DsEncodingError> {
    let arguments: Value = match tool_call.get("arguments") {
        Some(Value::String(s)) => {
            serde_json::from_str(s).map_err(DsEncodingError::InvalidToolArgumentsJson)?
        }
        Some(v) if v.is_object() => v.clone(),
        _ => Value::Object(serde_json::Map::new()),
    };

    let obj = match arguments.as_object() {
        Some(obj) => obj,
        // Non-object payload — render nothing, matching Python behaviour
        // (`for k, v in arguments.items()` would raise; we tolerate it).
        None => return Ok(String::new()),
    };

    let mut parts = Vec::with_capacity(obj.len());
    for (k, v) in obj {
        let (is_str, value_str) = match v {
            Value::String(s) => ("true", s.clone()),
            other => ("false", to_json(other)),
        };
        parts.push(format!(
            "<{DSML}parameter name=\"{k}\" string=\"{is_str}\">{value_str}</{DSML}parameter>",
        ));
    }
    Ok(parts.join("\n"))
}

fn render_tools(tools: &[Value]) -> String {
    render_tools_template(&tool_schemas(tools))
}

/// Returns `true` when `index == last_user_idx`.
fn equals_last_user(index: usize, last_user_idx: Option<usize>) -> bool {
    last_user_idx == Some(index)
}

// ---------------------------------------------------------------------------
// render_message — direct port of the Python function with the same name:
// the message read into its parts, one function per role for the arms of its
// `match role`.
// ---------------------------------------------------------------------------

fn render_message(
    index: usize,
    messages: &[Value],
    thinking_mode: ThinkingMode,
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

    let tools_owned = tools_raw.map(|t| tools_from_openai_format(t));
    let tools = tools_owned.as_deref();

    let tool_calls_owned = tool_calls_raw.map(|tc| tool_calls_from_openai_format(tc));
    let tool_calls = tool_calls_owned.as_deref();

    // The thinking token after a user or developer message opens the thinking
    // at the last user turn in thinking mode and closes it everywhere else.
    let opens_thinking = equals_last_user(index, last_user_idx) && thinking_mode.is_thinking();

    match role {
        "system" => render_system(&mut prompt, content, tools, response_format),
        "developer" => {
            if content.is_empty() {
                return Err(DsEncodingError::InvalidMessage {
                    role: role.to_string(),
                    msg: msg.to_string(),
                });
            }
            render_developer(&mut prompt, content, tools, response_format, opens_thinking);
        }
        "user" => render_user(&mut prompt, content, opens_thinking),
        "tool" => {
            let position = tool_result_position(index, messages)?;
            // The last result of a block opens the thinking from the last user
            // turn on in thinking mode (not the user message's rule above).
            let block_opens_thinking =
                at_or_after_last_user(index, last_user_idx) && thinking_mode.is_thinking();
            render_tool(&mut prompt, position, content, block_opens_thinking);
        }
        "assistant" => {
            let assistant = AssistantParts {
                message: msg,
                content,
                tool_calls,
                reasoning_content,
                emits_thinking: thinking_mode.is_thinking()
                    && after_last_user(index, last_user_idx),
            };
            render_assistant(&mut prompt, &assistant)?;
        }
        other => return Err(DsEncodingError::UnknownRole(other.to_string())),
    }

    Ok(prompt)
}

/// The thinking token that follows a user, developer or final tool message:
/// the opener when the turn thinks, the closer otherwise.
fn thinking_token(opens_thinking: bool) -> &'static str {
    if opens_thinking {
        THINKING_START
    } else {
        THINKING_END
    }
}

/// The tools and the response format a system or developer message carries,
/// each after a blank line.
fn render_tools_and_response_format(
    prompt: &mut String,
    tools: Option<&[Value]>,
    response_format: Option<&Value>,
) {
    if let Some(tools) = tools.filter(|t| !t.is_empty()) {
        prompt.push_str("\n\n");
        prompt.push_str(&render_tools(tools));
    }
    if let Some(rf) = response_format {
        prompt.push_str("\n\n");
        prompt.push_str(&response_format_block(&to_json(rf)));
    }
}

/// A system message: its content (`system_msg_template` is "{content}"),
/// then the tools and the response format it carries.
fn render_system(
    prompt: &mut String,
    content: &str,
    tools: Option<&[Value]>,
    response_format: Option<&Value>,
) {
    prompt.push_str(content);
    render_tools_and_response_format(prompt, tools, response_format);
}

/// A developer message: a user message made of the tools and the response
/// format it carries and its content under a heading, then the thinking
/// token (the content is never empty: the caller refused an empty one).
fn render_developer(
    prompt: &mut String,
    content: &str,
    tools: Option<&[Value]>,
    response_format: Option<&Value>,
    opens_thinking: bool,
) {
    let mut content_developer = String::new();
    render_tools_and_response_format(&mut content_developer, tools, response_format);
    let _ = write!(content_developer, "\n\n# The user's message is: {content}");

    prompt.push_str(&user_msg(&content_developer));
    prompt.push_str(thinking_token(opens_thinking));
}

/// A user message: its content between the user and assistant tokens, then
/// the thinking token.
fn render_user(prompt: &mut String, content: &str, opens_thinking: bool) {
    prompt.push_str(&user_msg(content));
    prompt.push_str(thinking_token(opens_thinking));
}

/// Where a tool message stands in its block of results.
struct ToolResultPosition {
    /// The result's order among the originating assistant turn's tool calls,
    /// from one.
    order: usize,
    /// How many tool calls that turn made.
    count: usize,
}

/// Walks back over the consecutive tool messages before `index` to the
/// assistant turn that made the calls (the same logic as the Python) and
/// places the message among that turn's calls; a block with no assistant
/// turn before it, or with more results than calls, is refused.
fn tool_result_position(
    index: usize,
    messages: &[Value],
) -> Result<ToolResultPosition, DsEncodingError> {
    let mut prev_assistant_idx: isize = index as isize - 1;
    while prev_assistant_idx >= 0
        && messages[prev_assistant_idx as usize]
            .get("role")
            .and_then(|v| v.as_str())
            == Some("tool")
    {
        prev_assistant_idx -= 1;
    }

    let assistant_role = if prev_assistant_idx >= 0 {
        messages[prev_assistant_idx as usize]
            .get("role")
            .and_then(|v| v.as_str())
    } else {
        None
    };

    let valid_anchor =
        index == 0 || (prev_assistant_idx >= 0 && assistant_role == Some("assistant"));
    if !valid_anchor {
        let anchor_idx = prev_assistant_idx.max(0) as usize;
        return Err(DsEncodingError::InvalidToolMessages {
            index,
            context: messages[anchor_idx].to_string(),
        });
    }

    let assistant_tool_calls = if prev_assistant_idx >= 0 {
        messages[prev_assistant_idx as usize]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0)
    } else {
        0
    };

    let tool_call_order = (index as isize - prev_assistant_idx) as usize;
    if assistant_tool_calls == 0 || assistant_tool_calls < tool_call_order {
        return Err(DsEncodingError::NoToolCalls);
    }

    Ok(ToolResultPosition {
        order: tool_call_order,
        count: assistant_tool_calls,
    })
}

/// A tool message: the block's opener before its first result, the result
/// (`tool_output_template` is "\n<result>{content}</result>"), and after the
/// block's last result its closer and the thinking token.
fn render_tool(
    prompt: &mut String,
    position: ToolResultPosition,
    content: &str,
    opens_thinking: bool,
) {
    if position.order == 1 {
        prompt.push_str("\n\n<function_results>");
    }

    let _ = write!(prompt, "\n<result>{content}</result>");

    if position.order == position.count {
        prompt.push_str("\n</function_results>");
        prompt.push_str("\n\n");
        prompt.push_str(thinking_token(opens_thinking));
    }
}

/// What an assistant message renders from: its content, its tool calls, its
/// reasoning, and whether the turn's thinking goes out (thinking mode, after
/// the last user message); the message itself names a refused one.
struct AssistantParts<'a> {
    message: &'a Value,
    content: &'a str,
    tool_calls: Option<&'a [Value]>,
    reasoning_content: &'a str,
    emits_thinking: bool,
}

/// An assistant message (`assistant_msg_template` is
/// "{reasoning}{content}{tool_calls}<｜end▁of▁sentence｜>"): its reasoning and
/// the thinking end token when the turn's thinking goes out (a turn with
/// neither reasoning nor tool calls is refused then), its content, its tool
/// calls as a DSML block, and the EOS token.
fn render_assistant(
    prompt: &mut String,
    parts: &AssistantParts<'_>,
) -> Result<(), DsEncodingError> {
    let mut thinking_part = String::new();

    let mut tool_calls_content = String::new();
    if let Some(tcs) = parts.tool_calls.filter(|t| !t.is_empty()) {
        let mut rendered = Vec::with_capacity(tcs.len());
        for tc in tcs {
            let name = tc.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = encode_arguments_to_dsml(tc)?;
            rendered.push(format!(
                "<{DSML}invoke name=\"{name}\">\n{args}\n</{DSML}invoke>",
            ));
        }
        let joined = rendered.join("\n");
        let _ = write!(
            tool_calls_content,
            "\n\n<{DSML}function_calls>\n{joined}\n</{DSML}function_calls>"
        );
    }

    let summary_content = parts.content;

    if parts.emits_thinking {
        let has_reasoning = !parts.reasoning_content.is_empty();
        let has_tool_calls = parts.tool_calls.is_some_and(|t| !t.is_empty());
        if !has_reasoning && !has_tool_calls {
            return Err(DsEncodingError::MissingReasoningOrToolCalls(
                parts.message.to_string(),
            ));
        }
        thinking_part.push_str(parts.reasoning_content);
        thinking_part.push_str(THINKING_END);
    }

    prompt.push_str(&thinking_part);
    prompt.push_str(summary_content);
    prompt.push_str(&tool_calls_content);
    prompt.push_str(EOS);
    Ok(())
}

// ---------------------------------------------------------------------------
// drop_thinking_messages
// ---------------------------------------------------------------------------

fn drop_thinking_messages(messages: &[Value]) -> Vec<Value> {
    let last_user_idx = find_last_user_index(messages);
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());

    for (idx, msg) in messages.iter().enumerate() {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let always_keep =
            matches!(role, "user" | "system" | "tool") || at_or_after_last_user(idx, last_user_idx);

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
        // Other roles before last_user_idx are dropped, matching Python.
    }

    out
}

// ---------------------------------------------------------------------------
// encode_messages — public entry point
// ---------------------------------------------------------------------------

/// Encode a list of OpenAI-style messages into a DeepSeek V3.2 prompt string.
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
    render_messages(messages, thinking_mode, params)
        .map(|marked| marks::strip(&marked).into_owned())
}

/// [`encode_messages`] with the renderer's own control tokens marked (see
/// `marks`): the form the tokenizer wrapper encodes by segments, so that a
/// control-token spelling typed in a message stays text. Request strings
/// lose any mark a client wrote into them first.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the same signature as `encode_messages`"
)]
pub(crate) fn render_messages(
    messages: &[Value],
    thinking_mode: ThinkingMode,
    params: &EncodeParams,
) -> Result<String, DsEncodingError> {
    let messages = marks::without_marks(messages);
    // The engine's renderer rewrites a `developer` message as a `system` one
    // (and merges the system messages when one is not first) before any
    // tokenizer mode's own encoder sees the conversation, so the vendor
    // encoding's developer branch is never reached there; the same rule here.
    let mut full_messages: Vec<Value> =
        crate::chat_template::developer_messages_as_system(&messages)
            .unwrap_or_else(|| messages.into_owned());

    let mut prompt = if params.add_default_bos_token {
        BOS.to_string()
    } else {
        String::new()
    };

    if thinking_mode.is_thinking() && params.drop_thinking {
        full_messages = drop_thinking_messages(&full_messages);
    }

    for idx in 0..full_messages.len() {
        prompt.push_str(&render_message(idx, &full_messages, thinking_mode)?);
    }

    Ok(prompt)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        super::deepseek_common::{ASSISTANT_SP_TOKEN, USER_SP_TOKEN},
        *,
    };

    fn user(text: &str) -> Value {
        json!({ "role": "user", "content": text })
    }

    fn assistant_with_reasoning(reasoning: &str, content: &str) -> Value {
        json!({
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": content,
        })
    }

    #[test]
    fn one_turn_user_chat_mode_closes_think() {
        let msgs = [user("Hello")];
        let out = encode_messages(&msgs, ThinkingMode::Chat, &EncodeParams::default()).unwrap();

        let expected =
            format!("{BOS_TOKEN}{USER_SP_TOKEN}Hello{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}",);
        assert_eq!(out, expected);
    }

    #[test]
    fn one_turn_user_thinking_mode_opens_think() {
        let msgs = [user("Hello")];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();

        let expected =
            format!("{BOS_TOKEN}{USER_SP_TOKEN}Hello{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}",);
        assert_eq!(out, expected);
    }

    #[test]
    fn drop_thinking_strips_earlier_reasoning() {
        // Three-turn conversation: the assistant turn at index 1 carries
        // reasoning_content. The Python encoder only emits reasoning on
        // assistant messages strictly *after* the last user turn, so an
        // earlier assistant turn never leaks reasoning regardless of the
        // drop_thinking flag. The flag controls whether the field is
        // *retained on the message dict* — verify the rendered prompt is
        // unchanged in either direction.
        let msgs = [
            user("Q1"),
            assistant_with_reasoning("private thought", "A1"),
            user("Q2"),
        ];

        let out_drop =
            encode_messages(&msgs, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();
        assert!(!out_drop.contains("private thought"));

        let params = EncodeParams {
            drop_thinking: false,
            ..EncodeParams::default()
        };
        let out_keep = encode_messages(&msgs, ThinkingMode::Thinking, &params).unwrap();
        assert!(!out_keep.contains("private thought"));

        // Sanity: the most-recent assistant turn (after the last user) DOES
        // emit reasoning_content, proving the dropper acted only on the
        // earlier turn.
        let msgs2 = [user("Q1"), assistant_with_reasoning("recent thought", "A1")];
        let out_recent =
            encode_messages(&msgs2, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();
        assert!(out_recent.contains("recent thought"));
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

        assert!(out.contains(&format!("<{DSML_TOKEN}function_calls>")));
        assert!(out.contains(&format!("<{DSML_TOKEN}invoke name=\"search\">")));
        assert!(out.contains(&format!(
            "<{DSML_TOKEN}parameter name=\"query\" string=\"true\">deepseek</{DSML_TOKEN}parameter>"
        )));
        assert!(out.contains(&format!(
            "<{DSML_TOKEN}parameter name=\"limit\" string=\"false\">5</{DSML_TOKEN}parameter>"
        )));
        assert!(out.contains(&format!("</{DSML_TOKEN}function_calls>")));
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

    #[test]
    fn skip_bos_when_disabled() {
        let msgs = [user("Hi")];
        let params = EncodeParams {
            add_default_bos_token: false,
            ..EncodeParams::default()
        };
        let out = encode_messages(&msgs, ThinkingMode::Chat, &params).unwrap();
        assert!(!out.starts_with(BOS_TOKEN));
        assert!(out.starts_with(USER_SP_TOKEN));
    }

    #[test]
    fn drop_thinking_does_not_overrun_when_filtering_shrinks_messages() {
        // `drop_thinking_messages` can leave `full_messages` shorter than
        // `messages` (the roles it drops before the last user turn), so the
        // outer loop must iterate the filtered length, not the original, or
        // it walks off the end and returns IndexOutOfRange. A developer
        // message is a system message by the time it gets there and stays.
        let msgs = [
            json!({ "role": "developer", "content": "earlier developer note" }),
            json!({ "role": "user", "content": "now" }),
        ];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &EncodeParams::default())
            .expect("filtered message length must not blow up the loop");
        assert!(
            out.contains("now") && out.contains("earlier developer note"),
            "messages missing from prompt: {out}"
        );
    }

    /// The engine's renderer hands every tokenizer mode a `developer` message
    /// as a `system` one: the recorded reference of the developer-role case
    /// (the content as the system message, no user token before it), the same
    /// as a system message in both modes.
    #[test]
    fn developer_message_is_the_system_message_as_the_engine_hands_it() {
        let msgs = [
            json!({ "role": "developer", "content": "Answer in one word." }),
            json!({ "role": "user", "content": "Colour of the sky?" }),
        ];
        let out = encode_messages(&msgs, ThinkingMode::Chat, &EncodeParams::default()).unwrap();
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜>Answer in one word.<｜User｜>Colour of the sky?<｜Assistant｜></think>"
        );
        let as_system = [
            json!({ "role": "system", "content": "Answer in one word." }),
            json!({ "role": "user", "content": "Colour of the sky?" }),
        ];
        for mode in [ThinkingMode::Chat, ThinkingMode::Thinking] {
            assert_eq!(
                encode_messages(&msgs, mode, &EncodeParams::default()).unwrap(),
                encode_messages(&as_system, mode, &EncodeParams::default()).unwrap()
            );
        }
    }
}
