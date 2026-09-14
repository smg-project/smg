// Ported from https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py

use std::fmt::Write as _;

use serde_json::{json, Value};
use thiserror::Error;

// `find_last_user_index` and `at_or_after_last_user` are byte-for-byte
// identical to the V4-family copies; shared via `deepseek_common` rather than
// duplicated a third time. V3.2's other tool/drop-thinking helpers are
// genuinely different algorithms (not just a naming coincidence) and are
// intentionally kept local to this file.
use super::deepseek_common::{at_or_after_last_user, find_last_user_index};

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
// Special-token constants — copied verbatim from the Python source.
// ---------------------------------------------------------------------------

pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub const THINKING_START_TOKEN: &str = "<think>";
pub const THINKING_END_TOKEN: &str = "</think>";
pub const DSML_TOKEN: &str = "｜DSML｜";

const USER_PREFIX: &str = "<｜User｜>";
const ASSISTANT_SUFFIX: &str = "<｜Assistant｜>";

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

/// Mirrors `TOOLS_SYSTEM_TEMPLATE` from the Python source, including the
/// blank lines between paragraphs that the upstream template preserves.
fn render_tools_template(tool_schemas: &str) -> String {
    let dsml = DSML_TOKEN;
    let tstart = THINKING_START_TOKEN;
    let tend = THINKING_END_TOKEN;
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
    format!("{USER_PREFIX}{content}{ASSISTANT_SUFFIX}")
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Mirrors the Python `to_json` helper: `json.dumps(value, ensure_ascii=False)`,
/// which uses spaced `", "` / `": "` separators. Compact `serde_json::to_string`
/// would change the prompt bytes vLLM renders from.
fn to_json(value: &Value) -> String {
    crate::json_dumps::to_string(value)
}

/// `[tool["function"] for tool in tools]`
fn tools_from_openai_format(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|t| t.get("function").cloned())
        .collect()
}

/// `[{ "name": tc["function"]["name"], "arguments": tc["function"]["arguments"] } for tc in tool_calls]`
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
            "<{DSML_TOKEN}parameter name=\"{k}\" string=\"{is_str}\">{value_str}</{DSML_TOKEN}parameter>",
        ));
    }
    Ok(parts.join("\n"))
}

fn render_tools(tools: &[Value]) -> String {
    let schemas: Vec<String> = tools.iter().map(to_json).collect();
    render_tools_template(&schemas.join("\n"))
}

/// Returns `true` when `index > last_user_idx` in the Python sense.
fn after_last_user(index: usize, last_user_idx: Option<usize>) -> bool {
    match last_user_idx {
        Some(idx) => index > idx,
        None => true,
    }
}

/// Returns `true` when `index == last_user_idx`.
fn equals_last_user(index: usize, last_user_idx: Option<usize>) -> bool {
    last_user_idx == Some(index)
}

// ---------------------------------------------------------------------------
// render_message — direct port of the Python function with the same name.
// ---------------------------------------------------------------------------

#[expect(
    clippy::too_many_lines,
    reason = "mirrors the Python render_message function 1:1 for sync-ability"
)]
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

    match role {
        "system" => {
            // system_msg_template is "{content}"
            prompt.push_str(content);
            if let Some(tools) = tools.filter(|t| !t.is_empty()) {
                prompt.push_str("\n\n");
                prompt.push_str(&render_tools(tools));
            }
            if let Some(rf) = response_format {
                prompt.push_str("\n\n");
                prompt.push_str(&response_format_block(&to_json(rf)));
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
            if let Some(tools) = tools.filter(|t| !t.is_empty()) {
                content_developer.push_str("\n\n");
                content_developer.push_str(&render_tools(tools));
            }
            if let Some(rf) = response_format {
                content_developer.push_str("\n\n");
                content_developer.push_str(&response_format_block(&to_json(rf)));
            }
            let _ = write!(content_developer, "\n\n# The user's message is: {content}");

            prompt.push_str(&user_msg(&content_developer));

            if equals_last_user(index, last_user_idx) && thinking_mode.is_thinking() {
                prompt.push_str(THINKING_START_TOKEN);
            } else {
                prompt.push_str(THINKING_END_TOKEN);
            }
        }
        "user" => {
            prompt.push_str(&user_msg(content));
            if equals_last_user(index, last_user_idx) && thinking_mode.is_thinking() {
                prompt.push_str(THINKING_START_TOKEN);
            } else {
                prompt.push_str(THINKING_END_TOKEN);
            }
        }
        "tool" => {
            // Walk back over consecutive tool messages to find the originating
            // assistant turn — same logic as Python.
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

            if tool_call_order == 1 {
                prompt.push_str("\n\n<function_results>");
            }

            // tool_output_template = "\n<result>{content}</result>"
            let _ = write!(prompt, "\n<result>{content}</result>");

            if tool_call_order == assistant_tool_calls {
                prompt.push_str("\n</function_results>");

                if at_or_after_last_user(index, last_user_idx) && thinking_mode.is_thinking() {
                    prompt.push_str("\n\n");
                    prompt.push_str(THINKING_START_TOKEN);
                } else {
                    prompt.push_str("\n\n");
                    prompt.push_str(THINKING_END_TOKEN);
                }
            }
        }
        "assistant" => {
            let mut thinking_part = String::new();

            let mut tool_calls_content = String::new();
            if let Some(tcs) = tool_calls.filter(|t| !t.is_empty()) {
                let mut rendered = Vec::with_capacity(tcs.len());
                for tc in tcs {
                    let name = tc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args = encode_arguments_to_dsml(tc)?;
                    rendered.push(format!(
                        "<{DSML_TOKEN}invoke name=\"{name}\">\n{args}\n</{DSML_TOKEN}invoke>",
                    ));
                }
                let joined = rendered.join("\n");
                let _ = write!(
                    tool_calls_content,
                    "\n\n<{DSML_TOKEN}function_calls>\n{joined}\n</{DSML_TOKEN}function_calls>"
                );
            }

            let summary_content = content;

            if thinking_mode.is_thinking() && after_last_user(index, last_user_idx) {
                let has_reasoning = !reasoning_content.is_empty();
                let has_tool_calls = tool_calls.is_some_and(|t| !t.is_empty());
                if !has_reasoning && !has_tool_calls {
                    return Err(DsEncodingError::MissingReasoningOrToolCalls(
                        msg.to_string(),
                    ));
                }
                thinking_part.push_str(reasoning_content);
                thinking_part.push_str(THINKING_END_TOKEN);
            }

            // assistant_msg_template = "{reasoning}{content}{tool_calls}<｜end▁of▁sentence｜>"
            prompt.push_str(&thinking_part);
            prompt.push_str(summary_content);
            prompt.push_str(&tool_calls_content);
            prompt.push_str(EOS_TOKEN);
        }
        other => return Err(DsEncodingError::UnknownRole(other.to_string())),
    }

    Ok(prompt)
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
    let mut full_messages: Vec<Value> = messages.to_vec();

    let mut prompt = if params.add_default_bos_token {
        BOS_TOKEN.to_string()
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

    use super::*;

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
            format!("{BOS_TOKEN}{USER_PREFIX}Hello{ASSISTANT_SUFFIX}{THINKING_END_TOKEN}",);
        assert_eq!(out, expected);
    }

    #[test]
    fn one_turn_user_thinking_mode_opens_think() {
        let msgs = [user("Hello")];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();

        let expected =
            format!("{BOS_TOKEN}{USER_PREFIX}Hello{ASSISTANT_SUFFIX}{THINKING_START_TOKEN}",);
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
        assert!(out.starts_with(USER_PREFIX));
    }

    #[test]
    fn drop_thinking_does_not_overrun_when_filtering_shrinks_messages() {
        // `drop_thinking_messages` removes developer-role messages before the
        // last user turn, so `full_messages.len() < messages.len()`. The
        // outer loop must iterate the filtered length, not the original, or
        // it walks off the end and returns IndexOutOfRange.
        let msgs = [
            json!({ "role": "developer", "content": "earlier developer note" }),
            json!({ "role": "user", "content": "now" }),
        ];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &EncodeParams::default())
            .expect("filtered message length must not blow up the loop");
        assert!(
            out.contains("now"),
            "user message missing from prompt: {out}"
        );
    }
}
