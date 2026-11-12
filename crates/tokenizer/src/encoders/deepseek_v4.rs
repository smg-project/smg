// Ported from the `encoding/encoding_dsv4.py` shipped with the DeepSeek V4
// checkpoints (deepseek-ai/DeepSeek-V4-Flash and -Flash-0731 on HF). The two
// revisions differ only in the reasoning-effort block, and the serving engine
// renders every V4 checkpoint with its own copy of the encoder, which carries
// the refreshed table: that table is the one here, whatever the checkpoint's
// own file says (the engine never reads it).

use std::fmt::Write as _;

use serde_json::Value;

// `DsEncodingError` lives in `deepseek_common` (one type for the three
// renderers) and is re-exported here unchanged: it is part of this module's
// public API (returned from `encode_messages`), and `pub(super)` items are
// only reachable within `encoders` on their own.
pub use super::deepseek_common::DsEncodingError;
// Reuse the public ThinkingMode enum from the V3.2 module to keep the
// "thinking" / "chat" mode invariant identical across DeepSeek versions.
pub use super::deepseek_v32::ThinkingMode;
// The message read, message-shape preprocessing, drop-thinking, DSML argument
// rendering, the tools template and the tools-and-response-format section are
// the V4.1 renderer's too, apart from the tag strings; see `deepseek_common`
// for the shared implementation and its own doc comments.
use super::{
    deepseek_common::{
        after_last_user, at_or_after_last_user, drop_thinking_messages, encode_arguments_to_dsml,
        find_last_user_index, merge_tool_messages, push_tools_and_response_format,
        render_system_message, render_v4_tools_template, sort_tool_results_by_call_order,
        task_sp_token, to_json, tool_schemas, DsmlTags, IncomingBlocks, Turn,
    },
    RendererProfile,
};
use crate::{
    chat_template::{ThinkingKeyName, ThinkingToggle},
    traits::RendererCapabilities,
};

/// Native effort names the engine's copy of the encoder accepts (the keys of
/// its effort table).
const NATIVE_EFFORT_VALUES: &[&str] = &["low", "high", "max"];

/// What this renderer answers around a render (see [`RendererProfile`]).
pub(crate) const PROFILE: RendererProfile = RendererProfile {
    // No opinion on content parts: the (usually absent) template's format.
    content_format: None,
    // Thinking defaults ON like the engine's own server: `thinking: false`,
    // its `enable_thinking` alias or `reasoning_effort: "none"` turns it off.
    thinking_toggle: ThinkingToggle::DefaultOn,
    thinking_key_name: ThinkingKeyName::Thinking,
    native_effort_values: NATIVE_EFFORT_VALUES,
    // `reasoning_effort` is read as the engine's own server reads it:
    // `"none"` is chat mode, every other value (`"minimal"` included, which
    // the engine maps to the low effort) keeps thinking on. Declaring the
    // off word keeps the gateway's parser arming on that rule.
    native_effort_off_values: &["none"],
    // `<｜Assistant｜><think>` ends the prompt when thinking is on.
    think_in_prefill: true,
    // The shim reads the engine's `enable_thinking` alias too. The encoder
    // writes a system message after the first as bare text with no role
    // marker, and opens the assistant turn after it only when it ends the
    // conversation or an assistant turn follows (`render_turn_open`); on
    // the Messages path the gateway renders a system message sent after
    // the first as a user turn, which sets the text off from the turn
    // before it (a role marker of its own after an assistant turn, a blank
    // line inside a user turn); both forms take the header when the message
    // ends the conversation or an assistant turn follows.
    capabilities: RendererCapabilities {
        enable_thinking_alias: true,
        leading_system_only: true,
        ..RendererProfile::NO_CAPABILITIES
    },
    thinking_mode: None,
};

/// Reasoning effort for the V4 prompt prefix: the levels of the table the
/// serving engine's copy of the encoder renders (`low` writes no prefix,
/// `high` the text the checkpoint's original file reserved for `max`, `max`
/// the refreshed, stronger one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
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

// ---------------------------------------------------------------------------
// Special-token constants: the family's shared ones come from `deepseek_common`
// (re-exported under this renderer's path), this version's own follow. Copied
// verbatim from the Python source.
// ---------------------------------------------------------------------------
pub use super::deepseek_common::{
    BOS_TOKEN, DSML_TOKEN, EOS_TOKEN, THINKING_END_TOKEN, THINKING_START_TOKEN,
};
use super::{
    deepseek_common::{
        mark_grounding_tokens, ASSISTANT, BOS, DSML, EOS, LATEST_REMINDER, THINKING_END,
        THINKING_START, USER,
    },
    marks,
};
const TOOL_CALLS_BLOCK_NAME: &str = "tool_calls";
/// V4's DSML tag strings: bare `invoke`/`parameter`, no leading space. The
/// V4.1 renderer builds its own `DsmlTags` with a leading space on each and
/// passes it to the same shared [`encode_arguments_to_dsml`].
const DSML_TAGS: DsmlTags = DsmlTags {
    invoke: "invoke",
    parameter: "parameter",
};

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------
// Original `max` prefix; the 0731 refresh reuses this exact text for `high`.
const REASONING_EFFORT_ABSOLUTE_MAXIMUM: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
const REASONING_EFFORT_BEYOND_MAXIMUM: &str = "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

/// Mirrors V4's `TOOLS_TEMPLATE`: the V4 family's text with this dialect's
/// block name (`tool_calls`, not `function_calls` like V3.2) and bare tags.
fn render_tools(tools: &[Value]) -> String {
    render_v4_tools_template(TOOL_CALLS_BLOCK_NAME, &DSML_TAGS, &tool_schemas(tools))
}

// ---------------------------------------------------------------------------
// render_message — direct port of the V4 Python function with the same name:
// the message read into its parts (the family's `Turn`), one function per
// role for the arms of its `match role`, and the turn opener after it.
// ---------------------------------------------------------------------------
fn render_message(
    index: usize,
    messages: &[Value],
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    reasoning_effort: Option<ReasoningEffort>,
) -> Result<String, DsEncodingError> {
    let turn = Turn::read(index, messages)?;
    let last_user_idx = find_last_user_index(messages);
    let mut prompt = String::new();

    // Reasoning effort prefix: index 0, thinking mode only, the table the
    // engine's copy of the encoder renders for every V4 checkpoint.
    if index == 0 && thinking_mode == ThinkingMode::Thinking {
        match reasoning_effort {
            Some(ReasoningEffort::High) => prompt.push_str(REASONING_EFFORT_ABSOLUTE_MAXIMUM),
            Some(ReasoningEffort::Max) => prompt.push_str(REASONING_EFFORT_BEYOND_MAXIMUM),
            None | Some(ReasoningEffort::Low) => {}
        }
    }

    match turn.role {
        "system" => render_system_message(&mut prompt, &turn, render_tools),
        "developer" => render_developer(&mut prompt, &turn)?,
        "user" => render_user(&mut prompt, turn.msg, turn.content),
        "latest_reminder" => {
            prompt.push_str(LATEST_REMINDER);
            prompt.push_str(turn.content);
        }
        "tool" => {
            return Err(DsEncodingError::UnmergedToolRole(index));
        }
        "assistant" => {
            // prev_has_task: if previous message had a task, this is a task
            // output (no thinking).
            let prev_has_task = index >= 1
                && messages[index - 1].get("task").is_some()
                && !messages[index - 1]
                    .get("task")
                    .map(Value::is_null)
                    .unwrap_or(true);
            // The turn's thinking goes out: thinking mode, the previous
            // message gave no task (a task output has no thinking), and the
            // thinking is kept (not dropped, or after the last user message).
            let emits_thinking = thinking_mode == ThinkingMode::Thinking
                && !prev_has_task
                && (!drop_thinking || after_last_user(index, last_user_idx));
            render_assistant(&mut prompt, &turn, emits_thinking);
        }
        other => return Err(DsEncodingError::UnknownRole(other.to_string())),
    }

    render_turn_open(
        &mut prompt,
        messages,
        index,
        turn.role,
        thinking_mode,
        drop_thinking,
        last_user_idx,
    )?;
    Ok(prompt)
}

/// A developer message: the user token, its content, then the tools and
/// the response format it carries; an empty one is refused.
fn render_developer(prompt: &mut String, turn: &Turn<'_>) -> Result<(), DsEncodingError> {
    let content = turn.developer_content()?;
    let mut content_developer = String::new();
    content_developer.push_str(USER);
    content_developer.push_str(content);
    push_tools_and_response_format(
        &mut content_developer,
        turn.tools.as_deref(),
        turn.response_format,
        render_tools,
    );
    prompt.push_str(&content_developer);
    Ok(())
}

/// A user message: the user token, then its content blocks (tool results
/// mixed with text) joined, or its plain content.
fn render_user(prompt: &mut String, msg: &Value, content: &str) {
    prompt.push_str(USER);
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
                    let tool_content = render_tool_result_content(block.get("content"));
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

/// A tool result's content as text: its text parts joined (another part
/// kind named as unsupported), a string as it is, any other JSON as JSON,
/// nothing for none.
fn render_tool_result_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::Array(items)) => {
            let mut text_parts: Vec<String> = Vec::with_capacity(items.len());
            for b in items {
                let bt = b.get("type").and_then(|v| v.as_str()).unwrap_or("");
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
    }
}

/// An assistant message: its thinking (reasoning and the thinking end
/// token) when the turn keeps it (`emits_thinking`), its content, its tool
/// calls as a DSML block, and the EOS token unless the message says
/// otherwise.
fn render_assistant(prompt: &mut String, turn: &Turn<'_>, emits_thinking: bool) {
    let mut thinking_part = String::new();
    let mut tc_content = String::new();
    if let Some(tcs) = turn.tool_calls.as_deref() {
        let mut tc_list = Vec::with_capacity(tcs.len());
        for tc in tcs {
            let name = tc.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let arguments = tc.get("arguments").unwrap_or(&Value::Null);
            let args = encode_arguments_to_dsml(arguments, &DSML_TAGS);
            tc_list.push(format!(
                "<{DSML}{invoke} name=\"{name}\">\n{args}\n</{DSML}{invoke}>",
                invoke = DSML_TAGS.invoke,
            ));
        }
        let joined = tc_list.join("\n");
        let _ = write!(
            tc_content,
            "\n\n<{DSML}{TOOL_CALLS_BLOCK_NAME}>\n{joined}\n</{DSML}{TOOL_CALLS_BLOCK_NAME}>"
        );
    }
    if emits_thinking {
        thinking_part.push_str(&mark_grounding_tokens(turn.reasoning_content));
        thinking_part.push_str(THINKING_END);
    }
    prompt.push_str(&thinking_part);
    prompt.push_str(&mark_grounding_tokens(turn.content));
    prompt.push_str(&tc_content);
    if !turn.wo_eos() {
        prompt.push_str(EOS);
    }
}

/// The transition tokens after a message, by what follows it: nothing
/// before a message that is neither the assistant's nor a latest reminder;
/// a task's token (an action task after the assistant token and the
/// thinking token); after a user or developer message, or a system message
/// that takes the header ([`system_opens_the_turn`]), the assistant token
/// and the thinking token that opens or closes the thinking.
fn render_turn_open(
    prompt: &mut String,
    messages: &[Value],
    index: usize,
    role: &str,
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    last_user_idx: Option<usize>,
) -> Result<(), DsEncodingError> {
    // Append transition tokens based on what follows.
    if let Some(next) = messages.get(index + 1) {
        let next_role = next.get("role").and_then(|v| v.as_str()).unwrap_or("");
        if !matches!(next_role, "assistant" | "latest_reminder") {
            return Ok(());
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
            prompt.push_str(ASSISTANT);
            prompt.push_str(if thinking_mode == ThinkingMode::Thinking {
                THINKING_START
            } else {
                THINKING_END
            });
            prompt.push_str(sp_token);
        } else {
            // Non-action tasks: append task sp token directly after the message.
            prompt.push_str(sp_token);
        }
    } else if matches!(role, "user" | "developer") || system_opens_the_turn(messages, index, role) {
        // Normal generation: append Assistant + thinking token.
        prompt.push_str(ASSISTANT);
        let opens_thinking = thinking_mode == ThinkingMode::Thinking
            && (!drop_thinking || at_or_after_last_user(index, last_user_idx));
        if opens_thinking {
            prompt.push_str(THINKING_START);
        } else {
            prompt.push_str(THINKING_END);
        }
    }
    Ok(())
}

/// Whether a system message takes the assistant header, as the serving
/// engine's copy of the encoder renders it: a trailing system message opens
/// generation, and one an assistant turn follows opens that history turn;
/// one a user turn, a tool result or a latest reminder follows stays bare
/// text (the checkpoint's own encoder file opens the turn after user and
/// developer turns only).
fn system_opens_the_turn(messages: &[Value], index: usize, role: &str) -> bool {
    role == "system"
        && messages
            .get(index + 1)
            .is_none_or(|next| next.get("role").and_then(Value::as_str) == Some("assistant"))
}

// ---------------------------------------------------------------------------
// encode_messages — public entry point
// ---------------------------------------------------------------------------
// Preprocessing (merge tool messages, sort tool results, drop-thinking) now
// lives in `deepseek_common`, shared with the upcoming V4.1 renderer; see
// that module for `merge_tool_messages`, `sort_tool_results_by_call_order`
// and `drop_thinking_messages`, all used unchanged below.

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
    // The engine's renderer hands the conversation to this encoder with a
    // `developer` message as written (its developer-to-system rewrite belongs
    // to its template path, which this encoder never takes), so the vendor
    // encoding's own rules apply: dropped before the last user turn with the
    // thinking, a user turn otherwise.
    let messages = marks::without_marks(messages);
    // Preprocess: merge tool messages and sort tool results.
    let merged = merge_tool_messages(&messages, IncomingBlocks::Rebuilt);
    let mut full_messages = sort_tool_results_by_call_order(merged);
    let mut prompt = if params.add_default_bos_token {
        BOS.to_string()
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
    use serde_json::json;

    use super::{
        super::deepseek_common::{
            object_arguments_render_one_parameter_per_key_through, ASSISTANT_SP_TOKEN,
            LATEST_REMINDER_SP_TOKEN, USER_SP_TOKEN,
        },
        *,
    };

    // The task tokens as the recorded prompts spell them, independent of
    // `task_sp_token`.
    const TASK_ACTION: &str = "<｜action｜>";
    const TASK_QUERY: &str = "<｜query｜>";
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
    fn effort_levels_render_the_engines_table() {
        // The table the serving engine's copy of the encoder renders for
        // every V4 checkpoint: `low` and no effort write nothing, `high` the
        // text the checkpoint's original file reserved for `max`, `max` the
        // refreshed, stronger one; nothing outside thinking mode.
        let msgs = [user("Hello")];
        for (effort, expected_prefix) in [
            (None, None),
            (Some(ReasoningEffort::Low), None),
            (
                Some(ReasoningEffort::High),
                Some(REASONING_EFFORT_ABSOLUTE_MAXIMUM),
            ),
            (
                Some(ReasoningEffort::Max),
                Some(REASONING_EFFORT_BEYOND_MAXIMUM),
            ),
        ] {
            let params = EncodeParams {
                reasoning_effort: effort,
                ..EncodeParams::default()
            };
            let out = encode_messages(&msgs, ThinkingMode::Thinking, &params).unwrap();
            match expected_prefix {
                Some(prefix) => assert!(
                    out.starts_with(&format!("{BOS_TOKEN}{prefix}")),
                    "effort {effort:?}: {:?}",
                    &out[..120.min(out.len())]
                ),
                None => assert!(!out.contains("Reasoning Effort"), "effort {effort:?}"),
            }
            let out_chat = encode_messages(&msgs, ThinkingMode::Chat, &params).unwrap();
            assert!(
                !out_chat.contains("Reasoning Effort"),
                "effort {effort:?} outside thinking mode"
            );
        }
        assert_eq!(PROFILE.native_effort_values, &["low", "high", "max"]);
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
        object_arguments_render_one_parameter_per_key_through(
            |msgs| encode_messages(msgs, ThinkingMode::Chat, &EncodeParams::default()),
            "parameter",
        );
    }
    /// A system message after the first renders as the serving engine's copy
    /// of the encoder renders it: bare text with no role marker, and the
    /// assistant header after it when it ends the conversation or an
    /// assistant turn follows. The same messages as user turns, what the
    /// gateway's Messages path renders for a `leading_system_only` renderer,
    /// are set off from the turn before them by a blank line and take the
    /// same headers.
    #[test]
    fn a_system_message_after_the_first_opens_the_turn_it_ends_or_an_assistant_turn_follows() {
        let system = |text: &str| json!({ "role": "system", "content": text });
        let assistant = json!({ "role": "assistant", "content": "A" });
        let in_place = [
            system("S"),
            user("Task"),
            system("Env"),
            assistant.clone(),
            user("Q2"),
            system("Note"),
        ];
        let out =
            encode_messages(&in_place, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();
        assert_eq!(
            out,
            format!(
                "{BOS_TOKEN}S{USER_SP_TOKEN}TaskEnv{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}A{EOS_TOKEN}\
                 {USER_SP_TOKEN}Q2Note{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
            )
        );

        let as_user_turns = [
            system("S"),
            user("Task"),
            user("Env"),
            assistant,
            user("Q2"),
            user("Note"),
        ];
        let out = encode_messages(
            &as_user_turns,
            ThinkingMode::Thinking,
            &EncodeParams::default(),
        )
        .unwrap();
        assert_eq!(
            out,
            format!(
                "{BOS_TOKEN}S{USER_SP_TOKEN}Task\n\nEnv{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}A{EOS_TOKEN}\
                 {USER_SP_TOKEN}Q2\n\nNote{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
            )
        );
    }

    /// A system message that ends the conversation opens the assistant
    /// turn in both modes, the thinking token after the header as after a
    /// user turn; the checkpoint's own encoder file leaves the prompt
    /// without the header, the serving engine's copy writes it.
    #[test]
    fn a_trailing_system_message_opens_the_assistant_turn() {
        let msgs = [
            json!({ "role": "system", "content": "You are a terse assistant." }),
            user("Hello."),
            json!({ "role": "system", "content": "Answer in French." }),
        ];
        for (mode, think) in [
            (ThinkingMode::Thinking, THINKING_START_TOKEN),
            (ThinkingMode::Chat, THINKING_END_TOKEN),
        ] {
            let out = encode_messages(&msgs, mode, &EncodeParams::default()).unwrap();
            assert_eq!(
                out,
                format!(
                    "{BOS_TOKEN}You are a terse assistant.{USER_SP_TOKEN}Hello.Answer in French.\
                     {ASSISTANT_SP_TOKEN}{think}"
                )
            );
        }
    }

    /// The neighbours of the trailing case: a system message a user turn
    /// follows stays bare text with no header, one an assistant turn
    /// follows opens that history turn (closed thinking before the last
    /// user turn under the default `drop_thinking`).
    #[test]
    fn a_system_message_inside_the_conversation_takes_the_header_before_an_assistant_turn_only() {
        let note = json!({ "role": "system", "content": "Answer in French." });
        let before_user = [user("Hello."), note.clone(), user("How are you?")];
        let out = encode_messages(
            &before_user,
            ThinkingMode::Thinking,
            &EncodeParams::default(),
        )
        .unwrap();
        assert_eq!(
            out,
            format!(
                "{BOS_TOKEN}{USER_SP_TOKEN}Hello.Answer in French.{USER_SP_TOKEN}How are you?\
                 {ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
            )
        );

        let before_assistant = [
            user("Hello."),
            note,
            json!({ "role": "assistant", "content": "Bonjour." }),
            user("How are you?"),
        ];
        let out = encode_messages(
            &before_assistant,
            ThinkingMode::Thinking,
            &EncodeParams::default(),
        )
        .unwrap();
        assert_eq!(
            out,
            format!(
                "{BOS_TOKEN}{USER_SP_TOKEN}Hello.Answer in French.{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}\
                 Bonjour.{EOS_TOKEN}{USER_SP_TOKEN}How are you?{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
            )
        );
    }

    /// The first system message takes the header before a history assistant
    /// turn too: the serving engine's copy of the encoder has no index rule
    /// (the family's newer encoder leaves a first system message bare there).
    #[test]
    fn the_first_system_message_takes_the_header_before_an_assistant_turn() {
        let msgs = [
            json!({ "role": "system", "content": "S" }),
            json!({ "role": "assistant", "content": "Hi!" }),
            user("Q"),
        ];
        for (mode, think) in [
            (ThinkingMode::Thinking, THINKING_START_TOKEN),
            (ThinkingMode::Chat, THINKING_END_TOKEN),
        ] {
            let out = encode_messages(&msgs, mode, &EncodeParams::default()).unwrap();
            assert_eq!(
                out,
                format!(
                    "{BOS_TOKEN}S{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}Hi!{EOS_TOKEN}\
                     {USER_SP_TOKEN}Q{ASSISTANT_SP_TOKEN}{think}"
                )
            );
        }
    }

    /// A system message a latest reminder follows stays bare text with no
    /// header, where a user message before the reminder takes it.
    #[test]
    fn a_system_message_before_a_latest_reminder_stays_bare_text() {
        let reminder = json!({ "role": "latest_reminder", "content": "Reminder" });
        let after_system = [
            user("Hello."),
            json!({ "role": "system", "content": "Note" }),
            reminder.clone(),
        ];
        let out = encode_messages(
            &after_system,
            ThinkingMode::Thinking,
            &EncodeParams::default(),
        )
        .unwrap();
        assert_eq!(
            out,
            format!("{BOS_TOKEN}{USER_SP_TOKEN}Hello.Note{LATEST_REMINDER_SP_TOKEN}Reminder")
        );
        let after_user = [user("Hello."), reminder];
        let out = encode_messages(
            &after_user,
            ThinkingMode::Thinking,
            &EncodeParams::default(),
        )
        .unwrap();
        assert_eq!(
            out,
            format!(
                "{BOS_TOKEN}{USER_SP_TOKEN}Hello.{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}\
                 {LATEST_REMINDER_SP_TOKEN}Reminder"
            )
        );
    }

    #[test]
    fn unknown_role_errors() {
        let msgs = [json!({ "role": "moderator", "content": "hi" })];
        let err = encode_messages(&msgs, ThinkingMode::Chat, &EncodeParams::default()).unwrap_err();
        assert!(matches!(err, DsEncodingError::UnknownRole(ref r) if r == "moderator"));
    }

    fn developer(text: &str) -> Value {
        json!({ "role": "developer", "content": text })
    }

    /// The engine's renderer hands this encoder a `developer` message as
    /// written (its developer-to-system rewrite is the template path's), so
    /// the encoding's own rules apply: in thinking mode the thinking-drop
    /// removes a developer message before the last user turn entirely; in
    /// chat mode it is a user turn, the user token and the content. The
    /// recorded reference of the developer-role case is the thinking render,
    /// byte for byte; never the system message's prompt.
    #[test]
    fn developer_message_renders_as_the_encoder_developer_turn() {
        let msgs = [developer("Answer in one word."), user("Colour of the sky?")];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();
        assert_eq!(
            out,
            format!(
                "{BOS_TOKEN}{USER_SP_TOKEN}Colour of the sky?{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
            )
        );
        let out = encode_messages(&msgs, ThinkingMode::Chat, &EncodeParams::default()).unwrap();
        assert_eq!(
            out,
            format!(
                "{BOS_TOKEN}{USER_SP_TOKEN}Answer in one word.{USER_SP_TOKEN}Colour of the sky?\
                 {ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}"
            )
        );
        let as_system = [
            json!({ "role": "system", "content": "Answer in one word." }),
            user("Colour of the sky?"),
        ];
        for mode in [ThinkingMode::Thinking, ThinkingMode::Chat] {
            assert_ne!(
                encode_messages(&msgs, mode, &EncodeParams::default()).unwrap(),
                encode_messages(&as_system, mode, &EncodeParams::default()).unwrap()
            );
        }
    }

    /// Earlier in the conversation the same rule: gone with the thinking-drop,
    /// a user turn in chat mode (the assistant turn after it closes as it does
    /// after a user turn). An empty developer message is refused where it is
    /// rendered and dropped where the thinking-drop reaches it first.
    #[test]
    fn a_developer_message_before_the_last_user_turn_goes_with_the_thinking_drop() {
        let msgs = [
            developer("DEV"),
            user("q1"),
            json!({ "role": "assistant", "content": "a1" }),
            user("q2"),
        ];
        let out = encode_messages(&msgs, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();
        assert_eq!(
            out,
            format!(
                "{BOS_TOKEN}{USER_SP_TOKEN}q1{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}a1{EOS_TOKEN}\
                 {USER_SP_TOKEN}q2{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}"
            )
        );
        let out = encode_messages(&msgs, ThinkingMode::Chat, &EncodeParams::default()).unwrap();
        assert_eq!(
            out,
            format!(
                "{BOS_TOKEN}{USER_SP_TOKEN}DEV{USER_SP_TOKEN}q1{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}a1\
                 {EOS_TOKEN}{USER_SP_TOKEN}q2{ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}"
            )
        );
        let empty = [developer(""), user("q")];
        let out =
            encode_messages(&empty, ThinkingMode::Thinking, &EncodeParams::default()).unwrap();
        assert_eq!(
            out,
            format!("{BOS_TOKEN}{USER_SP_TOKEN}q{ASSISTANT_SP_TOKEN}{THINKING_START_TOKEN}")
        );
        let err =
            encode_messages(&empty, ThinkingMode::Chat, &EncodeParams::default()).unwrap_err();
        assert!(
            matches!(err, DsEncodingError::InvalidMessage { ref role, .. } if role == "developer"),
            "{err}"
        );
    }

    /// A developer message that ends the conversation is the last user turn:
    /// kept in both modes, the assistant header and the thinking token after
    /// it.
    #[test]
    fn a_trailing_developer_message_is_the_last_user_turn() {
        let msgs = [user("Colour of the sky?"), developer("Answer in one word.")];
        for (mode, thinking_token) in [
            (ThinkingMode::Thinking, THINKING_START_TOKEN),
            (ThinkingMode::Chat, THINKING_END_TOKEN),
        ] {
            let out = encode_messages(&msgs, mode, &EncodeParams::default()).unwrap();
            assert_eq!(
                out,
                format!(
                    "{BOS_TOKEN}{USER_SP_TOKEN}Colour of the sky?{USER_SP_TOKEN}Answer in one word.\
                     {ASSISTANT_SP_TOKEN}{thinking_token}"
                )
            );
        }
    }

    /// A developer message's own tools and response format render under its
    /// content after a blank line, as a system message's do; declared tools
    /// keep the thinking-drop off, so the message stays in thinking mode too.
    #[test]
    fn a_developer_message_carries_its_tools_and_response_format() {
        let with_tools = [json!({
            "role": "developer",
            "content": "Follow the policy.",
            "tools": [{ "type": "function", "function": { "name": "lookup_weather",
                "description": "Look up", "parameters": { "type": "object",
                "properties": { "query": { "type": "string" } }, "required": ["query"] } } }],
        })];
        let with_format = [json!({
            "role": "developer",
            "content": "Answer in one word.",
            "response_format": { "type": "json_object" },
        })];
        for (mode, thinking_token) in [
            (ThinkingMode::Thinking, THINKING_START_TOKEN),
            (ThinkingMode::Chat, THINKING_END_TOKEN),
        ] {
            let out = encode_messages(&with_tools, mode, &EncodeParams::default()).unwrap();
            assert!(
                out.starts_with(&format!(
                    "{BOS_TOKEN}{USER_SP_TOKEN}Follow the policy.\n\n## Tools\n\n"
                )) && out.contains(r#""name": "lookup_weather""#)
                    && out.ends_with(&format!(
                        "schemas to invoke tool calls.\n{ASSISTANT_SP_TOKEN}{thinking_token}"
                    )),
                "{out}"
            );
            let out = encode_messages(&with_format, mode, &EncodeParams::default()).unwrap();
            assert_eq!(
                out,
                format!(
                    "{BOS_TOKEN}{USER_SP_TOKEN}Answer in one word.\n\n## Response Format:\n\n\
                     You MUST strictly adhere to the following schema to reply:\n\
                     {{\"type\": \"json_object\"}}{ASSISTANT_SP_TOKEN}{thinking_token}"
                )
            );
        }
    }

    /// A system message renders its content first, then the tools block and
    /// the response-format section it carries, each after a blank line, in
    /// that order, before the user turn; the same in both thinking modes.
    #[test]
    fn a_system_message_renders_its_content_then_its_tools_then_its_response_format() {
        let messages = [
            json!({
                "role": "system",
                "content": "Use the tool.",
                "tools": [{ "type": "function", "function": { "name": "lookup_weather",
                    "description": "Look up", "parameters": { "type": "object",
                    "properties": { "query": { "type": "string" } }, "required": ["query"] } } }],
                "response_format": { "type": "json_object" },
            }),
            user("hello"),
        ];
        for (mode, thinking_token) in [
            (ThinkingMode::Thinking, THINKING_START_TOKEN),
            (ThinkingMode::Chat, THINKING_END_TOKEN),
        ] {
            let out = encode_messages(&messages, mode, &EncodeParams::default()).unwrap();
            let tools = out.find("## Tools\n").expect("the tools block");
            let format = out
                .find("## Response Format:\n\n")
                .expect("the response-format section");
            assert!(
                out.starts_with(&format!(
                    "{BOS_TOKEN}Use the tool.\n\n## Tools\n\nYou have access"
                )),
                "{out}"
            );
            assert!(tools < format, "{out}");
            assert!(
                out[tools..format].contains(r#""name": "lookup_weather""#)
                    && out.ends_with(&format!(
                        "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n\
                         {{\"type\": \"json_object\"}}{USER_SP_TOKEN}hello{ASSISTANT_SP_TOKEN}{thinking_token}"
                    )),
                "{out}"
            );
        }
    }

    /// The reference writes the response-format section under `if
    /// response_format:`, so `null` and `{}` render no section, on a system
    /// and on a developer message (the engine's own encoder copy renders
    /// these bytes).
    #[test]
    fn a_falsy_response_format_renders_no_section() {
        for (mode, thinking_token) in [
            (ThinkingMode::Chat, THINKING_END_TOKEN),
            (ThinkingMode::Thinking, THINKING_START_TOKEN),
        ] {
            for falsy in [json!(null), json!({})] {
                let messages = [
                    json!({ "role": "system", "content": "You are terse.", "response_format": falsy }),
                    user("hello"),
                ];
                let out = encode_messages(&messages, mode, &EncodeParams::default()).unwrap();
                assert_eq!(
                    out,
                    format!(
                        "{BOS_TOKEN}You are terse.{USER_SP_TOKEN}hello{ASSISTANT_SP_TOKEN}{thinking_token}"
                    ),
                    "{falsy}"
                );
            }
        }
        let messages = [
            json!({ "role": "developer", "content": "Answer in one line.", "response_format": null }),
            user("hello"),
        ];
        let out = encode_messages(&messages, ThinkingMode::Chat, &EncodeParams::default()).unwrap();
        assert_eq!(
            out,
            format!(
                "{BOS_TOKEN}{USER_SP_TOKEN}Answer in one line.{USER_SP_TOKEN}hello\
                 {ASSISTANT_SP_TOKEN}{THINKING_END_TOKEN}"
            )
        );
    }
}
