//! Helpers shared by the DeepSeek V4-family renderers (V4, V4.1, ...).
//!
//! `deepseek_v4.rs` and the upcoming `deepseek_v41.rs` are otherwise
//! standalone 1:1 ports of their respective Python encoders — message
//! merging, tool-result sorting, drop-thinking, and DSML argument rendering
//! are identical between the two revisions except for the DSML tag strings
//! (V4 uses bare `invoke`/`parameter`; V4.1 uses ` invoke`/` parameter` with a
//! leading space). This module holds exactly that shared behaviour so it is
//! implemented, and tested, once.
//!
//! V3.2 is intentionally *not* wired through this module beyond the two index
//! predicates it shares (`find_last_user_index`, `at_or_after_last_user`): its
//! tool-message handling and drop-thinking role set are genuinely different
//! algorithms that merely happen to share some function names, not the same
//! logic.

use serde_json::{json, Value};
use thiserror::Error;

const DSML_TOKEN: &str = "｜DSML｜";

/// DSML tag strings that differ between the V4-family DSML dialects.
///
/// V4 uses bare `invoke`/`parameter`; V4.1 uses ` invoke`/` parameter` (a
/// leading space on each). The block name (`tool_calls` vs. ` calls`) is not
/// included here since it is only ever used by each renderer's own
/// `render_message`/tools-template code, never by the shared helpers below.
#[derive(Debug)]
pub(super) struct DsmlTags {
    pub invoke: &'static str,
    pub parameter: &'static str,
}

/// Errors raised when a message list is malformed.
///
/// Shared by the V4-family renderers (V4, V4.1). V3.2 keeps its own
/// independent error type, since its message-shape invariants differ enough
/// that the variant sets aren't the same.
///
/// `pub`, not `pub(super)`: this is already part of `deepseek_v4`'s public
/// API (returned from `encode_messages`), re-exported unchanged from there.
/// `deepseek_common` itself stays a private module, so this is only ever
/// reachable through that re-export, never directly.
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
    // --- V4.1 only -----------------------------------------------------
    #[error("Message text contains the image special token `<｜deepseek_image｜>`; images must be sent as image content parts")]
    PlaceholderInText,
    #[error("Invalid reasoning effort `{0}`: expected an integer within [1, 100] or one of low, high, xhigh, max")]
    InvalidReasoningEffort(String),
    #[error("Unsupported content part type `{0}`: only text and image parts are supported")]
    UnsupportedContentPart(String),
}

/// Mirrors V4's `find_last_user_index`: returns `None` if no user/developer
/// message exists (Python returns -1).
///
/// V4.1 widens this definition (a mid-conversation system message also counts)
/// and therefore keeps its own version; only the "is this index at or after
/// the last user turn" predicate below is shared.
pub(super) fn find_last_user_index(messages: &[Value]) -> Option<usize> {
    for idx in (0..messages.len()).rev() {
        let role = messages[idx].get("role").and_then(|v| v.as_str());
        if matches!(role, Some("user") | Some("developer")) {
            return Some(idx);
        }
    }
    None
}

/// Returns `true` when `index >= last_user_idx` in the Python sense, treating
/// the "no user message" case (-1) as: every non-negative index satisfies it.
///
/// Used by [`drop_thinking_messages`] here and by every DeepSeek renderer's
/// `render_message` (V3.2, V4 and V4.1).
pub(super) fn at_or_after_last_user(index: usize, last_user_idx: Option<usize>) -> bool {
    match last_user_idx {
        Some(idx) => index >= idx,
        None => true,
    }
}

// ---------------------------------------------------------------------------
// Preprocessing: merge tool messages and sort tool results.
// ---------------------------------------------------------------------------
pub(super) fn merge_tool_messages(messages: &[Value]) -> Vec<Value> {
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
pub(super) fn sort_tool_results_by_call_order(messages: Vec<Value>) -> Vec<Value> {
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
///
/// `last_user_idx` is passed in rather than recomputed because the two
/// revisions disagree on what the "last user turn" is: V4 counts only
/// user/developer messages ([`find_last_user_index`]), while V4.1 also counts
/// mid-conversation system messages. The filtering itself is identical.
pub(super) fn drop_thinking_messages(
    messages: &[Value],
    last_user_idx: Option<usize>,
) -> Vec<Value> {
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

/// Encode a single tool call's `arguments` value into `<DSML_TOKEN><parameter>`
/// blocks, mirroring the reference `encode_arguments_to_dsml` (HF
/// `encoding.py`, identical in vLLM's port).
///
/// `arguments` may arrive as a JSON *string* (raw OpenAI tool_calls) or as an
/// already-parsed value — `model_gateway`'s `process_tool_call_arguments`
/// parses the client's string before the chat template runs, so an object
/// arrives as an object and any other payload (array, number, bool, null) as
/// that value. A string is JSON-parsed at most twice (tolerating a
/// double-encoded object), stopping at the first non-string result or the
/// first parse failure. Whatever is then not an object — an unparsable
/// string, a parsed or pre-parsed non-object, `Value::Null` for a call with no
/// `arguments` at all — renders as the single parameter `arguments` holding
/// the ORIGINAL value: `string="true"` with the raw text when it was a string,
/// otherwise `string="false"` with its JSON. Never errors (unlike V3.2, which
/// propagates a parse error).
pub(super) fn encode_arguments_to_dsml(arguments: &Value, tags: &DsmlTags) -> String {
    let mut parsed = arguments.clone();
    // Tolerate JSON strings, including double-encoded ones.
    for _ in 0..2 {
        let Value::String(text) = &parsed else {
            break;
        };
        match serde_json::from_str::<Value>(text) {
            Ok(value) => parsed = value,
            Err(_) => break,
        }
    }
    // The fallback wraps the ORIGINAL `arguments`, not the parsed value.
    let parameters: Vec<(&str, &Value)> = match parsed.as_object() {
        Some(object) => object
            .iter()
            .map(|(key, value)| (key.as_str(), value))
            .collect(),
        None => vec![("arguments", arguments)],
    };
    let mut parts = Vec::with_capacity(parameters.len());
    for (key, value) in parameters {
        let (is_str, value_str) = match value {
            Value::String(text) => ("true", text.clone()),
            other => ("false", to_json(other)),
        };
        parts.push(format!(
            "<{DSML_TOKEN}{p} name=\"{key}\" string=\"{is_str}\">{value_str}</{DSML_TOKEN}{p}>",
            p = tags.parameter,
        ));
    }
    parts.join("\n")
}

// Python's `to_json` is `json.dumps(value, ensure_ascii=False)`: spaced
// separators, raw UTF-8. Compact `serde_json::to_string` would change the
// prompt bytes vLLM trained on. Each renderer module keeps its own copy of
// this tiny wrapper (see `deepseek_v4::to_json`); this one is for
// `encode_arguments_to_dsml`'s exclusive use.
fn to_json(value: &Value) -> String {
    crate::json_dumps::to_string(value)
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;

    /// V4.1's dialect (a leading space on both tags), the one the fixture
    /// cases `assistant_tool_call_array_string_args` and
    /// `assistant_tool_call_double_encoded_args` record end to end.
    const V41_TAGS: DsmlTags = DsmlTags {
        invoke: " invoke",
        parameter: " parameter",
    };
    const V4_TAGS: DsmlTags = DsmlTags {
        invoke: "invoke",
        parameter: "parameter",
    };

    /// The helper is shared with V4: these are the three input shapes whose
    /// V4 output changed when it started mirroring the reference (they used
    /// to render an empty parameter list).
    #[test]
    fn v4_non_object_arguments_render_one_parameter_like_the_reference() {
        assert_eq!(
            encode_arguments_to_dsml(&Value::Null, &V4_TAGS),
            "<｜DSML｜parameter name=\"arguments\" string=\"false\">null</｜DSML｜parameter>"
        );
        assert_eq!(
            encode_arguments_to_dsml(&json!([1, 2]), &V4_TAGS),
            "<｜DSML｜parameter name=\"arguments\" string=\"false\">[1, 2]</｜DSML｜parameter>"
        );
        assert_eq!(
            encode_arguments_to_dsml(&json!("\"{\\\"a\\\": 1}\""), &V4_TAGS),
            "<｜DSML｜parameter name=\"a\" string=\"false\">1</｜DSML｜parameter>"
        );
    }

    #[test]
    fn object_arguments_render_one_parameter_per_key() {
        assert_eq!(
            encode_arguments_to_dsml(&json!({"n": 3, "s": "raw <x>"}), &V41_TAGS),
            "<｜DSML｜ parameter name=\"n\" string=\"false\">3</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"s\" string=\"true\">raw <x></｜DSML｜ parameter>"
        );
    }

    #[test]
    fn string_arguments_that_parse_to_an_object_render_its_keys() {
        assert_eq!(
            encode_arguments_to_dsml(&json!("{\"a\": 1}"), &V41_TAGS),
            "<｜DSML｜ parameter name=\"a\" string=\"false\">1</｜DSML｜ parameter>"
        );
    }

    #[test]
    fn raw_string_arguments_that_parse_to_an_array_wrap_the_raw_string() {
        // The reference wraps the ORIGINAL value: the client's raw string is
        // one string parameter, not the array it parses to.
        assert_eq!(
            encode_arguments_to_dsml(&json!("[1, 2]"), &V41_TAGS),
            "<｜DSML｜ parameter name=\"arguments\" string=\"true\">[1, 2]</｜DSML｜ parameter>"
        );
    }

    #[test]
    fn gateway_parsed_array_arguments_wrap_the_array_as_json() {
        // `process_tool_call_arguments` has already parsed the client's
        // `"[1, 2]"`, so the renderer receives the array itself.
        assert_eq!(
            encode_arguments_to_dsml(&json!([1, 2]), &V41_TAGS),
            "<｜DSML｜ parameter name=\"arguments\" string=\"false\">[1, 2]</｜DSML｜ parameter>"
        );
    }

    #[test]
    fn number_and_bool_arguments_wrap_the_value_as_json() {
        assert_eq!(
            encode_arguments_to_dsml(&json!(42), &V41_TAGS),
            "<｜DSML｜ parameter name=\"arguments\" string=\"false\">42</｜DSML｜ parameter>"
        );
        assert_eq!(
            encode_arguments_to_dsml(&json!(true), &V41_TAGS),
            "<｜DSML｜ parameter name=\"arguments\" string=\"false\">true</｜DSML｜ parameter>"
        );
    }

    #[test]
    fn null_arguments_wrap_null_as_json() {
        assert_eq!(
            encode_arguments_to_dsml(&Value::Null, &V41_TAGS),
            "<｜DSML｜ parameter name=\"arguments\" string=\"false\">null</｜DSML｜ parameter>"
        );
    }

    #[test]
    fn double_encoded_string_arguments_are_parsed_twice() {
        // The JSON string `"{\"a\": 1}"`: the first parse yields the text
        // `{"a": 1}`, the second the object.
        assert_eq!(
            encode_arguments_to_dsml(&json!("\"{\\\"a\\\": 1}\""), &V41_TAGS),
            "<｜DSML｜ parameter name=\"a\" string=\"false\">1</｜DSML｜ parameter>"
        );
    }

    #[test]
    fn triple_encoded_string_arguments_stop_after_two_parses() {
        // Encoding the object's text twice gives the double-encoded string
        // above; a third time is one level too many for the reference, so the
        // ORIGINAL text is wrapped verbatim.
        let twice = serde_json::to_string("{\"a\": 1}").unwrap();
        assert_eq!(twice, "\"{\\\"a\\\": 1}\"");
        let thrice = serde_json::to_string(&twice).unwrap();
        assert_eq!(
            encode_arguments_to_dsml(&Value::String(thrice.clone()), &V41_TAGS),
            format!("<｜DSML｜ parameter name=\"arguments\" string=\"true\">{thrice}</｜DSML｜ parameter>")
        );
    }

    #[test]
    fn unparsable_string_arguments_are_wrapped_verbatim() {
        assert_eq!(
            encode_arguments_to_dsml(&json!("{oops"), &V41_TAGS),
            "<｜DSML｜ parameter name=\"arguments\" string=\"true\">{oops</｜DSML｜ parameter>"
        );
    }
}
