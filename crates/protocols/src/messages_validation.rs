//! The public Messages API's own request rules, applied with the field-level
//! validation of [`CreateMessageRequest`]: the shapes the public API refuses
//! with `invalid_request_error` are refused here before any rendering, each
//! message naming the field as the public API names it.
//!
//! Deliberately not enforced, as documented extensions of this gateway:
//! unknown top-level fields and `rid` (forwarded to the engine), a `system`
//! role inside `messages` (kept in place for the clients that send one), the
//! vendor's model-specific couplings of thinking with `temperature` or a
//! forced `tool_choice` (a served model may take both), and the per-model
//! output-token cap. The rule that `thinking.budget_tokens` stays under
//! `max_tokens` reads the `anthropic-beta` header (the interleaved-thinking
//! beta lifts it), so the gateway's handler applies it.

use std::collections::HashSet;

use validator::ValidationError;

use crate::messages::{
    CreateMessageRequest, InputContent, InputContentBlock, InputMessage, Role, ThinkingConfig, Tool,
};

/// Fields of a custom tool definition the public API accepts that the typed
/// definition does not model yet; they pass through, anything else is refused
/// as the public API refuses it.
const PASSTHROUGH_CUSTOM_TOOL_FIELDS: &[&str] = &[
    "strict",
    "input_examples",
    "eager_input_streaming",
    "allowed_callers",
];

pub(crate) fn validate(req: &CreateMessageRequest) -> Result<(), ValidationError> {
    validate_sampling(req)?;
    validate_message_contents(&req.messages)?;
    validate_tool_pairing(&req.messages)?;
    validate_stop_sequences(req.stop_sequences.as_deref())?;
    validate_tools(req.tools.as_deref())?;
    validate_thinking(req)
}

fn invalid(code: &'static str, message: String) -> ValidationError {
    let mut e = ValidationError::new(code);
    e.message = Some(message.into());
    e
}

/// `temperature` and `top_p` take the public API's range, 0 to 1.
fn validate_sampling(req: &CreateMessageRequest) -> Result<(), ValidationError> {
    for (name, value) in [("temperature", req.temperature), ("top_p", req.top_p)] {
        if value.is_some_and(|v| !(0.0..=1.0).contains(&v)) {
            return Err(invalid(
                "value_out_of_range",
                format!("{name}: range: 0..1"),
            ));
        }
    }
    Ok(())
}

fn role_name(role: &Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
    }
}

/// Every message carries content, except the optional final assistant
/// message (a prefill), which may be empty but may not end in whitespace;
/// every text block carries text. A `system` message in the array is this
/// gateway's extension and is left to its own rules.
fn validate_message_contents(messages: &[InputMessage]) -> Result<(), ValidationError> {
    let last = messages.len().saturating_sub(1);
    for (index, message) in messages.iter().enumerate() {
        if message.role == Role::System {
            continue;
        }
        let final_assistant = index == last && message.role == Role::Assistant;
        let empty = match &message.content {
            InputContent::String(text) => text.trim().is_empty(),
            InputContent::Blocks(blocks) => blocks.is_empty(),
        };
        if empty && !final_assistant {
            return Err(invalid(
                "empty_content",
                format!(
                    "messages.{index}: {} messages must have non-empty content",
                    role_name(&message.role)
                ),
            ));
        }
        if let InputContent::Blocks(blocks) = &message.content {
            let empty_text = blocks.iter().any(|block| {
                matches!(block, InputContentBlock::Text(text) if text.text.trim().is_empty())
            });
            if empty_text {
                return Err(invalid(
                    "empty_text_block",
                    format!("messages.{index}: text content blocks must be non-empty"),
                ));
            }
        }
        if final_assistant {
            let tail = match &message.content {
                InputContent::String(text) => Some(text.as_str()),
                InputContent::Blocks(blocks) => blocks.last().and_then(|block| match block {
                    InputContentBlock::Text(text) => Some(text.text.as_str()),
                    _ => None,
                }),
            };
            if tail.is_some_and(|text| text.ends_with(char::is_whitespace)) {
                return Err(invalid(
                    "prefill_trailing_whitespace",
                    "messages: final assistant content cannot end with trailing whitespace"
                        .to_owned(),
                ));
            }
        }
    }
    Ok(())
}

/// A conversation turn: consecutive messages of one role, which the public
/// API combines into a single turn, with the index of the first.
struct Turn<'a> {
    role: Role,
    at: usize,
    messages: Vec<(usize, &'a InputMessage)>,
}

/// The user and assistant turns of `messages`, consecutive same-role
/// messages combined; a `system` message of this gateway's extension is
/// transparent.
fn turns(messages: &[InputMessage]) -> Vec<Turn<'_>> {
    let mut turns: Vec<Turn<'_>> = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role == Role::System {
            continue;
        }
        match turns.last_mut() {
            Some(turn) if turn.role == message.role => turn.messages.push((index, message)),
            _ => turns.push(Turn {
                role: message.role.clone(),
                at: index,
                messages: vec![(index, message)],
            }),
        }
    }
    turns
}

fn tool_use_ids<'a>(turn: &Turn<'a>) -> Vec<&'a str> {
    turn.messages
        .iter()
        .flat_map(|(_, message)| match &message.content {
            InputContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    InputContentBlock::ToolUse(tool_use) => Some(tool_use.id.as_str()),
                    _ => None,
                })
                .collect(),
            InputContent::String(_) => Vec::new(),
        })
        .collect()
}

/// A `tool_use` block is answered by a `tool_result` block in the very next
/// user turn, one per id, and a `tool_result` answers a `tool_use` of the
/// assistant turn right before it: the public API refuses a result nobody
/// asked for, a call nobody answered, and a call answered twice, naming the
/// message. Consecutive messages of one role form one turn, as the public
/// API combines them (parallel results split over two user messages answer
/// one assistant turn); a `system` message in between is skipped.
fn validate_tool_pairing(messages: &[InputMessage]) -> Result<(), ValidationError> {
    let unanswered = |at: usize, ids: &[&str]| {
        invalid(
            "tool_use_without_tool_result",
            format!(
                "messages.{at}: `tool_use` ids were found without `tool_result` blocks \
                 immediately after: {}. Each `tool_use` block must have a corresponding \
                 `tool_result` block in the next message.",
                ids.join(", ")
            ),
        )
    };
    let mut pending: Vec<&str> = Vec::new();
    let mut pending_at = 0;
    for turn in turns(messages) {
        match turn.role {
            Role::System => continue,
            Role::Assistant => {
                if !pending.is_empty() {
                    return Err(unanswered(pending_at, &pending));
                }
                pending = tool_use_ids(&turn);
                pending_at = turn.at;
            }
            Role::User => {
                let mut answered: HashSet<&str> = HashSet::new();
                for (index, message) in &turn.messages {
                    let InputContent::Blocks(blocks) = &message.content else {
                        continue;
                    };
                    for (position, block) in blocks.iter().enumerate() {
                        let InputContentBlock::ToolResult(result) = block else {
                            continue;
                        };
                        let id = result.tool_use_id.as_str();
                        if !pending.contains(&id) {
                            return Err(invalid(
                                "unexpected_tool_use_id",
                                format!(
                                    "messages.{index}.content.{position}: unexpected `tool_use_id` \
                                     found in `tool_result` blocks: {id}. Each `tool_result` block \
                                     must have a corresponding `tool_use` block in the previous \
                                     message."
                                ),
                            ));
                        }
                        if !answered.insert(id) {
                            return Err(invalid(
                                "duplicate_tool_result",
                                format!(
                                    "messages.{index}.content.{position}: `tool_result` blocks must \
                                     each answer a different `tool_use_id`: {id} is answered twice."
                                ),
                            ));
                        }
                    }
                }
                let missing: Vec<&str> = pending
                    .iter()
                    .copied()
                    .filter(|id| !answered.contains(id))
                    .collect();
                if !missing.is_empty() {
                    return Err(unanswered(pending_at, &missing));
                }
                pending.clear();
            }
        }
    }
    if pending.is_empty() {
        Ok(())
    } else {
        Err(unanswered(pending_at, &pending))
    }
}

/// A stop sequence is matched against generated text, so it needs text.
fn validate_stop_sequences(sequences: Option<&[String]>) -> Result<(), ValidationError> {
    if sequences
        .unwrap_or_default()
        .iter()
        .any(|sequence| sequence.trim().is_empty())
    {
        return Err(invalid(
            "invalid_stop_sequence",
            "stop_sequences: each stop sequence must contain non-whitespace".to_owned(),
        ));
    }
    Ok(())
}

fn tool_name(tool: &Tool) -> Option<&str> {
    match tool {
        Tool::Custom(tool) => Some(&tool.name),
        Tool::ToolSearch(tool) => Some(&tool.name),
        Tool::Bash(tool) => Some(&tool.name),
        Tool::TextEditor(tool) => Some(&tool.name),
        Tool::WebSearch(tool) => Some(&tool.name),
        Tool::McpToolset(_) => None,
    }
}

/// Tool names are unique; a custom tool's name matches
/// `^[a-zA-Z0-9_-]{1,128}$`, its input schema describes an object, and its
/// definition carries only the public API's fields.
fn validate_tools(tools: Option<&[Tool]>) -> Result<(), ValidationError> {
    let mut names: HashSet<&str> = HashSet::new();
    for (index, tool) in tools.unwrap_or_default().iter().enumerate() {
        if let Some(name) = tool_name(tool) {
            if !names.insert(name) {
                return Err(invalid(
                    "duplicate_tool_names",
                    "tools: Tool names must be unique.".to_owned(),
                ));
            }
        }
        let Tool::Custom(custom) = tool else {
            continue;
        };
        if custom.name.chars().count() > 128 {
            return Err(invalid(
                "tool_name_too_long",
                format!("tools.{index}.custom.name: String should have at most 128 characters"),
            ));
        }
        let well_formed = !custom.name.is_empty()
            && custom
                .name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !well_formed {
            return Err(invalid(
                "invalid_tool_name",
                format!(
                    "tools.{index}.custom.name: String should match pattern '^[a-zA-Z0-9_-]{{1,128}}$'"
                ),
            ));
        }
        if custom.input_schema.schema_type != "object" {
            return Err(invalid(
                "invalid_input_schema_type",
                format!("tools.{index}.custom.input_schema.type: Input should be 'object'"),
            ));
        }
        if let Some(field) = custom
            .extra
            .keys()
            .find(|key| !PASSTHROUGH_CUSTOM_TOOL_FIELDS.contains(&key.as_str()))
        {
            return Err(invalid(
                "unknown_tool_field",
                format!("tools.{index}.custom.{field}: Extra inputs are not permitted"),
            ));
        }
    }
    Ok(())
}

/// A thinking budget starts at 1024 tokens. (That it stays under
/// `max_tokens` depends on the `anthropic-beta` header, so the gateway's
/// handler checks it.)
fn validate_thinking(req: &CreateMessageRequest) -> Result<(), ValidationError> {
    if let Some(ThinkingConfig::Enabled { budget_tokens, .. }) = &req.thinking {
        if *budget_tokens < 1024 {
            return Err(invalid(
                "thinking_budget_too_small",
                "thinking.enabled.budget_tokens: Input should be greater than or equal to 1024"
                    .to_owned(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};
    use validator::Validate;

    fn request(body: Value) -> CreateMessageRequest {
        let mut body = body;
        let object = body.as_object_mut().expect("an object");
        object.entry("model").or_insert(json!("m"));
        object.entry("max_tokens").or_insert(json!(64));
        object
            .entry("messages")
            .or_insert(json!([{"role": "user", "content": "hi"}]));
        serde_json::from_value(body).expect("a well-formed request")
    }

    fn rejection(body: Value) -> String {
        request(body)
            .validate()
            .expect_err("the public API refuses this shape")
            .to_string()
    }

    fn accepted(body: Value) {
        request(body)
            .validate()
            .expect("the public API accepts this shape");
    }

    use super::*;

    #[test]
    fn sampling_parameters_take_the_public_range() {
        assert!(rejection(json!({"temperature": 2.0})).contains("temperature: range: 0..1"));
        assert!(rejection(json!({"temperature": -1.0})).contains("temperature: range: 0..1"));
        assert!(rejection(json!({"top_p": 1.5})).contains("top_p: range: 0..1"));
        accepted(json!({"temperature": 1.0, "top_p": 0.0}));
    }

    #[test]
    fn messages_carry_content_except_the_final_prefill() {
        assert!(
            rejection(json!({"messages": [{"role": "user", "content": ""}]}))
                .contains("messages.0: user messages must have non-empty content")
        );
        assert!(
            rejection(json!({"messages": [{"role": "user", "content": []}]}))
                .contains("messages.0: user messages must have non-empty content")
        );
        assert!(rejection(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "   "}
        ]}))
        .contains("messages.2: user messages must have non-empty content"));
        assert!(rejection(json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": ""}]}
        ]}))
        .contains("messages.0: text content blocks must be non-empty"));
        // The optional final assistant message may be empty, not end in whitespace.
        accepted(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""}
        ]}));
        assert!(rejection(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "The answer is "}
        ]}))
        .contains("final assistant content cannot end with trailing whitespace"));
        assert!(rejection(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "The answer is\n"}]}
        ]}))
        .contains("final assistant content cannot end with trailing whitespace"));
        // A system message in the array is this gateway's extension: untouched,
        // empty blocks and empty text alike.
        accepted(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": []},
            {"role": "system", "content": [{"type": "text", "text": ""}]},
            {"role": "user", "content": "again"}
        ]}));
    }

    fn tool_use(id: &str) -> Value {
        json!({"type": "tool_use", "id": id, "name": "get_weather", "input": {"city": "Paris"}})
    }

    fn tool_result(id: &str) -> Value {
        json!({"type": "tool_result", "tool_use_id": id, "content": "sunny"})
    }

    #[test]
    fn tool_use_and_tool_result_blocks_pair_up() {
        accepted(json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [tool_use("toolu_a"), tool_use("toolu_b")]},
            {"role": "user", "content": [tool_result("toolu_b"), tool_result("toolu_a")]}
        ]}));
        let missing = rejection(json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [tool_use("toolu_a"), tool_use("toolu_b")]},
            {"role": "user", "content": [tool_result("toolu_a")]}
        ]}));
        assert!(
            missing.contains("messages.1: `tool_use` ids were found without `tool_result` blocks immediately after: toolu_b"),
            "{missing}"
        );
        let unknown = rejection(json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [tool_use("toolu_a")]},
            {"role": "user", "content": [tool_result("toolu_wrong")]}
        ]}));
        assert!(
            unknown.contains("messages.2.content.0: unexpected `tool_use_id` found in `tool_result` blocks: toolu_wrong"),
            "{unknown}"
        );
        let orphan = rejection(json!({"messages": [
            {"role": "user", "content": [tool_result("toolu_probe")]}
        ]}));
        assert!(
            orphan.contains("messages.0.content.0: unexpected `tool_use_id` found in `tool_result` blocks: toolu_probe"),
            "{orphan}"
        );
        let trailing = rejection(json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [tool_use("toolu_a")]}
        ]}));
        assert!(
            trailing.contains("messages.1: `tool_use` ids were found without"),
            "{trailing}"
        );
        let twice = rejection(json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [tool_use("toolu_a")]},
            {"role": "user", "content": [tool_result("toolu_a"), tool_result("toolu_a")]}
        ]}));
        assert!(
            twice.contains("messages.2.content.1: `tool_result` blocks must each answer a different `tool_use_id`: toolu_a is answered twice."),
            "{twice}"
        );
    }

    /// Consecutive messages of one role are one turn, as the public API
    /// combines them: parallel results split over two user messages answer
    /// one assistant turn, and a tool_use followed by more assistant text is
    /// answered after the whole assistant turn; a system message in between
    /// changes nothing.
    #[test]
    fn consecutive_same_role_messages_pair_as_one_turn() {
        accepted(json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [tool_use("toolu_a"), tool_use("toolu_b")]},
            {"role": "user", "content": [tool_result("toolu_a")]},
            {"role": "user", "content": [tool_result("toolu_b")]}
        ]}));
        accepted(json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [tool_use("toolu_a")]},
            {"role": "assistant", "content": "Let me check that."},
            {"role": "system", "content": "stay brief"},
            {"role": "user", "content": [tool_result("toolu_a")]}
        ]}));
        let still_missing = rejection(json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [tool_use("toolu_a"), tool_use("toolu_b")]},
            {"role": "user", "content": [tool_result("toolu_a")]},
            {"role": "user", "content": "and tomorrow?"}
        ]}));
        assert!(still_missing.contains("messages.1: `tool_use` ids were found without `tool_result` blocks immediately after: toolu_b"), "{still_missing}");
    }

    #[test]
    fn stop_sequences_carry_text() {
        assert!(rejection(json!({"stop_sequences": ["   "]}))
            .contains("stop_sequences: each stop sequence must contain non-whitespace"));
        accepted(json!({"stop_sequences": ["END", " stop "]}));
    }

    fn custom_tool(name: &str) -> Value {
        json!({"name": name, "description": "d", "input_schema": {"type": "object", "properties": {}}})
    }

    #[test]
    fn tool_definitions_follow_the_public_contract() {
        assert!(
            rejection(json!({"tools": [custom_tool("a"), custom_tool("a")]}))
                .contains("tools: Tool names must be unique.")
        );
        assert!(rejection(json!({"tools": [custom_tool("get weather!")]}))
            .contains("tools.0.custom.name: String should match pattern '^[a-zA-Z0-9_-]{1,128}$'"));
        assert!(rejection(json!({"tools": [custom_tool(&"x".repeat(129))]}))
            .contains("tools.0.custom.name: String should have at most 128 characters"));
        assert!(rejection(json!({"tools": [
            {"name": "a", "input_schema": {"type": "string"}}
        ]}))
        .contains("tools.0.custom.input_schema.type: Input should be 'object'"));
        assert!(rejection(json!({"tools": [
            {"name": "a", "input_schema": {"type": "object"}, "banana": 1}
        ]}))
        .contains("tools.0.custom.banana: Extra inputs are not permitted"));
        // The public API's own newer tool fields pass through.
        accepted(json!({"tools": [
            {"name": "a", "input_schema": {"type": "object"}, "strict": true}
        ]}));
        accepted(json!({"tools": [custom_tool("get_weather-v2"), custom_tool("other")]}));
    }

    #[test]
    fn thinking_budget_is_bounded() {
        assert!(rejection(
            json!({"max_tokens": 4096, "thinking": {"type": "enabled", "budget_tokens": 1023}})
        )
        .contains("thinking.enabled.budget_tokens: Input should be greater than or equal to 1024"));
        accepted(
            json!({"max_tokens": 4096, "thinking": {"type": "enabled", "budget_tokens": 1024}}),
        );
        // Whether the budget stays under max_tokens depends on the beta
        // header: the handler decides, not the body alone.
        accepted(
            json!({"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 1024}}),
        );
        // The vendor's couplings of thinking with sampling and tool_choice are
        // model-specific: a served model takes both.
        accepted(
            json!({"max_tokens": 4096, "temperature": 0.6, "thinking": {"type": "enabled", "budget_tokens": 1024}}),
        );
    }

    #[test]
    fn an_unknown_field_in_a_message_does_not_deserialize() {
        let err = serde_json::from_value::<CreateMessageRequest>(json!({
            "model": "m", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi", "extra": 1}]
        }))
        .expect_err("the public API refuses an unknown field in a message")
        .to_string();
        assert!(err.contains("extra"), "{err}");
    }
}
