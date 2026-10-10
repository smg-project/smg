//! OpenAI's own contract: the public Chat Completions API's structured-output
//! and tool-message rules, applied to OpenAI-named models only. The default
//! profile ([`super::ProviderProfile::Generic`]) passes these shapes through
//! to the engine, whose grammar compiler validates what it needs.

use std::collections::HashSet;

use serde_json::Value;

use crate::{
    chat::{ChatCompletionRequest, ChatMessage, MessageContent},
    common::{ContentPart, ResponseFormat},
};

pub(super) fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    validate_tool_message_pairing(&req.messages)?;
    validate_json_object_prompt(req)?;
    validate_strict_schemas(req)
}

/// Tool messages answer the `tool_calls` of the assistant message right
/// before them, one message per call id, before the conversation goes on:
/// a `tool` message with no call to answer, a `tool_call_id` the preceding
/// assistant turn did not issue, and an assistant `tool_calls` turn left
/// without its tool messages are rejected the way the public API rejects
/// them, naming the message. Historical `arguments` are not inspected (the
/// public API takes any string there). Linear in the number of messages
/// and calls.
fn validate_tool_message_pairing(
    messages: &[ChatMessage],
) -> Result<(), validator::ValidationError> {
    // The call ids of the last assistant `tool_calls` turn still unanswered
    // (in issue order, for a stable message) and that turn's position.
    let mut pending: HashSet<&str> = HashSet::new();
    let mut order: Vec<&str> = Vec::new();
    let mut calls_at: Option<usize> = None;
    let unanswered = |at: usize, order: &[&str], pending: &HashSet<&str>| {
        let ids: Vec<&str> = order
            .iter()
            .copied()
            .filter(|id| pending.contains(id))
            .collect();
        invalid(
            "tool_calls_without_tool_messages",
            format!(
                "Invalid parameter: messages.[{at}].role: an assistant message with 'tool_calls' \
                 must be followed by tool messages responding to each 'tool_call_id'; the \
                 following tool_call_ids did not have response messages: {}",
                ids.join(", ")
            ),
        )
    };
    for (index, message) in messages.iter().enumerate() {
        match message {
            ChatMessage::Tool { tool_call_id, .. } => {
                let Some(at) = calls_at else {
                    return Err(invalid(
                        "tool_message_without_tool_calls",
                        format!(
                            "Invalid parameter: messages.[{index}].role: messages with role 'tool' \
                             must be a response to a preceding message with 'tool_calls'"
                        ),
                    ));
                };
                if !pending.remove(tool_call_id.as_str()) {
                    return Err(invalid(
                        "tool_call_id_not_found",
                        format!(
                            "Invalid parameter: messages.[{index}].tool_call_id: '{tool_call_id}' \
                             not found in the 'tool_calls' of messages.[{at}]"
                        ),
                    ));
                }
            }
            ChatMessage::Assistant { tool_calls, .. } => {
                if let Some(at) = calls_at.filter(|_| !pending.is_empty()) {
                    return Err(unanswered(at, &order, &pending));
                }
                order = tool_calls
                    .iter()
                    .flatten()
                    .map(|call| call.id.as_str())
                    .collect();
                pending = order.iter().copied().collect();
                calls_at = (!order.is_empty()).then_some(index);
            }
            _ => {
                if let Some(at) = calls_at.filter(|_| !pending.is_empty()) {
                    return Err(unanswered(at, &order, &pending));
                }
                calls_at = None;
            }
        }
    }
    match calls_at.filter(|_| !pending.is_empty()) {
        Some(at) => Err(unanswered(at, &order, &pending)),
        None => Ok(()),
    }
}

fn invalid(code: &'static str, message: String) -> validator::ValidationError {
    let mut e = validator::ValidationError::new(code);
    e.message = Some(message.into());
    e
}

/// `response_format: {"type": "json_object"}` needs the word "json" somewhere
/// in the messages, as the public API requires: without the instruction the
/// model may stream whitespace until the token cap.
fn validate_json_object_prompt(
    req: &ChatCompletionRequest,
) -> Result<(), validator::ValidationError> {
    if !matches!(req.response_format, Some(ResponseFormat::JsonObject)) {
        return Ok(());
    }
    let mentions_json = req.messages.iter().any(|message| {
        let content = match message {
            ChatMessage::System { content, .. }
            | ChatMessage::User { content, .. }
            | ChatMessage::Developer { content, .. }
            | ChatMessage::Tool { content, .. }
            | ChatMessage::Root { content, .. } => Some(content),
            ChatMessage::Assistant { content, .. } => content.as_ref(),
            ChatMessage::Function { content, .. } => return contains_json(content),
        };
        match content {
            Some(MessageContent::Text(text)) => contains_json(text),
            Some(MessageContent::Parts(parts)) => parts
                .iter()
                .any(|part| matches!(part, ContentPart::Text { text } if contains_json(text))),
            None => false,
        }
    });
    if mentions_json {
        return Ok(());
    }
    Err(invalid(
        "json_object_requires_json_in_messages",
        "'messages' must contain the word 'json' in some form to use 'response_format' of \
         type 'json_object'"
            .to_string(),
    ))
}

fn contains_json(text: &str) -> bool {
    text.to_ascii_lowercase().contains("json")
}

/// A strict schema (`response_format.json_schema.strict`, a tool's
/// `function.strict`) pins the output to exactly its keys, so every object
/// in it must say `"additionalProperties": false`, as the public API requires
/// before it accepts the schema.
fn validate_strict_schemas(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    if let Some(ResponseFormat::JsonSchema { json_schema }) = &req.response_format {
        if json_schema.strict == Some(true) {
            if let Some(context) = object_without_additional_properties_false(&json_schema.schema) {
                return Err(invalid(
                    "invalid_json_schema",
                    format!(
                        "Invalid schema for response_format '{}': in context={context}, \
                         'additionalProperties' is required to be supplied and to be false \
                         when 'strict' is true",
                        json_schema.name
                    ),
                ));
            }
        }
    }
    for (index, tool) in req.tools.iter().flatten().enumerate() {
        if tool.function.strict != Some(true) {
            continue;
        }
        if let Some(context) = object_without_additional_properties_false(&tool.function.parameters)
        {
            return Err(invalid(
                "invalid_function_parameters",
                format!(
                    "Invalid schema for function '{}' (tools[{index}].function.parameters): in \
                     context={context}, 'additionalProperties' is required to be supplied and \
                     to be false when 'strict' is true",
                    tool.function.name
                ),
            ));
        }
    }
    Ok(())
}

/// The JSON-pointer-like path of the first object schema in `schema` without
/// `"additionalProperties": false`, walking `properties`, array `items`,
/// `$defs`/`definitions` and the `anyOf`/`oneOf`/`allOf` branches; `None`
/// when every object schema has it.
fn object_without_additional_properties_false(schema: &Value) -> Option<String> {
    fn walk(schema: &Value, path: &mut Vec<String>) -> Option<String> {
        let object = schema.as_object()?;
        let is_object_schema = object.get("type").and_then(Value::as_str) == Some("object")
            || object.contains_key("properties");
        if is_object_schema && object.get("additionalProperties") != Some(&Value::Bool(false)) {
            return Some(if path.is_empty() {
                "()".to_string()
            } else {
                format!("('{}')", path.join("', '"))
            });
        }
        for key in ["properties", "$defs", "definitions"] {
            if let Some(children) = object.get(key).and_then(Value::as_object) {
                for (name, child) in children {
                    path.push(key.to_string());
                    path.push(name.clone());
                    let found = walk(child, path);
                    path.pop();
                    path.pop();
                    if found.is_some() {
                        return found;
                    }
                }
            }
        }
        if let Some(items) = object.get("items") {
            path.push("items".to_string());
            let found = walk(items, path);
            path.pop();
            if found.is_some() {
                return found;
            }
        }
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(branches) = object.get(key).and_then(Value::as_array) {
                for (index, branch) in branches.iter().enumerate() {
                    path.push(key.to_string());
                    path.push(index.to_string());
                    let found = walk(branch, path);
                    path.pop();
                    path.pop();
                    if found.is_some() {
                        return found;
                    }
                }
            }
        }
        None
    }
    walk(schema, &mut Vec::new())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use validator::Validate;

    use super::*;

    fn request(extra: Value) -> ChatCompletionRequest {
        let mut value = json!({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "Say hi in one word."}],
            "max_completion_tokens": 64
        });
        value
            .as_object_mut()
            .expect("object")
            .extend(extra.as_object().expect("object").clone());
        serde_json::from_value(value).expect("request deserializes")
    }

    /// The validation error codes and messages, joined, or `None` when valid.
    fn code(req: &ChatCompletionRequest) -> Option<String> {
        req.validate().err().map(|errors| {
            errors
                .field_errors()
                .values()
                .flat_map(|errors| errors.iter())
                .map(|error| {
                    format!(
                        "{}: {}",
                        error.code,
                        error.message.as_deref().unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>()
                .join(" | ")
        })
    }

    /// The same shapes under the default profile (a self-hosted model) are
    /// not judged here: strict schemas without the pin, an unprompted
    /// `json_object` and a loose tool history go through to the engine,
    /// while the structural rules of core validation still apply.
    #[test]
    fn the_default_profile_passes_the_openai_only_rules_through() {
        let generic = |extra: Value| {
            let mut value = json!({
                "model": "qwen3-8b",
                "messages": [{"role": "user", "content": "Say hi in one word."}],
                "max_completion_tokens": 64
            });
            value
                .as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            serde_json::from_value::<ChatCompletionRequest>(value).expect("request deserializes")
        };
        assert_eq!(
            super::super::ProviderProfile::for_model("qwen3-8b"),
            super::super::ProviderProfile::Generic
        );
        let strict_open = generic(json!({"tools": [{"type": "function", "function": {
            "name": "sub", "strict": true,
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
        }}]}));
        assert_eq!(code(&strict_open), None);
        let strict_format = generic(
            json!({"response_format": {"type": "json_schema", "json_schema": {
            "name": "person", "strict": true,
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}}),
        );
        assert_eq!(code(&strict_format), None);
        let unprompted = generic(json!({"response_format": {"type": "json_object"}}));
        assert_eq!(code(&unprompted), None);
        let loose_history = generic(json!({"messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "tool", "tool_call_id": "call_1", "content": "18C"}
        ]}));
        assert_eq!(code(&loose_history), None);
        // Structural rules are not profile-scoped.
        let bad_parameters = generic(
            json!({"tools": [{"type": "function", "function": {"name": "f", "parameters": "oops"}}]}),
        );
        assert!(code(&bad_parameters).is_some_and(|e| e.contains("invalid_type")));
    }

    #[test]
    fn json_object_needs_the_word_json_in_the_messages() {
        let bare = request(json!({"response_format": {"type": "json_object"}}));
        assert!(
            code(&bare).is_some_and(|e| e.contains("json_object_requires_json_in_messages")),
            "{:?}",
            code(&bare)
        );

        let told = request(json!({
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Answer as JSON."},
                {"role": "user", "content": "Say hi in one word."}
            ]
        }));
        assert_eq!(code(&told), None);

        let in_parts = request(json!({
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "reply in json"}]}]
        }));
        assert_eq!(code(&in_parts), None);

        // Only json_object carries the rule.
        let text = request(json!({"response_format": {"type": "text"}}));
        assert_eq!(code(&text), None);
    }

    #[test]
    fn strict_json_schema_needs_additional_properties_false_on_every_object() {
        let strict = |schema: Value| {
            request(
                json!({"response_format": {"type": "json_schema", "json_schema": {
                "name": "person", "strict": true, "schema": schema}}}),
            )
        };
        let root_open = strict(
            json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
        );
        assert!(code(&root_open).is_some_and(|e| e.contains("invalid_json_schema")));

        let nested_open = strict(json!({
            "type": "object", "additionalProperties": false, "required": ["address"],
            "properties": {"address": {"type": "object", "properties": {"city": {"type": "string"}}}}
        }));
        let error = code(&nested_open).expect("rejected");
        assert!(error.contains("('properties', 'address')"), "{error}");

        let closed = strict(json!({
            "type": "object", "additionalProperties": false, "required": ["tags", "address"],
            "properties": {
                "tags": {"type": "array", "items": {"type": "object", "additionalProperties": false, "properties": {}}},
                "address": {"anyOf": [{"type": "object", "additionalProperties": false, "properties": {}}, {"type": "null"}]}
            }
        }));
        assert_eq!(code(&closed), None);

        // Not strict: the schema is the caller's business.
        let lax = request(
            json!({"response_format": {"type": "json_schema", "json_schema": {
            "name": "person", "schema": {"type": "object", "properties": {}}}}}),
        );
        assert_eq!(code(&lax), None);
    }

    #[test]
    fn strict_tool_parameters_need_additional_properties_false() {
        let tool = |strict: bool, parameters: Value| {
            request(json!({"tools": [{"type": "function", "function": {
                "name": "get_weather", "strict": strict, "parameters": parameters}}]}))
        };
        let open = json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]});
        assert!(code(&tool(true, open.clone()))
            .is_some_and(|e| e.contains("invalid_function_parameters")));
        assert_eq!(code(&tool(false, open)), None);
        let closed = json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"], "additionalProperties": false});
        assert_eq!(code(&tool(true, closed)), None);
    }
    fn weather_call(id: &str) -> Value {
        json!({"role": "assistant", "content": null, "tool_calls": [{
            "id": id, "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
        }]})
    }

    #[test]
    fn a_tool_message_answers_the_preceding_tool_calls() {
        let round_trip = request(json!({"messages": [
            {"role": "user", "content": "Weather in Paris?"},
            weather_call("call_1"),
            {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
            {"role": "user", "content": "And tomorrow?"}
        ]}));
        assert_eq!(code(&round_trip), None);

        let two_calls_two_answers = request(json!({"messages": [
            {"role": "user", "content": "Weather in Paris and Rome?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "b", "content": "20C"},
            {"role": "tool", "tool_call_id": "a", "content": "18C"}
        ]}));
        assert_eq!(code(&two_calls_two_answers), None);

        let without_call = request(json!({"messages": [
            {"role": "user", "content": "Weather?"},
            {"role": "tool", "tool_call_id": "call_1", "content": "18C"}
        ]}));
        let error = code(&without_call).expect("rejected");
        assert!(
            error.contains("tool_message_without_tool_calls")
                && error.contains("messages.[1].role"),
            "{error}"
        );

        let wrong_id = request(json!({"messages": [
            {"role": "user", "content": "Weather?"},
            weather_call("call_1"),
            {"role": "tool", "tool_call_id": "call_9", "content": "18C"}
        ]}));
        let error = code(&wrong_id).expect("rejected");
        assert!(
            error.contains("tool_call_id_not_found") && error.contains("messages.[2].tool_call_id"),
            "{error}"
        );

        let unanswered = request(json!({"messages": [
            {"role": "user", "content": "Weather?"},
            weather_call("call_1"),
            {"role": "user", "content": "Never mind."}
        ]}));
        let error = code(&unanswered).expect("rejected");
        assert!(
            error.contains("tool_calls_without_tool_messages")
                && error.contains("messages.[1].role"),
            "{error}"
        );

        let trailing = request(json!({"messages": [
            {"role": "user", "content": "Weather?"},
            weather_call("call_1")
        ]}));
        assert!(code(&trailing).is_some_and(|e| e.contains("tool_calls_without_tool_messages")));
    }
}
