use std::collections::HashMap;

use openai_protocol::common::Tool;
use serde::de::{Deserialize, IgnoredAny};
use serde_json::{de::Deserializer, Value};

use crate::{
    errors::{ParserError, ParserResult},
    types::{StreamingParseResult, ToolCallItem},
};

/// `param_name -> declared JSON-schema type` for the named function (empty if the
/// function or its `properties` are absent). Lets XML-style parsers coerce by the
/// declared type instead of guessing from text (e.g. keep a numeric-looking
/// `string` as a string). For simple unions, a string branch takes precedence;
/// otherwise the first non-null scalar type is used. `$ref` resolution remains
/// the responsibility of the caller.
pub fn param_types_for_function(tools: &[Tool], func_name: &str) -> HashMap<String, String> {
    let mut types = HashMap::new();
    let Some(tool) = tools.iter().find(|t| t.function.name == func_name) else {
        return types;
    };
    if let Some(props) = tool
        .function
        .parameters
        .get("properties")
        .and_then(Value::as_object)
    {
        for (key, schema) in props {
            if let Some(ty) = declared_schema_type(schema) {
                types.insert(key.clone(), ty);
            }
        }
    }
    types
}

fn declared_schema_type(schema: &Value) -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();

    match schema.get("type") {
        Some(Value::String(ty)) => candidates.push(ty.clone()),
        Some(Value::Array(types)) => candidates.extend(
            types
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string),
        ),
        _ => {}
    }

    for keyword in ["anyOf", "oneOf"] {
        if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
            candidates.extend(branches.iter().filter_map(declared_schema_type));
        }
    }

    candidates
        .iter()
        .find(|ty| ty.as_str() == "string")
        .or_else(|| candidates.iter().find(|ty| ty.as_str() != "null"))
        .cloned()
}

/// Coerce a raw value by its declared JSON-schema type. For `string`, a JSON
/// string literal (`"4"`) is unwrapped while bare text (`4`, `true`) is kept
/// verbatim; other types are parsed. `None` (unknown type or parse failure)
/// means the caller should infer.
pub fn coerce_by_schema_type(text: &str, declared_type: Option<&str>) -> Option<Value> {
    match declared_type? {
        "string" => Some(Value::String(
            serde_json::from_str::<String>(text).unwrap_or_else(|_| text.to_string()),
        )),
        "integer" => text
            .trim()
            .parse::<i64>()
            .ok()
            .map(|n| Value::Number(n.into())),
        // Parse as JSON so large integers keep their precision (a plain f64 parse
        // would round e.g. 9007199254740993).
        "number" => serde_json::from_str::<Value>(text.trim())
            .ok()
            .filter(Value::is_number),
        "boolean" => match text.trim() {
            "true" | "True" => Some(Value::Bool(true)),
            "false" | "False" => Some(Value::Bool(false)),
            _ => None,
        },
        "object" | "array" => serde_json::from_str::<Value>(text).ok(),
        _ => None,
    }
}

/// Get a mapping of tool names to their indices
pub fn get_tool_indices(tools: &[Tool]) -> HashMap<String, usize> {
    tools
        .iter()
        .enumerate()
        .map(|(i, tool)| (tool.function.name.clone(), i))
        .collect()
}

/// Find the common prefix of two strings
/// Used for incremental argument streaming when partial JSON returns different intermediate states
pub fn find_common_prefix(s1: &str, s2: &str) -> String {
    s1.chars()
        .zip(s2.chars())
        .take_while(|(c1, c2)| c1 == c2)
        .map(|(c1, _)| c1)
        .collect()
}

/// Get unstreamed tool call arguments
/// Returns tool call items for arguments that have been parsed but not yet streamed
/// This ensures tool calls are properly completed even if the model generates final arguments in the last chunk
pub fn get_unstreamed_args(
    prev_tool_call_arr: &[Value],
    streamed_args_for_tool: &[String],
) -> Option<Vec<ToolCallItem>> {
    // Check if we have tool calls being tracked
    if prev_tool_call_arr.is_empty() || streamed_args_for_tool.is_empty() {
        return None;
    }

    // Get the last tool call that was being processed
    let tool_index = prev_tool_call_arr.len() - 1;
    if tool_index >= streamed_args_for_tool.len() {
        return None;
    }

    // Get expected vs actual arguments
    let expected_args = prev_tool_call_arr[tool_index].get("arguments")?;
    let expected_str = serde_json::to_string(expected_args).ok()?;
    let actual_str = &streamed_args_for_tool[tool_index];

    // Check if there are remaining arguments to send
    let remaining = if expected_str.starts_with(actual_str) {
        &expected_str[actual_str.len()..]
    } else {
        return None;
    };

    if remaining.is_empty() {
        return None;
    }

    // Return the remaining arguments as a ToolCallItem
    Some(vec![ToolCallItem {
        tool_index,
        name: None, // No name for argument deltas
        parameters: remaining.to_string(),
    }])
}

/// Check if a buffer ends with a partial occurrence of a token
/// Returns Some(length) if there's a partial match, None otherwise
pub fn ends_with_partial_token(buffer: &str, token: &str) -> Option<usize> {
    if buffer.is_empty() || token.is_empty() {
        return None;
    }

    token
        .char_indices()
        .skip(1)
        .map(|(i, _)| i)
        .find(|&i| buffer.ends_with(&token[..i]))
}

/// Reset state for the current tool being parsed (used when skipping invalid tools).
/// This preserves the parser's overall state (current_tool_id, prev_tool_call_arr)
/// but clears the state specific to the current incomplete tool.
pub fn reset_current_tool_state(
    buffer: &mut String,
    current_tool_name_sent: &mut bool,
    streamed_args_for_tool: &mut Vec<String>,
    prev_tool_call_arr: &[Value],
) {
    buffer.clear();
    *current_tool_name_sent = false;

    // Only pop if we added an entry for the current (invalid) tool
    // streamed_args_for_tool should match prev_tool_call_arr length for completed tools
    if streamed_args_for_tool.len() > prev_tool_call_arr.len() {
        streamed_args_for_tool.pop();
    }
}

/// Reset the entire parser state (used at the start of a new request).
/// Clears all accumulated tool calls and resets all state to initial values.
pub fn reset_parser_state(
    buffer: &mut String,
    prev_tool_call_arr: &mut Vec<Value>,
    current_tool_id: &mut i32,
    current_tool_name_sent: &mut bool,
    streamed_args_for_tool: &mut Vec<String>,
) {
    buffer.clear();
    prev_tool_call_arr.clear();
    *current_tool_id = -1;
    *current_tool_name_sent = false;
    streamed_args_for_tool.clear();
}

/// Ensure arrays have capacity for the given tool ID
pub fn ensure_capacity(
    current_tool_id: i32,
    prev_tool_call_arr: &mut Vec<Value>,
    streamed_args_for_tool: &mut Vec<String>,
) {
    if current_tool_id < 0 {
        return;
    }
    let needed = (current_tool_id + 1) as usize;

    if prev_tool_call_arr.len() < needed {
        prev_tool_call_arr.resize_with(needed, || Value::Null);
    }
    if streamed_args_for_tool.len() < needed {
        streamed_args_for_tool.resize_with(needed, String::new);
    }
}

/// Check if a string contains complete, valid JSON
pub fn is_complete_json(input: &str) -> bool {
    let mut de = Deserializer::from_str(input);
    IgnoredAny::deserialize(&mut de).is_ok() && de.end().is_ok()
}

/// If the object has "parameters" but not "arguments", copy parameters to arguments.
pub fn normalize_arguments_field(mut obj: Value) -> Value {
    if obj.get("arguments").is_none() {
        if let Some(params) = obj.get("parameters").cloned() {
            if let Value::Object(ref mut map) = obj {
                map.insert("arguments".to_string(), params);
            }
        }
    }
    obj
}

/// If the object has "tool_name" but not "name", copy tool_name to name.
pub fn normalize_name_field(mut obj: Value) -> Value {
    if obj.get("name").is_none() {
        if let Some(tool_name) = obj.get("tool_name").cloned() {
            if let Value::Object(ref mut map) = obj {
                map.insert("name".to_string(), tool_name);
            }
        }
    }
    obj
}

/// Normalize both the name and arguments fields (e.g. Cohere's "tool_name"/"parameters").
pub fn normalize_tool_call_fields(obj: Value) -> Value {
    let obj = normalize_name_field(obj);
    normalize_arguments_field(obj)
}

/// Handle JSON tool-call streaming (parse partial JSON, validate names, stream the
/// name then arguments, advance the buffer) for the JSON, Llama, Mistral, and Qwen
/// parsers. `start_idx` is where JSON begins in `current_text`; `current_tool_id ==
/// -1` means no active tool.
#[expect(clippy::too_many_arguments)]
pub(crate) fn handle_json_tool_streaming(
    current_text: &str,
    start_idx: usize,
    partial_json: &mut crate::partial_json::PartialJson,
    tool_indices: &HashMap<String, usize>,
    buffer: &mut String,
    current_tool_id: &mut i32,
    current_tool_name_sent: &mut bool,
    streamed_args_for_tool: &mut Vec<String>,
    prev_tool_call_arr: &mut Vec<Value>,
) -> ParserResult<StreamingParseResult> {
    // Check if we have content to parse
    if start_idx >= current_text.len() {
        return Ok(StreamingParseResult::default());
    }

    // Extract JSON string from current position
    let json_str = &current_text[start_idx..];

    // When current_tool_name_sent is false, don't allow partial strings to avoid
    // parsing incomplete tool names as empty strings
    let allow_partial_strings = *current_tool_name_sent;

    // Parse partial JSON
    let (obj, end_idx) = match partial_json.parse_value(json_str, allow_partial_strings) {
        Ok(result) => result,
        Err(_) => {
            return Ok(StreamingParseResult::default());
        }
    };

    // Check if JSON is complete - validate only the parsed portion
    // Ensure end_idx is on a valid UTF-8 character boundary
    let safe_end_idx = if json_str.is_char_boundary(end_idx) {
        end_idx
    } else {
        // Find the nearest valid character boundary before end_idx
        (0..end_idx)
            .rev()
            .find(|&i| json_str.is_char_boundary(i))
            .unwrap_or(0)
    };
    let is_complete = is_complete_json(&json_str[..safe_end_idx]);

    // Normalize all tool call fields first (handles tool_name -> name, parameters -> arguments)
    // This must happen before validation since different LLMs use different field names
    let current_tool_call = normalize_tool_call_fields(obj);

    // Validate tool name if present
    if let Some(name) = current_tool_call.get("name").and_then(|v| v.as_str()) {
        if !tool_indices.contains_key(name) {
            // Invalid tool name - skip this tool, preserve indexing for next tool
            tracing::debug!("Invalid tool name '{}' - skipping", name);
            reset_current_tool_state(
                buffer,
                current_tool_name_sent,
                streamed_args_for_tool,
                prev_tool_call_arr,
            );
            return Ok(StreamingParseResult::default());
        }
    }

    let mut result = StreamingParseResult::default();

    // Case 1: Handle tool name streaming
    if !*current_tool_name_sent {
        if let Some(function_name) = current_tool_call.get("name").and_then(|v| v.as_str()) {
            if tool_indices.contains_key(function_name) {
                // Initialize if first tool
                if *current_tool_id == -1 {
                    *current_tool_id = 0;
                    streamed_args_for_tool.push(String::new());
                } else if *current_tool_id as usize >= streamed_args_for_tool.len() {
                    // Ensure capacity for subsequent tools
                    ensure_capacity(*current_tool_id, prev_tool_call_arr, streamed_args_for_tool);
                }

                // Send tool name with empty parameters
                *current_tool_name_sent = true;
                result.calls.push(ToolCallItem {
                    tool_index: *current_tool_id as usize,
                    name: Some(function_name.to_string()),
                    parameters: String::new(),
                });
            }
        }
    }
    // Case 2: Handle streaming arguments
    else if let Some(cur_arguments) = current_tool_call.get("arguments") {
        let tool_id = *current_tool_id as usize;
        let sent = streamed_args_for_tool
            .get(tool_id)
            .map(|s| s.len())
            .unwrap_or(0);
        let cur_args_json = serde_json::to_string(cur_arguments)
            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        // Get prev_arguments (matches Python's structure)
        let prev_arguments = if tool_id < prev_tool_call_arr.len() {
            prev_tool_call_arr[tool_id].get("arguments")
        } else {
            None
        };

        // Calculate diff: everything after we've already sent
        let mut argument_diff = None;

        if is_complete {
            // Python: argument_diff = cur_args_json[sent:]
            // Rust needs bounds check (Python returns "" automatically)
            argument_diff = if sent < cur_args_json.len() {
                Some(cur_args_json[sent..].to_string())
            } else {
                Some(String::new())
            };
        } else if let Some(prev_args) = prev_arguments {
            let prev_args_json = serde_json::to_string(prev_args)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

            if cur_args_json != prev_args_json {
                let prefix = find_common_prefix(&prev_args_json, &cur_args_json);
                argument_diff = if sent < prefix.len() {
                    Some(prefix[sent..].to_string())
                } else {
                    Some(String::new())
                };
            }
        }

        // Send diff if present
        if let Some(diff) = argument_diff {
            if !diff.is_empty() {
                if tool_id < streamed_args_for_tool.len() {
                    streamed_args_for_tool[tool_id].push_str(&diff);
                }
                result.calls.push(ToolCallItem {
                    tool_index: tool_id,
                    name: None,
                    parameters: diff,
                });
            }
        }

        // Update prev_tool_call_arr with current state
        if *current_tool_id >= 0 {
            ensure_capacity(*current_tool_id, prev_tool_call_arr, streamed_args_for_tool);

            if tool_id < prev_tool_call_arr.len() {
                prev_tool_call_arr[tool_id] = current_tool_call;
            }
        }

        // If complete, advance to next tool
        if is_complete {
            *buffer = current_text[start_idx + end_idx..].to_string();
            *current_tool_name_sent = false;
            *current_tool_id += 1;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ends_with_partial_token() {
        assert!(ends_with_partial_token("hello <|py", "<|python_tag|>").is_some());
        assert!(ends_with_partial_token("hello <|python_tag", "<|python_tag|>").is_some());
        assert!(ends_with_partial_token("hello <|python_tag|>", "<|python_tag|>").is_none());
        assert!(ends_with_partial_token("", "<|python_tag|>").is_none());
        assert!(ends_with_partial_token("hello world", "<|python_tag|>").is_none());
    }

    #[test]
    fn test_ends_with_partial_token_multibyte() {
        // U+FF5C "｜" is 3 bytes, so token[..i] must only be sliced at char boundaries.
        let token = "<｜tool_calls_begin｜>";

        // Buffer not ending in '<' must not panic and yields no partial match.
        assert_eq!(
            ends_with_partial_token("some plain ascii text", token),
            None
        );

        // Ending in '<' matches a 1-byte prefix.
        assert_eq!(ends_with_partial_token("hello <", token), Some(1));

        // Ending in "<｜" matches the prefix up to the next char boundary (1 + 3 bytes).
        assert_eq!(ends_with_partial_token("hello <｜", token), Some(4));

        // A complete token at the end is not a partial match.
        assert_eq!(
            ends_with_partial_token("x <｜tool_calls_begin｜>", token),
            None
        );
    }

    #[test]
    fn test_reset_current_tool_state() {
        let mut buffer = String::from("partial json");
        let mut current_tool_name_sent = true;
        let mut streamed_args = vec!["tool0_args".to_string(), "tool1_partial".to_string()];
        let prev_tools = vec![serde_json::json!({"name": "tool0"})];

        reset_current_tool_state(
            &mut buffer,
            &mut current_tool_name_sent,
            &mut streamed_args,
            &prev_tools,
        );

        assert_eq!(buffer, "");
        assert!(!current_tool_name_sent);
        assert_eq!(streamed_args.len(), 1); // Popped the partial tool1 args
        assert_eq!(streamed_args[0], "tool0_args");
    }

    #[test]
    fn test_reset_current_tool_state_no_pop_when_synced() {
        let mut buffer = String::from("partial json");
        let mut current_tool_name_sent = true;
        let mut streamed_args = vec!["tool0_args".to_string()];
        let prev_tools = vec![serde_json::json!({"name": "tool0"})];

        reset_current_tool_state(
            &mut buffer,
            &mut current_tool_name_sent,
            &mut streamed_args,
            &prev_tools,
        );

        assert_eq!(buffer, "");
        assert!(!current_tool_name_sent);
        assert_eq!(streamed_args.len(), 1); // No pop, lengths matched
    }

    #[test]
    fn test_reset_parser_state() {
        let mut buffer = String::from("some buffer");
        let mut prev_tools = vec![serde_json::json!({"name": "tool0"})];
        let mut current_tool_id = 5;
        let mut current_tool_name_sent = true;
        let mut streamed_args = vec!["args".to_string()];

        reset_parser_state(
            &mut buffer,
            &mut prev_tools,
            &mut current_tool_id,
            &mut current_tool_name_sent,
            &mut streamed_args,
        );

        assert_eq!(buffer, "");
        assert_eq!(prev_tools.len(), 0);
        assert_eq!(current_tool_id, -1);
        assert!(!current_tool_name_sent);
        assert_eq!(streamed_args.len(), 0);
    }

    #[test]
    fn test_ensure_capacity() {
        let mut prev_tools = vec![];
        let mut streamed_args = vec![];

        ensure_capacity(2, &mut prev_tools, &mut streamed_args);

        assert_eq!(prev_tools.len(), 3);
        assert_eq!(streamed_args.len(), 3);
        assert_eq!(prev_tools[0], Value::Null);
        assert_eq!(streamed_args[0], "");
    }

    #[test]
    fn test_ensure_capacity_negative_id() {
        let mut prev_tools = vec![];
        let mut streamed_args = vec![];

        ensure_capacity(-1, &mut prev_tools, &mut streamed_args);

        // Should not resize for negative ID
        assert_eq!(prev_tools.len(), 0);
        assert_eq!(streamed_args.len(), 0);
    }

    #[test]
    fn test_is_complete_json() {
        assert!(is_complete_json(r#"{"name": "test"}"#));
        assert!(is_complete_json("[1, 2, 3]"));
        assert!(is_complete_json("42"));
        assert!(is_complete_json("true"));
        assert!(!is_complete_json(r#"{"name": "#));
        assert!(!is_complete_json("[1, 2,"));
    }

    #[test]
    fn test_normalize_arguments_field() {
        // Case 1: Has parameters, no arguments
        let obj = serde_json::json!({
            "name": "test",
            "parameters": {"key": "value"}
        });
        let normalized = normalize_arguments_field(obj);
        assert_eq!(
            normalized.get("arguments").unwrap(),
            &serde_json::json!({"key": "value"})
        );

        // Case 2: Already has arguments
        let obj = serde_json::json!({
            "name": "test",
            "arguments": {"key": "value"}
        });
        let normalized = normalize_arguments_field(obj.clone());
        assert_eq!(normalized, obj);

        // Case 3: No parameters or arguments
        let obj = serde_json::json!({"name": "test"});
        let normalized = normalize_arguments_field(obj.clone());
        assert_eq!(normalized, obj);
    }

    #[test]
    fn test_normalize_name_field() {
        // Case 1: Has tool_name, no name (Cohere format)
        let obj = serde_json::json!({
            "tool_name": "search",
            "parameters": {"query": "test"}
        });
        let normalized = normalize_name_field(obj);
        assert_eq!(normalized.get("name").unwrap(), "search");

        // Case 2: Already has name (standard format)
        let obj = serde_json::json!({
            "name": "test",
            "arguments": {"key": "value"}
        });
        let normalized = normalize_name_field(obj.clone());
        assert_eq!(normalized, obj);

        // Case 3: Has both tool_name and name - name takes precedence
        let obj = serde_json::json!({
            "tool_name": "cohere_name",
            "name": "standard_name",
            "parameters": {}
        });
        let normalized = normalize_name_field(obj);
        assert_eq!(normalized.get("name").unwrap(), "standard_name");

        // Case 4: No name or tool_name
        let obj = serde_json::json!({"parameters": {}});
        let normalized = normalize_name_field(obj.clone());
        assert!(normalized.get("name").is_none());
    }

    #[test]
    fn test_normalize_tool_call_fields() {
        // Case 1: Full Cohere format with tool_name and parameters
        let obj = serde_json::json!({
            "tool_name": "search",
            "parameters": {"query": "rust programming"}
        });
        let normalized = normalize_tool_call_fields(obj);
        assert_eq!(normalized.get("name").unwrap(), "search");
        assert_eq!(
            normalized.get("arguments").unwrap(),
            &serde_json::json!({"query": "rust programming"})
        );

        // Case 2: Standard format - should remain unchanged
        let obj = serde_json::json!({
            "name": "test",
            "arguments": {"key": "value"}
        });
        let normalized = normalize_tool_call_fields(obj.clone());
        assert_eq!(normalized, obj);

        // Case 3: Mixed format (name + parameters)
        let obj = serde_json::json!({
            "name": "test",
            "parameters": {"key": "value"}
        });
        let normalized = normalize_tool_call_fields(obj);
        assert_eq!(normalized.get("name").unwrap(), "test");
        assert_eq!(
            normalized.get("arguments").unwrap(),
            &serde_json::json!({"key": "value"})
        );
    }

    #[test]
    fn test_coerce_by_schema_type() {
        use serde_json::json;
        // string: numeric/bool/array-looking text stays a string; a JSON string
        // literal is unwrapped.
        assert_eq!(coerce_by_schema_type("4", Some("string")), Some(json!("4")));
        assert_eq!(
            coerce_by_schema_type("true", Some("string")),
            Some(json!("true"))
        );
        assert_eq!(
            coerce_by_schema_type("[60,30]", Some("string")),
            Some(json!("[60,30]"))
        );
        assert_eq!(
            coerce_by_schema_type("\"4\"", Some("string")),
            Some(json!("4"))
        );

        assert_eq!(coerce_by_schema_type("4", Some("integer")), Some(json!(4)));
        assert_eq!(
            coerce_by_schema_type("true", Some("boolean")),
            Some(json!(true))
        );
        assert_eq!(
            coerce_by_schema_type("[60,30]", Some("array")),
            Some(json!([60, 30]))
        );
        // number keeps integer precision (no f64 rounding).
        assert_eq!(
            coerce_by_schema_type("9007199254740993", Some("number")),
            Some(json!(9007199254740993i64))
        );

        // unknown type / value that fails to parse -> None (caller infers)
        assert_eq!(coerce_by_schema_type("4", None), None);
        assert_eq!(coerce_by_schema_type("abc", Some("integer")), None);
    }
}
