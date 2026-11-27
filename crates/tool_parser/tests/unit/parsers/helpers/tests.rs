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
