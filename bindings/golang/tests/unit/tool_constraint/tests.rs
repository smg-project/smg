use serde_json::{json, Value};

use super::*;

/// A request with two tools, `get_weather` second.
fn request(tool_choice: Value) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "unmapped-model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "function": {"name": "get_time", "parameters":
                {"type": "object", "properties": {"zone": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "get_weather", "parameters":
                {"type": "object", "properties": {"city": {"type": "string"}}}}}
        ],
        "tool_choice": tool_choice
    }))
    .unwrap()
}

/// A request for a provider profile that declares tools on messages:
/// `get_weather` on a system message, `tools` as given at the request
/// level (a list, or null for none).
fn request_with_message_tool(tools: Value, tool_choice: Value) -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "kimi-k3",
        "messages": [
            {"role": "system", "content": "", "tools": [
                {"type": "function", "function": {"name": "get_weather", "parameters":
                    {"type": "object", "properties": {"city": {"type": "string"}}}}}
            ]},
            {"role": "user", "content": "hi"}
        ],
        "tools": tools,
        "tool_choice": tool_choice
    }))
    .unwrap()
}

fn time_tool() -> Value {
    json!([{"type": "function", "function": {"name": "get_time", "parameters":
        {"type": "object", "properties": {"zone": {"type": "string"}}}}}])
}

fn schema_of(request: &ChatCompletionRequest) -> Value {
    match tool_constraint_for(request).unwrap() {
        Some((kind, schema)) if kind == "json_schema" => serde_json::from_str(&schema).unwrap(),
        other => panic!("expected a JSON schema constraint, got {other:?}"),
    }
}

fn tool_names(json: &str) -> Vec<String> {
    serde_json::from_str::<Vec<Tool>>(json)
        .unwrap()
        .into_iter()
        .map(|tool| tool.function.name)
        .collect()
}

/// A choice naming the second tool is constrained with that tool's
/// parameter schema, not the first tool's.
#[test]
fn a_named_choice_is_constrained_with_the_named_functions_schema() {
    let schema = schema_of(&request(
        json!({"type": "function", "function": {"name": "get_weather"}}),
    ));
    assert_eq!(
        schema,
        json!({"type": "object", "properties": {"city": {"type": "string"}}})
    );
}

/// A choice that needs a tool on a request without tools is refused with
/// the gateway's words; `auto` and `none` without tools constrain nothing.
#[test]
fn a_choice_that_needs_a_tool_is_refused_without_tools() {
    let without_tools = |tool_choice: Value| -> ChatCompletionRequest {
        serde_json::from_value(json!({
            "model": "unmapped-model",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": tool_choice
        }))
        .unwrap()
    };
    for tool_choice in [
        json!("required"),
        json!({"type": "function", "function": {"name": "get_weather"}}),
        json!({"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "get_weather"}]}),
    ] {
        let error = tool_constraint_for(&without_tools(tool_choice.clone())).unwrap_err();
        assert!(
            error.contains("'tool_choice' is only allowed when 'tools' are specified"),
            "{tool_choice}: {error}"
        );
    }
    for tool_choice in [json!("auto"), json!("none")] {
        assert_eq!(
            tool_constraint_for(&without_tools(tool_choice.clone())).unwrap(),
            None,
            "{tool_choice}"
        );
    }
}

/// `required` keeps every tool: the forced array may name either.
#[test]
fn a_required_choice_keeps_every_tool() {
    let schema = schema_of(&request(json!("required"))).to_string();
    assert!(
        schema.contains("get_time") && schema.contains("get_weather"),
        "{schema}"
    );
}

/// `auto` forces nothing.
#[test]
fn an_auto_choice_forces_nothing() {
    assert!(tool_constraint_for(&request(json!("auto")))
        .unwrap()
        .is_none());
}

/// A choice that leaves no tool to call (a function the request does not
/// declare, an allowed-tools list naming none of its tools) is an error,
/// not a request served without a constraint.
#[test]
fn a_choice_that_names_no_declared_tool_is_an_error() {
    let undeclared = request(json!({"type": "function", "function": {"name": "get_news"}}));
    let error = tool_constraint_for(&undeclared).unwrap_err();
    assert!(error.contains("'get_news' not found in 'tools'"), "{error}");
    let allowed = request(json!({
        "type": "allowed_tools", "mode": "required",
        "tools": [{"type": "function", "name": "get_news"}]
    }));
    let error = tool_constraint_for(&allowed).unwrap_err();
    assert!(error.contains("not found in 'tools'"), "{error}");
}

/// `required` constrains to every tool the model sees: the request-level
/// one and the message-declared one.
#[test]
fn a_required_choice_keeps_a_message_declared_tool() {
    let schema = schema_of(&request_with_message_tool(time_tool(), json!("required"))).to_string();
    assert!(
        schema.contains("get_time") && schema.contains("get_weather"),
        "{schema}"
    );
}

/// A choice naming a message-declared tool is constrained with its schema.
#[test]
fn a_choice_naming_a_message_declared_tool_is_constrained_with_its_schema() {
    let schema = schema_of(&request_with_message_tool(
        time_tool(),
        json!({"type": "function", "function": {"name": "get_weather"}}),
    ));
    assert_eq!(
        schema,
        json!({"type": "object", "properties": {"city": {"type": "string"}}})
    );
}

/// A request whose only tool is message-declared is constrained too.
#[test]
fn a_request_whose_only_tool_is_message_declared_is_constrained() {
    let schema = schema_of(&request_with_message_tool(Value::Null, json!("required"))).to_string();
    assert!(schema.contains("get_weather"), "{schema}");
}

/// The engine keeps its special tokens for the parser when tools are
/// declared at either level and nothing forces plain JSON: a request whose
/// only tool is message-declared keeps them; a JSON-schema constraint, a
/// `none` choice, no tools at all or an empty `tools` list (what the Go
/// wrappers send when the caller sets none) do not.
#[test]
fn special_tokens_are_kept_for_message_declared_tools() {
    let only_message_tool = request_with_message_tool(Value::Null, json!("auto"));
    assert!(tools_keep_special_tokens(&only_message_tool, None));
    let json_schema = ("json_schema".to_string(), "{}".to_string());
    assert!(!tools_keep_special_tokens(
        &only_message_tool,
        Some(&json_schema)
    ));
    let none_choice = request_with_message_tool(Value::Null, json!("none"));
    assert!(!tools_keep_special_tokens(&none_choice, None));
    let no_tools: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    assert!(!tools_keep_special_tokens(&no_tools, None));
    let empty_tools: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}], "tools": []
    }))
    .unwrap();
    assert!(!tools_keep_special_tokens(&empty_tools, None));
}

/// The response parser's tool list is the same set: the request-level
/// tools first, then the message-declared ones; none when the request
/// declares none.
#[test]
fn the_parser_tools_include_the_message_declared_tool() {
    let json =
        effective_tools_json(&request_with_message_tool(time_tool(), json!("auto"))).unwrap();
    assert_eq!(tool_names(&json), ["get_time", "get_weather"]);
    let json =
        effective_tools_json(&request_with_message_tool(Value::Null, json!("auto"))).unwrap();
    assert_eq!(tool_names(&json), ["get_weather"]);
    let none: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "kimi-k3", "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    assert!(effective_tools_json(&none).is_none());
}
