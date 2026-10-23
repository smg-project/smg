//! The tools of a request as the bindings read them: the decoding constraint
//! of its tool choice, and the tool list its response parser resolves calls
//! against.

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{Tool, ToolChoice, ToolChoiceValue},
};
use tool_parser::ToolConstraint;

use super::runtime::PARSER_FACTORY;

/// Every tool the model sees: the request-level `tools` followed by the tools
/// declared on messages, as the request's provider profile defines them (the
/// gateway's `effective_tools`). The constraint and the response parser both
/// work from this set, so a forced call can land on a message-declared tool
/// and a call to one comes back as a call, not as text.
fn effective_tools(request: &ChatCompletionRequest) -> Vec<Tool> {
    request.effective_tools().cloned().collect()
}

/// The tools the response parser resolves calls against, as the JSON array
/// the converter takes; `None` when the request declares none.
pub(crate) fn effective_tools_json(request: &ChatCompletionRequest) -> Option<String> {
    let tools = effective_tools(request);
    if tools.is_empty() {
        return None;
    }
    serde_json::to_string(&tools).ok()
}

/// Whether the engine must keep its special tokens for the response parser:
/// tools at either level (request-level or message-declared), a tool choice
/// that allows calls, and no JSON-schema constraint (whose output is plain
/// JSON from the first token, with no markers to keep). The request paths
/// derive `skip_special_tokens = false` from it.
pub(crate) fn tools_keep_special_tokens(
    request: &ChatCompletionRequest,
    constraint: Option<&(String, String)>,
) -> bool {
    constraint.is_none_or(|(kind, _)| kind != "json_schema")
        && !effective_tools(request).is_empty()
        && !matches!(
            request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        )
}

/// The constraint the engine is held to for the request's tool choice, built
/// from the request's effective tools narrowed by the choice (the named
/// function, an allowed-tools list), the gateway's callable set. Built from
/// `tools` as a whole, a named choice is constrained with the first tool's
/// schema. `None` when the request declares no tool or its choice forces no
/// call; an error when the choice narrows the tools to none (a function the
/// request does not declare, an allowed-tools list naming none of them),
/// which the gateway's validation rejects.
pub(crate) fn tool_constraint_for(
    request: &ChatCompletionRequest,
) -> Result<Option<ToolConstraint>, String> {
    let Some(tool_choice) = request.tool_choice.as_ref() else {
        return Ok(None);
    };
    let tools = effective_tools(request);
    if tools.is_empty() {
        return Ok(None);
    }
    let narrowed = tool_choice.narrow_tools(&tools);
    let tools: &[Tool] = match &narrowed {
        Some(narrowed) if narrowed.is_empty() => return Err(narrowed_to_none(tool_choice)),
        Some(narrowed) => narrowed,
        None => &tools,
    };
    // No parser is configured here, so the registry falls back to a JSON
    // schema, which carries no reasoning prefix.
    PARSER_FACTORY
        .registry()
        .generate_tool_constraint(None, tools, tool_choice, false)
}

/// The error of a choice that leaves no tool to call, worded as the
/// gateway's validation words it.
fn narrowed_to_none(tool_choice: &ToolChoice) -> String {
    match tool_choice {
        ToolChoice::Function { function, .. } => format!(
            "Invalid value for 'tool_choice': function '{}' not found in 'tools'.",
            function.name
        ),
        ToolChoice::AllowedTools { .. } => {
            "Invalid value for 'tool_choice.tools': the listed tools are not found in 'tools'."
                .to_string()
        }
        ToolChoice::Value(_) => "Invalid value for 'tool_choice': no tool to call.".to_string(),
    }
}

#[cfg(test)]
mod tests {
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
            Some(ToolConstraint::JsonSchema(schema)) => serde_json::from_str(&schema).unwrap(),
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
        let schema =
            schema_of(&request_with_message_tool(time_tool(), json!("required"))).to_string();
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
        let schema =
            schema_of(&request_with_message_tool(Value::Null, json!("required"))).to_string();
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
}
