//! The decoding constraint of a request's tool choice.

use openai_protocol::{
    chat::ChatCompletionRequest,
    common::{Tool, ToolChoice},
};
use tool_parser::ToolConstraint;

use super::runtime::PARSER_FACTORY;

/// The constraint the engine is held to for the request's tool choice, built
/// from the request's tools narrowed by the choice (the named function, an
/// allowed-tools list), the way the gateway's preparation stage narrows the
/// tools it constrains with; the bindings' response parser reads the same
/// request-level tools. Built from `tools` as a whole, a named choice is
/// constrained with the first tool's schema. `None` when the request has no
/// tools or its choice forces no call; an error when the choice narrows the
/// tools to none (a function the request does not declare, an allowed-tools
/// list naming none of them), which the gateway's validation rejects.
pub(crate) fn tool_constraint_for(
    request: &ChatCompletionRequest,
) -> Result<Option<ToolConstraint>, String> {
    let (Some(tools), Some(tool_choice)) = (request.tools.as_ref(), request.tool_choice.as_ref())
    else {
        return Ok(None);
    };
    let narrowed = tool_choice.narrow_tools(tools);
    let tools: &[Tool] = match &narrowed {
        Some(narrowed) if narrowed.is_empty() => return Err(narrowed_to_none(tool_choice)),
        Some(narrowed) => narrowed,
        None => tools,
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

    fn schema_of(request: &ChatCompletionRequest) -> Value {
        match tool_constraint_for(request).unwrap() {
            Some(ToolConstraint::JsonSchema(schema)) => serde_json::from_str(&schema).unwrap(),
            other => panic!("expected a JSON schema constraint, got {other:?}"),
        }
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
}
