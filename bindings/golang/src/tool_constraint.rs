//! The decoding constraint of a request's tool choice.

use openai_protocol::chat::ChatCompletionRequest;
use tool_parser::ToolConstraint;

use super::runtime::PARSER_FACTORY;

/// The constraint the engine is held to for the request's tool choice, built
/// from the request's callable tools: the tools narrowed to the named function
/// when the choice names one, the set the gateway's preparation stage builds
/// its constraint from. Built from `tools` as a whole, a named choice is
/// constrained with the first tool's schema. `None` when the request has no
/// tools or its choice forces no call.
pub(crate) fn tool_constraint_for(
    request: &ChatCompletionRequest,
) -> Result<Option<ToolConstraint>, String> {
    let (Some(_), Some(tool_choice)) = (request.tools.as_ref(), request.tool_choice.as_ref())
    else {
        return Ok(None);
    };
    let tools = request.callable_tools();
    // No parser is configured here, so the registry falls back to a JSON
    // schema, which carries no reasoning prefix.
    PARSER_FACTORY
        .registry()
        .generate_tool_constraint(None, &tools, tool_choice, false)
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

    /// `auto` forces nothing, and neither does a choice naming a function the
    /// request does not declare (nothing is callable, so nothing is forced).
    #[test]
    fn auto_and_an_undeclared_function_force_nothing() {
        assert!(tool_constraint_for(&request(json!("auto")))
            .unwrap()
            .is_none());
        let undeclared = request(json!({"type": "function", "function": {"name": "get_news"}}));
        assert!(tool_constraint_for(&undeclared).unwrap().is_none());
    }
}
