//! GLM-4.7 MoE Parser Integration Tests
mod common;

use common::create_test_tools;
use openai_protocol::common::{ToolChoice, ToolChoiceValue};
use tool_parser::{Glm4MoeParser, ParserFactory, ToolConstraint, ToolParser};

#[tokio::test]
async fn test_glm47_complete_parsing() {
    let parser = Glm4MoeParser::glm47();

    let input = r"Let me search for that.
<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value><arg_key>date</arg_key><arg_value>2024-12-25</arg_value></tool_call>
The weather will be...";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me search for that.\n");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Beijing");
    assert_eq!(args["date"], "2024-12-25");
}

#[tokio::test]
async fn test_glm47_multiple_tools() {
    let parser = Glm4MoeParser::glm47();

    let input = r"<tool_call>search<arg_key>query</arg_key><arg_value>rust tutorials</arg_value></tool_call><tool_call>translate<arg_key>text</arg_key><arg_value>Hello World</arg_value><arg_key>target_lang</arg_key><arg_value>zh</arg_value></tool_call>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_glm47_type_conversion() {
    let parser = Glm4MoeParser::glm47();

    let input = r"<tool_call>process<arg_key>count</arg_key><arg_value>42</arg_value><arg_key>rate</arg_key><arg_value>1.5</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>data</arg_key><arg_value>null</arg_value><arg_key>text</arg_key><arg_value>string value</arg_value></tool_call>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["count"], 42);
    assert_eq!(args["rate"], 1.5);
    assert_eq!(args["enabled"], true);
    assert_eq!(args["data"], serde_json::Value::Null);
    assert_eq!(args["text"], "string value");
}

#[tokio::test]
async fn test_glm47_streaming() {
    let mut parser = Glm4MoeParser::glm47();

    let tools = create_test_tools();

    // Simulate streaming chunks
    let chunks = vec![
        "<tool_call>",
        "get_weather",
        "<arg_key>city</arg_key>",
        "<arg_value>Shanghai</arg_value>",
        "<arg_key>units</arg_key>",
        "<arg_value>celsius</arg_value>",
        "</tool_call>",
    ];

    let mut found_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
}

#[test]
fn test_glm47_format_detection() {
    let parser = Glm4MoeParser::glm47();

    // Should detect GLM-4 format
    assert!(parser.has_tool_markers("<tool_call>"));
    assert!(parser.has_tool_markers("text with <tool_call> marker"));

    // Should not detect other formats
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<｜tool▁calls▁begin｜>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_glm5_routes_to_glm47_moe() {
    // GLM-5.x must route to glm47_moe, not the catch-all glm-* -> json mapping.
    let factory = ParserFactory::new();
    let input =
        r"<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>";
    for model in ["glm-5", "glm-5.1", "glm-5.2", "glm-5.2-fp8"] {
        let parser = factory
            .registry()
            .create_for_model(model)
            .unwrap_or_else(|| panic!("no parser for {model}"));
        let (_, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1, "{model} should extract one tool call");
        assert_eq!(tools[0].function.name, "get_weather", "{model}");
    }
}

/// #2548: a GLM-4.7-family request that offers no tools must not carry the
/// full-assistant EBNF — an engine launched without a grammar backend
/// (TokenSpeed's default) rejects every constrained request, and plain chat
/// has nothing to constrain. Tools in auto mode still get the grammar, and
/// `tool_choice: none` with tools offered gets the grammar that forbids calls.
#[test]
fn test_glm47_chat_constraint_only_when_tools_are_offered() {
    let factory = ParserFactory::new();
    let registry = factory.registry();
    let parser = Some("glm47_moe");
    let auto = ToolChoice::Value(ToolChoiceValue::Auto);

    let none_offered = registry
        .generate_chat_constraint(parser, &[], &auto, true)
        .unwrap();
    assert!(
        none_offered.is_none(),
        "no tools, no grammar: {none_offered:?}"
    );
    // The response side reads the same predicate, so it must agree.
    assert!(!registry.uses_full_assistant_constraint(parser, &[]));

    let tools = create_test_tools();
    match registry
        .generate_chat_constraint(parser, &tools, &auto, true)
        .unwrap()
    {
        Some(ToolConstraint::Ebnf(grammar)) => {
            assert!(grammar.contains("tool_calls ::= tool_call*"), "{grammar}");
        }
        other => panic!("tools in auto mode must yield the EBNF, got {other:?}"),
    }

    let none = ToolChoice::Value(ToolChoiceValue::None);
    match registry
        .generate_chat_constraint(parser, &tools, &none, true)
        .unwrap()
    {
        Some(ToolConstraint::Ebnf(grammar)) => {
            assert!(grammar.contains("tool_calls ::= \"\""), "{grammar}");
        }
        other => panic!("tool_choice none must forbid calls with a grammar, got {other:?}"),
    }
}

#[tokio::test]
async fn test_python_literals() {
    let parser = Glm4MoeParser::glm47();

    let input = r"<tool_call>test_func<arg_key>bool_true</arg_key><arg_value>True</arg_value><arg_key>bool_false</arg_key><arg_value>False</arg_value><arg_key>none_val</arg_key><arg_value>None</arg_value></tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test_func");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["bool_true"], true);
    assert_eq!(args["bool_false"], false);
    assert_eq!(args["none_val"], serde_json::Value::Null);
}

#[tokio::test]
async fn test_glm47_nested_json_in_arg_values() {
    let parser = Glm4MoeParser::glm47();

    let input = r#"<tool_call>process<arg_key>data</arg_key><arg_value>{"nested": {"key": "value"}}</arg_value><arg_key>list</arg_key><arg_value>[1, 2, 3]</arg_value></tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"].is_object());
    assert!(args["list"].is_array());
}
