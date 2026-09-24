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

/// GLM-4.7 follows the same contract as every other native tool format:
/// `auto` and `none` send no constraint (an engine launched without a grammar
/// backend, TokenSpeed's default, keeps serving tool calls), and `required`
/// or a named function sends the structural tag with a forced call.
#[test]
fn test_glm47_constrains_only_forced_tool_choices_with_a_structural_tag() {
    let factory = ParserFactory::new();
    let registry = factory.registry();
    let parser = Some("glm47_moe");
    let tools = create_test_tools();
    assert!(registry.has_structural_tag("glm47_moe"));

    for choice in [
        ToolChoice::Value(ToolChoiceValue::Auto),
        ToolChoice::Value(ToolChoiceValue::None),
    ] {
        let constraint = registry
            .generate_tool_constraint(parser, &tools, &choice, false)
            .unwrap();
        assert!(
            constraint.is_none(),
            "{choice:?} must not constrain: {constraint:?}"
        );
    }
    assert!(registry
        .generate_tool_constraint(
            parser,
            &[],
            &ToolChoice::Value(ToolChoiceValue::Required),
            false
        )
        .unwrap()
        .is_none());

    let structural_tag = |constraint: Option<ToolConstraint>| -> serde_json::Value {
        let Some(ToolConstraint::StructuralTag(tag)) = constraint else {
            panic!("expected the structural tag, got {constraint:?}");
        };
        serde_json::from_str(&tag).unwrap()
    };

    let required = structural_tag(
        registry
            .generate_tool_constraint(
                parser,
                &tools,
                &ToolChoice::Value(ToolChoiceValue::Required),
                false,
            )
            .unwrap(),
    );
    assert_eq!(required["format"]["type"], "triggered_tags");
    assert_eq!(required["format"]["at_least_one"], true);
    assert_eq!(
        required["format"]["tags"].as_array().unwrap().len(),
        tools.len()
    );

    // A named function used to fall back to a JSON schema (pure-JSON output);
    // it is now the structural tag with a forced call. The gateway narrows
    // `tools` to the named one before asking the registry, so the tag carries
    // exactly that tool.
    let named: ToolChoice = serde_json::from_value(serde_json::json!({
        "type": "function",
        "function": {"name": tools[0].function.name}
    }))
    .unwrap();
    let named_tag = structural_tag(
        registry
            .generate_tool_constraint(parser, &tools[..1], &named, false)
            .unwrap(),
    );
    assert_eq!(named_tag["format"]["at_least_one"], true);
    let tags = named_tag["format"]["tags"].as_array().unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(
        tags[0]["begin"],
        format!("<tool_call>{}", tools[0].function.name)
    );

    // Allowed tools: forced only in required mode; auto mode is unconstrained.
    let allowed = |mode: &str| -> ToolChoice {
        serde_json::from_value(serde_json::json!({
            "type": "allowed_tools",
            "mode": mode,
            "tools": [{"type": "function", "name": tools[0].function.name}]
        }))
        .unwrap()
    };
    let allowed_required = structural_tag(
        registry
            .generate_tool_constraint(parser, &tools[..1], &allowed("required"), false)
            .unwrap(),
    );
    assert_eq!(allowed_required["format"]["at_least_one"], true);
    assert!(registry
        .generate_tool_constraint(parser, &tools[..1], &allowed("auto"), false)
        .unwrap()
        .is_none());
}

/// On a thinking prompt (GLM-4.7 with `enable_thinking` on, GLM-5 always) the
/// forced call must follow the model's reasoning, as xgrammar's built-in
/// `glm_4_7` tag lays it out with `reasoning=True`:
/// `sequence[<free text></think>, <calls>]`. Without a forced choice there is
/// still no constraint, thinking or not.
#[test]
fn test_glm47_forced_choice_on_a_thinking_prompt_reasons_first() {
    let factory = ParserFactory::new();
    let registry = factory.registry();
    let parser = Some("glm47_moe");
    let tools = create_test_tools();
    assert!(registry.has_reasoning_prefix(parser));
    assert!(!registry.has_reasoning_prefix(Some("mistral")));
    assert!(!registry.has_reasoning_prefix(None));

    let required = ToolChoice::Value(ToolChoiceValue::Required);
    let tag = |reasoning: bool| -> serde_json::Value {
        match registry
            .generate_tool_constraint(parser, &tools, &required, reasoning)
            .unwrap()
        {
            Some(ToolConstraint::StructuralTag(tag)) => serde_json::from_str(&tag).unwrap(),
            other => panic!("expected the structural tag, got {other:?}"),
        }
    };
    let plain = tag(false);
    let thinking = tag(true);
    assert_eq!(thinking["format"]["type"], "sequence");
    let elements = thinking["format"]["elements"].as_array().unwrap();
    assert_eq!(elements.len(), 2);
    assert_eq!(elements[0]["type"], "tag");
    assert_eq!(elements[0]["begin"], "");
    assert_eq!(elements[0]["end"], "</think>");
    assert_eq!(elements[0]["content"]["type"], "any_text");
    assert_eq!(
        elements[1], plain["format"],
        "the calls part is the non-thinking tag"
    );

    assert!(registry
        .generate_tool_constraint(
            parser,
            &tools,
            &ToolChoice::Value(ToolChoiceValue::Auto),
            true
        )
        .unwrap()
        .is_none());
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
