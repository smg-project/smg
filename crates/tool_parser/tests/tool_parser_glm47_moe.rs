//! GLM-4.7 MoE Parser Integration Tests
mod common;

use common::create_test_tools;
use openai_protocol::common::{Function, Tool};
use serde_json::{json, Value};
use tool_parser::{Glm4MoeParser, ParserFactory, ToolParser};

fn schema_tool(properties: Value) -> Vec<Tool> {
    vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "lookup".to_string(),
            description: None,
            parameters: json!({"type": "object", "properties": properties}),
            strict: None,
        },
    }]
}

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

    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
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

    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["count"], 42);
    assert_eq!(args["rate"], 1.5);
    assert_eq!(args["enabled"], true);
    assert_eq!(args["data"], Value::Null);
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

#[tokio::test]
async fn test_glm47_streaming_holds_partial_tool_marker_at_every_split() {
    let tools = create_test_tools();
    let input = "回答：<tool_call>get_weather<arg_key>city</arg_key><arg_value>杭州</arg_value></tool_call>";
    let expected_prefix = "回答：";

    for split in input.char_indices().map(|(index, _)| index).skip(1) {
        let mut parser = Glm4MoeParser::glm47();
        let mut normal_text = String::new();
        let mut calls = Vec::new();

        for chunk in [&input[..split], &input[split..]] {
            let result = parser.parse_incremental(chunk, &tools).await.unwrap();
            normal_text.push_str(&result.normal_text);
            calls.extend(result.calls);
        }

        assert_eq!(normal_text, expected_prefix, "split at byte {split}");
        assert_eq!(calls.len(), 1, "split at byte {split}");
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
        let args: Value = serde_json::from_str(&calls[0].parameters).unwrap();
        assert_eq!(args["city"], "杭州", "split at byte {split}");
    }
}

#[tokio::test]
async fn test_glm47_streaming_parses_two_calls_from_one_chunk() {
    let mut parser = Glm4MoeParser::glm47();
    let tools = create_test_tools();
    let input = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>杭州</arg_value></tool_call>\
                 <tool_call>translate<arg_key>text</arg_key><arg_value>天气</arg_value><arg_key>target_lang</arg_key><arg_value>en</arg_value></tool_call>";

    let result = parser.parse_incremental(input, &tools).await.unwrap();

    assert_eq!(result.calls.len(), 2);
    assert_eq!(result.calls[0].tool_index, 0);
    assert_eq!(result.calls[0].name.as_deref(), Some("get_weather"));
    assert_eq!(result.calls[1].tool_index, 1);
    assert_eq!(result.calls[1].name.as_deref(), Some("translate"));
}

#[tokio::test]
async fn test_glm47_finalize_accepts_complete_cjk_call_and_rejects_partial_eof() {
    let tools = schema_tool(json!({"query": {"type": "string"}}));
    let mut complete = Glm4MoeParser::glm47();
    let parsed = complete
        .parse_incremental(
            "<tool_call>lookup<arg_key>query</arg_key><arg_value>查询人事</arg_value></tool_call>",
            &tools,
        )
        .await
        .unwrap();
    assert_eq!(parsed.calls.len(), 1);
    let final_result = complete.finalize(&tools).await.unwrap();
    assert!(final_result.normal_text.is_empty());
    assert!(final_result.calls.is_empty());

    let mut unclosed = Glm4MoeParser::glm47();
    unclosed
        .parse_incremental(
            "<tool_call>lookup<arg_key>query</arg_key><arg_value>查询人事",
            &tools,
        )
        .await
        .unwrap();
    assert!(unclosed.finalize(&tools).await.is_err());

    let mut partial_marker = Glm4MoeParser::glm47();
    partial_marker
        .parse_incremental("正文<tool_", &tools)
        .await
        .unwrap();
    assert!(partial_marker.finalize(&tools).await.is_err());
}

#[tokio::test]
async fn test_glm47_schema_keeps_string_values_and_whitespace() {
    let tools = schema_tool(json!({
        "plain": {"type": "string"},
        "nullable": {"type": ["string", "null"]},
        "any": {"anyOf": [{"type": "null"}, {"type": "string"}]},
        "one": {"oneOf": [{"type": "integer"}, {"type": "string"}]}
    }));
    let input = "<tool_call>lookup\
                 <arg_key>plain</arg_key><arg_value>  00123  </arg_value>\
                 <arg_key>nullable</arg_key><arg_value>00123</arg_value>\
                 <arg_key>any</arg_key><arg_value>00123</arg_value>\
                 <arg_key>one</arg_key><arg_value>00123</arg_value>\
                 </tool_call>";

    let (_, calls) = Glm4MoeParser::glm47()
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();

    assert_eq!(args["plain"], "  00123  ");
    assert_eq!(args["nullable"], "00123");
    assert_eq!(args["any"], "00123");
    assert_eq!(args["one"], "00123");
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

#[tokio::test]
async fn test_glm_parser_aliases_and_model_name_variants() {
    let factory = ParserFactory::new();
    let input =
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>杭州</arg_value></tool_call>";

    for alias in ["glm45", "glm47"] {
        assert!(factory.registry().create_parser(alias).is_some(), "{alias}");
    }

    for model in [
        "GLM-5.2",
        "THUDM/GLM-5.2",
        "zai-org/glm-5.2-fp8",
        "/models/THUDM/GLM-5.2",
    ] {
        let parser = factory
            .registry()
            .create_for_model(model)
            .unwrap_or_else(|| panic!("no parser for {model}"));
        let (_, calls) = parser.parse_complete(input).await.unwrap();
        assert_eq!(calls.len(), 1, "{model}");
        assert_eq!(calls[0].function.name, "get_weather", "{model}");
    }
}

#[tokio::test]
async fn test_python_literals() {
    let parser = Glm4MoeParser::glm47();

    let input = r"<tool_call>test_func<arg_key>bool_true</arg_key><arg_value>True</arg_value><arg_key>bool_false</arg_key><arg_value>False</arg_value><arg_key>none_val</arg_key><arg_value>None</arg_value></tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test_func");

    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["bool_true"], true);
    assert_eq!(args["bool_false"], false);
    assert_eq!(args["none_val"], Value::Null);
}

#[tokio::test]
async fn test_glm47_nested_json_in_arg_values() {
    let parser = Glm4MoeParser::glm47();

    let input = r#"<tool_call>process<arg_key>data</arg_key><arg_value>{"nested": {"key": "value"}}</arg_value><arg_key>list</arg_key><arg_value>[1, 2, 3]</arg_value></tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"].is_object());
    assert!(args["list"].is_array());
}
