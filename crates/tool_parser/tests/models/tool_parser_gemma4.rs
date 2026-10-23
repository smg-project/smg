use openai_protocol::common::{Tool, ToolChoice, ToolChoiceValue};
use serde_json::{json, Value};
use tool_parser::{Gemma4Parser, ParserFactory, ToolConstraint, ToolParser};

const WEATHER: &str = r#"<|tool_call>call:get_weather{city:<|"|>San Francisco<|"|>}<tool_call|>"#;

#[expect(clippy::unwrap_used, reason = "literal test fixture must deserialize")]
fn tools() -> Vec<Tool> {
    serde_json::from_value(json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "count": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                    "metadata": {
                        "type": "object",
                        "properties": {"labels": {"type": "array", "items": {"type": "integer"}}}
                    }
                }
            }
        }
    }]))
    .unwrap()
}

#[tokio::test]
async fn complete_native_weather_call_keeps_surrounding_content() {
    let parser = Gemma4Parser::new();
    let input = format!("Checking. {WEATHER}Done.");
    let (text, calls) = parser
        .parse_complete_with_tools(&input, &tools())
        .await
        .unwrap();
    assert_eq!(text, "Checking. Done.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(calls[0].function.arguments, r#"{"city": "San Francisco"}"#);
}

#[tokio::test]
async fn accepts_native_bare_prefix_parentheses_and_eos_after_arguments() {
    for input in [
        r#"<|tool_call>:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#,
        r#"<|tool_call>call:get_weather(city:<|"|>Paris<|"|>)<tool_call|>"#,
        r#"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}"#,
        r#"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<turn|>"#,
        r#"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|><|tool_response>"#,
    ] {
        let (text, calls) = Gemma4Parser::new().parse_complete(input).await.unwrap();
        assert_eq!(text, "", "{input}");
        assert_eq!(calls.len(), 1, "{input}");
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"city": "Paris"}"#);
    }
}

#[tokio::test]
async fn nested_values_and_special_token_strings_match_native_arguments() {
    let input = r#"<|tool_call>call:get_weather{city:<|"|>"Paris" {x}, [y] \ <tool_call|> 杭州<|"|>,count:42,enabled:true,metadata:{labels:[1,2]},extra:[<|"|>a,b<|"|>,{key:<|"|>value<|"|>},false]}<tool_call|>"#;
    let (_, raw_calls) = Gemma4Parser::new().parse_complete(input).await.unwrap();
    let raw: Value = serde_json::from_str(&raw_calls[0].function.arguments).unwrap();
    assert_eq!(raw["count"], "42");
    assert_eq!(raw["enabled"], "true");
    assert_eq!(raw["metadata"]["labels"], json!(["1", "2"]));
    assert_eq!(raw["extra"], json!(["a,b", {"key": "value"}, "false"]));

    let (_, calls) = Gemma4Parser::new()
        .parse_complete_with_tools(input, &tools())
        .await
        .unwrap();
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "\"Paris\" {x}, [y] \\ <tool_call|> 杭州");
    assert_eq!(args["count"], 42);
    assert_eq!(args["enabled"], true);
    assert_eq!(args["metadata"]["labels"], json!([1, 2]));
}

#[tokio::test]
async fn primitive_schemas_coerce_bare_values_and_preserve_native_quoted_strings() {
    let input = r#"<|tool_call>call:typed{number:42,quoted_number:<|"|>42<|"|>,enabled:true,quoted_enabled:<|"|>true<|"|>,nothing:null,quoted_nothing:<|"|>null<|"|>,as_string:42,nested:{values:[42,<|"|>42<|"|>]}}<tool_call|>"#;
    let expected = json!({
        "number": 42, "quoted_number": "42", "enabled": true,
        "quoted_enabled": "true", "nothing": null, "quoted_nothing": "null",
        "as_string": "42", "nested": {"values": [42, "42"]}
    });
    for style in [1, 2, 0] {
        let primitive = |kind: &str| match style {
            0 => json!({"type": kind}),
            1 => json!({"type": [kind, "null"]}),
            _ => json!({"anyOf": [{"type": kind}, {"type": "null"}]}),
        };
        let tools: Vec<Tool> = serde_json::from_value(json!([{
            "type": "function", "function": {
                "name": "typed", "parameters": {"type": "object", "properties": {
                    "number": primitive("integer"), "quoted_number": primitive("integer"),
                    "enabled": primitive("boolean"), "quoted_enabled": primitive("boolean"),
                    "nothing": primitive("null"), "quoted_nothing": primitive("null"),
                    "as_string": {"type": "string", "anyOf": [{"type": "string"}, {"type": "integer"}]},
                    "nested": {"type": "object", "properties": {
                        "values": {"type": "array", "items": primitive("integer")}
                    }}
                }}
            }
        }]))
        .unwrap();
        let (_, calls) = Gemma4Parser::new()
            .parse_complete_with_tools(input, &tools)
            .await
            .unwrap();
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, expected, "schema style {style}");

        let mut parser = Gemma4Parser::new();
        let mut streamed_calls = Vec::new();
        for character in input.chars() {
            let result = parser
                .parse_incremental(&character.to_string(), &tools)
                .await
                .unwrap();
            assert!(result.normal_text.is_empty());
            streamed_calls.extend(result.calls);
        }
        assert_eq!(streamed_calls.len(), 1);
        let args: Value = serde_json::from_str(&streamed_calls[0].parameters).unwrap();
        assert_eq!(args, expected, "streaming schema style {style}");
    }
}

#[tokio::test]
async fn unknown_schemas_keep_raw_primitive_text_in_complete_and_streaming_calls() {
    let input = r#"<|tool_call>call:raw{number:42,quoted_number:<|"|>42<|"|>,enabled:true,nothing:null}<tool_call|>"#;
    let expected =
        json!({"number": "42", "quoted_number": "42", "enabled": "true", "nothing": "null"});
    let unknown_schema: Vec<Tool> = serde_json::from_value(json!([{
        "type": "function", "function": {
            "name": "raw", "parameters": {"type": "object", "properties": {
                "number": {}, "quoted_number": {}, "enabled": {}, "nothing": {}
            }}
        }
    }]))
    .unwrap();
    for tools in [Vec::new(), unknown_schema] {
        let (_, calls) = Gemma4Parser::new()
            .parse_complete_with_tools(input, &tools)
            .await
            .unwrap();
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args, expected);
        let mut parser = Gemma4Parser::new();
        let result = parser.parse_incremental(input, &tools).await.unwrap();
        let args: Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
        assert_eq!(args, expected);
    }
}

#[tokio::test]
async fn every_chunk_split_has_the_same_native_call_and_no_marker_leak() {
    let input = format!("Before {WEATHER}<|tool_response>After");
    let tools = tools();
    let (expected_text, expected_calls) = Gemma4Parser::new()
        .parse_complete_with_tools(&input, &tools)
        .await
        .unwrap();
    for split in 0..=input.len() {
        let mut parser = Gemma4Parser::new();
        let mut text = String::new();
        let mut calls = Vec::new();
        for chunk in [&input[..split], &input[split..]] {
            let parsed = parser.parse_incremental(chunk, &tools).await.unwrap();
            text.push_str(&parsed.normal_text);
            calls.extend(parsed.calls);
        }
        text.push_str(&parser.take_unstreamed_normal_text());
        assert_eq!(text, expected_text, "split {split}");
        assert_eq!(calls.len(), 1, "split {split}");
        assert_eq!(calls[0].tool_index, 0);
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(calls[0].parameters, expected_calls[0].function.arguments);
        assert!(parser.get_unstreamed_tool_args().is_none());
    }
}

#[tokio::test]
async fn one_character_stream_preserves_unicode_and_sequential_call_indices() {
    let input = r#"<|tool_call>call:get_weather{city:<|"|>杭州<|"|>}<tool_call|><|tool_call>:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#;
    let tools = tools();
    let mut parser = Gemma4Parser::new();
    let mut text = String::new();
    let mut calls = Vec::new();
    for character in input.chars() {
        let parsed = parser
            .parse_incremental(&character.to_string(), &tools)
            .await
            .unwrap();
        text.push_str(&parsed.normal_text);
        calls.extend(parsed.calls);
    }
    text.push_str(&parser.take_unstreamed_normal_text());
    assert_eq!(text, "");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].tool_index, 0);
    assert_eq!(calls[1].tool_index, 1);
    assert_eq!(calls[0].parameters, r#"{"city": "杭州"}"#);
    assert_eq!(calls[1].parameters, r#"{"city": "Paris"}"#);
}

#[tokio::test]
async fn malformed_and_incomplete_frames_remain_visible_and_reset_discards_state() {
    for input in [
        "plain text <|tool_ca",
        "<|tool_call>call:",
        "<|tool_call>call:get_weather{city:",
        r#"<|tool_call>call:get_weather{city:<|"|>Paris}<tool_call|>"#,
        "<|tool_call>call:get_weather{city:Paris]<tool_call|>",
        "<|tool_call>call:get_weather{city:}<tool_call|>",
        "<|tool_call>call:{city:Paris}<tool_call|>",
    ] {
        let (text, calls) = Gemma4Parser::new().parse_complete(input).await.unwrap();
        assert_eq!(text, input);
        assert!(calls.is_empty());
        let mut parser = Gemma4Parser::new();
        let mut streamed = String::new();
        for ch in input.chars() {
            let parsed = parser
                .parse_incremental(&ch.to_string(), &tools())
                .await
                .unwrap();
            streamed.push_str(&parsed.normal_text);
            assert!(parsed.calls.is_empty());
        }
        streamed.push_str(&parser.take_unstreamed_normal_text());
        assert_eq!(streamed, input);
    }
    let mut parser = Gemma4Parser::new();
    parser
        .parse_incremental("<|tool_call>call:", &tools())
        .await
        .unwrap();
    parser.reset();
    let parsed = parser.parse_incremental(WEATHER, &tools()).await.unwrap();
    assert_eq!(parsed.calls.len(), 1);
    assert_eq!(parsed.calls[0].tool_index, 0);
    assert_eq!(parsed.normal_text, "");
}

#[tokio::test]
async fn malformed_frame_does_not_hide_a_later_valid_call() {
    let invalid = "<|tool_call>call:get_weather{city:Paris]<tool_call|>";
    let (text, calls) = Gemma4Parser::new()
        .parse_complete(&format!("{invalid}{WEATHER}"))
        .await
        .unwrap();
    assert_eq!(text, invalid);
    assert_eq!(calls.len(), 1);
}

#[test]
fn gemma4_auto_required_and_named_use_native_output_without_forced_json() {
    let factory = ParserFactory::new();
    let registry = factory.registry();
    let tools = tools();
    assert!(registry.create_parser("gemma4").is_some());
    assert_eq!(
        registry
            .resolve_model_to_parser("google/gemma-4-12B-it")
            .as_deref(),
        Some("gemma4")
    );
    assert!(factory.list_parsers().contains(&"gemma4".to_owned()));
    assert!(registry.uses_native_tool_format(Some("gemma4")));
    assert!(!registry.has_structural_tag("gemma4"));

    let named: ToolChoice = serde_json::from_value(json!({
        "type": "function", "function": {"name": "get_weather"}
    }))
    .unwrap();
    let allowed: ToolChoice = serde_json::from_value(json!({
        "type": "allowed_tools", "mode": "required",
        "tools": [{"type": "function", "name": "get_weather"}]
    }))
    .unwrap();
    for choice in [
        ToolChoice::Value(ToolChoiceValue::Auto),
        ToolChoice::Value(ToolChoiceValue::None),
        ToolChoice::Value(ToolChoiceValue::Required),
        named,
        allowed,
    ] {
        for reasoning in [false, true] {
            assert!(registry
                .generate_tool_constraint(Some("gemma4"), &tools, &choice, reasoning)
                .unwrap()
                .is_none());
        }
    }

    let required = ToolChoice::Value(ToolChoiceValue::Required);
    for parser in [None, Some("json"), Some("qwen")] {
        assert!(!registry.uses_native_tool_format(parser));
        assert!(matches!(
            registry
                .generate_tool_constraint(parser, &tools, &required, false)
                .unwrap(),
            Some(ToolConstraint::JsonSchema(_))
        ));
    }
    assert!(registry.uses_native_tool_format(Some("mistral")));
    assert!(matches!(
        registry
            .generate_tool_constraint(Some("mistral"), &tools, &required, false)
            .unwrap(),
        Some(ToolConstraint::StructuralTag(_))
    ));
}
