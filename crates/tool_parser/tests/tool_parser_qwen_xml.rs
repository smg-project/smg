//! Qwen XML Parser Integration Tests
//!
//! Tests for the Qwen XML parser which handles XML format:
//! <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>
mod common;

use common::{create_test_tools, streaming_helpers};
use serde_json::json;
use tool_parser::{parsers::QwenXmlParser, traits::ToolParser};

// Exercise the same terminal getters as the gRPC streaming caller. Fixtures are synthetic.
#[expect(clippy::unwrap_used, reason = "assertions in a shared test helper")]
async fn assert_streamed_arguments(
    input: &str,
    expected: &[(&str, serde_json::Value)],
    complete: bool,
) {
    let tools = create_test_tools();
    let mut feeds = vec![
        vec![input.to_string()],
        input.chars().map(|c| c.to_string()).collect(),
    ];
    // Also cover every possible two-chunk split, including coalesced calls.
    for (split, _) in input.char_indices().skip(1) {
        feeds.push(vec![input[..split].to_string(), input[split..].to_string()]);
    }
    for chunks in feeds {
        let mut parser = QwenXmlParser::new();
        let mut deltas = Vec::new();
        for chunk in &chunks {
            let result = parser.parse_incremental(chunk, &tools).await.unwrap();
            assert!(result.normal_text.is_empty());
            deltas.extend(result.calls);
        }
        assert!(parser.take_unstreamed_normal_text().is_empty());
        let pending = parser.get_unstreamed_tool_args();
        if complete {
            assert!(pending.is_none(), "completed calls must already be closed");
            assert!(parser
                .parse_incremental("", &tools)
                .await
                .unwrap()
                .calls
                .is_empty());
        }
        deltas.extend(pending.unwrap_or_default());
        let names: Vec<_> = deltas
            .iter()
            .filter_map(|d| d.name.as_deref().map(|name| (d.tool_index, name)))
            .collect();
        let expected_names: Vec<_> = expected
            .iter()
            .enumerate()
            .map(|(index, (name, _))| (index, *name))
            .collect();
        assert_eq!(names, expected_names, "chunks: {chunks:?}");
        assert!(deltas.iter().all(|d| d.tool_index < expected.len()));
        for (index, (_, expected_args)) in expected.iter().enumerate() {
            let args: String = deltas
                .iter()
                .filter(|d| d.tool_index == index)
                .map(|d| d.parameters.as_str())
                .collect();
            let parsed = serde_json::from_str::<serde_json::Value>(&args);
            assert_eq!(
                parsed.as_ref().ok(),
                Some(expected_args),
                "args: {args}; chunks: {chunks:?}"
            );
        }
        parser.reset();
        assert!(parser.get_unstreamed_tool_args().is_none());
    }
}

#[tokio::test]
async fn test_qwen_xml_streaming_braces_inside_string_do_not_close_object() {
    let input = "<tool_call><function=get_weather><parameter=city>echo '}'</parameter></function></tool_call>";
    assert_streamed_arguments(input, &[("get_weather", json!({"city": "echo '}'"}))], true).await;
}

#[tokio::test]
async fn test_qwen_xml_eos_closes_only_complete_parameter_values() {
    let input = "<tool_call><function=get_weather><parameter=city>Tokyo</parameter>";
    for suffix in ["", "<parameter=units>cels"] {
        assert_streamed_arguments(
            &format!("{input}{suffix}"),
            &[("get_weather", json!({"city": "Tokyo"}))],
            false,
        )
        .await;
    }
}

#[tokio::test]
async fn test_qwen_xml_eos_does_not_invent_unfinished_parameter() {
    assert_streamed_arguments(
        "<tool_call><function=get_weather><parameter=city>Tok",
        &[("get_weather", json!({}))],
        false,
    )
    .await;
}

#[tokio::test]
async fn test_qwen_xml_eos_closes_last_of_multiple_calls() {
    let input = concat!(
        "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>",
        "<tool_call><function=get_weather><parameter=city>Tokyo</parameter>",
    );
    assert_streamed_arguments(
        input,
        &[
            ("get_weather", json!({"city": "Paris"})),
            ("get_weather", json!({"city": "Tokyo"})),
        ],
        false,
    )
    .await;
}

#[tokio::test]
async fn test_qwen_xml_coalesced_calls_do_not_share_parameters() {
    let input = concat!(
        "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>",
        "<tool_call><function=get_weather><parameter=city>Tokyo</parameter><parameter=units>celsius</parameter></function></tool_call>",
    );
    assert_streamed_arguments(
        input,
        &[
            ("get_weather", json!({"city": "Paris"})),
            ("get_weather", json!({"city": "Tokyo", "units": "celsius"})),
        ],
        true,
    )
    .await;
}

#[tokio::test]
async fn test_qwen_xml_empty_calls_and_nested_values_are_closed_once() {
    let input = concat!(
        "<tool_call><function=get_time></function></tool_call>",
        r#"<tool_call><function=process><parameter=data>{"x":"}"}</parameter></function></tool_call>"#,
    );
    assert_streamed_arguments(
        input,
        &[
            ("get_time", json!({})),
            ("process", json!({"data": {"x": "}"}})),
        ],
        true,
    )
    .await;
}

#[tokio::test]
async fn test_qwen_xml_single_tool() {
    let parser = QwenXmlParser::new();
    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Beijing</parameter>
<parameter=units>celsius</parameter>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Beijing");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_qwen_xml_multiple_sequential_tools() {
    let parser = QwenXmlParser::new();
    let input = r"Let me help you with that.
<tool_call>
<function=search>
<parameter=query>Qwen model</parameter>
</function>
</tool_call>
<tool_call>
<function=translate>
<parameter=text>Hello</parameter>
<parameter=to>zh</parameter>
</function>
</tool_call>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "Let me help you with that.\n");
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_qwen_xml_nested_json_in_parameters() {
    let parser = QwenXmlParser::new();
    let input = r#"<tool_call>
<function=process_data>
<parameter=config>{"nested": {"value": [1, 2, 3]}}</parameter>
<parameter=enabled>true</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process_data");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // JSON values should be parsed
    assert_eq!(args["config"]["nested"]["value"], json!([1, 2, 3]));
    assert_eq!(args["enabled"], true);
}

#[tokio::test]
async fn test_qwen_xml_string_parameters() {
    let parser = QwenXmlParser::new();
    let input = r"<tool_call>
<function=process>
<parameter=text>Hello World</parameter>
<parameter=number>42</parameter>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Hello World");
    // JSON numbers should be parsed as numbers (consistent with Python's json.loads)
    assert_eq!(args["number"], 42);
}

#[tokio::test]
async fn test_qwen_xml_empty_arguments() {
    let parser = QwenXmlParser::new();
    let input = r"<tool_call>
<function=get_time>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_time");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args, json!({}));
}

#[tokio::test]
async fn test_qwen_xml_multiline_parameter_values() {
    let parser = QwenXmlParser::new();
    let input = r"<tool_call>
<function=write_file>
<parameter=content>Line 1
Line 2
Line 3</parameter>
<parameter=path>/tmp/test.txt</parameter>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["content"], "Line 1\nLine 2\nLine 3");
    assert_eq!(args["path"], "/tmp/test.txt");
}

#[tokio::test]
async fn test_qwen_xml_format_detection() {
    let parser = QwenXmlParser::new();

    assert!(parser.has_tool_markers("<tool_call>"));
    assert!(parser.has_tool_markers("Some text <tool_call>"));
    assert!(!parser.has_tool_markers("Just plain text"));
    assert!(!parser.has_tool_markers("<function=test>")); // Without tool_call tags
}

#[tokio::test]
async fn test_qwen_xml_incomplete_tags() {
    let parser = QwenXmlParser::new();

    // Missing closing tag
    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Beijing</parameter>";
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);

    // Missing opening tag
    let input = r"<parameter=city>Beijing</parameter>
</function>
</tool_call>";
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
}

#[tokio::test]
async fn test_qwen_xml_streaming_basic() {
    let mut parser = QwenXmlParser::new();
    let tools = create_test_tools();

    // Simulate streaming chunks
    let chunks = vec![
        "<tool_call>",
        r"<function=get_weather>",
        r"<parameter=city>Shanghai</parameter>",
        r"<parameter=units>celsius</parameter>",
        "</function>",
        "</tool_call>",
    ];

    let mut found_name = false;
    let mut found_params = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            if !call.parameters.is_empty() {
                found_params = true;
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
    assert!(found_params, "Should have streamed parameters");
}

#[tokio::test]
async fn test_qwen_xml_streaming_incremental_json() {
    let mut parser = QwenXmlParser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<tool_call>",
        r"<function=get_weather>",
        r"<parameter=city>Paris</parameter>",
        r"<parameter=units>metric</parameter>",
        "</function></tool_call>",
    ];

    let mut json_fragments = Vec::new();
    let mut found_function = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(_name) = call.name {
                found_function = true;
            }
            if !call.parameters.is_empty() {
                json_fragments.push(call.parameters.clone());
            }
        }
    }

    assert!(found_function);

    // Verify JSON was built incrementally
    assert!(!json_fragments.is_empty());

    // First fragment should start with opening brace
    if let Some(first) = json_fragments.first() {
        assert!(
            first.starts_with('{'),
            "First JSON fragment should start with '{{': {first}",
        );
    }
}

#[tokio::test]
async fn test_qwen_xml_streaming_partial_tags() {
    let mut parser = QwenXmlParser::new();
    let tools = create_test_tools();

    // Chunks split mid-tag
    let chunks = vec![
        "<tool_c",
        "all><function=",
        r"get_weather><param",
        r"eter=city>Bei",
        "jing</parameter></func",
        "tion></tool_call>",
    ];

    let mut found_name = false;
    let mut buffer = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        buffer.push_str(&result.normal_text);

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
        }
    }

    assert!(
        found_name,
        "Should have parsed function name from partial chunks"
    );
}

#[tokio::test]
async fn test_qwen_xml_multiple_tools_boundary() {
    let mut parser = QwenXmlParser::new();
    let tools = create_test_tools();

    // Tool boundary at chunk boundary
    let chunks = vec![
        r"<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>",
        r"<tool_call><function=search><parameter=query>weather forecast</parameter></function></tool_call>",
    ];

    let mut tool_names = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                tool_names.push(name);
            }
        }
    }

    assert_eq!(tool_names.len(), 2);
    assert_eq!(tool_names[0], "get_weather");
    assert_eq!(tool_names[1], "search");
}

#[tokio::test]
async fn test_qwen_xml_invalid_function_name() {
    let mut parser = QwenXmlParser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<tool_call>",
        r"<function=invalid_function>",
        r"<parameter=param>value</parameter>",
        "</function></tool_call>",
    ];

    let mut found_invalid = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        // Invalid function should be skipped
        for call in result.calls {
            if let Some(name) = call.name {
                if name == "invalid_function" {
                    found_invalid = true;
                }
            }
        }
    }

    assert!(!found_invalid, "Invalid function should not be parsed");
}

#[tokio::test]
async fn test_qwen_xml_type_conversion() {
    let parser = QwenXmlParser::new();

    let input = r"<tool_call>
<function=process>
<parameter=count>42</parameter>
<parameter=rate>1.5</parameter>
<parameter=enabled>true</parameter>
<parameter=data>null</parameter>
<parameter=text>string value</parameter>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // JSON values should be parsed
    assert_eq!(args["count"], 42);
    assert_eq!(args["rate"], 1.5);
    assert_eq!(args["enabled"], true);
    assert_eq!(args["data"], serde_json::Value::Null);
    assert_eq!(args["text"], "string value");
}

#[tokio::test]
async fn test_qwen_xml_special_characters_in_values() {
    let parser = QwenXmlParser::new();

    let input = r#"<tool_call>
<function=process>
<parameter=text>Special chars: @#$%^&*()</parameter>
<parameter=emoji>🦀 Rust 🚀</parameter>
<parameter=quotes>"double" and 'single' quotes</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Special chars: @#$%^&*()");
    assert_eq!(args["emoji"], "🦀 Rust 🚀");
    assert_eq!(args["quotes"], "\"double\" and 'single' quotes");
}

#[tokio::test]
async fn test_qwen_xml_whitespace_handling() {
    let parser = QwenXmlParser::new();

    // Test with various whitespace scenarios
    let input = r"<tool_call>
    <function=process>
        <parameter=trimmed>  spaces around  </parameter>
        <parameter=newlines>
            Line 1
            Line 2
        </parameter>
    </function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // Values should preserve internal whitespace but trim edges
    assert_eq!(args["trimmed"], "spaces around");
    assert!(args["newlines"].as_str().unwrap().contains("Line 1"));
    assert!(args["newlines"].as_str().unwrap().contains("Line 2"));
}

#[tokio::test]
async fn test_qwen_xml_no_tools() {
    // Test input with no tool calls at all
    let parser = QwenXmlParser::new();

    let input = r"This is just a normal response without any tool calls.
I can provide information directly without using any tools.
Even if I mention function names like get_weather or search,
they are not actual tool calls unless properly formatted.";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // No tools should be extracted
    assert_eq!(
        tools.len(),
        0,
        "Should not extract any tools from plain text"
    );

    // All content should be returned as normal text
    assert_eq!(
        normal_text, input,
        "All content should be returned as normal text when no tools present"
    );
}

#[tokio::test]
async fn test_qwen_xml_streaming_state_reset() {
    let mut parser = QwenXmlParser::new();
    let tools = create_test_tools();

    // First tool
    let chunks1 = vec![
        r"<tool_call><function=get_weather>",
        r"<parameter=city>London</parameter>",
        "</function></tool_call>",
    ];

    for chunk in chunks1 {
        parser.parse_incremental(chunk, &tools).await.unwrap();
    }

    // Second tool - state should be reset
    let chunks2 = vec![
        r"<tool_call><function=search>",
        r"<parameter=query>rust</parameter>",
        "</function></tool_call>",
    ];

    let mut second_tool_name = None;
    for chunk in chunks2 {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                second_tool_name = Some(name);
            }
        }
    }

    assert_eq!(second_tool_name, Some("search".to_string()));
}

#[tokio::test]
async fn test_qwen_xml_realistic_chunks() {
    let tools = create_test_tools();
    let mut parser = QwenXmlParser::new();

    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
<parameter=units>celsius</parameter>
</function>
</tool_call>";
    let chunks = streaming_helpers::create_realistic_chunks(input);

    assert!(chunks.len() > 20, "Should have many small chunks");

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed tool name");
}

#[tokio::test]
async fn test_qwen_xml_xml_tag_arrives_in_parts() {
    let tools = create_test_tools();
    let mut parser = QwenXmlParser::new();

    let chunks = vec![
        "<to", "ol_", "cal", "l>", "<fun", "cti", "on=", "get", "_we", "ath", "er>", "<par", "ame",
        "ter=", "cit", "y>", "Tok", "yo", "</", "par", "ame", "ter>", "</", "func", "tion>", "</",
        "too", "l_c", "all>",
    ];

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed tool name");
}

#[tokio::test]
async fn test_qwen_xml_content_before_and_after_tool_calls() {
    let parser = QwenXmlParser::new();

    let input = r"I'll analyze the weather for you now.
<tool_call>
<function=get_weather>
<parameter=city>Boston</parameter>
<parameter=state>MA</parameter>
</function>
</tool_call>
Based on the analysis, here's what I found.";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Verify tool extraction
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    // Verify content preservation (only text before tool call is returned)
    assert!(normal_text.contains("I'll analyze the weather for you now."));
    // Text after tool call is not included in parse_complete
    assert!(!normal_text.contains("Based on the analysis, here's what I found."));

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Boston");
    assert_eq!(args["state"], "MA");
}

#[tokio::test]
async fn test_qwen_xml_incomplete_tool_call() {
    let parser = QwenXmlParser::new();

    // Incomplete tool call - missing closing tag
    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Chicago</parameter>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Should not extract incomplete tool calls
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should return as normal text
}

#[tokio::test]
async fn test_qwen_xml_malformed_function_tag() {
    let parser = QwenXmlParser::new();

    // Malformed function tag - missing name attribute
    let input = r"<tool_call>
<function>
<parameter=city>Miami</parameter>
</function>
</tool_call>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Should not extract tool calls with malformed function tags
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input);
}

#[tokio::test]
async fn test_qwen_xml_many_parameters() {
    let parser = QwenXmlParser::new();

    let mut params_xml = String::new();
    for i in 1..=20 {
        params_xml.push_str(&format!(
            r"<parameter=param{i}>value{i}</parameter>
"
        ));
    }

    let input = format!(
        r"<tool_call>
<function=complex_func>
{params_xml}
</function>
</tool_call>"
    );

    let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "complex_func");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();

    // Verify all 20 parameters are parsed
    for i in 1..=20 {
        let key = format!("param{i}");
        let expected_value = format!("value{i}");
        assert_eq!(args[key], expected_value);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[tokio::test]
async fn test_qwen_xml_malformed_xml_missing_parameter_close() {
    let parser = QwenXmlParser::new();

    // Missing </parameter> closing tag - parser regex won't match incomplete parameter
    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Beijing
</function>
</tool_call>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // The parser extracts the tool call but with empty arguments since
    // the parameter block is malformed (no </parameter>)
    // This is acceptable behavior - we extract what we can
    if tools.is_empty() {
        // If no tools extracted, input returned as normal text
        assert_eq!(normal_text, input);
    } else {
        // If tool extracted, it should have the function name
        assert_eq!(tools[0].function.name, "get_weather");
    }
}

#[tokio::test]
async fn test_qwen_xml_malformed_xml_unclosed_function() {
    let parser = QwenXmlParser::new();

    // Missing </function> closing tag
    let input = r"<tool_call>
<function=get_weather>
<parameter=city>Beijing</parameter>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Parser should still extract the tool since it has complete tool_call tags
    // and the function name + parameters are present
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
}

#[tokio::test]
async fn test_qwen_xml_malformed_xml_nested_tool_calls() {
    let parser = QwenXmlParser::new();

    // Nested tool_call tags (invalid)
    let input = r"<tool_call>
<function=outer>
<tool_call>
<function=inner>
</function>
</tool_call>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Should handle gracefully - may parse first complete tool_call
    // The exact behavior depends on regex matching
    assert!(tools.len() <= 1);
}

#[tokio::test]
async fn test_qwen_xml_unicode_parameter_names() {
    let parser = QwenXmlParser::new();

    // Unicode characters in parameter names (Chinese, Japanese, emoji)
    let input = r"<tool_call>
<function=process>
<parameter=城市>北京</parameter>
<parameter=天気>晴れ</parameter>
<parameter=emoji_key>🌍🌎🌏</parameter>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["城市"], "北京");
    assert_eq!(args["天気"], "晴れ");
    assert_eq!(args["emoji_key"], "🌍🌎🌏");
}

#[tokio::test]
async fn test_qwen_xml_unicode_function_name() {
    let parser = QwenXmlParser::new();

    // Unicode function name
    let input = r"<tool_call>
<function=获取天气>
<parameter=location>上海</parameter>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "获取天气");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "上海");
}

#[tokio::test]
async fn test_qwen_xml_very_large_parameter_value() {
    let parser = QwenXmlParser::new();

    // Generate a large parameter value (100KB)
    let large_value: String = "x".repeat(100_000);

    let input = format!(
        r"<tool_call>
<function=process_large>
<parameter=data>{large_value}</parameter>
</function>
</tool_call>"
    );

    let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process_large");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["data"].as_str().unwrap().len(), 100_000);
}

#[tokio::test]
async fn test_qwen_xml_very_large_nested_json_parameter() {
    let parser = QwenXmlParser::new();

    // Generate moderately nested JSON structure (10 levels to avoid stack overflow)
    let mut nested_json = String::from(r#"{"level": 0}"#);
    for i in 1..=10 {
        nested_json = format!(r#"{{"level": {i}, "child": {nested_json}}}"#);
    }

    let input = format!(
        r"<tool_call>
<function=process_nested>
<parameter=config>{nested_json}</parameter>
</function>
</tool_call>"
    );

    let (_normal_text, tools) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process_nested");

    // Verify the nested JSON was parsed correctly
    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["config"].is_object());
    assert_eq!(args["config"]["level"], 10);
}

#[tokio::test]
async fn test_qwen_xml_streaming_malformed_recovery() {
    let mut parser = QwenXmlParser::new();
    let tools = create_test_tools();

    // First: malformed tool call (invalid function name)
    // Second: valid tool call
    let chunks = vec![
        r"<tool_call><function=invalid_func><parameter=x>1</parameter></function></tool_call>",
        r"<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>",
    ];

    let mut valid_tool_found = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                if name == "get_weather" {
                    valid_tool_found = true;
                }
            }
        }
    }

    assert!(
        valid_tool_found,
        "Should recover and parse valid tool after invalid one"
    );
}

#[tokio::test]
async fn test_qwen_xml_parameter_with_xml_like_content() {
    let parser = QwenXmlParser::new();

    // Parameter value contains XML-like content that shouldn't be parsed as tags
    let input = r#"<tool_call>
<function=process>
<parameter=html_content><div class="test"><span>Hello</span></div></parameter>
<parameter=xml_snippet><root><child attr="value"/></root></parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["html_content"]
        .as_str()
        .unwrap()
        .contains("<div class=\"test\">"));
    assert!(args["xml_snippet"]
        .as_str()
        .unwrap()
        .contains("<root><child"));
}

#[tokio::test]
async fn test_qwen_xml_empty_parameter_value() {
    let parser = QwenXmlParser::new();

    let input = r"<tool_call>
<function=process>
<parameter=empty></parameter>
<parameter=whitespace>   </parameter>
<parameter=normal>value</parameter>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["empty"], "");
    assert_eq!(args["whitespace"], ""); // Trimmed
    assert_eq!(args["normal"], "value");
}

// ============================================================================
// Literal Value (HTML-entity parity) and Python Literal Tests
//
// Argument values are treated literally — entity-like substrings are NOT
// decoded, matching Qwen's official API (verified on Qwen3-Coder and Qwen3.5).
// Regression guard for #1888.
// ============================================================================

#[tokio::test]
async fn test_qwen_xml_preserves_html_entities() {
    let parser = QwenXmlParser::new();

    // Named HTML entities in parameter values must be preserved verbatim.
    let input = r"<tool_call>
<function=process>
<parameter=ampersand>Tom &amp; Jerry</parameter>
<parameter=comparison>5 &lt; 10 &amp;&amp; 10 &gt; 5</parameter>
<parameter=quotes>&quot;Hello&quot; &amp; &apos;World&apos;</parameter>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["ampersand"], "Tom &amp; Jerry");
    assert_eq!(args["comparison"], "5 &lt; 10 &amp;&amp; 10 &gt; 5");
    assert_eq!(args["quotes"], "&quot;Hello&quot; &amp; &apos;World&apos;");
}

#[tokio::test]
async fn test_qwen_xml_preserves_numeric_entity_text() {
    let parser = QwenXmlParser::new();

    // Numeric/hex entity text must be preserved verbatim (not decoded).
    let input = r"<tool_call>
<function=process>
<parameter=decimal>&#60;tag&#62;</parameter>
<parameter=hex>&#x3C;tag&#x3E;</parameter>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["decimal"], "&#60;tag&#62;");
    assert_eq!(args["hex"], "&#x3C;tag&#x3E;");
}

#[tokio::test]
async fn test_qwen_xml_python_literals() {
    let parser = QwenXmlParser::new();

    // Test Python-style literals (True, False, None)
    let input = r"<tool_call>
<function=process>
<parameter=py_true>True</parameter>
<parameter=py_false>False</parameter>
<parameter=py_none>None</parameter>
<parameter=json_true>true</parameter>
<parameter=json_false>false</parameter>
<parameter=json_null>null</parameter>
</function>
</tool_call>";

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // Python literals should be converted
    assert_eq!(args["py_true"], true);
    assert_eq!(args["py_false"], false);
    assert_eq!(args["py_none"], serde_json::Value::Null);
    // JSON literals should also work
    assert_eq!(args["json_true"], true);
    assert_eq!(args["json_false"], false);
    assert_eq!(args["json_null"], serde_json::Value::Null);
}

#[tokio::test]
async fn test_qwen_xml_mixed_html_and_json() {
    let parser = QwenXmlParser::new();

    // Entity-like text alongside a JSON-valued parameter.
    let input = r#"<tool_call>
<function=search>
<parameter=query>price &lt; 100 &amp;&amp; rating &gt; 4</parameter>
<parameter=config>{"operator": "&amp;&amp;", "escape": true}</parameter>
</function>
</tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // Non-JSON value stays a literal string with entities intact.
    assert_eq!(args["query"], "price &lt; 100 &amp;&amp; rating &gt; 4");
    // JSON value parses as an object; entities inside its strings are preserved
    // (JSON has no notion of HTML entities, so they are ordinary characters).
    assert!(args["config"].is_object());
    assert_eq!(args["config"]["operator"], "&amp;&amp;");
    assert_eq!(args["config"]["escape"], true);
}
