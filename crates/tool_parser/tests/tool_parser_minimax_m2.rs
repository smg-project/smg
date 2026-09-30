//! MiniMax M2 Parser Integration Tests
mod common;

use common::create_test_tools;
use openai_protocol::common::{Function, Tool};
use serde_json::json;
use tool_parser::{MinimaxM2Parser, ToolParser};

/// Tools matching issue #1743: `update_task(taskId: string, subject: string)`.
fn update_task_tools() -> Vec<Tool> {
    vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "update_task".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "taskId":  {"type": "string"},
                    "subject": {"type": "string"}
                },
                "required": ["taskId", "subject"]
            }),
            strict: None,
        },
    }]
}

const UPDATE_TASK_CALL: &str = r#"<minimax:tool_call>
<invoke name="update_task">
<parameter name="taskId">3</parameter>
<parameter name="subject">X</parameter>
</invoke>
</minimax:tool_call>"#;

/// Issue #1743: a `string`-typed parameter whose value looks numeric must be
/// emitted as a JSON string when the schema is available, not coerced to a number.
#[tokio::test]
async fn test_minimax_string_arg_kept_as_string_with_schema() {
    let parser = MinimaxM2Parser::new();
    let tools = update_task_tools();

    let (_normal, calls) = parser
        .parse_complete_with_tools(UPDATE_TASK_CALL, &tools)
        .await
        .unwrap();

    assert_eq!(calls.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["taskId"],
        json!("3"),
        "string schema -> must stay string"
    );
    assert_eq!(args["subject"], json!("X"));
}

/// Numeric schema types are still coerced: an `integer` field stays a number.
#[tokio::test]
async fn test_minimax_integer_arg_coerced_with_schema() {
    let parser = MinimaxM2Parser::new();
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: Function {
            name: "update_task".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": { "taskId": {"type": "integer"} }
            }),
            strict: None,
        },
    }];

    let (_normal, calls) = parser
        .parse_complete_with_tools(UPDATE_TASK_CALL, &tools)
        .await
        .unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["taskId"], json!(3), "integer schema -> number");
}

/// Backward compatible: with no schema, text-based inference is unchanged
/// (a numeric-looking value becomes a number).
#[tokio::test]
async fn test_minimax_no_schema_infers_number() {
    let parser = MinimaxM2Parser::new();
    let (_normal, calls) = parser.parse_complete(UPDATE_TASK_CALL).await.unwrap();
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["taskId"],
        json!(3),
        "no schema -> inferred number (unchanged)"
    );
}

/// Streaming honors the schema too (tools were previously received but ignored).
#[tokio::test]
async fn test_minimax_streaming_string_arg_kept_as_string_with_schema() {
    let mut parser = MinimaxM2Parser::new();
    let tools = update_task_tools();

    let mut args_json = String::new();
    for chunk in [
        "<minimax:tool_call>",
        r#"<invoke name="update_task">"#,
        r#"<parameter name="taskId">3</parameter>"#,
        r#"<parameter name="subject">X</parameter>"#,
        "</invoke>",
        "</minimax:tool_call>",
    ] {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if call.name.is_none() {
                args_json.push_str(&call.parameters);
            }
        }
    }

    let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
    assert_eq!(
        args["taskId"],
        json!("3"),
        "streaming string schema -> stay string"
    );
}

#[tokio::test]
async fn test_minimax_complete_parsing() {
    let parser = MinimaxM2Parser::new();

    let input = r#"Let me search for that.
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Beijing</parameter>
<parameter name="date">2024-12-25</parameter>
</invoke>
</minimax:tool_call>
The weather will be..."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me search for that.\n");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Beijing");
    assert_eq!(args["date"], "2024-12-25");
}

#[tokio::test]
async fn test_minimax_multiple_tools() {
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="search">
<parameter name="query">rust tutorials</parameter>
</invoke>
</minimax:tool_call>
<minimax:tool_call>
<invoke name="translate">
<parameter name="text">Hello World</parameter>
<parameter name="target_lang">zh</parameter>
</invoke>
</minimax:tool_call>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_minimax_parallel_invokes_in_one_wrapper() {
    // MiniMax M2 emits parallel calls as multiple <invoke> blocks inside a SINGLE
    // <minimax:tool_call> wrapper (Anthropic-style). All must be extracted — the
    // earlier single-capture parse dropped every invoke after the first, zeroing
    // the parallel/parallel_multiple BFCL categories.
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="get_current_weather">
<parameter name="location">Beijing</parameter>
</invoke>
<invoke name="get_current_weather">
<parameter name="location">Shanghai</parameter>
</invoke>
</minimax:tool_call>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2, "both invokes in one wrapper must be parsed");
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "get_current_weather");
    assert_eq!(tools[1].function.name, "get_current_weather");
    assert!(tools[0].function.arguments.contains("Beijing"));
    assert!(tools[1].function.arguments.contains("Shanghai"));
}

#[tokio::test]
async fn test_minimax_type_conversion() {
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="process">
<parameter name="count">42</parameter>
<parameter name="rate">1.5</parameter>
<parameter name="enabled">true</parameter>
<parameter name="data">null</parameter>
<parameter name="text">string value</parameter>
</invoke>
</minimax:tool_call>"#;

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
async fn test_minimax_streaming_basic() {
    let mut parser = MinimaxM2Parser::new();

    let tools = create_test_tools();

    // Simulate streaming chunks
    let chunks = vec![
        "<minimax:tool_call>",
        r#"<invoke name="get_weather">"#,
        r#"<parameter name="city">Shanghai</parameter>"#,
        r#"<parameter name="units">celsius</parameter>"#,
        "</invoke>",
        "</minimax:tool_call>",
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
async fn test_minimax_streaming_parallel_invokes_in_one_wrapper() {
    // Two <invoke> blocks inside ONE <minimax:tool_call> wrapper must both stream.
    let mut parser = MinimaxM2Parser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<minimax:tool_call>",
        r#"<invoke name="get_weather"><parameter name="city">Beijing</parameter></invoke>"#,
        r#"<invoke name="get_weather"><parameter name="city">Shanghai</parameter></invoke>"#,
        "</minimax:tool_call>",
    ];

    let mut names = Vec::new();
    let mut max_index = 0usize;
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                names.push(name);
            }
            max_index = max_index.max(call.tool_index);
        }
    }

    assert_eq!(
        names.len(),
        2,
        "both invokes must stream a tool name, got {names:?}"
    );
    assert_eq!(max_index, 1, "second invoke must use tool_index 1");
}

#[test]
fn test_minimax_format_detection() {
    let parser = MinimaxM2Parser::new();

    // Should detect MiniMax format
    assert!(parser.has_tool_markers("<minimax:tool_call>"));
    assert!(parser.has_tool_markers("text with <minimax:tool_call> marker"));

    // Should not detect other formats
    assert!(!parser.has_tool_markers("<tool_call>")); // GLM4 format
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<｜tool▁calls▁begin｜>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_minimax_python_literals() {
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="test_func">
<parameter name="bool_true">True</parameter>
<parameter name="bool_false">False</parameter>
<parameter name="none_val">None</parameter>
</invoke>
</minimax:tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test_func");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["bool_true"], true);
    assert_eq!(args["bool_false"], false);
    assert_eq!(args["none_val"], serde_json::Value::Null);
}

#[tokio::test]
async fn test_minimax_nested_json_in_parameters() {
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="process">
<parameter name="data">{"nested": {"key": "value"}}</parameter>
<parameter name="list">[1, 2, 3]</parameter>
</invoke>
</minimax:tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // JSON-like strings are kept as strings, not parsed as JSON
    // This matches the behavior of other parsers like GLM4 MOE
    assert!(args["data"].is_string());
    assert_eq!(args["data"], r#"{"nested": {"key": "value"}}"#);
    assert!(args["list"].is_string());
    assert_eq!(args["list"], "[1, 2, 3]");
}

#[tokio::test]
async fn test_minimax_xml_entities() {
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="process">
<parameter name="html">&lt;div&gt;content&lt;/div&gt;</parameter>
<parameter name="text">Quote: &quot;hello&quot;</parameter>
<parameter name="code">if (a &amp;&amp; b) { }</parameter>
</invoke>
</minimax:tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["html"], "<div>content</div>");
    assert_eq!(args["text"], "Quote: \"hello\"");
    assert_eq!(args["code"], "if (a && b) { }");
}

#[tokio::test]
async fn test_minimax_streaming_partial_tags() {
    let mut parser = MinimaxM2Parser::new();
    let tools = create_test_tools();

    // Chunks split mid-tag
    let chunks = vec![
        "<minimax:tool_c",
        "all><invoke na",
        r#"me="get_weather"><param"#,
        r#"eter name="city">Bei"#,
        "jing</parameter></inv",
        "oke></minimax:tool_call>",
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
    assert_eq!(buffer, "");
}

#[tokio::test]
async fn test_minimax_streaming_incremental_json() {
    let mut parser = MinimaxM2Parser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<minimax:tool_call>",
        r#"<invoke name="get_weather">"#,
        r#"<parameter name="city">Paris</parameter>"#,
        r#"<parameter name="units">metric</parameter>"#,
        "</invoke></minimax:tool_call>",
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

    // Last fragment should be closing brace
    if let Some(last) = json_fragments.last() {
        assert!(
            last.contains('}'),
            "Last JSON fragment should contain '}}': {last}",
        );
    }
}

#[tokio::test]
async fn test_minimax_multiple_tools_boundary() {
    let mut parser = MinimaxM2Parser::new();
    let tools = create_test_tools();

    // Tool boundary at chunk boundary
    let chunks = vec![
        r#"<minimax:tool_call><invoke name="get_weather"><parameter name="city">Tokyo</parameter></invoke></minimax:tool_call>"#,
        r#"<minimax:tool_call><invoke name="search"><parameter name="query">weather forecast</parameter></invoke></minimax:tool_call>"#,
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
async fn test_minimax_unknown_function_name_is_forwarded() {
    let mut parser = MinimaxM2Parser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<minimax:tool_call>",
        r#"<invoke name="invalid_function">"#,
        r#"<parameter name="param">value</parameter>"#,
        "</invoke></minimax:tool_call>",
    ];

    let mut found_invalid = false;
    let mut all_normal_text = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_normal_text.push_str(&result.normal_text);

        // Unknown/hallucinated function names are forwarded as tool calls so the
        // client gets a clean error, rather than leaking the raw markup as text.
        for call in result.calls {
            if let Some(name) = call.name {
                if name == "invalid_function" {
                    found_invalid = true;
                }
            }
        }
    }

    assert!(
        found_invalid,
        "Unknown function name should be forwarded as a tool call"
    );
    assert!(
        !all_normal_text.contains("<invoke"),
        "Tool-call markup must not leak into normal text"
    );
}

#[tokio::test]
async fn test_minimax_empty_parameters() {
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="simple_func">
</invoke>
</minimax:tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "simple_func");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args, serde_json::json!({}));
}

#[tokio::test]
async fn test_minimax_streaming_empty_parameters() {
    // A streamed call without parameters ends with `{}`, as parse_complete returns.
    let tools = create_test_tools();
    for chunks in [
        vec!["<minimax:tool_call>\n<invoke name=\"get_weather\">\n</invoke>\n</minimax:tool_call>"],
        vec![
            "<minimax:tool_call>",
            "\n<invoke name=\"get_weather\">",
            "\n</invoke>",
            "\n</minimax:tool_call>",
        ],
    ] {
        let mut parser = MinimaxM2Parser::new();
        let mut name = None;
        let mut arguments = String::new();
        for chunk in chunks {
            let result = parser.parse_incremental(chunk, &tools).await.unwrap();
            for call in result.calls {
                assert_eq!(call.tool_index, 0);
                name = call.name.or(name);
                arguments.push_str(&call.parameters);
            }
        }
        assert_eq!(name.as_deref(), Some("get_weather"));
        assert_eq!(arguments, "{}");
    }
}

#[tokio::test]
async fn test_minimax_multiline_parameter_values() {
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="process">
<parameter name="multiline">line1
line2
line3</parameter>
<parameter name="unicode">你好世界 🌍</parameter>
</invoke>
</minimax:tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["multiline"], "line1\nline2\nline3");
    assert_eq!(args["unicode"], "你好世界 🌍");
}

#[tokio::test]
async fn test_minimax_nested_xml_like_content() {
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="process">
<parameter name="template"><html><body>Hello</body></html></parameter>
<parameter name="config">{"key": "<value>nested</value>"}</parameter>
</invoke>
</minimax:tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["template"], "<html><body>Hello</body></html>");

    // The nested JSON with XML-like content
    let config =
        serde_json::from_str::<serde_json::Value>(args["config"].as_str().unwrap()).unwrap();
    assert_eq!(config["key"], "<value>nested</value>");
}

#[tokio::test]
async fn test_minimax_streaming_state_reset() {
    let mut parser = MinimaxM2Parser::new();
    let tools = create_test_tools();

    // First tool
    let chunks1 = vec![
        r#"<minimax:tool_call><invoke name="get_weather">"#,
        r#"<parameter name="city">London</parameter>"#,
        "</invoke></minimax:tool_call>",
    ];

    for chunk in chunks1 {
        parser.parse_incremental(chunk, &tools).await.unwrap();
    }

    // Second tool - state should be reset
    let chunks2 = vec![
        r#"<minimax:tool_call><invoke name="search">"#,
        r#"<parameter name="query">rust</parameter>"#,
        "</invoke></minimax:tool_call>",
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
async fn test_minimax_many_parameters() {
    let parser = MinimaxM2Parser::new();

    let mut params_xml = String::new();
    for i in 1..=20 {
        params_xml.push_str(&format!(
            r#"<parameter name="param{i}">value{i}</parameter>
"#,
        ));
    }

    let input = format!(
        r#"<minimax:tool_call>
<invoke name="complex_func">
{params_xml}
</invoke>
</minimax:tool_call>"#,
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

#[tokio::test]
async fn test_minimax_character_by_character_streaming() {
    // Test character-by-character streaming to simulate real-world streaming
    let mut parser = MinimaxM2Parser::new();
    let tools = create_test_tools();

    let complete_text = r#"Let me help you. <minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Seattle</parameter>
<parameter name="units">celsius</parameter>
</invoke>
</minimax:tool_call> Here are the results."#;

    let mut content_collected = String::new();
    let mut tool_name_found = false;
    let mut parameters_found = Vec::new();

    // Stream character by character - feed only one character at a time
    for i in 0..complete_text.len() {
        let delta = &complete_text[i..i + 1];
        let result = parser.parse_incremental(delta, &tools).await.unwrap();
        content_collected.push_str(&result.normal_text);

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                tool_name_found = true;
            }
            if !call.parameters.is_empty() && !parameters_found.contains(&call.parameters) {
                parameters_found.push(call.parameters.clone());
            }
        }
    }

    assert!(
        tool_name_found,
        "Should find tool name during character-by-character streaming"
    );
    assert!(
        !parameters_found.is_empty(),
        "Should find parameters during streaming"
    );

    // Should have initial content and final content
    assert!(content_collected.contains("Let me help you."));
    assert!(content_collected.contains("Here are the results."));
}

#[tokio::test]
async fn test_minimax_content_before_and_after_tool_calls() {
    let parser = MinimaxM2Parser::new();

    let input = r#"I'll analyze the weather for you now.
<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Boston</parameter>
<parameter name="state">MA</parameter>
</invoke>
</minimax:tool_call>
Based on the analysis, here's what I found."#;

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
async fn test_minimax_incomplete_tool_call() {
    let parser = MinimaxM2Parser::new();

    // Incomplete tool call - missing closing tag
    let input = r#"<minimax:tool_call>
<invoke name="get_weather">
<parameter name="city">Chicago</parameter>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Should not extract incomplete tool calls
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should return as normal text
}

#[tokio::test]
async fn test_minimax_malformed_invoke_tag() {
    let parser = MinimaxM2Parser::new();

    // Malformed invoke tag - missing name attribute
    let input = r#"<minimax:tool_call>
<invoke>
<parameter name="city">Miami</parameter>
</invoke>
</minimax:tool_call>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Should not extract tool calls with malformed invoke tags
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input);
}

#[tokio::test]
async fn test_minimax_streaming_with_invalid_function_progressive() {
    let mut parser = MinimaxM2Parser::new();
    let tools = create_test_tools();

    // Progressive chunks building an invalid function call
    let chunks = vec![
        "<minimax:tool_call>",
        r#"<invoke name="invalid_function">"#,
        r#"<parameter name="test">value</parameter>"#,
        "</invoke>",
        "</minimax:tool_call>",
    ];

    let mut all_normal_text = String::new();
    let mut forwarded_name = None;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_normal_text.push_str(&result.normal_text);

        for call in result.calls {
            if let Some(name) = call.name {
                forwarded_name = Some(name);
            }
        }
    }

    // The unknown name is forwarded as a tool call, and its markup is NOT
    // leaked into the user-visible normal text.
    assert_eq!(
        forwarded_name.as_deref(),
        Some("invalid_function"),
        "Unknown function should be forwarded as a tool call"
    );
    assert!(
        !all_normal_text.contains("<invoke"),
        "Tool-call markup must not leak into normal text"
    );
}

#[tokio::test]
async fn test_minimax_rapid_streaming_bursts() {
    // Test handling of rapid streaming bursts (multiple chunks at once)
    let mut parser = MinimaxM2Parser::new();
    let tools = create_test_tools();

    let chunks = vec![
        "<minimax:tool_call><invoke name=\"search\"><parameter name=\"query\">",
        "rust programming",
        "</parameter></invoke></minimax:tool_call>",
    ];

    let mut found_function = false;
    let mut parameters = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "search");
                found_function = true;
            }
            if !call.parameters.is_empty() {
                parameters.push(call.parameters.clone());
            }
        }
    }

    assert!(found_function);

    // Verify that parameters were streamed correctly
    let final_params = parameters.join("");
    assert!(final_params.contains("rust programming"));
}

#[tokio::test]
async fn test_minimax_special_characters_in_values() {
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="process">
<parameter name="text">Special chars: @#$%^&*()</parameter>
<parameter name="emoji">🦀 Rust 🚀</parameter>
<parameter name="quotes">"double" and 'single' quotes</parameter>
</invoke>
</minimax:tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Special chars: @#$%^&*()");
    assert_eq!(args["emoji"], "🦀 Rust 🚀");
    assert_eq!(args["quotes"], "\"double\" and 'single' quotes");
}

#[tokio::test]
async fn test_minimax_whitespace_handling() {
    let parser = MinimaxM2Parser::new();

    // Test with various whitespace scenarios
    let input = r#"<minimax:tool_call>
    <invoke name="process">
        <parameter name="trimmed">  spaces around  </parameter>
        <parameter name="newlines">
            Line 1
            Line 2
        </parameter>
        <parameter name="tabs">	tab	separated	</parameter>
    </invoke>
</minimax:tool_call>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    // Values should preserve internal whitespace but may trim edges based on parser design
    assert!(args["newlines"].as_str().unwrap().contains("Line 1"));
    assert!(args["newlines"].as_str().unwrap().contains("Line 2"));
    assert_eq!(args["tabs"], "\ttab\tseparated\t");
}

#[tokio::test]
async fn test_minimax_no_tools() {
    // Test input with no tool calls at all
    let parser = MinimaxM2Parser::new();

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
async fn test_minimax_invalid_json_in_parameters() {
    // Test handling of invalid JSON in parameter values
    let parser = MinimaxM2Parser::new();

    let input = r#"<minimax:tool_call>
<invoke name="process">
<parameter name="valid">{"key": "value"}</parameter>
<parameter name="invalid">{invalid json: no quotes}</parameter>
<parameter name="broken">[1, 2, unclosed</parameter>
<parameter name="mixed">Some text {"partial": json} more text</parameter>
</invoke>
</minimax:tool_call>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();

    // Tool should still be extracted despite invalid JSON in parameters
    assert_eq!(
        tools.len(),
        1,
        "Should extract tool even with invalid JSON in parameters"
    );
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();

    // Parameters are stored as strings, not parsed as JSON
    // Even invalid JSON should be preserved as string values
    assert!(args["valid"].is_string());
    assert_eq!(args["valid"], r#"{"key": "value"}"#);

    assert!(args["invalid"].is_string());
    assert_eq!(args["invalid"], "{invalid json: no quotes}");

    assert!(args["broken"].is_string());
    assert_eq!(args["broken"], "[1, 2, unclosed");

    assert!(args["mixed"].is_string());
    assert_eq!(args["mixed"], r#"Some text {"partial": json} more text"#);

    assert_eq!(normal_text, "");
}
