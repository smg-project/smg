//! DeepSeek V3.1 Parser Integration Tests

use tool_parser::{DeepSeek31Parser, ParserFactory, ToolParser};

use crate::common::create_test_tools;

#[tokio::test]
async fn test_deepseek31_complete_single_tool() {
    let parser = DeepSeek31Parser::new();

    let input = concat!(
        "Let me check that for you.",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>",
        r#"{"location": "Tokyo", "units": "celsius"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "Let me check that for you.");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_deepseek31_complete_multiple_tools() {
    let parser = DeepSeek31Parser::new();

    let input = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>search<｜tool▁sep｜>",
        r#"{"query": "rust programming"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁call▁begin｜>translate<｜tool▁sep｜>",
        r#"{"text": "Hello World", "to": "ja"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_deepseek31_complete_nested_json() {
    let parser = DeepSeek31Parser::new();

    let input = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>process<｜tool▁sep｜>",
        r#"{"data": {"nested": {"deep": [1, 2, 3]}}}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert!(args["data"]["nested"]["deep"].is_array());
}

#[tokio::test]
async fn test_deepseek31_complete_malformed_json() {
    let parser = DeepSeek31Parser::new();

    let input = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>search<｜tool▁sep｜>",
        "{invalid json}",
        "<｜tool▁call▁end｜>",
        "<｜tool▁call▁begin｜>translate<｜tool▁sep｜>",
        r#"{"text": "hello", "to": "ja"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "translate");
}

#[test]
fn test_deepseek31_format_detection() {
    let parser = DeepSeek31Parser::new();

    assert!(parser.has_tool_markers("<｜tool▁calls▁begin｜>"));
    assert!(parser.has_tool_markers("text with <｜tool▁calls▁begin｜> marker"));

    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_deepseek31_no_tool_calls() {
    let parser = DeepSeek31Parser::new();

    let input = "Just a normal response with no tools.";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, input);
    assert!(tools.is_empty());
}

#[tokio::test]
async fn test_deepseek31_streaming_single_tool() {
    let tools = create_test_tools();
    let mut parser = DeepSeek31Parser::new();

    let chunks = vec![
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
        "get_weather",
        "<｜tool▁sep｜>",
        r#"{"location"#,
        r#": "Beijing"#,
        r#", "units": "#,
        r#""metric"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    ];

    let mut found_name = false;
    let mut collected_args = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            if !call.parameters.is_empty() {
                collected_args.push_str(&call.parameters);
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
    assert!(!collected_args.is_empty(), "Should have streamed arguments");
}

#[tokio::test]
async fn test_deepseek31_streaming_multiple_tools() {
    let tools = create_test_tools();
    let mut parser = DeepSeek31Parser::new();

    let chunks = vec![
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>search<｜tool▁sep｜>",
        r#"{"query": "rust"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>",
        r#"{"location": "Tokyo"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    ];

    let mut tool_names: Vec<String> = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                tool_names.push(name);
            }
        }
    }

    assert_eq!(tool_names, vec!["search", "get_weather"]);
}

#[tokio::test]
async fn test_deepseek31_streaming_text_before_tools() {
    let tools = create_test_tools();
    let mut parser = DeepSeek31Parser::new();

    let chunks = vec![
        "Here is ",
        "the result",
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>",
        r#"{"query": "test"}"#,
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ];

    let mut normal_text = String::new();
    let mut found_tool = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        normal_text.push_str(&result.normal_text);
        for call in result.calls {
            if call.name.is_some() {
                found_tool = true;
            }
        }
    }

    assert_eq!(normal_text, "Here is the result");
    assert!(found_tool);
}

#[tokio::test]
async fn test_deepseek31_streaming_end_tokens_stripped() {
    let tools = create_test_tools();
    let mut parser = DeepSeek31Parser::new();

    // Both <｜tool▁calls▁end｜> and <｜end▁of▁sentence｜> must not leak into normal_text
    for end_token in ["<｜tool▁calls▁end｜>", "<｜end▁of▁sentence｜>"] {
        parser.reset();
        let result = parser.parse_incremental(end_token, &tools).await.unwrap();
        assert!(
            result.normal_text.is_empty() || !result.normal_text.contains(end_token),
            "end token '{end_token}' should be stripped from normal_text"
        );
    }
}

#[tokio::test]
async fn test_deepseek31_streaming_end_marker_not_leaked_into_args() {
    let tools = create_test_tools();
    let mut parser = DeepSeek31Parser::new();

    // End markers arrive in the same chunk as the final JSON bytes.
    // The partial_tool_call_regex greedily captures everything after <｜tool▁sep｜>,
    // including trailing end tokens — these must not leak into streamed arguments.
    // Uses the realistic three-token sequence the model emits.
    let chunks = vec![
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>",
        r#"{"query": "rust"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#,
    ];

    let mut collected_args = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            collected_args.push_str(&call.parameters);
        }
    }

    assert!(
        !collected_args.is_empty(),
        "expected streamed argument chunks in marker-leak scenario"
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&collected_args).expect("streamed args should be valid JSON");
    assert_eq!(parsed["query"], "rust");

    for marker in [
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
        "<｜end▁of▁sentence｜>",
    ] {
        assert!(
            !collected_args.contains(marker),
            "end marker '{marker}' must not leak into streamed arguments: {collected_args}"
        );
    }
}

#[tokio::test]
async fn test_deepseek31_factory_registration() {
    let factory = ParserFactory::new();

    assert!(factory.has_parser("deepseek31"));

    // Verify V3.1 model names resolve to a parser that handles V3.1 format
    // (raw JSON after tool_sep, no code block wrapping)
    let v31_input = concat!(
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>search<｜tool▁sep｜>",
        r#"{"query": "rust"}"#,
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    );
    for model in [
        "deepseek-v3.1",
        "deepseek-v3.1-terminus",
        "deepseek-ai/DeepSeek-V3.1",
        "deepseek-ai/DeepSeek-V3.1-Terminus",
    ] {
        let parser = factory
            .registry()
            .create_for_model(model)
            .expect("parser should exist");
        let (_text, calls) = parser.parse_complete(v31_input).await.unwrap();
        assert_eq!(calls.len(), 1, "model {model} should parse V3.1 format");
        assert_eq!(calls[0].function.name, "search");
    }

    // Existing V3 mappings still work
    assert!(factory.registry().has_parser_for_model("deepseek-v3"));
    assert!(factory
        .registry()
        .has_parser_for_model("deepseek-ai/DeepSeek-V3-0324"));
}
