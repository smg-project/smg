//! Mistral Parser Integration Tests
//!
//! Tests for the Mistral parser which handles [TOOL_CALLS] format
mod common;

use openai_protocol::common::ToolChoice;
use serde_json::json;
use tool_parser::{MistralParser, ParserFactory, ToolConstraint, ToolParser};

#[test]
fn test_mistral_structural_tag_call_cardinality() {
    let tools = common::create_test_tools();
    for at_least_one in [true, false] {
        let tag = MistralParser::build_structural_tag(&tools[..2], at_least_one);
        let format = &tag["format"];
        let calls = &format["tags"][0]["content"];
        assert_eq!(format["at_least_one"], at_least_one);
        assert_eq!(calls["type"], "tags_with_separator");
        assert_eq!(calls["at_least_one"], at_least_one);
        assert!(!calls["stop_after_first"].as_bool().unwrap_or(false));
    }
}

#[test]
fn test_mistral_tool_choice_constraints() {
    let factory = ParserFactory::new();
    let registry = factory.registry();
    let tools = common::create_test_tools();
    let tools = &tools[1..3];

    for (choice, selected) in [
        (json!("required"), tools),
        (
            json!({"type": "function", "function": {"name": "get_weather"}}),
            &tools[..1],
        ),
        (
            json!({"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "get_weather"}]}),
            &tools[..1],
        ),
    ] {
        // The gateway filters named/allowed tools before asking the registry.
        let choice: ToolChoice = serde_json::from_value(choice).unwrap();
        let Some(ToolConstraint::StructuralTag(tag)) = registry
            .generate_tool_constraint(Some("mistral"), selected, &choice, false)
            .unwrap()
        else {
            panic!("expected a structural tag for {choice:?}");
        };
        let tag: serde_json::Value = serde_json::from_str(&tag).unwrap();
        let calls = &tag["format"]["tags"][0]["content"];
        assert_eq!(calls["at_least_one"], true, "{choice:?}");
        assert_eq!(calls["tags"].as_array().unwrap().len(), selected.len());
    }

    for choice in [
        json!("auto"),
        json!("none"),
        json!({"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "get_weather"}]}),
    ] {
        let choice: ToolChoice = serde_json::from_value(choice).unwrap();
        assert!(registry
            .generate_tool_constraint(Some("mistral"), tools, &choice, false)
            .unwrap()
            .is_none());
    }
}

#[tokio::test]
async fn test_mistral_single_tool() {
    let parser = MistralParser::new();
    let input = r#"Let me search for that.
[TOOL_CALLS] [{"name": "search_web", "arguments": {"query": "latest news", "max_results": 5}}]"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me search for that.\n");
    assert_eq!(tools[0].function.name, "search_web");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["query"], "latest news");
    assert_eq!(args["max_results"], 5);
}

#[tokio::test]
async fn test_mistral_multiple_tools() {
    let parser = MistralParser::new();
    let input = r#"I'll help you with both tasks.
[TOOL_CALLS] [
    {"name": "get_weather", "arguments": {"city": "Tokyo", "units": "celsius"}},
    {"name": "search_news", "arguments": {"query": "AI developments", "limit": 10}}
]"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "I'll help you with both tasks.\n");

    assert_eq!(tools[0].function.name, "get_weather");
    let args0: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args0["city"], "Tokyo");

    assert_eq!(tools[1].function.name, "search_news");
    let args1: serde_json::Value = serde_json::from_str(&tools[1].function.arguments).unwrap();
    assert_eq!(args1["query"], "AI developments");
}

#[tokio::test]
async fn test_mistral_nested_json() {
    let parser = MistralParser::new();
    let input = r#"Processing complex data.
[TOOL_CALLS] [{"name": "process_data", "arguments": {"config": {"nested": {"value": [1, 2, 3]}}, "enabled": true}}]"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Processing complex data.\n");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["config"]["nested"]["value"], json!([1, 2, 3]));
    assert_eq!(args["enabled"], true);
}

#[tokio::test]
async fn test_mistral_with_text_after() {
    let parser = MistralParser::new();
    let input = r#"[TOOL_CALLS] [{"name": "test", "arguments": {}}]

And here's some text after the tool call that should be ignored."#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test");
}

#[tokio::test]
async fn test_mistral_empty_arguments() {
    let parser = MistralParser::new();
    let input = r#"[TOOL_CALLS] [{"name": "ping", "arguments": {}}]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "ping");
}

#[tokio::test]
async fn test_mistral_with_brackets_in_strings() {
    let parser = MistralParser::new();
    let input = r#"[TOOL_CALLS] [{"name": "echo", "arguments": {"text": "Array notation: arr[0] = value[1]"}}]"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Array notation: arr[0] = value[1]");
}

#[tokio::test]
async fn test_mistral_format_detection() {
    let parser = MistralParser::new();

    assert!(parser.has_tool_markers("[TOOL_CALLS] ["));
    assert!(parser.has_tool_markers("Some text [TOOL_CALLS] ["));
    assert!(!parser.has_tool_markers("Just plain text"));
    assert!(!parser.has_tool_markers("[{\"name\": \"test\"}]")); // JSON array without TOOL_CALLS
}

#[tokio::test]
async fn test_mistral_malformed_json() {
    let parser = MistralParser::new();

    // Missing closing bracket
    let input = r#"[TOOL_CALLS] [{"name": "test", "arguments": {}"#;
    if let Ok((_normal_text, tools)) = parser.parse_complete(input).await {
        assert_eq!(tools.len(), 0);
    }
    // Error is also acceptable for malformed input

    // Invalid JSON inside
    let input = r#"[TOOL_CALLS] [{"name": invalid}]"#;
    if let Ok((_normal_text, tools)) = parser.parse_complete(input).await {
        assert_eq!(tools.len(), 0);
    }
    // Error is also acceptable for malformed input
}

#[tokio::test]
async fn test_mistral_real_world_output() {
    let parser = MistralParser::new();

    // Actual output from Mistral model
    let input = r#"I'll search for information about Rust programming and check the weather in San Francisco.

[TOOL_CALLS] [
    {
        "name": "web_search",
        "arguments": {
            "query": "Rust programming language features 2024",
            "max_results": 3,
            "include_snippets": true
        }
    },
    {
        "name": "get_weather",
        "arguments": {
            "location": "San Francisco, CA",
            "units": "fahrenheit",
            "include_forecast": false
        }
    }
]

Let me execute these searches for you."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "I'll search for information about Rust programming and check the weather in San Francisco.\n\n");
    assert_eq!(tools[0].function.name, "web_search");
    assert_eq!(tools[1].function.name, "get_weather");
}

#[tokio::test]
async fn test_mistral_streaming_closing_bracket() {
    use openai_protocol::common::Tool;

    // Test that closing ] is stripped for Mistral array format
    let mut parser = MistralParser::new();

    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: openai_protocol::common::Function {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: json!({}),
            strict: None,
            extra: Default::default(),
        },
    }];

    let chunks = vec![
        "[TOOL_CALLS] ",
        "[{",
        "\"",
        "name",
        "\":",
        "\"",
        "get",
        "_weather",
        "\",",
        "\"",
        "arguments",
        "\":",
        "{",
        "\"",
        "city",
        "\":",
        "\"",
        "Paris",
        "\"",
        "}",
        "}",
        "]",
        " Here's",
        " the weather",
        " info",
    ];

    let mut all_normal_text = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_normal_text.push_str(&result.normal_text);
    }

    // Should emit only the third chunk as normal text, NOT the ]
    assert_eq!(
        all_normal_text, " Here's the weather info",
        "Should not emit ] for Mistral array format, got: '{all_normal_text}'",
    );
}

#[tokio::test]
async fn test_mistral_streaming_bracket_in_text_after_tools() {
    use openai_protocol::common::Tool;

    // Test that ] in normal text AFTER tool calls is preserved
    let mut parser = MistralParser::new();

    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: openai_protocol::common::Function {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: json!({}),
            strict: None,
            extra: Default::default(),
        },
    }];

    let chunks = vec![
        "[TOOL_CALLS] ",
        "[",
        "{",
        "\"name",
        "\":",
        "\"get_weather",
        "\",",
        "\"arguments",
        "\":",
        "{\"",
        "city",
        "\":",
        "\"Paris",
        "\"}",
        "}",
        "]",
        " Array",
        " notation:",
        " arr",
        "[",
        "0",
        "]",
    ];

    let mut all_normal_text = String::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_normal_text.push_str(&result.normal_text);
    }

    // Should preserve ] in normal text after tools complete
    assert_eq!(
        all_normal_text, " Array notation: arr[0]",
        "Should preserve ] in normal text after tools, got: '{all_normal_text}'",
    );
}
