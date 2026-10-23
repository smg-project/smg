//! Cohere Parser Integration Tests
//!
//! Tests for the Cohere parser which handles <|START_ACTION|>...<|END_ACTION|> format

mod common;

use serde_json::json;
use tool_parser::{CohereParser, ToolParser};

#[tokio::test]
async fn test_cohere_single_tool() {
    let parser = CohereParser::new();
    let input = r#"<|START_RESPONSE|>Let me search for that.<|END_RESPONSE|>
<|START_ACTION|>
{"tool_name": "search_web", "parameters": {"query": "latest news", "max_results": 5}}
<|END_ACTION|>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me search for that.");
    assert_eq!(tools[0].function.name, "search_web");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["query"], "latest news");
    assert_eq!(args["max_results"], 5);
}

#[tokio::test]
async fn test_cohere_multiple_tools_array() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>
[
    {"tool_name": "get_weather", "parameters": {"city": "Tokyo", "units": "celsius"}},
    {"tool_name": "search_news", "parameters": {"query": "AI developments", "limit": 10}}
]
<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);

    assert_eq!(tools[0].function.name, "get_weather");
    let args0: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args0["city"], "Tokyo");

    assert_eq!(tools[1].function.name, "search_news");
    let args1: serde_json::Value = serde_json::from_str(&tools[1].function.arguments).unwrap();
    assert_eq!(args1["query"], "AI developments");
}

#[tokio::test]
async fn test_cohere_nested_json() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>
{"tool_name": "process_data", "parameters": {"config": {"nested": {"value": [1, 2, 3]}}, "enabled": true}}
<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["config"]["nested"]["value"], json!([1, 2, 3]));
    assert_eq!(args["enabled"], true);
}

#[tokio::test]
async fn test_cohere_with_text_before_and_after() {
    let parser = CohereParser::new();
    let input = r#"<|START_RESPONSE|>I'll help with that.<|END_RESPONSE|>
<|START_ACTION|>{"tool_name": "test", "parameters": {}}<|END_ACTION|>
<|START_TEXT|>Here's some follow-up text.<|END_TEXT|>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test");
    assert!(normal_text.contains("I'll help with that."));
    assert!(normal_text.contains("Here's some follow-up text."));
}

#[tokio::test]
async fn test_cohere_empty_parameters() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>{"tool_name": "ping"}<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "ping");
    assert_eq!(tools[0].function.arguments, "{}");
}

#[tokio::test]
async fn test_cohere_with_special_chars_in_strings() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>{"tool_name": "echo", "parameters": {"text": "Array notation: arr[0] = <value>"}}<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Array notation: arr[0] = <value>");
}

#[tokio::test]
async fn test_cohere_format_detection() {
    let parser = CohereParser::new();

    assert!(parser.has_tool_markers("<|START_ACTION|>"));
    assert!(parser.has_tool_markers("<|END_ACTION|>"));
    assert!(parser.has_tool_markers("Some text <|START_ACTION|> more"));
    assert!(!parser.has_tool_markers("Just plain text"));
    assert!(!parser.has_tool_markers("[TOOL_CALLS]")); // Mistral format
    assert!(!parser.has_tool_markers("<|python_tag|>")); // Llama format
}

#[tokio::test]
async fn test_cohere_malformed_json() {
    let parser = CohereParser::new();

    // Missing closing bracket
    let input = r#"<|START_ACTION|>{"tool_name": "test", "parameters": {}<|END_ACTION|>"#;
    if let Ok((_, tools)) = parser.parse_complete(input).await {
        // Either returns empty tools or error is acceptable
        assert!(tools.is_empty() || tools.len() == 1);
    }

    // Invalid JSON inside
    let input = r#"<|START_ACTION|>{"tool_name": invalid}<|END_ACTION|>"#;
    if let Ok((_, tools)) = parser.parse_complete(input).await {
        assert_eq!(tools.len(), 0);
    }
}

#[tokio::test]
async fn test_cohere_no_tool_calls() {
    let parser = CohereParser::new();
    let input = "<|START_RESPONSE|>Hello, how can I help you today?<|END_RESPONSE|>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, "Hello, how can I help you today?");
}

#[tokio::test]
async fn test_cohere_real_world_output() {
    let parser = CohereParser::new();

    // Simulated real output from Cohere Command model
    let input = r#"<|START_RESPONSE|>I'll search for information about Rust programming and check the weather in San Francisco.<|END_RESPONSE|>
<|START_ACTION|>
[
    {
        "tool_name": "web_search",
        "parameters": {
            "query": "Rust programming language features 2024",
            "max_results": 3,
            "include_snippets": true
        }
    },
    {
        "tool_name": "get_weather",
        "parameters": {
            "location": "San Francisco, CA",
            "units": "fahrenheit",
            "include_forecast": false
        }
    }
]
<|END_ACTION|>
<|START_TEXT|>Let me execute these searches for you.<|END_TEXT|>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert!(normal_text.contains("I'll search for information about Rust programming"));
    assert!(normal_text.contains("Let me execute these searches"));
    assert_eq!(tools[0].function.name, "web_search");
    assert_eq!(tools[1].function.name, "get_weather");
}

#[tokio::test]
async fn test_cohere_alternative_field_names() {
    let parser = CohereParser::new();

    // Some Cohere outputs might use "name" and "arguments" instead
    let input = r#"<|START_ACTION|>{"name": "test", "arguments": {"key": "value"}}<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["key"], "value");
}

#[tokio::test]
async fn test_cohere_streaming_basic() {
    use openai_protocol::common::Tool;

    let mut parser = CohereParser::new();

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
        "<|START_RESPONSE|>",
        "Let me ",
        "check ",
        "that.",
        "<|END_RESPONSE|>",
        "<|START_ACTION|>",
        "{",
        "\"tool_name\"",
        ":",
        "\"get_weather\"",
        ",",
        "\"parameters\"",
        ":",
        "{",
        "\"city\"",
        ":",
        "\"Paris\"",
        "}",
        "}",
        "<|END_ACTION|>",
    ];

    let mut all_normal_text = String::new();
    let mut tool_names = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        all_normal_text.push_str(&result.normal_text);
        for call in result.calls {
            if let Some(name) = call.name {
                tool_names.push(name);
            }
        }
    }

    assert!(
        all_normal_text.contains("Let me check that."),
        "Should capture normal text, got: '{all_normal_text}'",
    );
    assert_eq!(tool_names.len(), 1, "Should have one tool call");
    assert_eq!(tool_names[0], "get_weather");
}

#[tokio::test]
async fn test_cohere_streaming_empty_parameters() {
    // A streamed call without parameters ends with `{}`, as parse_complete returns.
    let tools = common::create_test_tools();
    for chunks in [
        vec![
            "<|START_ACTION|>",
            r#"{"tool_name": "get_weather"}"#,
            "<|END_ACTION|>",
        ],
        vec![
            "<|START_ACTION|>",
            "{",
            "\"tool_name\"",
            ": ",
            "\"get_weather\"",
            "}",
            "<|END_ACTION|>",
        ],
    ] {
        let mut parser = CohereParser::new();
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
async fn test_cohere_reset() {
    let mut parser = CohereParser::new();

    // Parse something first
    let input = r#"<|START_ACTION|>{"tool_name": "test", "parameters": {}}<|END_ACTION|>"#;
    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    // Reset parser
    parser.reset();

    // Parse again - should work from clean state
    let (_, tools2) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools2.len(), 1);
}

#[tokio::test]
async fn test_cohere_multiple_action_blocks() {
    let parser = CohereParser::new();

    // Multiple separate action blocks (less common but possible)
    let input = r#"<|START_RESPONSE|>First task done.<|END_RESPONSE|>
<|START_ACTION|>{"tool_name": "task1", "parameters": {"id": 1}}<|END_ACTION|>
<|START_RESPONSE|>Second task.<|END_RESPONSE|>
<|START_ACTION|>{"tool_name": "task2", "parameters": {"id": 2}}<|END_ACTION|>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "task1");
    assert_eq!(tools[1].function.name, "task2");
    assert!(normal_text.contains("First task done."));
    assert!(normal_text.contains("Second task."));
}

#[tokio::test]
async fn test_cohere_escaped_quotes_in_parameters() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>{"tool_name": "echo", "parameters": {"text": "He said \"hello\""}}<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], r#"He said "hello""#);
}

#[tokio::test]
async fn test_cohere_unicode_in_parameters() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>{"tool_name": "translate", "parameters": {"text": "こんにちは世界", "emoji": "🚀"}}<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "こんにちは世界");
    assert_eq!(args["emoji"], "🚀");
}

#[tokio::test]
async fn test_cohere_whitespace_handling() {
    let parser = CohereParser::new();
    // Extra whitespace around JSON
    let input = r#"<|START_ACTION|>

    {"tool_name": "test", "parameters": {}}

<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "test");
}

#[tokio::test]
async fn test_cohere_tool_call_id_field() {
    let parser = CohereParser::new();
    // CMD3+ format includes tool_call_id
    let input = r#"<|START_ACTION|>{"tool_call_id": "call_123", "tool_name": "search", "parameters": {"q": "test"}}<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "search");
    // Note: tool_call_id is currently ignored - this test documents current behavior
}

#[tokio::test]
async fn test_cohere_end_action_marker_inside_string_param() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>{"tool_name": "echo", "parameters": {"text": "say <|END_ACTION|> please"}}<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "echo");
    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "say <|END_ACTION|> please");
}

#[tokio::test]
async fn test_cohere_unclosed_string_does_not_false_terminate() {
    let parser = CohereParser::new();
    // Unclosed quote: do not treat an in-string END_ACTION (or a later one via
    // find/rfind fallback) as a complete action — streaming may still be open.
    let input = r#"<|START_ACTION|>{"tool_name": "echo", "parameters": {"text": "say <|END_ACTION|> please}<|END_ACTION|>"#;
    let (normal, tools) = parser.parse_complete(input).await.unwrap();
    assert!(tools.is_empty());
    assert!(normal.contains("<|START_ACTION|>"));
}
