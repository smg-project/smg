use super::*;

#[tokio::test]
async fn test_single_tool_call() {
    let parser = CohereParser::new();
    let input = r#"<|START_RESPONSE|>Let me search for that.<|END_RESPONSE|>
<|START_ACTION|>
{"tool_name": "search", "parameters": {"query": "rust programming"}}
<|END_ACTION|>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me search for that.");
    assert_eq!(tools[0].function.name, "search");

    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["query"], "rust programming");
}

#[tokio::test]
async fn test_multiple_tool_calls_array() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>
[
  {"tool_name": "search", "parameters": {"query": "rust"}},
  {"tool_name": "get_weather", "parameters": {"city": "Paris"}}
]
<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "get_weather");
}

#[tokio::test]
async fn test_no_tool_calls() {
    let parser = CohereParser::new();
    let input = "<|START_RESPONSE|>Hello, how can I help?<|END_RESPONSE|>";

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, "Hello, how can I help?");
}

#[tokio::test]
async fn test_has_tool_markers() {
    let parser = CohereParser::new();

    assert!(parser.has_tool_markers("<|START_ACTION|>"));
    assert!(parser.has_tool_markers("<|END_ACTION|>"));
    assert!(parser.has_tool_markers("Some text <|START_ACTION|> more"));
    assert!(!parser.has_tool_markers("Just plain text"));
    assert!(!parser.has_tool_markers("[TOOL_CALLS]")); // Mistral format
}

#[tokio::test]
async fn test_empty_parameters() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>{"tool_name": "ping"}<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "ping");
    assert_eq!(tools[0].function.arguments, "{}");
}

#[tokio::test]
async fn test_nested_json() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>
{"tool_name": "process", "parameters": {"config": {"nested": {"value": [1, 2, 3]}}}}
<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);

    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(
        args["config"]["nested"]["value"],
        serde_json::json!([1, 2, 3])
    );
}

#[tokio::test]
async fn test_text_markers_cleaned() {
    let parser = CohereParser::new();
    let input = r#"<|START_TEXT|>Some intro<|END_TEXT|>
<|START_ACTION|>{"tool_name": "test", "parameters": {}}<|END_ACTION|>
<|START_TEXT|>Conclusion<|END_TEXT|>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert!(normal_text.contains("Some intro"));
    assert!(normal_text.contains("Conclusion"));
    assert!(!normal_text.contains("<|START_TEXT|>"));
    assert!(!normal_text.contains("<|END_TEXT|>"));
}

#[tokio::test]
async fn test_malformed_json() {
    let parser = CohereParser::new();
    let input = r#"<|START_ACTION|>{"tool_name": invalid}<|END_ACTION|>"#;

    let (_, tools) = parser.parse_complete(input).await.unwrap();
    // Should gracefully handle malformed JSON
    assert_eq!(tools.len(), 0);
}
