use super::*;

#[tokio::test]
async fn test_with_marker() {
    let parser = SarashinaParser::new();
    let input = "<|tool_calls|>[{'name': 'get_weather', 'arguments': {'city': 'Tokyo'}}]";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Tokyo");
}

#[tokio::test]
async fn test_markerless_bare_list() {
    // This is what the server actually sees (special token stripped).
    let parser = SarashinaParser::new();
    let input = "[{'name': 'get_weather', 'arguments': {'city': 'Tokyo'}}]";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Tokyo");
}

#[tokio::test]
async fn test_leading_text_with_marker() {
    let parser = SarashinaParser::new();
    let input = "Let me check.<|tool_calls|>[{'name': 'ping', 'arguments': {}}]";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(normal_text, "Let me check.");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.arguments, "{}");
}

#[tokio::test]
async fn test_multiple_tool_calls() {
    let parser = SarashinaParser::new();
    let input = "[{'name': 'search', 'arguments': {'q': 'rust'}}, \
             {'name': 'get_weather', 'arguments': {'city': 'Paris'}}]";
    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "get_weather");
}

#[tokio::test]
async fn test_nested_arguments() {
    let parser = SarashinaParser::new();
    let input = "[{'name': 'process', 'arguments': {'config': {'nested': {'values': [1, 2, 3]}}}}]";
    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(
        args["config"]["nested"]["values"],
        serde_json::json!([1, 2, 3])
    );
}

#[tokio::test]
async fn test_python_literals_coerced() {
    let parser = SarashinaParser::new();
    let input = "[{'name': 'flags', 'arguments': {'a': True, 'b': False, 'c': None}}]";
    let (_, tools) = parser.parse_complete(input).await.unwrap();
    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["a"], serde_json::json!(true));
    assert_eq!(args["b"], serde_json::json!(false));
    assert_eq!(args["c"], Value::Null);
}

#[tokio::test]
async fn test_apostrophe_in_value() {
    let parser = SarashinaParser::new();
    let input = "[{'name': 'say', 'arguments': {'text': 'it\\'s [done]'}}]";
    let (_, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    let args: Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "it's [done]");
}

#[tokio::test]
async fn test_plain_text_untouched() {
    let parser = SarashinaParser::new();
    let input = "The weather in Tokyo is sunny.";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, "The weather in Tokyo is sunny.");
}

#[tokio::test]
async fn test_non_toolcall_list_untouched() {
    // A bare list that is not a tool-call structure must be left as text.
    let parser = SarashinaParser::new();
    let input = "[1, 2, 3]";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, "[1, 2, 3]");
}

#[tokio::test]
async fn test_has_tool_markers() {
    let parser = SarashinaParser::new();
    assert!(parser.has_tool_markers("<|tool_calls|>[]"));
    assert!(parser.has_tool_markers("[{'name': 'x', 'arguments': {}}]"));
    assert!(!parser.has_tool_markers("just text"));
    assert!(!parser.has_tool_markers("[TOOL_CALLS]")); // Mistral format
}

#[tokio::test]
async fn test_streaming_markerless_chunked() {
    let mut parser = SarashinaParser::new();
    let chunks = [
        "[{'name': 'get_",
        "weather', 'arguments': {'city': 'To",
        "kyo'}}]",
    ];
    let mut calls = Vec::new();
    for c in chunks {
        let r = parser.parse_incremental(c, &[]).await.unwrap();
        calls.extend(r.calls);
    }
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
    let args: Value = serde_json::from_str(&calls[0].parameters).unwrap();
    assert_eq!(args["city"], "Tokyo");
}

#[tokio::test]
async fn test_streaming_with_marker_chunked() {
    let mut parser = SarashinaParser::new();
    let chunks = [
        "Sure.",
        "<|tool_ca",
        "lls|>[{'name': 'ping', 'arg",
        "uments': {}}]",
    ];
    let mut normal = String::new();
    let mut calls = Vec::new();
    for c in chunks {
        let r = parser.parse_incremental(c, &[]).await.unwrap();
        normal.push_str(&r.normal_text);
        calls.extend(r.calls);
    }
    assert_eq!(normal, "Sure.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name.as_deref(), Some("ping"));
}

#[tokio::test]
async fn test_streaming_markerless_whitespace_split() {
    // The `[` and the first `{` arrive in different chunks, separated by
    // whitespace — the look-ahead must keep buffering, not flush as text.
    let mut parser = SarashinaParser::new();
    let chunks = ["[ ", "  {'name': 'ping', 'arguments': {}}]"];
    let mut normal = String::new();
    let mut calls = Vec::new();
    for c in chunks {
        let r = parser.parse_incremental(c, &[]).await.unwrap();
        normal.push_str(&r.normal_text);
        calls.extend(r.calls);
    }
    assert_eq!(normal, "");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name.as_deref(), Some("ping"));
}

#[tokio::test]
async fn test_marker_not_leaked_on_malformed() {
    // Marker present but payload unbalanced: no tool call, and the
    // `<|tool_calls|>` special token must not leak into the text.
    let parser = SarashinaParser::new();
    let input = "before <|tool_calls|>[{'name': 'x'";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert!(!normal_text.contains("<|tool_calls|>"));
    assert!(normal_text.starts_with("before"));
}
