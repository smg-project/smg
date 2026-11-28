use serde_json::json;
use tool_parser::{InklingParser, ParserFactory, ToolParser};

use crate::common::create_test_tools;

#[tokio::test]
async fn test_inkling_complete_parsing() {
    let parser = InklingParser::new();
    let tools = create_test_tools();
    let input = concat!(
        "<|message_model|><|content_text|>I'll check.<|end_message|>",
        "<|message_model|>search<|content_invoke_tool_json|>{\"name\":\"search\",\"args\":",
        "{\"query\":\"Rust {language}\"}}<|end_message|>",
        "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":",
        "{\"city\":\"Tokyo\"}}<|end_message|>",
        "<|content_model_end_sampling|>"
    );

    let (normal_text, calls) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();

    assert_eq!(normal_text, "I'll check.");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "search");
    assert_eq!(calls[1].function.name, "get_weather");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
        json!({"query": "Rust {language}"})
    );
}

#[tokio::test]
async fn test_inkling_rejects_undefined_tool() {
    let parser = InklingParser::new();
    let tools = create_test_tools();
    let input = concat!(
        "Before",
        "<|content_invoke_tool_json|>",
        "{\"name\":\"not_declared\",\"args\":{}}",
        "<|end_message|>"
    );

    let (normal_text, calls) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    assert!(calls.is_empty());
    assert_eq!(normal_text, "Before");
}

#[tokio::test]
async fn test_inkling_streaming_buffers_split_control_tokens() {
    let tools = create_test_tools();
    let mut parser = InklingParser::new();
    let chunks = [
        "<|message_",
        "model|><|content_",
        "text|>Before<|end_mes",
        "sage|><|message_model|>sea",
        "rch<|content_invoke_tool_json|>{\"name\":\"sea",
        "rch\",\"args\":{\"query\":\"Rust {lang}",
        "\"}}<|end_mes",
        "sage|><|content_",
        "text|>After<|content_model_end_",
        "sampling|>",
    ];

    let mut normal_text = String::new();
    let mut names = Vec::new();
    let mut arguments = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        normal_text.push_str(&result.normal_text);
        for call in result.calls {
            if let Some(name) = call.name {
                names.push((call.tool_index, name));
            }
            arguments.push_str(&call.parameters);
        }
    }

    assert_eq!(normal_text, "BeforeAfter");
    assert_eq!(names, vec![(0, "search".to_string())]);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
        json!({"query": "Rust {lang}"})
    );
}

#[tokio::test]
async fn test_inkling_streaming_skips_undefined_call() {
    let tools = create_test_tools();
    let mut parser = InklingParser::new();
    let chunks = [
        "<|content_invoke_tool_json|>{\"name\":\"unknown\",",
        "\"args\":{}}<|end_mes",
        "sage|><|content_invoke_tool_json|>",
        "{\"name\":\"search\",\"args\":{}}<|end_message|>",
    ];

    let mut calls = Vec::new();
    for chunk in chunks {
        calls.extend(parser.parse_incremental(chunk, &tools).await.unwrap().calls);
    }

    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].tool_index, 0);
    assert_eq!(calls[0].name.as_deref(), Some("search"));
    assert_eq!(calls[1].parameters, "{}");
}

#[test]
fn test_inkling_factory_and_structural_tag() {
    let factory = ParserFactory::new();
    assert!(factory.has_parser("inkling"));
    assert_eq!(
        factory
            .registry()
            .resolve_model_to_parser("Inkling-7B")
            .as_deref(),
        Some("inkling")
    );
    // Namespaced / differently-cased ids must still resolve (case-insensitive substring).
    assert_eq!(
        factory
            .registry()
            .resolve_model_to_parser("org/Inkling-Chat")
            .as_deref(),
        Some("inkling")
    );
    assert!(factory.registry().has_structural_tag("inkling"));

    let tools = create_test_tools();
    let tag = InklingParser::build_structural_tag(&tools[..1], true);
    assert_eq!(tag["format"]["type"], "triggered_tags");
    assert_eq!(
        tag["format"]["tags"][0]["begin"],
        "<|content_invoke_tool_json|>{\"name\":\"search\",\"args\":"
    );
    assert_eq!(tag["format"]["tags"][0]["end"], "}<|end_message|>");
    assert_eq!(tag["format"]["at_least_one"], true);
}

#[tokio::test]
async fn test_inkling_namespaced_model_parses_tool_calls() {
    let factory = ParserFactory::new();
    let parser = factory
        .get_parser("org/Inkling-Chat")
        .expect("namespaced inkling id should resolve to a parser");
    let tools = create_test_tools();
    let input = concat!(
        "<|message_model|>search<|content_invoke_tool_json|>{\"name\":\"search\",\"args\":",
        "{\"query\":\"Rust\"}}<|end_message|><|content_model_end_sampling|>"
    );

    let (_normal_text, calls) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
}

#[tokio::test]
async fn test_inkling_complete_recovers_after_malformed_tool_frame() {
    let input = concat!(
        "<|message_model|><|content_text|>Before<|end_message|>",
        "<|content_invoke_tool_json|>not-json<|end_message|>",
        "<|message_model|><|content_text|>After malformed.<|end_message|>",
        "<|content_invoke_tool_json|>{\"name\":\"search\",\"args\":{}}<|end_message|>",
        "<|message_model|><|content_text|>Tail<|end_message|>",
        "<|content_model_end_sampling|>"
    );
    let tools = create_test_tools();
    let parser = InklingParser::new();

    let (normal_text, calls) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();

    assert_eq!(normal_text, "BeforeAfter malformed.Tail");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
    assert_eq!(calls[0].function.arguments, "{}");
}

#[tokio::test]
async fn test_inkling_text_mode_is_safely_suppressed() {
    let input = concat!(
        "<|message_model|><|content_text|>Before<|end_message|>",
        "<|content_invoke_tool_text|>search for weather in SF<|end_message|>",
        "<|content_model_end_sampling|>"
    );
    let tools = create_test_tools();

    let parser = InklingParser::new();
    let (normal_text, calls) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    assert_eq!(normal_text, "Before");
    assert!(calls.is_empty());
}

/// A JSON answer whose strings spell every kind of protocol token, an escaped
/// quote before one of them, and non-ASCII text.
const QUOTED_PROTOCOL_ANSWER: &str = concat!(
    "{\"call\":\"<|content_invoke_tool_json|>{\\\"name\\\":\\\"search\\\"}<|end_message|>\",",
    "\"<|message_model|>key\":\"\\\"<|content_thinking|>é<|content_text|>\",",
    "\"text\":\"<|content_invoke_tool_text|>日本<|content_model_end_sampling|>\"}"
);

#[expect(clippy::unwrap_used, reason = "test helper — panics are intentional")]
async fn stream_json_mode(chunks: &[&str]) -> (String, Vec<String>, String) {
    let tools = create_test_tools();
    let mut parser = InklingParser::new();
    parser.set_json_mode(true);
    let mut normal_text = String::new();
    let mut names = Vec::new();
    let mut arguments = String::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        normal_text.push_str(&result.normal_text);
        for call in result.calls {
            names.extend(call.name);
            arguments.push_str(&call.parameters);
        }
    }
    (normal_text, names, arguments)
}

#[tokio::test]
async fn test_inkling_json_mode_keeps_quoted_protocol_text_complete() {
    let tools = create_test_tools();
    let mut parser = InklingParser::new();
    parser.set_json_mode(true);

    let (normal_text, calls) = parser
        .parse_complete_with_tools(QUOTED_PROTOCOL_ANSWER, &tools)
        .await
        .unwrap();

    assert!(calls.is_empty());
    assert_eq!(normal_text, QUOTED_PROTOCOL_ANSWER);
    serde_json::from_str::<serde_json::Value>(&normal_text).unwrap();
}

#[tokio::test]
async fn test_inkling_json_mode_keeps_quoted_protocol_text_streaming() {
    let characters = QUOTED_PROTOCOL_ANSWER
        .char_indices()
        .map(|(index, _)| index)
        .chain([QUOTED_PROTOCOL_ANSWER.len()])
        .collect::<Vec<_>>();

    // Every split into two chunks, then one chunk per character.
    for split in &characters {
        let (head, tail) = QUOTED_PROTOCOL_ANSWER.split_at(*split);
        let (normal_text, names, _) = stream_json_mode(&[head, tail]).await;
        assert_eq!(normal_text, QUOTED_PROTOCOL_ANSWER, "split at byte {split}");
        assert!(names.is_empty(), "split at byte {split}");
    }
    let chunks = characters
        .windows(2)
        .map(|bounds| &QUOTED_PROTOCOL_ANSWER[bounds[0]..bounds[1]])
        .collect::<Vec<_>>();
    let (normal_text, names, _) = stream_json_mode(&chunks).await;
    assert_eq!(normal_text, QUOTED_PROTOCOL_ANSWER);
    assert!(names.is_empty());
}

#[tokio::test]
async fn test_inkling_json_mode_still_parses_tool_frames_outside_strings() {
    let tools = create_test_tools();
    let input = concat!(
        "{\"note\":\"<|end_message|>\"}",
        "<|message_model|>search<|content_invoke_tool_json|>",
        "{\"name\":\"search\",\"args\":{\"query\":\"\\\"<|end_message|>\"}}",
        "<|end_message|><|content_model_end_sampling|>"
    );
    let mut parser = InklingParser::new();
    parser.set_json_mode(true);

    let (normal_text, calls) = parser
        .parse_complete_with_tools(input, &tools)
        .await
        .unwrap();
    assert_eq!(normal_text, "{\"note\":\"<|end_message|>\"}");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "search");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
        json!({"query": "\"<|end_message|>"})
    );

    let split = input.find("<|message_model|>").unwrap() + 3;
    let (head, tail) = input.split_at(split);
    let (normal_text, names, arguments) = stream_json_mode(&[head, tail]).await;
    assert_eq!(normal_text, "{\"note\":\"<|end_message|>\"}");
    assert_eq!(names, vec!["search".to_string()]);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
        json!({"query": "\"<|end_message|>"})
    );
}

#[tokio::test]
async fn test_inkling_reset_clears_json_quote_state() {
    let tools = create_test_tools();
    let mut parser = InklingParser::new();
    parser.set_json_mode(true);
    parser
        .parse_incremental("{\"open\":\"unterminated", &tools)
        .await
        .unwrap();

    parser.reset();
    let result = parser
        .parse_incremental("<|content_text|>{\"a\":1}<|end_message|>", &tools)
        .await
        .unwrap();
    assert_eq!(result.normal_text, "{\"a\":1}");
}

#[tokio::test]
async fn test_inkling_without_json_mode_still_strips_framing() {
    let tools = create_test_tools();
    let parser = InklingParser::new();
    let (normal_text, calls) = parser
        .parse_complete_with_tools("{\"v\":\"a<|end_message|>b\"}", &tools)
        .await
        .unwrap();
    assert!(calls.is_empty());
    assert_eq!(normal_text, "{\"v\":\"ab\"}");
}
