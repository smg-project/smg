//! Streaming buffered-content flush tests
//!
//! Regression tests for the streaming content-loss bug observed on PR #2261's
//! e2e runs (all four engines, deterministic): Llama-3.2-1B with
//! `--tool-call-parser llama`, tools present, `tool_choice=auto`, streaming.
//! When the model emits `{`-prefixed text that never becomes a valid declared
//! tool call, the incremental parser buffered everything while waiting for a
//! tool call that never materialized and the client received a completely
//! empty stream — no tool_call deltas AND no content deltas — while the
//! non-streaming path correctly fell back to returning the text as content.
//!
//! Two properties are covered here:
//! 1. Text that is definitively not a declared tool call (complete JSON with
//!    a missing or undeclared name) is surfaced as normal text mid-stream
//!    instead of being buffered forever or silently dropped.
//! 2. Text still buffered at end of stream (truncated JSON, partial start
//!    markers) is recoverable via the end-of-stream flush instead of being
//!    swallowed.
mod common;

use common::{create_test_tools, streaming_helpers};
use tool_parser::{
    types::ToolCallItem, CohereParser, JsonParser, LlamaParser, MistralParser, QwenParser,
    ToolParser,
};

/// Representative Llama-3.2-1B output for "What's the weather in Tokyo?" with
/// declared tools: `{`-prefixed JSON that never becomes a valid declared tool
/// call because the function name is under a non-standard key (the parser
/// only recognizes `name`). The CI worker-log artifacts do not record raw
/// generations, so this reproduces the failure class rather than the literal
/// string.
const NON_TOOL_JSON: &str =
    r#"{"type": "function", "function": "get_weather", "parameters": {"city": "Tokyo"}}"#;

/// Valid, complete JSON in llama tool shape, but the name is not among the
/// declared tools.
const UNDECLARED_TOOL_JSON: &str =
    r#"{"name": "get_stock_price", "parameters": {"ticker": "AAPL"}}"#;

/// Drive a parser through chunked streaming and collect everything it emits.
#[expect(
    clippy::unwrap_used,
    reason = "test helper; allow-unwrap-in-tests only covers #[test] fns"
)]
async fn stream_chunks(
    parser: &mut dyn ToolParser,
    chunks: &[&str],
) -> (String, Vec<ToolCallItem>) {
    let tools = create_test_tools();
    let mut normal_text = String::new();
    let mut calls = Vec::new();
    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        normal_text.push_str(&result.normal_text);
        calls.extend(result.calls);
    }
    (normal_text, calls)
}

// ============================================================================
// Property 1: definitively-not-a-tool JSON must surface as normal text
// ============================================================================

#[tokio::test]
async fn test_llama_streaming_non_tool_json_not_swallowed() {
    let mut parser = LlamaParser::new();
    // Realistic 2-3 char chunks: many chunk boundaries inside the JSON.
    let chunks = streaming_helpers::create_realistic_chunks(NON_TOOL_JSON);
    let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();

    let (normal_text, calls) = stream_chunks(&mut parser, &chunk_refs).await;

    assert!(
        calls.is_empty(),
        "no declared tool call was made: {calls:?}"
    );
    assert_eq!(
        normal_text, NON_TOOL_JSON,
        "non-tool JSON text must be streamed as content, not swallowed"
    );
}

#[tokio::test]
async fn test_llama_streaming_undeclared_tool_name_surfaces_as_text() {
    let mut parser = LlamaParser::new();
    // Chunk boundaries inside the JSON, including inside the (undeclared) name.
    let chunks = [
        r#"{"name": "get_st"#,
        r#"ock_price", "para"#,
        r#"meters": {"ticker": "AAPL"}}"#,
    ];

    let (normal_text, calls) = stream_chunks(&mut parser, &chunks).await;

    assert!(
        calls.is_empty(),
        "undeclared tool must not be emitted: {calls:?}"
    );
    assert_eq!(
        normal_text, UNDECLARED_TOOL_JSON,
        "undeclared-tool JSON must be surfaced as content, not dropped"
    );
}

#[tokio::test]
async fn test_json_streaming_non_tool_json_not_swallowed() {
    let mut parser = JsonParser::new();
    let input = r#"{"result": 42, "status": "ok"}"#;
    let chunks = streaming_helpers::create_realistic_chunks(input);
    let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();

    let (normal_text, calls) = stream_chunks(&mut parser, &chunk_refs).await;

    assert!(calls.is_empty());
    assert_eq!(
        normal_text, input,
        "non-tool JSON text must be streamed as content, not swallowed"
    );
}

#[tokio::test]
async fn test_mistral_streaming_undeclared_tool_name_not_dropped() {
    let mut parser = MistralParser::new();
    let input = r#"[TOOL_CALLS] [{"name": "frobnicate", "arguments": {"x": 1}}]"#;
    let chunks = [
        "[TOOL_CALLS] ",
        r#"[{"name": "frobni"#,
        r#"cate", "arguments"#,
        r#"": {"x": 1}}]"#,
    ];

    let (normal_text, calls) = stream_chunks(&mut parser, &chunks).await;

    assert!(
        calls.is_empty(),
        "undeclared tool must not be emitted: {calls:?}"
    );
    // Content-preserving: everything the model emitted must reach the client
    // as text (the non-streaming path never drops it silently either).
    let compact: String = normal_text.chars().filter(|c| !c.is_whitespace()).collect();
    let expected: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(compact, expected, "undeclared-tool text was dropped");
}

#[tokio::test]
async fn test_qwen_streaming_non_tool_json_in_markers_not_swallowed() {
    let mut parser = QwenParser::new();
    // Complete JSON inside qwen markers but with no "name" field at all.
    let chunks = ["<tool_call>\n", r#"{"foo": "#, "1}", "\n</tool_call>"];

    let (normal_text, calls) = stream_chunks(&mut parser, &chunks).await;

    assert!(calls.is_empty());
    assert!(
        normal_text.contains(r#"{"foo": 1}"#),
        "non-tool JSON inside markers must be surfaced as content, got {normal_text:?}"
    );
}

// ============================================================================
// Property 1b: legitimate tool calls must be unaffected
// ============================================================================

#[tokio::test]
async fn test_llama_streaming_valid_tool_call_still_works() {
    let mut parser = LlamaParser::new();
    let chunks = [
        r#"{"name": "get_we"#,
        r#"ather", "parameters"#,
        r#"": {"city": "Tokyo"}}"#,
    ];

    let (normal_text, calls) = stream_chunks(&mut parser, &chunks).await;

    assert_eq!(normal_text, "", "no content expected for a valid tool call");
    assert!(
        calls
            .iter()
            .any(|c| c.name.as_deref() == Some("get_weather")),
        "declared tool call must still be announced: {calls:?}"
    );
    let args: String = calls.iter().map(|c| c.parameters.as_str()).collect();
    assert!(
        args.contains("Tokyo"),
        "arguments must be streamed: {args:?}"
    );
}

#[tokio::test]
async fn test_llama_streaming_partial_declared_name_keeps_buffering() {
    let mut parser = LlamaParser::new();
    let tools = create_test_tools();

    // "get_we" is a prefix of the declared "get_weather": the parser must NOT
    // bail out to normal text while the name string is still open.
    let result = parser
        .parse_incremental(r#"{"name": "get_we"#, &tools)
        .await
        .unwrap();
    assert_eq!(result.normal_text, "");
    assert!(result.calls.is_empty());

    let result = parser
        .parse_incremental(r#"ather", "parameters": {"city": "Paris"}}"#, &tools)
        .await
        .unwrap();
    let mut calls = result.calls;
    calls.extend(parser.parse_incremental("", &tools).await.unwrap().calls);
    assert!(
        calls
            .iter()
            .any(|c| c.name.as_deref() == Some("get_weather")),
        "declared tool call must be announced after the name completes: {calls:?}"
    );
}

// ============================================================================
// Property 2: end-of-stream flush recovers buffered text
// ============================================================================

#[tokio::test]
async fn test_llama_truncated_json_flushed_at_end_of_stream() {
    let mut parser = LlamaParser::new();
    let tools = create_test_tools();

    // Stream ends mid-name: the parser must hold it while streaming (it could
    // still become "get_weather"), but surface it at end of stream.
    let truncated = r#"{"name": "get_wea"#;
    let result = parser.parse_incremental(truncated, &tools).await.unwrap();
    assert_eq!(result.normal_text, "");
    assert!(result.calls.is_empty());

    assert_eq!(
        parser.take_unstreamed_normal_text(),
        truncated,
        "text buffered at end of stream must be flushed as content"
    );
    // Drained: a second flush returns nothing.
    assert_eq!(parser.take_unstreamed_normal_text(), "");
}

#[tokio::test]
async fn test_llama_partial_bot_token_flushed_at_end_of_stream() {
    let mut parser = LlamaParser::new();
    let tools = create_test_tools();

    let text = "Sure, let me check <|py";
    let result = parser.parse_incremental(text, &tools).await.unwrap();
    assert_eq!(result.normal_text, "");
    assert!(result.calls.is_empty());

    assert_eq!(parser.take_unstreamed_normal_text(), text);
}

#[tokio::test]
async fn test_llama_flush_empty_after_completed_tool_call() {
    let mut parser = LlamaParser::new();
    let tools = create_test_tools();

    let result = parser
        .parse_incremental(
            r#"{"name": "get_weather", "parameters": {"city": "Tokyo"}}"#,
            &tools,
        )
        .await
        .unwrap();
    let mut calls = result.calls;
    calls.extend(parser.parse_incremental("", &tools).await.unwrap().calls);
    assert!(calls
        .iter()
        .any(|c| c.name.as_deref() == Some("get_weather")));

    assert_eq!(
        parser.take_unstreamed_normal_text(),
        "",
        "no content must be invented after a real tool call"
    );
}

#[tokio::test]
async fn test_llama_flush_empty_after_announced_tool_truncated_args() {
    let mut parser = LlamaParser::new();
    let tools = create_test_tools();

    // Name announced, arguments truncated at end of stream: the buffered tail
    // is tool syntax, not content — flushing it as text would duplicate the
    // tool call. Remaining args are recovered via get_unstreamed_tool_args.
    let result = parser
        .parse_incremental(
            r#"{"name": "get_weather", "parameters": {"city": "Par"#,
            &tools,
        )
        .await
        .unwrap();
    assert!(result
        .calls
        .iter()
        .any(|c| c.name.as_deref() == Some("get_weather")));

    assert_eq!(
        parser.take_unstreamed_normal_text(),
        "",
        "announced tool call tail must not be re-emitted as content"
    );
}

#[tokio::test]
async fn test_json_truncated_json_flushed_at_end_of_stream() {
    let mut parser = JsonParser::new();
    let tools = create_test_tools();

    let truncated = r#"{"name": "sea"#;
    let result = parser.parse_incremental(truncated, &tools).await.unwrap();
    assert_eq!(result.normal_text, "");
    assert!(result.calls.is_empty());

    assert_eq!(parser.take_unstreamed_normal_text(), truncated);
}

#[tokio::test]
async fn test_mistral_truncated_tool_call_flushed_at_end_of_stream() {
    let mut parser = MistralParser::new();
    let tools = create_test_tools();

    let truncated = r#"[TOOL_CALLS] [{"unknown"#;
    let result = parser.parse_incremental(truncated, &tools).await.unwrap();
    assert_eq!(result.normal_text, "");
    assert!(result.calls.is_empty());

    assert_eq!(parser.take_unstreamed_normal_text(), truncated);
}

#[tokio::test]
async fn test_qwen_truncated_tool_call_flushed_at_end_of_stream() {
    let mut parser = QwenParser::new();
    let tools = create_test_tools();

    let truncated = "<tool_call>\n{\"a\": 1";
    let result = parser.parse_incremental(truncated, &tools).await.unwrap();
    assert_eq!(result.normal_text, "");
    assert!(result.calls.is_empty());

    assert_eq!(parser.take_unstreamed_normal_text(), truncated);
}

#[tokio::test]
async fn test_cohere_unterminated_action_flushed_at_end_of_stream() {
    let mut parser = CohereParser::new();
    let tools = create_test_tools();

    // START_ACTION seen, END_ACTION never arrives (truncated stream).
    let json_part = r#"{"tool_name": "search", "parameters": {"query": "x"#;
    let result = parser
        .parse_incremental(&format!("<|START_ACTION|>{json_part}"), &tools)
        .await
        .unwrap();
    assert_eq!(result.normal_text, "");
    assert!(result.calls.is_empty());

    assert_eq!(
        parser.take_unstreamed_normal_text(),
        json_part,
        "unterminated action block must be flushed as content"
    );
}

#[tokio::test]
async fn test_flush_empty_after_reset() {
    let mut parser = LlamaParser::new();
    let tools = create_test_tools();

    parser
        .parse_incremental(r#"{"name": "get_wea"#, &tools)
        .await
        .unwrap();
    parser.reset();

    assert_eq!(
        parser.take_unstreamed_normal_text(),
        "",
        "reset must clear the streaming buffer"
    );
}
