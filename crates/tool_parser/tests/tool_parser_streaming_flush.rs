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
//! Four properties are covered here:
//! 1. Text that is definitively not a declared tool call (complete JSON with
//!    a missing or undeclared name) is surfaced as normal text mid-stream
//!    instead of being buffered forever or silently dropped, while legitimate
//!    tool calls keep streaming unchanged.
//! 2. Text still buffered at end of stream (truncated JSON, partial start
//!    markers) is recoverable via the end-of-stream flush instead of being
//!    swallowed — and nothing is invented once a tool call was announced.
//! 3. A declared tool call trailing a non-tool value in the same chunk still
//!    becomes tool-call deltas rather than stranded text.
//! 4. A call whose name and arguments both land in one chunk streams *both*
//!    in that chunk. There is no later parser call to fall back on, so a name
//!    without arguments would reach the client as an unusable tool call.
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

type MakeParser = fn() -> Box<dyn ToolParser>;

fn llama() -> Box<dyn ToolParser> {
    Box::new(LlamaParser::new())
}
fn json() -> Box<dyn ToolParser> {
    Box::new(JsonParser::new())
}
fn mistral() -> Box<dyn ToolParser> {
    Box::new(MistralParser::new())
}
fn qwen() -> Box<dyn ToolParser> {
    Box::new(QwenParser::new())
}
fn cohere() -> Box<dyn ToolParser> {
    Box::new(CohereParser::new())
}

/// How a case chunks its input.
enum Feed {
    /// Explicit chunk boundaries, chosen to split mid-token.
    Chunks(&'static [&'static str]),
    /// Realistic 2-3 char chunks: many chunk boundaries inside the JSON.
    Realistic(&'static str),
}

/// Drive a parser through chunked streaming and collect everything it emits.
#[expect(
    clippy::unwrap_used,
    reason = "test helper; allow-unwrap-in-tests only covers #[test] fns"
)]
async fn stream(parser: &mut dyn ToolParser, feed: &Feed) -> (String, Vec<ToolCallItem>) {
    let owned_chunks;
    let owned_refs;
    let chunks: &[&str] = match feed {
        Feed::Chunks(chunks) => chunks,
        Feed::Realistic(input) => {
            owned_chunks = streaming_helpers::create_realistic_chunks(input);
            owned_refs = owned_chunks.iter().map(String::as_str).collect::<Vec<_>>();
            &owned_refs
        }
    };
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

fn announced(calls: &[ToolCallItem]) -> Option<&str> {
    calls.iter().find_map(|c| c.name.as_deref())
}

/// Everything the parser streamed as tool-call argument fragments.
fn streamed_args(calls: &[ToolCallItem]) -> String {
    calls.iter().map(|c| c.parameters.as_str()).collect()
}

// ============================================================================
// One table, three properties per row: what the client sees streamed, which
// declared tool (if any) was announced, and what the end-of-stream flush
// yields. Asserting all three for every case is what pins the bug: the
// regression produced an empty stream *and* an empty flush.
// ============================================================================

struct Case {
    label: &'static str,
    make: MakeParser,
    feed: Feed,
    /// Exactly what the client must receive as content.
    text: &'static str,
    /// The declared tool the parser must announce, if any.
    call: Option<&'static str>,
    /// A substring the streamed arguments must contain, when a call is made.
    args_contain: Option<&'static str>,
    /// What the end-of-stream flush must return.
    flush: &'static str,
}

/// Defaults for the common shape: streams nothing, announces nothing, flushes
/// nothing. Each row overrides only what it is actually about.
const BLANK: Case = Case {
    label: "",
    make: llama,
    feed: Feed::Chunks(&[]),
    text: "",
    call: None,
    args_contain: None,
    flush: "",
};

#[tokio::test]
async fn streaming_never_swallows_content() {
    let cases = [
        // --- definitively-not-a-tool JSON must surface as content mid-stream
        Case {
            label: "llama: no `name` key at all, split mid-token",
            feed: Feed::Realistic(NON_TOOL_JSON),
            text: NON_TOOL_JSON,
            ..BLANK
        },
        Case {
            label: "llama: undeclared name split across chunk boundaries",
            feed: Feed::Chunks(&[
                r#"{"name": "get_st"#,
                r#"ock_price", "para"#,
                r#"meters": {"ticker": "AAPL"}}"#,
            ]),
            text: UNDECLARED_TOOL_JSON,
            ..BLANK
        },
        Case {
            label: "json: plain answer that happens to be JSON",
            make: json,
            feed: Feed::Realistic(r#"{"result": 42, "status": "ok"}"#),
            text: r#"{"result": 42, "status": "ok"}"#,
            ..BLANK
        },
        Case {
            // Content-preserving, markers included: everything the model
            // emitted reaches the client, as the non-streaming path does too.
            label: "mistral: undeclared name inside [TOOL_CALLS] array",
            make: mistral,
            feed: Feed::Chunks(&[
                "[TOOL_CALLS] ",
                r#"[{"name": "frobni"#,
                r#"cate", "arguments"#,
                r#"": {"x": 1}}]"#,
            ]),
            text: r#"[TOOL_CALLS] [{"name": "frobnicate", "arguments": {"x": 1}}]"#,
            ..BLANK
        },
        Case {
            label: "qwen: complete JSON inside markers with no `name` key",
            make: qwen,
            feed: Feed::Chunks(&["<tool_call>\n", r#"{"foo": "#, "1}", "\n</tool_call>"]),
            text: "<tool_call>\n{\"foo\": 1}\n</tool_call>",
            ..BLANK
        },
        // --- legitimate tool calls keep streaming, and invent no content
        Case {
            label: "llama: a real declared call still streams name and args",
            feed: Feed::Chunks(&[
                r#"{"name": "get_we"#,
                r#"ather", "parameters"#, // codespell:ignore ather
                r#"": {"city": "Tokyo"}}"#,
            ]),
            call: Some("get_weather"),
            args_contain: Some(r#"{"city":"Tokyo"}"#),
            ..BLANK
        },
        Case {
            label: "llama: nothing invented after a completed tool call",
            feed: Feed::Chunks(&[r#"{"name": "get_weather", "parameters": {"city": "Tokyo"}}"#]),
            call: Some("get_weather"),
            args_contain: Some(r#"{"city":"Tokyo"}"#),
            ..BLANK
        },
        Case {
            // The buffered tail is tool syntax, not content: flushing it as
            // text would duplicate the announced call.
            label: "llama: announced call whose arguments are truncated",
            feed: Feed::Chunks(&[r#"{"name": "get_weather", "parameters": {"city": "Par"#]),
            call: Some("get_weather"),
            ..BLANK
        },
        // --- a call whose name and arguments arrive together must stream both
        //     in that single invocation: no later call ever comes
        Case {
            label: "json: whole declared call inside one chunk",
            make: json,
            feed: Feed::Chunks(&[r#"{"name": "get_weather", "arguments": {"city": "Tokyo"}}"#]),
            call: Some("get_weather"),
            args_contain: Some(r#"{"city":"Tokyo"}"#),
            ..BLANK
        },
        Case {
            label: "mistral: declared call after an undeclared one streams its args",
            make: mistral,
            feed: Feed::Chunks(&[
                r#"[TOOL_CALLS] [{"name": "bogus", "arguments": {}}, {"name": "get_weather", "arguments": {"city": "Paris"}}"#,
                "]",
            ]),
            text: r#"[TOOL_CALLS] [{"name": "bogus", "arguments": {}}, "#,
            call: Some("get_weather"),
            args_contain: Some(r#"{"city":"Paris"}"#),
            ..BLANK
        },
        Case {
            label: "cohere: complete action block streams name and arguments",
            make: cohere,
            feed: Feed::Chunks(&[
                r#"<|START_ACTION|>{"tool_name": "search", "parameters": {"query": "rust"}}<|END_ACTION|>"#,
                " done",
            ]),
            call: Some("search"),
            args_contain: Some(r#"{"query":"rust"}"#),
            ..BLANK
        },
        Case {
            // The helper bails out on the undeclared, still-incomplete value
            // and emits the whole chunk, so `normal_text_buffer` holds back
            // the trailing partial `</tool_call>`. With no tool call ever
            // announced that suffix is real text, not marker debris.
            label: "qwen: partial </tool_call> held in normal_text_buffer",
            make: qwen,
            feed: Feed::Chunks(&[r#"<tool_call>
{"name": "bogus"</tool"#]),
            text: "<tool_call>\n{\"name\": \"bogus\"",
            flush: "</tool",
            ..BLANK
        },
        Case {
            // Every other cohere emission path strips response markers; the
            // flush must too, rather than leaking control tokens as content.
            label: "cohere: response markers stripped from the flushed text",
            make: cohere,
            feed: Feed::Chunks(&["<|START_RESPONSE|>Hello<|START_AC"]),
            flush: "Hello<|START_AC",
            ..BLANK
        },
        // --- text still buffered at end of stream is recovered by the flush
        Case {
            label: "llama: stream ends mid-name (could still become get_weather)",
            feed: Feed::Chunks(&[r#"{"name": "get_wea"#]),
            flush: r#"{"name": "get_wea"#,
            ..BLANK
        },
        Case {
            label: "llama: partial <|python_tag|> held back at end of stream",
            feed: Feed::Chunks(&["Sure, let me check <|py"]),
            flush: "Sure, let me check <|py",
            ..BLANK
        },
        Case {
            label: "json: truncated prefix of the declared `search`",
            make: json,
            feed: Feed::Chunks(&[r#"{"name": "sea"#]),
            flush: r#"{"name": "sea"#,
            ..BLANK
        },
        Case {
            label: "mistral: truncated tool call after the marker",
            make: mistral,
            feed: Feed::Chunks(&[r#"[TOOL_CALLS] [{"unknown"#]),
            flush: r#"[TOOL_CALLS] [{"unknown"#,
            ..BLANK
        },
        Case {
            label: "qwen: truncated tool call after the marker",
            make: qwen,
            feed: Feed::Chunks(&["<tool_call>\n{\"a\": 1"]),
            flush: "<tool_call>\n{\"a\": 1",
            ..BLANK
        },
        Case {
            label: "cohere: START_ACTION seen, END_ACTION never arrived",
            make: cohere,
            feed: Feed::Chunks(&[
                r#"<|START_ACTION|>{"tool_name": "search", "parameters": {"query": "x"#,
            ]),
            flush: r#"{"tool_name": "search", "parameters": {"query": "x"#,
            ..BLANK
        },
        // --- a declared call trailing a non-tool value in the SAME chunk must
        //     still become tool-call deltas, never stranded text
        Case {
            label: "json: adjacent non-tool value then a declared call",
            make: json,
            feed: Feed::Chunks(&[
                r#"{"note": "checking"} {"name": "get_weather", "arguments": {"city": "Tokyo"}}"#,
            ]),
            text: r#"{"note": "checking"} "#,
            call: Some("get_weather"),
            args_contain: Some(r#"{"city":"Tokyo"}"#),
            ..BLANK
        },
        Case {
            label: "json: adjacent undeclared call then a declared call",
            make: json,
            feed: Feed::Chunks(&[concat!(
                r#"{"name": "bogus_tool", "arguments": {}}"#,
                r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#
            )]),
            text: r#"{"name": "bogus_tool", "arguments": {}}"#,
            call: Some("get_weather"),
            args_contain: Some(r#"{"city":"Paris"}"#),
            ..BLANK
        },
    ];

    for case in &cases {
        let mut parser = (case.make)();
        let (text, calls) = stream(parser.as_mut(), &case.feed).await;

        assert_eq!(text, case.text, "[{}] streamed content", case.label);
        assert_eq!(announced(&calls), case.call, "[{}] tool call", case.label);
        if let Some(fragment) = case.args_contain {
            let args = streamed_args(&calls);
            assert!(
                args.contains(fragment),
                "[{}] arguments must be streamed, got {args:?}",
                case.label
            );
        }
        assert_eq!(
            parser.take_unstreamed_normal_text(),
            case.flush,
            "[{}] end-of-stream flush",
            case.label
        );
        assert_eq!(
            parser.take_unstreamed_normal_text(),
            "",
            "[{}] flush must drain: a second take returns nothing",
            case.label
        );
    }
}

#[tokio::test]
async fn partial_declared_name_keeps_buffering() {
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
        .parse_incremental(r#"ather", "parameters": {"city": "Paris"}}"#, &tools) // codespell:ignore ather
        .await
        .unwrap();
    assert_eq!(
        announced(&result.calls),
        Some("get_weather"),
        "declared tool call must be announced once the name completes"
    );
}

#[tokio::test]
async fn reset_clears_the_streaming_buffers() {
    let tools = create_test_tools();

    // The shared buffer every helper-based parser holds.
    let mut llama = LlamaParser::new();
    llama
        .parse_incremental(r#"{"name": "get_wea"#, &tools)
        .await
        .unwrap();
    llama.reset();
    assert_eq!(
        llama.take_unstreamed_normal_text(),
        "",
        "reset must clear the streaming buffer"
    );

    // Qwen additionally parks a partial `</tool_call>` in `normal_text_buffer`,
    // which reset() must clear too or it bleeds into the next request.
    let held = r#"<tool_call>
{"name": "bogus"</tool"#;
    let mut qwen = QwenParser::new();
    qwen.parse_incremental(held, &tools).await.unwrap();
    let mut drained = QwenParser::new();
    drained.parse_incremental(held, &tools).await.unwrap();
    assert_eq!(
        drained.take_unstreamed_normal_text(),
        "</tool",
        "precondition: normal_text_buffer is holding the partial end token"
    );

    qwen.reset();
    assert_eq!(
        qwen.take_unstreamed_normal_text(),
        "",
        "reset must clear normal_text_buffer, not just the main buffer"
    );
}
