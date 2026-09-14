//! DeepSeek-V4.1 reasoning parser: `<think>`/`</think>` plus the spaced DSML
//! tool-call marker as an implicit end of reasoning, mirroring the reasoning
//! transitions of vLLM's `deepseek_v4` parser engine (inherited unchanged by
//! its `deepseek_v41` configuration, which only swaps in the spaced marker).

use reasoning_parser::{
    parsers::deepseek_v41::TOOL_BLOCK_START, ParseError, ParserFactory, ReasoningParser,
    DEFAULT_MAX_BUFFER_SIZE,
};

/// A chat-mode parser: the constructor state, nothing armed.
fn unarmed() -> Box<dyn ReasoningParser> {
    let parser = ParserFactory::new().create("deepseek_v41");
    assert_eq!(
        parser.model_type(),
        "deepseek_v41",
        "deepseek_v41 must be registered (the factory falls back to passthrough)"
    );
    parser
}

/// A thinking-mode parser the way the gateway arms it after a prompt that
/// ends in `<think>`.
fn armed() -> Box<dyn ReasoningParser> {
    let mut parser = unarmed();
    parser.mark_reasoning_started();
    parser.mark_think_start_stripped();
    parser
}

fn split(result: reasoning_parser::ParserResult) -> (String, String) {
    (result.reasoning_text, result.normal_text)
}

/// Feed `text` in chunks of `chunk_chars` characters (never splitting a
/// multi-byte `｜`) and concatenate the deltas.
#[expect(clippy::unwrap_used, reason = "test helper — panics are intentional")]
fn stream(parser: &mut dyn ReasoningParser, text: &str, chunk_chars: usize) -> (String, String) {
    let chars: Vec<char> = text.chars().collect();
    let (mut reasoning, mut normal) = (String::new(), String::new());
    for chunk in chars.chunks(chunk_chars) {
        let piece: String = chunk.iter().collect();
        let delta = parser
            .parse_reasoning_streaming_incremental(&piece)
            .unwrap();
        reasoning.push_str(&delta.reasoning_text);
        normal.push_str(&delta.normal_text);
    }
    (reasoning, normal)
}

#[test]
fn factory_registers_and_maps_v41_model_names_before_v4() {
    let factory = ParserFactory::new();
    assert!(factory.registry().has_parser("deepseek_v41"));
    for model in [
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "deepseek-v4.1-flash",
        "deepseek_v41",
        "DeepSeek-V41",
    ] {
        assert_eq!(
            factory
                .registry()
                .create_for_model(model)
                .unwrap()
                .model_type(),
            "deepseek_v41",
            "{model}"
        );
    }
    assert_eq!(
        factory
            .registry()
            .create_for_model("deepseek-ai/DeepSeek-V4-Flash")
            .unwrap()
            .model_type(),
        "deepseek_v4"
    );
}

#[test]
fn model_type_arming_and_special_token_requirement() {
    let parser = unarmed();
    assert_eq!(parser.model_type(), "deepseek_v41");
    assert!(!parser.requires_special_tokens());
    assert!(!parser.is_in_reasoning());
    assert!(armed().is_in_reasoning());
}

#[test]
fn prefill_armed_split_is_verbatim() {
    // Spec D9: no trimming — the trailing space before `</think>` survives.
    let (reasoning, normal) = split(
        armed()
            .detect_and_parse_reasoning("Plan. </think>12")
            .unwrap(),
    );
    assert_eq!(reasoning, "Plan. ");
    assert_eq!(normal, "12");
}

#[test]
fn vllm_control_cases() {
    // vLLM tests/parser/engine/test_deepseek_v41.py::test_reasoning_adapter_controls_and_usage
    assert_eq!(
        split(armed().detect_and_parse_reasoning("</think>12").unwrap()),
        (String::new(), "12".to_string())
    );
    assert_eq!(
        split(
            armed()
                .detect_and_parse_reasoning("Plan.</think>12")
                .unwrap()
        ),
        ("Plan.".to_string(), "12".to_string())
    );
    assert_eq!(
        split(unarmed().detect_and_parse_reasoning("12").unwrap()),
        (String::new(), "12".to_string())
    );
}

#[test]
fn chat_mode_unarmed_passes_everything_as_content() {
    let (reasoning, normal) = split(
        unarmed()
            .detect_and_parse_reasoning("plain answer")
            .unwrap(),
    );
    assert_eq!(normal, "plain answer");
    assert_eq!(reasoning, "");
}

#[test]
fn tool_block_ends_an_unclosed_reasoning_block() {
    let text = "thinking\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let (reasoning, normal) = split(armed().detect_and_parse_reasoning(text).unwrap());
    assert_eq!(reasoning, "thinking\n\n");
    assert_eq!(normal, &text["thinking\n\n".len()..]);
    assert!(normal.starts_with(TOOL_BLOCK_START));
}

#[test]
fn first_marker_wins_when_a_quoted_tool_marker_precedes_think_end() {
    // vLLM's lexer takes the first terminal; the tool states never interpret
    // think markers again, so the later `</think>` passes through verbatim.
    // (SGLang's non-streaming path keeps the whole buffer as reasoning; not ported.)
    let (reasoning, normal) = split(
        armed()
            .detect_and_parse_reasoning("quote <｜DSML｜ calls></think>x")
            .unwrap(),
    );
    assert_eq!(reasoning, "quote ");
    assert_eq!(normal, "<｜DSML｜ calls></think>x");
}

#[test]
fn only_the_full_spaced_marker_ends_reasoning() {
    // A bare `<｜DSML｜` (SGLang's cut point) or V4's unspaced spelling is
    // reasoning text for V4.1.
    let (reasoning, normal) = split(
        armed()
            .detect_and_parse_reasoning("a <｜DSML｜ b</think>c")
            .unwrap(),
    );
    assert_eq!(reasoning, "a <｜DSML｜ b");
    assert_eq!(normal, "c");
    let (reasoning, normal) = split(
        armed()
            .detect_and_parse_reasoning("a <｜DSML｜tool_calls></think>c")
            .unwrap(),
    );
    assert_eq!(reasoning, "a <｜DSML｜tool_calls>");
    assert_eq!(normal, "c");
}

#[test]
fn duplicate_think_start_inside_reasoning_is_absorbed() {
    let (reasoning, normal) = split(
        armed()
            .detect_and_parse_reasoning("<think>a</think>b")
            .unwrap(),
    );
    assert_eq!(reasoning, "a");
    assert_eq!(normal, "b");
}

#[test]
fn bare_think_end_in_content_is_absorbed() {
    let (reasoning, normal) = split(unarmed().detect_and_parse_reasoning("</think>12").unwrap());
    assert_eq!(reasoning, "");
    assert_eq!(normal, "12");
}

#[test]
fn think_start_in_content_enters_reasoning() {
    let (reasoning, normal) = split(
        unarmed()
            .detect_and_parse_reasoning("x<think>r</think>y")
            .unwrap(),
    );
    assert_eq!(reasoning, "r");
    assert_eq!(normal, "xy");
}

#[test]
fn think_markers_after_a_tool_block_are_not_interpreted() {
    let text = "hi <｜DSML｜ calls>x</think>y<think>z";
    let (reasoning, normal) = split(unarmed().detect_and_parse_reasoning(text).unwrap());
    assert_eq!(reasoning, "");
    assert_eq!(normal, text);
}

#[test]
fn streaming_reproduces_the_one_shot_split_at_every_chunk_size() {
    let cases: [(bool, &str); 7] = [
        (true, "Let me think</think>Answer with <｜DSML｜ inside content"),
        (
            true,
            "thinking\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
        ),
        (true, "quote <｜DSML｜ calls></think>x"),
        (true, "</think>12"),
        (false, "x<think>r</think>y"),
        (false, "Answer <｜DSML｜ calls>tail</think>z"),
        (true, "I think <｜DSML｜ calm</think>x"),
    ];
    for (arm, text) in cases {
        let make = || if arm { armed() } else { unarmed() };
        let expected = split(make().detect_and_parse_reasoning(text).unwrap());
        for chunk_chars in [1usize, 3, 7, 1000] {
            let mut parser = make();
            let streamed = stream(parser.as_mut(), text, chunk_chars);
            assert_eq!(streamed, expected, "chunk size {chunk_chars} for {text:?}");
        }
    }
}

#[test]
fn streaming_holds_back_a_partial_marker_until_it_resolves() {
    let mut parser = armed();
    let first = parser
        .parse_reasoning_streaming_incremental("Let me think</thi")
        .unwrap();
    assert_eq!(first.reasoning_text, "Let me think");
    assert_eq!(first.normal_text, "");
    let second = parser
        .parse_reasoning_streaming_incremental("nk>Answer <｜DSML｜ ca")
        .unwrap();
    assert_eq!(second.reasoning_text, "");
    assert_eq!(second.normal_text, "Answer ");
    let third = parser.parse_reasoning_streaming_incremental("lm").unwrap();
    assert_eq!(third.normal_text, "<｜DSML｜ calm");
    assert!(!parser.is_in_reasoning());
}

#[test]
fn reset_returns_to_chat_mode_and_clears_the_buffer() {
    let mut parser = armed();
    let held = parser
        .parse_reasoning_streaming_incremental("partial <")
        .unwrap();
    assert_eq!(held.reasoning_text, "partial ");
    parser.reset();
    assert!(!parser.is_in_reasoning());
    let (reasoning, normal) = split(parser.detect_and_parse_reasoning("plain").unwrap());
    assert_eq!(reasoning, "");
    assert_eq!(normal, "plain");
}

#[test]
fn buffer_overflow_is_an_error_on_both_paths() {
    let oversized = "a".repeat(DEFAULT_MAX_BUFFER_SIZE + 1);
    assert!(matches!(
        armed().detect_and_parse_reasoning(&oversized),
        Err(ParseError::BufferOverflow(size)) if size == DEFAULT_MAX_BUFFER_SIZE + 1
    ));
    let mut parser = armed();
    parser.parse_reasoning_streaming_incremental("<").unwrap();
    assert!(matches!(
        parser.parse_reasoning_streaming_incremental(&oversized),
        Err(ParseError::BufferOverflow(size)) if size == DEFAULT_MAX_BUFFER_SIZE + 2
    ));
}

#[test]
fn a_marker_prefix_that_diverges_late_stays_reasoning_text() {
    // `<｜DSML｜ cal` could still become the tool marker; once the next byte
    // diverges it is reasoning text, and `</think>` ends the block as usual.
    let (reasoning, normal) = split(
        armed()
            .detect_and_parse_reasoning("I think <｜DSML｜ calm</think>x")
            .unwrap(),
    );
    assert_eq!(reasoning, "I think <｜DSML｜ calm");
    assert_eq!(normal, "x");
}

#[test]
fn flush_emits_the_held_back_tail_once() {
    let mut parser = armed();
    let first = parser
        .parse_reasoning_streaming_incremental("held <｜DSML｜ ca")
        .unwrap();
    assert_eq!(first.reasoning_text, "held ");
    assert_eq!(
        parser.flush().unwrap(),
        reasoning_parser::ParserResult::reasoning("<｜DSML｜ ca".to_string())
    );
    assert!(parser.flush().unwrap().is_empty());

    let mut parser = unarmed();
    parser
        .parse_reasoning_streaming_incremental("answer </thi")
        .unwrap();
    assert_eq!(
        parser.flush().unwrap(),
        reasoning_parser::ParserResult::normal("</thi".to_string())
    );
    parser.reset();
    assert!(parser.flush().unwrap().is_empty());
}
