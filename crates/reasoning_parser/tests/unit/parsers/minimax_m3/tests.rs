use super::*;

fn collect_stream(parser: &mut MinimaxM3Parser, chunks: &[&str]) -> ParserResult {
    let mut result = ParserResult::default();
    for chunk in chunks {
        let parsed = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
        result.normal_text.push_str(&parsed.normal_text);
        result.reasoning_text.push_str(&parsed.reasoning_text);
    }
    result
}

#[test]
fn test_model_type() {
    let parser = MinimaxM3Parser::new();
    assert_eq!(parser.model_type(), "minimax_m3");
}

#[test]
fn test_fresh_parser_not_in_reasoning() {
    let parser = MinimaxM3Parser::new();
    // always_in_reasoning=false -> starts outside reasoning.
    assert!(!parser.is_in_reasoning());
}

#[test]
fn test_reasoning_extraction_complete() {
    let mut parser = MinimaxM3Parser::new();
    let result = parser
        .detect_and_parse_reasoning("<mm:think>thinking here</mm:think>normal text")
        .unwrap();
    assert_eq!(result.reasoning_text, "thinking here");
    assert_eq!(result.normal_text, "normal text");
}

#[test]
fn test_no_think_passthrough() {
    let mut parser = MinimaxM3Parser::new();
    // Without a start token, content is treated as normal text.
    let result = parser
        .detect_and_parse_reasoning("just a plain answer")
        .unwrap();
    assert_eq!(result.reasoning_text, "");
    assert_eq!(result.normal_text, "just a plain answer");
}

#[test]
fn test_nonstreaming_drops_repeated_leading_closers() {
    let mut parser = MinimaxM3Parser::new();
    for (input, expected) in [
        ("</mm:think>Hello.", "Hello."),
        ("</mm:think></mm:think>tool", "tool"),
    ] {
        let result = parser.detect_and_parse_reasoning(input).unwrap();
        assert_eq!(result, ParserResult::normal(expected.to_string()));
    }
}

#[test]
fn test_nonstreaming_preserves_non_leading_unmatched_closer() {
    let mut parser = MinimaxM3Parser::new();
    let result = parser
        .detect_and_parse_reasoning("visible</mm:think>text")
        .unwrap();
    assert_eq!(result.reasoning_text, "");
    assert_eq!(result.normal_text, "visible</mm:think>text");
}

#[test]
fn test_streaming_incremental_across_chunks() {
    let mut parser = MinimaxM3Parser::new();

    let r1 = parser
        .parse_reasoning_streaming_incremental("<mm:think>thinking ")
        .unwrap();
    assert_eq!(r1.reasoning_text, "thinking ");
    assert_eq!(r1.normal_text, "");

    let r2 = parser
        .parse_reasoning_streaming_incremental("about it</mm:think>the answer")
        .unwrap();
    assert_eq!(r2.reasoning_text, "about it");
    assert_eq!(r2.normal_text, "the answer");
}

#[test]
fn test_streaming_holds_end_marker_suffix_after_reasoning_text() {
    let mut parser = MinimaxM3Parser::new();
    let result = collect_stream(&mut parser, &["<mm:think>plan</mm:", "think>ANSWER"]);
    assert_eq!(result.reasoning_text, "plan");
    assert_eq!(result.normal_text, "ANSWER");
}

#[test]
fn test_streaming_drops_repeated_leading_closers_across_chunks() {
    let mut parser = MinimaxM3Parser::new();
    let result = collect_stream(
        &mut parser,
        &["</mm:", "think></mm:think>", "]<]minimax[>[<tool_call>"],
    );
    assert_eq!(result.reasoning_text, "");
    assert_eq!(result.normal_text, "]<]minimax[>[<tool_call>");
}

#[test]
fn test_streaming_one_character_chunks() {
    let text = "</mm:think></mm:think><mm:think>思考</mm:think>答案";
    let chunks: Vec<&str> = text
        .char_indices()
        .map(|(start, _)| start)
        .zip(
            text.char_indices()
                .map(|(start, _)| start)
                .skip(1)
                .chain(std::iter::once(text.len())),
        )
        .map(|(start, end)| &text[start..end])
        .collect();
    let mut parser = MinimaxM3Parser::new();
    let result = collect_stream(&mut parser, &chunks);
    assert_eq!(result.reasoning_text, "思考");
    assert_eq!(result.normal_text, "答案");
}

#[test]
fn test_prefilled_thinking_marked_splits_without_start_token() {
    // thinking_mode="enabled" prefills <mm:think> in the prompt, so the
    // output stream carries no start token; the pipeline marks the parser.
    let mut parser = MinimaxM3Parser::new();
    parser.mark_reasoning_started();
    let result = parser
        .detect_and_parse_reasoning("deep thoughts</mm:think>the answer")
        .unwrap();
    assert_eq!(result.reasoning_text, "deep thoughts");
    assert_eq!(result.normal_text, "the answer");

    let mut streaming = MinimaxM3Parser::new();
    streaming.mark_reasoning_started();
    let r1 = streaming
        .parse_reasoning_streaming_incremental("deep ")
        .unwrap();
    assert_eq!(r1.reasoning_text, "deep ");
    assert_eq!(r1.normal_text, "");
    let r2 = streaming
        .parse_reasoning_streaming_incremental("thoughts</mm:think>the answer")
        .unwrap();
    assert_eq!(r2.reasoning_text, "thoughts");
    assert_eq!(r2.normal_text, "the answer");
}

#[test]
fn test_prefilled_thinking_drops_duplicate_closers() {
    let mut parser = MinimaxM3Parser::new();
    parser.mark_reasoning_started();
    let result = collect_stream(
        &mut parser,
        &["<mm:think>deep</mm:", "think></mm:think>answer"],
    );
    assert_eq!(result.reasoning_text, "deep");
    assert_eq!(result.normal_text, "answer");
}

#[test]
fn test_reset_behavior() {
    let mut parser = MinimaxM3Parser::new();
    parser
        .parse_reasoning_streaming_incremental("<mm:think>partial reasoning")
        .unwrap();
    assert!(parser.is_in_reasoning());

    parser.reset();
    // After reset, state returns to the configured initial value (false).
    assert!(!parser.is_in_reasoning());

    let result = parser
        .detect_and_parse_reasoning("<mm:think>fresh</mm:think>done")
        .unwrap();
    assert_eq!(result.reasoning_text, "fresh");
    assert_eq!(result.normal_text, "done");
}

#[test]
fn test_buffer_overflow() {
    let mut parser = MinimaxM3Parser::new();
    let oversized = "x".repeat(DEFAULT_MAX_BUFFER_SIZE + 1);
    assert!(matches!(
        parser.detect_and_parse_reasoning(&oversized),
        Err(ParseError::BufferOverflow(_))
    ));
}
