use super::*;

#[test]
fn test_nano_v3_initial_state() {
    let mut parser = NanoV3Parser::new();

    // Without mark_reasoning_started(), text without <think> is normal content
    let result = parser
        .detect_and_parse_reasoning("This is normal content")
        .unwrap();
    assert_eq!(result.normal_text, "This is normal content");
    assert_eq!(result.reasoning_text, "");
}

#[test]
fn test_nano_v3_with_mark_reasoning_started() {
    let mut parser = NanoV3Parser::new();
    parser.mark_reasoning_started();

    // After mark_reasoning_started(), text is treated as reasoning
    let result = parser
        .detect_and_parse_reasoning("reasoning content</think>answer")
        .unwrap();
    assert_eq!(result.normal_text, "answer");
    assert_eq!(result.reasoning_text, "reasoning content");
}

#[test]
fn test_nano_v3_with_both_tokens() {
    let mut parser = NanoV3Parser::new();

    let result = parser
        .detect_and_parse_reasoning("<think>reasoning content</think>answer")
        .unwrap();
    assert_eq!(result.normal_text, "answer");
    assert_eq!(result.reasoning_text, "reasoning content");
}

#[test]
fn test_nano_v3_streaming() {
    let mut parser = NanoV3Parser::new();
    parser.mark_reasoning_started();

    let result1 = parser
        .parse_reasoning_streaming_incremental("reasoning text ")
        .unwrap();
    assert_eq!(result1.normal_text, "");
    assert_eq!(result1.reasoning_text, "reasoning text ");

    let result2 = parser
        .parse_reasoning_streaming_incremental("more reasoning</think>answer")
        .unwrap();
    assert_eq!(result2.normal_text, "answer");
    assert_eq!(result2.reasoning_text, "more reasoning");
}

#[test]
fn test_model_type() {
    let parser = NanoV3Parser::new();
    assert_eq!(parser.model_type(), "nano_v3");
}
