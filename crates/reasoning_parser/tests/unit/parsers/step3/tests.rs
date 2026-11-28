use super::*;

#[test]
fn test_step3_initial_state() {
    let mut parser = Step3Parser::new();

    // Should treat text as reasoning even without start token
    let result = parser
        .detect_and_parse_reasoning("This is reasoning content")
        .unwrap();
    assert_eq!(result.normal_text, "");
    assert_eq!(result.reasoning_text, "This is reasoning content");
}

#[test]
fn test_step3_with_end_token() {
    let mut parser = Step3Parser::new();

    // Should handle text with end token
    let result = parser
        .detect_and_parse_reasoning("reasoning content</think>answer")
        .unwrap();
    assert_eq!(result.normal_text, "answer");
    assert_eq!(result.reasoning_text, "reasoning content");
}

#[test]
fn test_step3_with_both_tokens() {
    let mut parser = Step3Parser::new();

    // Should handle both start and end tokens
    let result = parser
        .detect_and_parse_reasoning("<think>reasoning content</think>answer")
        .unwrap();
    assert_eq!(result.normal_text, "answer");
    assert_eq!(result.reasoning_text, "reasoning content");
}

#[test]
fn test_step3_streaming() {
    let mut parser = Step3Parser::new();

    // First chunk - treated as reasoning (always_in_reasoning=true)
    let result1 = parser
        .parse_reasoning_streaming_incremental("reasoning text ")
        .unwrap();
    assert_eq!(result1.normal_text, "");
    assert_eq!(result1.reasoning_text, "reasoning text ");

    // Second chunk - continues reasoning until end token
    let result2 = parser
        .parse_reasoning_streaming_incremental("more reasoning</think>answer")
        .unwrap();
    assert_eq!(result2.normal_text, "answer");
    assert_eq!(result2.reasoning_text, "more reasoning");
}

#[test]
fn test_model_type() {
    let parser = Step3Parser::new();
    assert_eq!(parser.model_type(), "step3");
}
