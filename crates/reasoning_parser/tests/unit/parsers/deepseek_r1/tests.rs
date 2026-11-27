use super::*;

#[test]
fn test_deepseek_r1_initial_state() {
    let mut parser = DeepSeekR1Parser::new();

    // Should treat text as reasoning even without start token
    let result = parser
        .detect_and_parse_reasoning("This is reasoning content")
        .unwrap();
    assert_eq!(result.normal_text, "");
    assert_eq!(result.reasoning_text, "This is reasoning content");
}

#[test]
fn test_deepseek_r1_with_end_token() {
    let mut parser = DeepSeekR1Parser::new();

    // Should extract reasoning until end token
    let result = parser
        .detect_and_parse_reasoning("reasoning content</think>normal content")
        .unwrap();
    assert_eq!(result.normal_text, "normal content");
    assert_eq!(result.reasoning_text, "reasoning content");
}

#[test]
fn test_deepseek_r1_streaming() {
    let mut parser = DeepSeekR1Parser::new();

    // First chunk - all reasoning
    let result1 = parser
        .parse_reasoning_streaming_incremental("thinking about")
        .unwrap();
    assert_eq!(result1.reasoning_text, "thinking about");
    assert_eq!(result1.normal_text, "");

    // Second chunk - ends reasoning
    let result2 = parser
        .parse_reasoning_streaming_incremental(" the problem</think>answer")
        .unwrap();
    assert_eq!(result2.reasoning_text, " the problem"); // Whitespace preserved
    assert_eq!(result2.normal_text, "answer");
}

#[test]
fn test_model_type() {
    let parser = DeepSeekR1Parser::new();
    assert_eq!(parser.model_type(), "deepseek_r1");
}
