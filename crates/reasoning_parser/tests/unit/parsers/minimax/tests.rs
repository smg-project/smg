use super::*;

#[test]
fn test_minimax_always_in_reasoning() {
    let mut parser = MiniMaxParser::new();

    // always_in_reasoning=true: treats all text as reasoning until </think>
    let result = parser
        .detect_and_parse_reasoning("reasoning content</think>normal content")
        .unwrap();
    assert_eq!(result.normal_text, "normal content");
    assert_eq!(result.reasoning_text, "reasoning content");
}

#[test]
fn test_minimax_without_end_token() {
    let mut parser = MiniMaxParser::new();

    // Should treat all content as reasoning when no end token
    let result = parser
        .detect_and_parse_reasoning("all reasoning content")
        .unwrap();
    assert_eq!(result.normal_text, "");
    assert_eq!(result.reasoning_text, "all reasoning content");
}

#[test]
fn test_minimax_streaming() {
    let mut parser = MiniMaxParser::new();

    // First chunk — already in reasoning mode
    let result1 = parser
        .parse_reasoning_streaming_incremental("thinking about")
        .unwrap();
    assert_eq!(result1.reasoning_text, "thinking about");
    assert_eq!(result1.normal_text, "");

    // End of reasoning
    let result2 = parser
        .parse_reasoning_streaming_incremental(" the problem</think>answer")
        .unwrap();
    assert_eq!(result2.reasoning_text, " the problem");
    assert_eq!(result2.normal_text, "answer");
}

#[test]
fn test_model_type() {
    let parser = MiniMaxParser::new();
    assert_eq!(parser.model_type(), "minimax");
}
