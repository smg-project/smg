use super::*;

#[test]
fn test_qwen3_initial_state() {
    let mut parser = Qwen3Parser::new();

    // Should NOT treat text as reasoning without start token
    let result = parser
        .detect_and_parse_reasoning("This is normal content")
        .unwrap();
    assert_eq!(result.normal_text, "This is normal content");
    assert_eq!(result.reasoning_text, "");
}

#[test]
fn test_qwen3_with_tokens() {
    let mut parser = Qwen3Parser::new();

    // Should extract reasoning with proper tokens
    let result = parser
        .detect_and_parse_reasoning("<think>reasoning</think>answer")
        .unwrap();
    assert_eq!(result.normal_text, "answer");
    assert_eq!(result.reasoning_text, "reasoning");
}

#[test]
fn test_qwen_thinking_initial_state() {
    let mut parser = QwenThinkingParser::new();

    // Should treat text as reasoning even without start token
    let result = parser
        .detect_and_parse_reasoning("This is reasoning content")
        .unwrap();
    assert_eq!(result.normal_text, "");
    assert_eq!(result.reasoning_text, "This is reasoning content");
}

#[test]
fn test_qwen3_streaming() {
    let mut parser = Qwen3Parser::new();

    // First chunk - normal text (no start token yet)
    let result1 = parser
        .parse_reasoning_streaming_incremental("normal text ")
        .unwrap();
    assert_eq!(result1.normal_text, "normal text ");
    assert_eq!(result1.reasoning_text, "");

    // Second chunk - enters reasoning
    let result2 = parser
        .parse_reasoning_streaming_incremental("<think>reasoning")
        .unwrap();
    assert_eq!(result2.normal_text, "");
    assert_eq!(result2.reasoning_text, "reasoning");
}

#[test]
fn test_model_types() {
    let qwen3 = Qwen3Parser::new();
    assert_eq!(qwen3.model_type(), "qwen3");

    let qwen_thinking = QwenThinkingParser::new();
    assert_eq!(qwen_thinking.model_type(), "qwen_thinking");
}
