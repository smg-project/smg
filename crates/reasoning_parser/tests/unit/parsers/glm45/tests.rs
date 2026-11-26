use super::*;

#[test]
fn test_glm45_initial_state() {
    let mut parser = Glm45Parser::new();

    // Should NOT treat text as reasoning without start token
    let result = parser
        .detect_and_parse_reasoning("This is normal content")
        .unwrap();
    assert_eq!(result.normal_text, "This is normal content");
    assert_eq!(result.reasoning_text, "");
}

#[test]
fn test_glm45_with_tokens() {
    let mut parser = Glm45Parser::new();

    // Should extract reasoning with proper tokens
    let result = parser
        .detect_and_parse_reasoning("<think>reasoning content</think>answer")
        .unwrap();
    assert_eq!(result.normal_text, "answer");
    assert_eq!(result.reasoning_text, "reasoning content");
}

#[test]
fn test_glm45_streaming() {
    let mut parser = Glm45Parser::new();

    // First chunk - normal text
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

    // Third chunk - exits reasoning
    let result3 = parser
        .parse_reasoning_streaming_incremental("</think>answer")
        .unwrap();
    assert_eq!(result3.normal_text, "answer");
    assert_eq!(result3.reasoning_text, "");
}

#[test]
fn test_model_type() {
    let parser = Glm45Parser::new();
    assert_eq!(parser.model_type(), "glm45");
}
