use super::*;

#[test]
fn non_streaming_forwards_input_verbatim() {
    let mut parser = PassthroughParser::new();
    let result = parser
        .detect_and_parse_reasoning("  <think>cot</think>answer  \n")
        .unwrap();
    assert_eq!(result.normal_text, "  <think>cot</think>answer  \n");
    assert_eq!(result.reasoning_text, "");
}

#[test]
fn streaming_forwards_chunks_verbatim() {
    let mut parser = PassthroughParser::new();
    let r1 = parser
        .parse_reasoning_streaming_incremental("  <think>cot")
        .unwrap();
    assert_eq!(r1.normal_text, "  <think>cot");
    assert_eq!(r1.reasoning_text, "");
    let r2 = parser
        .parse_reasoning_streaming_incremental("</think>answer  \n")
        .unwrap();
    assert_eq!(r2.normal_text, "</think>answer  \n");
    assert_eq!(r2.reasoning_text, "");
}
