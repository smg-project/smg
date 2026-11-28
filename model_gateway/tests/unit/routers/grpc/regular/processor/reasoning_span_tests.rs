use super::reasoning_span_tokens;

#[test]
fn reasoning_span_tokens_skips_the_delimiters_and_keeps_held_tokens() {
    // Tokens: "<think>" | "a" | "" (text held back) | "b" | "</think>" | "c"
    let text = "<think>ab</think>c";
    assert_eq!(
        reasoning_span_tokens(&[7, 8, 8, 9, 17, 18], text, "ab", "c"),
        3
    );
    // Reasoning cut off before its end: every token is reasoning.
    assert_eq!(reasoning_span_tokens(&[7, 8, 9], "<think>ab", "ab", ""), 2);
    // The span text is not in the output verbatim: everything before the
    // normal text, delimiters included.
    assert_eq!(
        reasoning_span_tokens(&[7, 8, 9, 17, 18], text, "xy", "c"),
        4
    );
    // Text the stop decoder held back to the end ("ST" of a partial
    // "STOP") went to the last token at the flush: the held tokens before
    // it are zero-width inside the span, or at its start.
    assert_eq!(
        reasoning_span_tokens(&[7, 9, 9, 11], "<think>abST", "abST", ""),
        3
    );
    assert_eq!(reasoning_span_tokens(&[7, 7, 9], "<think>ST", "ST", ""), 2);
    assert_eq!(reasoning_span_tokens(&[0, 2], "ST", "ST", ""), 2);
}
