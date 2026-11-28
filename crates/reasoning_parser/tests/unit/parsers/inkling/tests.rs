use super::*;

const TOOL_BLOCK: &str = concat!(
    "<|content_invoke_tool_json|>",
    r#"{"name":"search","args":{"q":"rust"}}"#,
    "<|end_message|>"
);
// Canonical typed-content fixture.
// Structured tool calls carry their tool name in the model-message header
// as well as in the JSON payload; only the payload reaches the tool parser.
const TYPED_OUTPUT: &str = concat!(
    "<|message_model|>",
    "<|content_thinking|>check sources<|end_message|>",
    "<|message_model|>",
    "<|content_text|>Here is the answer.<|end_message|>",
    "<|message_model|>search",
    "<|content_invoke_tool_json|>",
    r#"{"name":"search","args":{"q":"rust"}}"#,
    "<|end_message|>",
    "<|content_model_end_sampling|>"
);

#[test]
fn parses_typed_blocks_and_preserves_tool_framing() {
    let mut parser = InklingParser::new();
    let result = parser.detect_and_parse_reasoning(TYPED_OUTPUT).unwrap();

    assert_eq!(result.reasoning_text, "check sources");
    assert_eq!(
        result.normal_text,
        format!("Here is the answer.{TOOL_BLOCK}")
    );
    assert!(parser.requires_special_tokens());
}

#[test]
fn safely_suppresses_headerless_text_tool_frame() {
    let output = concat!(
        "<|content_invoke_tool_text|>search for rust<|end_message|>",
        "<|content_model_end_sampling|>"
    );
    let mut parser = InklingParser::new();
    let result = parser.detect_and_parse_reasoning(output).unwrap();

    assert_eq!(result.reasoning_text, "");
    assert_eq!(result.normal_text, "");
}

#[test]
fn streaming_buffers_control_tokens_at_every_chunk_boundary() {
    let expected_normal = format!("Here is the answer.{TOOL_BLOCK}");

    for split in TYPED_OUTPUT
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(TYPED_OUTPUT.len()))
    {
        let mut parser = InklingParser::new();
        let first = parser
            .parse_reasoning_streaming_incremental(&TYPED_OUTPUT[..split])
            .unwrap();
        let second = parser
            .parse_reasoning_streaming_incremental(&TYPED_OUTPUT[split..])
            .unwrap();

        assert_eq!(
            format!("{}{}", first.reasoning_text, second.reasoning_text),
            "check sources",
            "reasoning mismatch at split {split}"
        );
        assert_eq!(
            format!("{}{}", first.normal_text, second.normal_text),
            expected_normal,
            "content mismatch at split {split}"
        );
    }
}

#[test]
fn reset_clears_partial_token_and_reasoning_state() {
    let mut parser = InklingParser::new();
    parser
        .parse_reasoning_streaming_incremental("<|content_thinking|>work<|end_mes")
        .unwrap();
    assert!(parser.is_in_reasoning());

    parser.reset();
    let result = parser
        .parse_reasoning_streaming_incremental("plain answer")
        .unwrap();
    assert_eq!(result, ParserResult::normal("plain answer".to_string()));
}

#[test]
fn preserves_protocol_spellings_in_json_strings() {
    let payload = r#"{"literal":"<|content_thinking|>quoted<|end_message|>"}"#;
    let framed = format!("{MESSAGE_MODEL}{CONTENT_TEXT}{payload}{END_MESSAGE}");
    let mut parser = InklingParser::new();
    parser.set_json_mode(true);
    let result = parser.detect_and_parse_reasoning(&framed).unwrap();
    assert_eq!(result, ParserResult::normal(payload.to_string()));
}

fn parse_json_chunks(chunks: &[&str]) -> ParserResult {
    let mut parser = InklingParser::new();
    parser.set_json_mode(true);
    let mut result = ParserResult::default();
    for chunk in chunks {
        let parsed = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
        result.normal_text.push_str(&parsed.normal_text);
        result.reasoning_text.push_str(&parsed.reasoning_text);
    }
    let tail = parser.flush().unwrap();
    result.normal_text.push_str(&tail.normal_text);
    result.reasoning_text.push_str(&tail.reasoning_text);
    assert!(parser.flush().unwrap().is_empty());
    result
}

#[test]
fn json_keys_escapes_and_unicode_survive_every_partition() {
    for token in CONTROL_TOKENS {
        let payload = format!(r#"{{"{token}":"α\"{token}\\\\終"}}"#);
        let framed = format!(
            "{MESSAGE_MODEL}{CONTENT_THINKING}unmatched quote: \"{END_MESSAGE}\
                 {MESSAGE_MODEL}{CONTENT_TEXT}{payload}{END_MESSAGE}{CONTENT_MODEL_END_SAMPLING}"
        );
        let expected = ParserResult::new(payload, "unmatched quote: \"".to_string());
        for split in framed
            .char_indices()
            .map(|(index, _)| index)
            .chain([framed.len()])
        {
            assert_eq!(
                parse_json_chunks(&[&framed[..split], &framed[split..]]),
                expected,
                "token {token}, split {split}"
            );
        }
        let chunks: Vec<_> = framed
            .char_indices()
            .map(|(index, character)| &framed[index..index + character.len_utf8()])
            .collect();
        assert_eq!(parse_json_chunks(&chunks), expected);
    }
}

#[test]
fn json_continuations_scalars_and_incomplete_strings_preserve_text() {
    for payload in [
        r#"["<|message_model|>","<|end_message|>"]"#,
        r#""<|content_thinking|>""#,
        "42",
        "null",
        "true",
        r#"{"value":"<|content_thinking|>unfinished<|end_mes"#,
        r#"{"value":"escaped quote: \"<|end_message|>"#,
    ] {
        assert_eq!(
            parse_json_chunks(&[payload]),
            ParserResult::normal(payload.to_string())
        );
    }
}

#[test]
fn json_reset_and_complete_parse_do_not_reuse_quote_state() {
    let mut parser = InklingParser::new();
    parser.set_json_mode(true);
    parser
        .parse_reasoning_streaming_incremental("{\"value\":\"\\")
        .unwrap();
    let framed = format!("{CONTENT_TEXT}\"{CONTENT_THINKING}\"{END_MESSAGE}");
    let expected = ParserResult::normal(format!("\"{CONTENT_THINKING}\""));
    assert_eq!(
        parser.detect_and_parse_reasoning(&framed).unwrap(),
        expected
    );
    parser.reset();
    assert_eq!(
        parser
            .parse_reasoning_streaming_incremental(&framed)
            .unwrap(),
        expected
    );
    parser.reset();
    parser.set_json_mode(false);
    let parsed = parser.detect_and_parse_reasoning(&framed).unwrap();
    assert_eq!(parsed.reasoning_text, "\"");
    assert_eq!(parsed.normal_text, "\"");
}

#[test]
fn json_mode_preserves_existing_tool_block_behavior() {
    assert_eq!(
        parse_json_chunks(&[TYPED_OUTPUT]),
        ParserResult::new(
            format!("Here is the answer.{TOOL_BLOCK}"),
            "check sources".to_string()
        )
    );
}
