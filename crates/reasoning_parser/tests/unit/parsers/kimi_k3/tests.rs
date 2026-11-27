use super::*;

const OPEN: &str = "<|open|>";
const CLOSE: &str = "<|close|>";
const SEP: &str = "<|sep|>";
fn think_open() -> String {
    format!("{OPEN}think{SEP}")
}
fn think_close() -> String {
    format!("{CLOSE}think{SEP}")
}
fn response_open() -> String {
    format!("{OPEN}response{SEP}")
}
fn response_close() -> String {
    format!("{CLOSE}response{SEP}")
}
fn message_open() -> String {
    format!("{OPEN}message{SEP}")
}
fn message_close() -> String {
    format!("{CLOSE}message{SEP}")
}

// -------------------------------------------------------------------------
// Non-streaming tests
// -------------------------------------------------------------------------

#[test]
fn model_type_and_special_tokens() {
    let p = KimiK3Parser::new();
    assert_eq!(p.model_type(), "kimi_k3");
    assert!(p.requires_special_tokens());
}

#[test]
fn extract_with_xtml_tags() {
    let mut p = KimiK3Parser::new();
    let input = format!(
        "{}step{}{}answer",
        think_open(),
        think_close(),
        response_open()
    );
    let r = p.detect_and_parse_reasoning(&input).unwrap();
    assert_eq!(r.reasoning_text, "step");
    assert_eq!(r.normal_text, "answer");
}

#[test]
fn extract_with_generation_prefix_consumed() {
    // open think marker absent (was the prompt prefix)
    let mut p = KimiK3Parser::new();
    p.mark_reasoning_started();
    let input = format!("step{}{}answer", think_close(), response_open());
    let r = p.detect_and_parse_reasoning(&input).unwrap();
    assert_eq!(r.reasoning_text, "step");
    assert_eq!(r.normal_text, "answer");
}

#[test]
fn strips_response_wrapper_with_close_and_keeps_tools_channel() {
    let mut p = KimiK3Parser::new();
    let input = format!(
        "{}step{}{}answer{}{}tools{}CALLS{}tools{}",
        think_open(),
        think_close(),
        response_open(),
        response_close(),
        OPEN,
        SEP,
        CLOSE,
        SEP
    );
    let r = p.detect_and_parse_reasoning(&input).unwrap();
    assert_eq!(r.reasoning_text, "step");
    // response wrapper removed; tools channel preserved for the tool parser
    assert_eq!(
        r.normal_text,
        format!("answer{OPEN}tools{SEP}CALLS{CLOSE}tools{SEP}")
    );
}

#[test]
fn strips_message_wrapper_like_response_wrapper() {
    // The `message` wrapper is stripped symmetrically (open + close), just
    // like the `response` wrapper, leaving only the answer body.
    let mut p = KimiK3Parser::new();
    let input = format!(
        "{}r{}{}answer{}",
        think_open(),
        think_close(),
        message_open(),
        message_close()
    );
    let r = p.detect_and_parse_reasoning(&input).unwrap();
    assert_eq!(r.reasoning_text, "r");
    assert_eq!(r.normal_text, "answer");
}

#[test]
fn in_reasoning_forced_tools_without_think_close_is_not_swallowed() {
    // Regression: under tool_choice="required" the model can jump from the
    // prefilled think channel straight into tools with no think-close. The
    // tools channel must reach the tool parser, not land in reasoning.
    let mut p = KimiK3Parser::new();
    p.mark_reasoning_started();
    let input = format!(
        "{OPEN}tools{SEP}{OPEN}call tool=\"get_weather\" index=\"1\"{SEP}{CLOSE}call{SEP}{CLOSE}tools{SEP}{CLOSE}message{SEP}"
    );
    let r = p.detect_and_parse_reasoning(&input).unwrap();
    assert_eq!(r.reasoning_text, "");
    // tools channel preserved for the tool parser; message wrapper stripped.
    assert_eq!(
        r.normal_text,
        format!(
            "{OPEN}tools{SEP}{OPEN}call tool=\"get_weather\" index=\"1\"{SEP}{CLOSE}call{SEP}{CLOSE}tools{SEP}"
        )
    );
    assert!(!p.is_in_reasoning());
}

#[test]
fn in_reasoning_reasoning_then_tools_without_close_splits_at_tools_opener() {
    // Reasoning text followed directly by tools, still with no think-close.
    let mut p = KimiK3Parser::new();
    p.mark_reasoning_started();
    let input = format!("let me check{OPEN}tools{SEP}CALLS{CLOSE}tools{SEP}");
    let r = p.detect_and_parse_reasoning(&input).unwrap();
    assert_eq!(r.reasoning_text, "let me check");
    assert_eq!(
        r.normal_text,
        format!("{OPEN}tools{SEP}CALLS{CLOSE}tools{SEP}")
    );
    assert!(!p.is_in_reasoning());
}

#[test]
fn in_reasoning_response_opener_without_close_ends_reasoning() {
    // The `response` opener ends reasoning just like the `tools` opener,
    // and its wrapper is then stripped from content.
    let mut p = KimiK3Parser::new();
    p.mark_reasoning_started();
    let input = format!("thought{OPEN}response{SEP}answer{CLOSE}response{SEP}");
    let r = p.detect_and_parse_reasoning(&input).unwrap();
    assert_eq!(r.reasoning_text, "thought");
    assert_eq!(r.normal_text, "answer");
}

#[test]
fn in_reasoning_answer_closed_as_response_is_content_not_reasoning() {
    // Regression (16/700 BEAM answers): reasoning pre-marked for a prefilled
    // `<|open|>think<|sep|>`, then the model answers straight into it and
    // closes as `response`, leaving sibling closers and no opener at all.
    let mut p = KimiK3Parser::new();
    p.mark_reasoning_started();
    let input = format!("The answer is 42.{CLOSE}response{SEP}{CLOSE}message{SEP}");
    let r = p.detect_and_parse_reasoning(&input).unwrap();
    assert_eq!(r.normal_text, "The answer is 42.");
    assert_eq!(r.reasoning_text, "");
    assert!(!p.is_in_reasoning());
}

#[test]
fn observed_think_channel_ignores_a_stray_sibling_closer() {
    // With a real `<|open|>think<|sep|>` present the closer is malformed
    // output, not proof the deliberation was the answer: reasoning survives.
    let mut p = KimiK3Parser::new();
    let input = format!("{}deliberating{CLOSE}response{SEP}", think_open());
    let r = p.detect_and_parse_reasoning(&input).unwrap();
    assert_eq!(
        r.reasoning_text,
        format!("deliberating{CLOSE}response{SEP}")
    );
    assert_eq!(r.normal_text, "");
    assert!(p.is_in_reasoning());
}

#[test]
fn think_close_wins_over_a_later_sibling_closer() {
    // Terminators are ranked by marker position; the trailing closers are
    // just wrapper to strip. The ordering the new branch must not disturb.
    let mut p = KimiK3Parser::new();
    p.mark_reasoning_started();
    let input = format!(
        "step{}{}answer{}{}",
        think_close(),
        response_open(),
        response_close(),
        message_close()
    );
    let r = p.detect_and_parse_reasoning(&input).unwrap();
    assert_eq!(r.reasoning_text, "step");
    assert_eq!(r.normal_text, "answer");
}

#[test]
fn no_think_channel_is_all_content() {
    let mut p = KimiK3Parser::new();
    let r = p.detect_and_parse_reasoning("just an answer").unwrap();
    assert_eq!(r.reasoning_text, "");
    assert_eq!(r.normal_text, "just an answer");
}

#[test]
fn marker_free_text_while_in_reasoning_is_all_reasoning() {
    // Once the caller has told us we're already in the think channel
    // (e.g. the generation prompt pre-filled `<|open|>think<|sep|>`),
    // truncated output with no markers at all must come back as
    // reasoning in full, not be misclassified as content.
    let mut p = KimiK3Parser::new();
    p.mark_reasoning_started();
    let text = "still thinking, no markers";
    let r = p.detect_and_parse_reasoning(text).unwrap();
    assert_eq!(r.reasoning_text, text);
    assert_eq!(r.normal_text, "");
}

#[test]
fn detect_and_parse_reasoning_buffer_overflow_is_guarded() {
    let mut p = KimiK3Parser::new();
    let oversized = "a".repeat(DEFAULT_MAX_BUFFER_SIZE + 1);
    let result = p.detect_and_parse_reasoning(&oversized);
    assert!(matches!(
        result,
        Err(ParseError::BufferOverflow(size)) if size == DEFAULT_MAX_BUFFER_SIZE + 1
    ));
}

// -------------------------------------------------------------------------
// Streaming tests
// -------------------------------------------------------------------------

#[test]
fn streaming_thinking_disabled_routes_tools_to_normal_content() {
    let mut p = KimiK3Parser::new();
    let chunks = [
        "<|close|>response",
        "<|sep|>",
        "<|open|>tools",
        "<|sep|>",
        r#"<|open|>call tool="get_weather" index="1""#,
        "<|sep|>",
        "<|close|>call",
        "<|sep|>",
        "<|close|>tools",
        "<|sep|>",
    ];

    let mut reasoning = String::new();
    let mut normal = String::new();
    for chunk in chunks {
        let result = p.parse_reasoning_streaming_incremental(chunk).unwrap();
        reasoning.push_str(&result.reasoning_text);
        normal.push_str(&result.normal_text);
    }

    assert_eq!(reasoning, "");
    assert_eq!(
        normal,
        concat!(
            "<|open|>tools<|sep|>",
            r#"<|open|>call tool="get_weather" index="1"<|sep|>"#,
            "<|close|>call<|sep|><|close|>tools<|sep|>"
        )
    );
    assert!(!p.is_in_reasoning());
}

#[test]
fn streaming_answer_closed_as_response_recovers_content() {
    // Streaming counterpart of the regression above, with the closer split
    // across chunks so the widened holdback set is exercised too: a leaked
    // `res` would show up in `reasoning`.
    //
    // Recovery is partial by construction — the answer was already streamed
    // as reasoning deltas before the closer arrived, so it appears in both
    // fields. What matters is a complete `normal_text` and a cleared
    // `in_reasoning`, which gates the tool parser.
    let mut p = KimiK3Parser::new();
    p.mark_reasoning_started();
    let chunks = [
        "The answer ",
        "is 42.",
        "<|close|>res",
        "ponse<|sep|>",
        "<|close|>message<|sep|>",
    ];

    let mut reasoning = String::new();
    let mut normal = String::new();
    for chunk in chunks {
        let result = p.parse_reasoning_streaming_incremental(chunk).unwrap();
        reasoning.push_str(&result.reasoning_text);
        normal.push_str(&result.normal_text);
    }

    assert_eq!(normal, "The answer is 42.");
    assert_eq!(reasoning, "The answer is 42.");
    assert!(!p.is_in_reasoning());
}

#[test]
fn streaming_split_open_marker_held_back() {
    let mut p = KimiK3Parser::new();
    assert_eq!(
        p.parse_reasoning_streaming_incremental("<|open|>")
            .unwrap()
            .reasoning_text,
        ""
    );
    assert_eq!(
        p.parse_reasoning_streaming_incremental("think")
            .unwrap()
            .reasoning_text,
        ""
    );
    let r = p
        .parse_reasoning_streaming_incremental("<|sep|>step")
        .unwrap();
    assert_eq!(r.reasoning_text, "step");
}

#[test]
fn streaming_split_close_hands_content_downstream() {
    let mut p = KimiK3Parser::new();
    p.parse_reasoning_streaming_incremental("<|open|>think<|sep|>step")
        .unwrap();
    let partial = p
        .parse_reasoning_streaming_incremental("<|close|>")
        .unwrap();
    assert_eq!(partial.reasoning_text, ""); // partial close held back
    assert_eq!(partial.normal_text, "");
    let closed = p
        .parse_reasoning_streaming_incremental("think<|sep|><|open|>response<|sep|>answer")
        .unwrap();
    assert_eq!(closed.reasoning_text, "");
    assert_eq!(closed.normal_text, "answer"); // response-open stripped
}

#[test]
fn streaming_split_message_wrapper_markers_do_not_leak_into_content() {
    // Analogous to `streaming_split_close_hands_content_downstream`, but
    // the marker split during the post-reasoning content phase is the
    // `message` wrapper rather than the think-close marker.
    let mut p = KimiK3Parser::new();
    // Reach the content phase in one shot.
    p.parse_reasoning_streaming_incremental("<|open|>think<|sep|>step<|close|>think<|sep|>")
        .unwrap();

    // Split message-open marker across chunks.
    let partial_open = p
        .parse_reasoning_streaming_incremental("<|open|>mess")
        .unwrap();
    assert_eq!(partial_open.normal_text, ""); // partial open marker held back
    let after_open = p
        .parse_reasoning_streaming_incremental("age<|sep|>answer")
        .unwrap();
    assert_eq!(after_open.normal_text, "answer"); // message-open stripped, no leak

    // Split message-close marker across chunks.
    let partial_close = p
        .parse_reasoning_streaming_incremental(" more<|clo")
        .unwrap();
    assert_eq!(partial_close.normal_text, " more"); // partial close marker held back
    let after_close = p
        .parse_reasoning_streaming_incremental("se|>message<|sep|> tail")
        .unwrap();
    assert_eq!(after_close.normal_text, " tail"); // message-close stripped, no leak
}

#[test]
fn streaming_maintains_in_reasoning_flag() {
    let mut p = KimiK3Parser::new();
    // While accumulating reasoning content, the parser reports in-reasoning.
    p.parse_reasoning_streaming_incremental("<|open|>think<|sep|>step")
        .unwrap();
    assert!(p.is_in_reasoning());
    // A partial close marker is held back but we are still in reasoning.
    p.parse_reasoning_streaming_incremental("<|close|>")
        .unwrap();
    assert!(p.is_in_reasoning());
    // Completing the think-close transition clears the flag.
    p.parse_reasoning_streaming_incremental("think<|sep|><|open|>response<|sep|>answer")
        .unwrap();
    assert!(!p.is_in_reasoning());
}

#[test]
fn streaming_in_reasoning_transitions_on_tools_opener_without_think_close() {
    // Streaming counterpart of the forced-tools regression, with the
    // markers split across chunks.
    let mut p = KimiK3Parser::new();
    p.mark_reasoning_started();
    let chunks = [
        "<|open|>tools",
        "<|sep|>",
        r#"<|open|>call tool="get_weather" index="1""#,
        "<|sep|>",
        "<|close|>call",
        "<|sep|>",
        "<|close|>tools",
        "<|sep|>",
        "<|close|>message<|sep|>",
    ];

    let mut reasoning = String::new();
    let mut normal = String::new();
    for chunk in chunks {
        let r = p.parse_reasoning_streaming_incremental(chunk).unwrap();
        reasoning.push_str(&r.reasoning_text);
        normal.push_str(&r.normal_text);
    }

    assert_eq!(reasoning, "");
    assert_eq!(
        normal,
        concat!(
            "<|open|>tools<|sep|>",
            r#"<|open|>call tool="get_weather" index="1"<|sep|>"#,
            "<|close|>call<|sep|><|close|>tools<|sep|>"
        )
    );
    assert!(!p.is_in_reasoning());
}

#[test]
fn streaming_buffer_overflow_is_guarded() {
    let mut p = KimiK3Parser::new();
    let oversized = "a".repeat(DEFAULT_MAX_BUFFER_SIZE + 1);
    let result = p.parse_reasoning_streaming_incremental(&oversized);
    assert!(matches!(
        result,
        Err(ParseError::BufferOverflow(size)) if size == DEFAULT_MAX_BUFFER_SIZE + 1
    ));
}

#[test]
fn reset_clears_state() {
    let mut p = KimiK3Parser::new();
    // Put the parser into reasoning mode via streaming.
    p.parse_reasoning_streaming_incremental("<|open|>think<|sep|>step")
        .unwrap();
    assert!(p.is_in_reasoning() || !p.buffer.is_empty());
    p.reset();
    assert!(!p.is_in_reasoning());
    assert!(p.buffer.is_empty());
    // After reset, a fresh non-streaming parse should work correctly.
    let r = p.detect_and_parse_reasoning("just content").unwrap();
    assert_eq!(r.normal_text, "just content");
    assert_eq!(r.reasoning_text, "");
}

// -------------------------------------------------------------------------
// A tag the end of the output cut short (smg-lab #115)
// -------------------------------------------------------------------------

/// The turn's closing structure after an answer, one marker a token.
const CLOSING: [&str; 6] = [
    "<|close|>",
    "response",
    "<|sep|>",
    "<|close|>",
    "message",
    "<|sep|>",
];

/// Every chunk streamed, then the flush: what a client assembles from the deltas.
fn streamed(chunks: &[&str], prefilled: bool) -> ParserResult {
    let mut p = KimiK3Parser::new();
    if prefilled {
        p.mark_reasoning_started();
    }
    let mut got = ParserResult::default();
    for chunk in chunks {
        let r = p.parse_reasoning_streaming_incremental(chunk).unwrap();
        got.normal_text.push_str(&r.normal_text);
        got.reasoning_text.push_str(&r.reasoning_text);
    }
    let tail = p.flush().unwrap();
    got.normal_text.push_str(&tail.normal_text);
    got.reasoning_text.push_str(&tail.reasoning_text);
    got
}

/// The whole text at once: the non-streaming path.
fn parsed(text: &str, prefilled: bool) -> ParserResult {
    let mut p = KimiK3Parser::new();
    if prefilled {
        p.mark_reasoning_started();
    }
    p.detect_and_parse_reasoning(text).unwrap()
}

#[test]
fn a_cut_inside_the_closing_structure_leaks_no_marker_into_content() {
    // Thinking off: the prompt opened the response channel, the model answers and closes
    // the turn. Cut by `max_tokens` after 8 of its 10 tokens, the client received
    // `Pong.<|close|>message` (smg-lab #115); after 6, or at the natural end, `Pong.`.
    // Every cut reads `Pong.`, streamed and whole.
    let mut tokens = vec!["P", "ong", "."];
    tokens.extend(CLOSING);
    let expected = ParserResult::normal("Pong.".to_owned());
    for cut in 3..=tokens.len() {
        assert_eq!(
            streamed(&tokens[..cut], false),
            expected,
            "streamed, cut after {cut} tokens"
        );
        assert_eq!(
            parsed(&tokens[..cut].concat(), false),
            expected,
            "whole, cut after {cut} tokens"
        );
    }
}

#[test]
fn every_byte_cut_of_the_closing_structure_leaks_no_marker() {
    // A cut at every byte of the closing structure, fed as one chunk, byte by byte, and as
    // the whole text. From a structural token on, the tail is a tag cut short and goes;
    // inside a structural token, which no stream splits (it is one token of the
    // vocabulary), the bytes so far stay text, as every parser keeps a partial marker at
    // the end of the stream, and never amount to a marker.
    let closing: String = CLOSING.concat();
    for cut in 1..=closing.len() {
        let text = format!("Pong.{}", &closing[..cut]);
        let bytes: Vec<&str> = (0..text.len()).map(|at| &text[at..=at]).collect();
        for (how, got) in [
            ("one chunk", streamed(&[&text], false)),
            ("byte by byte", streamed(&bytes, false)),
            ("non-streaming", parsed(&text, false)),
        ] {
            assert_eq!(got.reasoning_text, "", "{how}, cut at byte {cut}");
            let rest = got
                .normal_text
                .strip_prefix("Pong.")
                .unwrap_or_else(|| panic!("{how}, cut at byte {cut}: {:?}", got.normal_text));
            let inside_a_token = rest.len() < CLOSE.len() && CLOSE.starts_with(rest);
            assert!(
                inside_a_token,
                "{how}, cut at byte {cut}: {rest:?} reached the client"
            );
        }
    }
}

#[test]
fn a_cut_after_a_thought_leaks_into_neither_field() {
    // Thinking on: the prompt opened the thought (the parser is told so), the model closes
    // it, answers in the response channel and closes the turn; the cut falls after any
    // token. The thought is the thought and the answer is the answer, nothing more.
    let tokens = [
        "Thought",
        "<|close|>",
        "think",
        "<|sep|>",
        "<|open|>",
        "response",
        "<|sep|>",
        "P",
        "ong",
        ".",
        "<|close|>",
        "response",
        "<|sep|>",
        "<|close|>",
        "message",
        "<|sep|>",
    ];
    for cut in 1..=tokens.len() {
        let content = if cut > 7 {
            tokens[7..cut.min(10)].concat()
        } else {
            String::new()
        };
        let expected = ParserResult::new(content, "Thought".to_owned());
        assert_eq!(
            streamed(&tokens[..cut], true),
            expected,
            "streamed, cut after {cut} tokens"
        );
        assert_eq!(
            parsed(&tokens[..cut].concat(), true),
            expected,
            "whole, cut after {cut} tokens"
        );
    }
}

#[test]
fn only_a_tag_begun_by_a_whole_structural_token_is_cut() {
    // What goes at the end of the output: a structural token with nothing, or part of a
    // tag's word, after it. What stays: a tail shorter than a token, which may be text, and
    // a token followed by text no tag begins with, which is the model's to explain.
    for tail in [
        "<|close|>",
        "<|close|>message",
        "<|close|>mes",
        "<|close|>response",
        "<|open|>",
        "<|open|>respo",
    ] {
        let text = format!("Pong.{tail}");
        let expected = ParserResult::normal("Pong.".to_owned());
        assert_eq!(parsed(&text, false), expected, "{tail:?}");
        assert_eq!(streamed(&[&text], false), expected, "{tail:?}");
    }
    // The `tools` opener cut short goes from a whole text too; streamed, the content phase
    // hands the tools channel downstream as it arrives, so that opener is the tool parser's.
    assert_eq!(
        parsed("Pong.<|open|>tools", false),
        ParserResult::normal("Pong.".to_owned())
    );
    for tail in ["<", "<|clo", "<|close|>x"] {
        let text = format!("Pong.{tail}");
        let expected = ParserResult::normal(text.clone());
        assert_eq!(parsed(&text, false), expected, "{tail:?}");
        assert_eq!(streamed(&[&text], false), expected, "{tail:?}");
    }
}
