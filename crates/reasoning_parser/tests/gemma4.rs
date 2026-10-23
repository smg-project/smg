use reasoning_parser::{parsers::Gemma4Parser, traits::ReasoningParser, ParserFactory};

#[test]
fn gemma4_channel_and_implicit_tool_end_at_every_chunk_boundary() {
    for (text, expected_content) in [
        ("<|channel>thought\n推理🙂<channel|>answer", "answer"),
        (
            "<|channel>thought\n推理🙂<|tool_call>call:weather{city:<|\"|>東京<|\"|>}<tool_call|>",
            "<|tool_call>call:weather{city:<|\"|>東京<|\"|>}<tool_call|>",
        ),
    ] {
        for boundary in (0..=text.len()).filter(|i| text.is_char_boundary(*i)) {
            let mut parser = Gemma4Parser::new();
            let mut content = String::new();
            let mut reasoning = String::new();
            for chunk in [&text[..boundary], &text[boundary..]] {
                let result = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
                content.push_str(&result.normal_text);
                reasoning.push_str(&result.reasoning_text);
            }
            let result = parser.flush().unwrap();
            content.push_str(&result.normal_text);
            reasoning.push_str(&result.reasoning_text);
            assert_eq!(reasoning, "推理🙂", "boundary {boundary}");
            assert_eq!(content, expected_content, "boundary {boundary}");
        }
    }
}

#[test]
fn gemma4_direct_answer_and_prompt_prefilled_continuation() {
    assert!(!Gemma4Parser::prompt_ends_in_reasoning("<|turn>model\n"));
    assert!(!Gemma4Parser::prompt_ends_in_reasoning(
        "<|turn>model\n<|channel>thought\n<channel|>"
    ));
    assert!(Gemma4Parser::prompt_ends_in_reasoning(
        "tool result<|channel>thought\n"
    ));
    let mut parser = Gemma4Parser::new();
    assert_eq!(
        parser
            .detect_and_parse_reasoning("answer")
            .unwrap()
            .normal_text,
        "answer"
    );
    parser.mark_reasoning_started();
    let result = parser
        .detect_and_parse_reasoning("analysis<channel|>answer")
        .unwrap();
    assert_eq!(result.reasoning_text, "analysis");
    assert_eq!(result.normal_text, "answer");
}

#[test]
fn gemma4_character_chunks_flush_and_reset() {
    let mut parser = Gemma4Parser::new();
    let mut reasoning = String::new();
    let mut content = String::new();
    for ch in "<|channel>thought\nwork<channel|>done".chars() {
        let result = parser
            .parse_reasoning_streaming_incremental(&ch.to_string())
            .unwrap();
        reasoning.push_str(&result.reasoning_text);
        content.push_str(&result.normal_text);
    }
    assert_eq!(reasoning, "work");
    assert_eq!(content, "done");
    parser.reset();
    parser.mark_reasoning_started();
    assert_eq!(
        parser
            .parse_reasoning_streaming_incremental("work<chan")
            .unwrap()
            .reasoning_text,
        "work"
    );
    assert_eq!(parser.flush().unwrap().reasoning_text, "<chan");
    assert!(parser.flush().unwrap().is_empty());
    parser.reset();
    assert!(!parser.is_in_reasoning());
    assert!(parser.requires_special_tokens());
    assert_eq!(
        ParserFactory::new()
            .create("google/gemma-4-12B-it")
            .model_type(),
        "gemma4"
    );
}

#[test]
fn gemma4_suppressed_opener_and_normal_thought_prefix() {
    let text = "thought\nanalysis<channel|>answer";
    for boundary in 0..=text.len() {
        let mut parser = Gemma4Parser::new();
        let mut reasoning = String::new();
        let mut content = String::new();
        for chunk in [&text[..boundary], &text[boundary..]] {
            let result = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
            reasoning.push_str(&result.reasoning_text);
            content.push_str(&result.normal_text);
        }
        let result = parser.flush().unwrap();
        reasoning.push_str(&result.reasoning_text);
        content.push_str(&result.normal_text);
        assert_eq!(reasoning, "analysis", "boundary {boundary}");
        assert_eq!(content, "answer");
    }
    let mut parser = Gemma4Parser::new();
    assert_eq!(
        parser
            .detect_and_parse_reasoning("thoughtful answer")
            .unwrap()
            .normal_text,
        "thoughtful answer"
    );
}

#[test]
fn gemma4_closing_channel_requires_prompt_prefill_for_reasoning() {
    let text = "analysis<channel|>answer";
    for prefilled in [false, true] {
        let expected_reasoning = if prefilled { "analysis" } else { "" };
        let expected_content = if prefilled {
            "answer"
        } else {
            "analysisanswer"
        };
        let mut parser = Gemma4Parser::new();
        if prefilled {
            parser.mark_reasoning_started();
        }
        let complete = parser.detect_and_parse_reasoning(text).unwrap();
        assert_eq!(complete.reasoning_text, expected_reasoning);
        assert_eq!(complete.normal_text, expected_content);
        for boundary in 0..=text.len() {
            let mut parser = Gemma4Parser::new();
            if prefilled {
                parser.mark_reasoning_started();
            }
            let mut content = String::new();
            let mut reasoning = String::new();
            for chunk in [&text[..boundary], &text[boundary..]] {
                let result = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
                content.push_str(&result.normal_text);
                reasoning.push_str(&result.reasoning_text);
            }
            let tail = parser.flush().unwrap();
            content.push_str(&tail.normal_text);
            reasoning.push_str(&tail.reasoning_text);
            assert_eq!(
                reasoning, expected_reasoning,
                "prefilled {prefilled}, boundary {boundary}"
            );
            assert_eq!(
                content, expected_content,
                "prefilled {prefilled}, boundary {boundary}"
            );
        }
    }
}

#[test]
fn gemma4_channel_tokens_inside_tool_strings_are_opaque() {
    let frame = "<|tool_call>call:lookup{q:<|\"|>literal <channel|> and <|channel>thought\n token<|\"|>}<tool_call|>";
    for boundary in 0..=frame.len() {
        let mut parser = Gemma4Parser::new();
        let mut reasoning = String::new();
        let mut content = String::new();
        for chunk in [&frame[..boundary], &frame[boundary..]] {
            let result = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
            reasoning.push_str(&result.reasoning_text);
            content.push_str(&result.normal_text);
        }
        let result = parser.flush().unwrap();
        reasoning.push_str(&result.reasoning_text);
        content.push_str(&result.normal_text);
        assert!(reasoning.is_empty(), "boundary {boundary}");
        assert_eq!(content, frame, "boundary {boundary}");
    }
    let input = frame.trim_end_matches("<tool_call|>");
    let result = Gemma4Parser::new()
        .detect_and_parse_reasoning(input)
        .unwrap();
    assert_eq!(result.normal_text, input);
    assert!(result.reasoning_text.is_empty());
}
