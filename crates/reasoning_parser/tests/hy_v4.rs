use reasoning_parser::{
    parsers::HyV4Parser,
    traits::{PromptReasoning, ReasoningParser},
};

#[test]
fn hy4_every_chunk_boundary_and_prompt_prefill() {
    for suffix in ["", ":6124c78e", ":another-checkpoint"] {
        for explicit in [true, false] {
            let text = format!(
                "{}推理🙂</think{suffix}>answer",
                if explicit {
                    format!("<think{suffix}>")
                } else {
                    String::new()
                }
            );
            for boundary in (0..=text.len()).filter(|i| text.is_char_boundary(*i)) {
                let mut p = HyV4Parser::new();
                p.mark_reasoning_started();
                let mut reasoning = String::new();
                let mut content = String::new();
                for chunk in [&text[..boundary], &text[boundary..]] {
                    let r = p.parse_reasoning_streaming_incremental(chunk).unwrap();
                    reasoning.push_str(&r.reasoning_text);
                    content.push_str(&r.normal_text);
                }
                let r = p.flush().unwrap();
                reasoning.push_str(&r.reasoning_text);
                content.push_str(&r.normal_text);
                assert_eq!(reasoning, "推理🙂", "boundary {boundary}");
                assert_eq!(content, "answer");
            }
        }
    }
}
#[test]
fn hy4_no_think_reset_and_truncated_marker() {
    let mut p = HyV4Parser::new();
    assert_eq!(
        p.detect_and_parse_reasoning("answer").unwrap().normal_text,
        "answer"
    );
    p.mark_reasoning_started();
    assert_eq!(
        p.parse_reasoning_streaming_incremental("thought</thi")
            .unwrap()
            .reasoning_text,
        "thought"
    );
    assert_eq!(p.flush().unwrap().reasoning_text, "</thi");
    assert!(p.flush().unwrap().is_empty());
    p.reset();
    assert!(!p.is_in_reasoning());
    assert!(p.requires_special_tokens());
}

#[test]
fn hy4_factory_and_bytewise_stream() {
    let factory = reasoning_parser::factory::ParserFactory::new();
    let mut p = factory.create("tencent/Hy4-preview-FP8");
    assert_eq!(p.model_type(), "hy_v4");
    p.mark_reasoning_started();
    let mut reasoning = String::new();
    let mut content = String::new();
    for ch in "abc</think:6124c78e>ok".chars() {
        let r = p
            .parse_reasoning_streaming_incremental(&ch.to_string())
            .unwrap();
        reasoning.push_str(&r.reasoning_text);
        content.push_str(&r.normal_text);
    }
    assert_eq!(reasoning, "abc");
    assert_eq!(content, "ok");
}

#[test]
fn hy4_prompt_reasoning_reads_the_last_think_marker() {
    let p = HyV4Parser::new();
    let turn = "<｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜>";
    // The generation prompt opens the thought, with or without a checkpoint suffix.
    assert_eq!(
        p.prompt_reasoning(&format!("{turn}<think:opensource>\n")),
        PromptReasoning::Open
    );
    assert_eq!(
        p.prompt_reasoning(&format!("{turn}<think>")),
        PromptReasoning::Open
    );
    // `no_think` prefills an empty thought.
    assert_eq!(
        p.prompt_reasoning(&format!("{turn}<think:6124c78e></think:6124c78e>")),
        PromptReasoning::Closed
    );
    assert_eq!(p.prompt_reasoning(turn), PromptReasoning::Absent);
    // Earlier turns do not count; only the tail does.
    assert_eq!(
        p.prompt_reasoning(&format!(
            "{turn}<think:opensource>old</think:opensource>answer{turn}<think:opensource>\n"
        )),
        PromptReasoning::Open
    );
    // Text the output parser would not take for a marker is not one here either.
    for tail in [
        "<thinking>",
        "<think:>",
        "<think: x>",
        "<think:a<b>",
        "<think:opensource",
    ] {
        assert_eq!(
            p.prompt_reasoning(&format!("{turn}{tail}")),
            PromptReasoning::Absent,
            "{tail}"
        );
    }
}
