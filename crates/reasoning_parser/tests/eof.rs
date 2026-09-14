use reasoning_parser::{
    BaseReasoningParser, ParserConfig, ParserFactory, ParserResult, ReasoningParser,
};

#[test]
fn flush_preserves_partial_markers_for_every_parser() {
    let factory = ParserFactory::new();
    for name in factory.registry().list_parsers() {
        if name == "passthrough" {
            continue;
        }
        let tail = if name == "kimi" { "◁" } else { "<" };
        for in_reasoning in [false, true] {
            let mut parser = factory.registry().create_parser(&name).unwrap();
            if in_reasoning {
                parser.mark_reasoning_started();
            }
            let expected = if parser.is_in_reasoning() {
                ParserResult::reasoning(tail.to_owned())
            } else {
                ParserResult::normal(tail.to_owned())
            };
            let mut actual = parser.parse_reasoning_streaming_incremental(tail).unwrap();
            let final_text = parser.flush().unwrap();
            actual.normal_text.push_str(&final_text.normal_text);
            actual.reasoning_text.push_str(&final_text.reasoning_text);
            assert_eq!(actual, expected, "{name}, prefill={in_reasoning}");
            assert!(parser.flush().unwrap().is_empty(), "{name} repeated flush");
            parser.reset();
            assert!(parser.flush().unwrap().is_empty(), "{name} reset");
        }
    }
}

#[test]
fn flush_emits_reasoning_when_incremental_output_is_disabled() {
    let mut parser = BaseReasoningParser::new(ParserConfig {
        stream_reasoning: false,
        ..Default::default()
    });
    assert!(parser
        .parse_reasoning_streaming_incremental("<think>unfinished</thi")
        .unwrap()
        .is_empty());
    assert_eq!(
        parser.flush().unwrap(),
        ParserResult::reasoning("unfinished</thi".to_owned())
    );
    assert!(parser.flush().unwrap().is_empty());
}

#[test]
fn flush_emits_only_the_held_tail() {
    for (name, chunks, expected) in [
        (
            "qwen3",
            vec!["<think>reason", "</thi"],
            ParserResult::reasoning("reason</thi".to_owned()),
        ),
        (
            "qwen3",
            vec!["<think>reason</think>answer", "<thi"],
            ParserResult::new("answer<thi".to_owned(), "reason".to_owned()),
        ),
        (
            "minimax_m3",
            vec!["<mm:think>reason</mm:thi"],
            ParserResult::reasoning("reason</mm:thi".to_owned()),
        ),
        (
            "kimi_k3",
            vec!["<|open|>think<|sep|>reason<|clo"],
            ParserResult::reasoning("reason<|clo".to_owned()),
        ),
        (
            "kimi_k3",
            vec![
                "<|open|>think<|sep|>reason<|close|>think<|sep|><|open|>response<|sep|>answer<|clo",
            ],
            ParserResult::new("answer<|clo".to_owned(), "reason".to_owned()),
        ),
    ] {
        let mut parser = ParserFactory::new().registry().create_parser(name).unwrap();
        let mut actual = ParserResult::default();
        for chunk in chunks {
            let result = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
            actual.normal_text.push_str(&result.normal_text);
            actual.reasoning_text.push_str(&result.reasoning_text);
        }
        let tail = parser.flush().unwrap();
        actual.normal_text.push_str(&tail.normal_text);
        actual.reasoning_text.push_str(&tail.reasoning_text);
        assert_eq!(actual, expected, "{name}");
        assert!(parser.flush().unwrap().is_empty(), "{name} repeated flush");
        if name == "kimi_k3" {
            parser.reset();
            let next = parser.parse_reasoning_streaming_incremental(
                "<|open|>message<|sep|><|open|>think<|sep|>new reason<|close|>think<|sep|><|open|>response<|sep|>new answer<|close|>response<|sep|><|close|>message<|sep|>",
            ).unwrap();
            assert_eq!(
                next,
                ParserResult::new("new answer".to_owned(), "new reason".to_owned())
            );
            assert!(parser.flush().unwrap().is_empty());
        }
    }
}
