use openai_protocol::chat::ChatCompletionRequest;
use smg_grpc_client::tokenspeed_proto::GenerateComplete;

use super::*;

#[expect(
    dead_code,
    reason = "this shared fixture also supports multi-turn MCP tests"
)]
mod scripted_tokenizer {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_harness/src/scripted_tokenizer.rs"
    ));
}

/// The Messages tool choice read as its chat form decides the JSON-schema
/// path exactly as the chat choice does: a named tool and `any` force a
/// call, `auto` and `none` do not, and a native format never takes it.
#[test]
fn a_messages_choice_forces_a_call_as_its_chat_form_does() {
    let named = messages::ToolChoice::Tool {
        name: "lookup".to_string(),
        disable_parallel_tool_use: None,
    };
    let any = messages::ToolChoice::Any {
        disable_parallel_tool_use: None,
    };
    let auto = messages::ToolChoice::Auto {
        disable_parallel_tool_use: None,
    };
    for (choice, forces) in [
        (named, true),
        (any, true),
        (auto, false),
        (messages::ToolChoice::None, false),
    ] {
        let chat = ToolChoice::from(&choice);
        assert_eq!(
            forced_call_shapes_output(false, Some(&chat)),
            forces,
            "{choice:?}"
        );
        assert!(
            !forced_call_shapes_output(true, Some(&chat)),
            "{choice:?} native"
        );
    }
    assert!(!forced_call_shapes_output(false, None));
}

/// Without a parser the split leaves the text whole.
#[test]
fn splitting_without_a_reasoning_parser_leaves_the_text_whole() {
    let processor = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    );
    let split = processor.split_reasoning(
        "<think>a</think>b".to_string(),
        &[8, 9, 17],
        ReasoningSource {
            wanted: true,
            in_prefill: false,
            json_mode: false,
            parser_name: None,
            model: "no-such-model",
        },
    );
    assert_eq!(split.content, "<think>a</think>b");
    assert_eq!((split.reasoning, split.reasoning_tokens), (None, 0));
}

/// One decode for every reader: the text, where each token's text ends,
/// whether a stop fired; the finish and the matched stop read from it.
#[test]
fn decoding_a_choice_reports_its_text_token_ends_and_stop() {
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(scripted_tokenizer::ScriptedTokenizer::from_chunks(vec![
            "Hel".into(),
            "lo ".into(),
            "world".into(),
            "!".into(),
        ]));
    let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
        finish_reason: "length".into(),
        ..Default::default()
    });

    // Without a stop: every token's text end, the last at the text's end.
    let mut decoder = StopSequenceDecoder::new(
        tokenizer.clone(),
        llm_tokenizer::StopSequenceConfig::default(),
        false,
    );
    let decoded = decode_complete(&mut decoder, &[100, 101, 102, 103], "req", 0).unwrap();
    assert_eq!(decoded.text, "Hello world!");
    assert_eq!(decoded.token_ends, vec![3, 6, 11, 12]);
    assert!(!decoded.stopped);
    assert_eq!(
        finish_after_local_stop(decoded.stopped, &complete),
        "length"
    );
    assert_eq!(matched_stop_value(&decoder, &complete), None);

    // A string stop: the text ends before it, the stop fired and is the
    // finish and the matched stop, whatever the engine said.
    let mut decoder = StopSequenceDecoder::new(
        tokenizer.clone(),
        llm_tokenizer::StopSequenceConfig {
            stop_sequences: vec!["world".to_string()],
            ..Default::default()
        },
        false,
    );
    let decoded = decode_complete(&mut decoder, &[100, 101, 102, 103], "req", 0).unwrap();
    assert_eq!(decoded.text, "Hello ");
    assert!(decoded.stopped);
    assert_eq!(decoded.token_ends.last(), Some(&decoded.text.len()));
    assert_eq!(finish_after_local_stop(decoded.stopped, &complete), "stop");
    assert_eq!(
        matched_stop_value(&decoder, &complete),
        Some(serde_json::Value::String("world".into()))
    );
    assert_eq!(
        matched_stop_string(&decoder, &complete).as_deref(),
        Some("world")
    );
}

#[tokio::test]
async fn parsed_tool_call_preserves_engine_length_finish() {
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(scripted_tokenizer::ScriptedTokenizer::new(
        "<tool_call>\n{\"name\":\"user_tool\",\"arguments\":{}}\n</tool_call>",
    ));
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model":"test-model","messages":[{"role":"user","content":"call the tool"}],
        "tools":[{"type":"function","function":{"name":"user_tool","parameters":{"type":"object","properties":{}}}}]
    })).unwrap();
    let processor = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    );
    for (finish, expected) in [("stop", "tool_calls"), ("length", "length")] {
        let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
            output_ids: vec![100],
            finish_reason: finish.into(),
            ..Default::default()
        });
        let mut decoder = StopSequenceDecoder::new(
            tokenizer.clone(),
            llm_tokenizer::StopSequenceConfig::default(),
            false,
        );
        let (choice, _) = processor
            .process_single_choice(
                "test-request",
                &complete,
                0,
                &ChatResponseSpec::from(&request),
                "test-model",
                &tokenizer,
                &mut decoder,
                0,
                false,
                true,
                None,
                Some("qwen"),
            )
            .await
            .unwrap();
        assert_eq!(choice.finish_reason.as_deref(), Some(expected));
        assert_eq!(choice.message.tool_calls.as_ref().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn vllm_json_tool_nonstream_keeps_prefilled_reasoning() {
    let processor = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    );
    for (tool_choice, output) in [
        (
            serde_json::json!("required"),
            r#"plan</think>[{"name":"lookup","parameters":{"q":"Paris"}}]"#,
        ),
        (
            serde_json::json!({"type":"function", "function":{"name":"lookup"}}),
            r#"plan</think>{"q":"Paris"}"#,
        ),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"test-model", "messages":[], "separate_reasoning":true,
            "tool_choice":tool_choice,
            "tools":[{"type":"function","function":{"name":"lookup",
                "parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}]
        }))
        .unwrap();
        let mut spec = ChatResponseSpec::from(&request);
        spec.reasoning_in_prefill = true;
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(scripted_tokenizer::ScriptedTokenizer::new(output));
        let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
            output_ids: vec![100],
            finish_reason: "stop".into(),
            ..Default::default()
        });
        let mut decoder = StopSequenceDecoder::new(
            tokenizer.clone(),
            llm_tokenizer::StopSequenceConfig::default(),
            false,
        );
        let (choice, _) = processor
            .process_single_choice(
                "test-request",
                &complete,
                0,
                &spec,
                "test-model",
                &tokenizer,
                &mut decoder,
                0,
                true,
                true,
                Some("qwen3"),
                Some("qwen_coder"),
            )
            .await
            .unwrap();
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        assert!(choice.message.content.is_none());
        assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
        let calls = choice.message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                calls[0].function.arguments.as_deref().unwrap()
            )
            .unwrap(),
            serde_json::json!({"q":"Paris"})
        );
    }
}

#[tokio::test]
async fn gemma4_nonstream_native_tool_choices_preserve_reasoning_and_arguments() {
    let output = r#"<|channel>thought
plan<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#;
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(scripted_tokenizer::ScriptedTokenizer::new(output));
    let processor = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    );
    for tool_choice in [
        serde_json::json!("auto"),
        serde_json::json!("required"),
        serde_json::json!({"type":"function", "function":{"name":"get_weather"}}),
    ] {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"test-model", "messages":[], "separate_reasoning":true,
            "tool_choice":tool_choice,
            "tools":[{"type":"function","function":{"name":"get_weather",
                "parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
        }))
        .unwrap();
        let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
            output_ids: vec![100],
            finish_reason: "stop".into(),
            ..Default::default()
        });
        let mut decoder = StopSequenceDecoder::new(
            tokenizer.clone(),
            llm_tokenizer::StopSequenceConfig::default(),
            false,
        );
        let (choice, _) = processor
            .process_single_choice(
                "test-request",
                &complete,
                0,
                &ChatResponseSpec::from(&request),
                "test-model",
                &tokenizer,
                &mut decoder,
                0,
                true,
                true,
                Some("gemma4"),
                Some("gemma4"),
            )
            .await
            .unwrap();
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
        assert!(choice.message.content.is_none());
        assert_eq!(choice.message.reasoning_content.as_deref(), Some("plan"));
        let calls = choice.message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                calls[0].function.arguments.as_deref().unwrap()
            )
            .unwrap(),
            serde_json::json!({"city":"Paris"})
        );
    }
}

/// `parallel_tool_calls: false` returns the first of the calls a model
/// writes; unset or true, all of them.
#[tokio::test]
async fn parallel_tool_calls_false_returns_the_first_call() {
    let call = |q: u32| {
        format!("<tool_call>\n{{\"name\":\"user_tool\",\"arguments\":{{\"q\":{q}}}}}\n</tool_call>")
    };
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(scripted_tokenizer::ScriptedTokenizer::new(
        &format!("{}\n{}", call(1), call(2)),
    ));
    let processor = ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::disabled(),
    );
    for (parallel, calls) in [(None, 2), (Some(true), 2), (Some(false), 1)] {
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"test-model","messages":[{"role":"user","content":"call the tool"}],
            "tools":[{"type":"function","function":{"name":"user_tool","parameters":{"type":"object","properties":{"q":{"type":"integer"}}}}}],
            "parallel_tool_calls": parallel
        })).unwrap();
        let complete = ProtoGenerateComplete::TokenSpeed(GenerateComplete {
            output_ids: vec![100],
            finish_reason: "stop".into(),
            ..Default::default()
        });
        let mut decoder = StopSequenceDecoder::new(
            tokenizer.clone(),
            llm_tokenizer::StopSequenceConfig::default(),
            false,
        );
        let (choice, _) = processor
            .process_single_choice(
                "test-request",
                &complete,
                0,
                &ChatResponseSpec::from(&request),
                "test-model",
                &tokenizer,
                &mut decoder,
                0,
                false,
                true,
                None,
                Some("qwen"),
            )
            .await
            .unwrap();
        let got = choice.message.tool_calls.unwrap();
        assert_eq!(got.len(), calls, "{parallel:?}");
        assert_eq!(got[0].function.arguments.as_deref(), Some("{\"q\":1}"));
        assert_eq!(choice.finish_reason.as_deref(), Some("tool_calls"));
    }
}
