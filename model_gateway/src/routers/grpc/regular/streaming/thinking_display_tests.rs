//! `thinking.display` on the Messages path: `omitted` returns the thinking blocks without their
//! text, `summarized`, `updates` and no display return it, under the reasoning parser and under
//! Symphony alike, in the stream and in the message; the usage is the same under every value.

use openai_protocol::messages::CreateMessageRequest;
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::proto::generate_response::Response as GenerationEvent;
use symphony::{formats::qwen3, CallSyntax};

use super::{
    eof_tests::{chunk, complete, dispatch, scripted_stream, CharacterTokenizer},
    symphony_tests::{message_blocks, message_events},
    *,
};
use crate::{
    routers::grpc::{
        context,
        regular::{processor::ResponseProcessor, symphony::Assignment},
        spec::MessagesResponseSpec,
    },
    worker::WorkerRegistry,
};

/// Every `display` a request can state, and none, with whether the thinking text is returned.
const DISPLAYS: [(Option<&str>, bool); 4] = [
    (None, true),
    (Some("summarized"), true),
    (Some("updates"), true),
    (Some("omitted"), false),
];

/// The outputs: a thought then text, and text alone; with the thought the output has.
const OUTPUTS: [(&str, Option<&str>); 2] =
    [("<think>plan</think>Hello", Some("plan")), ("Hello", None)];

/// The spec of a request with adaptive thinking and `display`, read by the reasoning parser or,
/// `with_symphony`, by Symphony's Qwen3 table.
fn spec(display: Option<&str>, with_symphony: bool) -> MessagesResponseSpec {
    let mut thinking = json!({"type": "adaptive"});
    if let Some(display) = display {
        thinking["display"] = json!(display);
    }
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model": "m", "max_tokens": 64, "messages": [], "thinking": thinking
    }))
    .expect("the request deserializes");
    let mut spec = MessagesResponseSpec::from(&request);
    if with_symphony {
        spec.symphony = Some(Assignment {
            #[cfg(feature = "diagnostic-capture")]
            recipe: None,
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: "<|im_start|>assistant\n".to_string(),
        });
    }
    spec
}

/// The non-streaming processor with the Qwen3 reasoning parser.
fn response_processor() -> ResponseProcessor {
    ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::with_flags(
            Arc::new(WorkerRegistry::new()),
            None,
            Some("qwen3".to_string()),
        ),
    )
}

/// The thinking text a display returns for `thought`: the thought, or nothing under `omitted`.
fn returned_thought(thought: &str, returned: bool) -> &str {
    if returned {
        thought
    } else {
        ""
    }
}

/// One usage under every display, the thinking tokens counted when the output had a thought.
fn assert_usage_the_same(usages: &[Value], thought: Option<&str>) {
    assert!(
        usages.iter().all(|usage| *usage == usages[0]),
        "{usages:#?}"
    );
    let thinking_tokens = usages[0]["output_tokens_details"]["thinking_tokens"]
        .as_u64()
        .expect("thinking tokens, the request asked for thinking");
    assert_eq!(thinking_tokens > 0, thought.is_some(), "{usages:#?}");
}

#[tokio::test]
async fn the_thinking_text_leaves_the_stream_unless_the_display_is_omitted() {
    for with_symphony in [false, true] {
        for (output, thought) in OUTPUTS {
            let mut usages = Vec::new();
            for (display, returned) in DISPLAYS {
                // Cut inside the thought text, so the deltas join; the markers stay whole, as
                // an engine's token boundaries leave them.
                let cut = output.find("an</").unwrap_or(output.len() / 2);
                let frames = vec![
                    chunk(0, &output[..cut]),
                    chunk(0, &output[cut..]),
                    complete(0, "stop"),
                ];
                let (result, events) =
                    message_events(frames, spec(display, with_symphony), None).await;
                let case = format!("{display:?} on {output:?}, symphony {with_symphony}");
                assert!(result.is_ok(), "{case}: {result:?}");

                let mut expected = Vec::new();
                if let Some(thought) = thought {
                    expected.push((
                        "thinking".to_string(),
                        String::new(),
                        Value::String(returned_thought(thought, returned).to_string()),
                    ));
                }
                expected.push((
                    "text".to_string(),
                    String::new(),
                    Value::String("Hello".to_string()),
                ));
                assert_eq!(message_blocks(&events), expected, "{case}");
                let thinking_deltas = events
                    .iter()
                    .filter(|event| event["delta"]["type"] == "thinking_delta")
                    .count();
                assert_eq!(
                    thinking_deltas > 0,
                    thought.is_some() && returned,
                    "{case}: {events:#?}"
                );
                let message_delta = events
                    .iter()
                    .find(|event| event["type"] == "message_delta")
                    .unwrap_or_else(|| panic!("{case}: a message_delta"));
                usages.push(message_delta["usage"].clone());
            }
            assert_usage_the_same(&usages, thought);
        }
    }
}

#[tokio::test]
async fn the_thinking_text_is_in_the_message_unless_the_display_is_omitted() {
    for with_symphony in [false, true] {
        for (output, thought) in OUTPUTS {
            let mut usages = Vec::new();
            for (display, returned) in DISPLAYS {
                let mut frame = complete(0, "stop");
                if let Some(GenerationEvent::Complete(complete)) = &mut frame.response {
                    complete.output_ids = output.chars().map(u32::from).collect();
                    complete.completion_tokens = complete.output_ids.len() as u32;
                }
                let (stream, server) = scripted_stream(vec![frame], "0").await;
                let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
                let mut decoder =
                    utils::create_stop_decoder(&tokenizer, None, None, false, false, true);
                let (message, _counts) = response_processor()
                    .process_non_streaming_messages_response(
                        context::ExecutionResult::Single { stream },
                        spec(display, with_symphony),
                        dispatch(),
                        tokenizer,
                        &mut decoder,
                    )
                    .await
                    .unwrap_or_else(|response| panic!("{}", response.status()));
                server.abort();
                let message = serde_json::to_value(message).expect("serializable");
                let case = format!("{display:?} on {output:?}, symphony {with_symphony}");

                let mut expected = Vec::new();
                if let Some(thought) = thought {
                    expected.push(json!({
                        "type": "thinking",
                        "thinking": returned_thought(thought, returned),
                        "signature": "",
                    }));
                }
                expected.push(json!({"type": "text", "text": "Hello"}));
                assert_eq!(message["content"], Value::Array(expected), "{case}");
                usages.push(message["usage"].clone());
            }
            assert_usage_the_same(&usages, thought);
        }
    }
}
