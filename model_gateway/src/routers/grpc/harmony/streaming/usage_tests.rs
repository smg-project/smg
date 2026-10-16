//! The usage object of a Harmony chat completion is the same whether its
//! answer is streamed or not: the final usage of a stream driven over
//! scripted engine frames equals, field for field, the usage of the
//! non-streamed answer built from the same frames, with the reasoning tokens
//! the Harmony parser counts and every detail counter present. The usage of a
//! Responses API iteration is checked the same way.

use std::sync::Arc;

use openai_protocol::{chat::ChatCompletionRequest, responses::ResponsesRequest};
use serde_json::{json, Value};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};

use super::HarmonyStreamingProcessor;
use crate::{
    observability::metrics::metrics_labels,
    routers::{
        common::sse::sse_channel,
        grpc::{
            common::responses::streaming::ResponseStreamEventEmitter,
            context,
            harmony::{
                processor::{HarmonyResponseProcessor, ResponsesIterationResult},
                try_harmony_encoding,
            },
            regular::streaming::eof_tests::scripted_stream,
            utils,
        },
    },
};

/// The engine's counters for one request, as its `Complete` frames carry them.
#[derive(Clone, Copy)]
struct Counters {
    prompt: u32,
    cached: u32,
    spec_accepted: u32,
    spec_drafted: u32,
}

/// The tokens of one assistant turn in the Harmony format: the `analysis`
/// channel (the reasoning the parser counts) when given, then the `final`
/// channel's `text`.
pub(super) fn harmony_tokens(analysis: Option<&str>, text: &str) -> Vec<u32> {
    let mut turn = String::new();
    if let Some(analysis) = analysis {
        turn.push_str("<|channel|>analysis<|message|>");
        turn.push_str(analysis);
        turn.push_str("<|end|><|start|>assistant");
    }
    turn.push_str("<|channel|>final<|message|>");
    turn.push_str(text);
    turn.push_str("<|return|>");
    try_harmony_encoding()
        .expect("harmony encoding must load")
        .tokenizer()
        .encode_with_special_tokens(&turn)
}

/// The frames of one choice: `token_ids` in chunks of three, then the
/// terminal frame with `counters` and the whole output (the streamed path
/// parses the chunks, the non-streamed one the terminal frame).
fn choice_frames(
    index: u32,
    token_ids: &[u32],
    counters: Counters,
) -> Vec<proto::GenerateResponse> {
    let mut frames: Vec<proto::GenerateResponse> = token_ids
        .chunks(3)
        .map(|ids| proto::GenerateResponse {
            response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
                index,
                completion_tokens: ids.len() as u32,
                token_ids: ids.to_vec(),
                ..Default::default()
            })),
        })
        .collect();
    frames.push(proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index,
            finish_reason: "stop".to_string(),
            prompt_tokens: counters.prompt,
            completion_tokens: token_ids.len() as u32,
            cached_tokens: counters.cached,
            spec_accepted_tokens: counters.spec_accepted,
            spec_draft_tokens: counters.spec_drafted,
            output_ids: token_ids.to_vec(),
            ..Default::default()
        })),
    });
    frames
}

/// The frames of `choices` choices that each generate `token_ids`.
fn frames(token_ids: &[u32], choices: u32, counters: Counters) -> Vec<proto::GenerateResponse> {
    (0..choices)
        .flat_map(|index| choice_frames(index, token_ids, counters))
        .collect()
}

fn dispatch() -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: "harmony-usage".to_string(),
        model: "harmony-test".to_string(),
        created: 1,
        weight_version: None,
    }
}

fn chat_request(choices: u32, stream: bool) -> ChatCompletionRequest {
    let mut request = json!({
        "model": "harmony-test", "messages": [], "n": choices, "stream": stream,
    });
    if stream {
        request["stream_options"] = json!({"include_usage": true});
    }
    serde_json::from_value(request).expect("chat request")
}

/// The `usage` of the usage-only chunk that ends a Harmony chat stream with
/// `stream_options.include_usage`.
async fn streamed_chat_usage(frames: Vec<proto::GenerateResponse>, choices: u32) -> Value {
    let (stream, server) = scripted_stream(frames, "0").await;
    let dispatch = dispatch();
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_CHAT);
    let response = Arc::new(HarmonyStreamingProcessor::new())
        .process_streaming_chat_response(
            context::ExecutionResult::Single { stream },
            Arc::new(chat_request(choices, true)),
            dispatch,
            Vec::new(),
            None,
            source,
        )
        .await;
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("the stream's body");
    server.abort();
    let events: Vec<Value> = std::str::from_utf8(&body)
        .expect("UTF-8 SSE")
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE JSON"))
        .collect();
    assert!(
        events.iter().all(|event| event.get("error").is_none()),
        "the stream completes without an error event: {events:?}"
    );
    let usage_chunk = events
        .iter()
        .rfind(|event| event["choices"].as_array().is_some_and(Vec::is_empty))
        .expect("a usage-only chunk");
    usage_chunk["usage"].clone()
}

async fn non_streamed_chat_usage(frames: Vec<proto::GenerateResponse>, choices: u32) -> Value {
    let (stream, server) = scripted_stream(frames, "0").await;
    let response = HarmonyResponseProcessor::new()
        .process_non_streaming_chat_response(
            context::ExecutionResult::Single { stream },
            Arc::new(chat_request(choices, false)),
            dispatch(),
            &[],
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();
    serde_json::to_value(response).expect("a JSON response")["usage"].clone()
}

/// The detail counters the OpenAI API sends on every chat completion.
const CHAT_DETAIL_COUNTERS: [&str; 6] = [
    "/prompt_tokens_details/cached_tokens",
    "/prompt_tokens_details/audio_tokens",
    "/completion_tokens_details/reasoning_tokens",
    "/completion_tokens_details/audio_tokens",
    "/completion_tokens_details/accepted_prediction_tokens",
    "/completion_tokens_details/rejected_prediction_tokens",
];

#[tokio::test]
async fn harmony_chat_usage_is_the_same_streamed_and_not() {
    let plain = Counters {
        prompt: 10,
        cached: 0,
        spec_accepted: 0,
        spec_drafted: 0,
    };
    let cached = Counters { cached: 4, ..plain };
    let speculative = Counters {
        spec_accepted: 2,
        spec_drafted: 3,
        ..cached
    };
    for (name, analysis, choices, counters) in [
        ("final channel only", None, 1, plain),
        (
            "analysis then final, cached prompt",
            Some("The user greets; a short answer will do."),
            1,
            cached,
        ),
        (
            "analysis then final, speculative decoding, two choices",
            Some("Two choices think alike."),
            2,
            speculative,
        ),
    ] {
        let tokens = harmony_tokens(analysis, "Hello there!");
        let streamed = streamed_chat_usage(frames(&tokens, choices, counters), choices).await;
        let whole = non_streamed_chat_usage(frames(&tokens, choices, counters), choices).await;
        assert_eq!(
            streamed, whole,
            "{name}: the streamed usage chunk vs the non-streamed usage"
        );
        for pointer in CHAT_DETAIL_COUNTERS {
            assert!(
                streamed.pointer(pointer).is_some_and(Value::is_number),
                "{name}: {pointer} missing from {streamed}"
            );
        }
        assert_eq!(
            streamed["prompt_tokens_details"]["cached_tokens"], counters.cached,
            "{name}: {streamed}"
        );
        let reasoning = streamed["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .expect("a reasoning count");
        if analysis.is_some() {
            assert!(
                reasoning > 0,
                "{name}: the tokens the parser counted on the analysis channel: {streamed}"
            );
        } else {
            assert_eq!(reasoning, 0, "{name}: {streamed}");
        }
    }
}

/// The `usage` an iteration result carries.
fn iteration_usage(result: ResponsesIterationResult) -> Value {
    let usage = match result {
        ResponsesIterationResult::Completed { usage, .. } => usage,
        ResponsesIterationResult::ToolCallsFound { usage, .. } => usage,
    };
    serde_json::to_value(usage).expect("a JSON usage")
}

#[tokio::test]
async fn harmony_responses_iteration_usage_is_the_same_streamed_and_not() {
    let counters = Counters {
        prompt: 12,
        cached: 4,
        spec_accepted: 2,
        spec_drafted: 3,
    };
    let tokens = harmony_tokens(Some("Think, then answer."), "Done.");

    let (stream, server) = scripted_stream(frames(&tokens, 1, counters), "0").await;
    let mut emitter = ResponseStreamEventEmitter::new(
        "resp_harmony-usage".to_string(),
        "harmony-test".to_string(),
        1,
    );
    let (tx, mut rx) = sse_channel();
    // The events are read as they come, so the processor never waits on the
    // channel.
    let drain = async { while rx.recv().await.is_some() {} };
    let iteration = async {
        let result = HarmonyStreamingProcessor::process_responses_iteration_stream(
            context::ExecutionResult::Single { stream },
            &mut emitter,
            &tx,
            None,
            None,
        )
        .await;
        drop(tx);
        result
    };
    let (streamed, ()) = tokio::join!(iteration, drain);
    let streamed = streamed.expect("the streamed iteration");
    server.abort();

    let (stream, server) = scripted_stream(frames(&tokens, 1, counters), "0").await;
    let request: ResponsesRequest =
        serde_json::from_value(json!({"model": "harmony-test", "input": "hi"}))
            .expect("responses request");
    let whole = HarmonyResponseProcessor::new()
        .process_responses_iteration(
            context::ExecutionResult::Single { stream },
            Arc::new(request),
            dispatch(),
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();

    let (streamed, whole) = (iteration_usage(streamed), iteration_usage(whole));
    assert_eq!(
        streamed, whole,
        "the streamed iteration's usage vs the non-streamed one"
    );
    assert_eq!(
        streamed["prompt_tokens_details"]["cached_tokens"], 4,
        "{streamed}"
    );
    assert!(
        streamed["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .is_some_and(|reasoning| reasoning > 0),
        "{streamed}"
    );
    assert_eq!(
        streamed["completion_tokens_details"]["accepted_prediction_tokens"], 2,
        "{streamed}"
    );
}
