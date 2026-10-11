//! The usage object of a request is the same whether its answer is streamed
//! or not. For a chat completion and for a legacy completion, the final usage
//! of a stream driven over scripted engine frames equals, field for field, the
//! usage of the non-streamed answer built from the same frames.

use openai_protocol::{chat::ChatCompletionRequest, completion::CompletionRequest};
use smg_grpc_client::vllm_engine::{proto, proto::generate_response::Response as GenerationEvent};

use super::{
    eof_tests::{chunk, collect_events, processor, scripted_stream, CharacterTokenizer},
    *,
};
use crate::{routers::grpc::regular::processor::ResponseProcessor, worker::WorkerRegistry};

/// The engine's counters for one request, as its `Complete` frames carry them.
#[derive(Clone, Copy)]
struct Counters {
    prompt: u32,
    cached: u32,
    spec_accepted: u32,
    spec_drafted: u32,
}

/// The frames of one choice: `texts` as delta chunks, then the terminal frame
/// with `counters` and the whole text as `output_ids` (the streamed path
/// decodes the chunks, the non-streamed one the terminal frame).
fn choice_frames(index: u32, texts: &[&str], counters: Counters) -> Vec<proto::GenerateResponse> {
    let output_ids: Vec<u32> = texts.concat().chars().map(u32::from).collect();
    let mut frames: Vec<_> = texts.iter().map(|text| chunk(index, text)).collect();
    frames.push(proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index,
            finish_reason: "stop".to_string(),
            prompt_tokens: counters.prompt,
            completion_tokens: output_ids.len() as u32,
            cached_tokens: counters.cached,
            spec_accepted_tokens: counters.spec_accepted,
            spec_draft_tokens: counters.spec_drafted,
            output_ids,
            ..Default::default()
        })),
    });
    frames
}

/// The frames of `choices` choices that each generate `texts`.
fn frames(texts: &[&str], choices: u32, counters: Counters) -> Vec<proto::GenerateResponse> {
    (0..choices)
        .flat_map(|index| choice_frames(index, texts, counters))
        .collect()
}

fn dispatch() -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: "cmpl-usage".to_string(),
        model: "eof-test".to_string(),
        created: 1,
        weight_version: None,
    }
}

/// The non-streamed processor with the parsers of [`processor`].
fn response_processor() -> ResponseProcessor {
    ResponseProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::new(
            Arc::new(WorkerRegistry::new()),
            None,
            Some("qwen3".to_string()),
        ),
    )
}

fn tokenizer() -> Arc<dyn Tokenizer> {
    Arc::new(CharacterTokenizer::default())
}

/// The `usage` of the usage-only chunk that ends a stream with
/// `stream_options.include_usage`.
fn usage_chunk(events: &[Value]) -> Value {
    let event = events
        .iter()
        .rfind(|event| event["choices"].as_array().is_some_and(Vec::is_empty))
        .expect("a usage-only chunk");
    event["usage"].clone()
}

async fn streamed_chat_usage(
    frames: Vec<proto::GenerateResponse>,
    request: &ChatCompletionRequest,
) -> Value {
    let (stream, server) = scripted_stream(frames, "0").await;
    let (tx, rx) = sse_channel();
    let result = processor(false)
        .process_streaming_chunks(
            stream,
            dispatch(),
            tokenizer(),
            (None, None, false, false, false),
            ChatResponseSpec::from(request),
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    usage_chunk(&events)
}

async fn non_streamed_chat_usage(
    frames: Vec<proto::GenerateResponse>,
    request: &ChatCompletionRequest,
) -> Value {
    let (stream, server) = scripted_stream(frames, "0").await;
    let tokenizer = tokenizer();
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let response = response_processor()
        .process_non_streaming_chat_response(
            context::ExecutionResult::Single { stream },
            ChatResponseSpec::from(request),
            dispatch(),
            tokenizer,
            &mut decoder,
            false,
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();
    serde_json::to_value(response).expect("a JSON response")["usage"].clone()
}

async fn streamed_completion_usage(
    frames: Vec<proto::GenerateResponse>,
    request: &CompletionRequest,
) -> Value {
    let (stream, server) = scripted_stream(frames, "0").await;
    let dispatch = dispatch();
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_COMPLETIONS);
    let response = Arc::new(processor(false))
        .process_completion_streaming_response(
            context::ExecutionResult::Single { stream },
            CompletionResponseSpec::from(request),
            dispatch,
            tokenizer(),
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
    usage_chunk(&events)
}

async fn non_streamed_completion_usage(
    frames: Vec<proto::GenerateResponse>,
    request: &CompletionRequest,
) -> Value {
    let (stream, server) = scripted_stream(frames, "0").await;
    let tokenizer = tokenizer();
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let response = response_processor()
        .process_non_streaming_completion_response(
            context::ExecutionResult::Single { stream },
            CompletionResponseSpec::from(request),
            dispatch(),
            tokenizer,
            &mut decoder,
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
async fn chat_usage_is_the_same_streamed_and_not() {
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
    for (name, texts, separate_reasoning, choices, counters, reasoning) in [
        ("plain", vec!["hi"], false, 1, plain, 0),
        (
            "reasoning the gateway's parser split off, cached prompt",
            vec!["<think>", "ab", "</think>", "de"],
            true,
            1,
            cached,
            2,
        ),
        (
            "speculative decoding, two choices",
            vec!["ok"],
            false,
            2,
            speculative,
            0,
        ),
    ] {
        let request = |stream: bool| -> ChatCompletionRequest {
            let mut request = json!({
                "model": "eof-test", "messages": [], "n": choices,
                "separate_reasoning": separate_reasoning, "stream": stream,
            });
            if stream {
                request["stream_options"] = json!({"include_usage": true});
            }
            serde_json::from_value(request).expect("chat request")
        };
        let streamed = streamed_chat_usage(frames(&texts, choices, counters), &request(true)).await;
        let whole =
            non_streamed_chat_usage(frames(&texts, choices, counters), &request(false)).await;
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
        assert_eq!(
            streamed["completion_tokens_details"]["reasoning_tokens"], reasoning,
            "{name}: {streamed}"
        );
    }
}

#[tokio::test]
async fn completion_usage_is_the_same_streamed_and_not() {
    let plain = Counters {
        prompt: 3,
        cached: 0,
        spec_accepted: 0,
        spec_drafted: 0,
    };
    let cached = Counters { cached: 2, ..plain };
    let speculative = Counters {
        spec_accepted: 2,
        spec_drafted: 3,
        ..cached
    };
    for (name, choices, counters) in [
        ("plain", 1, plain),
        ("cached prompt", 1, cached),
        ("speculative decoding, two choices", 2, speculative),
    ] {
        let request = |stream: bool| -> CompletionRequest {
            let mut request = json!({
                "model": "eof-test", "prompt": "hi", "n": choices, "stream": stream,
            });
            if stream {
                request["stream_options"] = json!({"include_usage": true});
            }
            serde_json::from_value(request).expect("completion request")
        };
        let streamed =
            streamed_completion_usage(frames(&["ok"], choices, counters), &request(true)).await;
        let whole =
            non_streamed_completion_usage(frames(&["ok"], choices, counters), &request(false))
                .await;
        assert_eq!(
            streamed, whole,
            "{name}: the streamed usage chunk vs the non-streamed usage"
        );
        assert_eq!(
            whole["prompt_tokens_details"]["cached_tokens"], counters.cached,
            "{name}: {whole}"
        );
    }
}
