//! The usage object of a request is the same whether its answer is streamed
//! or not. For a chat completion and for a legacy completion, the final usage
//! of a stream driven over scripted engine frames equals, field for field, the
//! usage of the non-streamed answer built from the same frames; the frames of
//! the scheduler proto, whose terminal frame carries the engine's own
//! reasoning count, drive the same comparison.

use futures::stream;
use openai_protocol::{chat::ChatCompletionRequest, completion::CompletionRequest};
use smg_grpc_client::{
    sglang_scheduler::{
        proto as scheduler, proto::generate_response::Response as SchedulerEvent,
        AbortOnDropStream as SchedulerStream, SglangSchedulerClient,
    },
    vllm_engine::{proto, proto::generate_response::Response as GenerationEvent},
};
use tokio::task::JoinHandle;

use super::{
    eof_tests::{
        chunk, collect_events, data_frame, decoded_stream, mock_worker_endpoint, processor,
        scripted_stream, trailers_frame, CharacterTokenizer,
    },
    *,
};
use crate::{routers::grpc::regular::processor::ResponseProcessor, worker::WorkerRegistry};

/// A scripted worker stream and the mock server that owns its connection.
type Scripted = (ProtoStream, JoinHandle<()>);

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

/// [`frames`] as a scripted stream of the engine proto.
async fn engine_stream(frames: Vec<proto::GenerateResponse>) -> Scripted {
    scripted_stream(frames, "0").await
}

/// The frames of one choice in the scheduler proto, the one shape whose
/// terminal frame carries the engine's own reasoning count: `texts` as
/// delta chunks with the running count, then the terminal frame with
/// `counters`, `reasoning` and the whole text as `output_ids`.
fn scheduler_choice_frames(
    index: u32,
    texts: &[&str],
    counters: Counters,
    reasoning: u32,
) -> Vec<scheduler::GenerateResponse> {
    let output_ids: Vec<u32> = texts.concat().chars().map(u32::from).collect();
    let mut generated = 0;
    let mut frames: Vec<_> = texts
        .iter()
        .map(|text| {
            let token_ids: Vec<u32> = text.chars().map(u32::from).collect();
            generated += token_ids.len() as u32;
            scheduler::GenerateResponse {
                request_id: String::new(),
                response: Some(SchedulerEvent::Chunk(scheduler::GenerateStreamChunk {
                    index,
                    completion_tokens: generated,
                    token_ids,
                    ..Default::default()
                })),
            }
        })
        .collect();
    frames.push(scheduler::GenerateResponse {
        request_id: String::new(),
        response: Some(SchedulerEvent::Complete(scheduler::GenerateComplete {
            index,
            finish_reason: "stop".to_string(),
            prompt_tokens: counters.prompt,
            completion_tokens: output_ids.len() as u32,
            cached_tokens: counters.cached,
            reasoning_tokens: reasoning,
            spec_accepted_tokens: counters.spec_accepted,
            spec_draft_tokens: counters.spec_drafted,
            output_ids,
            ..Default::default()
        })),
    });
    frames
}

/// The scheduler-proto frames of `choices` choices that each generate
/// `texts`, every one reporting `reasoning` reasoning tokens of its own.
fn scheduler_frames(
    texts: &[&str],
    choices: u32,
    counters: Counters,
    reasoning: u32,
) -> Vec<scheduler::GenerateResponse> {
    (0..choices)
        .flat_map(|index| scheduler_choice_frames(index, texts, counters, reasoning))
        .collect()
}

/// [`scheduler_frames`] as a scripted stream of the scheduler proto, owned
/// by a scheduler client as a worker stream that was never dispatched.
async fn scheduler_stream(frames: Vec<scheduler::GenerateResponse>) -> Scripted {
    let body: Vec<Result<_, tonic::Status>> = frames
        .iter()
        .map(|frame| Ok(data_frame(frame)))
        .chain(std::iter::once(Ok(trailers_frame("0"))))
        .collect();
    let (url, server) = mock_worker_endpoint().await;
    let client = SglangSchedulerClient::connect(&url)
        .await
        .expect("connect mock worker");
    let stream = SchedulerStream::new(
        decoded_stream(stream::iter(body)),
        "cmpl-usage".to_string(),
        client,
    );
    // No generation was sent to the mock, so there is nothing to cancel.
    stream.mark_completed();
    (ProtoStream::Sglang(stream), server)
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

async fn streamed_chat_usage((stream, server): Scripted, request: &ChatCompletionRequest) -> Value {
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
    (stream, server): Scripted,
    request: &ChatCompletionRequest,
) -> Value {
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
    (stream, server): Scripted,
    request: &CompletionRequest,
) -> Value {
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
    (stream, server): Scripted,
    request: &CompletionRequest,
) -> Value {
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
        let streamed = streamed_chat_usage(
            engine_stream(frames(&texts, choices, counters)).await,
            &request(true),
        )
        .await;
        let whole = non_streamed_chat_usage(
            engine_stream(frames(&texts, choices, counters)).await,
            &request(false),
        )
        .await;
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
        let streamed = streamed_completion_usage(
            engine_stream(frames(&["ok"], choices, counters)).await,
            &request(true),
        )
        .await;
        let whole = non_streamed_completion_usage(
            engine_stream(frames(&["ok"], choices, counters)).await,
            &request(false),
        )
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

/// The engine's own reasoning count (the scheduler proto's terminal frame
/// carries one; the engine proto's does not) reaches
/// `completion_tokens_details.reasoning_tokens` on both shapes, streamed and
/// not, summed over the choices: on the chat shape whether or not the
/// gateway's parser ran (an engine that counts reasoning itself is not
/// overruled, even when it counts fewer tokens than the parser split off),
/// on the legacy completion shape through the non-streamed answer's sum of
/// the terminal frames' counts.
#[tokio::test]
async fn the_engines_own_reasoning_count_is_the_same_streamed_and_not() {
    let counters = Counters {
        prompt: 10,
        cached: 4,
        spec_accepted: 0,
        spec_drafted: 0,
    };
    let choices = 2;
    for (name, texts, separate_reasoning, reasoning) in [
        ("the gateway's parser off", vec!["ab", "cd"], false, 3),
        // The parser splits two tokens off per choice; the engine's count of
        // one stands, below the parser's, so a rule that took the higher of
        // the two would be caught here.
        (
            "the gateway's parser on",
            vec!["<think>", "ab", "</think>", "de"],
            true,
            1,
        ),
    ] {
        let summed = json!(reasoning * choices);
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
        let streamed = streamed_chat_usage(
            scheduler_stream(scheduler_frames(&texts, choices, counters, reasoning)).await,
            &request(true),
        )
        .await;
        let whole = non_streamed_chat_usage(
            scheduler_stream(scheduler_frames(&texts, choices, counters, reasoning)).await,
            &request(false),
        )
        .await;
        assert_eq!(
            streamed, whole,
            "chat, {name}: the streamed usage chunk vs the non-streamed usage"
        );
        assert_eq!(
            streamed["completion_tokens_details"]["reasoning_tokens"], summed,
            "chat, {name}: {streamed}"
        );
    }

    let reasoning = 3;
    let summed = json!(reasoning * choices);
    let request = |stream: bool| -> CompletionRequest {
        let mut request = json!({
            "model": "eof-test", "prompt": "hi", "n": choices, "stream": stream,
        });
        if stream {
            request["stream_options"] = json!({"include_usage": true});
        }
        serde_json::from_value(request).expect("completion request")
    };
    let streamed = streamed_completion_usage(
        scheduler_stream(scheduler_frames(&["ok"], choices, counters, reasoning)).await,
        &request(true),
    )
    .await;
    let whole = non_streamed_completion_usage(
        scheduler_stream(scheduler_frames(&["ok"], choices, counters, reasoning)).await,
        &request(false),
    )
    .await;
    assert_eq!(
        streamed, whole,
        "completion: the streamed usage chunk vs the non-streamed usage"
    );
    assert_eq!(
        whole["completion_tokens_details"]["reasoning_tokens"], summed,
        "completion: {whole}"
    );
}
