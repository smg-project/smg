//! The usage object of a request is the same whether its answer is streamed
//! or not. For a chat completion and for a legacy completion, the final usage
//! of a stream driven over scripted engine frames equals, field for field, the
//! usage of the non-streamed answer built from the same frames; the frames of
//! the scheduler proto, whose terminal frame carries the engine's own
//! reasoning count, drive the same comparison.
//!
//! The settlement of a request's reservation is checked the same way, on the
//! chat, completion, generate and messages shapes: it carries the engine's
//! cached and reasoning counts beside the prompt and completion counts, the
//! same streamed and not, zero where the engine's frames carry none.

use futures::stream;
use openai_protocol::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    common::StringOrArray,
    completion::{CompletionRequest, CompletionResponse},
    generate::GenerateResponse,
};
use smg_grpc_client::{
    sglang_scheduler::{
        proto as scheduler, proto::generate_response::Response as SchedulerEvent,
        AbortOnDropStream as SchedulerStream, SglangSchedulerClient,
    },
    vllm_engine::{proto, proto::generate_response::Response as GenerationEvent},
};
use symphony::{formats::qwen3, CallSyntax};
use tokio::{sync::oneshot, task::JoinHandle};

use super::{
    super::*,
    eof::{
        chunk, collect_events, data_frame, decoded_stream, gated_stream, messages_spec,
        mock_worker_endpoint, processor, response_processor, runtime, scripted_stream,
        trailers_frame, CharacterTokenizer,
    },
    symphony_chat::assigned_message,
};
use crate::{rate_limit::test_support::RecordingBackend, routers::grpc::pipeline::RequestPipeline};

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
                media_identity: None,
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
        media_identity: None,
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
    (stream, server): Scripted,
    request: &ChatCompletionRequest,
    reservation: Option<Arc<SharedReservationHandle>>,
) -> Value {
    let (tx, rx) = sse_channel();
    let result = processor(false)
        .process_streaming_chunks(
            stream,
            dispatch(),
            tokenizer(),
            (None, None, false, false, false),
            ChatResponseSpec::from(request),
            &tx,
            reservation,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    usage_chunk(&events)
}

/// The non-streamed chat answer with the engine's counters it was built from.
async fn non_streamed_chat(
    (stream, server): Scripted,
    request: &ChatCompletionRequest,
) -> (ChatCompletionResponse, TokenCounts) {
    let tokenizer = tokenizer();
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let answer = response_processor()
        .process_non_streaming_chat_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            ChatResponseSpec::from(request),
            dispatch(),
            tokenizer,
            &mut decoder,
            false,
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();
    answer
}

async fn non_streamed_chat_usage(scripted: Scripted, request: &ChatCompletionRequest) -> Value {
    let (response, _) = non_streamed_chat(scripted, request).await;
    serde_json::to_value(response).expect("a JSON response")["usage"].clone()
}

async fn streamed_completion_usage(
    (stream, server): Scripted,
    request: &CompletionRequest,
    reservation: Option<Arc<SharedReservationHandle>>,
) -> Value {
    let dispatch = dispatch();
    let source = utils::StreamSource::new(None, &dispatch, metrics_labels::ENDPOINT_COMPLETIONS);
    let response = Arc::new(processor(false))
        .process_completion_streaming_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            CompletionResponseSpec::new(request, Vec::new()),
            dispatch,
            tokenizer(),
            reservation,
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

async fn non_streamed_completion(
    (stream, server): Scripted,
    request: &CompletionRequest,
) -> CompletionResponse {
    let tokenizer = tokenizer();
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let response = response_processor()
        .process_non_streaming_completion_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            CompletionResponseSpec::new(request, Vec::new()),
            dispatch(),
            &tokenizer,
            &mut decoder,
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();
    response
}

async fn non_streamed_completion_usage(scripted: Scripted, request: &CompletionRequest) -> Value {
    let response = non_streamed_completion(scripted, request).await;
    serde_json::to_value(response).expect("a JSON response")["usage"].clone()
}

/// Drive one of the two generate streaming paths (the one with input
/// logprobs serves the disaggregated mode too) over `stream` with a
/// reservation to settle; the path must succeed.
async fn streamed_generate(
    (stream, server): Scripted,
    with_input_logprobs: bool,
    choices: u32,
    reservation: Option<Arc<SharedReservationHandle>>,
) {
    let ctx = GenerateStreamContext {
        request_id: "cmpl-usage".to_string(),
        weight_version: "default".to_string(),
        return_logprob: false,
        backend_type: "test",
        model: "eof-test".to_string(),
        expected_choices: choices,
    };
    let (tx, rx) = sse_channel();
    let result = if with_input_logprobs {
        StreamingProcessor::process_generate_streaming_with_input_logprobs(
            tokenizer(),
            stream,
            ctx,
            None,
            &tx,
            None,
            reservation,
        )
        .await
    } else {
        StreamingProcessor::process_generate_streaming(tokenizer(), stream, ctx, &tx, reservation)
            .await
    };
    drop(tx);
    let _events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
}

async fn non_streamed_generate(
    (stream, server): Scripted,
    metrics_endpoint: &'static str,
) -> Vec<GenerateResponse> {
    let tokenizer = tokenizer();
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let responses = response_processor()
        .process_non_streaming_generate_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            dispatch(),
            &mut decoder,
            false,
            Instant::now(),
            metrics_endpoint,
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();
    responses
}

/// Drive the Messages streaming path over `stream` with a reservation to
/// settle and return its events; the path must succeed.
async fn streamed_messages(
    (stream, server): Scripted,
    spec: MessagesResponseSpec,
    reservation: Option<Arc<SharedReservationHandle>>,
) -> Vec<Value> {
    let (tx, rx) = sse_channel();
    let result = processor(false)
        .process_messages_streaming_chunks(
            stream,
            dispatch(),
            tokenizer(),
            (None, None, false, false, false),
            spec,
            &tx,
            reservation,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    events
}

/// The non-streamed message with the engine's counters it was built from.
async fn non_streamed_messages(
    (stream, server): Scripted,
    spec: MessagesResponseSpec,
) -> (Message, TokenCounts) {
    let tokenizer = tokenizer();
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let answer = response_processor()
        .process_non_streaming_messages_response(
            context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            },
            spec,
            dispatch(),
            tokenizer,
            &mut decoder,
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();
    answer
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
            None,
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
            None,
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
            None,
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
        None,
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

/// The engine's counters a settlement test scripts: `reasoning` per choice
/// is carried by the scheduler proto's terminal frame alone.
const SETTLED: Counters = Counters {
    prompt: 10,
    cached: 4,
    spec_accepted: 0,
    spec_drafted: 0,
};

/// The settlement of a chat request carries the engine's own cached and
/// reasoning counts, streamed and not: the counts of the scheduler proto's
/// terminal frame, whatever the gateway's parser split off; zero from the
/// engine proto's, which carries no reasoning count, though the parser's
/// split is what the usage object then shows. A stream that ends without
/// a terminal frame is closed with the reserved amount instead, as before.
#[tokio::test]
async fn chat_settlement_carries_the_engines_counts_streamed_and_not() {
    let texts = ["<think>", "ab", "</think>", "de"];
    let choices = 2;
    let completion = texts.concat().chars().count() as u32 * choices;
    let request = |stream: bool| -> ChatCompletionRequest {
        let mut request = json!({
            "model": "eof-test", "messages": [], "n": choices,
            "separate_reasoning": true, "stream": stream,
        });
        if stream {
            request["stream_options"] = json!({"include_usage": true});
        }
        serde_json::from_value(request).expect("chat request")
    };

    // The engine counts one reasoning token per choice; the parser splits two.
    let expected = UsageSettlement {
        actual_input_tokens: SETTLED.prompt,
        completion_tokens: completion,
        cached_input_tokens: SETTLED.cached,
        reasoning_tokens: choices,
    };
    let (handle, backend) = RecordingBackend::reservation();
    streamed_chat_usage(
        scheduler_stream(scheduler_frames(&texts, choices, SETTLED, 1)).await,
        &request(true),
        Some(handle),
    )
    .await;
    assert_eq!(backend.settlements(), vec![expected]);
    assert_eq!(backend.closes(), 0);
    let (_, counts) = non_streamed_chat(
        scheduler_stream(scheduler_frames(&texts, choices, SETTLED, 1)).await,
        &request(false),
    )
    .await;
    assert_eq!(counts.settlement(), expected);

    // No reasoning count and a cold prompt from the engine: zeros, though the
    // usage object shows the two tokens per choice the parser split off.
    let cold = Counters {
        cached: 0,
        ..SETTLED
    };
    let expected = UsageSettlement {
        actual_input_tokens: SETTLED.prompt,
        completion_tokens: completion,
        cached_input_tokens: 0,
        reasoning_tokens: 0,
    };
    let (handle, backend) = RecordingBackend::reservation();
    let usage = streamed_chat_usage(
        engine_stream(frames(&texts, choices, cold)).await,
        &request(true),
        Some(handle),
    )
    .await;
    assert_eq!(
        usage["completion_tokens_details"]["reasoning_tokens"],
        2 * choices,
        "{usage}"
    );
    assert_eq!(backend.settlements(), vec![expected]);
    let (response, counts) = non_streamed_chat(
        engine_stream(frames(&texts, choices, cold)).await,
        &request(false),
    )
    .await;
    assert_eq!(counts.settlement(), expected);
    let usage = serde_json::to_value(response).expect("a JSON response")["usage"].clone();
    assert_eq!(
        usage["completion_tokens_details"]["reasoning_tokens"],
        2 * choices,
        "{usage}"
    );

    // Chunks and a clean EOF, no terminal frame: no authoritative usage.
    let (handle, backend) = RecordingBackend::reservation();
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "messages": [], "stream": true,
        "stream_options": {"include_usage": true},
    }))
    .expect("chat request");
    streamed_chat_usage(
        engine_stream(vec![chunk(0, "hi")]).await,
        &request,
        Some(handle),
    )
    .await;
    assert!(backend.settlements().is_empty());
    assert_eq!(backend.closes(), 1);
}

/// The two cases the completion, generate and messages settlement tests
/// drive, each with the settlement it expects: the engine's counts (a
/// scheduler stream whose terminal frames
/// carry `reasoning` reasoning tokens per choice and the cache hit of
/// `SETTLED`) and no counts from the engine (an engine stream, no cache
/// hit, no reasoning); `completion` is the completion count the choices
/// generate together.
fn settlement_cases(
    reasoning: u32,
    completion: u32,
    choices: u32,
) -> [(&'static str, Option<u32>, Counters, UsageSettlement); 2] {
    [
        (
            "the engine's counts",
            Some(reasoning),
            SETTLED,
            UsageSettlement {
                actual_input_tokens: SETTLED.prompt,
                completion_tokens: completion,
                cached_input_tokens: SETTLED.cached,
                reasoning_tokens: reasoning * choices,
            },
        ),
        (
            "no counts from the engine",
            None,
            Counters {
                cached: 0,
                ..SETTLED
            },
            UsageSettlement {
                actual_input_tokens: SETTLED.prompt,
                completion_tokens: completion,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
        ),
    ]
}

/// The settlement of a legacy completion carries the engine's counts summed
/// over the choices, streamed and not; the non-streamed answer's usage
/// object is those counts, so the pipeline settles from it.
#[tokio::test]
async fn completion_settlement_carries_the_engines_counts_streamed_and_not() {
    let choices = 2;
    let request = |stream: bool| -> CompletionRequest {
        let mut request = json!({
            "model": "eof-test", "prompt": "hi", "n": choices, "stream": stream,
        });
        if stream {
            request["stream_options"] = json!({"include_usage": true});
        }
        serde_json::from_value(request).expect("completion request")
    };
    for (name, scheduler_reasoning, counters, expected) in settlement_cases(3, 2 * choices, choices)
    {
        let scripted = || async {
            match scheduler_reasoning {
                Some(reasoning) => {
                    scheduler_stream(scheduler_frames(&["ok"], choices, counters, reasoning)).await
                }
                None => engine_stream(frames(&["ok"], choices, counters)).await,
            }
        };
        let (handle, backend) = RecordingBackend::reservation();
        streamed_completion_usage(scripted().await, &request(true), Some(handle)).await;
        assert_eq!(backend.settlements(), vec![expected], "{name}: streamed");
        let response = non_streamed_completion(scripted().await, &request(false)).await;
        assert_eq!(
            RequestPipeline::completion_settlement(response.usage.as_ref()),
            expected,
            "{name}: non-streamed"
        );
    }
}

/// The settlement of a generate request carries the engine's counts on both
/// streaming paths and on the non-streamed answer: the shared prompt and its
/// cache hit once, each choice's completion and reasoning counts summed.
#[tokio::test]
async fn generate_settlement_carries_the_engines_counts_streamed_and_not() {
    let choices = 2;
    for (name, scheduler_reasoning, counters, expected) in settlement_cases(3, 2 * choices, choices)
    {
        let scripted = || async {
            match scheduler_reasoning {
                Some(reasoning) => {
                    scheduler_stream(scheduler_frames(&["ok"], choices, counters, reasoning)).await
                }
                None => engine_stream(frames(&["ok"], choices, counters)).await,
            }
        };
        for with_input_logprobs in [false, true] {
            let (handle, backend) = RecordingBackend::reservation();
            streamed_generate(scripted().await, with_input_logprobs, choices, Some(handle)).await;
            assert_eq!(
                backend.settlements(),
                vec![expected],
                "{name}: streamed, input logprobs {with_input_logprobs}"
            );
        }
        let responses =
            non_streamed_generate(scripted().await, metrics_labels::ENDPOINT_GENERATE).await;
        assert_eq!(
            RequestPipeline::generate_settlement(&responses),
            expected,
            "{name}: non-streamed"
        );
    }
}

/// The settlement of a Messages request carries the engine's counts, streamed
/// and not: the engine's cache hit and its own thinking count, zero where its
/// frames carry none. The Messages usage reports the same cache hit as
/// `cache_read_input_tokens` beside `input_tokens` = the prompt outside the
/// cache, on the final `message_delta` and on the collected message (integers
/// either way; the `message_start` skeleton, which precedes the engine's
/// frames, reports the whole rendered prompt and a zero cache hit), and
/// the greater of the engine's and the parser's thinking count.
#[tokio::test]
async fn messages_settlement_carries_the_engines_counts_streamed_and_not() {
    let texts = ["<think>", "ab", "</think>", "de"];
    let completion = texts.concat().chars().count() as u32;
    for (name, scheduler_reasoning, counters, expected) in settlement_cases(1, completion, 1) {
        let scripted = || async {
            match scheduler_reasoning {
                Some(reasoning) => {
                    scheduler_stream(scheduler_frames(&texts, 1, counters, reasoning)).await
                }
                None => engine_stream(frames(&texts, 1, counters)).await,
            }
        };
        let spec = || {
            let mut spec = messages_spec(true);
            spec.prompt_tokens = counters.prompt;
            spec
        };
        let (handle, backend) = RecordingBackend::reservation();
        let events = streamed_messages(scripted().await, spec(), Some(handle)).await;
        assert_eq!(backend.settlements(), vec![expected], "{name}: streamed");
        let delta = events
            .iter()
            .rfind(|event| event["type"] == "message_delta")
            .expect("a message_delta event");
        assert_eq!(
            delta["usage"]["cache_read_input_tokens"], expected.cached_input_tokens,
            "{name}: the message_delta usage reports the engine's cache hit"
        );
        assert_eq!(delta["usage"]["cache_creation_input_tokens"], 0, "{name}");
        assert_eq!(
            delta["usage"]["input_tokens"],
            expected.actual_input_tokens - expected.cached_input_tokens,
            "{name}: the message_delta input_tokens is the prompt outside the cache"
        );
        let start = events
            .iter()
            .find(|event| event["type"] == "message_start")
            .expect("a message_start event");
        assert_eq!(
            start["message"]["usage"]["input_tokens"], expected.actual_input_tokens,
            "{name}: the skeleton reports the whole rendered prompt"
        );
        assert_eq!(
            start["message"]["usage"]["cache_read_input_tokens"], 0,
            "{name}: the skeleton precedes the engine's frames"
        );
        let (message, counts) = non_streamed_messages(scripted().await, spec()).await;
        assert_eq!(counts.settlement(), expected, "{name}: non-streamed");
        assert_eq!(
            message.usage.cache_read_input_tokens,
            Some(expected.cached_input_tokens),
            "{name}: the collected message reports the engine's cache hit"
        );
        assert_eq!(message.usage.cache_creation_input_tokens, Some(0), "{name}");
        assert_eq!(
            message.usage.input_tokens,
            expected.actual_input_tokens - expected.cached_input_tokens,
            "{name}: the collected message's input_tokens is the prompt outside the cache"
        );
    }
}

/// The token and generation families of a collected answer are recorded as
/// a streamed answer's are: the chat, completion, generate and messages
/// shapes each render their counts and one generation observation under
/// their own endpoint, a transcription (decoded through the generate
/// processor) under its own, the prompt and its cache hit included, as the
/// streamed generate answer counts them; and no TTFT or TPOT, which a
/// collected answer has no first token for.
#[test]
fn a_collected_answer_records_its_token_and_generation_families() {
    use crate::observability::metrics::test_support::render_with_recorder;

    let counters = Counters {
        prompt: 12,
        cached: 4,
        spec_accepted: 0,
        spec_drafted: 0,
    };
    let chat_request: ChatCompletionRequest =
        serde_json::from_value(json!({"model": "eof-test", "messages": [], "n": 1}))
            .expect("chat request");
    let completion_request: CompletionRequest =
        serde_json::from_value(json!({"model": "eof-test", "prompt": "hi", "n": 1}))
            .expect("completion request");
    let rt = runtime();
    let rendered = render_with_recorder(|| {
        rt.block_on(async {
            non_streamed_chat(
                engine_stream(frames(&["ab", "c"], 1, counters)).await,
                &chat_request,
            )
            .await;
            non_streamed_completion(
                engine_stream(frames(&["ok"], 1, counters)).await,
                &completion_request,
            )
            .await;
            non_streamed_generate(
                engine_stream(frames(&["abcd"], 1, counters)).await,
                metrics_labels::ENDPOINT_GENERATE,
            )
            .await;
            // A transcription decodes through the generate processor and
            // records under its own endpoint, the one its request count has.
            non_streamed_generate(
                engine_stream(frames(&["hello"], 1, counters)).await,
                metrics_labels::ENDPOINT_AUDIO_TRANSCRIPTIONS,
            )
            .await;
            non_streamed_messages(
                engine_stream(frames(&["hi"], 1, counters)).await,
                messages_spec(false),
            )
            .await;
        });
    });
    for expected in [
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"chat\",token_type=\"input\"} 12",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"chat\",token_type=\"cached\"} 4",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"chat\",token_type=\"output\"} 3",
        "smg_router_generation_duration_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"chat\"} 1",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"completions\",token_type=\"input\"} 12",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"completions\",token_type=\"cached\"} 4",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"completions\",token_type=\"output\"} 2",
        "smg_router_generation_duration_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"completions\"} 1",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"generate\",token_type=\"input\"} 12",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"generate\",token_type=\"cached\"} 4",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"generate\",token_type=\"output\"} 4",
        "smg_router_generation_duration_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"generate\"} 1",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"audio_transcriptions\",token_type=\"input\"} 12",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"audio_transcriptions\",token_type=\"cached\"} 4",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"audio_transcriptions\",token_type=\"output\"} 5",
        "smg_router_generation_duration_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"audio_transcriptions\"} 1",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"messages\",token_type=\"input\"} 12",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"messages\",token_type=\"cached\"} 4",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"messages\",token_type=\"output\"} 2",
        "smg_router_generation_duration_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"messages\"} 1",
    ] {
        assert!(
            rendered.lines().any(|line| line == expected),
            "missing {expected} in:\n{rendered}"
        );
    }
    for absent in [
        "smg_router_ttft_seconds",
        "smg_router_tpot_seconds",
        "smg_router_time_to_first_token_seconds",
        "smg_router_request_time_per_output_token_seconds",
    ] {
        assert!(
            !rendered.contains(absent),
            "unexpected {absent} in:\n{rendered}"
        );
    }
}

/// A collected Messages answer a Symphony table reads records the token and
/// generation families as one the legacy parsers read does: the engine's
/// counts under the messages endpoint and one generation observation, no
/// TTFT or TPOT (the whole-answer Symphony reading once returned before the
/// record was made).
#[test]
fn a_collected_symphony_message_records_its_token_and_generation_families() {
    use crate::observability::metrics::test_support::render_with_recorder;

    let counters = Counters {
        prompt: 12,
        cached: 4,
        spec_accepted: 0,
        spec_drafted: 0,
    };
    let rt = runtime();
    let rendered = render_with_recorder(|| {
        rt.block_on(async {
            let (_, counts) = non_streamed_messages(
                engine_stream(frames(&["hi"], 1, counters)).await,
                assigned_message(qwen3(CallSyntax::Json), "<|im_start|>assistant\n", false),
            )
            .await;
            assert_eq!(
                (counts.prompt, counts.cached, counts.completion),
                (12, 4, 2)
            );
        });
    });
    for expected in [
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"messages\",token_type=\"input\"} 12",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"messages\",token_type=\"cached\"} 4",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"messages\",token_type=\"output\"} 2",
        "smg_router_generation_duration_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"eof-test\",endpoint=\"messages\"} 1",
    ] {
        assert!(
            rendered.lines().any(|line| line == expected),
            "missing {expected} in:\n{rendered}"
        );
    }
    for absent in [
        "smg_router_ttft_seconds",
        "smg_router_tpot_seconds",
        "smg_router_time_to_first_token_seconds",
        "smg_router_request_time_per_output_token_seconds",
    ] {
        assert!(
            !rendered.contains(absent),
            "unexpected {absent} in:\n{rendered}"
        );
    }
}

/// An unwatched stream whose sender closed after the read: the first chunk
/// reached the client, the client dropped the response, and the engine's
/// `Complete` was read (nothing watched the client, so the read went on to
/// it). The terminal sends fail on the closed sender and the path returns
/// that failure, after the reservation was settled with the engine's
/// counts: nothing is closed as reserved-only.
#[tokio::test]
async fn a_chat_stream_whose_client_left_before_the_terminal_sends_settles_the_engines_counts() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "messages": [], "stream": true,
        "stream_options": {"include_usage": true},
    }))
    .expect("chat request");
    let (release, gate) = oneshot::channel();
    let mut frames = choice_frames(0, &["hi"], SETTLED);
    let complete = frames.pop().expect("the terminal frame");
    let (stream, server) = gated_stream(frames, gate, vec![complete], "0").await;
    let (tx, mut rx) = sse_channel();
    let (handle, backend) = RecordingBackend::reservation();
    let processor = processor(false);
    let streamed = processor.process_streaming_chunks(
        stream,
        dispatch(),
        tokenizer(),
        (None, None, false, false, false),
        ChatResponseSpec::from(&request),
        &tx,
        Some(handle),
    );
    let client = async {
        let first = rx.recv().await.expect("the first chunk reaches the client");
        assert!(first.is_ok(), "{first:?}");
        drop(rx);
        release.send(()).expect("the gate is waited on");
    };
    let (result, ()) = tokio::join!(streamed, client);
    server.abort();
    assert!(
        result.is_err(),
        "the terminal sends fail on the closed sender: {result:?}"
    );
    assert_eq!(
        backend.settlements(),
        vec![UsageSettlement {
            actual_input_tokens: SETTLED.prompt,
            completion_tokens: 2,
            cached_input_tokens: SETTLED.cached,
            reasoning_tokens: 0,
        }]
    );
    assert_eq!(backend.closes(), 0);
}

/// The same for the Messages path: its terminal sends fail on the closed
/// sender after the engine's counts were settled.
#[tokio::test]
async fn a_messages_stream_whose_client_left_before_the_terminal_sends_settles_the_engines_counts()
{
    let (release, gate) = oneshot::channel();
    let mut frames = choice_frames(0, &["hi"], SETTLED);
    let complete = frames.pop().expect("the terminal frame");
    let (stream, server) = gated_stream(frames, gate, vec![complete], "0").await;
    let (tx, mut rx) = sse_channel();
    let (handle, backend) = RecordingBackend::reservation();
    let processor = processor(false);
    let streamed = processor.process_messages_streaming_chunks(
        stream,
        dispatch(),
        tokenizer(),
        (None, None, false, false, false),
        messages_spec(false),
        &tx,
        Some(handle),
    );
    let client = async {
        // message_start, then the content block's start and the delta.
        let mut text = String::new();
        while !text.contains("content_block_delta") {
            let frame = rx.recv().await.expect("the first delta reaches the client");
            text.push_str(std::str::from_utf8(&frame.expect("an SSE write")).expect("UTF-8"));
        }
        drop(rx);
        release.send(()).expect("the gate is waited on");
    };
    let (result, ()) = tokio::join!(streamed, client);
    server.abort();
    assert!(
        result.is_err(),
        "the terminal sends fail on the closed sender: {result:?}"
    );
    assert_eq!(
        backend.settlements(),
        vec![UsageSettlement {
            actual_input_tokens: SETTLED.prompt,
            completion_tokens: 2,
            cached_input_tokens: SETTLED.cached,
            reasoning_tokens: 0,
        }]
    );
    assert_eq!(backend.closes(), 0);
}

/// The same two paths over the stream as production reads it, watched for
/// the client's leaving (`until_client_gone`): the client's drop ends the
/// read before the engine's `Complete`, so the path sends nothing more and
/// returns `Ok`, and the reservation, with no terminal frame read, is closed
/// as reserved-only, not settled.
#[tokio::test]
async fn a_watched_stream_whose_client_leaves_sends_nothing_more_and_closes_its_reservation() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "messages": [], "stream": true,
        "stream_options": {"include_usage": true},
    }))
    .expect("chat request");
    for messages in [false, true] {
        let (release, gate) = oneshot::channel();
        let mut frames = choice_frames(0, &["hi"], SETTLED);
        let complete = frames.pop().expect("the terminal frame");
        let (stream, server) = gated_stream(frames, gate, vec![complete], "0").await;
        let (tx, mut rx) = sse_channel();
        let stream = stream.until_client_gone(&tx, "chatcmpl-eof");
        let (handle, backend) = RecordingBackend::reservation();
        let processor = processor(false);
        let streamed = async {
            if messages {
                processor
                    .process_messages_streaming_chunks(
                        stream,
                        dispatch(),
                        tokenizer(),
                        (None, None, false, false, false),
                        messages_spec(false),
                        &tx,
                        Some(handle),
                    )
                    .await
            } else {
                processor
                    .process_streaming_chunks(
                        stream,
                        dispatch(),
                        tokenizer(),
                        (None, None, false, false, false),
                        ChatResponseSpec::from(&request),
                        &tx,
                        Some(handle),
                    )
                    .await
            }
        };
        let client = async {
            let mut text = String::new();
            while !text.contains("hi") {
                let frame = rx.recv().await.expect("the first delta reaches the client");
                text.push_str(std::str::from_utf8(&frame.expect("an SSE write")).expect("UTF-8"));
            }
            drop(rx);
            release.send(()).expect("the gate is waited on");
        };
        let (result, ()) = tokio::join!(streamed, client);
        server.abort();
        assert!(result.is_ok(), "messages {messages}: {result:?}");
        assert!(
            backend.settlements().is_empty(),
            "messages {messages}: no terminal frame was read: {:?}",
            backend.settlements()
        );
        assert_eq!(backend.closes(), 1, "messages {messages}");
    }
}

/// The watched stream again, with text held back at the departure: the stop
/// decoder keeps `hi` as the start of a possible `hiya`, so nothing of it was
/// sent when the client drops. The departure ends the read before the
/// decoder's end-of-stream flush (which would fail on the closed sender), so
/// the path returns `Ok` and closes the reservation as reserved-only, on
/// both paths.
#[tokio::test]
async fn a_watched_stream_whose_client_leaves_with_held_text_ends_its_read_at_the_departure() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "messages": [], "stream": true,
        "stream_options": {"include_usage": true},
    }))
    .expect("chat request");
    for messages in [false, true] {
        let (release, gate) = oneshot::channel();
        let mut frames = choice_frames(0, &["x", "hi"], SETTLED);
        let complete = frames.pop().expect("the terminal frame");
        let (stream, server) = gated_stream(frames, gate, vec![complete], "0").await;
        let (tx, mut rx) = sse_channel();
        let stream = stream.until_client_gone(&tx, "chatcmpl-eof");
        let (handle, backend) = RecordingBackend::reservation();
        let processor = processor(false);
        let stop = Some(StringOrArray::Array(vec!["hiya".to_string()]));
        let streamed = async {
            if messages {
                processor
                    .process_messages_streaming_chunks(
                        stream,
                        dispatch(),
                        tokenizer(),
                        (stop, None, false, false, false),
                        messages_spec(false),
                        &tx,
                        Some(handle),
                    )
                    .await
            } else {
                processor
                    .process_streaming_chunks(
                        stream,
                        dispatch(),
                        tokenizer(),
                        (stop, None, false, false, false),
                        ChatResponseSpec::from(&request),
                        &tx,
                        Some(handle),
                    )
                    .await
            }
        };
        // The delta's own payload, not a character the frame envelope carries
        // too: a Messages content block opens with `"type":"text"`.
        let delta = if messages {
            r#""text":"x""#
        } else {
            r#""content":"x""#
        };
        let client = async {
            let mut text = String::new();
            while !text.contains(delta) {
                let frame = rx.recv().await.expect("the first delta reaches the client");
                text.push_str(std::str::from_utf8(&frame.expect("an SSE write")).expect("UTF-8"));
            }
            assert!(
                !text.contains("hi"),
                "messages {messages}: `hi` is held back: {text}"
            );
            drop(rx);
            release.send(()).expect("the gate is waited on");
        };
        let (result, ()) = tokio::join!(streamed, client);
        server.abort();
        assert!(result.is_ok(), "messages {messages}: {result:?}");
        assert!(
            backend.settlements().is_empty(),
            "messages {messages}: no terminal frame was read: {:?}",
            backend.settlements()
        );
        assert_eq!(backend.closes(), 1, "messages {messages}");
    }
}

/// The completions processor on the same departure: the read ends at the
/// client's leaving and the open choices are not finished (the held `hi`
/// would go to the closed sender), so the path returns `Ok` as the chat and
/// Messages paths do.
#[tokio::test]
async fn a_watched_completion_stream_whose_client_leaves_with_held_text_ends_at_the_departure() {
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "prompt": "p", "stream": true,
    }))
    .expect("completion request");
    let (release, gate) = oneshot::channel();
    let mut frames = choice_frames(0, &["x", "hi"], SETTLED);
    let complete = frames.pop().expect("the terminal frame");
    let (stream, server) = gated_stream(frames, gate, vec![complete], "0").await;
    let (tx, mut rx) = sse_channel();
    let stream = stream.until_client_gone(&tx, "cmpl-eof");
    let processor = processor(false);
    let stop = Some(StringOrArray::Array(vec!["hiya".to_string()]));
    let spec = CompletionResponseSpec::new(&request, Vec::new());
    let streamed = processor.process_completion_streaming_chunks(
        stream,
        dispatch(),
        tokenizer(),
        (stop, None, false, false, false),
        &spec,
        "p",
        0,
        &tx,
        None,
    );
    let client = async {
        let mut text = String::new();
        // The delta's own payload: every completion frame's envelope carries
        // an `x` (`"object":"text_completion"`, the `"text"` key).
        while !text.contains(r#""text":"x""#) {
            let frame = rx.recv().await.expect("the first delta reaches the client");
            text.push_str(std::str::from_utf8(&frame.expect("an SSE write")).expect("UTF-8"));
        }
        assert!(!text.contains("hi"), "`hi` is held back: {text}");
        drop(rx);
        release.send(()).expect("the gate is waited on");
    };
    let (result, ()) = tokio::join!(streamed, client);
    server.abort();
    assert!(result.is_ok(), "{:?}", result.err());
}

#[tokio::test]
async fn spec_usage_regression_streamed_and_nonstreamed_formatting() {
    for (first, second, rejected) in [
        ((2, 0), (1, 10), None),
        ((0, 0), (0, 10), None),
        ((0, 7), (0, 0), None),
        ((0, 7), (0, 10), Some(17)),
        ((4, 2), (1, 10), None),
        ((0, 0), (1, 10), None),
        ((2, 3), (1, 10), Some(10)),
        ((2, 2), (1, 1), Some(0)),
        ((0, 0), (0, 0), Some(0)),
    ] {
        let counters = |(a, d)| Counters {
            prompt: 10,
            cached: 4,
            spec_accepted: a,
            spec_drafted: d,
        };
        let mut data = choice_frames(0, &["ok"], counters(first));
        data.extend(choice_frames(1, &["ok"], counters(second)));
        let chat = |stream| {
            serde_json::from_value(json!({
                "model": "eof-test", "messages": [], "n": 2, "stream": stream,
                "stream_options": {"include_usage": true}
            }))
            .unwrap()
        };
        let completion = |stream| {
            serde_json::from_value(json!({
                "model": "eof-test", "prompt": "hi", "n": 2, "stream": stream,
                "stream_options": {"include_usage": true}
            }))
            .unwrap()
        };
        let chat_stream =
            streamed_chat_usage(engine_stream(data.clone()).await, &chat(true), None).await;
        let chat_whole =
            non_streamed_chat_usage(engine_stream(data.clone()).await, &chat(false)).await;
        let completion_stream =
            streamed_completion_usage(engine_stream(data.clone()).await, &completion(true), None)
                .await;
        let completion_whole =
            non_streamed_completion_usage(engine_stream(data).await, &completion(false)).await;
        assert_eq!(chat_stream, chat_whole);
        assert_eq!(completion_stream, completion_whole);
        for (usage, defaults) in [(chat_stream, true), (completion_stream, false)] {
            let details = &usage["completion_tokens_details"];
            if !defaults && first == (0, 0) && second == (0, 0) {
                assert!(
                    details.is_null(),
                    "legacy completion has no speculative measurements: {usage}"
                );
            } else {
                assert_eq!(details["accepted_prediction_tokens"], first.0 + second.0);
                match rejected {
                    Some(n) => assert_eq!(details["rejected_prediction_tokens"], n),
                    None => assert!(
                        details.get("rejected_prediction_tokens").is_none(),
                        "{usage}"
                    ),
                }
            }
        }
    }
}

#[test]
fn mixed_stream_snapshot_preserves_unknown_rejected() {
    let mut usage = ChatStreamUsage::default();
    usage.choices.insert(
        0,
        ChatStreamTokenCounts {
            spec_accepted: 2,
            spec_drafted: 0,
            ..Default::default()
        },
    );
    usage.choices.insert(
        1,
        ChatStreamTokenCounts {
            spec_accepted: 1,
            spec_drafted: 10,
            ..Default::default()
        },
    );
    let json = serde_json::to_value(usage.snapshot(0)).unwrap();
    assert_eq!(
        json["completion_tokens_details"]["accepted_prediction_tokens"],
        3
    );
    assert!(
        json["completion_tokens_details"]
            .get("rejected_prediction_tokens")
            .is_none(),
        "{json}"
    );
}

#[tokio::test]
async fn continuous_spec_usage_waits_for_unseen_choice() {
    let counters = Counters {
        prompt: 10,
        cached: 4,
        spec_accepted: 2,
        spec_drafted: 2,
    };
    let mut data = choice_frames(0, &["ok"], counters);
    data.extend(choice_frames(1, &["ok"], counters));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "messages": [], "n": 2, "stream": true,
        "stream_options": {"include_usage": true, "continuous_usage_stats": true}
    }))
    .unwrap();
    let (stream, server) = engine_stream(data).await;
    let (tx, rx) = sse_channel();
    let result = processor(false)
        .process_streaming_chunks(
            stream,
            dispatch(),
            tokenizer(),
            (
                Some(StringOrArray::String("okay".to_string())),
                None,
                false,
                false,
                false,
            ),
            ChatResponseSpec::from(&request),
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    let intermediate: Vec<_> = events
        .iter()
        .filter(|event| {
            event["usage"]["completion_tokens_details"]["accepted_prediction_tokens"] == 2
        })
        .collect();
    assert!(!intermediate.is_empty(), "{events:?}");
    for event in intermediate {
        let usage = &event["usage"];
        assert_eq!(usage["prompt_tokens"], 10);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 4);
        assert!(
            usage["completion_tokens_details"]
                .get("rejected_prediction_tokens")
                .is_none(),
            "unseen choice must remain unknown: {event}"
        );
    }
    let final_usage = usage_chunk(&events);
    assert_eq!(
        final_usage["completion_tokens_details"]["accepted_prediction_tokens"],
        4
    );
    assert_eq!(
        final_usage["completion_tokens_details"]["rejected_prediction_tokens"],
        0
    );
}

#[test]
fn continuous_spec_usage_terminal_coverage_controls() {
    for (first, second, rejected) in [
        ((0, 0), (0, 0), Some(0)),
        ((2, 2), (1, 1), Some(0)),
        ((2, 3), (1, 10), Some(10)),
        ((2, 0), (1, 10), None),
        ((0, 7), (0, 0), None),
        ((4, 2), (1, 10), None),
    ] {
        let complete = |index, (accepted, drafted)| {
            ProtoGenerateComplete::Vllm(Box::new(proto::GenerateComplete {
                index,
                prompt_tokens: 10,
                cached_tokens: 4,
                spec_accepted_tokens: accepted,
                spec_draft_tokens: drafted,
                ..Default::default()
            }))
        };
        let snapshot =
            |tracker: &ChatStreamUsage| serde_json::to_value(tracker.snapshot(3)).unwrap();
        let mut tracker = ChatStreamUsage::new(2);
        tracker.reasoning_parsed = true;
        tracker.record_reasoning(0, 3);
        tracker.record_complete(&complete(0, first));
        // A duplicate terminal for choice 0 cannot stand in for choice 1.
        tracker.record_complete(&complete(0, first));
        let usage = snapshot(&tracker);
        assert_eq!(usage["prompt_tokens"], 7);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 3);
        assert_eq!(
            usage["completion_tokens_details"]["accepted_prediction_tokens"],
            first.0
        );
        if first == (0, 0) {
            assert_eq!(
                usage["completion_tokens_details"]["rejected_prediction_tokens"],
                0
            );
        } else {
            assert!(
                usage["completion_tokens_details"]
                    .get("rejected_prediction_tokens")
                    .is_none(),
                "{usage}"
            );
        }
        tracker.record_chunk(&ProtoGenerateStreamChunk::Vllm(
            proto::GenerateStreamChunk {
                index: 1,
                token_ids: vec![1, 2],
                ..Default::default()
            },
        ));
        let usage = snapshot(&tracker);
        assert_eq!(usage["completion_tokens"], 2);
        if first == (0, 0) {
            assert_eq!(
                usage["completion_tokens_details"]["rejected_prediction_tokens"],
                0
            );
        } else {
            assert!(
                usage["completion_tokens_details"]
                    .get("rejected_prediction_tokens")
                    .is_none(),
                "{usage}"
            );
        }
        tracker.record_complete(&complete(1, second));
        let usage = snapshot(&tracker);
        assert_eq!(usage["completion_tokens"], 2);
        assert_eq!(
            usage["completion_tokens_details"]["accepted_prediction_tokens"],
            first.0 + second.0
        );
        match rejected {
            Some(value) => assert_eq!(
                usage["completion_tokens_details"]["rejected_prediction_tokens"],
                value
            ),
            None => assert!(
                usage["completion_tokens_details"]
                    .get("rejected_prediction_tokens")
                    .is_none(),
                "{usage}"
            ),
        }
    }
}

/// Clean EOF may return the completed subset, but cannot certify rejections
/// for the requested choice that never supplied its terminal counters.
#[tokio::test]
async fn collected_chat_missing_choice_keeps_rejections_unknown() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "eof-test", "messages": [], "n": 2,
        "separate_reasoning": false,
    }))
    .unwrap();
    let counters = Counters {
        prompt: 10,
        cached: 4,
        spec_accepted: 2,
        spec_drafted: 2,
    };
    let (response, counts) = non_streamed_chat(
        scheduler_stream(scheduler_choice_frames(0, &["ok"], counters, 3)).await,
        &request,
    )
    .await;
    assert_eq!(
        response.choices.len(),
        1,
        "preserve the partial answer policy"
    );
    assert_eq!(
        counts.settlement(),
        UsageSettlement {
            actual_input_tokens: 10,
            completion_tokens: 2,
            cached_input_tokens: 4,
            reasoning_tokens: 3,
        }
    );
    let usage = serde_json::to_value(response).unwrap()["usage"].clone();
    assert_eq!(
        usage["completion_tokens_details"]["accepted_prediction_tokens"],
        2
    );
    assert!(
        usage["completion_tokens_details"]
            .get("rejected_prediction_tokens")
            .is_none(),
        "{usage}"
    );
}

#[tokio::test]
async fn collected_completion_missing_choice_in_one_prompt_keeps_rejections_unknown() {
    for (first_indices, second_indices) in [
        (&[1, 0][..], &[0][..]),
        (&[1, 0][..], &[1, 0][..]),
        (&[1, 0][..], &[0, 0][..]),
        (&[0][..], &[1, 0][..]),
    ] {
        let request: CompletionRequest = serde_json::from_value(json!({
            "model": "eof-test", "prompt": ["first", "second"], "n": 2,
        }))
        .unwrap();
        let counters = Counters {
            prompt: 10,
            cached: 4,
            spec_accepted: 2,
            spec_drafted: 2,
        };
        let first: Vec<_> = first_indices
            .iter()
            .flat_map(|&index| scheduler_choice_frames(index, &["ok"], counters, 3))
            .collect();
        let mut results = Vec::new();
        let mut servers = Vec::new();
        let second: Vec<_> = second_indices
            .iter()
            .flat_map(|&index| scheduler_choice_frames(index, &["ok"], counters, 3))
            .collect();
        for frames in [first, second] {
            let (stream, server) = scheduler_stream(frames).await;
            results.push(context::ExecutionResult::Single {
                stream,
                dispatch_start: Instant::now(),
            });
            servers.push(server);
        }
        let tokenizer = tokenizer();
        let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
        let response = response_processor()
            .process_non_streaming_completion_response(
                context::ExecutionResult::Batch { results },
                CompletionResponseSpec::new(&request, Vec::new()),
                dispatch(),
                &tokenizer,
                &mut decoder,
            )
            .await
            .unwrap_or_else(|response| panic!("{}", response.status()));
        for server in servers {
            server.abort();
        }
        let count = first_indices.len() + second_indices.len();
        assert_eq!(
            response.choices.iter().map(|c| c.index).collect::<Vec<_>>(),
            (0..first_indices.len() as u32)
                .chain((0..second_indices.len() as u32).map(|i| 2 + i))
                .collect::<Vec<_>>()
        );
        let usage = serde_json::to_value(response).unwrap()["usage"].clone();
        assert_eq!(usage["prompt_tokens"], 20);
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 8);
        assert_eq!(usage["completion_tokens"], 2 * count);
        assert_eq!(
            usage["completion_tokens_details"]["reasoning_tokens"],
            3 * count
        );
        assert_eq!(
            usage["completion_tokens_details"]["accepted_prediction_tokens"],
            2 * count
        );
        let rejected = usage["completion_tokens_details"].get("rejected_prediction_tokens");
        if first_indices == [1, 0] && second_indices == [1, 0] {
            assert_eq!(rejected, Some(&json!(0)));
        } else {
            assert!(rejected.is_none(), "{usage}");
        }
    }
}

/// Compatibility controls drive both actual collected consumers, including
/// ambiguous legacy duplicate indices and scalar zero proposal sentinels.
#[tokio::test]
async fn collected_regular_membership_and_proposal_controls() {
    let cases = [
        (Some(2), &[(0, 0, 0)][..], None),
        (Some(2), &[(0, 0, 10)][..], None),
        (Some(2), &[(0, 2, 2), (0, 2, 2)][..], None),
        (Some(2), &[(0, 2, 2), (2, 2, 2)][..], None),
        (Some(2), &[(1, 2, 2), (0, 2, 2)][..], Some(0)),
        (Some(2), &[(1, 1, 4), (0, 2, 5)][..], Some(6)),
        (Some(2), &[(0, 0, 0), (1, 0, 0)][..], Some(0)),
        (Some(2), &[(0, 2, 0), (1, 1, 10)][..], None),
        (Some(2), &[(0, 0, 0), (1, 0, 10)][..], None),
        (Some(2), &[(0, 4, 2), (1, 0, 10)][..], None),
        (Some(1), &[(0, 0, 5)][..], Some(5)),
        (None, &[(0, 2, 2)][..], Some(0)),
        (Some(0), &[(0, 2, 2)][..], Some(0)),
    ];
    for (n, entries, rejected) in cases {
        let frames: Vec<_> = entries
            .iter()
            .flat_map(|&(index, accepted, drafted)| {
                scheduler_choice_frames(
                    index,
                    &["ok"],
                    Counters {
                        prompt: 10,
                        cached: 4,
                        spec_accepted: accepted,
                        spec_drafted: drafted,
                    },
                    3,
                )
            })
            .collect();
        let chat: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "eof-test", "messages": [], "n": n, "separate_reasoning": false,
        }))
        .unwrap();
        let completion: CompletionRequest = serde_json::from_value(json!({
            "model": "eof-test", "prompt": "hi", "n": n,
        }))
        .unwrap();
        let chat_usage =
            non_streamed_chat_usage(scheduler_stream(frames.clone()).await, &chat).await;
        let completion_usage =
            non_streamed_completion_usage(scheduler_stream(frames).await, &completion).await;
        for (is_chat, usage) in [(true, chat_usage), (false, completion_usage)] {
            assert_eq!(usage["prompt_tokens"], 10);
            assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 4);
            assert_eq!(usage["completion_tokens"], 2 * entries.len());
            let details = &usage["completion_tokens_details"];
            assert_eq!(details["reasoning_tokens"], 3 * entries.len());
            if !is_chat && rejected == Some(0) && entries.iter().all(|e| e.1 == 0 && e.2 == 0) {
                // Legacy completions do not add chat's all-zero schema defaults.
                assert!(
                    details.get("accepted_prediction_tokens").is_none(),
                    "{usage}"
                );
                assert!(
                    details.get("rejected_prediction_tokens").is_none(),
                    "{usage}"
                );
            } else {
                assert_eq!(
                    details["accepted_prediction_tokens"],
                    entries.iter().map(|e| e.1).sum::<u32>()
                );
                assert_eq!(
                    details.get("rejected_prediction_tokens"),
                    rejected.map(|v| json!(v)).as_ref(),
                    "n={n:?}, {entries:?}: {usage}"
                );
            }
        }
    }
}
