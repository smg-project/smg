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
use tokio::task::JoinHandle;

use super::{
    eof_tests::{
        chunk, collect_events, data_frame, decoded_stream, mock_worker_endpoint, processor,
        scripted_stream, trailers_frame, CharacterTokenizer,
    },
    *,
};
use crate::{
    rate_limit::test_support::RecordingBackend,
    routers::grpc::{pipeline::RequestPipeline, regular::processor::ResponseProcessor},
    worker::WorkerRegistry,
};

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
            context::ExecutionResult::Single { stream },
            CompletionResponseSpec::from(request),
            dispatch,
            tokenizer(),
            reservation,
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
            context::ExecutionResult::Single { stream },
            CompletionResponseSpec::from(request),
            dispatch(),
            tokenizer,
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

async fn non_streamed_generate((stream, server): Scripted) -> Vec<GenerateResponse> {
    let tokenizer = tokenizer();
    let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
    let responses = response_processor()
        .process_non_streaming_generate_response(
            context::ExecutionResult::Single { stream },
            dispatch(),
            &mut decoder,
            false,
            Instant::now(),
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
    server.abort();
    responses
}

/// A Messages request with thinking on or off, no tools.
fn messages_spec(thinking: bool) -> MessagesResponseSpec {
    MessagesResponseSpec {
        thinking_requested: thinking,
        thinking_text_returned: true,
        reasoning_in_prefill: false,
        tool_choice: None,
        has_tools: false,
        history_tool_calls_count: 0,
        chat_tools: Vec::new().into(),
        stop_sequences: None,
        symphony: None,
        first_tool_call_only: false,
    }
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
            context::ExecutionResult::Single { stream },
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
    for (name, scheduler_reasoning, counters, expected) in [
        (
            "the engine's counts",
            Some(3),
            SETTLED,
            UsageSettlement {
                actual_input_tokens: SETTLED.prompt,
                completion_tokens: 2 * choices,
                cached_input_tokens: SETTLED.cached,
                reasoning_tokens: 3 * choices,
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
                completion_tokens: 2 * choices,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
        ),
    ] {
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
    for (name, scheduler_reasoning, counters, expected) in [
        (
            "the engine's counts",
            Some(3),
            SETTLED,
            UsageSettlement {
                actual_input_tokens: SETTLED.prompt,
                completion_tokens: 2 * choices,
                cached_input_tokens: SETTLED.cached,
                reasoning_tokens: 3 * choices,
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
                completion_tokens: 2 * choices,
                cached_input_tokens: 0,
                reasoning_tokens: 0,
            },
        ),
    ] {
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
        let responses = non_streamed_generate(scripted().await).await;
        assert_eq!(
            RequestPipeline::generate_settlement(&responses),
            expected,
            "{name}: non-streamed"
        );
    }
}

/// The settlement of a Messages request carries the engine's counts, streamed
/// and not, though the Messages usage reports no cache hits and the greater
/// of the engine's and the parser's thinking count: the engine's cache hit
/// and its own thinking count, zero where its frames carry none.
#[tokio::test]
async fn messages_settlement_carries_the_engines_counts_streamed_and_not() {
    let texts = ["<think>", "ab", "</think>", "de"];
    let completion = texts.concat().chars().count() as u32;
    for (name, scheduler_reasoning, counters, expected) in [
        (
            "the engine's counts",
            Some(1),
            SETTLED,
            UsageSettlement {
                actual_input_tokens: SETTLED.prompt,
                completion_tokens: completion,
                cached_input_tokens: SETTLED.cached,
                reasoning_tokens: 1,
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
    ] {
        let scripted = || async {
            match scheduler_reasoning {
                Some(reasoning) => {
                    scheduler_stream(scheduler_frames(&texts, 1, counters, reasoning)).await
                }
                None => engine_stream(frames(&texts, 1, counters)).await,
            }
        };
        let (handle, backend) = RecordingBackend::reservation();
        let events = streamed_messages(scripted().await, messages_spec(true), Some(handle)).await;
        assert_eq!(backend.settlements(), vec![expected], "{name}: streamed");
        let delta = events
            .iter()
            .rfind(|event| event["type"] == "message_delta")
            .expect("a message_delta event");
        assert_eq!(
            delta["usage"]["cache_read_input_tokens"], 0,
            "{name}: the Messages usage reports no cache hits"
        );
        let (message, counts) = non_streamed_messages(scripted().await, messages_spec(true)).await;
        assert_eq!(counts.settlement(), expected, "{name}: non-streamed");
        assert_eq!(message.usage.cache_read_input_tokens, Some(0), "{name}");
    }
}
