//! Test gateway parsers and SSE events with exact gRPC frames.

use std::time::Duration;

use axum::http::{HeaderMap, HeaderValue, StatusCode};
use bytes::BufMut;
use futures::{stream, StreamExt as _};
use http_body::Frame;
use http_body_util::StreamBody;
use llm_tokenizer::{traits::Encoding, SpecialTokens};
use openai_protocol::chat::ChatCompletionRequest;
use prost::Message as ProstMessage;
use smg_grpc_client::vllm_engine::{
    proto, proto::generate_response::Response as GenerationEvent, AbortOnDropStream,
    VllmEngineClient,
};
use tokio::{net::TcpListener, sync::oneshot, task::JoinHandle, time::timeout};
use tonic::codec::Codec;
use tool_parser::types::ToolCallItem;

use super::*;
use crate::{
    routers::{common::sse::SseReceiver, grpc::regular::processor::ResponseProcessor},
    worker::WorkerRegistry,
};

#[derive(Default)]
pub(super) struct CharacterTokenizer {
    special_tokens: SpecialTokens,
}

impl llm_tokenizer::traits::Encoder for CharacterTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
        Ok(Encoding::Plain(text.chars().map(u32::from).collect()))
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }
}

impl llm_tokenizer::traits::Decoder for CharacterTokenizer {
    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> anyhow::Result<String> {
        ids.iter()
            .map(|id| char::from_u32(*id).ok_or_else(|| anyhow::anyhow!("invalid character token")))
            .collect()
    }
}

impl Tokenizer for CharacterTokenizer {
    fn vocab_size(&self) -> usize {
        128
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        token.chars().next().map(u32::from)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        char::from_u32(id).map(|character| character.to_string())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(super) fn chunk(index: u32, text: &str) -> proto::GenerateResponse {
    let token_ids: Vec<_> = text.chars().map(u32::from).collect();
    proto::GenerateResponse {
        response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
            index,
            completion_tokens: token_ids.len() as u32,
            token_ids,
            ..Default::default()
        })),
    }
}

/// A chunk that also reports one logprob per token.
pub(super) fn chunk_with_logprobs(index: u32, text: &str) -> proto::GenerateResponse {
    let mut response = chunk(index, text);
    if let Some(GenerationEvent::Chunk(chunk)) = &mut response.response {
        chunk.output_logprobs = Some(proto::OutputLogProbs {
            token_logprobs: vec![-0.5; chunk.token_ids.len()],
            token_ids: chunk.token_ids.clone(),
            top_logprobs: Vec::new(),
        });
    }
    response
}

pub(super) fn complete(index: u32, reason: &str) -> proto::GenerateResponse {
    complete_with_prompt(index, reason, 1)
}

pub(super) fn complete_with_prompt(
    index: u32,
    reason: &str,
    prompt_tokens: u32,
) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index,
            finish_reason: reason.to_string(),
            prompt_tokens,
            completion_tokens: 12,
            ..Default::default()
        })),
    }
}

/// The mock server accepts the client connection. This test supplies the
/// response frames and controls EOF.
pub(super) async fn scripted_stream(
    responses: Vec<proto::GenerateResponse>,
    grpc_status: &'static str,
) -> (ProtoStream, JoinHandle<()>) {
    let frames: Vec<Result<Frame<Bytes>, tonic::Status>> = responses
        .iter()
        .map(|response| Ok(data_frame(response)))
        .chain(std::iter::once(Ok(trailers_frame(grpc_status))))
        .collect();
    let (client, server) = mock_client().await;
    let stream = response_stream(stream::iter(frames), client);
    (stream, server)
}

/// [`scripted_stream`] with the frames after `first` held back until `gate`
/// fires: the processor sees the first engine output on its own.
pub(super) async fn gated_stream(
    first: Vec<proto::GenerateResponse>,
    gate: oneshot::Receiver<()>,
    rest: Vec<proto::GenerateResponse>,
    grpc_status: &'static str,
) -> (ProtoStream, JoinHandle<()>) {
    let head: Vec<Result<Frame<Bytes>, tonic::Status>> = first
        .iter()
        .map(|response| Ok(data_frame(response)))
        .collect();
    let tail: Vec<Result<Frame<Bytes>, tonic::Status>> = rest
        .iter()
        .map(|response| Ok(data_frame(response)))
        .chain(std::iter::once(Ok(trailers_frame(grpc_status))))
        .collect();
    let (client, server) = mock_client().await;
    let gated = stream::once(async move {
        let _ = gate.await;
        stream::iter(tail)
    })
    .flatten();
    let stream = response_stream(stream::iter(head).chain(gated), client);
    (stream, server)
}

/// One length-prefixed gRPC data frame carrying `response`.
fn data_frame(response: &proto::GenerateResponse) -> Frame<Bytes> {
    let encoded = response.encode_to_vec();
    let mut frame = Vec::with_capacity(encoded.len() + 5);
    frame.put_u8(0);
    frame.put_u32(encoded.len() as u32);
    frame.extend_from_slice(&encoded);
    Frame::data(Bytes::from(frame))
}

/// The trailers frame that ends a gRPC response with `grpc_status`.
fn trailers_frame(grpc_status: &'static str) -> Frame<Bytes> {
    let mut trailers = HeaderMap::new();
    trailers.insert("grpc-status", HeaderValue::from_static(grpc_status));
    Frame::trailers(trailers)
}

/// A decoded response stream over `body`, owned by `client` as a worker
/// stream that was never dispatched (nothing to cancel on drop).
fn response_stream<S>(body: S, client: VllmEngineClient) -> ProtoStream
where
    S: futures::Stream<Item = Result<Frame<Bytes>, tonic::Status>> + Send + 'static,
{
    let mut codec =
        tonic_prost::ProstCodec::<proto::GenerateResponse, proto::GenerateResponse>::default();
    let stream = tonic::Streaming::new_response(
        codec.decoder(),
        StreamBody::new(body),
        StatusCode::OK,
        None,
        None,
    );
    let stream = AbortOnDropStream::new(stream, "eof-test".to_string(), client);
    // No generation was sent to the mock, so there is nothing to cancel.
    stream.mark_completed();
    ProtoStream::Vllm(stream)
}

/// The mock worker a test's client connects to. The test supplies the
/// response frames itself; the server only accepts the connection.
#[expect(
    clippy::disallowed_methods,
    reason = "bounded test fixture; server task is explicitly aborted"
)]
async fn mock_client() -> (VllmEngineClient, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock worker");
    let port = listener.local_addr().expect("mock worker address").port();
    let config = Arc::new(mock_worker::config::Config {
        admin_port: None,
        context_length: 32768,
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: "eof-test".to_string(),
        tokenizer_path: "eof-test".to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 0,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
        ..mock_worker::config::Config::default()
    });
    let server = tokio::spawn(mock_worker::grpc::serve_with_listener(config, listener));
    let client = VllmEngineClient::connect(&format!("http://127.0.0.1:{port}"))
        .await
        .expect("connect mock worker");
    (client, server)
}

pub(super) fn processor(with_tools: bool) -> StreamingProcessor {
    StreamingProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::new(
            Arc::new(WorkerRegistry::new()),
            with_tools.then(|| "json".to_string()),
            Some("qwen3".to_string()),
        ),
        "vllm",
    )
}

fn dispatch() -> context::DispatchMetadata {
    context::DispatchMetadata {
        request_id: "chatcmpl-eof".to_string(),
        model: "eof-test".to_string(),
        created: 1,
        weight_version: None,
    }
}

fn chat_spec(with_tools: bool) -> ChatResponseSpec {
    chat_spec_with(with_tools, None)
}

/// [`chat_spec`] with a `tool_choice` (`"required"`, a named function, ...).
fn chat_spec_with(with_tools: bool, tool_choice: Option<Value>) -> ChatResponseSpec {
    let mut request = serde_json::json!({
        "model": "eof-test", "messages": [], "stream": true,
        "separate_reasoning": true, "n": 2
    });
    if with_tools {
        request["tools"] = serde_json::json!([{
            "type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object", "properties": {}}
            }
        }]);
    }
    if let Some(tool_choice) = tool_choice {
        request["tool_choice"] = tool_choice;
    }
    ChatResponseSpec::from(
        &serde_json::from_value::<ChatCompletionRequest>(request).expect("chat request"),
    )
}

async fn collect_events(mut rx: SseReceiver) -> Vec<Value> {
    let mut events = Vec::new();
    while let Some(frame) = rx.recv().await {
        let frame = frame.expect("successful SSE write");
        let text = std::str::from_utf8(&frame).expect("UTF-8 SSE");
        for line in text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                events.push(serde_json::from_str(data).expect("SSE JSON"));
            }
        }
    }
    events
}

async fn chat_events(
    responses: Vec<proto::GenerateResponse>,
    with_tools: bool,
    grpc_status: &'static str,
) -> (Result<(), String>, Vec<Value>) {
    chat_events_with_spec(responses, with_tools, chat_spec(with_tools), grpc_status).await
}

async fn chat_events_with_spec(
    responses: Vec<proto::GenerateResponse>,
    with_tools: bool,
    spec: ChatResponseSpec,
    grpc_status: &'static str,
) -> (Result<(), String>, Vec<Value>) {
    let (stream, server) = scripted_stream(responses, grpc_status).await;
    let (tx, rx) = sse_channel();
    let result = processor(with_tools)
        .process_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, false),
            spec,
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    (result, events)
}

fn chat_text(events: &[Value], index: u32, field: &str) -> String {
    events
        .iter()
        .filter_map(|event| {
            let choice = &event["choices"][0];
            if choice["index"] == index {
                choice["delta"][field].as_str()
            } else {
                None
            }
        })
        .collect()
}

#[tokio::test]
async fn chat_role_chunk_goes_out_on_the_first_engine_output() {
    // The engine's first output carries nothing visible (as a special token the
    // decoder skips does); the first visible text is held back behind the gate.
    let (gate_tx, gate_rx) = oneshot::channel();
    let (stream, server) = gated_stream(
        vec![chunk(0, "")],
        gate_rx,
        vec![chunk(0, "hi"), complete(0, "stop")],
        "0",
    )
    .await;
    let (tx, mut rx) = sse_channel();
    #[expect(
        clippy::disallowed_methods,
        reason = "bounded test fixture; the task ends with the stream"
    )]
    let task = tokio::spawn(async move {
        let result = processor(false)
            .process_streaming_chunks(
                stream,
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                chat_spec(false),
                &tx,
                None,
            )
            .await;
        drop(tx);
        result
    });
    // The role chunk opens the choice on the engine's first output, not on
    // its first visible text one step later.
    let first = timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("the role chunk arrives before any visible text")
        .expect("stream open")
        .expect("successful SSE write");
    let text = std::str::from_utf8(&first).expect("UTF-8 SSE");
    let data = text.trim().strip_prefix("data: ").expect("SSE data line");
    let event: Value = serde_json::from_str(data).expect("SSE JSON");
    assert_eq!(event["choices"][0]["delta"]["role"], "assistant");
    // The role chunk carries an empty content, the shape the engines' own
    // servers send (role once, content ""), not a null.
    assert_eq!(event["choices"][0]["delta"]["content"], "");
    gate_tx.send(()).expect("open the gate");
    let events = collect_events(rx).await;
    task.await.expect("stream task").expect("stream processed");
    assert_eq!(chat_text(&events, 0, "content"), "hi");
    server.abort();
}

#[tokio::test]
async fn chat_eof_preserves_reasoning_and_normal_tails_per_choice() {
    for send_complete in [true, false] {
        let mut responses = vec![
            chunk(0, "<think>first"),
            chunk(1, "answer"),
            chunk(0, "</thi"),
            chunk(1, "<thi"),
        ];
        if send_complete {
            responses.extend([complete(1, "stop"), complete(0, "length")]);
        }
        let (result, events) = chat_events(responses, false, "0").await;
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(chat_text(&events, 0, "reasoning_content"), "first</thi");
        assert_eq!(chat_text(&events, 0, "content"), "");
        assert_eq!(chat_text(&events, 1, "content"), "answer<thi");
        assert_eq!(chat_text(&events, 1, "reasoning_content"), "");
        for index in [0, 1] {
            let mut finished = false;
            for event in &events {
                let choice = &event["choices"][0];
                if choice["index"] == index {
                    if !choice["finish_reason"].is_null() {
                        finished = true;
                        assert_eq!(
                            choice["finish_reason"],
                            if index == 0 { "length" } else { "stop" }
                        );
                    } else if choice["delta"]["content"].is_string()
                        || choice["delta"]["reasoning_content"].is_string()
                    {
                        assert!(!finished, "tail emitted after choice finished: {events:?}");
                    }
                }
            }
            assert_eq!(finished, send_complete);
        }
    }
}

/// A stream carries `role` once, in the first delta the OpenAI API opens a
/// text stream with (`{"role": "assistant", "content": "", "refusal": null}`,
/// whether the model then answers with text, reasoning or a tool call it
/// chose itself); later deltas carry neither the role nor a
/// `reasoning_content: null` placeholder.
#[tokio::test]
async fn chat_stream_sends_role_once_in_an_empty_first_content_delta() {
    let text = vec![chunk(0, "Hi"), chunk(0, "!"), complete(0, "stop")];
    let reasoning = vec![chunk(0, "<think>why</think>answer"), complete(0, "stop")];
    let tool = vec![
        chunk(0, r#"{"name": "lookup", "arguments": {}}"#),
        complete(0, "stop"),
    ];
    for (frames, with_tools, content) in [
        (text, false, "Hi!"),
        (reasoning, false, "answer"),
        (tool, true, ""),
    ] {
        let (result, events) = chat_events(frames, with_tools, "0").await;
        assert!(result.is_ok(), "{result:?}");
        let deltas: Vec<&Value> = events
            .iter()
            .filter(|event| event["choices"][0]["index"] == 0)
            .map(|event| &event["choices"][0]["delta"])
            .collect();
        assert!(deltas.len() > 2, "{events:?}");
        assert_eq!(
            deltas[0],
            &serde_json::json!({"role": "assistant", "content": "", "refusal": null}),
            "{events:?}"
        );
        for delta in &deltas[1..] {
            assert!(delta.get("role").is_none(), "role repeated: {events:?}");
            assert!(
                delta.get("refusal").is_none(),
                "refusal repeated: {events:?}"
            );
            assert!(
                delta.get("reasoning_content").is_none_or(Value::is_string),
                "null reasoning placeholder: {events:?}"
            );
        }
        assert_eq!(chat_text(&events, 0, "content"), content, "{events:?}");
        assert_eq!(
            deltas.iter().any(|delta| delta["tool_calls"].is_array()),
            with_tools,
            "{events:?}"
        );
    }
}

/// A request that forces a tool call (`tool_choice: "required"`, a named
/// function) opens as the OpenAI API opens a tool-call stream:
/// `{"role": "assistant", "content": null}`, no `refusal`; the opening chunk
/// still goes out on the engine's first output.
#[tokio::test]
async fn chat_stream_opens_a_forced_tool_call_with_a_null_content() {
    for tool_choice in [
        serde_json::json!("required"),
        serde_json::json!({"type": "function", "function": {"name": "lookup"}}),
    ] {
        let frames = vec![
            chunk(0, r#"{"name": "lookup", "arguments": {}}"#),
            complete(0, "stop"),
        ];
        let spec = chat_spec_with(true, Some(tool_choice.clone()));
        let (result, events) = chat_events_with_spec(frames, true, spec, "0").await;
        assert!(result.is_ok(), "{result:?}");
        let first = &events[0]["choices"][0];
        assert_eq!(first["index"], 0, "{events:?}");
        assert_eq!(
            first["delta"],
            serde_json::json!({"role": "assistant", "content": null}),
            "{tool_choice}: {events:?}"
        );
        assert!(first["logprobs"].is_null() && first["finish_reason"].is_null());
    }
}

#[tokio::test]
async fn chat_eof_tail_passes_through_buffered_tool_text_in_order() {
    let (result, events) = chat_events(
        vec![chunk(0, "["), chunk(0, "<"), complete(0, "stop")],
        true,
        "0",
    )
    .await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "content"), "[<");
    assert!(events
        .iter()
        .all(|event| event["choices"][0]["delta"]["tool_calls"].is_null()));
}

/// The Kimi K3 parsers on an output `max_tokens` cuts inside the turn's closing
/// structure: no marker reaches the content, streamed or whole.
#[tokio::test]
async fn kimi_k3_cut_inside_the_closing_structure_keeps_the_markers_out_of_content() {
    // Thinking off: the prompt opened the response channel; the model answers and
    // closes the turn, one marker a token. Cut after 8 of the 10 tokens the client
    // used to receive `Pong.<|close|>message` (smg-lab #115); after 6, and at the
    // natural end, `Pong.`.
    let tokens = [
        "P",
        "ong",
        ".",
        "<|close|>",
        "response",
        "<|sep|>",
        "<|close|>",
        "message",
        "<|sep|>",
    ];
    for (cut, finish) in [(6, "length"), (8, "length"), (9, "stop")] {
        let text: String = tokens[..cut].concat();
        let resolver = || {
            utils::ParserResolver::new(
                Arc::new(WorkerRegistry::new()),
                Some("kimi_k3".to_string()),
                Some("kimi_k3".to_string()),
            )
        };
        // Streamed, a chunk per token.
        let mut frames: Vec<_> = tokens[..cut].iter().map(|token| chunk(0, token)).collect();
        frames.push(complete(0, finish));
        let (stream, server) = scripted_stream(frames, "0").await;
        let (tx, rx) = sse_channel();
        let processor = StreamingProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            resolver(),
            "vllm",
        );
        let result = processor
            .process_streaming_chunks(
                stream,
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                chat_spec(false),
                &tx,
                None,
            )
            .await;
        drop(tx);
        let events = collect_events(rx).await;
        server.abort();
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(
            chat_text(&events, 0, "content"),
            "Pong.",
            "streamed, cut after {cut} tokens"
        );
        assert_eq!(chat_text(&events, 0, "reasoning_content"), "");
        assert!(
            events
                .iter()
                .any(|event| event["choices"][0]["finish_reason"] == finish),
            "{events:?}"
        );
        // Whole, the non-streaming path.
        let mut last = complete(0, finish);
        if let Some(GenerationEvent::Complete(complete)) = &mut last.response {
            complete.output_ids = text.chars().map(u32::from).collect();
        }
        let (stream, server) = scripted_stream(vec![chunk(0, &text), last], "0").await;
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
        let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
        let response = ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            resolver(),
        )
        .process_non_streaming_chat_response(
            context::ExecutionResult::Single { stream },
            chat_spec(false),
            dispatch(),
            tokenizer,
            &mut decoder,
            false,
        )
        .await
        .unwrap_or_else(|response| panic!("{}", response.status()));
        server.abort();
        let response = serde_json::to_value(response).unwrap();
        assert_eq!(
            response["choices"][0]["message"]["content"], "Pong.",
            "whole, cut after {cut} tokens"
        );
        assert_eq!(response["choices"][0]["finish_reason"], finish);
    }
}

#[tokio::test]
async fn chat_stream_error_does_not_flush_a_success_tail() {
    let (result, events) = chat_events(
        vec![chunk(0, "<think>first"), chunk(0, "</thi")],
        false,
        "13",
    )
    .await;
    assert!(result.is_err());
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "first");
    assert!(events
        .iter()
        .all(|event| event["choices"][0]["finish_reason"].is_null()));
}

/// Token logprobs reach the client on reasoning deltas as they do on content
/// deltas: a chunk whose text is all reasoning carries its logprobs there.
#[tokio::test]
async fn chat_stream_keeps_logprobs_on_reasoning_deltas() {
    let (result, events) = chat_events(
        vec![
            chunk_with_logprobs(0, "<think>why"),
            chunk_with_logprobs(0, "</think>ok"),
            complete(0, "stop"),
        ],
        false,
        "0",
    )
    .await;
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "why");
    assert_eq!(chat_text(&events, 0, "content"), "ok");
    let mut with_text = 0;
    for event in &events {
        let choice = &event["choices"][0];
        let delta = &choice["delta"];
        let has_text = delta["reasoning_content"].is_string()
            || delta["content"]
                .as_str()
                .is_some_and(|text| !text.is_empty());
        if has_text {
            with_text += 1;
            assert!(
                choice["logprobs"]["content"]
                    .as_array()
                    .is_some_and(|items| !items.is_empty()),
                "text delta without logprobs: {event}"
            );
        }
    }
    assert!(with_text >= 2, "{events:?}");
}

#[tokio::test]
async fn chat_usage_chunk_excludes_unbilled_prompt_tokens() {
    for (unbilled, continuous) in [(0, false), (3, false), (0, true), (3, true)] {
        let mut first = chunk(0, "hi");
        if let Some(GenerationEvent::Chunk(chunk)) = &mut first.response {
            chunk.prompt_tokens = 10;
            chunk.cached_tokens = 10;
        }
        let mut last = complete_with_prompt(0, "stop", 10);
        if let Some(GenerationEvent::Complete(complete)) = &mut last.response {
            complete.cached_tokens = 10;
        }
        let (stream, server) = scripted_stream(vec![first, last], "0").await;
        let (tx, rx) = sse_channel();
        let request = serde_json::json!({
            "model": "eof-test", "messages": [], "stream": true,
            "stream_options": {"include_usage": true, "continuous_usage_stats": continuous}
        });
        let mut spec = ChatResponseSpec::from(
            &serde_json::from_value::<ChatCompletionRequest>(request).expect("chat request"),
        );
        spec.unbilled_prompt_tokens = unbilled;
        let result = processor(false)
            .process_streaming_chunks(
                stream,
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                spec,
                &tx,
                None,
            )
            .await;
        drop(tx);
        let events = collect_events(rx).await;
        server.abort();
        assert!(result.is_ok(), "{result:?}");

        let final_event = events.last().expect("final usage event");
        assert!(final_event["choices"].as_array().unwrap().is_empty());
        let usage = &final_event["usage"];
        for event in &events[..events.len() - 1] {
            let snapshot = &event["usage"];
            if continuous {
                assert_eq!(snapshot["prompt_tokens"], 10 - unbilled, "{event}");
                assert_eq!(
                    snapshot["prompt_tokens_details"]["cached_tokens"],
                    10 - unbilled
                );
                assert_eq!(
                    snapshot["total_tokens"].as_u64().unwrap(),
                    u64::from(10 - unbilled) + snapshot["completion_tokens"].as_u64().unwrap()
                );
                if !event["choices"][0]["finish_reason"].is_null() {
                    assert_eq!(snapshot, usage, "finish and final usage must agree");
                }
            } else {
                assert!(snapshot.is_null(), "{event}");
            }
        }
        assert!(events
            .iter()
            .any(|event| event["choices"][0]["delta"]["content"] == "hi"));
        assert!(events
            .iter()
            .any(|event| event["choices"][0]["finish_reason"] == "stop"));
        assert_eq!(usage["prompt_tokens"], 10 - unbilled);
        assert_eq!(
            usage["total_tokens"],
            usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap()
        );
    }
}

/// The engine on this path does not count reasoning tokens; the gateway's
/// reasoning parser does, and reports them whenever it ran (zero included),
/// as an engine with a reasoning parser does.
#[tokio::test]
async fn chat_usage_counts_the_tokens_the_reasoning_parser_split_off() {
    for (texts, reasoning, expected) in [
        (vec!["<think>", "ab", "</think>", "de"], "ab", 2),
        // Both delimiters in the reasoning chunk, alone and with content:
        // they are not reasoning tokens.
        (vec!["<think>ab</think>", "de"], "ab", 2),
        (vec!["<think>ab</think>de"], "ab", 2),
        (vec!["hi"], "", 0),
    ] {
        let mut responses: Vec<_> = texts.iter().map(|text| chunk(0, text)).collect();
        responses.push(complete(0, "stop"));
        let (stream, server) = scripted_stream(responses, "0").await;
        let (tx, rx) = sse_channel();
        let request = serde_json::json!({
            "model": "eof-test", "messages": [], "stream": true,
            "separate_reasoning": true,
            "stream_options": {"include_usage": true, "continuous_usage_stats": true}
        });
        let spec = ChatResponseSpec::from(
            &serde_json::from_value::<ChatCompletionRequest>(request).expect("chat request"),
        );
        let result = processor(false)
            .process_streaming_chunks(
                stream,
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                spec,
                &tx,
                None,
            )
            .await;
        drop(tx);
        let events = collect_events(rx).await;
        server.abort();
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(chat_text(&events, 0, "reasoning_content"), reasoning);
        let usage = &events.last().expect("final usage event")["usage"];
        assert_eq!(
            usage["completion_tokens_details"]["reasoning_tokens"], expected,
            "{usage}"
        );
        // The continuous usage never counts fewer reasoning tokens than the
        // chunk before it: the content chunk after a crossed delimiter agrees
        // with the reasoning chunk of its frame.
        let mut last = 0;
        for event in &events {
            if let Some(count) =
                event["usage"]["completion_tokens_details"]["reasoning_tokens"].as_u64()
            {
                assert!(count >= last, "reasoning count fell from {last}: {event}");
                last = count;
            }
        }
    }
}

#[tokio::test]
async fn messages_eof_emits_thinking_tail_before_block_stop() {
    for send_complete in [true, false] {
        let mut responses = vec![chunk(0, "<think>first"), chunk(0, "</thi")];
        if send_complete {
            responses.push(complete(0, "length"));
        }
        let (stream, server) = scripted_stream(responses, "0").await;
        let (tx, rx) = sse_channel();
        let spec = MessagesResponseSpec {
            thinking: Some(messages::ThinkingConfig::Enabled {
                budget_tokens: 1024,
                display: None,
            }),
            tool_choice: None,
            has_tools: false,
            history_tool_calls_count: 0,
            chat_tools: Vec::new(),
            stop_sequences: None,
        };
        let result = processor(false)
            .process_messages_streaming_chunks(
                stream,
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                spec,
                &tx,
                None,
            )
            .await;
        drop(tx);
        let events = collect_events(rx).await;
        server.abort();
        assert!(result.is_ok(), "{result:?}");
        let thinking: String = events
            .iter()
            .filter_map(|event| event["delta"]["thinking"].as_str())
            .collect();
        assert_eq!(thinking, "first</thi");
        let types: Vec<_> = events
            .iter()
            .filter_map(|event| event["type"].as_str())
            .collect();
        assert_eq!(
            types,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert_eq!(events[1]["content_block"]["type"], "thinking");
        assert_eq!(
            events[5]["delta"]["stop_reason"],
            if send_complete {
                "max_tokens"
            } else {
                "end_turn"
            }
        );
    }
}

/// A Messages response to `text` through the `qwen_xml` tool parser, which
/// reports a call's name before its arguments: the type of each content block
/// and the stop reason.
async fn qwen_xml_messages(text: &str, finish: &str, stream: bool) -> (Vec<String>, String) {
    let resolver = || {
        utils::ParserResolver::new(
            Arc::new(WorkerRegistry::new()),
            Some("qwen_xml".to_string()),
            None,
        )
    };
    let request = serde_json::from_value::<messages::CreateMessageRequest>(serde_json::json!({
        "model": "eof-test", "max_tokens": 16,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"name": "lookup", "input_schema": {"type": "object", "properties": {}}}]
    }))
    .expect("messages request");
    let spec = MessagesResponseSpec::from(&request);
    let mut last = complete(0, finish);
    if let Some(GenerationEvent::Complete(complete)) = &mut last.response {
        complete.output_ids = text.chars().map(u32::from).collect();
    }
    let (grpc_stream, server) = scripted_stream(vec![chunk(0, text), last], "0").await;
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
    let read = if stream {
        let processor = StreamingProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            resolver(),
            "vllm",
        );
        let (tx, rx) = sse_channel();
        let result = processor
            .process_messages_streaming_chunks(
                grpc_stream,
                dispatch(),
                tokenizer,
                (None, None, false, false, false),
                spec,
                &tx,
                None,
            )
            .await;
        drop(tx);
        assert!(result.is_ok(), "{result:?}");
        let events = collect_events(rx).await;
        let blocks = events
            .iter()
            .filter(|event| event["type"] == "content_block_start")
            .map(|event| event["content_block"]["type"].as_str().unwrap().to_string())
            .collect();
        let delta = events.iter().find(|event| event["type"] == "message_delta");
        let stop_reason = &delta.unwrap()["delta"]["stop_reason"];
        (blocks, stop_reason.as_str().unwrap().to_string())
    } else {
        let processor = ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            resolver(),
        );
        let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
        let message = processor
            .process_non_streaming_messages_response(
                context::ExecutionResult::Single {
                    stream: grpc_stream,
                },
                spec,
                dispatch(),
                tokenizer,
                &mut decoder,
            )
            .await
            .unwrap_or_else(|response| panic!("{}", response.status()));
        let message = serde_json::to_value(message).unwrap();
        let blocks = message["content"].as_array().unwrap().iter();
        let blocks = blocks
            .map(|block| block["type"].as_str().unwrap().to_string())
            .collect();
        (blocks, message["stop_reason"].as_str().unwrap().to_string())
    };
    server.abort();
    read
}

#[tokio::test]
async fn messages_truncated_after_a_tool_call_starts_stop_at_max_tokens() {
    // The output ends inside a call the parser already named, or after a
    // whole call. Only an output that finished calls a tool.
    let cut = "<tool_call>\n<function=lookup>\n<parameter=city>\nPar";
    let whole = "<tool_call>\n<function=lookup>\n</function>\n</tool_call>\nThen";
    for (text, stream) in [(cut, true), (whole, true), (whole, false)] {
        for (finish, stop_reason) in [("length", "max_tokens"), ("stop", "tool_use")] {
            let (blocks, got) = qwen_xml_messages(text, finish, stream).await;
            assert!(
                blocks.iter().any(|block| block == "tool_use"),
                "{text:?} {finish} stream={stream}: {blocks:?}"
            );
            assert_eq!(got, stop_reason, "{text:?} {finish} stream={stream}");
        }
    }
}

#[tokio::test]
async fn deepseek_usage_rides_on_the_last_finish_chunk() {
    for options in [
        serde_json::json!(null),
        serde_json::json!({"include_usage": false}),
        serde_json::json!({"include_usage": true}),
        serde_json::json!({"include_usage": true, "continuous_usage_stats": true}),
    ] {
        for (choices, unbilled) in [(1, 0), (2, 3)] {
            let mut frames = Vec::new();
            for index in 0..choices {
                frames.push(chunk(index, "<think>reason</think>answer<thi"));
                let mut final_frame = complete_with_prompt(index, "length", 10);
                if let Some(GenerationEvent::Complete(complete)) = &mut final_frame.response {
                    complete.cached_tokens = 8;
                    complete.completion_tokens = 31;
                }
                frames.push(final_frame);
            }
            let (stream, server) = scripted_stream(frames, "0").await;
            let (tx, rx) = sse_channel();
            let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "deepseek-flash", "messages": [], "stream": true, "n": choices,
                "stream_options": options, "separate_reasoning": true
            }))
            .unwrap();
            let mut spec = ChatResponseSpec::from(&req);
            spec.unbilled_prompt_tokens = unbilled;
            let result = processor(false)
                .process_streaming_chunks(
                    stream,
                    dispatch(),
                    Arc::new(CharacterTokenizer::default()),
                    (None, None, false, false, false),
                    spec,
                    &tx,
                    None,
                )
                .await;
            drop(tx);
            let events = collect_events(rx).await;
            server.abort();
            assert!(result.is_ok(), "{result:?}");
            assert!(events.len() > 2, "{events:?}");
            assert!(
                events
                    .iter()
                    .all(|event| event["choices"].as_array().unwrap().len() == 1),
                "no usage-only chunk: {events:?}"
            );
            let last = events.last().unwrap();
            assert_eq!(last["choices"][0]["finish_reason"], "length");
            assert!(last["choices"][0]["delta"]["content"].is_null());
            assert_eq!(last["usage"]["prompt_tokens"], 10 - unbilled);
            assert_eq!(last["usage"]["completion_tokens"], 31 * choices);
            assert_eq!(last["usage"]["total_tokens"], 10 - unbilled + 31 * choices);
            assert_eq!(
                last["usage"]["prompt_tokens_details"]["cached_tokens"],
                8.min(10 - unbilled)
            );
            // The six tokens of `reason` per choice, split out of the one
            // chunk by the gateway's reasoning parser.
            assert_eq!(
                last["usage"]["completion_tokens_details"]["reasoning_tokens"],
                6 * choices
            );
            for event in &events[..events.len() - 1] {
                if options["include_usage"] == true {
                    assert_eq!(event.get("usage"), Some(&Value::Null), "{event}");
                } else {
                    assert!(event.get("usage").is_none(), "{event}");
                }
            }
            for index in 0..choices {
                assert_eq!(chat_text(&events, index, "content"), "answer<thi");
                assert_eq!(chat_text(&events, index, "reasoning_content"), "reason");
            }
        }
    }
}

#[tokio::test]
async fn deepseek_does_not_emit_aggregate_usage_without_complete_frames() {
    for (status, partial) in [("0", false), ("13", false), ("0", true)] {
        let mut frames = vec![chunk(0, "hello")];
        if partial {
            frames.extend([complete(0, "stop"), chunk(1, "unfinished")]);
        }
        let (stream, server) = scripted_stream(frames, status).await;
        let (tx, rx) = sse_channel();
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "deepseek-flash", "messages": [], "stream": true, "n": if partial {2} else {1},
            "stream_options": {"include_usage": true}
        }))
        .unwrap();
        let result = processor(false)
            .process_streaming_chunks(
                stream,
                dispatch(),
                Arc::new(CharacterTokenizer::default()),
                (None, None, false, false, false),
                ChatResponseSpec::from(&req),
                &tx,
                None,
            )
            .await;
        drop(tx);
        let events = collect_events(rx).await;
        server.abort();
        assert_eq!(result.is_ok(), status == "0");
        assert!(!events.is_empty());
        assert!(
            events
                .iter()
                .all(|event| event.get("usage") == Some(&Value::Null)),
            "{events:?}"
        );
    }
}

/// Reasoning for every chunk but "text"; `is_in_reasoning` reports `.0`.
struct ReasoningButText(bool);

impl ReasoningParser for ReasoningButText {
    fn detect_and_parse_reasoning(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, reasoning_parser::ParseError> {
        Ok(ParserResult::normal(text.to_string()))
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
    ) -> Result<ParserResult, reasoning_parser::ParseError> {
        Ok(match text {
            "text" => ParserResult::normal(text.to_string()),
            _ => ParserResult::reasoning(text.to_string()),
        })
    }

    fn reset(&mut self) {}

    fn model_type(&self) -> &str {
        "reasoning-but-text"
    }

    fn is_in_reasoning(&self) -> bool {
        self.0
    }

    fn mark_reasoning_started(&mut self) {}

    fn mark_think_start_stripped(&mut self) {}
}

/// Returns the chunk as normal text and reports one whole `lookup` call on
/// its first parse.
#[derive(Default)]
struct CallFirst {
    called: bool,
}

#[async_trait::async_trait]
impl ToolParser for CallFirst {
    async fn parse_complete(
        &self,
        output: &str,
    ) -> tool_parser::errors::ParserResult<(String, Vec<tool_parser::ToolCall>)> {
        Ok((output.to_string(), Vec::new()))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        _tools: &[Tool],
    ) -> tool_parser::errors::ParserResult<StreamingParseResult> {
        let calls = (!std::mem::replace(&mut self.called, true)).then(|| ToolCallItem {
            tool_index: 0,
            name: Some("lookup".to_string()),
            parameters: "{}".to_string(),
        });
        Ok(StreamingParseResult {
            normal_text: chunk.to_string(),
            calls: calls.into_iter().collect(),
        })
    }

    fn has_tool_markers(&self, _text: &str) -> bool {
        false
    }
}

/// A processor with the stub parsers above.
fn stub_processor(in_reasoning: bool) -> StreamingProcessor {
    let reasoning = ReasoningParserFactory::new();
    reasoning
        .registry()
        .register_parser("reasoning-but-text", move || {
            Box::new(ReasoningButText(in_reasoning))
        });
    let tools = ToolParserFactory::new();
    tools
        .registry()
        .register_parser("call-first", || Box::new(CallFirst::default()));
    let resolver = utils::ParserResolver::new(
        Arc::new(WorkerRegistry::new()),
        Some("call-first".to_string()),
        Some("reasoning-but-text".to_string()),
    );
    StreamingProcessor::new(tools, reasoning, resolver, "vllm")
}

/// The content block starts and stops from `messages_blocks_and_inputs`.
async fn messages_blocks(
    processor: StreamingProcessor,
    tool_choice: Option<messages::ToolChoice>,
    texts: &[&str],
) -> Vec<String> {
    messages_blocks_and_inputs(processor, tool_choice, texts)
        .await
        .0
}

/// The content block starts and stops of a Messages stream of `texts`, and
/// the input of each `tool_use` block joined from its `input_json_delta`s as
/// a client SDK builds it (the joined text if it is not complete JSON). A
/// block starts only when none is open, and every delta goes to the open
/// block and matches its type.
async fn messages_blocks_and_inputs(
    processor: StreamingProcessor,
    tool_choice: Option<messages::ToolChoice>,
    texts: &[&str],
) -> (Vec<String>, Vec<Value>) {
    let spec = MessagesResponseSpec {
        thinking: Some(messages::ThinkingConfig::Enabled {
            budget_tokens: 1024,
            display: None,
        }),
        tool_choice,
        has_tools: true,
        history_tool_calls_count: 0,
        chat_tools: chat_spec(true).tools.unwrap(),
        stop_sequences: None,
    };
    let mut frames: Vec<_> = texts.iter().map(|text| chunk(0, text)).collect();
    frames.push(complete(0, "stop"));
    let (stream, server) = scripted_stream(frames, "0").await;
    let (tx, rx) = sse_channel();
    let result = processor
        .process_messages_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, false),
            spec,
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    assert!(result.is_ok(), "{result:?}");
    let mut open = None;
    let mut blocks = Vec::new();
    let mut inputs = Vec::new();
    for event in &events {
        let index = &event["index"];
        match event["type"].as_str() {
            Some("content_block_start") => {
                let kind = event["content_block"]["type"].as_str().unwrap_or_default();
                blocks.push(format!("start {index} {kind:?}"));
                assert!(open.replace((index, kind)).is_none(), "{blocks:?}");
                if kind == "tool_use" {
                    inputs.push(String::new());
                }
            }
            Some("content_block_delta") => {
                let delta = &event["delta"];
                let kind = match delta["type"].as_str() {
                    Some("text_delta") => "text",
                    Some("input_json_delta") => "tool_use",
                    Some("thinking_delta" | "signature_delta") => "thinking",
                    other => panic!("unexpected delta {other:?}"),
                };
                assert_eq!(open, Some((index, kind)), "delta {delta} after {blocks:?}");
                if let Some(json) = delta["partial_json"].as_str() {
                    inputs.last_mut().expect("open tool_use").push_str(json);
                }
            }
            Some("content_block_stop") => {
                blocks.push(format!("stop {index}"));
                assert_eq!(open.take().map(|open| open.0), Some(index), "{blocks:?}");
            }
            _ => {}
        }
    }
    let inputs = inputs
        .iter()
        .map(|json| match json.as_str() {
            "" => serde_json::json!({}),
            json => serde_json::from_str(json).unwrap_or_else(|_| json.into()),
        })
        .collect();
    (blocks, inputs)
}

#[tokio::test]
async fn messages_blocks_do_not_overlap_when_reasoning_calls_and_text_alternate() {
    // Reasoning, then a call with no text between; text; reasoning again.
    assert_eq!(
        messages_blocks(stub_processor(false), None, &["a", "text", "b"]).await,
        [
            "start 0 \"thinking\"",
            "stop 0",
            "start 1 \"tool_use\"",
            "stop 1",
            "start 2 \"text\"",
            "stop 2",
            "start 3 \"thinking\"",
            "stop 3",
        ]
    );
    // Text the reasoning parser returns while it stays in reasoning. Such text
    // skips the tool parser and follows the chunk's reasoning, so a chunk like
    // `</think>answer<think>more` still comes out in the wrong order; only the
    // block boundaries are checked here.
    assert_eq!(
        messages_blocks(stub_processor(true), None, &["a", "text", "b"]).await,
        [
            "start 0 \"thinking\"",
            "stop 0",
            "start 1 \"text\"",
            "stop 1",
            "start 2 \"thinking\"",
            "stop 2",
        ]
    );
    // Reasoning right after a call.
    assert_eq!(
        messages_blocks(stub_processor(false), None, &["a", "b"]).await,
        [
            "start 0 \"thinking\"",
            "stop 0",
            "start 1 \"tool_use\"",
            "stop 1",
            "start 2 \"thinking\"",
            "stop 2",
        ]
    );
}

/// A processor with the registered parsers `tool` and `reasoning`.
fn named_processor(tool: &str, reasoning: &str) -> StreamingProcessor {
    StreamingProcessor::new(
        ToolParserFactory::new(),
        ReasoningParserFactory::new(),
        utils::ParserResolver::new(
            Arc::new(WorkerRegistry::new()),
            Some(tool.to_string()),
            Some(reasoning.to_string()),
        ),
        "vllm",
    )
}

#[tokio::test]
async fn messages_blocks_do_not_overlap_with_deepseek_parsers() {
    // `</think>` ends reasoning without text, and the arguments of a specific
    // tool follow in the next chunk.
    let tool = messages::ToolChoice::Tool {
        name: "lookup".to_string(),
        disable_parallel_tool_use: None,
    };
    assert_eq!(
        messages_blocks(
            named_processor("deepseek", "deepseek_r1"),
            Some(tool),
            &["plan", "</think>", "{}"],
        )
        .await,
        [
            "start 0 \"thinking\"",
            "stop 0",
            "start 1 \"tool_use\"",
            "stop 1",
        ]
    );
    // A reasoning parser that enters reasoning again after text.
    assert_eq!(
        messages_blocks(
            named_processor("deepseek_v41", "deepseek_v41"),
            None,
            &[
                "<think>plan",
                "</think>",
                "answer",
                "<think>",
                "more",
                "</think>",
                "done"
            ],
        )
        .await,
        [
            "start 0 \"thinking\"",
            "stop 0",
            "start 1 \"text\"",
            "stop 1",
            "start 2 \"thinking\"",
            "stop 2",
            "start 3 \"text\"",
            "stop 3",
        ]
    );
    // Reasoning again while the tool parser holds text that it releases at
    // the end of the stream.
    assert_eq!(
        messages_blocks(
            named_processor("json", "deepseek_v41"),
            None,
            &["<think>plan", "</think>", "{", "<think>", "more"],
        )
        .await,
        [
            "start 0 \"thinking\"",
            "stop 0",
            "start 1 \"thinking\"",
            "stop 1",
            "start 2 \"text\"",
            "stop 2",
        ]
    );
}

#[tokio::test]
async fn messages_tool_arguments_precede_text_in_the_same_chunk() {
    // With multi-token chunks, qwen_xml returns the arguments that finish a
    // call together with the text after the call.
    let (blocks, inputs) = messages_blocks_and_inputs(
        named_processor("qwen_xml", "qwen3"),
        None,
        &[
            "<tool_call>\n<function=lookup>\n<parameter=q>\n1",
            "\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=lookup>\n",
            "<parameter=q>\n2\n</parameter>\n</function>\n</tool_call>",
        ],
    )
    .await;
    assert_eq!(
        blocks,
        [
            "start 0 \"tool_use\"",
            "stop 0",
            "start 1 \"text\"",
            "stop 1",
            "start 2 \"tool_use\"",
            "stop 2",
        ]
    );
    assert_eq!(
        inputs,
        [serde_json::json!({"q": 1}), serde_json::json!({"q": 2})]
    );
    for (texts, input) in [
        (
            [
                "<tool_call>\n<function=lookup>\n<parameter=q>\nPar",
                "is\n</parameter>\n</function>\n</tool_call>\nDone.",
            ],
            serde_json::json!({"q": "Paris"}),
        ),
        (
            [
                "<tool_call>\n<function=lookup>\n<parameter=a>\nx\n</parameter>\n",
                "<parameter=b>\ny\n</parameter>\n</function>\n</tool_call>\n",
            ],
            serde_json::json!({"a": "x", "b": "y"}),
        ),
    ] {
        let (blocks, inputs) =
            messages_blocks_and_inputs(named_processor("qwen_xml", "qwen3"), None, &texts).await;
        assert_eq!(
            blocks,
            [
                "start 0 \"tool_use\"",
                "stop 0",
                "start 1 \"text\"",
                "stop 1"
            ]
        );
        assert_eq!(inputs, [input]);
    }
}

#[tokio::test]
async fn messages_tool_arguments_need_an_open_block() {
    // Reasoning that starts only after the specific tool's block stops that
    // block; the arguments after it have no block to go to and are dropped.
    let tool = messages::ToolChoice::Tool {
        name: "lookup".to_string(),
        disable_parallel_tool_use: None,
    };
    let (blocks, inputs) = messages_blocks_and_inputs(
        named_processor("qwen", "qwen3"),
        Some(tool),
        &["<thi", "nk>plan", "</think>", "{}"],
    )
    .await;
    assert_eq!(
        blocks,
        [
            "start 0 \"tool_use\"",
            "stop 0",
            "start 1 \"thinking\"",
            "stop 1",
        ]
    );
    assert_eq!(inputs, [serde_json::json!({})]);
    // Until qwen3 strips a `<think>`, it takes the first one anywhere as the
    // start of reasoning, so one inside an argument stops the call's block.
    // The rest of the arguments are dropped, in the middle of the stream and
    // at its end, where qwen_xml releases the closing brace.
    for (texts, input) in [
        (
            [
                "<tool_call>\n<function=lookup>\n<parameter=q>\nA ",
                "<think>x</think> B\n</parameter>\n",
                "</function>\n</tool_call>",
            ],
            serde_json::json!({}),
        ),
        (
            [
                "<tool_call>\n<function=lookup>\n<parameter=q>\nA\n</parameter>\n",
                "<parameter=r>\nB ",
                "<think>x",
            ],
            serde_json::json!("{\"q\": \"A\""),
        ),
    ] {
        let (blocks, inputs) =
            messages_blocks_and_inputs(named_processor("qwen_xml", "qwen3"), None, &texts).await;
        assert_eq!(
            blocks,
            [
                "start 0 \"tool_use\"",
                "stop 0",
                "start 1 \"thinking\"",
                "stop 1",
            ]
        );
        assert_eq!(inputs, [input]);
    }
}
