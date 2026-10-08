//! Test gateway parsers and SSE events with exact gRPC frames.

use std::time::Duration;

use axum::http::{HeaderMap, HeaderValue, StatusCode};
use bytes::BufMut;
use http_body::Frame;
use http_body_util::StreamBody;
use llm_tokenizer::{traits::Encoding, SpecialTokens};
use openai_protocol::chat::ChatCompletionRequest;
use prost::Message as ProstMessage;
use smg_grpc_client::{
    sglang_scheduler::proto as sglang_proto,
    vllm_engine::{
        proto, proto::generate_response::Response as GenerationEvent, AbortOnDropStream,
        VllmEngineClient,
    },
};
use tokio::{net::TcpListener, task::JoinHandle};
use tonic::codec::Codec;
use tool_parser::types::ToolCallItem;

use super::*;
use crate::{
    routers::{common::sse::SseReceiver, grpc::regular::processor::ResponseProcessor},
    worker::WorkerRegistry,
};

#[derive(Default)]
struct CharacterTokenizer {
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
    fn eos_token_ids(&self) -> &[u32] {
        &[0]
    }
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

fn chunk(index: u32, text: &str) -> proto::GenerateResponse {
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

fn complete(index: u32, reason: &str) -> proto::GenerateResponse {
    complete_with_prompt(index, reason, 1)
}

fn complete_with_prompt(index: u32, reason: &str, prompt_tokens: u32) -> proto::GenerateResponse {
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
#[expect(
    clippy::disallowed_methods,
    reason = "bounded test fixture; server task is explicitly aborted"
)]
async fn scripted_stream(
    responses: Vec<proto::GenerateResponse>,
    grpc_status: &'static str,
) -> (ProtoStream, JoinHandle<()>) {
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
    let mut frames = Vec::new();
    for response in responses {
        let encoded = response.encode_to_vec();
        let mut frame = Vec::with_capacity(encoded.len() + 5);
        frame.put_u8(0);
        frame.put_u32(encoded.len() as u32);
        frame.extend_from_slice(&encoded);
        frames.push(Ok::<_, tonic::Status>(Frame::data(Bytes::from(frame))));
    }
    let mut trailers = HeaderMap::new();
    trailers.insert("grpc-status", HeaderValue::from_static(grpc_status));
    frames.push(Ok(Frame::trailers(trailers)));
    let mut codec =
        tonic_prost::ProstCodec::<proto::GenerateResponse, proto::GenerateResponse>::default();
    let stream = tonic::Streaming::new_response(
        codec.decoder(),
        StreamBody::new(futures::stream::iter(frames)),
        StatusCode::OK,
        None,
        None,
    );
    let stream = AbortOnDropStream::new(stream, "eof-test".to_string(), client);
    // No generation was sent to the mock, so there is nothing to cancel.
    stream.mark_completed();
    (ProtoStream::Vllm(stream), server)
}

#[expect(
    clippy::disallowed_methods,
    reason = "bounded fixture server is explicitly aborted"
)]
async fn scripted_sglang_stream(
    responses: Vec<sglang_proto::GenerateResponse>,
    grpc_status: &'static str,
) -> (ProtoStream, JoinHandle<()>) {
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
    let client = smg_grpc_client::sglang_scheduler::SglangSchedulerClient::connect(&format!(
        "http://127.0.0.1:{port}"
    ))
    .await
    .expect("connect mock worker");
    let mut frames = Vec::new();
    for response in responses {
        let encoded = response.encode_to_vec();
        let mut frame = Vec::with_capacity(encoded.len() + 5);
        frame.put_u8(0);
        frame.put_u32(encoded.len() as u32);
        frame.extend_from_slice(&encoded);
        frames.push(Ok::<_, tonic::Status>(Frame::data(Bytes::from(frame))));
    }
    let mut trailers = HeaderMap::new();
    trailers.insert("grpc-status", HeaderValue::from_static(grpc_status));
    frames.push(Ok(Frame::trailers(trailers)));
    let mut codec = tonic_prost::ProstCodec::<
        sglang_proto::GenerateResponse,
        sglang_proto::GenerateResponse,
    >::default();
    let stream = tonic::Streaming::new_response(
        codec.decoder(),
        StreamBody::new(futures::stream::iter(frames)),
        StatusCode::OK,
        None,
        None,
    );
    let stream = smg_grpc_client::sglang_scheduler::AbortOnDropStream::new(
        stream,
        "eof-test".to_string(),
        client,
    );
    // No generation was sent to the mock, so there is nothing to cancel.
    stream.mark_completed();
    (ProtoStream::Sglang(stream), server)
}

fn processor(with_tools: bool) -> StreamingProcessor {
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
    let (stream, server) = scripted_stream(responses, grpc_status).await;
    let (tx, rx) = sse_channel();
    let result = processor(with_tools)
        .process_streaming_chunks(
            stream,
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            (None, None, false, false, false),
            chat_spec(with_tools),
            &tx,
            None,
        )
        .await;
    drop(tx);
    let events = collect_events(rx).await;
    server.abort();
    (result, events)
}

#[tokio::test]
async fn reasoning_wire_names_match_for_framed_streaming_and_nonstreaming_responses() {
    for (text, reasoning) in [
        ("<think>reason</think>answer", Some("reason")),
        ("answer", None),
    ] {
        let (result, events) =
            chat_events(vec![chunk(0, text), complete(0, "stop")], false, "0").await;
        result.unwrap();
        assert_eq!(
            chat_text(&events, 0, "reasoning"),
            reasoning.unwrap_or_default()
        );
        for delta in events
            .iter()
            .flat_map(|event| event["choices"].as_array().unwrap())
            .map(|choice| &choice["delta"])
        {
            assert_eq!(delta.get("reasoning"), delta.get("reasoning_content"));
            assert!(delta.get("reasoning").is_none_or(Value::is_string));
        }

        let mut last = complete(0, "stop");
        if let Some(GenerationEvent::Complete(complete)) = &mut last.response {
            complete.output_ids = text.chars().map(u32::from).collect();
        }
        let (stream, server) = scripted_stream(vec![last], "0").await;
        let processor = ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            utils::ParserResolver::new(Arc::new(WorkerRegistry::new()), None, Some("qwen3".into())),
        );
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
        let mut decoder = utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
        let response = processor
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
        let wire = serde_json::to_value(response).unwrap();
        let message = &wire["choices"][0]["message"];
        assert_eq!(message.get("reasoning").and_then(Value::as_str), reasoning);
        assert_eq!(message.get("reasoning"), message.get("reasoning_content"));
        if reasoning.is_none() {
            assert!(message.get("reasoning").is_none());
        }
    }
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

fn assert_chat_envelope(events: &[Value], indices: &[u32]) {
    for index in indices {
        let choices: Vec<_> = events
            .iter()
            .flat_map(|event| {
                assert_eq!(event["id"], "chatcmpl-eof");
                event["choices"].as_array().expect("choices array")
            })
            .filter(|choice| choice["index"] == *index)
            .collect();
        assert_eq!(
            choices.first().expect("choice initialization")["delta"],
            serde_json::json!({"role": "assistant", "content": ""})
        );
        for choice in &choices[1..] {
            assert!(choice["delta"].get("role").is_none(), "{choice}");
        }
        for choice in &choices {
            assert!(
                choice["delta"]
                    .get("reasoning_content")
                    .is_none_or(Value::is_string),
                "{choice}"
            );
        }
        let finishes: Vec<_> = choices
            .iter()
            .enumerate()
            .filter(|(_, choice)| !choice["finish_reason"].is_null())
            .collect();
        assert_eq!(finishes.len(), 1, "{choices:?}");
        assert_eq!(finishes[0].0, choices.len() - 1, "{choices:?}");
    }
}

#[tokio::test]
async fn chat_envelope_initializes_each_choice_once_and_preserves_eof_tails() {
    let (result, events) = chat_events(
        vec![
            chunk(0, "<think>reason</think>answer"),
            chunk(1, "second"),
            chunk(0, "<thi"),
            complete(1, "stop"),
            complete(0, "length"),
        ],
        false,
        "0",
    )
    .await;
    result.expect("successful stream");
    assert_chat_envelope(&events, &[0, 1]);
    assert_eq!(chat_text(&events, 0, "reasoning_content"), "reason");
    assert_eq!(chat_text(&events, 0, "content"), "answer<thi");
    assert_eq!(chat_text(&events, 1, "content"), "second");
}

#[tokio::test]
async fn chat_envelope_tool_name_and_arguments_do_not_repeat_role() {
    let (result, events) = chat_events(
        vec![
            chunk(0, r#"{"name":"lookup","arguments":{"city":""#),
            chunk(0, r#"Paris"}}"#),
            complete(0, "stop"),
        ],
        true,
        "0",
    )
    .await;
    result.expect("successful tool stream");
    assert_chat_envelope(&events, &[0]);
    let calls: Vec<_> = events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
        .collect();
    assert!(calls
        .iter()
        .any(|call| call["function"]["name"] == "lookup"));
    let arguments: String = calls
        .iter()
        .filter_map(|call| call["function"]["arguments"].as_str())
        .collect();
    assert_eq!(
        serde_json::from_str::<Value>(&arguments).unwrap(),
        serde_json::json!({"city": "Paris"})
    );
}

#[tokio::test]
async fn chat_envelope_http_body_keeps_empty_choice_usage_and_done_contract() {
    use axum::body::to_bytes;

    let (stream, server) = scripted_stream(
        vec![chunk(0, "answer"), complete(1, "stop"), complete(0, "stop")],
        "0",
    )
    .await;
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "eof-test", "messages": [], "stream": true, "n": 2,
        "stream_options": {"include_usage": true}
    }))
    .unwrap();
    let response = Arc::new(processor(false))
        .process_streaming_response(
            context::ExecutionResult::Single { stream },
            ChatResponseSpec::from(&request),
            dispatch(),
            Arc::new(CharacterTokenizer::default()),
            false,
            None,
        )
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = tokio::time::timeout(
        Duration::from_secs(5),
        to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .unwrap()
    .unwrap();
    server.abort();
    let text = std::str::from_utf8(&body).unwrap();
    let frames: Vec<_> = text
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .collect();
    assert_eq!(frames.last(), Some(&"data: [DONE]"));
    assert_eq!(
        frames
            .iter()
            .filter(|frame| **frame == "data: [DONE]")
            .count(),
        1
    );
    let events: Vec<Value> = frames[..frames.len() - 1]
        .iter()
        .map(|frame| serde_json::from_str(frame.strip_prefix("data: ").unwrap()).unwrap())
        .collect();
    assert_chat_envelope(&events, &[0, 1]);
    assert_eq!(chat_text(&events, 0, "content"), "answer");
    assert_eq!(chat_text(&events, 1, "content"), "");
    let usage: Vec<_> = events
        .iter()
        .filter(|event| event["choices"].as_array().unwrap().is_empty())
        .collect();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0], events.last().unwrap());
    assert!(usage[0]["usage"].is_object());
}

#[tokio::test]
async fn chat_envelope_harmony_empty_choices_still_initialize_before_finishing() {
    use axum::body::to_bytes;

    use crate::routers::grpc::harmony::streaming::HarmonyStreamingProcessor;

    let (stream, server) =
        scripted_stream(vec![complete(1, "length"), complete(0, "stop")], "0").await;
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "eof-test", "messages": [], "stream": true, "n": 2,
        "stream_options": {"include_usage": true}
    }))
    .unwrap();
    let response = Arc::new(HarmonyStreamingProcessor::new())
        .process_streaming_chat_response(
            context::ExecutionResult::Single { stream },
            Arc::new(request),
            dispatch(),
            Vec::new(),
            None,
        )
        .await;
    let body = tokio::time::timeout(
        Duration::from_secs(5),
        to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .unwrap()
    .unwrap();
    server.abort();
    let text = std::str::from_utf8(&body).unwrap();
    assert_eq!(text.matches("data: [DONE]").count(), 1);
    let events: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert_chat_envelope(&events, &[0, 1]);
    assert_eq!(events.last().unwrap()["choices"], serde_json::json!([]));
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
            assert!(last["usage"]["completion_tokens_details"]["reasoning_tokens"].is_null());
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

#[tokio::test]
async fn parser_usage_omits_ambiguous_boundaries_and_reports_zero_without_reasoning() {
    for streaming in [true, false] {
        for (text, expected) in [
            ("<think>abc</think>answer", 3),
            ("answer", 0),
            ("<think>abc\0answer", 3),
        ] {
            let ids: Vec<_> = text.chars().map(u32::from).collect();
            let mut last = complete(0, "stop");
            if let Some(GenerationEvent::Complete(complete)) = &mut last.response {
                complete.output_ids = ids.clone();
                complete.completion_tokens = ids.len() as u32;
            }
            let (stream, server) = scripted_stream(vec![chunk(0, text), last], "0").await;
            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model":"eof-test", "messages":[], "stream":streaming,
                "separate_reasoning":true, "stream_options":{"include_usage":true}
            }))
            .unwrap();
            let tokenizer: Arc<dyn Tokenizer> = Arc::new(CharacterTokenizer::default());
            let usage = if streaming {
                let (tx, rx) = sse_channel();
                processor(false)
                    .process_streaming_chunks(
                        stream,
                        dispatch(),
                        tokenizer,
                        (None, None, false, false, false),
                        ChatResponseSpec::from(&request),
                        &tx,
                        None,
                    )
                    .await
                    .unwrap();
                drop(tx);
                let events = collect_events(rx).await;
                assert_eq!(
                    chat_text(&events, 0, "reasoning_content"),
                    if expected > 0 { "abc" } else { "" }
                );
                events.last().unwrap()["usage"].clone()
            } else {
                let response_processor = ResponseProcessor::new(
                    ToolParserFactory::new(),
                    ReasoningParserFactory::new(),
                    utils::ParserResolver::new(
                        Arc::new(WorkerRegistry::new()),
                        None,
                        Some("qwen3".into()),
                    ),
                );
                let mut decoder =
                    utils::create_stop_decoder(&tokenizer, None, None, false, false, false);
                let response = response_processor
                    .process_non_streaming_chat_response(
                        context::ExecutionResult::Single { stream },
                        ChatResponseSpec::from(&request),
                        dispatch(),
                        tokenizer,
                        &mut decoder,
                        false,
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    response.choices[0]
                        .message
                        .reasoning_content
                        .as_deref()
                        .unwrap_or(""),
                    if expected > 0 { "abc" } else { "" }
                );
                serde_json::to_value(response.usage).unwrap()
            };
            server.abort();
            assert_eq!(
                usage["completion_tokens_details"]["reasoning_tokens"],
                if expected == 0 {
                    serde_json::json!(0)
                } else {
                    Value::Null
                },
                "stream={streaming}, usage={usage}"
            );
            assert_eq!(usage["completion_tokens"], ids.len() as u32);
        }
    }
}

#[expect(
    dead_code,
    reason = "shared tokenizer fixture also supports multi-turn tests"
)]
mod reasoning_tokenizer {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/common/scripted_tokenizer.rs"
    ));
}

#[tokio::test]
async fn parser_reasoning_usage_counts_engine_word_tokens_for_each_choice() {
    for streaming in [true, false] {
        for eos_stop in [false, true] {
            for mixed_unknown in [false, true] {
                for parser_enabled in [true, false] {
                    let tokenizer: Arc<dyn Tokenizer> =
                        Arc::new(reasoning_tokenizer::ScriptedTokenizer::from_chunks(vec![
                            "<think>".into(),
                            "one long reasoning token".into(),
                            "🦀".into(),
                            "</think>".into(),
                            "answer".into(),
                            "reason</think>answer".into(),
                        ]));
                    let mut frames = Vec::new();
                    for (index, batches) in [
                        (
                            0,
                            if eos_stop {
                                vec![vec![100, 101], vec![102, 999, 104]]
                            } else {
                                vec![vec![100, 101], vec![102, 103, 104]]
                            },
                        ),
                        (
                            1,
                            if mixed_unknown {
                                vec![vec![100, 105]]
                            } else {
                                vec![vec![104]]
                            },
                        ),
                    ] {
                        let ids: Vec<_> = batches.iter().flatten().copied().collect();
                        for batch in batches {
                            frames.push(proto::GenerateResponse {
                                response: Some(GenerationEvent::Chunk(
                                    proto::GenerateStreamChunk {
                                        index,
                                        completion_tokens: batch.len() as u32,
                                        token_ids: batch,
                                        ..Default::default()
                                    },
                                )),
                            });
                        }
                        if !(streaming && mixed_unknown && eos_stop && index == 1) {
                            frames.push(proto::GenerateResponse {
                                response: Some(GenerationEvent::Complete(
                                    proto::GenerateComplete {
                                        index,
                                        output_ids: ids.clone(),
                                        completion_tokens: ids.len() as u32,
                                        prompt_tokens: 1,
                                        finish_reason: "stop".into(),
                                        ..Default::default()
                                    },
                                )),
                            });
                        }
                    }
                    let (stream, server) = if mixed_unknown {
                        use smg_grpc_client::sglang_scheduler::proto as sg;
                        let frames = frames
                            .into_iter()
                            .map(|frame| {
                                let response = match frame.response.unwrap() {
                                    GenerationEvent::Chunk(c) => {
                                        sg::generate_response::Response::Chunk(
                                            sg::GenerateStreamChunk {
                                                index: c.index,
                                                token_ids: c.token_ids,
                                                completion_tokens: c.completion_tokens,
                                                ..Default::default()
                                            },
                                        )
                                    }
                                    GenerationEvent::Complete(c) => {
                                        sg::generate_response::Response::Complete(
                                            sg::GenerateComplete {
                                                index: c.index,
                                                output_ids: c.output_ids,
                                                prompt_tokens: c.prompt_tokens,
                                                completion_tokens: c.completion_tokens,
                                                finish_reason: c.finish_reason,
                                                reasoning_tokens: if c.index == 0 { 2 } else { 0 },
                                                ..Default::default()
                                            },
                                        )
                                    }
                                };
                                sg::GenerateResponse {
                                    response: Some(response),
                                    ..Default::default()
                                }
                            })
                            .collect();
                        scripted_sglang_stream(frames, "0").await
                    } else {
                        scripted_stream(frames, "0").await
                    };
                    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"eof-test", "messages":[], "stream":streaming, "n":2, "separate_reasoning":parser_enabled,
            "stop_token_ids": if eos_stop { vec![999] } else { vec![] },
            "stream_options":{"include_usage":true,"continuous_usage_stats":true}
        }))
        .unwrap();
                    let usage = if streaming {
                        let (tx, rx) = sse_channel();
                        Box::pin(processor(false).process_streaming_chunks(
                            stream,
                            dispatch(),
                            tokenizer,
                            (None, request.stop_token_ids.clone(), false, false, false),
                            ChatResponseSpec::from(&request),
                            &tx,
                            None,
                        ))
                        .await
                        .unwrap();
                        drop(tx);
                        let events = collect_events(rx).await;
                        assert_eq!(
                            chat_text(&events, 0, "reasoning_content"),
                            if parser_enabled {
                                "one long reasoning token🦀"
                            } else {
                                ""
                            }
                        );
                        assert_eq!(
                            chat_text(&events, 1, "reasoning_content"),
                            if mixed_unknown && parser_enabled {
                                "reason"
                            } else {
                                ""
                            }
                        );
                        assert_eq!(
                            events
                                .iter()
                                .filter_map(|event| event["usage"]["completion_tokens_details"]
                                    ["reasoning_tokens"]
                                    .as_u64())
                                .next_back(),
                            if parser_enabled {
                                if mixed_unknown {
                                    None
                                } else {
                                    Some(2)
                                }
                            } else if mixed_unknown {
                                Some(2)
                            } else {
                                None
                            }
                        );
                        events.last().unwrap()["usage"].clone()
                    } else {
                        let p = ResponseProcessor::new(
                            ToolParserFactory::new(),
                            ReasoningParserFactory::new(),
                            utils::ParserResolver::new(
                                Arc::new(WorkerRegistry::new()),
                                None,
                                Some("qwen3".into()),
                            ),
                        );
                        let mut decoder = utils::create_stop_decoder(
                            &tokenizer,
                            None,
                            request.stop_token_ids.as_ref(),
                            false,
                            false,
                            false,
                        );
                        let response = Box::pin(p.process_non_streaming_chat_response(
                            context::ExecutionResult::Single { stream },
                            ChatResponseSpec::from(&request),
                            dispatch(),
                            tokenizer,
                            &mut decoder,
                            false,
                        ))
                        .await
                        .unwrap();
                        assert_eq!(
                            response.choices[0].message.reasoning_content.as_deref(),
                            if parser_enabled {
                                Some("one long reasoning token🦀")
                            } else {
                                None
                            }
                        );
                        serde_json::to_value(response.usage).unwrap()
                    };
                    server.abort();
                    assert_eq!(
                        usage["completion_tokens_details"]["reasoning_tokens"],
                        if parser_enabled == mixed_unknown {
                            Value::Null
                        } else {
                            serde_json::json!(2)
                        },
                        "{usage}"
                    );
                    assert_eq!(
                        usage["completion_tokens"],
                        if streaming && mixed_unknown && eos_stop {
                            5 // Preserve existing settlement: only completed choices contribute.
                        } else if mixed_unknown {
                            7
                        } else {
                            6
                        }
                    );
                    assert_eq!(usage["prompt_tokens"], 1);
                }
            }
        }
    }
}
