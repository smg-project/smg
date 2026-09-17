//! Test gateway parsers and SSE events with exact gRPC frames.

use std::time::Duration;

use axum::http::{HeaderMap, HeaderValue, StatusCode};
use bytes::BufMut;
use http_body::Frame;
use http_body_util::StreamBody;
use llm_tokenizer::{traits::Encoding, SpecialTokens};
use openai_protocol::chat::ChatCompletionRequest;
use prost::Message as ProstMessage;
use smg_grpc_client::vllm_engine::{
    proto, proto::generate_response::Response as GenerationEvent, AbortOnDropStream,
    VllmEngineClient,
};
use tokio::{net::TcpListener, task::JoinHandle};
use tonic::codec::Codec;

use super::*;
use crate::{routers::common::sse::SseReceiver, worker::WorkerRegistry};

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
    for unbilled in [0, 3] {
        let (stream, server) = scripted_stream(
            vec![chunk(0, "hi"), complete_with_prompt(0, "stop", 10)],
            "0",
        )
        .await;
        let (tx, rx) = sse_channel();
        let request = serde_json::json!({
            "model": "eof-test", "messages": [], "stream": true,
            "stream_options": {"include_usage": true}
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

        let usage = events
            .iter()
            .find_map(|event| event.get("usage").filter(|usage| !usage.is_null()))
            .expect("usage chunk");
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
