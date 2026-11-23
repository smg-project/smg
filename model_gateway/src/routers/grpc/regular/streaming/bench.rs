//! The stream state machines under a benchmark. A chat, completion, generate
//! or Messages stream is driven the way the streaming tests drive theirs:
//! over an engine stream of pre-encoded gRPC frames decoded by the client's
//! own codec (no socket under it), writing to the bounded SSE channel a
//! client reads from, with the bytes written as the result. The gRPC router
//! module re-exports `EngineScript`, `StreamBench` and `ForcedChat`, the face
//! `benches/stream_state_machines.rs`, `benches/forced_call_stream.rs` and
//! `benches/symphony_messages_stream.rs` see; the module is compiled for the
//! crate's test targets only.

#![expect(
    clippy::expect_used,
    reason = "a benchmark fixture: a failure of its own plumbing is a failure, not a measurement"
)]

use std::sync::Arc;

use axum::http::{HeaderMap, HeaderValue, StatusCode};
use bytes::{BufMut, Bytes};
use futures::{stream, Stream, StreamExt as _};
use http_body::Frame;
use http_body_util::StreamBody;
use llm_tokenizer::traits::Tokenizer;
use openai_protocol::{
    chat::ChatCompletionRequest, completion::CompletionRequest, profile::ProviderProfile,
};
use prost::Message as ProstMessage;
use reasoning_parser::ParserFactory as ReasoningParserFactory;
use serde_json::json;
use smg_grpc_client::{
    codec::SizedProstCodec,
    vllm_engine::{
        proto, proto::generate_response::Response as GenerationEvent, AbortOnDropStream,
        VllmEngineClient,
    },
};
use symphony::{
    formats::{qwen3, Table},
    grammar::GrammarStart,
    CallSyntax,
};
use tokio::{net::TcpListener, task::JoinHandle};
use tonic::{
    codec::Codec,
    service::Routes,
    transport::{server::TcpIncoming, Server},
};
use tool_parser::ParserFactory as ToolParserFactory;

use super::{character_tokenizer::CharacterTokenizer, GenerateStreamContext, StreamingProcessor};
use crate::routers::{
    common::{
        snapshot::WorkerRegistry,
        sse::{sse_channel, SseReceiver},
    },
    grpc::{
        context::DispatchMetadata,
        proto_wrapper::ProtoStream,
        regular::symphony::{assign_for, Asks, Assignment, Forbids, Reading},
        spec::{ChatResponseSpec, CompletionResponseSpec, MessagesResponseSpec},
        utils::ParserResolver,
    },
};

/// The model the requests name.
const MODEL: &str = "bench-model";
/// The completion request's prompt (echoed by no scenario here).
const COMPLETION_PROMPT: &str = "Say hello";

/// The engine's side of one request, pre-encoded: one gRPC data frame per
/// message, a chunk of one token per character of the text and the `stop`
/// completion after them, read again for every iteration.
pub struct EngineScript {
    frames: Vec<Bytes>,
    tokens: usize,
}

impl EngineScript {
    /// One chunk per character of `text` (a character is a token under the
    /// character tokenizer), then the completion of choice 0.
    pub fn single_tokens(text: &str) -> Self {
        let tokens = text.chars().count();
        let mut frames: Vec<Bytes> = text
            .chars()
            .map(|character| data_frame(&chunk(u32::from(character))))
            .collect();
        frames.push(data_frame(&complete(tokens)));
        Self { frames, tokens }
    }

    /// [`Self::single_tokens`] with each chunk carrying its token's logprobs:
    /// the sampled logprob and `candidates` ranked alternatives, the shape of
    /// a logprobs stream's chunk on the wire.
    pub fn single_tokens_with_logprobs(text: &str, candidates: u32) -> Self {
        let tokens = text.chars().count();
        let mut frames: Vec<Bytes> = text
            .chars()
            .map(|character| data_frame(&chunk_with_logprobs(u32::from(character), candidates)))
            .collect();
        frames.push(data_frame(&complete(tokens)));
        Self { frames, tokens }
    }

    /// The tokens of the script: its chunks, one each.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// One reading of the script: its data frames, then the trailers that
    /// end the stream as the engine ends a finished request.
    fn reading(&self) -> impl Stream<Item = Result<Frame<Bytes>, tonic::Status>> + Send + 'static {
        stream::iter(self.frames.clone())
            .map(|frame| Ok(Frame::data(frame)))
            .chain(stream::iter([Ok(trailers_frame())]))
    }
}

/// A chunk carrying the one token `id` of choice 0.
fn chunk(id: u32) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
            index: 0,
            token_ids: vec![id],
            completion_tokens: 1,
            ..Default::default()
        })),
    }
}

/// [`chunk`] with the token's logprobs: the sampled logprob and `candidates`
/// ranked alternatives (the ids after the token's, one value each).
fn chunk_with_logprobs(id: u32, candidates: u32) -> proto::GenerateResponse {
    let value = |k: u32| -0.5 - f32::from(u8::try_from(k).unwrap_or(u8::MAX));
    proto::GenerateResponse {
        response: Some(GenerationEvent::Chunk(proto::GenerateStreamChunk {
            index: 0,
            token_ids: vec![id],
            completion_tokens: 1,
            output_logprobs: Some(proto::OutputLogProbs {
                token_logprobs: vec![-0.125],
                token_ids: vec![id],
                top_logprobs: vec![proto::TopLogProbs {
                    values: (0..candidates).map(value).collect(),
                    token_ids: (0..candidates).map(|k| id + 1 + k).collect(),
                }],
            }),
            ..Default::default()
        })),
    }
}

/// The `stop` completion of choice 0 after `tokens` tokens.
fn complete(tokens: usize) -> proto::GenerateResponse {
    proto::GenerateResponse {
        response: Some(GenerationEvent::Complete(proto::GenerateComplete {
            index: 0,
            finish_reason: "stop".to_string(),
            prompt_tokens: 8,
            completion_tokens: u32::try_from(tokens).unwrap_or(u32::MAX),
            ..Default::default()
        })),
    }
}

/// One length-prefixed gRPC data frame carrying `message`.
fn data_frame<M: ProstMessage>(message: &M) -> Bytes {
    let encoded = message.encode_to_vec();
    let mut frame = Vec::with_capacity(encoded.len() + 5);
    frame.put_u8(0);
    frame.put_u32(u32::try_from(encoded.len()).unwrap_or(u32::MAX));
    frame.extend_from_slice(&encoded);
    Bytes::from(frame)
}

/// The trailers that end a gRPC response with the status OK.
fn trailers_frame() -> Frame<Bytes> {
    let mut trailers = HeaderMap::new();
    trailers.insert("grpc-status", HeaderValue::from_static("0"));
    Frame::trailers(trailers)
}

/// A chat request whose `tool_choice` forces a call, read as the gateway reads
/// it under the JSON schema the older registry writes for such a call: the
/// request, its model's table and the prompt's tail, and the response's spec
/// with the reader the gateway assigns (`assign_for`, the regular router's).
pub struct ForcedChat {
    request: ChatCompletionRequest,
    table: Table,
    prompt: &'static str,
    schema: (String, String),
    spec: ChatResponseSpec,
}

impl ForcedChat {
    /// The request on a model whose table reads a thought (Qwen3), the
    /// reasoning separated: `tool_choice` names the declared function when
    /// `named` and requires a call otherwise.
    pub fn on_a_thought_table(named: bool) -> Self {
        Self::new(Table::Qwen3, "<|im_start|>assistant\n", named, true)
    }

    /// The same request on a model whose table reads no thought (Llama 4),
    /// which separates none.
    pub fn on_a_plain_table(named: bool) -> Self {
        Self::new(
            Table::Llama4,
            "<|header_start|>assistant<|header_end|>",
            named,
            false,
        )
    }

    fn new(table: Table, prompt: &'static str, named: bool, separate_reasoning: bool) -> Self {
        let tool_choice = if named {
            json!({"type": "function", "function": {"name": "lookup"}})
        } else {
            json!("required")
        };
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "separate_reasoning": separate_reasoning,
            "tool_choice": tool_choice,
            "tools": [{"type": "function", "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
            }}]
        }))
        .expect("a chat request");
        // The schema's text as the older registry writes it: the named function's parameters,
        // or the root of a required call's list.
        let schema = if named {
            r#"{"type":"object","properties":{"q":{"type":"string"}}}"#
        } else {
            r#"{"type":"array","minItems":1,"items":{"type":"object"}}"#
        };
        let mut forced = Self {
            spec: ChatResponseSpec::from(&request),
            request,
            table,
            prompt,
            schema: ("json_schema".to_string(), schema.to_string()),
        };
        forced.spec.symphony = forced.assignment();
        forced
    }

    /// The gateway's assignment of the response's reader, which request
    /// building asks once per request: whether a table reads the response.
    pub fn assign(&self) -> bool {
        self.assignment().is_some()
    }

    fn assignment(&self) -> Option<Assignment> {
        assign_for(
            Reading::Table(self.table.shared(), false),
            Asks::Chat(&self.request),
            Some(&self.schema),
            self.prompt,
            ProviderProfile::for_model(MODEL),
            || GrammarStart::Unknown,
        )
    }
}

/// The stream drivers, ready to run a script: one streaming processor over
/// the legacy parser factories (the tool parser `json`, the reasoning parser
/// `qwen3`), the character tokenizer, the chat request's and the Messages'
/// specs under the legacy parsers and under Symphony's Qwen3 table, the
/// completion's spec, and the loopback peer the engine streams are owned
/// against (never dispatched to: the abort on drop is disarmed).
pub struct StreamBench {
    processor: StreamingProcessor,
    tokenizer: Arc<dyn Tokenizer>,
    client: VllmEngineClient,
    peer: JoinHandle<()>,
    chat: ChatResponseSpec,
    chat_symphony: ChatResponseSpec,
    chat_logprobs: ChatResponseSpec,
    chat_symphony_logprobs: ChatResponseSpec,
    completion: CompletionResponseSpec,
    messages: MessagesResponseSpec,
    messages_symphony: MessagesResponseSpec,
}

impl StreamBench {
    /// Start the peer and build the drivers.
    pub async fn start() -> Self {
        let (url, peer) = loopback_peer().await;
        let client = VllmEngineClient::connect(&url)
            .await
            .expect("a connection to the loopback peer");
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "separate_reasoning": true,
            "tools": [{"type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object", "properties": {}}
            }}]
        }))
        .expect("a chat request");
        let chat = ChatResponseSpec::from(&request);
        let qwen3_assignment = || Assignment {
            format: Arc::new(qwen3(CallSyntax::Json)),
            prompt: "<|im_start|>assistant\n".to_string(),
            forbids_calls: Forbids::Nothing,
            past_the_end: false,
            forcing: None,
            opener: None,
            forced_ids: None,
            json_answer: false,
            span: tracing::Span::none(),
            marked: Arc::default(),
        };
        let mut chat_symphony = chat.clone();
        chat_symphony.symphony = Some(qwen3_assignment());
        // The plain chat stream asking for the tokens' logprobs with five
        // candidates each: no tools and no reasoning split, so every token
        // goes out as a bare text delta carrying them.
        let logprobs_request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "logprobs": true,
            "top_logprobs": 5
        }))
        .expect("a chat request with logprobs");
        let chat_logprobs = ChatResponseSpec::from(&logprobs_request);
        // The same stream under Symphony's Qwen3 table: every token goes out
        // as a text event of the step, which carries its logprobs.
        let mut chat_symphony_logprobs = chat_logprobs.clone();
        chat_symphony_logprobs.symphony = Some(qwen3_assignment());
        let completion_request: CompletionRequest = serde_json::from_value(json!({
            "model": MODEL, "prompt": COMPLETION_PROMPT, "stream": true
        }))
        .expect("a completion request");
        // The resolver of a configured gateway: the parser names checked
        // against the factories' registries, as `--tool-call-parser` and
        // `--reasoning-parser` are.
        let tool_parsers = ToolParserFactory::new();
        let reasoning_parsers = ReasoningParserFactory::new();
        let resolver = ParserResolver::new(
            Arc::new(WorkerRegistry::new()),
            Some("json".to_string()),
            Some("qwen3".to_string()),
            &tool_parsers,
            &reasoning_parsers,
        );
        let processor = StreamingProcessor::new(tool_parsers, reasoning_parsers, resolver, "vllm");
        let messages = MessagesResponseSpec {
            thinking_requested: true,
            thinking_text_returned: true,
            reasoning_in_prefill: false,
            prompt_tokens: 8,
            max_tokens: 4096,
            context_window: None,
            tool_choice: None,
            has_tools: false,
            history_tool_calls_count: 0,
            chat_tools: Vec::new().into(),
            stop_sequences: None,
            symphony: None,
            first_tool_call_only: false,
            parser_names: None,
        };
        let mut messages_symphony = messages.clone();
        messages_symphony.symphony = Some(qwen3_assignment());
        Self {
            processor,
            tokenizer: Arc::new(CharacterTokenizer::default()),
            client,
            peer,
            chat,
            chat_symphony,
            chat_logprobs,
            chat_symphony_logprobs,
            completion: CompletionResponseSpec::new(&completion_request, Vec::new()),
            messages,
            messages_symphony,
        }
    }

    /// A chat stream under the legacy parsers: the bytes written.
    pub async fn chat(&self, script: &EngineScript) -> usize {
        self.chat_under(&self.chat, script).await
    }

    /// A chat stream under Symphony's Qwen3 table: the bytes written.
    pub async fn chat_symphony(&self, script: &EngineScript) -> usize {
        self.chat_under(&self.chat_symphony, script).await
    }

    /// The response stream of a chat request that forces a call, under the
    /// reader the gateway assigns it: the bytes written.
    pub async fn chat_forced(&self, forced: &ForcedChat, script: &EngineScript) -> usize {
        self.chat_under(&forced.spec, script).await
    }

    /// A plain chat stream asking for the tokens' logprobs with five
    /// candidates each (a script of [`EngineScript::single_tokens_with_logprobs`]):
    /// the bytes written.
    pub async fn chat_logprobs(&self, script: &EngineScript) -> usize {
        self.chat_under(&self.chat_logprobs, script).await
    }

    /// [`Self::chat_logprobs`] under Symphony's Qwen3 table (the step writer's
    /// path): the bytes written.
    pub async fn chat_symphony_logprobs(&self, script: &EngineScript) -> usize {
        self.chat_under(&self.chat_symphony_logprobs, script).await
    }

    /// The tokenizer the streams decode with (a character is a token).
    pub fn tokenizer(&self) -> &Arc<dyn Tokenizer> {
        &self.tokenizer
    }

    async fn chat_under(&self, spec: &ChatResponseSpec, script: &EngineScript) -> usize {
        let stream = self.engine_stream(script);
        let (tx, rx) = sse_channel();
        let driven = async {
            let result = self
                .processor
                .process_streaming_chunks(
                    stream,
                    dispatch("chatcmpl-bench"),
                    Arc::clone(&self.tokenizer),
                    (None, None, false, false, false),
                    spec.clone(),
                    &tx,
                    None,
                )
                .await;
            drop(tx);
            result
        };
        let (result, bytes) = tokio::join!(driven, drain(rx));
        assert!(result.is_ok(), "the chat stream failed: {result:?}");
        bytes
    }

    /// A completion stream: the bytes written.
    pub async fn completion(&self, script: &EngineScript) -> usize {
        let stream = self.engine_stream(script);
        let (tx, rx) = sse_channel();
        let driven = async {
            let result = self
                .processor
                .process_completion_streaming_chunks(
                    stream,
                    dispatch("cmpl-bench"),
                    Arc::clone(&self.tokenizer),
                    (None, None, false, false, false),
                    &self.completion,
                    COMPLETION_PROMPT,
                    0,
                    &tx,
                    None,
                )
                .await;
            drop(tx);
            result.map(drop)
        };
        let (result, bytes) = tokio::join!(driven, drain(rx));
        assert!(result.is_ok(), "the completion stream failed: {result:?}");
        bytes
    }

    /// A generate stream: the bytes written.
    pub async fn generate(&self, script: &EngineScript) -> usize {
        let stream = self.engine_stream(script);
        let context = GenerateStreamContext {
            request_id: "gen-bench".to_string(),
            weight_version: "default".to_string(),
            return_logprob: false,
            backend_type: "bench",
            model: MODEL.to_string(),
            expected_choices: 1,
            dispatch_wait: None,
        };
        let (tx, rx) = sse_channel();
        let driven = async {
            let result = StreamingProcessor::process_generate_streaming(
                Arc::clone(&self.tokenizer),
                stream,
                context,
                &tx,
                None,
            )
            .await;
            drop(tx);
            result
        };
        let (result, bytes) = tokio::join!(driven, drain(rx));
        assert!(result.is_ok(), "the generate stream failed: {result:?}");
        bytes
    }

    /// A Messages stream with thinking requested, under the legacy parsers:
    /// the bytes written.
    pub async fn messages(&self, script: &EngineScript) -> usize {
        self.messages_under(&self.messages, script).await
    }

    /// A Messages stream with thinking requested, under Symphony's Qwen3
    /// table: the bytes written.
    pub async fn messages_symphony(&self, script: &EngineScript) -> usize {
        self.messages_under(&self.messages_symphony, script).await
    }

    async fn messages_under(&self, spec: &MessagesResponseSpec, script: &EngineScript) -> usize {
        let stream = self.engine_stream(script);
        let (tx, rx) = sse_channel();
        let driven = async {
            let result = self
                .processor
                .process_messages_streaming_chunks(
                    stream,
                    dispatch("msg-bench"),
                    Arc::clone(&self.tokenizer),
                    (None, None, false, false, false),
                    spec.clone(),
                    &tx,
                    None,
                )
                .await;
            drop(tx);
            result
        };
        let (result, bytes) = tokio::join!(driven, drain(rx));
        assert!(result.is_ok(), "the Messages stream failed: {result:?}");
        bytes
    }

    /// The engine stream of one reading of `script`: decoded by the client's
    /// codec from the pre-encoded frames, owned against the peer and never
    /// dispatched to it, so dropping it aborts nothing.
    fn engine_stream(&self, script: &EngineScript) -> ProtoStream {
        let mut codec =
            SizedProstCodec::<proto::GenerateResponse, Box<proto::GenerateResponse>>::default();
        let decoded = tonic::Streaming::new_response(
            codec.decoder(),
            StreamBody::new(script.reading()),
            StatusCode::OK,
            None,
            None,
        );
        let stream = AbortOnDropStream::new(decoded, "bench".to_string(), self.client.clone());
        stream.mark_completed();
        ProtoStream::Vllm(stream)
    }
}

impl Drop for StreamBench {
    fn drop(&mut self) {
        self.peer.abort();
    }
}

/// The dispatch of a benchmark request.
fn dispatch(request_id: &str) -> DispatchMetadata {
    DispatchMetadata {
        request_id: request_id.to_string(),
        model: MODEL.to_string(),
        created: 1_700_000_000,
        weight_version: None,
    }
}

/// The bytes the client's side reads until the sender is dropped.
async fn drain(mut rx: SseReceiver) -> usize {
    let mut bytes = 0;
    while let Some(frame) = rx.recv().await {
        bytes += frame.expect("an SSE write").len();
    }
    bytes
}

/// A loopback peer speaking HTTP/2 with no service behind it: what the
/// client connects to, so the engine streams have an owner. It serves until
/// the bench drops it.
#[expect(
    clippy::disallowed_methods,
    reason = "the peer's task is aborted with the bench that owns it; nothing waits for it"
)]
async fn loopback_peer() -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback port");
    let addr = listener.local_addr().expect("the bound address");
    let server = tokio::spawn(async move {
        let _ = Server::builder()
            .add_routes(Routes::default())
            .serve_with_incoming(TcpIncoming::from(listener))
            .await;
    });
    (format!("http://{addr}"), server)
}
