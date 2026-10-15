//! `/v1/completions` with `echo` and `logprobs`, through the gRPC router and a
//! scripted TokenSpeed worker: the response lists every prompt token with its
//! logprob and alternatives, then the generated tokens, unary and streamed,
//! for a prompt given as token ids or as text. A backend that returns no
//! prompt logprobs gets the request refused instead of answered with
//! positions missing.

#[path = "common/mod.rs"]
mod common;

use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{body::to_bytes, http::StatusCode, response::Response as HttpResponse};
use futures::{stream, Stream};
use llm_tokenizer::{
    traits::Tokenizer, MockTokenizer, TiktokenModel, TiktokenTokenizer, TokenizerRegistry,
};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_grpc_client::{common_proto as common_pb, tokenspeed_scheduler::tokenspeed_proto as ts};
use tokio::{net::TcpListener, task::JoinHandle, time::timeout};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};
use ts::{
    generate_response::Response as GenResp,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};

const MODEL: &str = "completion-echo-logprobs-model";
/// The one token the worker generates (`token` in the mock vocabulary) and
/// its logprob.
const GENERATED: u32 = 4;
const GENERATED_LOGPROB: f32 = -0.25;

/// A TokenSpeed worker that scores the prompt it is given: position `i > 0`
/// of the scored prompt has logprob `-i / 2`, its alternatives are the token
/// itself then ids 9, 8, 7, ... one nat apart. It generates one token.
struct EchoWorker {
    /// Whether it returns prompt logprobs (an older engine does not).
    prompt_logprobs: bool,
    requests: Arc<Mutex<Vec<ts::GenerateRequest>>>,
}

fn alternatives(first: u32, value: f32, k: usize) -> ts::TopLogProbs {
    let ids: Vec<u32> = std::iter::once(first)
        .chain((1..=9u32).rev().filter(|&id| id != first))
        .take(k)
        .collect();
    ts::TopLogProbs {
        values: (0..ids.len()).map(|j| value - j as f32).collect(),
        token_ids: ids,
    }
}

impl EchoWorker {
    fn responses(
        request: &ts::GenerateRequest,
        prompt_logprobs: bool,
    ) -> Vec<ts::GenerateResponse> {
        let ids = request
            .tokenized
            .as_ref()
            .map(|t| t.input_ids.clone())
            .unwrap_or_default();
        let k = request.top_logprobs_num.max(0) as usize;
        let start = request.logprob_start_len.unwrap_or(-1);
        let input = (request.return_logprob && start >= 0 && prompt_logprobs).then(|| {
            let scored = &ids[start as usize..];
            ts::InputLogProbs {
                token_logprobs: (0..scored.len())
                    .map(|i| ts::InputTokenLogProb {
                        value: (i > 0).then(|| -(i as f32) / 2.0),
                    })
                    .collect(),
                token_ids: scored.to_vec(),
                top_logprobs: scored
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| {
                        if i == 0 || k == 0 {
                            ts::TopLogProbs::default()
                        } else {
                            alternatives(id, -(i as f32) / 2.0, k)
                        }
                    })
                    .collect(),
            }
        });
        let output = request.return_logprob.then(|| ts::OutputLogProbs {
            token_logprobs: vec![GENERATED_LOGPROB],
            token_ids: vec![GENERATED],
            top_logprobs: vec![if k == 0 {
                ts::TopLogProbs::default()
            } else {
                alternatives(GENERATED, GENERATED_LOGPROB, k)
            }],
        });
        let prompt_tokens = ids.len() as u32;
        let mut frames = Vec::new();
        if request.stream {
            frames.push(ts::GenerateResponse {
                request_id: request.request_id.clone(),
                response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                    token_ids: vec![GENERATED],
                    prompt_tokens,
                    completion_tokens: 1,
                    output_logprobs: output.clone(),
                    input_logprobs: input.clone(),
                    ..Default::default()
                })),
            });
        }
        frames.push(ts::GenerateResponse {
            request_id: request.request_id.clone(),
            response: Some(GenResp::Complete(ts::GenerateComplete {
                output_ids: vec![GENERATED],
                finish_reason: "length".to_string(),
                prompt_tokens,
                completion_tokens: 1,
                output_logprobs: output,
                input_logprobs: input,
                ..Default::default()
            })),
        });
        frames
    }
}

type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;
type KvStream = Pin<Box<dyn Stream<Item = Result<common_pb::KvEventBatch, Status>> + Send>>;
type TokenizerStream =
    Pin<Box<dyn Stream<Item = Result<common_pb::GetTokenizerChunk, Status>> + Send>>;

#[tonic::async_trait]
impl TokenSpeedScheduler for EchoWorker {
    type GenerateStream = GenStream;
    type SubscribeKvEventsStream = KvStream;
    type GetTokenizerStream = TokenizerStream;

    #[expect(
        clippy::unwrap_used,
        reason = "test worker: a poisoned lock fails the test"
    )]
    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        let frames = Self::responses(&request, self.prompt_logprobs);
        self.requests.lock().unwrap().push(request);
        Ok(Response::new(Box::pin(stream::iter(
            frames.into_iter().map(Ok),
        ))))
    }

    async fn health_check(
        &self,
        _: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Err(Status::unimplemented("echo worker"))
    }

    async fn abort(
        &self,
        _: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        Err(Status::unimplemented("echo worker"))
    }

    async fn get_model_info(
        &self,
        _: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("echo worker"))
    }

    async fn get_server_info(
        &self,
        _: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("echo worker"))
    }

    async fn get_loads(
        &self,
        _: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("echo worker"))
    }

    async fn subscribe_kv_events(
        &self,
        _: Request<common_pb::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("echo worker"))
    }

    async fn flush_cache(
        &self,
        _: Request<common_pb::FlushCacheRequest>,
    ) -> Result<Response<common_pb::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("echo worker"))
    }

    async fn start_profile(
        &self,
        _: Request<common_pb::StartProfileRequest>,
    ) -> Result<Response<common_pb::ProfileResponse>, Status> {
        Err(Status::unimplemented("echo worker"))
    }

    async fn stop_profile(
        &self,
        _: Request<common_pb::StopProfileRequest>,
    ) -> Result<Response<common_pb::ProfileResponse>, Status> {
        Err(Status::unimplemented("echo worker"))
    }

    async fn get_tokenizer(
        &self,
        _: Request<common_pb::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("echo worker"))
    }
}

struct Gateway {
    router: Box<dyn RouterTrait>,
    requests: Arc<Mutex<Vec<ts::GenerateRequest>>>,
    worker: JoinHandle<()>,
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.worker.abort();
    }
}

impl Gateway {
    #[expect(clippy::unwrap_used, reason = "test helper")]
    fn last_request(&self) -> ts::GenerateRequest {
        self.requests.lock().unwrap().last().cloned().unwrap()
    }
}

/// The gateway with the mock vocabulary, in front of an [`EchoWorker`].
async fn serve(prompt_logprobs: bool) -> Gateway {
    serve_with(prompt_logprobs, Arc::new(MockTokenizer::new())).await
}

#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper; the worker task is aborted when the gateway is dropped"
)]
async fn serve_with(prompt_logprobs: bool, tokenizer: Arc<dyn Tokenizer>) -> Gateway {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let worker = EchoWorker {
        prompt_logprobs,
        requests: requests.clone(),
    };
    let worker = tokio::spawn(async move {
        Server::builder()
            .add_service(TokenSpeedSchedulerServer::new(worker))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("echo worker failed");
    });
    let mut config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    let tokenizers = Arc::new(TokenizerRegistry::new());
    tokenizers
        .load(MODEL, MODEL, "test", || async move { Ok(tokenizer) })
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    let worker_entry = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::TokenSpeed)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    context
        .worker_registry
        .register(Arc::new(worker_entry))
        .unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    Gateway {
        router,
        requests,
        worker,
    }
}

#[expect(clippy::unwrap_used, clippy::expect_used, reason = "test helper")]
async fn send(gateway: &Gateway, body: Value) -> (StatusCode, Vec<u8>) {
    let request = serde_json::from_value(body).unwrap();
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let response: HttpResponse = gateway
        .router
        .route_completion(None, &tenant, request, MODEL)
        .await;
    let status = response.status();
    let bytes = timeout(
        Duration::from_secs(30),
        to_bytes(response.into_body(), 1 << 20),
    )
    .await
    .expect("request should finish")
    .unwrap();
    (status, bytes.to_vec())
}

/// The data events of an SSE body, `[DONE]` excluded.
#[expect(clippy::unwrap_used, reason = "test helper")]
fn events(body: &[u8]) -> Vec<Value> {
    String::from_utf8(body.to_vec())
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

/// The logprobs of every event of choice 0, joined in order.
fn joined_logprobs(events: &[Value]) -> Value {
    let mut joined =
        json!({"tokens": [], "token_logprobs": [], "top_logprobs": [], "text_offset": []});
    for event in events {
        let Some(logprobs) = event["choices"][0].get("logprobs") else {
            continue;
        };
        for key in ["tokens", "token_logprobs", "top_logprobs", "text_offset"] {
            if let (Some(list), Some(part)) = (joined[key].as_array_mut(), logprobs[key].as_array())
            {
                list.extend(part.iter().cloned());
            }
        }
    }
    joined
}

fn request(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL,
        "prompt": [1, 2, 3],
        "echo": true,
        "logprobs": 5,
        "max_tokens": 1,
        "temperature": 0,
    });
    if let (Some(body), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        body.extend(extra.clone());
    }
    body
}

/// What a client that scores the prompt `Hello world test` (ids 1, 2, 3)
/// reads, with the generated `token`.
#[expect(clippy::expect_used, reason = "test helper")]
fn assert_echoed(logprobs: &Value) {
    assert_eq!(
        logprobs["tokens"],
        json!(["Hello", "world", "test", "token"])
    );
    assert_eq!(
        logprobs["token_logprobs"],
        json!([null, -0.5, -1.0, GENERATED_LOGPROB])
    );
    let top = logprobs["top_logprobs"].as_array().expect("top_logprobs");
    assert_eq!(top.len(), 4);
    assert!(
        top[0].is_null(),
        "the first prompt token has no alternatives"
    );
    for (position, first) in [(1, "world"), (2, "test"), (3, "token")] {
        let alternatives = top[position].as_object().expect("alternatives");
        assert_eq!(
            alternatives.len(),
            5,
            "position {position}: {alternatives:?}"
        );
        assert_eq!(alternatives.keys().next().map(String::as_str), Some(first));
    }
    assert_eq!(top[1]["world"], json!(-0.5));
    // Where each token starts in the returned `Hello world testtoken`.
    assert_eq!(logprobs["text_offset"], json!([0, 6, 12, 16]));
}

/// Every offset indexes into `text`, at the token's own text.
#[expect(clippy::expect_used, reason = "test helper")]
fn assert_offsets_index_into(text: &str, logprobs: &Value) {
    let chars: Vec<char> = text.chars().collect();
    let tokens = logprobs["tokens"].as_array().expect("tokens");
    let offsets = logprobs["text_offset"].as_array().expect("text_offset");
    assert_eq!(tokens.len(), offsets.len());
    for (token, offset) in tokens.iter().zip(offsets) {
        let token = token.as_str().expect("token text");
        let offset = offset.as_u64().expect("offset") as usize;
        let at: String = chars
            .iter()
            .skip(offset)
            .take(token.chars().count())
            .collect();
        assert_eq!(at, token, "offset {offset} of {token:?} in {text:?}");
    }
}

/// The scoring request: token ids, `echo`, five alternatives, one
/// generated token. Every prompt position is listed, unary and streamed, and
/// the engine was asked to score the prompt from its start.
#[tokio::test]
async fn an_echoed_token_prompt_lists_every_prompt_logprob() {
    let gateway = serve(true).await;

    let (status, body) = send(&gateway, request(json!({}))).await;
    let response: Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(status, StatusCode::OK, "{response}");
    let choice = &response["choices"][0];
    assert_eq!(choice["text"], "Hello world testtoken");
    assert_echoed(&choice["logprobs"]);
    assert_offsets_index_into("Hello world testtoken", &choice["logprobs"]);
    let sent = gateway.last_request();
    assert_eq!(sent.logprob_start_len, Some(0));
    assert_eq!(sent.top_logprobs_num, 5);
    assert!(sent.return_logprob);
    assert_eq!(
        sent.tokenized.map(|t| t.input_ids),
        Some(vec![1, 2, 3]),
        "token ids go to the engine as given"
    );

    let (status, body) = send(&gateway, request(json!({"stream": true}))).await;
    assert_eq!(status, StatusCode::OK);
    let events = events(&body);
    let text: String = events
        .iter()
        .filter_map(|event| event["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(text, "Hello world testtoken");
    assert_echoed(&joined_logprobs(&events));
    assert_offsets_index_into(&text, &joined_logprobs(&events));
}

/// A streamed echo whose generated token the stop decoder holds (a partial
/// stop sequence): the echo goes out alone with the prompt's logprobs, and
/// the token's logprob goes with the event that releases its text.
#[tokio::test]
async fn a_held_token_keeps_its_logprob_for_the_event_with_its_text() {
    let gateway = serve(true).await;
    let (status, body) = send(
        &gateway,
        request(json!({"stream": true, "stop": ["token!"]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let events = events(&body);
    let listed: Vec<(Value, Value)> = events
        .iter()
        .map(|event| {
            let choice = &event["choices"][0];
            (choice["text"].clone(), choice["logprobs"]["tokens"].clone())
        })
        .collect();
    assert_eq!(
        listed,
        vec![
            (json!("Hello world test"), json!(["Hello", "world", "test"])),
            (json!("token"), json!(["token"])),
            (json!(""), Value::Null),
        ]
    );
    assert_echoed(&joined_logprobs(&events));
}

/// A text prompt is tokenized and echoed as written; its tokens are listed
/// the same way.
#[tokio::test]
async fn an_echoed_text_prompt_lists_its_tokens() {
    let gateway = serve(true).await;
    let (status, body) = send(&gateway, request(json!({"prompt": "Hello world test"}))).await;
    let response: Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_echoed(&response["choices"][0]["logprobs"]);
    assert_offsets_index_into("Hello world testtoken", &response["choices"][0]["logprobs"]);
}

/// A byte-level vocabulary splits a character it has no token for into
/// byte tokens, each of which decodes to U+FFFD on its own. The offsets
/// still index into the returned text: every other token is found at its
/// offset, those after the character included.
#[tokio::test]
async fn offsets_index_into_the_text_past_a_character_split_into_bytes() {
    let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).expect("cl100k_base");
    let gateway = serve_with(true, Arc::new(tokenizer)).await;
    // cl100k_base has no token for U+1FAE0: ` 🫠` is three byte tokens.
    let prompt = "Hi 🫠 there";
    let (status, body) = send(&gateway, request(json!({"prompt": prompt}))).await;
    let response: Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(status, StatusCode::OK, "{response}");
    let choice = &response["choices"][0];
    let text = choice["text"].as_str().expect("text");
    assert!(text.starts_with(prompt), "{text:?}");
    let chars: Vec<char> = text.chars().collect();
    let tokens = choice["logprobs"]["tokens"].as_array().expect("tokens");
    let offsets = choice["logprobs"]["text_offset"]
        .as_array()
        .expect("text_offset");
    let mut split = 0;
    for (token, offset) in tokens.iter().zip(offsets) {
        let token = token.as_str().expect("token text");
        if token.contains('\u{fffd}') {
            split += 1;
            continue;
        }
        let offset = offset.as_u64().expect("offset") as usize;
        let at: String = chars
            .iter()
            .skip(offset)
            .take(token.chars().count())
            .collect();
        assert_eq!(
            at, token,
            "offset {offset} of {token:?} in {text:?}: {offsets:?}"
        );
    }
    assert_eq!(split, 3, "the emoji is split into byte tokens: {tokens:?}");
}

/// Without `echo` only the generated tokens are listed, and no prompt
/// position is scored; `logprobs: 0` lists no alternatives.
#[tokio::test]
async fn logprobs_without_echo_list_the_generated_tokens() {
    let gateway = serve(true).await;
    let (status, body) = send(&gateway, request(json!({"echo": false, "logprobs": 0}))).await;
    let response: Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(status, StatusCode::OK, "{response}");
    let logprobs = &response["choices"][0]["logprobs"];
    assert_eq!(logprobs["tokens"], json!(["token"]));
    assert_eq!(logprobs["token_logprobs"], json!([GENERATED_LOGPROB]));
    assert_eq!(logprobs["top_logprobs"], json!([null]));
    assert_eq!(logprobs["text_offset"], json!([0]));
    assert_eq!(response["choices"][0]["text"], "token");
    assert_eq!(gateway.last_request().logprob_start_len, Some(-1));

    // No logprobs asked for: none in the response.
    let (status, body) = send(&gateway, request(json!({"logprobs": null}))).await;
    let response: Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(status, StatusCode::OK, "{response}");
    assert!(response["choices"][0].get("logprobs").is_none());
}

/// A batch of token-id prompts: every choice lists its own prompt.
#[tokio::test]
async fn a_batch_of_token_prompts_lists_each_prompt() {
    let gateway = serve(true).await;
    let (status, body) = send(&gateway, request(json!({"prompt": [[1, 2], [3]]}))).await;
    let response: Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(status, StatusCode::OK, "{response}");
    let tokens: Vec<Value> = response["choices"]
        .as_array()
        .expect("choices")
        .iter()
        .map(|choice| choice["logprobs"]["tokens"].clone())
        .collect();
    assert_eq!(
        tokens,
        vec![json!(["Hello", "world", "token"]), json!(["test", "token"])]
    );
}

/// A backend that returns no prompt logprobs gets the request refused, by
/// name, before the answer (unary) or as the stream's error event.
#[tokio::test]
async fn an_echo_the_backend_cannot_score_is_refused() {
    let gateway = serve(false).await;
    let (status, body) = send(&gateway, request(json!({}))).await;
    let response: Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["error"]["code"], "prompt_logprobs_unavailable");

    let (status, body) = send(&gateway, request(json!({"stream": true}))).await;
    assert_eq!(status, StatusCode::OK);
    let events = events(&body);
    let error = events
        .iter()
        .find_map(|event| event.get("error"))
        .expect("an error event");
    assert!(
        error["message"]
            .as_str()
            .is_some_and(|m| m.contains("prompt's logprobs")),
        "{error}"
    );
    assert!(
        events
            .iter()
            .all(|event| event["choices"][0]["text"] != "Hello world test"),
        "no echo is sent without its logprobs"
    );
}

/// A token id the tokenizer does not know is refused before any engine sees
/// it.
#[tokio::test]
async fn an_unknown_token_id_is_refused() {
    let gateway = serve(true).await;
    let (status, body) = send(&gateway, request(json!({"prompt": [1, 123456]}))).await;
    let response: Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(status, StatusCode::BAD_REQUEST, "{response}");
    assert_eq!(response["error"]["code"], "invalid_prompt_token");
    assert!(gateway
        .requests
        .lock()
        .map(|r| r.is_empty())
        .unwrap_or(false));
}
