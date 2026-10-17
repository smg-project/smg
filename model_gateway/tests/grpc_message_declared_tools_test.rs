//! A chat request whose tools are all declared on messages (a provider profile that lets a
//! system or developer message declare tools) is served as a request with tools, end to end
//! through the gRPC router and a scripted worker: the gateway keeps the special tokens when it
//! decodes the engine's token ids (the engine clients ask every chat request's engine for raw
//! ids and pin the wire setting; the gateway's own decoder is where the derived setting acts),
//! so the call markers reach the tool parser and the call comes back as a tool call rather
//! than as text, whole and streamed. The tokenizer drops every `<|...|>` chunk when asked to
//! skip special tokens, as a checkpoint's vocabulary does for its control tokens, so a request
//! decoded with them skipped loses the markers. The worker records what the gateway sends.
//!
//! The requests do not ask for the reasoning split: the model's reasoning parser keeps the
//! special tokens on its own account when they do, whatever the tools say, which would hide the
//! gate under test.

#[path = "common/mod.rs"]
mod common;
#[expect(dead_code, reason = "a shared fixture; this test uses part of it")]
#[path = "common/scripted_tokenizer.rs"]
mod scripted_tokenizer;
#[path = "common/scripted_worker.rs"]
mod scripted_worker;

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{body::to_bytes, http::StatusCode, response::Response};
use llm_tokenizer::{traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{
    chat::ChatCompletionRequest, model_card::ModelCard, validated::Normalizable,
    worker::HealthCheckConfig,
};
use scripted_tokenizer::ScriptedTokenizer;
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto as ts;
use tokio::{net::TcpListener, task::JoinHandle, time::timeout};
use validator::Validate;

/// A model id under the profile that reads tools declared on system and developer messages;
/// its parsers read the channel format below.
const MODEL: &str = "kimi-k3-scripted";
const EOS: &str = "<|im_end|>";
/// The model's output: text in its response channel, then one call in its tools channel.
/// Every `<|...|>` chunk is a special token.
const OUTPUT: &str = concat!(
    "<|open|>response<|sep|>Checking.<|close|>response<|sep|>",
    "<|open|>tools<|sep|>",
    "<|open|>call tool=\"get_weather\" index=\"0\"<|sep|>",
    "<|open|>argument key=\"city\" type=\"string\"<|sep|>Paris<|close|>argument<|sep|>",
    "<|close|>call<|sep|>",
    "<|close|>tools<|sep|>",
    "<|im_end|>",
);

/// Model output as tokens: each `<|...|>` alone, other text four characters at a time.
fn tokens(output: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut rest = output;
    while !rest.is_empty() {
        let len = if rest.starts_with("<|") {
            rest.find("|>").map_or(rest.len(), |end| end + 2)
        } else {
            let special = rest.find("<|").unwrap_or(rest.len());
            rest.char_indices()
                .nth(4)
                .map_or(special, |(at, _)| at.min(special))
        };
        tokens.push(rest[..len].to_string());
        rest = &rest[len..];
    }
    tokens
}

struct Gateway {
    router: Box<dyn RouterTrait>,
    worker: JoinHandle<()>,
    requests: Arc<Mutex<Vec<ts::GenerateRequest>>>,
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.worker.abort();
    }
}

impl Gateway {
    /// `skip_special_tokens` as the worker received it with the last request: the engine is
    /// asked for raw ids for every chat request, whichever way the tools are declared.
    #[expect(clippy::unwrap_used, clippy::expect_used, reason = "test helper")]
    fn last_skip_special_tokens(&self) -> bool {
        let requests = self.requests.lock().unwrap();
        let request = requests.last().expect("the worker received a request");
        request
            .sampling_params
            .as_ref()
            .expect("sampling params")
            .skip_special_tokens
    }
}

#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "test helper; the scripted worker is aborted when the gateway is dropped"
)]
async fn serve() -> Gateway {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let chunks = tokens(OUTPUT);
    let requests = Arc::new(Mutex::new(Vec::new()));
    let worker = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: u32::try_from(chunks.len()).unwrap(),
            finish_reasons: vec!["stop"],
            generation: Default::default(),
            requests: requests.clone(),
        }
        .serve(listener),
    );
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
    // The parsers are the model's own, resolved from its name.
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(ScriptedTokenizer::from_chunks(chunks).with_special_tokens(EOS));
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
        worker,
        requests,
    }
}

fn tenant() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("test-tenant"))
}

#[expect(clippy::unwrap_used, clippy::expect_used, reason = "test helper")]
async fn read(response: impl std::future::Future<Output = Response>) -> String {
    let response = timeout(Duration::from_secs(30), response)
        .await
        .expect("request should finish");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

/// JSON payloads of an SSE body.
#[expect(clippy::unwrap_used, reason = "test helper")]
fn events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

fn tool(name: &str) -> Value {
    json!({"type": "function", "function": {"name": name, "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}}
    }}})
}

fn user() -> Value {
    json!({"role": "user", "content": "weather in Paris?"})
}

/// A system message declaring `tools`, which the model's profile reads as the request's.
fn declaring(tools: &[Value]) -> Value {
    json!({"role": "system", "content": "", "tools": tools})
}

/// The request's tools and messages for each way of declaring the tool the model calls.
fn request_level_only() -> Value {
    json!({"messages": [user()], "tools": [tool("get_weather")]})
}

fn message_level_only() -> Value {
    json!({"messages": [declaring(&[tool("get_weather")]), user()]})
}

fn both() -> Value {
    json!({"messages": [declaring(&[tool("get_weather")]), user()], "tools": [tool("get_time")]})
}

/// A chat response in one shape for unary and streaming requests.
#[derive(Debug, Default, PartialEq)]
struct Chat {
    content: String,
    calls: Vec<(String, Value)>,
    finish: String,
}

/// A chat request as the gateway's HTTP layer hands it on: normalized and validated under
/// the model's profile, without the reasoning split (see the module's note).
#[expect(clippy::unwrap_used, clippy::expect_used, reason = "test helper")]
fn chat_request(extra: Value) -> ChatCompletionRequest {
    let mut request = json!({"model": MODEL, "max_tokens": 64, "separate_reasoning": false});
    request
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let mut request: ChatCompletionRequest = serde_json::from_value(request).unwrap();
    request.normalize();
    request.validate().expect("the gateway accepts the request");
    request
}

#[expect(clippy::unwrap_used, reason = "test helper")]
async fn chat(gateway: &Gateway, extra: Value) -> Chat {
    let stream = extra["stream"] == true;
    let request = chat_request(extra);
    let body = read(gateway.router.route_chat(None, &tenant(), request, MODEL)).await;
    let arguments = |args: &str| match args {
        "" => Value::Null,
        args => serde_json::from_str(args).unwrap(),
    };
    if !stream {
        let response: Value = serde_json::from_str(&body).unwrap();
        let choice = &response["choices"][0];
        let message = &choice["message"];
        let calls = message["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|call| {
                let name = call["function"]["name"].as_str().unwrap().to_string();
                (
                    name,
                    arguments(call["function"]["arguments"].as_str().unwrap()),
                )
            })
            .collect();
        return Chat {
            content: message["content"].as_str().unwrap_or("").to_string(),
            calls,
            finish: choice["finish_reason"].as_str().unwrap().to_string(),
        };
    }
    let mut chat = Chat::default();
    let mut calls: Vec<(String, String)> = Vec::new();
    for delta in events(&body)
        .iter()
        .filter_map(|event| event["choices"].get(0))
    {
        chat.content
            .push_str(delta["delta"]["content"].as_str().unwrap_or(""));
        for call in delta["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let index = call["index"].as_u64().unwrap() as usize;
            if let Some(name) = call["function"]["name"].as_str() {
                assert_eq!(index, calls.len(), "one name per call, in order");
                calls.push((name.to_string(), String::new()));
            }
            calls[index]
                .1
                .push_str(call["function"]["arguments"].as_str().unwrap_or(""));
        }
        if let Some(finish) = delta["finish_reason"].as_str() {
            assert!(chat.finish.is_empty(), "a single finish reason");
            chat.finish = finish.to_string();
        }
    }
    chat.calls = calls
        .into_iter()
        .map(|(name, args)| (name, arguments(&args)))
        .collect();
    chat
}

fn weather_call() -> (String, Value) {
    ("get_weather".to_string(), json!({"city": "Paris"}))
}

#[expect(clippy::unwrap_used, reason = "test helper")]
fn streamed(mut extra: Value, stream: bool) -> Value {
    extra
        .as_object_mut()
        .unwrap()
        .insert("stream".to_string(), json!(stream));
    extra
}

/// The call markers survive the gateway's decoding and the call comes back as a call, wherever
/// the request declares the tool: at the request level, on a message, or both. The engine is
/// asked the same thing each time.
#[tokio::test]
async fn a_tool_declared_on_a_message_keeps_the_call_markers_and_comes_back_as_a_call() {
    let gateway = serve().await;
    for stream in [false, true] {
        for (case, extra) in [
            ("message-level only", message_level_only()),
            ("request-level only", request_level_only()),
            ("both", both()),
        ] {
            let chat = chat(&gateway, streamed(extra, stream)).await;
            assert_eq!(
                chat.calls,
                vec![weather_call()],
                "{case}, stream={stream}: {chat:?}"
            );
            assert_eq!(chat.finish, "tool_calls", "{case}, stream={stream}");
            assert_eq!(chat.content, "Checking.", "{case}, stream={stream}");
            assert!(
                gateway.last_skip_special_tokens(),
                "{case}, stream={stream}: the engine is asked for raw ids, as for every chat request"
            );
        }
    }
}

/// `tool_choice: auto`, written or defaulted, reads the same.
#[tokio::test]
async fn a_tool_declared_on_a_message_is_offered_under_an_explicit_auto() {
    let gateway = serve().await;
    let mut extra = message_level_only();
    extra["tool_choice"] = json!("auto");
    let chat = chat(&gateway, extra).await;
    assert_eq!(chat.calls, vec![weather_call()], "{chat:?}");
    assert_eq!(chat.finish, "tool_calls");
}

/// Calls switched off leave the request's own setting and parse nothing: `tool_choice: none`
/// over a message's declaration, and an empty request-level list beside one (the normalizer
/// answers the empty list with `none`, as it does without the message). The markers then go
/// with the special tokens, as the request asked, and the call's text is content.
#[tokio::test]
async fn calls_switched_off_leave_the_requests_special_token_setting() {
    let gateway = serve().await;
    let mut switched_off = message_level_only();
    switched_off["tool_choice"] = json!("none");
    let mut empty_list = message_level_only();
    empty_list["tools"] = json!([]);
    for (case, extra) in [
        ("tool_choice none", switched_off),
        ("empty list", empty_list),
    ] {
        let chat = chat(&gateway, extra).await;
        assert!(chat.calls.is_empty(), "{case}: {chat:?}");
        assert_eq!(chat.finish, "stop", "{case}");
        assert!(
            !chat.content.contains("<|open|>") && chat.content.contains("get_weather"),
            "{case}: the call's text, its markers dropped, is content: {chat:?}"
        );
    }
}
