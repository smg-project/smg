//! Parsers selected from a tokenizer's `response_template`, end to end
//! through the gRPC router and a scripted worker. Every `<|...|>` chunk is a
//! special token and `<|im_end|>` is the EOS token, which the stop decoder
//! removes, so the last open region closes when the output ends.

#[path = "common/mod.rs"]
mod common;

#[expect(dead_code, reason = "a shared fixture; this test uses part of it")]
#[path = "common/scripted_tokenizer.rs"]
mod scripted_tokenizer;

#[path = "common/scripted_worker.rs"]
mod scripted_worker;

use std::{sync::Arc, time::Duration};

use axum::{body::to_bytes, http::StatusCode, response::Response};
use llm_tokenizer::{traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use scripted_tokenizer::ScriptedTokenizer;
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::{net::TcpListener, task::JoinHandle, time::timeout};

const MODEL: &str = "response-template-model";
const EOS: &str = "<|im_end|>";
/// Where the chat template leaves the assistant message: thinking is open.
const THINKING: &str = "<|im_start|>assistant\n<think>\n";
/// Thinking off: the chat template writes an empty thinking block.
const NO_THINKING: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
const OUTPUT: &str = concat!(
    "plan the call\n</think>\n\nChecking.\n",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
    "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>",
);

/// The `transformers serve` template for qwen3_5 checkpoints.
fn qwen3_5() -> Value {
    json!({
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
            "tool_calls": {
                "open_pattern": "\\s*<tool_call>\\s*<function=(?P<name>[^>\\n]+)>",
                "close_pattern": "</function>\\s*</tool_call>",
                "repeats": true,
                "content": "xml-inline",
                "content_args": {
                    "tag_pattern": "<parameter=(?P<key>[^>\\n]+)>\\s*(?P<value>.*?)\\s*</parameter>"
                },
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"}
                }
            },
            "content": {"close_pattern": "\\s*(?:<\\|im_end\\|>|<\\|endoftext\\|>)", "content": "text"}
        }
    })
}

/// Model output as tokens: each `<|...|>` alone, other text four characters
/// at a time.
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

struct Setup {
    template: Option<Value>,
    prompt_tail: &'static str,
    /// What the prompt tokens decode to in place of `prompt_tail`.
    decoded_prompt_tail: Option<&'static str>,
    output: &'static str,
    /// The output's tokens, in place of `tokens(output)`.
    chunks: Option<&'static [&'static str]>,
    finish: &'static str,
    card: ModelCard,
    reasoning_parser: Option<&'static str>,
    tool_parser: Option<&'static str>,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            template: Some(qwen3_5()),
            prompt_tail: THINKING,
            decoded_prompt_tail: None,
            output: OUTPUT,
            chunks: None,
            finish: "stop",
            card: ModelCard::new(MODEL),
            reasoning_parser: None,
            tool_parser: None,
        }
    }
}

struct Gateway {
    router: Box<dyn RouterTrait>,
    worker: JoinHandle<()>,
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.worker.abort();
    }
}

#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "test helper; the scripted worker is aborted when the gateway is dropped"
)]
async fn serve(setup: Setup) -> Gateway {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let chunks = match setup.chunks {
        Some(chunks) => chunks.iter().map(|chunk| chunk.to_string()).collect(),
        None => tokens(setup.output),
    };
    let worker = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: u32::try_from(chunks.len()).unwrap(),
            finish_reasons: vec![setup.finish],
            generation: Default::default(),
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
    config.reasoning_parser = setup.reasoning_parser.map(str::to_string);
    config.tool_call_parser = setup.tool_parser.map(str::to_string);
    let tokenizer = ScriptedTokenizer::from_chunks(chunks)
        .with_special_tokens(EOS)
        .with_prompt_tail(setup.prompt_tail)
        .with_prompt_ids(setup.decoded_prompt_tail);
    let tokenizer: Arc<dyn Tokenizer> = match setup.template {
        Some(template) => Arc::new(tokenizer.with_response_template(template)),
        None => Arc::new(tokenizer),
    };
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
        .model(setup.card)
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
    Gateway { router, worker }
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

/// The status and `error` object of a response the gateway refuses.
#[expect(clippy::unwrap_used, clippy::expect_used, reason = "test helper")]
async fn refusal(response: impl std::future::Future<Output = Response>) -> (StatusCode, Value) {
    let response = timeout(Duration::from_secs(30), response)
        .await
        .expect("request should finish");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    (status, body["error"].clone())
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

fn tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}
        }}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}
    ])
}

/// A chat response in one shape for unary and streaming requests.
#[derive(Debug, Default, PartialEq)]
struct Chat {
    reasoning: String,
    content: String,
    calls: Vec<(String, Value)>,
    finish: String,
}

fn chat_of(reasoning: &str, content: &str, calls: &[(&str, Value)], finish: &str) -> Chat {
    Chat {
        reasoning: reasoning.to_string(),
        content: content.to_string(),
        calls: calls
            .iter()
            .map(|(name, args)| (name.to_string(), args.clone()))
            .collect(),
        finish: finish.to_string(),
    }
}

#[expect(clippy::unwrap_used, reason = "test helper")]
async fn chat(gateway: &Gateway, extra: Value) -> Chat {
    let mut request = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "weather?"}],
        "max_tokens": 64,
    });
    request
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    let stream = request["stream"] == true;
    let request = serde_json::from_value(request).unwrap();
    let body = read(gateway.router.route_chat(None, &tenant(), request, MODEL)).await;
    // A call whose arguments never came reads as null.
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
        let text = |key: &str| message[key].as_str().unwrap_or("").to_string();
        return Chat {
            reasoning: text("reasoning_content"),
            content: text("content"),
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
        let text = |key: &str| delta["delta"][key].as_str().unwrap_or("").to_string();
        chat.reasoning.push_str(&text("reasoning_content"));
        chat.content.push_str(&text("content"));
        for call in delta["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let index = call["index"].as_u64().unwrap() as usize;
            if let Some(name) = call["function"]["name"].as_str() {
                assert_eq!(index, calls.len(), "one name per call, in order");
                assert!(call["id"].is_string(), "a new call has an id");
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

/// A Messages response as `[type, text or tool input]` per content block,
/// rebuilt from the deltas for a streaming request. Streamed blocks must not
/// overlap.
#[expect(clippy::unwrap_used, reason = "test helper")]
async fn messages(gateway: &Gateway, stream: bool, with_tools: bool) -> Vec<Value> {
    let body = messages_body(gateway, stream, with_tools).await;
    assert!(!body.contains("<tool_call>"), "no framing in {body}");
    let mut blocks: Vec<Value> = Vec::new();
    if !stream {
        let response: Value = serde_json::from_str(&body).unwrap();
        blocks.extend(response["content"].as_array().unwrap().iter().cloned());
    }
    let mut open = None;
    for event in events(&body) {
        if event["type"] == "content_block_start" {
            assert_eq!(open, None, "a block starts while another is open");
            assert_eq!(event["index"], blocks.len(), "block indices in order");
            open = Some(event["index"].clone());
            blocks.push(event["content_block"].clone());
        } else if event["type"] == "content_block_stop" {
            assert_eq!(open.take(), Some(event["index"].clone()));
        } else if event["type"] == "content_block_delta" {
            let (block, delta) = (blocks.last_mut().unwrap(), &event["delta"]);
            for (key, part) in [
                ("thinking", "thinking"),
                ("text", "text"),
                ("json", "partial_json"),
            ] {
                if let Some(part) = delta[part].as_str() {
                    block[key] = json!(block[key].as_str().unwrap_or("").to_string() + part);
                }
            }
        }
    }
    blocks
        .iter()
        .map(|block| {
            let input = match block["json"].as_str() {
                Some(json) => serde_json::from_str(json).unwrap(),
                None => block["input"].clone(),
            };
            let body = [&block["thinking"], &block["text"], &input];
            let body = body.into_iter().find(|v| !v.is_null()).unwrap().clone();
            json!([block["type"], body])
        })
        .collect()
}

#[expect(clippy::unwrap_used, reason = "test helper")]
async fn messages_body(gateway: &Gateway, stream: bool, with_tools: bool) -> String {
    let tools = json!([
        {"name": "get_weather", "input_schema": tools()[0]["function"]["parameters"]},
        {"name": "get_time", "input_schema": {"type": "object"}}
    ]);
    let request = serde_json::from_value(json!({
        "model": MODEL, "max_tokens": 64, "stream": stream,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": if with_tools { tools } else { Value::Null },
    }))
    .unwrap();
    read(
        gateway
            .router
            .route_messages(None, &tenant(), request, MODEL),
    )
    .await
}

/// The stop reason of a Messages response with tools.
#[expect(clippy::unwrap_used, reason = "test helper")]
async fn messages_stop_reason(gateway: &Gateway, stream: bool) -> Value {
    let body = messages_body(gateway, stream, true).await;
    if !stream {
        let response: Value = serde_json::from_str(&body).unwrap();
        return response["stop_reason"].clone();
    }
    let events = events(&body);
    let delta = events.iter().find(|e| e["type"] == "message_delta");
    delta.unwrap()["delta"]["stop_reason"].clone()
}

#[expect(clippy::unwrap_used, reason = "test helper")]
async fn responses_body(gateway: &Gateway, stream: bool, tools: Value) -> String {
    let request = serde_json::from_value(json!({
        "model": MODEL, "input": "weather?", "store": false, "stream": stream, "tools": tools,
    }))
    .unwrap();
    read(
        gateway
            .router
            .route_responses(None, &tenant(), request, MODEL),
    )
    .await
}

/// A Responses response, or the response of a stream's terminal event.
#[expect(clippy::unwrap_used, reason = "test helper")]
async fn responses(gateway: &Gateway, stream: bool, tools: Value) -> Value {
    let body = responses_body(gateway, stream, tools).await;
    if stream {
        return events(&body).pop().unwrap()["response"].clone();
    }
    serde_json::from_str(&body).unwrap()
}

/// `get_weather` as a Responses function tool.
fn weather_function() -> Value {
    let parameters = &tools()[0]["function"]["parameters"];
    json!([{"type": "function", "name": "get_weather", "parameters": parameters}])
}

fn weather() -> (&'static str, Value) {
    ("get_weather", json!({"city": "Paris", "days": 2}))
}

#[tokio::test]
async fn chat_reads_the_fields_of_the_template() {
    let gateway = serve(Setup::default()).await;
    let request = json!({"tools": tools()});
    // A complete output reads the parsed message, stripped as transformers
    // strips text fields; the call closes when the output ends.
    assert_eq!(
        chat(&gateway, request.clone()).await,
        chat_of("plan the call", "Checking.", &[weather()], "tool_calls")
    );
    let mut request = request;
    request["stream"] = json!(true);
    assert_eq!(
        chat(&gateway, request).await,
        chat_of(
            "plan the call\n",
            "\n\nChecking.",
            &[weather()],
            "tool_calls"
        )
    );
}

#[tokio::test]
async fn requests_without_tools_drop_the_calls() {
    let gateway = serve(Setup::default()).await;
    for request in [json!({}), json!({"tools": tools(), "tool_choice": "none"})] {
        assert_eq!(
            chat(&gateway, request.clone()).await,
            chat_of("plan the call", "Checking.", &[], "stop"),
            "{request}"
        );
        let mut request = request;
        request["stream"] = json!(true);
        assert_eq!(
            chat(&gateway, request.clone()).await,
            chat_of("plan the call\n", "\n\nChecking.", &[], "stop"),
            "{request}"
        );
    }
    assert_eq!(
        messages(&gateway, false, false).await,
        [
            json!(["thinking", "plan the call"]),
            json!(["text", "Checking."])
        ]
    );
    assert_eq!(
        messages(&gateway, true, false).await,
        [
            json!(["thinking", "plan the call\n"]),
            json!(["text", "\n\nChecking."])
        ]
    );
}

#[tokio::test]
async fn messages_and_responses_read_the_fields_of_the_template() {
    let gateway = serve(Setup::default()).await;
    let input = json!({"city": "Paris", "days": 2});
    assert_eq!(
        messages(&gateway, false, true).await,
        [
            json!(["thinking", "plan the call"]),
            json!(["text", "Checking."]),
            json!(["tool_use", input]),
        ]
    );
    assert_eq!(
        messages(&gateway, true, true).await,
        [
            json!(["thinking", "plan the call\n"]),
            json!(["text", "\n\nChecking."]),
            json!(["tool_use", input]),
        ]
    );

    for stream in [false, true] {
        let request = serde_json::from_value(json!({
            "model": MODEL, "input": "weather?", "store": false, "stream": stream,
            "tools": [{"type": "function", "name": "get_weather",
                       "parameters": tools()[0]["function"]["parameters"]}],
        }))
        .unwrap();
        let body = read(
            gateway
                .router
                .route_responses(None, &tenant(), request, MODEL),
        )
        .await;
        assert!(body.contains("plan the call") && body.contains("Checking."));
        assert!(body.contains("get_weather"), "{body}");
        assert!(!body.contains("<tool_call>") && !body.contains("</think>"));
        if !stream {
            let response: Value = serde_json::from_str(&body).unwrap();
            let call = response["output"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["type"] == "function_call")
                .unwrap();
            let args: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(args, input);
        }
    }
}

#[tokio::test]
async fn reasoning_and_calls_can_alternate() {
    let setup = Setup {
        output: concat!(
            "look it up\n</think>\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>",
            "<think>\nthen answer\n</think>\n\nDone.<|im_end|>",
        ),
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let request = json!({"tools": tools(), "stream": true});
    assert_eq!(
        chat(&gateway, request).await,
        chat_of(
            "look it up\n\nthen answer\n",
            "\n\nDone.",
            &[("get_time", json!({}))],
            "tool_calls"
        )
    );
    // A complete output keeps the last thinking region, as transformers does.
    assert_eq!(
        messages(&gateway, false, true).await,
        [
            json!(["thinking", "then answer"]),
            json!(["text", "Done."]),
            json!(["tool_use", {}]),
        ]
    );
    assert_eq!(
        messages(&gateway, true, true).await,
        [
            json!(["thinking", "look it up\n"]),
            json!(["tool_use", {}]),
            json!(["thinking", "\nthen answer\n"]),
            json!(["text", "\n\nDone."]),
        ]
    );
}

#[tokio::test]
async fn a_stream_names_a_call_when_its_region_opens() {
    let gateway = serve(Setup::default()).await;
    for separate_reasoning in [true, false] {
        let request = serde_json::from_value(json!({
            "model": MODEL, "messages": [{"role": "user", "content": "weather?"}],
            "max_tokens": 64, "stream": true, "tools": tools(),
            "separate_reasoning": separate_reasoning,
        }))
        .unwrap();
        let body = read(gateway.router.route_chat(None, &tenant(), request, MODEL)).await;
        let events = events(&body);
        let call = |event: &Value| event["choices"][0]["delta"]["tool_calls"][0].clone();
        let named = events.iter().map(call).position(|call| !call.is_null());
        let args = events.iter().map(call).position(|call| {
            call["function"]["arguments"]
                .as_str()
                .is_some_and(|args| !args.is_empty())
        });
        // The name comes with the token after the opener, which could grow
        // until then; the arguments when the region closes, here at the end.
        let (named, args) = (named.unwrap(), args.unwrap());
        assert_eq!(call(&events[named])["function"]["name"], "get_weather");
        assert!(call(&events[named])["function"]["arguments"].is_null());
        assert!(named < args, "{events:?}");
    }
}

/// qwen3_5 output whose call closes in the chunk that goes on with the answer.
const CLOSE_THEN_ANSWER: &[&str] = &[
    "plan the call\n</think>\n\nChecking.\n",
    "<tool_call>\n<function=get_weather>\n",
    "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n",
    "</function>\n</tool_call>\n\nDone.",
    "<|im_end|>",
];

#[tokio::test]
async fn output_after_a_call_in_one_chunk_follows_its_arguments() {
    let setup = Setup {
        chunks: Some(CLOSE_THEN_ANSWER),
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let stream = json!({"tools": tools(), "stream": true});
    assert_eq!(
        chat(&gateway, stream).await,
        chat_of(
            "plan the call\n",
            "\n\nChecking.\n\nDone.",
            &[weather()],
            "tool_calls"
        )
    );
    // The text after the close starts a block after the call's.
    let input = json!({"city": "Paris", "days": 2});
    assert_eq!(
        messages(&gateway, true, true).await,
        [
            json!(["thinking", "plan the call\n"]),
            json!(["text", "\n\nChecking."]),
            json!(["tool_use", input]),
            json!(["text", "\n\nDone."]),
        ]
    );
    let events = events(&responses_body(&gateway, true, weather_function()).await);
    let position = |kind: &str, text: &str| {
        events.iter().position(|e| {
            e["type"] == kind
                && e["delta"]
                    .as_str()
                    .is_some_and(|delta| delta.contains(text))
        })
    };
    let args = position("response.function_call_arguments.delta", "Paris").unwrap();
    let done = position("response.output_text.delta", "Done.").unwrap();
    assert!(args < done, "{events:?}");
}

/// A template whose opener names each call, holding JSON arguments.
fn json_calls() -> Value {
    json!({
        "start_anchor_pattern": "<\\|bot\\|>",
        "fields": {
            "tool_calls": {
                "open_pattern": "<\\|call\\|>(?P<name>\\w+)",
                "close": "<|end_call|>",
                "repeats": true,
                "content": "json",
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"}
                }
            },
            "content": {}
        }
    })
}

#[tokio::test]
async fn a_streamed_call_cut_short_keeps_its_name() {
    let setup = Setup {
        template: Some(json_calls()),
        prompt_tail: "<|bot|>",
        output: "<|call|>get_weather{\"city\": \"Par",
        finish: "length",
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    // A stream reported the name before the output ended inside the call;
    // the region, from its opener, is content after it.
    let region = "<|call|>get_weather{\"city\": \"Par";
    let stream = json!({"tools": tools(), "stream": true});
    assert_eq!(
        chat(&gateway, stream).await,
        chat_of("", region, &[("get_weather", Value::Null)], "length")
    );
    assert_eq!(
        messages(&gateway, true, true).await,
        [json!(["tool_use", {}]), json!(["text", region])]
    );
    assert_eq!(messages_stop_reason(&gateway, true).await, "max_tokens");
    // A complete output reads the region as content too, and has no call.
    let unary = chat(&gateway, json!({"tools": tools()})).await;
    assert_eq!(unary, chat_of("", region, &[], "length"));
    assert_eq!(messages_stop_reason(&gateway, false).await, "max_tokens");
    for stream in [false, true] {
        let response = responses(&gateway, stream, weather_function()).await;
        assert_eq!(response["status"], "incomplete", "{response}");
        let calls: Vec<_> = response["output"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item["type"] == "function_call")
            .map(|item| {
                (
                    item["name"].clone(),
                    item["arguments"].clone(),
                    item["status"].clone(),
                )
            })
            .collect();
        let expected = if stream {
            vec![(json!("get_weather"), json!(""), json!("incomplete"))]
        } else {
            vec![]
        };
        assert_eq!(calls, expected, "{response}");
    }
}

#[tokio::test]
async fn a_streamed_call_that_does_not_parse_keeps_its_name() {
    let setup = Setup {
        template: Some(json_calls()),
        prompt_tail: "<|bot|>",
        output: "<|call|>get_weather{\"city\": Paris}<|end_call|>Sorry.<|im_end|>",
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    // The region, from its opener, is content after the name, and so is the
    // rest of the output, whatever the chunks.
    let rest = "<|call|>get_weather{\"city\": Paris}<|end_call|>Sorry.";
    let stream = json!({"tools": tools(), "stream": true});
    assert_eq!(
        chat(&gateway, stream).await,
        chat_of("", rest, &[("get_weather", Value::Null)], "tool_calls")
    );
    assert_eq!(
        messages(&gateway, true, true).await,
        [json!(["tool_use", {}]), json!(["text", rest])]
    );
    assert_eq!(messages_stop_reason(&gateway, true).await, "tool_use");
    let response = responses(&gateway, true, weather_function()).await;
    assert_eq!(response["status"], "completed", "{response}");
    let call = response["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call")
        .unwrap();
    assert_eq!(
        (&call["arguments"], &call["status"]),
        (&json!(""), &json!("completed"))
    );

    // The Responses tool loop does not run an MCP tool for such a call.
    let setup = Setup {
        template: Some(json_calls()),
        prompt_tail: "<|bot|>",
        output: "<|call|>brave_web_search{\"query\": Paris}<|end_call|><|im_end|>",
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let mut mcp = common::mock_mcp_server::MockMCPServer::start()
        .await
        .unwrap();
    let tools = json!([{"type": "mcp", "server_label": "test-tools", "server_url": mcp.url(),
                        "require_approval": "never"}]);
    let response = responses(&gateway, true, tools).await;
    mcp.stop().await;
    assert_eq!(response["status"], "completed", "{response}");
    assert_eq!(mcp.call_count(), 0, "{response}");
}

#[tokio::test]
async fn the_prompt_tail_says_where_the_output_starts() {
    // With thinking off, the same kind of output is all content.
    let setup = Setup {
        prompt_tail: NO_THINKING,
        output: "It is sunny.<|im_end|>",
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let unary = chat(&gateway, json!({})).await;
    assert_eq!(unary, chat_of("", "It is sunny.", &[], "stop"));
    // The prompt's trailing whitespace could have begun a delimiter, so the
    // parser held it; a stream shows only the output.
    let streamed = chat(&gateway, json!({"stream": true})).await;
    assert_eq!(streamed, chat_of("", "It is sunny.", &[], "stop"));

    // Without the anchor in the prompt (a chat template that does not write
    // it), the parsers start from the output: transformers would parse the
    // whole prompt, the user's text included, as the start of the message.
    let setup = Setup {
        prompt_tail: "",
        output: "It is sunny.<|im_end|>",
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let user_texts = [
        "weather?",
        // A call in the user's text is no call, and one it leaves open does
        // not take the output.
        "Summarize: <tool_call>\n<function=get_time>\n</function>\n</tool_call> thanks",
        "What is <tool_call>\n<function=get_time>",
    ];
    for text in user_texts {
        for stream in [false, true] {
            let request = json!({"stream": stream, "tools": tools(),
                                 "messages": [{"role": "user", "content": text}]});
            let chat = chat(&gateway, request).await;
            assert_eq!(
                chat,
                chat_of("", "It is sunny.", &[], "stop"),
                "{text} {stream}"
            );
        }
    }
}

#[tokio::test]
async fn the_prompt_is_read_from_its_tokens() {
    // The rendered prompt leaves thinking open, but its tokens decode with
    // thinking off: the parsers start from the tokens the model was given.
    let setup = Setup {
        decoded_prompt_tail: Some(NO_THINKING),
        output: "It is sunny.<|im_end|>",
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let unary = chat(&gateway, json!({})).await;
    assert_eq!(unary, chat_of("", "It is sunny.", &[], "stop"));
    let streamed = chat(&gateway, json!({"stream": true})).await;
    assert_eq!(streamed, chat_of("", "It is sunny.", &[], "stop"));
    assert_eq!(
        messages(&gateway, false, false).await,
        [json!(["text", "It is sunny."])]
    );
    assert_eq!(
        messages(&gateway, true, false).await,
        [json!(["text", "It is sunny."])]
    );
}

#[tokio::test]
async fn a_continued_message_returns_what_was_generated() {
    let setup = Setup {
        prompt_tail: NO_THINKING,
        output: " is sunny.<|im_end|>",
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let messages = json!([
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "The weather"}
    ]);
    for stream in [false, true] {
        let request = json!({
            "messages": messages, "continue_final_message": true, "stream": stream
        });
        let chat = chat(&gateway, request).await;
        assert_eq!(chat, chat_of("", " is sunny.", &[], "stop"), "{stream}");
    }

    // The end of the message the client sent could have begun a delimiter,
    // so the parser held it; it is not sent back with the output.
    let cases = [
        ("The weather ", "is sunny.<|im_end|>", "is sunny."),
        ("so x <", "= 5<|im_end|>", "= 5"),
    ];
    for (prefix, output, generated) in cases {
        let setup = Setup {
            prompt_tail: NO_THINKING,
            output,
            ..Setup::default()
        };
        let gateway = serve(setup).await;
        let messages = json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": prefix}
        ]);
        for stream in [false, true] {
            let request = json!({
                "messages": messages, "continue_final_message": true, "stream": stream
            });
            let chat = chat(&gateway, request).await;
            assert_eq!(
                chat,
                chat_of("", generated, &[], "stop"),
                "{prefix:?} {stream}"
            );
        }
    }
}

#[tokio::test]
async fn a_forced_output_the_template_cannot_frame_is_refused() {
    // A forced tool choice or a JSON response_format would be decoded as JSON
    // from the first token, which opens no region of a template whose content
    // needs an opener.
    const TOOL: &str = "unsupported_tool_choice";
    const FORMAT: &str = "unsupported_response_format";
    let setup = Setup {
        template: Some(framed()),
        prompt_tail: "<|bot|>",
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let tenant = tenant();
    let check = |(status, error): (StatusCode, Value), code: &str, case: &Value| {
        assert_eq!(status, StatusCode::BAD_REQUEST, "{case}: {error}");
        assert_eq!(error["code"], code, "{case}: {error}");
        let message = error["message"].as_str().unwrap();
        assert!(message.contains("response template"), "{message}");
    };
    let named = json!({"type": "function", "function": {"name": "get_weather"}});
    let allowed = json!({"type": "allowed_tools", "mode": "required",
                         "tools": [{"type": "function", "name": "get_weather"}]});
    let schema = json!({"type": "json_schema",
                        "json_schema": {"name": "weather", "schema": {"type": "object"}}});
    let chat_cases = [
        (json!({"tools": tools(), "tool_choice": "required"}), TOOL),
        (json!({"tools": tools(), "tool_choice": named}), TOOL),
        (json!({"tools": tools(), "tool_choice": allowed}), TOOL),
        (json!({"response_format": schema}), FORMAT),
        (json!({"response_format": {"type": "json_object"}}), FORMAT),
    ];
    for (extra, code) in chat_cases {
        for stream in [false, true] {
            let mut request = json!({
                "model": MODEL, "max_tokens": 64, "stream": stream,
                "messages": [{"role": "user", "content": "weather?"}],
            });
            request
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let case = request.clone();
            let request = serde_json::from_value(request).unwrap();
            let response = gateway.router.route_chat(None, &tenant, request, MODEL);
            check(refusal(response).await, code, &case);
        }
    }

    let weather_tool = json!([{"name": "get_weather",
                               "input_schema": tools()[0]["function"]["parameters"]}]);
    for tool_choice in [
        json!({"type": "any"}),
        json!({"type": "tool", "name": "get_weather"}),
    ] {
        for stream in [false, true] {
            let case = json!({
                "model": MODEL, "max_tokens": 64, "stream": stream, "tools": weather_tool,
                "tool_choice": tool_choice, "messages": [{"role": "user", "content": "weather?"}],
            });
            let request = serde_json::from_value(case.clone()).unwrap();
            let response = gateway.router.route_messages(None, &tenant, request, MODEL);
            check(refusal(response).await, TOOL, &case);
        }
    }

    let format = json!({"format": {"type": "json_schema", "name": "weather",
                                   "schema": {"type": "object"}}});
    let case = json!({"model": MODEL, "input": "weather?", "store": false, "text": format});
    let request = serde_json::from_value(case.clone()).unwrap();
    let response = gateway
        .router
        .route_responses(None, &tenant, request, MODEL);
    check(refusal(response).await, FORMAT, &case);
}

#[tokio::test]
async fn tool_choice_required_reads_the_content_as_json() {
    let setup = Setup {
        prompt_tail: NO_THINKING,
        output: r#"[{"name": "get_weather", "parameters": {"city": "Paris", "days": 2}}]<|im_end|>"#,
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    for stream in [false, true] {
        let request = json!({"tools": tools(), "tool_choice": "required", "stream": stream});
        let chat = chat(&gateway, request).await;
        assert_eq!(chat.calls, [weather()].map(|(n, a)| (n.to_string(), a)));
        assert_eq!(chat.finish, "tool_calls");
    }
}

#[tokio::test]
async fn a_json_response_format_is_content_without_a_content_opener() {
    // qwen3_5's content has no opener, so JSON from the first token is content.
    let setup = Setup {
        prompt_tail: NO_THINKING,
        output: r#"{"city": "Paris"}<|im_end|>"#,
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let format = json!({"type": "json_schema",
                        "json_schema": {"name": "weather", "schema": {"type": "object"}}});
    for stream in [false, true] {
        let request = json!({"response_format": format, "stream": stream});
        let expected = chat_of("", r#"{"city": "Paris"}"#, &[], "stop");
        assert_eq!(chat(&gateway, request).await, expected);
    }
}

#[tokio::test]
async fn without_reasoning_separation_reasoning_stays_in_the_content() {
    let gateway = serve(Setup::default()).await;
    // The tool parser reads the template and keeps reasoning as content.
    for stream in [false, true] {
        let request = json!({"tools": tools(), "separate_reasoning": false, "stream": stream});
        assert_eq!(
            chat(&gateway, request).await,
            chat_of(
                "",
                "plan the call\n\n\nChecking.",
                &[weather()],
                "tool_calls"
            )
        );
    }
    // Without tools no parser runs.
    let chat = chat(&gateway, json!({"separate_reasoning": false})).await;
    assert!(
        chat.content.starts_with("plan the call\n</think>") && chat.content.contains("<tool_call>"),
        "{chat:?}"
    );
}

/// A template whose fields all open with special tokens, content too.
fn framed() -> Value {
    json!({
        "start_anchor_pattern": "<\\|bot\\|>",
        "fields": {
            "thinking": {"open": "<|think|>", "close": "<|done|>"},
            "content": {"open": "<|say|>", "close": "<|done|>"},
            "tool_calls": {
                "open_pattern": "<\\|tool\\|>(?P<name>[A-Za-z_][A-Za-z0-9_]*)<\\|args\\|>",
                "close": "<|done|>",
                "repeats": true,
                "content": "xml-inline",
                "content_args": {"tag_pattern": "<arg name=\"(?P<key>[^\"]+)\">(?P<value>.*?)</arg>"},
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"}
                }
            }
        }
    })
}

#[tokio::test]
async fn special_tokens_reach_the_parsers() {
    let setup = Setup {
        template: Some(framed()),
        prompt_tail: "<|bot|>",
        output: concat!(
            "<|think|>check the map<|done|><|say|>Paris it is.<|done|>",
            "<|tool|>get_weather<|args|><arg name=\"city\">Paris</arg>",
            "<arg name=\"days\">2</arg><|done|><|im_end|>",
        ),
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let expected = chat_of("check the map", "Paris it is.", &[weather()], "tool_calls");
    for stream in [false, true] {
        let request = json!({"tools": tools(), "stream": stream});
        assert_eq!(chat(&gateway, request).await, expected);
    }
}

#[tokio::test]
async fn a_server_call_without_json_arguments_does_not_run() {
    // A template can make a call's arguments text, which is not JSON: the
    // Responses tool loop does not run the MCP tool with them.
    let template = json!({
        "start_anchor_pattern": "<\\|bot\\|>",
        "fields": {
            "tool_calls": {
                "open_pattern": "<\\|tool\\|>(?P<name>\\w+)<\\|args\\|>",
                "close": "<|done|>",
                "repeats": true,
                "transform": {
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"}
                }
            },
            "content": {}
        }
    });
    let setup = Setup {
        template: Some(template),
        prompt_tail: "<|bot|>",
        output: "<|tool|>brave_web_search<|args|>weather in Paris<|done|><|im_end|>",
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let mut mcp = common::mock_mcp_server::MockMCPServer::start()
        .await
        .unwrap();
    let tools = json!([{"type": "mcp", "server_label": "test-tools", "server_url": mcp.url(),
                        "require_approval": "never"}]);
    for stream in [false, true] {
        let response = responses(&gateway, stream, tools.clone()).await;
        assert_eq!(response["status"], "completed", "{response}");
        let output = response["output"].as_array().unwrap();
        assert!(
            output.iter().all(|item| item["type"] == "mcp_list_tools"),
            "{response}"
        );
    }
    mcp.stop().await;
    assert_eq!(mcp.call_count(), 0);
}

#[tokio::test]
async fn explicit_parsers_turn_the_template_off() {
    // A configured reasoning parser: the template selects neither parser.
    let setup = Setup {
        output: "<think>deep</think>answer<|im_end|>",
        prompt_tail: "",
        reasoning_parser: Some("qwen3"),
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    assert_eq!(
        chat(&gateway, json!({})).await,
        chat_of("deep", "answer", &[], "stop")
    );

    // A card override for the tool parser: the reasoning is not separated.
    let setup = Setup {
        card: ModelCard::new(MODEL).with_tool_parser("json"),
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let chat = chat(&gateway, json!({})).await;
    assert_eq!(chat.reasoning, "");
    assert!(chat.content.contains("</think>"), "{chat:?}");
}

#[tokio::test]
async fn a_template_the_parsers_cannot_use_falls_back() {
    // Look-around is outside the regex subset: the template is refused and
    // the output passes through as before.
    let mut template = qwen3_5();
    template["fields"]["thinking"] = json!({"open_pattern": "<think>(?=\\n)", "close": "</think>"});
    let setup = Setup {
        template: Some(template),
        ..Setup::default()
    };
    let gateway = serve(setup).await;
    let chat = chat(&gateway, json!({})).await;
    assert_eq!(chat.reasoning, "");
    assert!(
        chat.content.starts_with("plan the call\n</think>"),
        "{chat:?}"
    );
}

/// The `transformers serve` template for qwen2 checkpoints, which names a call
/// inside its JSON.
fn qwen2() -> Value {
    json!({
        "defaults": {"role": "assistant"},
        "start_anchor": "<|im_start|>assistant\n",
        "fields": {
            "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
            "tool_calls": {
                "open_pattern": "\\s*<tool_call>", "close": "</tool_call>", "repeats": true,
                "content": "json", "transform": {"type": "function", "function": "{content}"}
            },
            "content": {
                "close_pattern": "\\s*(?:<\\|im_end\\|>|<\\|endoftext\\|>|<\\|eot_id\\|>)",
                "content": "text"
            }
        }
    })
}

/// The thinking, the text and the number of tool calls of a Messages response.
#[expect(clippy::unwrap_used, reason = "test helper")]
async fn messages_text(gateway: &Gateway, stream: bool) -> (String, String, usize) {
    let request = serde_json::from_value(json!({
        "model": MODEL, "max_tokens": 64, "stream": stream, "tools": [
            {"name": "get_weather", "input_schema": tools()[0]["function"]["parameters"]}],
        "messages": [{"role": "user", "content": "weather?"}],
    }))
    .unwrap();
    let body = read(
        gateway
            .router
            .route_messages(None, &tenant(), request, MODEL),
    )
    .await;
    let (mut thinking, mut text, mut calls) = (String::new(), String::new(), 0);
    let blocks: Vec<Value> = if stream {
        let events = events(&body);
        let starts = events.iter().filter(|e| e["type"] == "content_block_start");
        calls += starts
            .filter(|e| e["content_block"]["type"] == "tool_use")
            .count();
        events
            .iter()
            .filter(|e| e["type"] == "content_block_delta")
            .map(|e| e["delta"].clone())
            .collect()
    } else {
        let response: Value = serde_json::from_str(&body).unwrap();
        response["content"].as_array().unwrap().clone()
    };
    for block in blocks {
        thinking.push_str(block["thinking"].as_str().unwrap_or(""));
        text.push_str(block["text"].as_str().unwrap_or(""));
        calls += usize::from(block["type"] == "tool_use");
    }
    (thinking, text, calls)
}

/// The reasoning, the output text, the function calls and the status of a
/// Responses response.
#[expect(clippy::unwrap_used, reason = "test helper")]
async fn responses_text(gateway: &Gateway, stream: bool) -> (String, String, usize, Value) {
    let request = serde_json::from_value(json!({
        "model": MODEL, "input": "weather?", "store": false, "stream": stream,
        "tools": [{"type": "function", "name": "get_weather",
                   "parameters": tools()[0]["function"]["parameters"]}],
    }))
    .unwrap();
    let body = read(
        gateway
            .router
            .route_responses(None, &tenant(), request, MODEL),
    )
    .await;
    let response = if stream {
        events(&body).pop().unwrap()["response"].clone()
    } else {
        serde_json::from_str(&body).unwrap()
    };
    let (mut reasoning, mut text, mut calls) = (String::new(), String::new(), 0);
    for item in response["output"].as_array().unwrap() {
        for part in item["content"].as_array().into_iter().flatten() {
            let part_text = part["text"].as_str().unwrap_or("");
            match item["type"].as_str() {
                Some("reasoning") => reasoning.push_str(part_text),
                _ => text.push_str(part_text),
            }
        }
        calls += usize::from(item["type"] == "function_call");
    }
    (reasoning, text, calls, response["status"].clone())
}

#[tokio::test]
async fn a_region_that_fails_is_content_in_every_api() {
    // A call whose JSON does not parse, and one the output ends in: the
    // region, from its opener on, and the rest of the output are content, in
    // a stream as in a complete response; the reasoning before it stays
    // reasoning.
    let outputs = [
        concat!(
            "secret plan</think>Checking.\n<tool_call>\n",
            "{\"name\": \"get_weather\", \"arguments\": {\"city\": Paris}}\n</tool_call>\nSorry.<|im_end|>"
        ),
        "secret plan</think>Checking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Par",
    ];
    for output in outputs {
        let setup = Setup {
            template: Some(qwen2()),
            output,
            ..Setup::default()
        };
        let gateway = serve(setup).await;
        let content = output
            .trim_start_matches("secret plan</think>")
            .trim_end_matches(EOS);
        for stream in [false, true] {
            let request = json!({"tools": tools(), "stream": stream});
            assert_eq!(
                chat(&gateway, request).await,
                chat_of("secret plan", content, &[], "stop"),
                "{output} {stream}"
            );
            assert_eq!(
                messages_text(&gateway, stream).await,
                ("secret plan".to_string(), content.to_string(), 0),
                "{output} {stream}"
            );
            let (reasoning, text, calls, status) = responses_text(&gateway, stream).await;
            assert_eq!(
                (reasoning.as_str(), text.as_str(), calls, status),
                ("secret plan", content, 0, json!("completed")),
                "{output} {stream}"
            );
        }
    }
}
