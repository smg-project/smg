//! Parsers selected from a tokenizer `response_template`, end to end through
//! the gRPC router and a scripted worker. Every `<|...|>` chunk is a special
//! token and `<|eos|>` is the EOS token, so the tests also cover
//! special-token preservation and EOS removal by the stop decoder.

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
const EOS: &str = "<|eos|>";
const ARG_TAG: &str = "<arg name=\"(?P<key>[^\"]+)\">(?P<value>.*?)</arg>";
const OUTPUT: &str = concat!(
    "<|think|>check the map<|done|>",
    "<|say|>Paris it is.<|done|>",
    "<|tool|>get_weather<|args|><arg name=\"city\">Paris</arg><arg name=\"days\">2</arg>",
    "<|eos|>",
);

fn template(tag_pattern: &str) -> Value {
    json!({
        "start_anchor_pattern": "<\\|bot\\|>",
        "fields": {
            "thinking": {"open_pattern": "<\\|think\\|>", "close": "<|done|>", "content": "text"},
            "content": {"open_pattern": "<\\|say\\|>", "close": ["<|done|>", EOS], "content": "text"},
            "tool_calls": {
                "open_pattern": "<\\|tool\\|>(?P<name>[A-Za-z_][A-Za-z0-9_]*)<\\|args\\|>",
                "close": ["<|done|>", EOS],
                "repeats": true,
                "content": "xml-inline",
                "content_args": {"tag_pattern": tag_pattern, "value_parser": {"name": "text"}},
                "transform": {"name": "{name}", "arguments": "{content}"}
            }
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

fn tokenizer(template: Option<Value>, output: &str) -> Arc<dyn Tokenizer> {
    let tokenizer = ScriptedTokenizer::from_chunks(tokens(output)).with_special_tokens(EOS);
    match template {
        Some(template) => Arc::new(tokenizer.with_response_template(template)),
        None => Arc::new(tokenizer),
    }
}

struct Setup {
    template: Option<Value>,
    card: ModelCard,
    reasoning_parser: Option<&'static str>,
    tool_parser: Option<&'static str>,
    finish: &'static str,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            template: Some(template(ARG_TAG)),
            card: ModelCard::new(MODEL),
            // A template outranks the process-wide flags.
            reasoning_parser: Some("passthrough"),
            tool_parser: Some("qwen"),
            finish: "stop",
        }
    }
}

struct Gateway {
    router: Box<dyn RouterTrait>,
    tokenizers: Arc<TokenizerRegistry>,
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
async fn gateway(setup: Setup, output: &str) -> Gateway {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let worker = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: u32::try_from(tokens(output).len()).unwrap(),
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
    let tokenizers = Arc::new(TokenizerRegistry::new());
    load(&tokenizers, setup.template, output).await;
    let context =
        common::create_test_context_with_tokenizer_registry(config, tokenizers.clone()).await;
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
    Gateway {
        router,
        tokenizers,
        worker,
    }
}

#[expect(clippy::unwrap_used, reason = "test helper")]
async fn load(tokenizers: &TokenizerRegistry, template: Option<Value>, output: &str) {
    tokenizers.remove(MODEL);
    let tokenizer = tokenizer(template, output);
    tokenizers
        .load(MODEL, MODEL, "test", || async move { Ok(tokenizer) })
        .await
        .unwrap();
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

#[expect(clippy::unwrap_used, reason = "test helper")]
async fn chat(gateway: &Gateway, extra: Value) -> (Chat, Vec<Value>) {
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
    if !stream {
        let response: Value = serde_json::from_str(&body).unwrap();
        let choice = &response["choices"][0];
        let message = &choice["message"];
        let calls = message["tool_calls"]
            .as_array()
            .map_or_else(Vec::new, |calls| {
                calls
                    .iter()
                    .map(|call| {
                        let args = call["function"]["arguments"].as_str().unwrap();
                        let name = call["function"]["name"].as_str().unwrap().to_string();
                        (name, serde_json::from_str(args).unwrap())
                    })
                    .collect()
            });
        let chat = Chat {
            reasoning: message["reasoning_content"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            content: message["content"].as_str().unwrap_or("").to_string(),
            calls,
            finish: choice["finish_reason"].as_str().unwrap().to_string(),
        };
        return (chat, vec![]);
    }
    let events = events(&body);
    let mut chat = Chat::default();
    let mut args: Vec<(String, String)> = Vec::new();
    for delta in events.iter().filter_map(|event| event["choices"].get(0)) {
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
                assert_eq!(index, args.len(), "one name per call, in order");
                args.push((name.to_string(), String::new()));
            }
            args[index]
                .1
                .push_str(call["function"]["arguments"].as_str().unwrap_or(""));
        }
        if let Some(finish) = delta["finish_reason"].as_str() {
            assert!(chat.finish.is_empty(), "a single finish reason");
            chat.finish = finish.to_string();
        }
    }
    chat.calls = args
        .into_iter()
        .map(|(name, args)| (name, serde_json::from_str(&args).unwrap()))
        .collect();
    (chat, events)
}

/// A Messages response as `[type, text or tool input]` per content block,
/// rebuilt from the deltas for a streaming request.
#[expect(clippy::unwrap_used, reason = "test helper")]
async fn messages(gateway: &Gateway, stream: bool, with_tools: bool) -> Vec<Value> {
    let tools =
        json!([{"name": "get_weather", "input_schema": tools()[0]["function"]["parameters"]}]);
    let request = serde_json::from_value(json!({
        "model": MODEL, "max_tokens": 64, "stream": stream,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": if with_tools { tools } else { Value::Null },
    }))
    .unwrap();
    let body = read(
        gateway
            .router
            .route_messages(None, &tenant(), request, MODEL),
    )
    .await;
    assert!(!body.contains("<|"), "no framing in {body}");
    let mut blocks: Vec<Value> = Vec::new();
    if !stream {
        let response: Value = serde_json::from_str(&body).unwrap();
        blocks.extend(response["content"].as_array().unwrap().iter().cloned());
    }
    for event in events(&body) {
        if event["type"] == "content_block_start" {
            blocks.push(event["content_block"].clone());
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

fn count_deltas(events: &[Value], key: &str) -> usize {
    events
        .iter()
        .filter(|event| {
            event["choices"][0]["delta"][key]
                .as_str()
                .is_some_and(|t| !t.is_empty())
        })
        .count()
}

fn weather_call() -> (String, Value) {
    (
        "get_weather".to_string(),
        json!({"city": "Paris", "days": 2}),
    )
}

#[tokio::test]
async fn chat_splits_fields_and_calls_even_when_eos_hides_the_last_close() {
    let gateway = gateway(Setup::default(), OUTPUT).await;
    let expected = Chat {
        reasoning: "check the map".to_string(),
        content: "Paris it is.".to_string(),
        calls: vec![weather_call()],
        finish: "tool_calls".to_string(),
    };
    for extra in [json!({}), json!({"no_stop_trim": true})] {
        let mut request = json!({"tools": tools()});
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert_eq!(chat(&gateway, request.clone()).await.0, expected, "{extra}");
        request["stream"] = json!(true);
        let (streamed, events) = chat(&gateway, request).await;
        assert_eq!(streamed, expected, "{extra}");
        // Text is streamed as it arrives, not at the end.
        assert!(count_deltas(&events, "reasoning_content") >= 3);
        assert!(count_deltas(&events, "content") >= 3);
    }
}

#[tokio::test]
async fn tool_blocks_are_dropped_when_the_request_parses_no_tool_calls() {
    let gateway = gateway(Setup::default(), OUTPUT).await;
    let expected = Chat {
        reasoning: "check the map".to_string(),
        content: "Paris it is.".to_string(),
        finish: "stop".to_string(),
        ..Chat::default()
    };
    for request in [json!({}), json!({"tools": tools(), "tool_choice": "none"})] {
        for stream in [false, true] {
            let mut request = request.clone();
            request["stream"] = json!(stream);
            assert_eq!(
                chat(&gateway, request.clone()).await.0,
                expected,
                "{request}"
            );
        }
    }
    let blocks = [
        json!(["thinking", "check the map"]),
        json!(["text", "Paris it is."]),
    ];
    for stream in [false, true] {
        assert_eq!(
            messages(&gateway, stream, false).await,
            blocks,
            "stream: {stream}"
        );
    }
}

#[tokio::test]
async fn a_call_without_arguments_closed_by_eos_streams_its_name() {
    let output = "<|tool|>get_time<|args|><|eos|>";
    let gateway = gateway(Setup::default(), output).await;
    let expected = Chat {
        calls: vec![("get_time".to_string(), json!({}))],
        finish: "tool_calls".to_string(),
        ..Chat::default()
    };
    let request = json!({"tools": tools()});
    assert_eq!(chat(&gateway, request).await.0, expected);
    let request = json!({"tools": tools(), "stream": true});
    assert_eq!(chat(&gateway, request).await.0, expected);
}

#[tokio::test]
async fn an_opener_cut_off_by_the_length_limit_leaves_empty_content() {
    let setup = Setup {
        finish: "length",
        ..Setup::default()
    };
    let gateway = gateway(setup, "<|say|>").await;
    let expected = Chat {
        finish: "length".to_string(),
        ..Chat::default()
    };
    assert_eq!(chat(&gateway, json!({})).await.0, expected);
    assert_eq!(chat(&gateway, json!({"stream": true})).await.0, expected);
}

#[tokio::test]
async fn tool_choice_required_uses_the_json_schema_path() {
    let output = r#"[{"name": "get_weather", "parameters": {"city": "Paris", "days": 2}}]<|eos|>"#;
    let gateway = gateway(Setup::default(), output).await;
    let expected = Chat {
        calls: vec![weather_call()],
        finish: "tool_calls".to_string(),
        ..Chat::default()
    };
    let request = json!({"tools": tools(), "tool_choice": "required"});
    assert_eq!(chat(&gateway, request).await.0, expected);
    let request = json!({"tools": tools(), "tool_choice": "required", "stream": true});
    assert_eq!(chat(&gateway, request).await.0, expected);
}

#[tokio::test]
async fn without_reasoning_separation_the_framing_is_not_parsed() {
    // Known limitation, shared with other typed-output parsers: with
    // `separate_reasoning: false` no reasoning parser runs, special tokens
    // are skipped, and thinking text stays in the content.
    let gateway = gateway(Setup::default(), OUTPUT).await;
    let (chat, _) = chat(&gateway, json!({"separate_reasoning": false})).await;
    assert_eq!(chat.reasoning, "");
    assert!(
        chat.content.starts_with("check the mapParis it is."),
        "{chat:?}"
    );
}

#[tokio::test]
async fn a_card_override_replaces_only_its_own_parser() {
    let setup = Setup {
        card: ModelCard::new(MODEL).with_reasoning_parser("passthrough"),
        ..Setup::default()
    };
    let gateway = gateway(setup, OUTPUT).await;
    let (chat, _) = chat(&gateway, json!({"tools": tools()})).await;
    // The template tool parser still reads the calls, but thinking is no
    // longer separated from the content.
    assert_eq!(chat.reasoning, "");
    assert!(chat.content.contains("check the map"), "{chat:?}");
    assert_eq!(chat.calls, [weather_call()]);
}

#[tokio::test]
async fn an_unsupported_template_falls_back_to_the_configured_parsers() {
    let setup = Setup {
        template: Some(json!({"fields": {}})),
        reasoning_parser: Some("qwen3"),
        ..Setup::default()
    };
    let gateway = gateway(setup, "<think>deep</think>answer").await;
    let expected = Chat {
        reasoning: "deep".to_string(),
        content: "answer".to_string(),
        finish: "stop".to_string(),
        ..Chat::default()
    };
    assert_eq!(chat(&gateway, json!({})).await.0, expected);
}

#[tokio::test]
async fn a_changed_template_is_not_served_by_a_pooled_parser() {
    let param_tag = "<prm key=\"(?P<key>[^\"]+)\">(?P<value>.*?)</prm>";
    let first = "<|tool|>get_weather<|args|><arg name=\"city\">Paris</arg><|eos|>";
    let second = "<|tool|>get_weather<|args|><prm key=\"city\">Paris</prm><|eos|>";
    assert_eq!(tokens(first).len(), tokens(second).len());
    let gateway = gateway(Setup::default(), first).await;
    let expected = vec![("get_weather".to_string(), json!({"city": "Paris"}))];
    // Unary requests share a pooled tool parser per parser name.
    assert_eq!(
        chat(&gateway, json!({"tools": tools()})).await.0.calls,
        expected
    );
    load(&gateway.tokenizers, Some(template(param_tag)), second).await;
    assert_eq!(
        chat(&gateway, json!({"tools": tools()})).await.0.calls,
        expected
    );
}

#[tokio::test]
async fn responses_and_messages_use_the_template_parsers() {
    let gateway = gateway(Setup::default(), OUTPUT).await;

    let request = serde_json::from_value(json!({
        "model": MODEL, "input": "weather?", "store": false,
        "tools": [{"type": "function", "name": "get_weather", "parameters": tools()[0]["function"]["parameters"]}],
    }))
    .unwrap();
    let body = read(
        gateway
            .router
            .route_responses(None, &tenant(), request, MODEL),
    )
    .await;
    let response: Value = serde_json::from_str(&body).unwrap();
    let output = response["output"].as_array().unwrap();
    let call = output
        .iter()
        .find(|item| item["type"] == "function_call")
        .unwrap();
    assert_eq!(call["name"], "get_weather");
    let args: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"city": "Paris", "days": 2}));
    assert!(body.contains("check the map") && body.contains("Paris it is."));
    assert!(!body.contains("<|"), "no framing in {body}");

    let blocks = [
        json!(["thinking", "check the map"]),
        json!(["text", "Paris it is."]),
        json!(["tool_use", {"city": "Paris", "days": 2}]),
    ];
    for stream in [false, true] {
        assert_eq!(
            messages(&gateway, stream, true).await,
            blocks,
            "stream: {stream}"
        );
    }
}
