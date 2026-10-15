//! A model read through the table its tokenizer's `response_template` compiles into, end to end
//! through the gRPC router and a scripted worker: the parser name `hf` under `--parser symphony`.
//! The tokenizer ships transformers' `serve` template for Qwen 3.5 and the Qwen3.5-9B chat
//! template, so the table has a layout and writes structural tags. Every `<|...|>` chunk is a
//! special token and `<|im_end|>` is the EOS token, which the stop decoder removes.

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
    model_card::ModelCard,
    worker::{ConstrainedDecoding, HealthCheckConfig},
};
use scripted_tokenizer::ScriptedTokenizer;
use serde_json::{json, Value};
use smg::{
    config::{ParserChoice, RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto as ts;
use tokio::{net::TcpListener, task::JoinHandle, time::timeout};

const MODEL: &str = "response-template-model";
const EOS: &str = "<|im_end|>";
/// Where the chat template leaves the assistant message: thinking is open.
const THINKING: &str = "<|im_start|>assistant\n<think>\n";
/// A Qwen 3.5 output in the template's format: a thought, text, and one call.
const OUTPUT: &str = concat!(
    "plan the call\n</think>\n\nChecking.\n",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n",
    "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>",
);

const COMPILED: &str =
    include_str!("../../crates/response_template/tests/fixtures/compiled/templates.jsonl");
const CHAT: &str =
    include_str!("../../crates/response_template/tests/fixtures/chat_templates/templates.jsonl");

#[expect(clippy::expect_used, reason = "test fixture")]
fn fixture(rows: &str, id: &str) -> Value {
    rows.lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("a fixture row"))
        .find(|row| row["id"] == id)
        .expect("the fixture row")
}

/// transformers' `serve` response template for Qwen 3.5 checkpoints.
fn qwen3_5() -> Value {
    fixture(COMPILED, "serve/qwen3_5")["template"].clone()
}

/// The Qwen3.5-9B chat template.
#[expect(clippy::expect_used, reason = "test fixture")]
fn qwen3_5_chat_template() -> String {
    fixture(CHAT, "Qwen/Qwen3.5-9B")["template"]
        .as_str()
        .expect("a Jinja template")
        .to_string()
}

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

struct Setup {
    /// The tokenizer config's `response_template`, if it ships one.
    template: Option<Value>,
    /// Whether the tokenizer renders prompts with the Qwen3.5-9B chat template, which gives the
    /// table its layout; the mock's own rendering otherwise.
    chat_template: bool,
    prompt_tail: &'static str,
    output: &'static str,
    finish: &'static str,
    parser: ParserChoice,
    reasoning_parser: Option<&'static str>,
    tool_parser: Option<&'static str>,
    /// The worker's constrained-decoding policy, set on its spec.
    constrained_decoding: Option<ConstrainedDecoding>,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            template: Some(qwen3_5()),
            chat_template: true,
            prompt_tail: THINKING,
            output: OUTPUT,
            finish: "stop",
            parser: ParserChoice::Symphony,
            reasoning_parser: Some("hf"),
            tool_parser: Some("hf"),
            constrained_decoding: None,
        }
    }
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
    /// The constraint of the last request the worker received, as `(kind, value)`.
    #[expect(clippy::unwrap_used, reason = "test helper")]
    fn last_constraint(&self) -> Option<(String, String)> {
        let requests = self.requests.lock().unwrap();
        let params = requests.last()?.sampling_params.as_ref()?;
        params.constraint.as_ref().map(|constraint| {
            use ts::sampling_params::Constraint;
            match constraint {
                Constraint::StructuralTag(tag) => ("structural_tag".to_string(), tag.clone()),
                Constraint::JsonSchema(schema) => ("json_schema".to_string(), schema.clone()),
                Constraint::Regex(regex) => ("regex".to_string(), regex.clone()),
                Constraint::EbnfGrammar(grammar) => ("ebnf".to_string(), grammar.clone()),
            }
        })
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
    let chunks = tokens(setup.output);
    let requests = Arc::new(Mutex::new(Vec::new()));
    let worker = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: u32::try_from(chunks.len()).unwrap(),
            finish_reasons: vec![setup.finish],
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
    config.parser = setup.parser;
    config.reasoning_parser = setup.reasoning_parser.map(str::to_string);
    config.tool_call_parser = setup.tool_parser.map(str::to_string);
    // The chat template renders the generation prompt itself; the mock's own rendering gets
    // the tail the template would write.
    let mut tokenizer = ScriptedTokenizer::from_chunks(chunks).with_special_tokens(EOS);
    tokenizer = if setup.chat_template {
        tokenizer
            .with_chat_template(&qwen3_5_chat_template())
            .unwrap()
    } else {
        tokenizer.with_prompt_tail(setup.prompt_tail)
    };
    if let Some(template) = setup.template {
        tokenizer = tokenizer.with_response_template(template);
    }
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(tokenizer);
    let tokenizers = Arc::new(TokenizerRegistry::new());
    tokenizers
        .load(MODEL, MODEL, "test", || async move { Ok(tokenizer) })
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    let mut worker_entry = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::TokenSpeed)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        });
    if let Some(policy) = setup.constrained_decoding {
        worker_entry = worker_entry.constrained_decoding(policy);
    }
    let worker_entry = worker_entry.build();
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

#[expect(clippy::unwrap_used, reason = "test helper")]
fn chat_request(extra: Value) -> openai_protocol::chat::ChatCompletionRequest {
    let mut request = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "weather?"}],
        "max_tokens": 64,
    });
    request
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    serde_json::from_value(request).unwrap()
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

fn weather() -> (String, Value) {
    (
        "get_weather".to_string(),
        json!({"city": "Paris", "days": 2}),
    )
}

fn read_chat() -> Chat {
    Chat {
        reasoning: "plan the call".to_string(),
        content: "Checking.".to_string(),
        calls: vec![weather()],
        finish: "tool_calls".to_string(),
    }
}

/// A Messages response as `[type, text or tool input]` per content block, rebuilt from the
/// deltas for a streaming request, and its stop reason.
#[expect(clippy::unwrap_used, reason = "test helper")]
async fn messages(gateway: &Gateway, stream: bool) -> (Vec<Value>, Value) {
    let tools = json!([
        {"name": "get_weather", "input_schema": tools()[0]["function"]["parameters"]},
        {"name": "get_time", "input_schema": {"type": "object"}}
    ]);
    let request = serde_json::from_value(json!({
        "model": MODEL, "max_tokens": 64, "stream": stream,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": tools,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }))
    .unwrap();
    let body = read(
        gateway
            .router
            .route_messages(None, &tenant(), request, MODEL),
    )
    .await;
    assert!(!body.contains("<tool_call>"), "no framing in {body}");
    let mut blocks: Vec<Value> = Vec::new();
    let mut stop_reason = Value::Null;
    if !stream {
        let response: Value = serde_json::from_str(&body).unwrap();
        blocks.extend(response["content"].as_array().unwrap().iter().cloned());
        stop_reason = response["stop_reason"].clone();
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
        } else if event["type"] == "message_delta" {
            stop_reason = event["delta"]["stop_reason"].clone();
        }
    }
    let blocks = blocks
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
        .collect();
    (blocks, stop_reason)
}

/// A Responses response, or the response of a stream's terminal event.
#[expect(clippy::unwrap_used, reason = "test helper")]
async fn responses(gateway: &Gateway, stream: bool) -> Value {
    let parameters = &tools()[0]["function"]["parameters"];
    let request = serde_json::from_value(json!({
        "model": MODEL, "input": "weather?", "store": false, "stream": stream,
        "tools": [{"type": "function", "name": "get_weather", "parameters": parameters}],
    }))
    .unwrap();
    let body = read(
        gateway
            .router
            .route_responses(None, &tenant(), request, MODEL),
    )
    .await;
    if stream {
        return events(&body).pop().unwrap()["response"].clone();
    }
    serde_json::from_str(&body).unwrap()
}

#[tokio::test]
async fn hf_reads_chat_through_the_templates_table_whole_and_streamed() {
    let gateway = serve(Setup::default()).await;
    for stream in [false, true] {
        let chat = chat(&gateway, json!({"stream": stream, "tools": tools()})).await;
        assert_eq!(chat, read_chat(), "stream={stream}");
    }
    // Without tools nothing is constrained, and the call the model writes anyway is content,
    // as the older path shows it when it runs no tool parser.
    let chat = chat(&gateway, json!({})).await;
    assert!(chat.calls.is_empty(), "{chat:?}");
    assert!(chat.content.starts_with("Checking."), "{chat:?}");
    assert!(chat.content.contains("<tool_call>"), "{chat:?}");
    assert_eq!(chat.finish, "stop");
    assert_eq!(gateway.last_constraint(), None);
}

#[tokio::test]
async fn hf_gives_a_forced_call_and_a_json_answer_the_tables_tags() {
    let gateway = serve(Setup::default()).await;
    chat(
        &gateway,
        json!({"tools": tools(), "tool_choice": "required"}),
    )
    .await;
    let (kind, tag) = gateway.last_constraint().expect("a constraint");
    assert_eq!(kind, "structural_tag");
    let tag: Value = serde_json::from_str(&tag).expect("the tag as JSON");
    assert!(tag["format"].is_object(), "{tag}");
    // The answer's tag stands in for the response_format: the engine sees the tag alone.
    chat(
        &gateway,
        json!({"response_format": {"type": "json_object"}}),
    )
    .await;
    let (kind, _) = gateway.last_constraint().expect("a constraint");
    assert_eq!(kind, "structural_tag");
    let request = gateway.requests.lock().expect("requests").last().cloned();
    assert!(request.is_some());
}

/// A worker on `constrained_decoding: always` gets the table's tag for the calls a request
/// allows under `auto`, in Chat and in Messages; a request with no tools or `tool_choice: none`
/// stays free, as every request does on a worker that says nothing.
#[tokio::test]
async fn always_constrains_the_calls_a_request_allows_under_auto() {
    let gateway = serve(Setup {
        constrained_decoding: Some(ConstrainedDecoding::Always),
        ..Setup::default()
    })
    .await;
    chat(&gateway, json!({"tools": tools()})).await;
    let (kind, tag) = gateway.last_constraint().expect("a constraint under auto");
    assert_eq!(kind, "structural_tag");
    let tag: Value = serde_json::from_str(&tag).expect("the tag as JSON");
    assert!(tag["format"].is_object(), "{tag}");
    chat(&gateway, json!({"tools": null})).await;
    assert_eq!(gateway.last_constraint(), None);
    chat(&gateway, json!({"tools": tools(), "tool_choice": "none"})).await;
    assert_eq!(gateway.last_constraint(), None);
    messages(&gateway, false).await;
    let (kind, _) = gateway
        .last_constraint()
        .expect("a constraint for Messages");
    assert_eq!(kind, "structural_tag");

    let requested = serve(Setup::default()).await;
    chat(&requested, json!({"tools": tools()})).await;
    assert_eq!(requested.last_constraint(), None);

    // On a registry table, a JSON response_format is the engine's own schema constraint; the
    // auto tag must not displace it (nor a prompt that continues the assistant's message).
    let registry = serve(Setup {
        constrained_decoding: Some(ConstrainedDecoding::Always),
        tool_parser: Some("qwen"),
        reasoning_parser: None,
        ..Setup::default()
    })
    .await;
    chat(&registry, json!({"tools": tools()})).await;
    let (kind, _) = registry.last_constraint().expect("a constraint under auto");
    assert_eq!(kind, "structural_tag");
    chat(
        &registry,
        json!({"tools": tools(), "response_format": {"type": "json_object"}}),
    )
    .await;
    let (kind, _) = registry.last_constraint().expect("the schema constraint");
    assert_eq!(kind, "json_schema");
}

#[tokio::test]
async fn hf_refuses_what_the_templates_table_cannot_constrain() {
    // No layout: the mock's own chat template renders no probe.
    let gateway = serve(Setup {
        chat_template: false,
        ..Setup::default()
    })
    .await;
    let (status, error) = refusal(gateway.router.route_chat(
        None,
        &tenant(),
        chat_request(json!({"tools": tools(), "tool_choice": "required"})),
        MODEL,
    ))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["code"], "invalid_parser_configuration");
    assert!(
        error["message"]
            .as_str()
            .unwrap_or("")
            .contains("no structural tag for a forced tool choice"),
        "{error}"
    );
    // The table still reads a free turn.
    let chat = chat(&gateway, json!({"tools": tools()})).await;
    assert_eq!(chat, read_chat());
    // A model whose tokenizer ships no template is refused outright.
    let gateway = serve(Setup {
        template: None,
        ..Setup::default()
    })
    .await;
    let (status, error) = refusal(gateway.router.route_chat(
        None,
        &tenant(),
        chat_request(json!({})),
        MODEL,
    ))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["code"], "invalid_parser_configuration");
    assert_eq!(
        error["message"],
        "the tokenizer config has no response_template"
    );
}

#[tokio::test]
async fn hf_reads_messages_and_responses_through_the_templates_table() {
    let gateway = serve(Setup::default()).await;
    for stream in [false, true] {
        let (blocks, stop_reason) = messages(&gateway, stream).await;
        assert_eq!(
            blocks,
            vec![
                json!(["thinking", "plan the call"]),
                json!(["text", "Checking."]),
                json!(["tool_use", {"city": "Paris", "days": 2}]),
            ],
            "stream={stream}"
        );
        assert_eq!(stop_reason, "tool_use", "stream={stream}");
    }
    for stream in [false, true] {
        let response = responses(&gateway, stream).await;
        let items = response["output"].as_array().expect("output items");
        let kinds: Vec<&str> = items
            .iter()
            .filter_map(|item| item["type"].as_str())
            .collect();
        assert!(kinds.contains(&"reasoning"), "stream={stream}: {kinds:?}");
        assert!(kinds.contains(&"message"), "stream={stream}: {kinds:?}");
        let call = items
            .iter()
            .find(|item| item["type"] == "function_call")
            .expect("a function call");
        assert_eq!(call["name"], "get_weather");
        let arguments: Value =
            serde_json::from_str(call["arguments"].as_str().unwrap_or("")).expect("arguments");
        assert_eq!(arguments, json!({"city": "Paris", "days": 2}));
        assert_eq!(call["status"], "completed");
        assert_eq!(response["status"], "completed");
    }
}

#[tokio::test]
async fn the_legacy_parser_refuses_hf_at_startup() {
    // Under --parser legacy the hf name is no parser: the gateway refuses it when it starts.
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
    config.parser = ParserChoice::Legacy;
    config.tool_call_parser = Some("hf".to_string());
    let why = smg::app_context::AppContext::from_config(config, 30, None, None)
        .await
        .expect_err("refused");
    assert!(why.contains("--parser symphony"), "{why}");
}
