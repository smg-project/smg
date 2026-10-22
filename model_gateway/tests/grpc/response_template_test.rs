//! A model read through the table its tokenizer's `response_template` compiles into, end to end
//! through the gRPC router and a scripted worker: the parser name `hf` under `--parser symphony`.
//! The tokenizer ships transformers' `serve` template for Qwen 3.5 and the Qwen3.5-9B chat
//! template, so the table has a layout and writes structural tags. Every `<|...|>` chunk is a
//! special token and `<|im_end|>` is the EOS token, which the stop decoder removes.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{body::to_bytes, http::StatusCode, response::Response};
use llm_tokenizer::{chat_template::ChatTemplateParams, traits::Tokenizer, TokenizerRegistry};
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

use crate::{common, scripted_tokenizer, scripted_worker};

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
    include_str!("../../../crates/response_template/tests/fixtures/compiled/templates.jsonl");
const CHAT: &str =
    include_str!("../../../crates/response_template/tests/fixtures/chat_templates/templates.jsonl");

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
    /// The worker's labels (where its engine starts a grammar, say).
    labels: Vec<(&'static str, &'static str)>,
    /// The text of the vocabulary's end-of-sequence token.
    eos: &'static str,
    /// The tokenizer reports the rendered prompt as ending inside the model's thought.
    think_in_prefill: bool,
    /// Selection takes the workers in turn (round robin) instead of at random.
    round_robin: bool,
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
            labels: Vec::new(),
            eos: EOS,
            think_in_prefill: false,
            round_robin: false,
        }
    }
}

/// One scripted worker of the model: the labels on its registry entry (where its engine starts a
/// grammar, say) and the finish reasons it scripts, one per generation
/// (`scripted_worker::ScriptedWorker::finish_reasons`).
struct Engine {
    labels: Vec<(&'static str, &'static str)>,
    finish_reasons: Vec<&'static str>,
}

struct Gateway {
    router: Box<dyn RouterTrait>,
    workers: Vec<JoinHandle<()>>,
    /// Every request each worker received, one log per worker in the order they were served.
    requests: Vec<Arc<Mutex<Vec<ts::GenerateRequest>>>>,
}

impl Drop for Gateway {
    fn drop(&mut self) {
        for worker in &self.workers {
            worker.abort();
        }
    }
}

impl Gateway {
    /// The constraint of the last request the first worker received, as `(kind, value)`.
    #[expect(clippy::unwrap_used, reason = "test helper")]
    fn last_constraint(&self) -> Option<(String, String)> {
        let requests = self.requests[0].lock().unwrap();
        constraint_of(requests.last()?)
    }
}

/// The prompt the Qwen3.5-9B chat template renders for the request [`chat_request`] builds from
/// `extra`, its messages with its `chat_template_kwargs`: where the template leaves the model. A
/// precondition of a case, rendered by the fixture's template as the gateway renders it; the
/// engine's request is not read back (it carries token ids).
#[expect(clippy::expect_used, reason = "test helper")]
fn rendered(extra: &Value) -> String {
    let request = chat_request(extra.clone());
    let messages: Vec<Value> =
        serde_json::from_value(serde_json::to_value(&request.messages).expect("the messages"))
            .expect("the messages as a list");
    ScriptedTokenizer::from_chunks(Vec::new())
        .with_chat_template(&qwen3_5_chat_template())
        .expect("the chat template")
        .apply_chat_template(
            &messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                template_kwargs: request.chat_template_kwargs.as_ref(),
                ..Default::default()
            },
        )
        .expect("a rendered prompt")
}

/// A request's constraint as `(kind, value)`.
fn constraint_of(request: &ts::GenerateRequest) -> Option<(String, String)> {
    let params = request.sampling_params.as_ref()?;
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

/// The gateway over one worker, with the setup's labels and finish reason.
async fn serve(mut setup: Setup) -> Gateway {
    let engine = Engine {
        labels: std::mem::take(&mut setup.labels),
        finish_reasons: vec![setup.finish],
    };
    serve_engines(setup, vec![engine]).await
}

/// The gateway over one scripted worker per engine; the setup's own labels and finish reason are
/// those of [`serve`]'s one worker and are not read here.
#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "test helper; the scripted workers are aborted when the gateway is dropped"
)]
async fn serve_engines(setup: Setup, engines: Vec<Engine>) -> Gateway {
    let chunks = tokens(setup.output);
    let mut workers = Vec::new();
    let mut logs = Vec::new();
    let mut ports = Vec::new();
    for engine in &engines {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        ports.push(listener.local_addr().unwrap().port());
        let requests = Arc::new(Mutex::new(Vec::new()));
        workers.push(tokio::spawn(
            scripted_worker::ScriptedWorker {
                output_tokens: u32::try_from(chunks.len()).unwrap(),
                finish_reasons: engine.finish_reasons.clone(),
                generation: Default::default(),
                requests: requests.clone(),
            }
            .serve(listener),
        ));
        logs.push(requests);
    }
    let config = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection();
    let config = if setup.round_robin {
        config.round_robin_policy()
    } else {
        config.random_policy()
    };
    let mut config = config
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
    let mut tokenizer = ScriptedTokenizer::from_chunks(chunks).with_special_tokens(setup.eos);
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
    if setup.think_in_prefill {
        tokenizer = tokenizer.with_think_in_prefill();
    }
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(tokenizer);
    let tokenizers = Arc::new(TokenizerRegistry::new());
    tokenizers
        .load(MODEL, MODEL, "test", || async move { Ok(tokenizer) })
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    for (engine, port) in engines.iter().zip(ports) {
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
        for (key, value) in &engine.labels {
            worker_entry = worker_entry.label(*key, *value);
        }
        context
            .worker_registry
            .register(Arc::new(worker_entry.build()))
            .unwrap();
    }
    let router = RouterFactory::create_router(&context).await.unwrap();
    Gateway {
        router,
        workers,
        requests: logs,
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
    let request = gateway.requests[0]
        .lock()
        .expect("requests")
        .last()
        .cloned();
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

    // On a registry table too. A JSON response_format is the request's own constraint, which
    // the auto tag never displaces: the answer's constraint is sent instead; on a worker that
    // names no grammar start that is the engine's own schema, whatever the prompt
    // (`a_registry_tables_json_answer_follows_the_thought_the_prompt_leaves_open`).
    let registry = serve(Setup {
        constrained_decoding: Some(ConstrainedDecoding::Always),
        tool_parser: Some("qwen"),
        reasoning_parser: None,
        ..Setup::default()
    })
    .await;
    chat(&registry, json!({"tools": tools()})).await;
    let (kind, tag) = registry.last_constraint().expect("a constraint under auto");
    assert_eq!(kind, "structural_tag");
    assert_eq!(block_of(&tag), "triggered_tags", "{tag}");
    chat(
        &registry,
        json!({"tools": tools(), "response_format": {"type": "json_object"}}),
    )
    .await;
    let (kind, schema) = registry.last_constraint().expect("the answer's constraint");
    assert_eq!(kind, "json_schema");
    let schema: Value = serde_json::from_str(&schema).expect("the schema as JSON");
    assert_eq!(schema, json!({"type": "object"}));
}

/// A registry table's JSON `response_format` under `always` follows the prompt as the table and
/// the engine read it (`regular::symphony::reasoning_open`), not the thinking switch alone. The
/// Qwen 3.5 chat template's generation prompt ends in `<think>\n`, so the model writes its thought
/// first and the answer after its close. The engine's own schema constraint is sent wherever the
/// tag would be the schema alone: an engine that does not say where it starts the grammar (free
/// text before the schema would constrain nothing: the model may write a fence and prose around
/// its JSON, with the end of the turn masked until a whole JSON follows), one that starts it
/// after its reasoning parser has seen the close, and a request with thinking off, whose prompt
/// closes the thought (`<think>\n\n</think>\n\n`); an engine that runs the grammar from the first
/// token gets the closed thought, then the schema, so the JSON is not forced inside the thought,
/// which the table reads as reasoning.
#[tokio::test]
async fn a_registry_tables_json_answer_follows_the_thought_the_prompt_leaves_open() {
    let answer = json!({"tools": tools(), "response_format": {"type": "json_object"}});
    let registry = |labels: Vec<(&'static str, &'static str)>| Setup {
        constrained_decoding: Some(ConstrainedDecoding::Always),
        tool_parser: Some("qwen"),
        reasoning_parser: None,
        labels,
        ..Setup::default()
    };
    // The template's default leaves the thought open: for an engine that does not say where it
    // starts the grammar, the schema alone, the engine's own constraint.
    let prompt = rendered(&answer);
    assert!(prompt.ends_with(THINKING), "{prompt:?}");
    let gateway = serve(registry(Vec::new())).await;
    chat(&gateway, answer.clone()).await;
    let sent = gateway.last_constraint();
    assert_eq!(shape_of(sent.clone()), "not a tag");
    let (kind, schema) = sent.expect("the schema constraint");
    assert_eq!(kind, "json_schema");
    let schema: Value = serde_json::from_str(&schema).expect("the schema as JSON");
    assert_eq!(schema, json!({"type": "object"}));
    // Thinking off closes the thought in the prompt: the tag would be the schema alone, and the
    // engine's own constraint is sent, as before.
    let off = HashMap::from([("enable_thinking".to_string(), json!(false))]);
    let mut request = answer.clone();
    request["chat_template_kwargs"] = json!(off);
    let prompt = rendered(&request);
    assert!(prompt.ends_with("</think>\n\n"), "{prompt:?}");
    chat(&gateway, request).await;
    let (kind, _) = gateway.last_constraint().expect("the schema constraint");
    assert_eq!(kind, "json_schema");
    // An engine that starts the grammar after its reasoning parser has consumed the thought: the
    // schema alone, the engine's own constraint; one that runs it from the first token: the
    // thought closed by its own marker, then the schema.
    let after = ("constrained_decoding_start", "after_reasoning");
    let gateway = serve(registry(vec![after])).await;
    chat(&gateway, answer.clone()).await;
    let (kind, _) = gateway.last_constraint().expect("the schema constraint");
    assert_eq!(kind, "json_schema");
    let first = ("constrained_decoding_start", "first_token");
    let gateway = serve(registry(vec![first])).await;
    chat(&gateway, answer).await;
    let sent = gateway.last_constraint();
    assert_eq!(shape_of(sent.clone()), "the closed thought then the block");
    let (_, tag) = sent.expect("the answer's tag");
    assert_eq!(block_of(&tag), "json_schema", "{tag}");
}

/// The kind of the block a tag's format writes after what leads it (the free text or the closed
/// thought of a prompt that leaves the thought open, and the text the template writes after the
/// thought's close where the table carries one): the calls' `triggered_tags`, or an answer's
/// `json_schema`, the sequence's last element.
#[expect(clippy::expect_used, reason = "test helper")]
fn block_of(tag: &str) -> String {
    let tag: Value = serde_json::from_str(tag).expect("a tag");
    tag["format"]["elements"]
        .as_array()
        .and_then(|elements| elements.last())
        .and_then(|block| block["type"].as_str())
        .unwrap_or("")
        .to_string()
}

/// Which engine a tag under `auto`, or an answer's, is shaped for, the prompt leaving the thought
/// open: the free text before the block for an engine that does not say where it starts the
/// grammar, the block alone for one that starts it after its reasoning parser's gate, the closed
/// thought then the block for one that starts it at the first token.
#[expect(clippy::expect_used, reason = "test helper")]
fn shape_of(constraint: Option<(String, String)>) -> &'static str {
    let Some((kind, tag)) = constraint else {
        return "no constraint";
    };
    if kind != "structural_tag" {
        return "not a tag";
    }
    let tag: Value = serde_json::from_str(&tag).expect("a tag");
    let first = &tag["format"]["elements"][0];
    if first["type"] == "any_text" {
        "the free text then the block"
    } else if first["end"] == "</think>" {
        "the closed thought then the block"
    } else {
        "the block alone"
    }
}

/// Two workers of the model under `always`, whose engines start the grammar at different places
/// (one after its reasoning parser's gate, one that does not say): the `auto` tag reaches each in
/// the shape its own start takes, on the first attempt and on a retry alike. Each worker refuses
/// its first `Generate`, so the request crosses to the other worker and back before it is served:
/// three attempts, every one carrying the tag shaped for the worker it reached. On a registry
/// table whose block takes the free-text prefix (Qwen 3.5's tagged calls), the prompt leaving
/// the thought open, so the two shapes differ.
#[tokio::test]
async fn a_retry_takes_the_auto_tag_in_the_shape_its_own_workers_start_needs() {
    let gateway = serve_engines(
        Setup {
            chat_template: false,
            prompt_tail: THINKING,
            think_in_prefill: true,
            tool_parser: Some("qwen_coder"),
            reasoning_parser: None,
            constrained_decoding: Some(ConstrainedDecoding::Always),
            ..Setup::default()
        },
        vec![
            Engine {
                labels: vec![("constrained_decoding_start", "after_reasoning")],
                finish_reasons: vec!["unavailable", "stop"],
            },
            Engine {
                labels: vec![],
                finish_reasons: vec!["unavailable", "stop"],
            },
        ],
    )
    .await;
    let served = chat(&gateway, json!({"tools": tools()})).await;
    assert_eq!(served.calls, vec![weather()]);
    assert_eq!(served.finish, "tool_calls");
    assert_eq!(served.reasoning.trim(), "plan the call");
    // What each worker received: the shape of every tag, and the tags themselves for the message.
    let received = |log: &Arc<Mutex<Vec<ts::GenerateRequest>>>| {
        let log = log.lock().expect("the worker's log");
        let shapes: Vec<&'static str> = log
            .iter()
            .map(|request| shape_of(constraint_of(request)))
            .collect();
        let tags: Vec<Option<(String, String)>> = log.iter().map(constraint_of).collect();
        (shapes, tags)
    };
    let (gated, gated_tags) = received(&gateway.requests[0]);
    let (unsaid, unsaid_tags) = received(&gateway.requests[1]);
    assert_eq!(gated.len() + unsaid.len(), 3, "{gated:?} {unsaid:?}");
    assert!(
        !gated.is_empty() && gated.iter().all(|shape| *shape == "the block alone"),
        "after the gate: {gated:?} {gated_tags:?}"
    );
    assert!(
        !unsaid.is_empty()
            && unsaid
                .iter()
                .all(|shape| *shape == "the free text then the block"),
        "an engine that does not say: {unsaid:?} {unsaid_tags:?}"
    );
}

/// What one worker received: the shape of every request's constraint, and the constraints
/// themselves for the message.
type Received = (Vec<&'static str>, Vec<Option<(String, String)>>);

#[expect(clippy::expect_used, reason = "test helper")]
fn received(log: &Arc<Mutex<Vec<ts::GenerateRequest>>>) -> Received {
    let log = log.lock().expect("the worker's log");
    let shapes = log
        .iter()
        .map(|request| shape_of(constraint_of(request)))
        .collect();
    let constraints = log.iter().map(constraint_of).collect();
    (shapes, constraints)
}

/// Two workers of the model whose engines start the grammar at different places, one at the first
/// token and one that does not say, and a JSON `response_format` on a prompt that leaves the
/// thought open: selection takes the two in turn, and every request's constraint takes the shape
/// of the worker it reaches, the closed thought then the schema on the first, the schema alone
/// (the engine's own constraint) on the other. No request carries the free-text prefix, which
/// fits neither: the first-token engine would let the model write around its JSON, the other
/// cannot be told where the schema begins. On a registry table (Qwen 3.5's).
#[tokio::test]
async fn a_mixed_pools_json_answer_takes_the_shape_of_the_worker_each_request_reaches() {
    const REQUESTS: usize = 6;
    let answer = json!({"tools": tools(), "response_format": {"type": "json_object"}});
    assert!(rendered(&answer).ends_with(THINKING));
    let gateway = serve_engines(
        Setup {
            tool_parser: Some("qwen"),
            reasoning_parser: None,
            round_robin: true,
            ..Setup::default()
        },
        vec![
            Engine {
                labels: vec![("constrained_decoding_start", "first_token")],
                finish_reasons: vec!["stop"],
            },
            Engine {
                labels: vec![],
                finish_reasons: vec!["stop"],
            },
        ],
    )
    .await;
    for _ in 0..REQUESTS {
        chat(&gateway, answer.clone()).await;
    }
    let (first, first_tags) = received(&gateway.requests[0]);
    let (unsaid, unsaid_tags) = received(&gateway.requests[1]);
    assert_eq!(first.len() + unsaid.len(), REQUESTS, "{first:?} {unsaid:?}");
    assert!(
        !first.is_empty() && !unsaid.is_empty(),
        "{first:?} {unsaid:?}"
    );
    let wrong = first
        .iter()
        .filter(|shape| **shape != "the closed thought then the block")
        .count()
        + unsaid.iter().filter(|shape| **shape != "not a tag").count();
    assert_eq!(
        wrong, 0,
        "{wrong} of {REQUESTS} requests in a shape not the worker's own; first token: {first:?} \
         {first_tags:?}; does not say: {unsaid:?} {unsaid_tags:?}"
    );
    // The worker that does not say gets the request's own schema, as the engine takes it.
    for constraint in &unsaid_tags {
        let (kind, schema) = constraint.clone().expect("the schema constraint");
        assert_eq!(kind, "json_schema");
        let schema: Value = serde_json::from_str(&schema).expect("the schema as JSON");
        assert_eq!(schema, json!({"type": "object"}));
    }
    for constraint in &first_tags {
        let (_, tag) = constraint.clone().expect("the answer's tag");
        assert_eq!(block_of(&tag), "json_schema", "{tag}");
    }
}

/// The same two kinds of worker, each refusing its first `Generate`, so a JSON answer's request
/// crosses to the other worker and back before it is served: three attempts, every one carrying
/// the answer's constraint in the shape of the worker it reached, the schema alone behind the
/// reasoning parser's gate and the closed thought then the schema from the first token, whichever
/// kind the first attempt took. A retry re-derives the constraint for its worker, the one shape
/// standing in for the other.
#[tokio::test]
async fn a_retry_takes_the_json_answers_tag_in_the_shape_its_own_workers_start_needs() {
    let gateway = serve_engines(
        Setup {
            tool_parser: Some("qwen"),
            reasoning_parser: None,
            ..Setup::default()
        },
        vec![
            Engine {
                labels: vec![("constrained_decoding_start", "after_reasoning")],
                finish_reasons: vec!["unavailable", "stop"],
            },
            Engine {
                labels: vec![("constrained_decoding_start", "first_token")],
                finish_reasons: vec!["unavailable", "stop"],
            },
        ],
    )
    .await;
    let served = chat(
        &gateway,
        json!({"tools": tools(), "response_format": {"type": "json_object"}}),
    )
    .await;
    let (gated, gated_tags) = received(&gateway.requests[0]);
    let (first, first_tags) = received(&gateway.requests[1]);
    let wrong = gated.iter().filter(|shape| **shape != "not a tag").count()
        + first
            .iter()
            .filter(|shape| **shape != "the closed thought then the block")
            .count();
    assert!(!served.finish.is_empty(), "served after the retries");
    assert_eq!(gated.len() + first.len(), 3, "{gated:?} {first:?}");
    assert_eq!(
        wrong,
        0,
        "{wrong} of {} attempts in a shape not the worker's own; after the gate: {gated:?} \
         {gated_tags:?}; from the first token: {first:?} {first_tags:?}",
        gated.len() + first.len()
    );
    assert!(
        !gated.is_empty() && gated.iter().all(|shape| *shape == "not a tag"),
        "after the gate: {gated:?} {gated_tags:?}"
    );
    for constraint in &gated_tags {
        let (kind, schema) = constraint.clone().expect("the schema constraint");
        assert_eq!(kind, "json_schema");
        let schema: Value = serde_json::from_str(&schema).expect("the schema as JSON");
        assert_eq!(schema, json!({"type": "object"}));
    }
    assert!(
        !first.is_empty()
            && first
                .iter()
                .all(|shape| *shape == "the closed thought then the block"),
        "from the first token: {first:?} {first_tags:?}"
    );
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

/// A response template whose calls spell a whole turn (Beam's shape): a call's header names the
/// tool, its parameters are tags around raw text, and a row out of a call goes on to the next or
/// stops; `<|eos|>` is its end-of-sequence token.
fn turn_template() -> Value {
    json!({
      "start_anchor_pattern": "<\\|head\\|>assistant",
      "fields": {
        "thinking": {"open": "<|via|>notes<|body|>", "close": "<|stop|><|head|>assistant",
                     "content": "text"},
        "content": {"open": "<|via|>reply<|body|>", "close": ["<|done|>", "<|stop|>", "<|eos|>"],
                    "content": "text"},
        "tool_calls": {
          "open_pattern": " to=(?P<name>[^\\s<]+)<\\|via\\|>notes <\\|as\\|>xml<\\|body\\|>",
          "close": ["<|stop|><|head|>assistant", "<|eos|>"], "repeats": true,
          "content": "xml-inline",
          "content_args": {
            "tag_pattern":
              "<arg name=\"(?P<key>[^\"]+)\">\\n(?P<value>.*?)\\n(?:</arg>|(?=<arg name=\"))",
            "value_parser": {"name": "text", "args": {"strip": false}}
          },
          "transform": {"type": "function",
                        "function": {"name": "{name}", "arguments": "{content}"}}
        }
      }
    })
}

/// What a worker serving [`turn_template`]'s model under `policy`, with `labels`, received for a
/// chat request: the constraint's kind and the tag's format, or the refusal's status and message.
#[expect(clippy::expect_used, reason = "test helper")]
async fn turn_sent(
    policy: Option<ConstrainedDecoding>,
    labels: Vec<(&'static str, &'static str)>,
    extra: Value,
) -> Result<Option<(String, Value)>, (StatusCode, String)> {
    let gateway = serve(Setup {
        template: Some(turn_template()),
        chat_template: false,
        prompt_tail: "<|head|>assistant",
        output: "<|via|>notes<|body|>plan<|stop|><|head|>assistant<|via|>reply<|body|>Hello<|eos|>",
        eos: "<|eos|>",
        constrained_decoding: policy,
        labels,
        ..Setup::default()
    })
    .await;
    let tenant = tenant();
    let response = timeout(
        Duration::from_secs(30),
        gateway
            .router
            .route_chat(None, &tenant, chat_request(extra), MODEL),
    )
    .await
    .expect("request should finish");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.expect("body");
    if status != StatusCode::OK {
        let body: Value = serde_json::from_slice(&bytes).expect("an error body");
        let message = body["error"]["message"].as_str().unwrap_or("").to_string();
        return Err((status, message));
    }
    Ok(gateway.last_constraint().map(|(kind, value)| {
        let format = match kind.as_str() {
            "structural_tag" => {
                serde_json::from_str::<Value>(&value).expect("a tag")["format"].clone()
            }
            _ => Value::String(value),
        };
        (kind, format)
    }))
}

/// Whether a tag's format is the whole turn's grammar: it opens with the thought.
fn is_whole_turn(sent: Option<&(String, Value)>) -> bool {
    sent.is_some_and(|(kind, format)| {
        kind == "structural_tag" && format["elements"][0]["begin"] == "<|via|>notes<|body|>"
    })
}

#[tokio::test]
async fn a_worker_on_always_from_the_first_token_takes_the_whole_turns_grammar() {
    let first = ("constrained_decoding_start", "first_token");
    let always = Some(ConstrainedDecoding::Always);
    for extra in [
        json!({"tools": tools(), "tool_choice": "required"}),
        json!({"tools": tools()}),
        json!({"tools": null}),
        json!({"tools": tools(), "response_format": {"type": "json_object"}}),
    ] {
        let sent = turn_sent(always, vec![first], extra.clone()).await;
        assert!(
            sent.as_ref().is_ok_and(|sent| is_whole_turn(sent.as_ref())),
            "{extra}: {sent:?}"
        );
    }
    // An ebnf in xgrammar's syntax is the answer's; the engine's request carries the tag alone.
    let sent = turn_sent(always, vec![first], json!({"ebnf": "root ::= \"y\""}))
        .await
        .expect("served");
    assert!(is_whole_turn(sent.as_ref()), "{sent:?}");
    let (_, format) = sent.expect("a tag");
    assert_eq!(format["elements"][1]["elements"][1]["type"], "grammar");
}

#[tokio::test]
async fn the_selected_workers_policy_decides_and_what_no_grammar_holds_is_refused() {
    let first = ("constrained_decoding_start", "first_token");
    let after = ("constrained_decoding_start", "after_reasoning");
    let required = json!({"tools": tools(), "tool_choice": "required"});
    let no_tag = "no structural tag for a forced tool choice";
    let mut wrong = Vec::new();
    for (policy, labels, extra, why) in [
        // `requested`, set or not, is the behaviour before the policy: the table has no tag
        // for a forced call.
        (None, vec![first], required.clone(), no_tag),
        (
            Some(ConstrainedDecoding::Requested),
            vec![first],
            required.clone(),
            no_tag,
        ),
        (
            Some(ConstrainedDecoding::Requested),
            vec![after],
            required.clone(),
            no_tag,
        ),
        // `always` on an engine that starts the grammar after the reasoning, or does not say,
        // takes the tags: a forced call is refused, never sent free.
        (
            Some(ConstrainedDecoding::Always),
            vec![after],
            required.clone(),
            no_tag,
        ),
        (
            Some(ConstrainedDecoding::Always),
            vec![],
            required.clone(),
            no_tag,
        ),
        // What the whole turn's grammar cannot hold.
        (
            Some(ConstrainedDecoding::Always),
            vec![first],
            json!({"ebnf": "start: \"y\""}),
            "Lark",
        ),
        (
            Some(ConstrainedDecoding::Always),
            vec![first],
            json!({"tools": tools(), "regex": "y+", "parallel_tool_calls": false}),
            "parallel_tool_calls: false",
        ),
        (
            None,
            vec![first],
            json!({"ebnf": "start: \"y\""}),
            "would apply outside it",
        ),
    ] {
        match turn_sent(policy, labels.clone(), extra.clone()).await {
            Err((StatusCode::BAD_REQUEST, message)) if message.contains(why) => {}
            other => wrong.push(format!("{policy:?} {labels:?} {extra}: {other:?}")),
        }
    }
    assert!(wrong.is_empty(), "not refused as expected: {wrong:#?}");
}
