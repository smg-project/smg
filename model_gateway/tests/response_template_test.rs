//! Deterministic response-template gateway integration tests.
//!
//! Uses a synthetic tokenizer and fixed gRPC replies; no model download or GPU is needed.

#![recursion_limit = "512"]
#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration-test helpers fail immediately when a fixture or assertion is invalid"
)]

mod common;

use std::{
    collections::{BTreeMap, VecDeque},
    env,
    fs::{self, OpenOptions},
    io::{Cursor, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use async_trait::async_trait;
use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Request, StatusCode},
};
use llm_tokenizer::{
    CacheConfig, CachedTokenizer, Decoder, Encoder, Encoding, HuggingFaceTokenizer, LoadError,
    SequenceDecoderOutput, SpecialTokens, StopSequenceConfig, StopSequenceDecoder,
    TokenizerRegistry, TokenizerTrait,
};
use mock_worker::{
    config::Config as MockConfig,
    engine::EngineParams,
    grpc::{install_test_controls, remove_test_controls, GrpcTestControls, GrpcTestTokenizerReply},
};
use openai_protocol::worker::HealthCheckConfig;
use response_template_parser::{
    ClosePattern, ContentArgs, FieldTemplate, ParseOutput, ParserConfig, ResponseTemplate,
    ResponseTemplateError, ResponseTemplateParser, Transform, ValueParser, ValueParserArgs,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use smg::{
    config::RouterConfig,
    routers::{RouterFactory, RouterTrait},
    worker::{BasicWorkerBuilder, ConnectionMode, ModelCard, RuntimeType, Worker, WorkerType},
    workflow::tokenizer_registration::{
        create_tokenizer_registration_workflow, create_tokenizer_workflow_data, LoadTokenizerStep,
        TokenizerConfigRequest,
    },
};
use tempfile::TempDir;
use tower::ServiceExt;
use wfaas::{
    EventSubscriber, StepExecutor, WorkflowContext, WorkflowEngine, WorkflowError, WorkflowEvent,
    WorkflowId, WorkflowInstanceId,
};
use zip::{write::SimpleFileOptions, ZipWriter};

const MODEL: &str = "response-template-test";
const PREFIX: &str = "<|start|>assistant";
const CLOSE_ID: u32 = 10;
const USER_STOP_ID: u32 = 11;
const OVERFLOW_ID: u32 = 12;
const USER_STOP_LITERAL: &str = "<|external-stop|>";
const BYTE_BASE: u32 = 1_000;
const MIN_TOKENIZER_JSON: &str = r#"{
  "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
  "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
  "post_processor":null,"decoder":null,
  "model":{"type":"BPE","vocab":{"hello":0},"merges":[]}
}"#;

fn response_template() -> ResponseTemplate {
    let fields = BTreeMap::from([
        (
            "thinking".to_owned(),
            FieldTemplate {
                open_pattern: r"<\|channel\|>analysis<\|message\|>".to_owned(),
                close: ClosePattern::One("<|end|>".to_owned()),
                content: "text".to_owned(),
                content_args: None,
                repeats: false,
                transform: None,
            },
        ),
        (
            "content".to_owned(),
            FieldTemplate {
                open_pattern: r"<\|channel\|>final<\|message\|>".to_owned(),
                close: ClosePattern::Many(vec!["<|return|>".to_owned(), "<|end|>".to_owned()]),
                content: "text".to_owned(),
                content_args: None,
                repeats: false,
                transform: None,
            },
        ),
        (
            "tool_calls".to_owned(),
            FieldTemplate {
                open_pattern: concat!(
                    r"(?:<\|start\|>assistant )?to=(?P<name>.+?)",
                    r"<\|channel\|>analysis(?: <\|constrain\|>xml)?<\|message\|>"
                )
                .to_owned(),
                close: ClosePattern::Many(vec!["<|call|>".to_owned(), "<|end|>".to_owned()]),
                content: "xml-inline".to_owned(),
                content_args: Some(ContentArgs {
                    tag_pattern: concat!(
                        r#"<rfl:parameter name=\"(?P<key>[^\"]+)\">"#,
                        r"(?P<value>.*?)</rfl:parameter>"
                    )
                    .to_owned(),
                    value_parser: ValueParser {
                        name: "text".to_owned(),
                        args: ValueParserArgs { strip: true },
                    },
                }),
                repeats: true,
                transform: Some(Transform(json!({
                    "type": "function",
                    "function": {"name": "{name}", "arguments": "{content}"}
                }))),
            },
        ),
    ]);
    ResponseTemplate {
        defaults: BTreeMap::from([
            ("thinking".to_owned(), json!("")),
            ("content".to_owned(), json!("")),
            ("tool_calls".to_owned(), json!([])),
        ]),
        start_anchor_pattern: r"<\|start\|>assistant".to_owned(),
        fields,
    }
}

fn wire_output() -> String {
    concat!(
        "<|channel|>analysis<|message|>careful reasoning<|end|>",
        "<|start|>assistant to=weather.lookup<|channel|>analysis <|constrain|>xml<|message|>",
        "<rfl:parameter name=\"city\">Paris</rfl:parameter>",
        "<rfl:parameter name=\"days\">2</rfl:parameter><|call|>",
        "<|start|>assistant to=calendar.lookup<|channel|>analysis <|constrain|>xml<|message|>",
        "<rfl:parameter name=\"day\">tomorrow</rfl:parameter><|end|>",
        "<|channel|>final<|message|>The result is ready.<|return|>",
        "THIS_TEXT_MUST_NOT_BE_EMITTED"
    )
    .to_owned()
}

#[derive(Debug)]
struct OracleTokenizer {
    special: SpecialTokens,
    template: Option<Value>,
}

impl OracleTokenizer {
    fn new(template: Option<Value>) -> Self {
        Self {
            special: SpecialTokens {
                eos_token: Some("<|return|>".to_owned()),
                ..SpecialTokens::default()
            },
            template,
        }
    }

    fn token_ids(text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            if let Some(rest) = remaining.strip_prefix("<|return|>") {
                ids.push(CLOSE_ID);
                remaining = rest;
            } else if let Some(rest) = remaining.strip_prefix(USER_STOP_LITERAL) {
                ids.push(USER_STOP_ID);
                remaining = rest;
            } else {
                let byte = remaining.as_bytes()[0];
                ids.push(BYTE_BASE + u32::from(byte));
                remaining = &remaining[1..];
            }
        }
        ids
    }
}

impl Encoder for OracleTokenizer {
    fn encode(&self, input: &str, _add_special_tokens: bool) -> anyhow::Result<Encoding> {
        Ok(Encoding::Plain(Self::token_ids(input)))
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        inputs
            .iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect()
    }
}

impl Decoder for OracleTokenizer {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        let mut result = String::new();
        for id in token_ids {
            if *id == OVERFLOW_ID {
                result.push_str(&"x".repeat(ParserConfig::default().max_pending_bytes + 1));
            } else if *id == CLOSE_ID {
                if !skip_special_tokens {
                    result.push_str("<|return|>");
                }
            } else if *id == USER_STOP_ID {
                if !skip_special_tokens {
                    result.push_str(USER_STOP_LITERAL);
                }
            } else if let Some(byte) = id
                .checked_sub(BYTE_BASE)
                .and_then(|id| u8::try_from(id).ok())
            {
                result.push(char::from(byte));
            }
        }
        Ok(result)
    }
}

impl TokenizerTrait for OracleTokenizer {
    fn vocab_size(&self) -> usize {
        259
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match token {
            "<|return|>" => Some(CLOSE_ID),
            USER_STOP_LITERAL => Some(USER_STOP_ID),
            _ => None,
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        if id == CLOSE_ID {
            Some("<|return|>".to_owned())
        } else if id == USER_STOP_ID {
            Some(USER_STOP_LITERAL.to_owned())
        } else if id == OVERFLOW_ID {
            Some("x".repeat(ParserConfig::default().max_pending_bytes + 1))
        } else {
            id.checked_sub(BYTE_BASE)
                .and_then(|id| u8::try_from(id).ok())
                .map(|byte| char::from(byte).to_string())
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn response_template(&self) -> Option<&Value> {
        self.template.as_ref()
    }

    fn apply_chat_template(
        &self,
        _messages: &[Value],
        _params: llm_tokenizer::chat_template::ChatTemplateParams,
    ) -> anyhow::Result<String> {
        Ok(PREFIX.to_owned())
    }

    fn eos_token_ids(&self) -> &[u32] {
        &[CLOSE_ID]
    }
}

fn write_tokenizer_dir(template: Option<Value>) -> TempDir {
    let dir = TempDir::new().expect("temp tokenizer dir");
    fs::write(dir.path().join("tokenizer.json"), MIN_TOKENIZER_JSON).expect("write tokenizer");
    let config = template
        .map(|template| json!({"response_template": template}))
        .unwrap_or_else(|| json!({"chat_template": "{{ messages }}"}));
    fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::to_vec(&config).expect("serialize tokenizer config"),
    )
    .expect("write tokenizer config");
    dir
}

fn invalid_template() -> Value {
    let mut template = serde_json::to_value(response_template()).expect("template value");
    template["fields"]["thinking"]["content"] = json!("unsupported-text-parser");
    template
}

fn tokenizer_bundle(template: Option<Value>) -> (Vec<u8>, String) {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    writer
        .start_file("tokenizer.json", SimpleFileOptions::default())
        .expect("zip tokenizer member");
    writer
        .write_all(MIN_TOKENIZER_JSON.as_bytes())
        .expect("zip tokenizer payload");
    writer
        .start_file("tokenizer_config.json", SimpleFileOptions::default())
        .expect("zip config member");
    let config = template
        .map(|template| json!({"response_template": template}))
        .unwrap_or_else(|| json!({"chat_template": "{{ messages }}"}));
    writer
        .write_all(&serde_json::to_vec(&config).expect("serialize config"))
        .expect("zip config payload");
    let bytes = writer.finish().expect("finish zip").into_inner();
    let sha256 = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    (bytes, sha256)
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("ephemeral address")
        .port()
}

fn mock_config(port: u16) -> Arc<MockConfig> {
    Arc::new(MockConfig {
        host: "127.0.0.1".to_owned(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: MODEL.to_owned(),
        tokenizer_path: MODEL.to_owned(),
        gen_delay: Duration::ZERO,
        output_tokens: 1,
        realistic: false,
        engine: EngineParams::default(),
    })
}

fn router_config() -> RouterConfig {
    let mut config = RouterConfig::builder()
        .grpc_connection()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .request_timeout_secs(20)
        .worker_startup_timeout_secs(2)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(16)
        .queue_timeout_secs(20)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    config
}

fn register_grpc_worker(app: &smg::app_context::AppContext, port: u16) {
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..HealthCheckConfig::default()
            })
            .build(),
    );
    app.worker_registry
        .register(worker)
        .expect("register deterministic gRPC worker");
}

#[derive(Default)]
struct RecordedEvents(Mutex<Vec<WorkflowEvent>>);

#[async_trait]
impl EventSubscriber for RecordedEvents {
    async fn on_event(&self, event: &WorkflowEvent) {
        self.0
            .lock()
            .expect("event mutex poisoned")
            .push(event.clone());
    }
}

async fn registration(
    app: Arc<smg::app_context::AppContext>,
    source: &Path,
    name: &str,
) -> (Result<String, String>, Vec<WorkflowEvent>) {
    let engine = WorkflowEngine::new();
    engine
        .register_workflow(create_tokenizer_registration_workflow())
        .expect("register tokenizer workflow");
    let events = Arc::new(RecordedEvents::default());
    engine.event_bus().subscribe(events.clone()).await;
    let data = create_tokenizer_workflow_data(
        TokenizerConfigRequest {
            id: format!("{name}-id"),
            name: name.to_owned(),
            source: source.to_string_lossy().into_owned(),
            chat_template_path: None,
            cache_config: None,
            fail_on_duplicate: true,
        },
        app,
    );
    let id = engine
        .start_workflow(WorkflowId::new("tokenizer_registration"), data)
        .await
        .expect("start tokenizer registration workflow");
    let result = engine
        .wait_for_completion(id, "tokenizer registration", Duration::from_secs(15))
        .await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    let observed = events.0.lock().expect("event mutex poisoned").clone();
    (result, observed)
}

fn event_counts(events: &[WorkflowEvent]) -> (usize, usize, usize, usize, Vec<bool>, String) {
    let starts = events
        .iter()
        .filter(|event| matches!(event, WorkflowEvent::WorkflowStarted { .. }))
        .count();
    let attempts = events
        .iter()
        .filter(|event| matches!(event, WorkflowEvent::StepStarted { .. }))
        .count();
    let failures = events
        .iter()
        .filter(|event| matches!(event, WorkflowEvent::StepFailed { .. }))
        .count();
    let retries = events
        .iter()
        .filter(|event| matches!(event, WorkflowEvent::StepRetrying { .. }))
        .count();
    let will_retry = events
        .iter()
        .filter_map(|event| match event {
            WorkflowEvent::StepFailed { will_retry, .. } => Some(*will_retry),
            _ => None,
        })
        .collect();
    let error = events
        .iter()
        .find_map(|event| match event {
            WorkflowEvent::StepFailed { error, .. } => Some(error.clone()),
            _ => None,
        })
        .unwrap_or_default();
    (starts, attempts, failures, retries, will_retry, error)
}

fn parse_streamed(wire: &[u8], chunk_lengths: &[usize]) -> ParseOutput {
    let parser = ResponseTemplateParser::new(MODEL, response_template(), ParserConfig::default())
        .expect("valid response template");
    let mut stream = parser.stream(PREFIX);
    let mut output = ParseOutput::default();
    let mut cursor = 0;
    for length in chunk_lengths {
        let end = cursor + length;
        output.merge(stream.feed(&wire[cursor..end]).expect("stream feed"));
        cursor = end;
    }
    output.merge(stream.finish().expect("stream finish"));
    output
}

fn normalized(output: &ParseOutput) -> Value {
    json!({
        "reasoning_content": output.thinking,
        "content": output.content,
        "tool_calls": output.tool_calls.iter().map(|call| call.transformed.clone()).collect::<Vec<_>>()
    })
}

fn semantic_chat_payload(stream: bool) -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role":"user","content":"hello"}],
        "stream": stream,
        "tools": [
            {"type":"function","function":{"name":"weather.lookup","description":"weather","parameters":{"type":"object"}}},
            {"type":"function","function":{"name":"calendar.lookup","description":"calendar","parameters":{"type":"object"}}}
        ],
        "tool_choice": "required"
    })
}

fn visible_stop_chat_payload(stream: bool) -> Value {
    let mut payload = semantic_chat_payload(stream);
    payload["stop_token_ids"] = json!([USER_STOP_ID]);
    payload["no_stop_trim"] = json!(true);
    payload
}

async fn request_json(app: &axum::Router, uri: &str, payload: Value) -> Value {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_vec(&payload).expect("request json"),
        ))
        .expect("build request");
    let response = app.clone().oneshot(request).await.expect("route request");
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response body");
    serde_json::from_slice(&body).expect("response json")
}

async fn request_sse(app: &axum::Router, uri: &str, payload: Value) -> Vec<Value> {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_vec(&payload).expect("request json"),
        ))
        .expect("build request");
    let response = app.clone().oneshot(request).await.expect("route request");
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read response stream");
    String::from_utf8(body.to_vec())
        .expect("utf8 SSE")
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("SSE JSON event"))
        .collect()
}

fn contains_delimiter(value: &Value) -> bool {
    let text = value.to_string();
    [
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|call|>",
        "<|return|>",
        "<rfl:parameter",
        "THIS_TEXT_MUST_NOT_BE_EMITTED",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

fn marker_count(value: &Value, marker: &str) -> usize {
    value.to_string().matches(marker).count()
}

fn normalize_chat_message(message: &Value) -> Value {
    let calls = message["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|call| {
            let arguments = call["function"]["arguments"]
                .as_str()
                .and_then(|value| serde_json::from_str::<Value>(value).ok())
                .unwrap_or_else(|| json!({}));
            json!({
                "type": "function",
                "function": {"name": call["function"]["name"], "arguments": arguments}
            })
        })
        .collect::<Vec<_>>();
    json!({
        "reasoning_content": message["reasoning_content"].as_str().unwrap_or_default(),
        "content": message["content"].as_str().unwrap_or_default(),
        "tool_calls": calls
    })
}

fn normalize_chat_events(events: &[Value]) -> Value {
    let reasoning = events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect::<String>();
    let content = events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    let mut calls = BTreeMap::<u64, Value>::new();
    for call in events.iter().flat_map(|event| {
        event["choices"][0]["delta"]["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
    }) {
        let index = call["index"].as_u64().expect("streamed tool index");
        let arguments = call["function"]["arguments"]
            .as_str()
            .and_then(|value| serde_json::from_str::<Value>(value).ok())
            .unwrap_or_else(|| json!({}));
        calls.insert(
            index,
            json!({
                "type": "function",
                "function": {"name": call["function"]["name"], "arguments": arguments}
            }),
        );
    }
    json!({
        "reasoning_content": reasoning,
        "content": content,
        "tool_calls": calls.into_values().collect::<Vec<_>>()
    })
}

fn responses_fields_mapped(response: &Value) -> bool {
    let Some(output) = response["output"].as_array() else {
        return false;
    };
    let content = output
        .iter()
        .find(|item| item["type"] == "message")
        .and_then(|item| item["content"].as_array())
        .and_then(|parts| parts.first())
        .and_then(|part| part["text"].as_str());
    let reasoning_item = output.iter().find(|item| item["type"] == "reasoning");
    let reasoning = reasoning_item.and_then(|item| {
        item["content"].as_str().or_else(|| {
            item["content"]
                .as_array()
                .and_then(|parts| parts.first())
                .and_then(|part| part["text"].as_str())
        })
    });
    let calls = output
        .iter()
        .filter(|item| item["type"] == "function_call")
        .collect::<Vec<_>>();
    let call_matches = |index: usize, name: &str, arguments: Value| {
        calls.get(index).is_some_and(|call| {
            call["name"] == name
                && call["arguments"]
                    .as_str()
                    .and_then(|value| serde_json::from_str::<Value>(value).ok())
                    == Some(arguments)
        })
    };
    content == Some("The result is ready.")
        && reasoning == Some("careful reasoning")
        && calls.len() == 2
        && call_matches(0, "weather.lookup", json!({"city":"Paris","days":"2"}))
        && call_matches(1, "calendar.lookup", json!({"day":"tomorrow"}))
}

fn write_summary(summary: &Value) {
    let Some(directory) = env::var_os("RESPONSE_TEMPLATE_GATEWAY_SUMMARY_DIR") else {
        return;
    };
    let path = PathBuf::from(directory).join("gateway-summary.json");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap_or_else(|error| panic!("refusing to overwrite {}: {error}", path.display()));
    file.write_all(&serde_json::to_vec_pretty(summary).expect("summary JSON"))
        .expect("write gateway summary");
    file.flush().expect("flush gateway summary");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::disallowed_methods,
    reason = "test server tasks are explicitly aborted after each scenario"
)]
async fn response_template_gateway_semantics() {
    let template_value = serde_json::to_value(response_template()).expect("template value");

    // Trait, concrete HuggingFace loader, and cache wrapper must all retain
    // the exact raw checkpoint value.
    let hf_dir = write_tokenizer_dir(Some(template_value.clone()));
    let hf_path = hf_dir.path().join("tokenizer.json");
    let hf = HuggingFaceTokenizer::from_file(hf_path.to_str().expect("utf8 path"))
        .expect("load HuggingFace tokenizer");
    let response_template_retained_through_huggingface =
        hf.response_template() == Some(&template_value);
    assert!(response_template_retained_through_huggingface);
    let as_trait: Arc<dyn TokenizerTrait> = Arc::new(hf);
    let response_template_retained_through_trait =
        as_trait.response_template() == Some(&template_value);
    assert!(response_template_retained_through_trait);
    let cached = CachedTokenizer::new(as_trait, CacheConfig::default());
    let response_template_retained_through_cached_tokenizer =
        cached.response_template() == Some(&template_value);
    assert!(response_template_retained_through_cached_tokenizer);

    let load_variant = LoadError::InvalidResponseTemplate {
        model_name: MODEL.to_owned(),
        field: "response_template.fields.thinking.content".to_owned(),
        limit: 0,
        reason: "unsupported".to_owned(),
    };
    let typed_registry = TokenizerRegistry::new();
    let expected_load_variant = load_variant.clone();
    let preserved_load_error = typed_registry
        .load_typed("typed-id", MODEL, "typed-source", || async move {
            Err(expected_load_variant)
        })
        .await
        .expect_err("registry must preserve typed loader errors");
    let load_error_invalid_response_template_variant = matches!(
        &preserved_load_error,
        LoadError::InvalidResponseTemplate { .. }
    ) && preserved_load_error == load_variant
        && typed_registry.is_empty();
    assert!(load_error_invalid_response_template_variant);

    // Local invalid configuration: execute the real workflow, including its
    // event bus, and prove recognition happens before any remote fetch.
    let local_invalid_dir = write_tokenizer_dir(Some(invalid_template()));
    let local_port = free_port();
    let local_get_tokenizer_calls = Arc::new(AtomicUsize::new(0));
    install_test_controls(
        local_port,
        GrpcTestControls {
            output_ids: None,
            tokenizer_replies: VecDeque::from([GrpcTestTokenizerReply::Unimplemented]),
            get_tokenizer_calls: local_get_tokenizer_calls.clone(),
        },
    );
    let local_server = tokio::spawn(mock_worker::grpc::serve(
        mock_config(local_port),
        "127.0.0.1".to_owned(),
        local_port,
    ));
    tokio::time::sleep(Duration::from_millis(75)).await;
    let local_app = common::create_test_context(router_config()).await;
    register_grpc_worker(&local_app, local_port);
    let direct_local_data = create_tokenizer_workflow_data(
        TokenizerConfigRequest {
            id: "direct-local-invalid-id".to_owned(),
            name: MODEL.to_owned(),
            source: local_invalid_dir.path().to_string_lossy().into_owned(),
            chat_template_path: None,
            cache_config: None,
            fail_on_duplicate: true,
        },
        local_app.clone(),
    );
    let mut direct_local_context =
        WorkflowContext::new(WorkflowInstanceId::new(), direct_local_data);
    let direct_local_error = LoadTokenizerStep
        .execute(&mut direct_local_context)
        .await
        .expect_err("local invalid template must be typed configuration failure");
    let workflow_error_invalid_configuration_variant = matches!(
        direct_local_error,
        WorkflowError::InvalidConfiguration { .. }
    );
    assert!(workflow_error_invalid_configuration_variant);
    let (local_result, local_events) =
        registration(local_app.clone(), local_invalid_dir.path(), MODEL).await;
    let (starts, attempts, step_failed_events, local_retry_events, local_retry, local_message) =
        event_counts(&local_events);
    assert!(local_result.is_err());
    assert_eq!(
        (starts, attempts, step_failed_events, local_retry_events),
        (1, 1, 1, 0)
    );
    assert_eq!(local_retry, vec![false]);
    assert!(local_message.contains(MODEL));
    assert!(local_message.contains("thinking"));
    assert!(local_app.tokenizer_registry.is_empty());
    let observed_local_get_tokenizer_calls = local_get_tokenizer_calls.load(Ordering::SeqCst);
    assert_eq!(observed_local_get_tokenizer_calls, 0);
    local_server.abort();
    remove_test_controls(local_port);

    // Remote-only invalid bundle: local lookup fails, one healthy worker is
    // queried, the typed template error terminates traversal, and the other
    // healthy worker remains untouched.
    let invalid_source = TempDir::new().expect("empty local tokenizer source");
    let (invalid_zip, invalid_sha) = tokenizer_bundle(Some(invalid_template()));
    let first_port = free_port();
    let second_port = loop {
        let candidate = free_port();
        if candidate != first_port {
            break candidate;
        }
    };
    let first_calls = Arc::new(AtomicUsize::new(0));
    let second_calls = Arc::new(AtomicUsize::new(0));
    for (port, calls) in [
        (first_port, first_calls.clone()),
        (second_port, second_calls.clone()),
    ] {
        install_test_controls(
            port,
            GrpcTestControls {
                output_ids: None,
                tokenizer_replies: VecDeque::from([GrpcTestTokenizerReply::Bundle {
                    data: invalid_zip.clone(),
                    sha256: invalid_sha.clone(),
                }]),
                get_tokenizer_calls: calls,
            },
        );
    }
    let first_server = tokio::spawn(mock_worker::grpc::serve(
        mock_config(first_port),
        "127.0.0.1".to_owned(),
        first_port,
    ));
    let second_server = tokio::spawn(mock_worker::grpc::serve(
        mock_config(second_port),
        "127.0.0.1".to_owned(),
        second_port,
    ));
    tokio::time::sleep(Duration::from_millis(75)).await;
    let remote_app = common::create_test_context(router_config()).await;
    register_grpc_worker(&remote_app, first_port);
    register_grpc_worker(&remote_app, second_port);
    let (remote_result, remote_events) =
        registration(remote_app.clone(), invalid_source.path(), MODEL).await;
    let (_, remote_attempts, _, remote_retries, remote_retry, remote_message) =
        event_counts(&remote_events);
    let calls = [
        first_calls.load(Ordering::SeqCst),
        second_calls.load(Ordering::SeqCst),
    ];
    assert!(remote_result.is_err());
    assert_eq!(calls.iter().sum::<usize>(), 1);
    assert_eq!(calls.iter().filter(|count| **count == 0).count(), 1);
    assert_eq!(remote_attempts, 1);
    assert_eq!(remote_retries, 0);
    assert_eq!(remote_retry, vec![false]);
    assert!(remote_message.contains("Invalid configuration"));
    assert!(remote_app.tokenizer_registry.is_empty());
    let remote_bundle_parser_error =
        ResponseTemplateParser::from_json(MODEL, &invalid_template(), ParserConfig::default())
            .expect_err("the served bundle template must be typed-invalid");
    let remote_bundle_error_typed = matches!(
        remote_bundle_parser_error,
        ResponseTemplateError::InvalidTemplate { .. }
    ) && remote_retry == vec![false]
        && calls.iter().sum::<usize>() == 1;
    assert!(remote_bundle_error_typed);
    first_server.abort();
    second_server.abort();
    remove_test_controls(first_port);
    remove_test_controls(second_port);

    // A transport error remains retryable.  The first GetTokenizer fails,
    // the workflow retries, and the second fetch returns a template-less
    // tokenizer which is inserted normally.
    let fallback_source = TempDir::new().expect("empty fallback source");
    let (valid_zip, valid_sha) = tokenizer_bundle(None);
    let retry_port = free_port();
    let retry_calls = Arc::new(AtomicUsize::new(0));
    install_test_controls(
        retry_port,
        GrpcTestControls {
            output_ids: None,
            tokenizer_replies: VecDeque::from([
                GrpcTestTokenizerReply::Unavailable("temporary I/O failure".to_owned()),
                GrpcTestTokenizerReply::Bundle {
                    data: valid_zip.clone(),
                    sha256: valid_sha.clone(),
                },
            ]),
            get_tokenizer_calls: retry_calls.clone(),
        },
    );
    let retry_server = tokio::spawn(mock_worker::grpc::serve(
        mock_config(retry_port),
        "127.0.0.1".to_owned(),
        retry_port,
    ));
    tokio::time::sleep(Duration::from_millis(75)).await;
    let retry_app = common::create_test_context(router_config()).await;
    register_grpc_worker(&retry_app, retry_port);
    let (retry_result, retry_events) =
        registration(retry_app.clone(), fallback_source.path(), MODEL).await;
    let (_, retry_attempts, _, observed_retry_events, retry_flags, _) = event_counts(&retry_events);
    assert!(retry_result.is_ok());
    assert_eq!(retry_attempts, 2);
    assert_eq!(observed_retry_events, 1);
    assert_eq!(retry_flags, vec![true]);
    assert_eq!(retry_calls.load(Ordering::SeqCst), 2);
    assert_eq!(retry_app.tokenizer_registry.len(), 1);
    let registered = retry_app
        .tokenizer_registry
        .get(MODEL)
        .expect("fallback tokenizer registered");
    assert!(registered.response_template().is_none());
    retry_server.abort();
    remove_test_controls(retry_port);

    let misleading_error = WorkflowError::StepFailed {
        step_id: wfaas::StepId::new("load_tokenizer"),
        message: "I/O failure while reading invalid response template".to_owned(),
    };
    let misleading_io_string_retryable = <LoadTokenizerStep as StepExecutor<
        smg::workflow::TokenizerWorkflowData,
    >>::is_retryable(&LoadTokenizerStep, &misleading_error);
    assert!(misleading_io_string_retryable);

    let misleading_root = TempDir::new().expect("misleading I/O root");
    let misleading_source = misleading_root
        .path()
        .join("invalid response template ordinary io directory");
    fs::create_dir(&misleading_source).expect("create misleading ordinary I/O source");
    let misleading_port = free_port();
    let misleading_calls = Arc::new(AtomicUsize::new(0));
    install_test_controls(
        misleading_port,
        GrpcTestControls {
            output_ids: None,
            tokenizer_replies: VecDeque::from([GrpcTestTokenizerReply::Bundle {
                data: valid_zip,
                sha256: valid_sha,
            }]),
            get_tokenizer_calls: misleading_calls.clone(),
        },
    );
    let misleading_server = tokio::spawn(mock_worker::grpc::serve(
        mock_config(misleading_port),
        "127.0.0.1".to_owned(),
        misleading_port,
    ));
    tokio::time::sleep(Duration::from_millis(75)).await;
    let misleading_app = common::create_test_context(router_config()).await;
    register_grpc_worker(&misleading_app, misleading_port);
    let (misleading_result, misleading_events) =
        registration(misleading_app.clone(), &misleading_source, MODEL).await;
    assert!(misleading_result.is_ok());
    assert_eq!(misleading_calls.load(Ordering::SeqCst), 1);
    assert!(misleading_events
        .iter()
        .any(|event| matches!(event, WorkflowEvent::WorkflowCompleted { .. })));
    assert_eq!(misleading_app.tokenizer_registry.len(), 1);
    let misleading_io_string_fell_back = misleading_result.is_ok()
        && misleading_calls.load(Ordering::SeqCst) == 1
        && misleading_app.tokenizer_registry.len() == 1;
    misleading_server.abort();
    remove_test_controls(misleading_port);

    // Semantic oracle: exact one-chunk parsing equals every possible two-way
    // byte split and byte-at-a-time delivery, including delimiter interiors.
    let complete_wire = wire_output();
    let semantic_wire = complete_wire
        .strip_suffix("THIS_TEXT_MUST_NOT_BE_EMITTED")
        .expect("later-text suffix");
    let parser = ResponseTemplateParser::new(MODEL, response_template(), ParserConfig::default())
        .expect("valid parser");
    let nonstream = parser
        .parse_complete(PREFIX, semantic_wire)
        .expect("complete parse");
    let baseline = parse_streamed(semantic_wire.as_bytes(), &[semantic_wire.len()]);
    let mut mock_split_offsets = 0usize;
    for offset in 0..=semantic_wire.len() {
        let replay = parse_streamed(
            semantic_wire.as_bytes(),
            &[offset, semantic_wire.len() - offset],
        );
        assert_eq!(replay, baseline, "response changed at byte split {offset}");
        mock_split_offsets += 1;
    }
    let bytewise = parse_streamed(semantic_wire.as_bytes(), &vec![1; semantic_wire.len()]);
    assert_eq!(bytewise, baseline);
    mock_split_offsets += semantic_wire.len();
    assert_eq!(nonstream, baseline);
    assert_eq!(baseline.thinking, "careful reasoning");
    assert_eq!(baseline.content, "The result is ready.");
    assert_eq!(baseline.tool_calls.len(), 2);
    assert_eq!(baseline.tool_calls[0].name, "weather.lookup");
    assert_eq!(
        baseline.tool_calls[0].arguments,
        serde_json::from_value::<serde_json::Map<String, Value>>(
            json!({"city":"Paris","days":"2"})
        )
        .unwrap()
    );
    assert_eq!(baseline.tool_calls[1].name, "calendar.lookup");
    assert_eq!(
        baseline.tool_calls[0].transformed,
        json!({"type":"function","function":{"name":"weather.lookup","arguments":{"city":"Paris","days":"2"}}})
    );
    let streaming_equals_nonstreaming = bytewise == nonstream;
    let mapped = normalized(&baseline);
    let reasoning_content_toolcalls_mapped = mapped["reasoning_content"] == "careful reasoning"
        && mapped["content"] == "The result is ready."
        && mapped["tool_calls"]
            .as_array()
            .is_some_and(|calls| calls.len() == 2);
    assert!(streaming_equals_nonstreaming);
    assert!(reasoning_content_toolcalls_mapped);
    assert!(!contains_delimiter(&mapped));

    // Drive the ordinary Chat and Responses routes through the deterministic
    // mock-worker token sequence.  CLOSE_ID is both EOS and the template close,
    // so the gateway must expose it to the parser before stopping and hide all
    // later backend text.
    let semantic_port = free_port();
    install_test_controls(
        semantic_port,
        GrpcTestControls {
            output_ids: Some(OracleTokenizer::token_ids(&complete_wire)),
            tokenizer_replies: VecDeque::new(),
            get_tokenizer_calls: Arc::new(AtomicUsize::new(0)),
        },
    );
    let semantic_server = tokio::spawn(mock_worker::grpc::serve(
        mock_config(semantic_port),
        "127.0.0.1".to_owned(),
        semantic_port,
    ));
    tokio::time::sleep(Duration::from_millis(75)).await;
    let semantic_registry = Arc::new(TokenizerRegistry::new());
    let oracle: Arc<dyn TokenizerTrait> = Arc::new(OracleTokenizer::new(Some(template_value)));
    semantic_registry
        .load("oracle-id", MODEL, "oracle", || async move { Ok(oracle) })
        .await
        .expect("register oracle tokenizer");
    let semantic_app = common::create_test_context_with_tokenizer_registry(
        router_config(),
        semantic_registry.clone(),
    )
    .await;
    register_grpc_worker(&semantic_app, semantic_port);
    let router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&semantic_app)
            .await
            .expect("build semantic router"),
    );
    let app = common::test_app::create_test_app_with_context(router, semantic_app.clone());

    let chat_nonstream =
        request_json(&app, "/v1/chat/completions", semantic_chat_payload(false)).await;
    let chat_events = request_sse(&app, "/v1/chat/completions", semantic_chat_payload(true)).await;
    let chat_terminal_finish_reasons = chat_events
        .iter()
        .filter(|event| event["choices"][0]["finish_reason"].is_string())
        .count();
    let chat_nonstream_normalized =
        normalize_chat_message(&chat_nonstream["choices"][0]["message"]);
    let chat_stream_normalized = normalize_chat_events(&chat_events);
    assert_eq!(chat_stream_normalized, chat_nonstream_normalized);
    assert_eq!(chat_stream_normalized, mapped);
    let streamed_chat_content = chat_stream_normalized["content"]
        .as_str()
        .expect("stream content")
        .to_owned();
    assert_eq!(chat_terminal_finish_reasons, 1);
    assert_eq!(
        chat_nonstream["choices"][0]["message"]["reasoning_content"],
        "careful reasoning"
    );
    assert_eq!(
        chat_nonstream["choices"][0]["message"]["content"],
        "The result is ready."
    );
    assert_eq!(
        chat_nonstream["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(!contains_delimiter(&chat_nonstream));
    assert!(!contains_delimiter(&json!(chat_events)));

    let responses_nonstream = request_json(
        &app,
        "/v1/responses",
        json!({"model":MODEL,"input":"hello","stream":false,"store":false}),
    )
    .await;
    let response_events = request_sse(
        &app,
        "/v1/responses",
        json!({"model":MODEL,"input":"hello","stream":true,"store":false}),
    )
    .await;
    let completed_positions = response_events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| (event["type"] == "response.completed").then_some(index))
        .collect::<Vec<_>>();
    assert_eq!(completed_positions.len(), 1);
    let events_after_completed = response_events.len() - completed_positions[0] - 1;
    assert_eq!(events_after_completed, 0);
    assert_eq!(responses_nonstream["status"], "completed");
    let responses_nonstream_fields_mapped = responses_fields_mapped(&responses_nonstream);
    let responses_stream_fields_mapped =
        responses_fields_mapped(&response_events[completed_positions[0]]["response"]);
    assert!(responses_nonstream_fields_mapped);
    assert!(
        responses_stream_fields_mapped,
        "completed Responses stream lost mapped fields: terminal={} events={}",
        response_events[completed_positions[0]]["response"],
        json!(response_events)
    );
    assert!(!contains_delimiter(&responses_nonstream));
    assert!(!contains_delimiter(&json!(response_events)));
    let template_eos_close_handoff_nonstreaming =
        chat_nonstream_normalized["content"] == "The result is ready.";
    let template_eos_close_handoff_streaming = streamed_chat_content == "The result is ready.";
    let template_eos_close_terminations = usize::from(template_eos_close_handoff_nonstreaming)
        + usize::from(template_eos_close_handoff_streaming);
    let public_template_outputs = json!({
        "chat_nonstream": chat_nonstream.clone(),
        "chat_stream": chat_events.clone(),
        "responses_nonstream": responses_nonstream.clone(),
        "responses_stream": response_events.clone(),
    });
    let template_eos_close_leakage = marker_count(&public_template_outputs, "<|return|>");
    let template_eos_later_text_emitted =
        marker_count(&public_template_outputs, "THIS_TEXT_MUST_NOT_BE_EMITTED");
    assert_eq!(template_eos_close_terminations, 2);
    assert_eq!(template_eos_close_leakage, 0);
    assert_eq!(template_eos_later_text_emitted, 0);

    // A caller-owned visible stop is not private response-template framing.
    // Close the content field normally, then stop on a distinct user token:
    // both public Chat modes must retain that literal in content while still
    // consuming the template's own close delimiter.
    let external_stop_wire = format!(
        "{}<|end|>{USER_STOP_LITERAL}EXTERNAL_STOP_LATER_TEXT",
        semantic_wire
            .strip_suffix("<|return|>")
            .expect("semantic wire ends with its content close")
    );
    install_test_controls(
        semantic_port,
        GrpcTestControls {
            output_ids: Some(OracleTokenizer::token_ids(&external_stop_wire)),
            tokenizer_replies: VecDeque::new(),
            get_tokenizer_calls: Arc::new(AtomicUsize::new(0)),
        },
    );
    let visible_stop_nonstream = request_json(
        &app,
        "/v1/chat/completions",
        visible_stop_chat_payload(false),
    )
    .await;
    let visible_stop_events = request_sse(
        &app,
        "/v1/chat/completions",
        visible_stop_chat_payload(true),
    )
    .await;
    let visible_stop_complete =
        normalize_chat_message(&visible_stop_nonstream["choices"][0]["message"]);
    let visible_stop_stream = normalize_chat_events(&visible_stop_events);
    assert_eq!(visible_stop_stream, visible_stop_complete);
    assert_eq!(
        visible_stop_complete["content"],
        format!("The result is ready.{USER_STOP_LITERAL}")
    );
    assert_eq!(
        visible_stop_events
            .iter()
            .filter(|event| event["choices"][0]["finish_reason"].is_string())
            .count(),
        1
    );
    assert_eq!(marker_count(&visible_stop_nonstream, "<|end|>"), 0);
    assert_eq!(marker_count(&json!(visible_stop_events), "<|end|>"), 0);
    assert_eq!(
        marker_count(&visible_stop_nonstream, "EXTERNAL_STOP_LATER_TEXT"),
        0
    );
    assert_eq!(
        marker_count(&json!(visible_stop_events), "EXTERNAL_STOP_LATER_TEXT"),
        0
    );
    assert!(semantic_registry
        .get(MODEL)
        .and_then(|tokenizer| tokenizer.response_template().cloned())
        .is_some());

    // Cross the production StreamingProcessor's frozen default pending cap
    // through a real mocked Chat request. One synthetic tokenizer token
    // decodes to cap+1 ordinary bytes, avoiding a multi-million-frame gRPC
    // fixture while exercising the same production decoder/parser boundary.
    install_test_controls(
        semantic_port,
        GrpcTestControls {
            output_ids: Some(vec![OVERFLOW_ID]),
            tokenizer_replies: VecDeque::new(),
            get_tokenizer_calls: Arc::new(AtomicUsize::new(0)),
        },
    );
    let registry_before_gateway_overflow = semantic_registry.len();
    let gateway_overflow_events =
        request_sse(&app, "/v1/chat/completions", semantic_chat_payload(true)).await;
    let gateway_overflow_error = gateway_overflow_events
        .iter()
        .find_map(|event| event["error"]["message"].as_str())
        .expect("gateway overflow emits a terminal SSE error");
    assert!(gateway_overflow_events.iter().any(|event| {
        event["error"]["type"] == "response_template_pending_overflow"
            && event["error"]["message"] == gateway_overflow_error
    }));
    let expected_gateway_overflow = ResponseTemplateError::PendingOverflow {
        model_name: MODEL.to_owned(),
        field: "max_pending_bytes".to_owned(),
        limit: ParserConfig::default().max_pending_bytes,
    };
    let pending_overflow_is_runtime_failure =
        gateway_overflow_error == expected_gateway_overflow.to_string();
    assert!(pending_overflow_is_runtime_failure);
    let gateway_overflow_public = normalize_chat_events(&gateway_overflow_events);
    let pending_overflow_wire_bytes_leaked = gateway_overflow_public["reasoning_content"]
        .as_str()
        .map_or(0, str::len)
        + gateway_overflow_public["content"]
            .as_str()
            .map_or(0, str::len)
        + gateway_overflow_public["tool_calls"]
            .as_array()
            .map_or(0, Vec::len);
    let pending_overflow_registry_insertions = semantic_registry
        .len()
        .saturating_sub(registry_before_gateway_overflow);
    let pending_overflow_became_registration_error = gateway_overflow_error
        .contains("Invalid configuration")
        || pending_overflow_registry_insertions != 0;
    assert_eq!(pending_overflow_wire_bytes_leaked, 0);
    assert_eq!(pending_overflow_registry_insertions, 0);
    assert!(!pending_overflow_became_registration_error);
    semantic_server.abort();
    remove_test_controls(semantic_port);

    // Exercise the unchanged template-less branch through production Chat:
    // ordinary content keeps legacy precedence and EOS/later bytes stay hidden.
    let plain_wire = "plain legacy content<|return|>PLAIN_LATER_TEXT";
    let plain_port = free_port();
    install_test_controls(
        plain_port,
        GrpcTestControls {
            output_ids: Some(OracleTokenizer::token_ids(plain_wire)),
            tokenizer_replies: VecDeque::new(),
            get_tokenizer_calls: Arc::new(AtomicUsize::new(0)),
        },
    );
    let plain_server = tokio::spawn(mock_worker::grpc::serve(
        mock_config(plain_port),
        "127.0.0.1".to_owned(),
        plain_port,
    ));
    tokio::time::sleep(Duration::from_millis(75)).await;
    let plain_registry = Arc::new(TokenizerRegistry::new());
    let plain: Arc<dyn TokenizerTrait> = Arc::new(OracleTokenizer::new(None));
    let plain_for_registry = plain.clone();
    plain_registry
        .load("plain-id", MODEL, "plain", || async move {
            Ok(plain_for_registry)
        })
        .await
        .expect("register template-less oracle");
    let plain_app = common::create_test_context_with_tokenizer_registry(
        router_config(),
        plain_registry.clone(),
    )
    .await;
    register_grpc_worker(&plain_app, plain_port);
    let plain_router: Arc<dyn RouterTrait> = Arc::from(
        RouterFactory::create_router(&plain_app)
            .await
            .expect("build template-less router"),
    );
    let plain_axum = common::test_app::create_test_app_with_context(plain_router, plain_app);
    let plain_chat = request_json(
        &plain_axum,
        "/v1/chat/completions",
        json!({"model":MODEL,"messages":[{"role":"user","content":"hello"}],"stream":false}),
    )
    .await;
    let template_less_precedence_unchanged = plain_chat["choices"][0]["message"]["content"]
        == "plain legacy content"
        && plain_chat["choices"][0]["message"]["reasoning_content"].is_null()
        && plain_chat["choices"][0]["message"]["tool_calls"].is_null();
    let template_less_hidden_eos_unchanged = !contains_delimiter(&plain_chat)
        && marker_count(&plain_chat, "PLAIN_LATER_TEXT") == 0
        && plain_registry
            .get(MODEL)
            .is_some_and(|tokenizer| tokenizer.response_template().is_none());
    assert!(template_less_precedence_unchanged);
    assert!(template_less_hidden_eos_unchanged);
    plain_server.abort();
    remove_test_controls(plain_port);

    // Pin the low-level stop classification at its public seam as well.
    let mut hidden = StopSequenceDecoder::new(
        plain.clone(),
        StopSequenceConfig::default().with_stop_token(CLOSE_ID),
        false,
    );
    assert_eq!(
        hidden.process_token(CLOSE_ID).unwrap(),
        SequenceDecoderOutput::Stopped
    );
    let mut visible = StopSequenceDecoder::new(
        plain,
        StopSequenceConfig::default().with_visible_stop_token(CLOSE_ID),
        false,
    );
    assert_eq!(
        visible.process_token(CLOSE_ID).unwrap(),
        SequenceDecoderOutput::StoppedWithText("<|return|>".to_owned())
    );

    // PendingOverflow is request-time state, never a registration error.  Its
    // typed poison is sticky on subsequent feed and finish calls.
    let overflow_parser = ResponseTemplateParser::new(
        MODEL,
        response_template(),
        ParserConfig {
            max_pending_bytes: 8,
            max_structured_field_bytes: 4 * 1024 * 1024,
            max_body_bytes: 4 * 1024 * 1024,
        },
    )
    .expect("overflow parser");
    let mut overflow = overflow_parser.stream(PREFIX);
    let overflow_result = overflow.feed(b"xxxxxxxxx");
    let error = overflow_result.expect_err("pending cap + one");
    assert!(matches!(
        error,
        ResponseTemplateError::PendingOverflow { .. }
    ));
    let later = overflow.feed(b"later").expect_err("poisoned later feed");
    let finish = overflow.finish().expect_err("poisoned finish");
    let poisoned_on_later_feed = later == error;
    let poisoned_on_finish = finish == error;
    assert!(poisoned_on_later_feed);
    assert!(poisoned_on_finish);

    let invalid_template_terminal = local_result.is_err() && local_retry == vec![false];
    let recognised_before_remote_fetch = observed_local_get_tokenizer_calls == 0;
    let remote_invalid_configuration = remote_result.is_err()
        && remote_retry == vec![false]
        && remote_message.contains("Invalid configuration");
    let recoverable_io_retried = retry_attempts == 2
        && observed_retry_events == 1
        && retry_calls.load(Ordering::SeqCst) == 2;
    let recoverable_io_fell_back = retry_result.is_ok()
        && retry_app.tokenizer_registry.len() == 1
        && registered.response_template().is_none();
    let delimiter_leakage = contains_delimiter(&mapped)
        || contains_delimiter(&public_template_outputs)
        || contains_delimiter(&plain_chat);
    assert!(matches!(
        error,
        ResponseTemplateError::PendingOverflow { .. }
    ));
    let reasoning_content_toolcalls_mapped = reasoning_content_toolcalls_mapped
        && responses_nonstream_fields_mapped
        && responses_stream_fields_mapped;
    assert!(invalid_template_terminal);
    assert!(recognised_before_remote_fetch);
    assert!(remote_invalid_configuration);
    assert!(recoverable_io_retried);
    assert!(recoverable_io_fell_back);
    assert!(!delimiter_leakage);
    assert!(pending_overflow_is_runtime_failure);

    let summary = json!({
        "response_template_retained_through_trait": response_template_retained_through_trait,
        "response_template_retained_through_huggingface": response_template_retained_through_huggingface,
        "response_template_retained_through_cached_tokenizer": response_template_retained_through_cached_tokenizer,
        "load_error_invalid_response_template_variant": load_error_invalid_response_template_variant,
        "workflow_error_invalid_configuration_variant": workflow_error_invalid_configuration_variant,
        "invalid_template_terminal": invalid_template_terminal,
        "starts": starts,
        "attempts": attempts,
        "step_failed_events": step_failed_events,
        "will_retry": local_retry[0],
        "retry_events": local_retry_events,
        "get_tokenizer_calls": observed_local_get_tokenizer_calls,
        "registry_insertions": local_app.tokenizer_registry.len(),
        "message_names_model_and_field": local_message.contains(MODEL) && local_message.contains("thinking"),
        "recognised_before_remote_fetch": recognised_before_remote_fetch,
        "remote_invalid_get_tokenizer_calls": calls.iter().sum::<usize>(),
        "remote_invalid_second_worker_calls": *calls.iter().min().unwrap(),
        "remote_invalid_registry_insertions": remote_app.tokenizer_registry.len(),
        "remote_invalid_configuration": remote_invalid_configuration,
        "remote_invalid_will_retry": remote_retry[0],
        "remote_bundle_error_typed": remote_bundle_error_typed,
        "misleading_io_string_retryable": misleading_io_string_retryable,
        "misleading_io_string_fell_back": misleading_io_string_fell_back,
        "recoverable_io_retried": recoverable_io_retried,
        "recoverable_io_fell_back": recoverable_io_fell_back,
        "template_less_precedence_unchanged": template_less_precedence_unchanged,
        "template_eos_close_handoff_nonstreaming": template_eos_close_handoff_nonstreaming,
        "template_eos_close_handoff_streaming": template_eos_close_handoff_streaming,
        "template_eos_close_terminations": template_eos_close_terminations,
        "template_eos_close_leakage": template_eos_close_leakage,
        "template_eos_later_text_emitted": template_eos_later_text_emitted,
        "template_less_hidden_eos_unchanged": template_less_hidden_eos_unchanged,
        "mock_split_offsets": mock_split_offsets,
        "reasoning_content_toolcalls_mapped": reasoning_content_toolcalls_mapped,
        "tool_calls_parsed": baseline.tool_calls.len(),
        "streaming_equals_nonstreaming": streaming_equals_nonstreaming && chat_stream_normalized == chat_nonstream_normalized,
        "delimiter_leakage": delimiter_leakage,
        "chat_terminal_finish_reasons": chat_terminal_finish_reasons,
        "responses_completed_events": completed_positions.len(),
        "events_after_completed": events_after_completed,
        "nonstreaming_responses_status": responses_nonstream["status"].as_str().unwrap(),
        "pending_overflow_is_runtime_failure": pending_overflow_is_runtime_failure,
        "pending_overflow_became_registration_error": pending_overflow_became_registration_error,
        "pending_overflow_wire_bytes_leaked": pending_overflow_wire_bytes_leaked,
        "pending_overflow_registry_insertions": pending_overflow_registry_insertions,
        "poisoned_on_later_feed": poisoned_on_later_feed,
        "poisoned_on_finish": poisoned_on_finish
    });
    write_summary(&summary);
}
