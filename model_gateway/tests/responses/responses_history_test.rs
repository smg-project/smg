//! A Responses request that continues a stored turn carries that turn's tool
//! calls and their outputs to the engine, in order: through `conversation`
//! and through `previous_response_id`, over the real app with an in-process
//! scripted engine and the in-memory history backend; a conversation longer
//! than the history window reaches the engine with its newest items. The
//! engine receives the rendered prompt as ids; the test's tokenizer renders
//! the message list as JSON and encodes its bytes, so the prompt the engine
//! saw is read back.

use std::sync::{Arc, Mutex};

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Method, Request, StatusCode},
    Router,
};
use llm_tokenizer::{
    chat_template::ChatTemplateParams, traits::Tokenizer, Decoder, Encoder, Encoding,
    MockTokenizer, SpecialTokens, TokenizerRegistry,
};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    routers::RouterFactory,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_grpc_client::tokenspeed_scheduler::tokenspeed_proto as ts;
use tokio::net::TcpListener;
use tower::ServiceExt;

use crate::{common, scripted_worker};

const MODEL: &str = "history-model";

/// Prompt bytes are encoded above this id; the engine's generated tokens
/// (from 100) stay below it and decode to nothing.
const BYTE_BASE: u32 = 1000;

/// Renders the message list as JSON and encodes the text byte by byte, so
/// the prompt the engine received is read back from its ids.
struct PromptTokenizer {
    base: MockTokenizer,
}

impl PromptTokenizer {
    fn new() -> Self {
        Self {
            base: MockTokenizer::new().with_json_chat_template(),
        }
    }
}

fn prompt_text(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids
        .iter()
        .filter_map(|id| id.checked_sub(BYTE_BASE))
        .filter_map(|byte| u8::try_from(byte).ok())
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

impl Encoder for PromptTokenizer {
    fn encode(&self, text: &str, _add_special: bool) -> anyhow::Result<Encoding> {
        Ok(Encoding::Plain(
            text.bytes()
                .map(|byte| BYTE_BASE + u32::from(byte))
                .collect(),
        ))
    }

    fn encode_batch(&self, texts: &[&str], add_special: bool) -> anyhow::Result<Vec<Encoding>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special))
            .collect()
    }
}

impl Decoder for PromptTokenizer {
    fn decode(&self, ids: &[u32], _skip_special: bool) -> anyhow::Result<String> {
        Ok(prompt_text(ids))
    }
}

impl Tokenizer for PromptTokenizer {
    fn vocab_size(&self) -> usize {
        BYTE_BASE as usize + 256
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.base.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.base.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.base.id_to_token(id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn apply_chat_template(
        &self,
        messages: &[Value],
        params: ChatTemplateParams,
    ) -> anyhow::Result<String> {
        self.base.apply_chat_template(messages, params)
    }
}

/// The app over one scripted engine whose requests are kept in `requests`.
/// The engine's task is aborted when the handle drops.
#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process engine is aborted with its handle"
)]
async fn app() -> (
    Router,
    Arc<Mutex<Vec<ts::GenerateRequest>>>,
    tokio::task::AbortHandle,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let engine = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: 2,
            finish_reasons: vec!["stop"],
            generation: Default::default(),
            requests: requests.clone(),
        }
        .serve(listener),
    );
    let worker = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
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
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(PromptTokenizer::new());
    tokenizers
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let app = common::test_app::create_test_app_with_context(Arc::from(router), context);
    (app, requests, engine.abort_handle())
}

/// One call of the app.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn call(app: &Router, method: Method, path: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

/// The messages of the prompt the engine received last, as the rendered
/// message list.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper: failures should panic"
)]
fn last_prompt_messages(requests: &Mutex<Vec<ts::GenerateRequest>>) -> Vec<Value> {
    let requests = requests.lock().unwrap();
    let last = requests.last().expect("the engine received a request");
    let ids = &last
        .tokenized
        .as_ref()
        .expect("a tokenized prompt")
        .input_ids;
    let prompt: Value = serde_json::from_str(&prompt_text(ids)).expect("the rendered message list");
    prompt["messages"].as_array().expect("messages").clone()
}

/// The text of each message of the prompt the engine received last.
fn last_prompt_texts(requests: &Mutex<Vec<ts::GenerateRequest>>) -> Vec<String> {
    last_prompt_messages(requests)
        .iter()
        .filter_map(|message| match &message["content"] {
            Value::String(text) => Some(text.clone()),
            Value::Array(parts) => parts
                .iter()
                .find_map(|part| part["text"].as_str().map(str::to_string)),
            _ => None,
        })
        .collect()
}

/// The turns of the prompt the engine received last: the role of each
/// message, with the call id of a tool call or a tool result.
fn last_prompt_turns(requests: &Mutex<Vec<ts::GenerateRequest>>) -> Vec<String> {
    last_prompt_messages(requests)
        .iter()
        .map(|message| {
            let role = message["role"].as_str().unwrap_or("?");
            if let Some(calls) = message["tool_calls"].as_array() {
                let ids: Vec<&str> = calls
                    .iter()
                    .filter_map(|call| call["id"].as_str())
                    .collect();
                return format!("{role}+calls:{}", ids.join(","));
            }
            match message["tool_call_id"].as_str() {
                Some(call_id) => format!("{role}:{call_id}"),
                None => role.to_string(),
            }
        })
        .collect()
}

/// The stored turn asked the client to run `calc` (call_A); the client's
/// answer in the next request reaches the engine after the question and the
/// call it answers.
#[tokio::test]
async fn a_conversation_turn_reaches_the_engine_with_the_stored_call_before_its_output() {
    let (app, requests, engine) = app().await;

    let (status, conversation) = call(&app, Method::POST, "/v1/conversations", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{conversation}");
    let conversation_id = conversation["id"].as_str().expect("a conversation id");
    let (status, items) = call(
        &app,
        Method::POST,
        &format!("/v1/conversations/{conversation_id}/items"),
        json!({"items": [
            {"type": "message", "role": "user", "content": "What is 6*7?"},
            {"type": "function_call", "call_id": "call_A", "name": "calc",
             "arguments": "{\"expr\":\"6*7\"}", "status": "completed"}
        ]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{items}");

    let (status, response) = call(
        &app,
        Method::POST,
        "/v1/responses",
        json!({
            "model": MODEL,
            "conversation": conversation_id,
            "input": [{"type": "function_call_output", "call_id": "call_A", "output": "42"}],
            "max_output_tokens": 16,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed", "{response}");

    assert_eq!(
        last_prompt_turns(&requests),
        ["user", "assistant+calls:call_A", "tool:call_A"]
    );
    engine.abort();
}

/// A stored response whose input carried a call and its output replays
/// both, in order, in front of the next request's input.
#[tokio::test]
async fn a_previous_response_reaches_the_engine_with_its_call_and_output_in_order() {
    let (app, requests, engine) = app().await;

    let (status, first) = call(
        &app,
        Method::POST,
        "/v1/responses",
        json!({
            "model": MODEL,
            "input": [
                {"role": "user", "content": "What is 6*7?"},
                {"type": "function_call", "call_id": "call_A", "name": "calc",
                 "arguments": "{\"expr\":\"6*7\"}"},
                {"type": "function_call_output", "call_id": "call_A", "output": "42"}
            ],
            "max_output_tokens": 16,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let response_id = first["id"].as_str().expect("a response id");

    let (status, second) = call(
        &app,
        Method::POST,
        "/v1/responses",
        json!({
            "model": MODEL,
            "previous_response_id": response_id,
            "input": "Thanks",
            "max_output_tokens": 16,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");

    let turns = last_prompt_turns(&requests);
    let stored = ["user", "assistant+calls:call_A", "tool:call_A"];
    assert_eq!(&turns[..3], &stored, "{turns:?}");
    assert_eq!(turns.last().map(String::as_str), Some("user"), "{turns:?}");
    engine.abort();
}

/// Adds `items` to the conversation through the conversations API, which
/// takes 20 items a call.
async fn add_items(app: &Router, conversation_id: &str, items: &[Value]) {
    for batch in items.chunks(20) {
        let (status, added) = call(
            app,
            Method::POST,
            &format!("/v1/conversations/{conversation_id}/items"),
            json!({"items": batch}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{added}");
    }
}

/// Message `n` of a conversation: a user's for an odd `n`, the assistant's
/// for an even one, its text `item n`.
fn message_item(n: usize) -> Value {
    json!({
        "type": "message",
        "role": if n % 2 == 1 { "user" } else { "assistant" },
        "content": format!("item {n}"),
    })
}

/// A conversation of 120 items: the engine receives its newest 100, oldest
/// first, then the request's input; items 1 to 20 are not in the prompt.
#[tokio::test]
async fn a_conversation_longer_than_the_history_window_reaches_the_engine_with_its_newest_items() {
    let (app, requests, engine) = app().await;

    let (status, conversation) = call(&app, Method::POST, "/v1/conversations", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{conversation}");
    let conversation_id = conversation["id"].as_str().expect("a conversation id");
    let items: Vec<Value> = (1..=120).map(message_item).collect();
    add_items(&app, conversation_id, &items).await;

    let (status, response) = call(
        &app,
        Method::POST,
        "/v1/responses",
        json!({
            "model": MODEL,
            "conversation": conversation_id,
            "input": "item 121",
            "max_output_tokens": 16,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed", "{response}");

    let texts = last_prompt_texts(&requests);
    let expected: Vec<String> = (21..=121).map(|n| format!("item {n}")).collect();
    assert_eq!(
        texts.first().map(String::as_str),
        Some("item 21"),
        "the prompt begins at the oldest of the newest 100 items: {texts:?}"
    );
    assert_eq!(texts, expected);
    engine.abort();
}

/// A conversation one item longer than the window, whose first items are two
/// calls with their outputs: the window begins at the first output, inside a
/// turn the engine would not see the start of. The engine receives the window
/// from its first user message; the output before it and the call with its
/// output after it go with the turn the cut split.
#[tokio::test]
async fn a_history_window_that_begins_inside_a_turn_reaches_the_engine_from_its_user_message() {
    let (app, requests, engine) = app().await;

    let (status, conversation) = call(&app, Method::POST, "/v1/conversations", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{conversation}");
    let conversation_id = conversation["id"].as_str().expect("a conversation id");
    let call_item = |call_id: &str| {
        json!({"type": "function_call", "call_id": call_id, "name": "calc",
               "arguments": "{\"expr\":\"6*7\"}", "status": "completed"})
    };
    let output_item =
        |call_id: &str| json!({"type": "function_call_output", "call_id": call_id, "output": "42"});
    let mut items = vec![
        call_item("call_A"),
        output_item("call_A"),
        call_item("call_B"),
        output_item("call_B"),
    ];
    items.extend((5..=101).map(message_item));
    add_items(&app, conversation_id, &items).await;

    let (status, response) = call(
        &app,
        Method::POST,
        "/v1/responses",
        json!({
            "model": MODEL,
            "conversation": conversation_id,
            "input": "item 102",
            "max_output_tokens": 16,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed", "{response}");

    let turns = last_prompt_turns(&requests);
    assert_eq!(turns[0], "user", "{turns:?}");
    assert!(
        !turns
            .iter()
            .any(|turn| turn.starts_with("assistant+calls") || turn.starts_with("tool:")),
        "{turns:?}"
    );
    // the calls and outputs are the tail of the turn the cut split: the
    // messages reach the engine, then the request
    assert_eq!(turns.len(), 101 - 4 + 1, "{turns:?}");
    assert_eq!(
        last_prompt_texts(&requests).last().map(String::as_str),
        Some("item 102")
    );
    engine.abort();
}

/// A conversation of 150 items: the engine receives items 51 to 150, oldest
/// first, then the request's input; the cut lands on a user message, a turn
/// boundary, so the window reaches the engine whole.
#[tokio::test]
async fn a_conversation_of_150_items_reaches_the_engine_from_item_51() {
    let (app, requests, engine) = app().await;

    let (status, conversation) = call(&app, Method::POST, "/v1/conversations", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{conversation}");
    let conversation_id = conversation["id"].as_str().expect("a conversation id");
    let items: Vec<Value> = (1..=150).map(message_item).collect();
    add_items(&app, conversation_id, &items).await;

    let (status, response) = call(
        &app,
        Method::POST,
        "/v1/responses",
        json!({
            "model": MODEL,
            "conversation": conversation_id,
            "input": "item 151",
            "max_output_tokens": 16,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed", "{response}");

    let texts = last_prompt_texts(&requests);
    let expected: Vec<String> = (51..=151).map(|n| format!("item {n}")).collect();
    assert_eq!(texts, expected);
    engine.abort();
}

/// The items of a conversation of agent turns, each the user's message, the
/// model's call, its output and the model's answer: 37 whole turns, then a
/// user message and an open call, 150 items.
fn agent_turn_items() -> Vec<Value> {
    let call_item = |turn: usize| {
        json!({"type": "function_call", "call_id": format!("call_{turn}"), "name": "calc",
               "arguments": "{\"expr\":\"6*7\"}", "status": "completed"})
    };
    let output_item = |turn: usize| json!({"type": "function_call_output", "call_id": format!("call_{turn}"), "output": "42"});
    let mut items = Vec::new();
    for turn in 1..=37 {
        items.push(message_item(4 * turn - 3));
        items.push(call_item(turn));
        items.push(output_item(turn));
        items.push(message_item(4 * turn));
    }
    items.push(message_item(149));
    items.push(call_item(38));
    items
}

/// A window cut inside a turn (item 51 is the output of turn 13's call): the
/// engine's prompt begins at the next turn boundary, the user's message of
/// turn 14 (item 53), and ends with the request; the output and the answer
/// of turn 13 are not in it.
#[tokio::test]
async fn a_history_window_cut_inside_a_turn_reaches_the_engine_from_the_next_turn() {
    let (app, requests, engine) = app().await;

    let (status, conversation) = call(&app, Method::POST, "/v1/conversations", json!({})).await;
    assert_eq!(status, StatusCode::OK, "{conversation}");
    let conversation_id = conversation["id"].as_str().expect("a conversation id");
    add_items(&app, conversation_id, &agent_turn_items()).await;

    let (status, response) = call(
        &app,
        Method::POST,
        "/v1/responses",
        json!({
            "model": MODEL,
            "conversation": conversation_id,
            "input": "item 151",
            "max_output_tokens": 16,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["status"], "completed", "{response}");

    let turns = last_prompt_turns(&requests);
    assert_eq!(
        &turns[..4],
        [
            "user",
            "assistant+calls:call_14",
            "tool:call_14",
            "assistant"
        ],
        "{turns:?}"
    );
    assert!(!turns.contains(&"tool:call_13".to_string()), "{turns:?}");
    // items 53 to 150, then the request
    assert_eq!(turns.len(), 150 - 52 + 1, "{turns:?}");
    let texts = last_prompt_texts(&requests);
    assert_eq!(
        texts.first().map(String::as_str),
        Some("item 53"),
        "{texts:?}"
    );
    assert_eq!(texts.last().map(String::as_str), Some("item 151"));
    engine.abort();
}
