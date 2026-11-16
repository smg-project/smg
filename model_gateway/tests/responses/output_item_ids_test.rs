//! Every output item of a Responses answer carries an id of its own, with the
//! prefix of its type (`msg_` a message, `rs_` a reasoning item, `fc_` a
//! function call), distinct from the `call_id` a function call carries, and
//! the same id on the streamed events, the terminal object and the stored
//! object.

use std::{collections::HashSet, sync::Arc, time::Duration};

use axum::{body::to_bytes, http::StatusCode};
use llm_tokenizer::{traits::Tokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::RouterFactory,
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_test_harness::scripted_tokenizer;
use tokio::{net::TcpListener, time::timeout};

use crate::{common, scripted_worker};

const MODEL: &str = "responses-item-ids-model";

/// What the model says, chunk by chunk, which parsers read it, and the tools
/// the request offers.
struct Script {
    chunks: &'static [&'static str],
    tool_parser: Option<&'static str>,
    reasoning_parser: Option<&'static str>,
    tools: Option<Value>,
}

/// One stored `POST /v1/responses` of `script`, streamed or not: the final
/// object (the terminal event's for a stream), the stored object read back
/// under the id the final object carries, and the stream's events (none for
/// the plain answer).
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process worker is aborted after the request"
)]
async fn answer(script: &Script, stream: bool) -> (Value, Value, Vec<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(
        scripted_worker::ScriptedWorker {
            output_tokens: u32::try_from(script.chunks.len() + 1)
                .expect("scripted output fits in the worker token count"),
            finish_reasons: vec!["stop"],
            generation: Default::default(),
            requests: Default::default(),
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
    config.tool_call_parser = script.tool_parser.map(str::to_string);
    config.reasoning_parser = script.reasoning_parser.map(str::to_string);
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> =
        Arc::new(scripted_tokenizer::ScriptedTokenizer::from_chunks(
            script
                .chunks
                .iter()
                .map(|chunk| chunk.to_string())
                .collect(),
        ));
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
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    let mut body = json!({
        "model": MODEL, "input": "Hello", "stream": stream, "store": true,
        "max_output_tokens": 16,
    });
    if let Some(tools) = &script.tools {
        body["tools"] = tools.clone();
    }
    let request = serde_json::from_value(body).unwrap();
    let (status, bytes) = timeout(Duration::from_secs(30), async {
        let response = router.route_responses(None, &tenant, request, MODEL).await;
        let status = response.status();
        (
            status,
            to_bytes(response.into_body(), 1024 * 1024).await.unwrap(),
        )
    })
    .await
    .expect("the Responses answer ends");
    server.abort();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let (object, events) = if stream {
        let body = std::str::from_utf8(&bytes).unwrap();
        let events: Vec<Value> = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).expect("every Responses SSE payload is JSON"))
            .collect();
        let terminal = events.last().unwrap();
        assert_eq!(terminal["type"], "response.completed", "{terminal}");
        (terminal["response"].clone(), events)
    } else {
        (serde_json::from_slice(&bytes).unwrap(), vec![])
    };
    let stored = context
        .response_storage
        .get_response(&smg_data_connector::ResponseId::from(
            object["id"].as_str().unwrap(),
        ))
        .await
        .unwrap()
        .expect("a stored response")
        .raw_response;
    (object, stored, events)
}

/// The item kinds of a response object's output, in order.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
fn kinds(object: &Value) -> Vec<&str> {
    object["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["type"].as_str().unwrap())
        .collect()
}

/// Every item of `object` has an id of its own with its kind's prefix, a
/// function call's distinct from its `call_id`; the stored object carries the
/// same items under the same ids; on a stream, each item's
/// `response.output_item.added` and `.done` name its id once and every
/// item-scoped event at its output index names the same id.
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test assertions: failures should panic; an unscripted item kind is a test bug"
)]
fn assert_item_ids(object: &Value, stored: &Value, events: &[Value]) {
    let output = object["output"].as_array().unwrap();
    assert!(!output.is_empty(), "{object}");
    let mut seen = HashSet::new();
    for (index, item) in output.iter().enumerate() {
        let kind = item["type"].as_str().unwrap();
        let id = item["id"].as_str().unwrap();
        // The id prefix the API gives an output item of each scripted kind.
        let prefix = match kind {
            "message" => "msg_",
            "reasoning" => "rs_",
            "function_call" => "fc_",
            other => panic!("an output item kind this test does not script: {other}"),
        };
        assert!(id.starts_with(prefix), "{item}");
        assert!(seen.insert(id.to_string()), "two items share the id {id}");
        if kind == "function_call" {
            let call_id = item["call_id"].as_str().unwrap();
            assert!(call_id.starts_with("call_"), "{item}");
            assert_ne!(id, call_id, "{item}");
        }
        assert_eq!(stored["output"][index]["type"], kind, "{stored}");
        assert_eq!(stored["output"][index]["id"], id, "{stored}");
        if events.is_empty() {
            continue;
        }
        for event_type in ["response.output_item.added", "response.output_item.done"] {
            assert_eq!(
                events
                    .iter()
                    .filter(|e| e["type"] == event_type && e["item"]["id"] == id)
                    .count(),
                1,
                "{event_type} of {id}"
            );
        }
        assert!(
            events
                .iter()
                .filter(|e| e["output_index"] == index && e["item_id"].is_string())
                .all(|e| e["item_id"] == id),
            "an item-scoped event at output index {index} names another id: {events:?}"
        );
    }
}

/// A turn that calls a function: the `function_call` item's id is its own
/// (`fc_`), the `call_id` the client echoes is another string, and the stored
/// object carries the item under the streamed id.
#[tokio::test]
async fn a_function_call_item_has_an_id_of_its_own_beside_its_call_id() {
    let script = Script {
        chunks: &["<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>"],
        tool_parser: Some("qwen"),
        reasoning_parser: None,
        tools: Some(json!([{
            "type": "function", "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }])),
    };
    for stream in [false, true] {
        let (object, stored, events) = answer(&script, stream).await;
        assert_eq!(
            kinds(&object),
            ["function_call"],
            "stream {stream}: {object}"
        );
        assert_item_ids(&object, &stored, &events);
    }
}

/// A turn that reasons before it answers: the reasoning item is `rs_`, the
/// message `msg_`, on the plain answer, the stream and the stored object.
#[tokio::test]
async fn a_reasoning_item_and_a_message_carry_their_kinds_prefixes() {
    let script = Script {
        chunks: &["<think>", "plan", "</think>", "Hello"],
        tool_parser: None,
        reasoning_parser: Some("qwen3"),
        tools: None,
    };
    for stream in [false, true] {
        let (object, stored, events) = answer(&script, stream).await;
        assert_eq!(
            kinds(&object),
            ["reasoning", "message"],
            "stream {stream}: {object}"
        );
        assert_item_ids(&object, &stored, &events);
    }
}
