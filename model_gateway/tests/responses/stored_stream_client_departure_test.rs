//! A stored streamed response whose client leaves mid-stream is retrievable
//! afterwards under the id `response.created` announced, with its status, and
//! a chain on that id is served.

use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Method, Request, StatusCode},
    Router,
};
use futures_util::StreamExt;
use llm_tokenizer::{traits::Tokenizer, TokenizerRegistry};
use metrics_exporter_prometheus::PrometheusHandle;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    observability::metrics::{start_prometheus, PrometheusConfig},
    routers::RouterFactory,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use smg_test_harness::{held_worker::HeldWorker, scripted_tokenizer::ScriptedTokenizer};
use tokio::net::TcpListener;
use tower::ServiceExt;

use crate::common;

const MODEL: &str = "responses-departure-model";
/// The model of the test that reads the tool-call count: a series no other
/// test of the binary touches.
const METRICS_MODEL: &str = "responses-departure-metrics-model";
/// The scripted engine's first generation: a call of the MCP server's search
/// tool; its second answers plainly.
const TOOL_CALL: &str =
    "<tool_call>\n{\"name\":\"brave_web_search\",\"arguments\":{\"query\":\"test\"}}\n</tool_call>";

/// The production route wrapper over one gRPC worker that answers its stream
/// at once and its first token two seconds later (the shape of a prompt
/// still being read when the client leaves), with the memory stores.
async fn app() -> (Router, tokio::task::AbortHandle) {
    app_with(
        HeldWorker {
            first_token_after: Duration::from_secs(2),
            output_tokens: 3,
        },
        None,
        MODEL,
    )
    .await
}

/// The route wrapper over the given held worker; with a scripted text, the
/// engine's generations decode to it (a tool call first, a plain answer
/// after one tool result) under the `qwen` tool-call parser.
#[expect(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process worker is aborted after the request"
)]
async fn app_with(
    engine: HeldWorker,
    scripted: Option<&str>,
    model: &str,
) -> (Router, tokio::task::AbortHandle) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let engine = tokio::spawn(engine.serve(listener));
    let worker = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(model))
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
    let tokenizers = match scripted {
        Some(text) => {
            config.tool_call_parser = Some("qwen".into());
            let tokenizers = Arc::new(TokenizerRegistry::new());
            let tokenizer: Arc<dyn Tokenizer> = Arc::new(
                ScriptedTokenizer::new(text).with_final_answer(1, "Final answer after tools"),
            );
            tokenizers
                .load(
                    "tokenizer-id",
                    model,
                    "test",
                    || async move { Ok(tokenizer) },
                )
                .await
                .unwrap();
            tokenizers
        }
        None => common::mock_tokenizer_registry(model).await,
    };
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context).await.unwrap();
    let app = common::test_app::create_test_app_with_context(Arc::from(router), context);
    (app, engine.abort_handle())
}

/// One request to the app: the status and the body's JSON.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn call(app: &Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(body.map_or_else(Body::empty, |body| Body::from(body.to_string())))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}
/// The id the stream announces in `response.created`; the body is dropped
/// once it is read, which is the client's leaving.
async fn announced_id(response: axum::response::Response) -> String {
    leave_when(response, |event| event["type"] == "response.created").await
}

/// The client reads the stream until an event `leave` names and drops the
/// body there; the id `response.created` announced comes back.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper: failures should panic"
)]
async fn leave_when(response: axum::response::Response, leave: impl Fn(&Value) -> bool) -> String {
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut id = None;
    let mut left = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while !left {
        let frame = tokio::time::timeout_at(deadline, body.next())
            .await
            .expect("the first frames arrive before the first token")
            .expect("the stream is open")
            .unwrap();
        let text = String::from_utf8_lossy(&frame);
        for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
            let event: Value = serde_json::from_str(data).unwrap();
            if event["type"] == "response.created" {
                id = event["response"]["id"].as_str().map(str::to_string);
            }
            left |= leave(&event);
        }
    }
    drop(body);
    id.unwrap()
}

/// The record under `id`, polled while the departure is being written.
#[expect(clippy::panic, reason = "test helper: failures should panic")]
async fn record_of(app: &Router, id: &str) -> Value {
    for _ in 0..50 {
        let (status, body) = call(app, Method::GET, &format!("/v1/responses/{id}"), None).await;
        if status == StatusCode::OK {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("the stored stream's record is not retrievable")
}

#[tokio::test]
async fn a_stored_stream_whose_client_leaves_is_retrievable_with_its_status() {
    let (app, engine) = app().await;
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": MODEL, "input": "Hello", "stream": true, "store": true, "max_output_tokens": 8})
                .to_string(),
        ))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let id = announced_id(response).await;

    // The record is written when the departure is met; the GET answers from it.
    let stored = record_of(&app, &id).await;
    assert_eq!(stored["id"], id.as_str());
    assert_eq!(stored["status"], "cancelled", "{stored}");
    assert!(stored["error"].is_null(), "{stored}");
    assert_eq!(stored["store"], true, "{stored}");

    // A chain on the id is served, not refused as a missing previous response.
    let (status, next) = call(
        &app,
        Method::POST,
        "/v1/responses",
        Some(json!({"model": MODEL, "input": "And then?", "previous_response_id": id, "max_output_tokens": 4})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{next}");
    assert_eq!(next["previous_response_id"], id.as_str(), "{next}");
    engine.abort();
}

/// The MCP tool-loop stream (a request with an MCP tool) records the same
/// departure the same way: `cancelled`, no error, under the id it announced,
/// where it used to record `failed` with a `stream_error`.
#[tokio::test]
async fn a_stored_tool_loop_stream_whose_client_leaves_is_recorded_as_cancelled() {
    let (app, engine) = app().await;
    let mut mcp = common::mock_mcp_server::MockMCPServer::start()
        .await
        .unwrap();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": MODEL, "input": "Hello", "stream": true, "store": true, "max_output_tokens": 8,
                "tools": [{"type": "mcp", "server_label": "test-tools", "server_url": mcp.url(),
                           "require_approval": "never"}]})
            .to_string(),
        ))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let id = announced_id(response).await;

    let stored = record_of(&app, &id).await;
    assert_eq!(stored["id"], id.as_str());
    assert_eq!(stored["status"], "cancelled", "{stored}");
    assert!(stored["error"].is_null(), "{stored}");
    assert_eq!(stored["store"], true, "{stored}");
    // The tool list streamed before the departure is kept.
    assert_eq!(stored["output"][0]["type"], "mcp_list_tools", "{stored}");
    mcp.stop().await;
    engine.abort();
}

/// A departure in the generation after a tool call is recorded as `cancelled`
/// (the earlier generation's finish does not stand for this one), with the
/// tool list and the completed call kept.
#[tokio::test]
async fn a_departure_in_the_generation_after_a_tool_call_is_recorded_as_cancelled() {
    let (app, engine) = app_with(
        HeldWorker {
            first_token_after: Duration::from_millis(300),
            output_tokens: 2,
        },
        Some(TOOL_CALL),
        MODEL,
    )
    .await;
    let mut mcp = common::mock_mcp_server::MockMCPServer::start()
        .await
        .unwrap();
    let response = app
        .clone()
        .oneshot(tool_loop_request(&mcp.url(), MODEL))
        .await
        .unwrap();
    let id = leave_when(response, |event| {
        event["type"] == "response.output_item.done" && event["item"]["type"] == "mcp_call"
    })
    .await;

    let stored = record_of(&app, &id).await;
    assert_eq!(stored["status"], "cancelled", "{stored}");
    assert!(stored["error"].is_null(), "{stored}");
    assert_eq!(stored["output"][0]["type"], "mcp_list_tools", "{stored}");
    assert_eq!(stored["output"][1]["type"], "mcp_call", "{stored}");
    assert_eq!(stored["output"][1]["status"], "completed", "{stored}");
    assert_eq!(mcp.call_count(), 1);
    mcp.stop().await;
    engine.abort();
}

/// A departure while the tool executes keeps the executed call, with its
/// output, in the record and in the tool-call count: the item is stored and
/// the call recorded before the sends that fail.
#[tokio::test]
async fn a_departure_during_the_tool_call_keeps_the_executed_call_in_the_record() {
    let handle = recorder();
    let counted = counted_tool_calls(&handle, METRICS_MODEL);
    let (app, engine) = app_with(
        HeldWorker {
            first_token_after: Duration::from_millis(300),
            output_tokens: 2,
        },
        Some(TOOL_CALL),
        METRICS_MODEL,
    )
    .await;
    let mut mcp =
        common::mock_mcp_server::MockMCPServer::start_with_latency(Duration::from_millis(300))
            .await
            .unwrap();
    let response = app
        .clone()
        .oneshot(tool_loop_request(&mcp.url(), METRICS_MODEL))
        .await
        .unwrap();
    let id = leave_when(response, |event| {
        event["type"] == "response.mcp_call_arguments.done"
    })
    .await;

    let stored = record_of(&app, &id).await;
    assert_eq!(stored["status"], "cancelled", "{stored}");
    assert!(stored["error"].is_null(), "{stored}");
    assert_eq!(stored["output"][1]["type"], "mcp_call", "{stored}");
    assert_eq!(stored["output"][1]["status"], "completed", "{stored}");
    assert_eq!(stored["output"][1]["name"], "brave_web_search", "{stored}");
    assert!(
        stored["output"][1]["output"]
            .as_str()
            .is_some_and(|output| output.contains("Mock search results for: test")),
        "{stored}"
    );
    assert_eq!(mcp.call_count(), 1);
    assert_eq!(
        counted_tool_calls(&handle, METRICS_MODEL),
        counted + 1,
        "the executed call is counted once although its client left"
    );
    mcp.stop().await;
    engine.abort();
}

/// A stored streamed request with the MCP server as a tool.
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
fn tool_loop_request(server_url: &str, model: &str) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({"model": model, "input": "Hello", "stream": true, "store": true, "max_output_tokens": 16,
                "tools": [{"type": "mcp", "server_label": "test-tools", "server_url": server_url,
                           "require_approval": "never"}]})
            .to_string(),
        ))
        .unwrap()
}

/// The binary's recorder, installed on the first call (a second install
/// panics) and shared by every test of the binary; the test that reads it owns
/// its series (a model label of its own).
fn recorder() -> PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE
        .get_or_init(|| {
            start_prometheus(PrometheusConfig {
                port: 0,
                host: "127.0.0.1".to_string(),
                duration_buckets: None,
            })
        })
        .clone()
}

/// The successful calls of the MCP server's tool counted under the given
/// model, summed over the rendered series (0 before the first).
fn counted_tool_calls(handle: &PrometheusHandle, model: &str) -> u64 {
    let model = format!("model=\"{model}\"");
    handle
        .render()
        .lines()
        .filter(|line| {
            line.starts_with("smg_mcp_tool_calls_total{")
                && line.contains(&model)
                && line.contains("tool_name=\"brave_web_search\"")
                && line.contains("result=\"success\"")
        })
        .filter_map(|line| line.rsplit(' ').next()?.parse::<u64>().ok())
        .sum()
}
