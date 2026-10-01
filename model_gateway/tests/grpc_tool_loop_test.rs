//! The gRPC Responses tool loop against a scripted worker and a mock MCP
//! server: a frozen event sequence per family, so a change to the loop is
//! judged against what the wire says today rather than by argument.
//!
//! The worker answers each turn with scripted token ids and the mock
//! tokenizer decodes them into exact text, so the first turn is a JSON tool
//! call the `json` parser recognizes and the second is the final answer.

mod common;

use std::{sync::Arc, time::Duration};

use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{
    model_card::ModelCard, responses::ResponsesRequest, worker::HealthCheckConfig,
};
use serde_json::{json, Value};
use smg::{
    config::RouterConfig,
    middleware::TenantRequestMeta,
    routers::{RouterFactory, RouterTrait},
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::net::TcpListener;

use crate::common::mock_mcp_server::MockMCPServer;

const MODEL: &str = "tool-loop-test-model";

/// Token ids the scripted worker answers with, decoded by the mock
/// tokenizer into exact text.
const TOOL_CALL_TOKEN: u32 = 2001;
const FINAL_TOKEN_A: u32 = 2002;
const FINAL_TOKEN_B: u32 = 2003;
const TOOL_CALL_TEXT: &str = r#"{"name": "brave_web_search", "arguments": {"query": "rust"}}"#;

/// The regular family's streaming envelope for one MCP call followed by a
/// two-token answer, as the wire says it today: the tool item closes after
/// execution with its output, the answer follows in its own message item,
/// and the stream ends with `[DONE]`.
const REGULAR_STREAMING_SEQUENCE: &[&str] = &[
    "response.created",
    "response.in_progress",
    "response.output_item.added(mcp_list_tools)",
    "response.mcp_list_tools.in_progress",
    "response.mcp_list_tools.completed",
    "response.output_item.done(mcp_list_tools)",
    "response.output_item.added(mcp_call)",
    "response.mcp_call.in_progress",
    "response.mcp_call_arguments.delta",
    "response.mcp_call_arguments.done",
    "response.mcp_call.completed",
    "response.output_item.done(mcp_call)",
    "response.output_item.added(message)",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done(message)",
    "response.completed",
    "[DONE]",
];

fn scripted_tokenizer() -> Arc<dyn Tokenizer> {
    Arc::new(MockTokenizer::new().with_tokens(&[
        (TOOL_CALL_TEXT, TOOL_CALL_TOKEN),
        ("Final", FINAL_TOKEN_A),
        ("answer", FINAL_TOKEN_B),
    ]))
}

#[expect(
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper - panicking on failure is intentional; the spawned mock \
              server task is fire-and-forget for the test process's lifetime"
)]
async fn start_scripted_grpc_worker(scripted_outputs: Vec<Vec<u32>>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock gRPC worker");
    let port = listener
        .local_addr()
        .expect("mock gRPC worker address")
        .port();
    let cfg = Arc::new(mock_worker::config::Config {
        host: "127.0.0.1".to_string(),
        http_base_port: 0,
        http_count: 0,
        grpc_base_port: port,
        grpc_count: 1,
        zmq_handshake: None,
        zmq_count: 0,
        zmq_start_index: 0,
        model_id: MODEL.to_string(),
        tokenizer_path: MODEL.to_string(),
        gen_delay: Duration::ZERO,
        output_tokens: 2,
        realistic: false,
        engine: mock_worker::engine::EngineParams::default(),
        scripted_outputs,
    });
    tokio::spawn(mock_worker::grpc::serve_with_listener(cfg, listener));
    port
}

/// A gRPC regular router over one scripted worker, with the `json` tool
/// parser and an MCP orchestrator ready for request-scoped servers.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test helper - panicking on failure is intentional"
)]
async fn build_router(grpc_port: u16) -> Box<dyn RouterTrait> {
    let mut config = RouterConfig::builder()
        .grpc_connection()
        .regular_mode(vec![])
        .random_policy()
        .tool_call_parser("json")
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .request_timeout_secs(60)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked();
    config.health_check.disable_health_check = true;

    let tokenizer_registry = Arc::new(TokenizerRegistry::new());
    let tokenizer = scripted_tokenizer();
    tokenizer_registry
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let app_context =
        common::create_test_context_with_tokenizer_registry(config, tokenizer_registry).await;
    let worker = BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{grpc_port}"))
        .worker_type(WorkerType::Regular)
        .connection_mode(ConnectionMode::Grpc)
        .runtime_type(RuntimeType::TokenSpeed)
        .model(ModelCard::new(MODEL))
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build();
    app_context
        .worker_registry
        .register(Arc::new(worker))
        .unwrap();
    RouterFactory::create_router(&app_context)
        .await
        .expect("gRPC router should build")
}

#[expect(clippy::unwrap_used, reason = "test helper")]
fn responses_request(mcp_url: &str, stream: bool) -> ResponsesRequest {
    serde_json::from_value(json!({
        "model": MODEL,
        "input": "Hello world",
        "stream": stream,
        "store": false,
        "max_tool_calls": 3,
        "tools": [{
            "type": "mcp",
            "server_label": "mock",
            "server_url": mcp_url,
            "require_approval": "never"
        }]
    }))
    .unwrap()
}

fn tenant_meta() -> TenantRequestMeta {
    TenantRequestMeta::new(TenantKey::new("tool-loop-test"))
}

/// One SSE frame: the `event:` name and its parsed `data:`; the `[DONE]`
/// marker is kept as an event named `[DONE]` so termination is part of the
/// frozen sequence.
fn parse_sse(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter(|block| !block.trim().is_empty())
        .filter_map(|block| {
            let mut event = None;
            let mut data = Vec::new();
            for line in block.lines() {
                if let Some(rest) = line.strip_prefix("event:") {
                    event = Some(rest.trim().to_string());
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data.push(rest.trim_start().to_string());
                }
            }
            let data = data.join("\n");
            if data == "[DONE]" {
                return Some(("[DONE]".to_string(), Value::Null));
            }
            let parsed: Value = serde_json::from_str(&data).ok()?;
            let name = event.or_else(|| parsed.get("type")?.as_str().map(str::to_string))?;
            Some((name, parsed))
        })
        .collect()
}

/// What a frame is, for the frozen sequence: the event type plus the item
/// type it carries when it announces or closes an output item.
fn frame_label(name: &str, data: &Value) -> String {
    match data.pointer("/item/type").and_then(Value::as_str) {
        Some(item) => format!("{name}({item})"),
        None => name.to_string(),
    }
}

#[expect(clippy::print_stdout, reason = "the sequence is printed for pinning")]
#[tokio::test]
async fn regular_family_streams_one_mcp_call_then_the_answer() {
    let mut mcp = MockMCPServer::start().await.expect("mock MCP server");
    let port = start_scripted_grpc_worker(vec![
        vec![TOOL_CALL_TOKEN],
        vec![FINAL_TOKEN_A, FINAL_TOKEN_B],
    ])
    .await;
    let router = build_router(port).await;

    let req = responses_request(&mcp.url(), true);
    let response = router
        .route_responses(None, &tenant_meta(), req, MODEL)
        .await;
    assert_eq!(response.status(), 200, "streaming responses request");
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let frames = parse_sse(std::str::from_utf8(&body).unwrap());
    let labels: Vec<String> = frames
        .iter()
        .map(|(name, data)| frame_label(name, data))
        .collect();
    println!("regular streaming sequence:\n  {}", labels.join("\n  "));

    // The frozen sequence. A loop change that moves an event must change
    // this list on purpose, in the same PR, with the reason.
    assert_eq!(
        labels,
        REGULAR_STREAMING_SEQUENCE
            .iter()
            .map(|l| (*l).to_string())
            .collect::<Vec<_>>(),
        "regular family streaming sequence"
    );

    // Structure that must hold whatever the exact envelope is.
    assert_eq!(
        labels.iter().filter(|l| *l == "response.completed").count(),
        1,
        "exactly one response.completed"
    );
    assert_eq!(labels.last().map(String::as_str), Some("[DONE]"));
    let call_idx = labels
        .iter()
        .position(|l| l == "response.output_item.added(mcp_call)")
        .expect("the tool call is announced");
    let text_idx = labels
        .iter()
        .position(|l| l == "response.output_text.delta")
        .expect("the answer streams as text");
    assert!(call_idx < text_idx, "tool events precede the answer text");
    let completed = frames
        .iter()
        .find(|(name, _)| name == "response.completed")
        .map(|(_, data)| data.clone())
        .unwrap();
    let outputs: Vec<&str> = completed["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|item| item["type"].as_str())
        .collect();
    assert!(
        outputs.contains(&"mcp_call"),
        "final output carries the call: {outputs:?}"
    );
    assert!(
        outputs.contains(&"message"),
        "final output carries the answer: {outputs:?}"
    );
    let call = completed["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "mcp_call")
        .unwrap();
    assert_eq!(call["name"], "brave_web_search");
    assert!(
        call["output"]
            .as_str()
            .is_some_and(|o| o.contains("Mock search results for: rust")),
        "the tool result is on the call item: {call}"
    );

    mcp.stop().await;
}

#[tokio::test]
async fn regular_family_answers_non_streaming_after_one_mcp_call() {
    let mut mcp = MockMCPServer::start().await.expect("mock MCP server");
    let port = start_scripted_grpc_worker(vec![
        vec![TOOL_CALL_TOKEN],
        vec![FINAL_TOKEN_A, FINAL_TOKEN_B],
    ])
    .await;
    let router = build_router(port).await;

    let req = responses_request(&mcp.url(), false);
    let response = router
        .route_responses(None, &tenant_meta(), req, MODEL)
        .await;
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let value: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(status, 200, "non-streaming responses request: {value}");
    let outputs: Vec<&str> = value["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|item| item["type"].as_str())
        .collect();
    assert_eq!(value["status"], "completed", "{value}");
    assert_eq!(
        outputs,
        ["mcp_list_tools", "mcp_call", "message"],
        "output order: {value}"
    );
    let message_text = value["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "message")
        .and_then(|m| m.pointer("/content/0/text"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    assert_eq!(message_text, "Final answer");

    mcp.stop().await;
}
