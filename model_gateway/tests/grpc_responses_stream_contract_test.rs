//! Exercise both regular Responses streaming entry points through a real gRPC
//! worker connection. Responses streams end with a JSON terminal event, rather
//! than the Chat Completions `[DONE]` sentinel.

#[path = "common/mod.rs"]
mod common;

#[path = "common/scripted_tokenizer.rs"]
mod scripted_tokenizer;

use std::{sync::Arc, time::Duration};

use axum::{body::to_bytes, http::StatusCode};
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use serde_json::{json, Value};
use smg::{
    config::{RouterConfig, RoutingMode},
    middleware::TenantRequestMeta,
    routers::RouterFactory,
    tenant::TenantKey,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::{net::TcpListener, time::timeout};

const MODEL: &str = "responses-stream-contract-model";

#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process worker is aborted after the request"
)]
async fn responses_events_with_output(
    tools: Value,
    output: Option<&str>,
    max_tool_calls: Option<u32>,
    expected_status: &str,
) -> Vec<Value> {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let worker_config = Arc::new(mock_worker::config::Config {
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
    });
    let server = tokio::spawn(mock_worker::grpc::serve_with_listener(
        worker_config,
        listener,
    ));
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
    if output.is_some() {
        config.tool_call_parser = Some("qwen".into());
    }
    let tokenizer: Arc<dyn Tokenizer> = match output {
        Some(text) => Arc::new(scripted_tokenizer::ScriptedTokenizer::new(text)),
        None => Arc::new(MockTokenizer::new()),
    };
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
    let request = serde_json::from_value(json!({
        "model": MODEL, "input": "Hello", "stream": true, "store": false,
        "max_output_tokens": 16, "tools": tools, "max_tool_calls": max_tool_calls,
    }))
    .unwrap();
    let (status, content_type, bytes) = timeout(Duration::from_secs(30), async {
        let response = router.route_responses(None, &tenant, request, MODEL).await;
        let status = response.status();
        let content_type = response.headers().get("content-type").cloned();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (status, content_type, bytes)
    })
    .await
    .expect("Responses stream should terminate");
    server.abort();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    assert_eq!(
        content_type
            .unwrap()
            .to_str()
            .unwrap()
            .split(';')
            .next()
            .unwrap(),
        "text/event-stream"
    );
    let body = std::str::from_utf8(&bytes).unwrap();
    let events: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).expect("every Responses SSE payload must be JSON"))
        .collect();
    assert_eq!(events.first().unwrap()["type"], "response.created");
    assert_eq!(
        events.last().unwrap()["type"],
        format!("response.{expected_status}")
    );
    assert_eq!(
        events.last().unwrap()["response"]["status"],
        expected_status
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(
                e["type"].as_str(),
                Some("response.completed" | "response.incomplete" | "response.failed")
            ))
            .count(),
        1
    );
    assert!(events
        .windows(2)
        .all(|pair| pair[0]["sequence_number"].as_u64().unwrap()
            < pair[1]["sequence_number"].as_u64().unwrap()));
    events
}

async fn responses_events(tools: Value) -> Vec<Value> {
    responses_events_with_output(tools, None, None, "completed").await
}

#[tokio::test]
async fn regular_responses_stream_ends_with_json_terminal_event() {
    responses_events(json!([])).await;
}

#[tokio::test]
async fn mcp_responses_stream_ends_with_json_terminal_event() {
    let mut mcp = common::mock_mcp_server::MockMCPServer::start()
        .await
        .unwrap();
    let events = responses_events(json!([{
        "type": "mcp", "server_label": "test-tools", "server_url": mcp.url(),
        "require_approval": "never",
    }]))
    .await;
    assert!(events.iter().any(|e| e["item"]["type"] == "mcp_list_tools"));
    mcp.stop().await;
}

#[tokio::test]
async fn mcp_client_function_return_emits_one_terminal_response() {
    let mut mcp = common::mock_mcp_server::MockMCPServer::start()
        .await
        .unwrap();
    let events = responses_events_with_output(json!([
        {"type":"mcp","server_label":"test-tools","server_url":mcp.url(),"require_approval":"never"},
        {"type":"function","name":"user_tool","parameters":{"type":"object","properties":{}}}
    ]), Some("<tool_call>\n{\"name\":\"user_tool\",\"arguments\":{}}\n</tool_call>"), None, "completed").await;
    mcp.stop().await;
    let calls: Vec<_> = events.last().unwrap()["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "function_call")
        .collect();
    assert_eq!(calls.len(), 1, "a client function is emitted once");
    assert_eq!(calls[0]["name"], "user_tool");
    assert_eq!(calls[0]["status"], "completed");
}

#[tokio::test]
async fn mcp_tool_limit_emits_failed_terminal_response() {
    let mut mcp = common::mock_mcp_server::MockMCPServer::start()
        .await
        .unwrap();
    let events = responses_events_with_output(json!([
        {"type":"mcp","server_label":"test-tools","server_url":mcp.url(),"require_approval":"never"}
    ]), Some("<tool_call>\n{\"name\":\"brave_web_search\",\"arguments\":{\"query\":\"test\"}}\n</tool_call>"), Some(0), "failed").await;
    mcp.stop().await;
    let terminal = events.last().unwrap();
    assert_eq!(
        terminal["response"]["error"]["code"],
        "max_tool_calls_exceeded"
    );
    assert!(terminal["response"]["incomplete_details"].is_null());
}
