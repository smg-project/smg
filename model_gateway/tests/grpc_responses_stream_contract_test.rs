//! Exercise both regular Responses streaming entry points through a real gRPC
//! worker connection. Responses streams end with a JSON terminal event, rather
//! than the Chat Completions `[DONE]` sentinel.

#[path = "common/mod.rs"]
mod common;

#[path = "common/scripted_tokenizer.rs"]
mod scripted_tokenizer;

use std::{error::Error, sync::Arc, time::Duration};

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

async fn responses_result_with_output(
    tools: Value,
    output: Option<&str>,
    max_tool_calls: Option<u32>,
    expected_status: &str,
    stream: bool,
) -> (Value, Vec<Value>) {
    responses_result_with_final_answer(tools, output, max_tool_calls, expected_status, stream, None)
        .await
}

#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process worker is aborted after the request"
)]
async fn responses_result_with_final_answer(
    tools: Value,
    output: Option<&str>,
    max_tool_calls: Option<u32>,
    expected_status: &str,
    stream: bool,
    final_answer_after: Option<usize>,
) -> (Value, Vec<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let output_chunks = output.map(|text| {
        text.split_inclusive("</tool_call>")
            .map(str::to_string)
            .collect::<Vec<_>>()
    });
    let output_tokens = output_chunks.as_ref().map_or(2, |chunks| {
        u32::try_from(chunks.len() + 1).expect("scripted output fits in the worker token count")
    });
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
        output_tokens,
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
    let tokenizer: Arc<dyn Tokenizer> = match output_chunks {
        Some(chunks) => {
            let mut tokenizer = if chunks.len() == 1 {
                scripted_tokenizer::ScriptedTokenizer::new(&chunks.concat())
            } else {
                scripted_tokenizer::ScriptedTokenizer::from_chunks(chunks)
            };
            if let Some(limit) = final_answer_after {
                tokenizer = tokenizer.with_final_answer(limit, "Final answer after tools");
            }
            Arc::new(tokenizer)
        }
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
        "model": MODEL, "input": "Hello", "stream": stream, "store": false,
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
    if !stream {
        let response: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(response["status"], expected_status, "{response}");
        return (response, vec![]);
    }
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
    (events.last().unwrap()["response"].clone(), events)
}

async fn responses_events_with_output(
    tools: Value,
    output: Option<&str>,
    max_tool_calls: Option<u32>,
    expected_status: &str,
) -> Vec<Value> {
    responses_result_with_output(tools, output, max_tool_calls, expected_status, true)
        .await
        .1
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

async fn mcp_repeated_tool_call_events(
    max_tool_calls: Option<u32>,
    expected_status: &str,
) -> Result<Vec<Value>, Box<dyn Error + Send + Sync>> {
    let mut mcp = common::mock_mcp_server::MockMCPServer::start().await?;
    let events = responses_events_with_output(json!([
        {"type":"mcp","server_label":"test-tools","server_url":mcp.url(),"require_approval":"never"}
    ]), Some("<tool_call>\n{\"name\":\"brave_web_search\",\"arguments\":{\"query\":\"test\"}}\n</tool_call>"), max_tool_calls, expected_status).await;
    mcp.stop().await;
    Ok(events)
}

#[tokio::test]
async fn mcp_user_tool_limit_emits_completed_terminal_response() {
    for limit in [0, 1] {
        let events = mcp_repeated_tool_call_events(Some(limit), "completed")
            .await
            .unwrap();
        let terminal = events.last().unwrap();
        assert!(terminal["response"]["error"].is_null());
        assert!(terminal["response"]["incomplete_details"].is_null());
        let executed_calls = events
            .iter()
            .filter(|event| {
                event["type"] == "response.output_item.done" && event["item"]["type"] == "mcp_call"
            })
            .count();
        assert_eq!(
            executed_calls, limit as usize,
            "ignore calls exceeding the user cap"
        );
    }
}

#[tokio::test]
async fn mcp_call_safety_limit_emits_failed_terminal_response() {
    let events = mcp_repeated_tool_call_events(None, "failed").await.unwrap();
    let terminal = events.last().unwrap();
    assert_eq!(
        terminal["response"]["error"]["code"],
        "max_tool_calls_exceeded"
    );
    assert_eq!(
        terminal["response"]["output"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item["type"] == "mcp_call")
            .count(),
        10
    );
}

/// Exercise the same generated MCP batch through both regular Responses paths.
async fn mcp_cap_response(
    stream: bool,
    cap: Option<u32>,
    batch_size: usize,
    expected_status: &str,
) -> Result<Value, Box<dyn Error + Send + Sync>> {
    let mut mcp = common::mock_mcp_server::MockMCPServer::start().await?;
    let output: String = (0..batch_size)
        .map(|index| {
            let call =
                json!({"name":"brave_web_search","arguments":{"query":format!("query-{index}")}});
            format!("<tool_call>\n{call}\n</tool_call>\n")
        })
        .collect();
    let (response, events) = responses_result_with_output(
        json!([{"type":"mcp","server_label":"test-tools","server_url":mcp.url(),"require_approval":"never"}]),
        Some(&output), cap, expected_status, stream,
    ).await;
    mcp.stop().await;
    assert!(
        !events.iter().any(|event| {
            event["item"]["type"] == "function_call"
                || event["type"]
                    .as_str()
                    .is_some_and(|kind| kind.starts_with("response.function_call_arguments."))
        }),
        "server-executed calls must not be streamed as client functions: {events:?}"
    );
    assert!(
        !response["output"]
            .as_array()
            .ok_or("Responses output must be an array")?
            .iter()
            .any(|item| item["type"] == "function_call"),
        "{response}"
    );
    Ok(response)
}

/// Check retained MCP results, including which calls from each batch ran.
fn assert_executed_mcp_queries(response: &Value, expected: &[&str]) {
    assert!(response["output"].is_array(), "{response}");
    let calls: Vec<_> = response["output"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|item| item["type"] == "mcp_call")
        .collect();
    assert_eq!(calls.len(), expected.len(), "{response}");
    for (call, expected_query) in calls.into_iter().zip(expected) {
        assert_eq!(call["name"], "brave_web_search");
        assert_eq!(call["status"], "completed");
        assert!(
            call["arguments"].as_str().is_some_and(|args| {
                serde_json::from_str::<Value>(args)
                    .is_ok_and(|args| args["query"] == *expected_query)
            }),
            "expected {expected_query}: {response}"
        );
        assert!(
            call["output"].as_str().is_some_and(
                |result| result.contains(&format!("Mock search results for: {expected_query}"))
            ),
            "expected {expected_query}: {response}"
        );
    }
}

#[tokio::test]
async fn mcp_cap_zero_completes_in_both_response_modes() {
    for stream in [false, true] {
        let response = mcp_cap_response(stream, Some(0), 1, "completed")
            .await
            .unwrap();
        assert!(response["error"].is_null());
        assert!(response["incomplete_details"].is_null());
        assert_executed_mcp_queries(&response, &[]);
    }
}

#[tokio::test]
async fn mcp_cap_preserves_previous_results_in_both_response_modes() {
    for stream in [false, true] {
        let response = mcp_cap_response(stream, Some(1), 1, "completed")
            .await
            .unwrap();
        assert!(response["error"].is_null());
        assert_executed_mcp_queries(&response, &["query-0"]);
    }
}

#[tokio::test]
async fn mcp_cap_executes_allowed_prefix_of_parallel_batch() {
    for stream in [true, false] {
        let response = mcp_cap_response(stream, Some(2), 3, "completed")
            .await
            .unwrap();
        assert!(response["error"].is_null());
        assert_executed_mcp_queries(&response, &["query-0", "query-1"]);
    }
}

#[tokio::test]
async fn mcp_cap_applies_remaining_budget_to_later_batches() {
    for stream in [false, true] {
        let response = mcp_cap_response(stream, Some(3), 2, "completed")
            .await
            .unwrap();
        assert!(response["error"].is_null());
        assert_executed_mcp_queries(&response, &["query-0", "query-1", "query-0"]);
    }
}

#[tokio::test]
async fn mcp_cap_at_internal_limit_still_completes_normally() {
    for stream in [true, false] {
        let response = mcp_cap_response(stream, Some(10), 1, "completed")
            .await
            .unwrap();
        assert!(response["error"].is_null());
        assert_executed_mcp_queries(&response, &["query-0"; 10]);
    }
}

#[tokio::test]
async fn mcp_cap_cannot_disable_internal_safety_limit() {
    for stream in [false, true] {
        for cap in [None, Some(20)] {
            for batch_size in [1, 12] {
                let response = mcp_cap_response(stream, cap, batch_size, "failed")
                    .await
                    .unwrap();
                assert_eq!(response["error"]["code"], "max_tool_calls_exceeded");
                let queries: Vec<_> = (0..10)
                    .map(|index| format!("query-{}", index % batch_size))
                    .collect();
                let expected: Vec<_> = queries.iter().map(String::as_str).collect();
                assert_executed_mcp_queries(&response, &expected);
            }
        }
    }
}

#[tokio::test]
async fn mcp_final_answer_survives_ten_executed_calls() {
    for stream in [false, true] {
        for cap in [None, Some(10), Some(20)] {
            for batch_size in [1, 2] {
                let mut mcp = common::mock_mcp_server::MockMCPServer::start()
                    .await
                    .unwrap();
                let output = (0..batch_size).map(|index| format!(
                    "<tool_call>\n{}\n</tool_call>\n",
                    json!({"name":"brave_web_search","arguments":{"query":format!("query-{index}")}})
                )).collect::<String>();
                let (response, _) = responses_result_with_final_answer(
                    json!([{"type":"mcp","server_label":"test-tools","server_url":mcp.url(),"require_approval":"never"}]),
                    Some(&output), cap, "completed", stream, Some(10),
                ).await;
                mcp.stop().await;
                assert!(
                    response["output"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|item| item["type"] == "message"
                            && item["content"][0]["text"] == "Final answer after tools"),
                    "stream={stream}, cap={cap:?}, batch={batch_size}: {response}"
                );
                let queries: Vec<_> = (0..10)
                    .map(|i| format!("query-{}", i % batch_size))
                    .collect();
                assert_executed_mcp_queries(
                    &response,
                    &queries.iter().map(String::as_str).collect::<Vec<_>>(),
                );
            }
        }
    }
}

#[tokio::test]
async fn mcp_usage_includes_every_model_call_in_both_modes() {
    for stream in [false, true] {
        let response = mcp_cap_response(stream, Some(1), 1, "completed")
            .await
            .unwrap();
        // Two model requests: one executed call, followed by an ignored call.
        assert_eq!(response["usage"]["input_tokens"], 2, "{response}");
        assert_eq!(response["usage"]["output_tokens"], 6, "{response}");
        assert_eq!(response["usage"]["total_tokens"], 8, "{response}");
    }
}
