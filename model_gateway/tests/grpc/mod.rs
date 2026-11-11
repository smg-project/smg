//! gRPC router integration tests, and the three helpers the template tests
//! share: the model output cut into tokens, the awaited body of a routed
//! request, and the JSON payloads of an SSE body.

use std::time::Duration;

use axum::{body::to_bytes, http::StatusCode, response::Response};
use serde_json::Value;
use tokio::time::timeout;

mod constraint_failure_test;
mod decisions_grpc_test;
mod grpc_chat_logprobs_test;
mod grpc_completion_logprobs_test;
mod grpc_context_length_test;
mod grpc_context_routing_test;
mod grpc_message_declared_tools_test;
mod grpc_worker_registration_gate_test;
mod grpc_worker_restart_test;
mod kimi_k3_tool_choice_none;
mod media_fault_breaker_test;
mod policy_explain_grpc_test;
mod response_template_test;
mod tenant_rate_limiting_grpc_test;
mod tool_choice_none_renders_the_tools;

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

/// The body of a routed request answered within 30 s with status 200.
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
