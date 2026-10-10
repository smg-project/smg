//! Integration tests for the Anthropic Messages API (`/v1/messages`)
//! against the HTTP backend, which proxies to sglang's native
//! `/v1/messages` endpoint.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};

#[tokio::test]
async fn test_v1_messages_proxy_success() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 18301,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;

    let app = ctx.create_app();

    let payload = json!({
        "model": "mock-model",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Hello, Claude!"}
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(CONTENT_TYPE, "application/json")
        .header("anthropic-version", "2023-06-01")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(body_json["type"], "message");
    assert_eq!(body_json["role"], "assistant");
    assert_eq!(body_json["model"], "mock-model");
    assert_eq!(body_json["stop_reason"], "end_turn");
    let content = body_json["content"].as_array().expect("content array");
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    assert!(body_json["usage"]["input_tokens"].is_number());

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_v1_messages_proxy_streaming() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 18302,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;

    let app = ctx.create_app();

    let payload = json!({
        "model": "mock-model",
        "max_tokens": 64,
        "stream": true,
        "messages": [
            {"role": "user", "content": "Stream me a haiku"}
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(CONTENT_TYPE, "application/json")
        .header("anthropic-version", "2023-06-01")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("text/event-stream"),
        "expected SSE content-type, got {content_type:?}"
    );

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = std::str::from_utf8(&body).expect("utf8");

    // Wire format: `event: <type>\ndata: <json>\n\n`
    let event_types: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("event: "))
        .collect();

    assert_eq!(event_types.first().copied(), Some("message_start"));
    assert_eq!(event_types.last().copied(), Some("message_stop"));
    assert!(event_types.contains(&"content_block_delta"));

    ctx.shutdown().await;
}

#[tokio::test]
async fn test_v1_messages_proxy_propagates_upstream_error() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 18303,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 1.0, // always fail
    }])
    .await;

    let app = ctx.create_app();

    let payload = json!({
        "model": "mock-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "fail please"}]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(CONTENT_TYPE, "application/json")
        .header("anthropic-version", "2023-06-01")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

    ctx.shutdown().await;
}

/// The public Messages API refuses a request without the `anthropic-version`
/// header before anything is routed; so does the gateway, on every backend.
#[tokio::test]
async fn test_v1_messages_requires_the_anthropic_version_header() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 18304,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;

    let app = ctx.create_app();

    let payload = json!({
        "model": "mock-model",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hello!"}]
    });

    for (version, expected) in [
        (None, StatusCode::BAD_REQUEST),
        (Some("1999-01-01"), StatusCode::BAD_REQUEST),
        (Some("2023-06-01"), StatusCode::OK),
    ] {
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(CONTENT_TYPE, "application/json");
        if let Some(version) = version {
            req = req.header("anthropic-version", version);
        }
        let req = req
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), expected, "anthropic-version {version:?}");
    }

    ctx.shutdown().await;
}
