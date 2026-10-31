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

/// A request without `anthropic-version` is served (this gateway's own SDK
/// clients send none); a malformed version is refused before anything is
/// routed, on every backend that takes the buffered path.
#[tokio::test]
async fn test_v1_messages_takes_a_missing_anthropic_version_and_refuses_a_malformed_one() {
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
        (None, StatusCode::OK),
        (Some("1999-01-01"), StatusCode::BAD_REQUEST),
        (Some("2023-02-30"), StatusCode::BAD_REQUEST),
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

/// A `top_p` of 0.0 or a `top_k` of 0 is refused with a 400 in the Messages
/// error envelope that names the bound, before anything is routed: the
/// engines' wires cannot tell either zero from a field the client left out.
/// A request inside the bounds (`top_p: 1.0, top_k: 1`) is served.
#[tokio::test]
async fn test_v1_messages_refuses_a_nucleus_of_nothing_and_a_top_k_of_zero_before_routing() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 18305,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;

    let app = ctx.create_app();

    for (sampling, expected, message) in [
        (
            json!({"top_p": 0.0}),
            StatusCode::BAD_REQUEST,
            "top_p: range: (0, 1]",
        ),
        (
            json!({"top_k": 0}),
            StatusCode::BAD_REQUEST,
            "top_k: range: 1..2147483647",
        ),
        (json!({"top_p": 1.0, "top_k": 1}), StatusCode::OK, ""),
    ] {
        let mut payload = json!({
            "model": "mock-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hello!"}]
        });
        for (field, value) in sampling.as_object().unwrap() {
            payload[field] = value.clone();
        }
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header(CONTENT_TYPE, "application/json")
            .header("anthropic-version", "2023-06-01")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), expected, "{sampling}");
        if expected == StatusCode::BAD_REQUEST {
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body_json["type"], "error", "{sampling}");
            assert_eq!(
                body_json["error"]["type"], "invalid_request_error",
                "{sampling}"
            );
            let text = body_json["error"]["message"].as_str().unwrap_or_default();
            assert!(text.contains(message), "{sampling}: {text}");
        }
    }

    ctx.shutdown().await;
}
