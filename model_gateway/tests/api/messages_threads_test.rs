//! Message threads on `/v1/messages` through the whole app: the create turn
//! kept under its id, the continue turn rebuilt before the backend sees it,
//! and the refusals in the Messages envelope with their codes.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    Router,
};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{
        set_request_recorder, HealthStatus, MockWorkerConfig, RequestRecorder, WorkerType,
    },
    AppTestContext,
};

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a malformed request or answer fails the test"
)]
async fn post(app: &Router, path: &str, payload: &Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .header("anthropic-version", "2023-06-01")
        .body(Body::from(serde_json::to_string(payload).unwrap()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

/// A continue turn reaches the backend as the whole conversation: the create
/// turn's `system` and `tools`, its messages, the stored answer and the new
/// messages, with no `thread` field; a continue from an unknown id and one
/// with `system`, and a count with `thread`, are refused before the backend.
#[tokio::test]
async fn test_v1_messages_thread_continue_rebuilds_the_conversation_for_the_backend() {
    let recorder = RequestRecorder::new();
    set_request_recorder(18306, Arc::clone(&recorder));
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 18306,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;
    let app = ctx.create_app();

    let (status, created) = post(
        &app,
        "/v1/messages",
        &json!({
            "model": "mock-model",
            "max_tokens": 64,
            "system": "be brief",
            "tools": [{"name": "f", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": "first"}],
            "thread": {"type": "create"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{created}");
    let id = created["id"].as_str().expect("the answer's id").to_string();

    let continued = |previous: &str, extra: Value| {
        let mut payload = json!({
            "model": "mock-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "second"}],
            "thread": {"type": "continue", "previous_message_id": previous}
        });
        for (field, value) in extra.as_object().unwrap() {
            payload[field] = value.clone();
        }
        payload
    };

    let (status, answer) = post(&app, "/v1/messages", &continued(&id, json!({}))).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_ne!(answer["id"], created["id"]);

    let bodies = recorder.bodies();
    assert_eq!(bodies.len(), 2, "one request per turn reaches the backend");
    let sent = &bodies[1];
    assert_eq!(sent["system"], "be brief");
    assert_eq!(sent["tools"].as_array().map(Vec::len), Some(1));
    assert!(sent.get("thread").is_none(), "{sent}");
    let messages = sent["messages"].as_array().expect("messages");
    let roles: Vec<&str> = messages
        .iter()
        .map(|message| message["role"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user"], "{sent}");
    assert_eq!(messages[0]["content"], "first");
    assert_eq!(messages[1]["content"][0]["type"], "text");
    assert_eq!(
        messages[1]["content"][0]["text"], created["content"][0]["text"],
        "the stored answer's text is the second message"
    );
    assert_eq!(messages[2]["content"], "second");

    let (status, refused) = post(&app, "/v1/messages", &continued("msg_unknown", json!({}))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{refused}");
    assert_eq!(refused["type"], "error");
    assert_eq!(refused["error"]["type"], "not_found_error");
    assert_eq!(
        refused["error"]["details"]["error_code"],
        "thread_not_found"
    );
    assert!(refused["request_id"].is_string(), "{refused}");

    let (status, refused) = post(
        &app,
        "/v1/messages",
        &continued(&id, json!({"system": "be brief"})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert_eq!(refused["error"]["type"], "invalid_request_error");
    let text = refused["error"]["message"].as_str().unwrap_or_default();
    assert!(
        text.contains("thread: a continue request sends no system"),
        "{text}"
    );

    let (status, refused) = post(
        &app,
        "/v1/messages/count_tokens",
        &json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "first"}],
            "thread": {"type": "create"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    let text = refused["error"]["message"].as_str().unwrap_or_default();
    assert!(
        text.contains("thread: count_tokens takes no thread"),
        "{text}"
    );

    assert_eq!(recorder.bodies().len(), 2, "the refusals reach no backend");
    ctx.shutdown().await;
}
