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
use smg::config::{RouterConfig, RoutingKeyOverrideConfig};
use tower::ServiceExt;

use super::messages_streaming_test::parse_sse_events;
use crate::common::{
    mock_worker::{
        set_request_recorder, HealthStatus, MockWorkerConfig, RequestRecorder, WorkerType,
    },
    AppTestContext,
};

/// The beta the public API requires on every request of a thread.
const THREADS_BETA: &str = "message-threads-2026-08-12";

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a malformed request or answer fails the test"
)]
async fn post_raw_with(
    app: &Router,
    path: &str,
    payload: &Value,
    beta: Option<&str>,
) -> (StatusCode, axum::body::Bytes) {
    let mut req = Request::builder()
        .method("POST")
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .header("anthropic-version", "2023-06-01");
    if let Some(beta) = beta {
        req = req.header("anthropic-beta", beta);
    }
    let req = req
        .body(Body::from(serde_json::to_string(payload).unwrap()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, body)
}

#[expect(
    clippy::unwrap_used,
    reason = "test helper: a malformed request or answer fails the test"
)]
async fn post_with(
    app: &Router,
    path: &str,
    payload: &Value,
    beta: Option<&str>,
) -> (StatusCode, Value) {
    let (status, body) = post_raw_with(app, path, payload, beta).await;
    (status, serde_json::from_slice(&body).unwrap())
}

/// A request as the public SDK sends a thread turn: with the threads beta.
async fn post(app: &Router, path: &str, payload: &Value) -> (StatusCode, Value) {
    post_with(app, path, payload, Some(THREADS_BETA)).await
}

/// A thread turn's answer as the SDK reads it, the body left as bytes: a
/// stream's events.
async fn post_raw(app: &Router, path: &str, payload: &Value) -> (StatusCode, axum::body::Bytes) {
    post_raw_with(app, path, payload, Some(THREADS_BETA)).await
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

/// A `thread` field without the message-threads beta header is refused with
/// the public API's 400 before the backend; with the header the same turn is
/// served. The header is what tells the gateway to parse the turn.
#[tokio::test]
async fn test_v1_messages_thread_without_the_beta_header_is_refused() {
    let recorder = RequestRecorder::new();
    set_request_recorder(18964, Arc::clone(&recorder));
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 18964,
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
        "messages": [{"role": "user", "content": "first"}],
        "thread": {"type": "create"}
    });

    let (status, refused) = post_with(&app, "/v1/messages", &payload, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert_eq!(refused["type"], "error");
    assert_eq!(refused["error"]["type"], "invalid_request_error");
    let message = refused["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("message-threads-2026-08-12"), "{refused}");
    assert_eq!(recorder.bodies().len(), 0, "the refusal reaches no backend");

    let (status, created) = post(&app, "/v1/messages", &payload).await;
    assert_eq!(status, StatusCode::OK, "{created}");
    assert_eq!(
        recorder.bodies().len(),
        1,
        "the turn with the header is served"
    );

    ctx.shutdown().await;
}

/// Under the routing-key override the turns of one client lineage pin to one
/// worker: four create turns whose `rid`s share a base, with no routing-key
/// header, all reach the worker the first one was placed on, where round
/// robin alone would spread them over both.
#[tokio::test]
async fn test_v1_messages_thread_turns_of_one_lineage_pin_to_one_worker_under_the_override() {
    let recorders = [RequestRecorder::new(), RequestRecorder::new()];
    let ports = [18966, 18967];
    for (port, recorder) in ports.iter().zip(&recorders) {
        set_request_recorder(*port, Arc::clone(recorder));
    }
    let mut config = RouterConfig::builder()
        .regular_mode(vec![])
        .round_robin_policy()
        .routing_key_override(RoutingKeyOverrideConfig {
            enabled: true,
            ..Default::default()
        })
        .host("127.0.0.1")
        .port(3003)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked();
    config.health_check.disable_health_check = true;
    let workers = ports
        .iter()
        .map(|port| MockWorkerConfig {
            port: *port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        })
        .collect();
    let ctx = AppTestContext::new_with_config(config, workers).await;
    let app = ctx.create_app();

    for turn in 1..=4 {
        let (status, answer) = post(
            &app,
            "/v1/messages",
            &json!({
                "model": "mock-model",
                "max_tokens": 16,
                "rid": format!("conv_t{turn}"),
                "messages": [{"role": "user", "content": "hi"}],
                "thread": {"type": "create"}
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{answer}");
    }

    let counts: Vec<usize> = recorders
        .iter()
        .map(|recorder| recorder.bodies().len())
        .collect();
    assert_eq!(counts.iter().sum::<usize>(), 4, "{counts:?}");
    assert!(
        counts.contains(&4),
        "one lineage, one worker; the turns spread as {counts:?}"
    );
    ctx.shutdown().await;
}

/// A streamed create is stored from its events: the continue from the id of
/// its `message_start` reaches the backend with the streamed answer's text.
#[tokio::test]
async fn test_v1_messages_thread_streamed_create_is_continued() {
    let recorder = RequestRecorder::new();
    set_request_recorder(18962, Arc::clone(&recorder));
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 18962,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;
    let app = ctx.create_app();

    let (status, raw) = post_raw(
        &app,
        "/v1/messages",
        &json!({
            "model": "mock-model",
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role": "user", "content": "first"}],
            "thread": {"type": "create"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let events = parse_sse_events(&raw);
    let id = events
        .iter()
        .find(|(name, _)| name == "message_start")
        .and_then(|(_, event)| event["message"]["id"].as_str())
        .expect("a message_start with an id")
        .to_string();
    let streamed_text: String = events
        .iter()
        .filter(|(name, _)| name == "content_block_delta")
        .filter_map(|(_, event)| event["delta"]["text"].as_str())
        .collect();
    assert!(!streamed_text.is_empty(), "{events:?}");

    let (status, answer) = post(
        &app,
        "/v1/messages",
        &json!({
            "model": "mock-model",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "second"}],
            "thread": {"type": "continue", "previous_message_id": id}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{answer}");

    let bodies = recorder.bodies();
    assert_eq!(bodies.len(), 2, "one request per turn reaches the backend");
    let messages = bodies[1]["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 3, "{}", bodies[1]);
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(
        messages[1]["content"][0]["text"], streamed_text,
        "the streamed answer is the second message"
    );
    assert_eq!(messages[2]["content"], "second");
    ctx.shutdown().await;
}
