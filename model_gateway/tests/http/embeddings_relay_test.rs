//! The HTTP relay forwards an embeddings or classify request as the client
//! wrote it: the fields the gateway's types do not model (the engines'
//! `truncate_prompt_tokens`, `add_special_tokens`, `normalize`, `priority`,
//! `extra_body`, ...) reach the upstream beside the typed ones, as they do on
//! the chat and completions routes.
#![expect(clippy::unwrap_used, reason = "test failures should panic")]

use std::sync::Arc;

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, Request, StatusCode},
};
use common::{
    mock_worker::{
        set_request_recorder, HealthStatus, MockWorkerConfig, RequestRecorder, WorkerType,
    },
    AppTestContext,
};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::common;

/// The status and JSON body of `body` posted to `path` through the app.
async fn post(app: &axum::Router, path: &str, body: &Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// The app over one mock worker on `port`, with a recorder of the bodies the
/// worker receives.
async fn app_with_recorder(port: u16) -> (AppTestContext, axum::Router, Arc<RequestRecorder>) {
    let recorder = RequestRecorder::new();
    set_request_recorder(port, recorder.clone());
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;
    let app = ctx.create_app();
    (ctx, app, recorder)
}

/// An embeddings body reaches the upstream whole: the typed fields and the
/// ones the type does not model, and nothing for the fields the client left
/// out.
#[tokio::test]
async fn the_relay_forwards_the_embeddings_fields_the_type_does_not_model() {
    let (ctx, app, recorder) = app_with_recorder(18131).await;
    let sent = json!({
        "model": "mock-model",
        "input": "hello",
        "encoding_format": "float",
        "dimensions": 4,
        "user": "u1",
        "truncate_prompt_tokens": 10,
        "add_special_tokens": false,
        "priority": 5,
        "normalize": false,
        "extra_body": {"k": 1}
    });

    let (status, answer) = post(&app, "/v1/embeddings", &sent).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["data"][0]["index"], 0, "{answer}");
    assert_eq!(recorder.only_body(), sent);

    ctx.shutdown().await;
}

/// A classify body reaches the upstream whole the same way.
#[tokio::test]
async fn the_relay_forwards_the_classify_fields_the_type_does_not_model() {
    let (ctx, app, recorder) = app_with_recorder(18132).await;
    let sent = json!({
        "model": "mock-model",
        "input": ["a", "b"],
        "priority": 3,
        "truncate_prompt_tokens": 5,
        "user": "u"
    });

    let (status, answer) = post(&app, "/v1/classify", &sent).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["data"].as_array().map(Vec::len), Some(2), "{answer}");
    assert_eq!(recorder.only_body(), sent);

    ctx.shutdown().await;
}
