//! A `/generate` request's sampling params through the HTTP router: the
//! worker receives SGLang's sampling keys as the client sent them.

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, Request, StatusCode},
};
use common::{
    mock_worker::{set_request_recorder, RequestRecorder},
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};
use serde_json::json;
use tower::ServiceExt;

use crate::common;

/// The router parses the body into the typed request and writes it back for
/// the worker, so each sampling key SGLang takes reaches the worker only if
/// the type keeps it. A structural tag dropped on the way is decoded
/// unconstrained.
#[tokio::test]
async fn a_generate_request_reaches_the_worker_with_its_sampling_keys() {
    let recorder = RequestRecorder::new();
    set_request_recorder(19321, recorder.clone());
    let ctx = AppTestContext::new_with_config(
        TestRouterConfig::round_robin(0),
        vec![TestWorkerConfig::healthy(19321)],
    )
    .await;
    let app = ctx.create_app();

    let sampling_params = json!({
        "max_new_tokens": 8,
        "structural_tag": r#"{"type":"structural_tag","format":{"type":"const_string","value":"ok"}}"#,
        "stop_regex": "END$",
        "logit_bias": {"42": -100.0},
        "beam_width": 2,
        "spaces_between_special_tokens": false,
        "stream_interval": 2,
        "watermark": {"enabled": true}
    });
    let payload = json!({
        "text": "Answer ok.",
        "sampling_params": sampling_params,
        "stream": false
    });
    let req = Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = recorder.only_body();
    assert_eq!(body["sampling_params"], sampling_params);
    ctx.shutdown().await;
}
