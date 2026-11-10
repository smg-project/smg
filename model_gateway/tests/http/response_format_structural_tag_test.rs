//! A `structural_tag` response format through the HTTP router: the gateway
//! accepts it and the worker receives the object as the client sent it.

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

/// The request holds the sections of the answer it names to a schema and
/// leaves the text around them free; the gateway judges nothing in the tag
/// (the engine's grammar compiler does), so the worker's body carries it key
/// for key, in the order the client wrote them, `type` first.
#[tokio::test]
async fn a_structural_tag_response_format_reaches_the_worker_as_sent() {
    let recorder = RequestRecorder::new();
    set_request_recorder(19320, recorder.clone());
    let ctx = AppTestContext::new_with_config(
        TestRouterConfig::round_robin(0),
        vec![TestWorkerConfig::healthy(19320)],
    )
    .await;
    let app = ctx.create_app();

    let payload = json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Report the weather for Berlin as <w>{json}</w>."}],
        "response_format": {
            "type": "structural_tag",
            "structures": [{
                "begin": "<w>",
                "schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "temp_c": {"type": "integer"}},
                    "required": ["city", "temp_c"],
                    "additionalProperties": false
                },
                "end": "</w>"
            }],
            "triggers": ["<w>"]
        },
        "stream": false
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = recorder.only_body();
    assert_eq!(body["response_format"], payload["response_format"]);
    // the keys in the client's order: the text the worker reads is the client's
    assert_eq!(
        body["response_format"].to_string(),
        payload["response_format"].to_string()
    );
    ctx.shutdown().await;
}
