//! A `/v1/audio/transcriptions` form through the HTTP router: the worker's
//! form carries the text fields the gateway's types do not model.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, Request, StatusCode},
};
use common::{
    mock_worker::{set_request_recorder, RequestRecorder},
    AppTestContext, TestWorkerConfig,
};
use tower::ServiceExt;

use crate::common;

/// The router rebuilds the worker's form from the parsed request, so a field
/// reaches the worker only if the parse keeps it: `include[]` (the logprobs a
/// transcription model returns) and `chunking_strategy` ride beside the
/// modeled fields.
#[tokio::test]
async fn a_transcription_form_reaches_the_worker_with_the_fields_the_type_does_not_model() {
    let recorder = RequestRecorder::new();
    set_request_recorder(19322, Arc::clone(&recorder));
    let ctx = AppTestContext::new(vec![TestWorkerConfig::healthy(19322)]).await;
    let app = ctx.create_app();

    let boundary = "form-test-boundary";
    let part = |name: &str, value: &str| {
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
        )
    };
    let form = [
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFmock\r\n"
        ),
        part("model", "mock-model"),
        part("include[]", "logprobs"),
        part("chunking_strategy", "auto"),
        part("language", "en"),
        format!("--{boundary}--\r\n"),
    ]
    .concat();
    let request = Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(form))
        .unwrap();

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let fields = recorder.only_body();
    assert_eq!(fields["include[]"], "logprobs", "{fields}");
    assert_eq!(fields["chunking_strategy"], "auto", "{fields}");
    assert_eq!(fields["language"], "en", "{fields}");
    assert_eq!(fields["model"], "mock-model", "{fields}");

    ctx.shutdown().await;
}
