#![allow(clippy::unwrap_used)]
//! A JSON body under a `Content-Encoding` the gateway does not decode is a
//! 415 in the route's envelope before the body is parsed: the public one on
//! the data plane, the control plane's on the management routes.

use axum::{
    body::Body,
    http::{
        header::{ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_TYPE},
        Request, StatusCode,
    },
};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};

/// The status, the `Accept-Encoding` header and the body of a POST of `body`
/// to `uri` under the `Content-Encoding` fields `encodings`.
async fn post(
    app: &axum::Router,
    uri: &str,
    encodings: &[&str],
    body: &'static str,
) -> (StatusCode, Option<String>, serde_json::Value) {
    let mut req = Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json");
    for encoding in encodings {
        req = req.header(CONTENT_ENCODING, *encoding);
    }
    let resp = app
        .clone()
        .oneshot(req.body(Body::from(body)).unwrap())
        .await
        .unwrap();
    let status = resp.status();
    let accept = resp
        .headers()
        .get(ACCEPT_ENCODING)
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned());
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::json!({"not_json": String::from_utf8_lossy(&bytes)}));
    (status, accept, body)
}

/// A compressed body is a 415 naming the encoding, with `Accept-Encoding:
/// identity`, in the public envelope on the chat route (`json_parse_error`)
/// and in the control plane's on the worker route (`UNSUPPORTED_MEDIA_TYPE`),
/// in whichever field of a repeated header the encoding stands; `identity`
/// is no encoding and the body is read as before.
#[tokio::test]
async fn a_compressed_body_is_a_415_in_the_envelope_on_both_planes() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 18134,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;
    let app = ctx.create_app();
    let chat = r#"{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}"#;

    for encodings in [&["gzip"][..], &["identity", "gzip"][..]] {
        let (status, accept, body) = post(&app, "/v1/chat/completions", encodings, chat).await;
        assert_eq!(
            status,
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "{encodings:?}: {body}"
        );
        assert_eq!(accept.as_deref(), Some("identity"), "{encodings:?}");
        assert_eq!(
            body["error"]["code"], "json_parse_error",
            "{encodings:?}: {body}"
        );
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("gzip"),
            "{encodings:?}: {body}"
        );
    }

    let (status, accept, body) =
        post(&app, "/workers", &["br"], r#"{"url":"http://127.0.0.1:1"}"#).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE, "{body}");
    assert_eq!(accept.as_deref(), Some("identity"), "{body}");
    assert_eq!(body["code"], "UNSUPPORTED_MEDIA_TYPE", "{body}");
    assert!(
        body["error"].as_str().unwrap_or("").contains("br"),
        "{body}"
    );

    let (status, accept, body) = post(&app, "/v1/chat/completions", &["identity"], chat).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(accept, None, "{body}");
    assert!(body.get("choices").is_some(), "{body}");
    ctx.shutdown().await;
}
