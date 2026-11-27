use axum::{
    body::Body,
    http::{
        header::{ACCEPT_ENCODING, CONTENT_TYPE},
        Request, StatusCode,
    },
    routing::post,
    Router,
};
use openai_protocol::worker::{WorkerErrorResponse, WorkerSpec};
use tower::ServiceExt;

use super::*;

fn app() -> Router {
    Router::new()
        .route(
            "/workers",
            post(|AdminJson(_): AdminJson<WorkerSpec>| async { "ok" }),
        )
        .route(
            "/optional",
            post(|body: Option<AdminJson<WorkerSpec>>| async move {
                if body.is_some() {
                    "some"
                } else {
                    "none"
                }
            }),
        )
}

async fn send(body: &'static str, content_type: Option<&'static str>) -> (StatusCode, String) {
    send_to("/workers", body, content_type).await
}

async fn send_to(
    path: &'static str,
    body: &'static str,
    content_type: Option<&'static str>,
) -> (StatusCode, String) {
    let mut request = Request::builder().method("POST").uri(path);
    if let Some(content_type) = content_type {
        request = request.header(CONTENT_TYPE, content_type);
    }
    let response = app()
        .oneshot(request.body(Body::from(body)).expect("request"))
        .await
        .expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// The envelope of a refused body: `code` and `error`, the error naming
/// what was wrong.
fn envelope(body: &str, text: &str) -> WorkerErrorResponse {
    serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("{body}: not the control plane's envelope ({e}): {text}"))
}

/// The status, the `Accept-Encoding` header and the body of an empty
/// JSON object posted under the `Content-Encoding` fields `encodings`.
async fn send_encoded(
    path: &'static str,
    encodings: &[&'static str],
    content_type: Option<&'static str>,
) -> (StatusCode, Option<String>, String) {
    let mut request = Request::builder().method("POST").uri(path);
    for encoding in encodings {
        request = request.header("content-encoding", *encoding);
    }
    if let Some(content_type) = content_type {
        request = request.header(CONTENT_TYPE, content_type);
    }
    let response = app()
        .oneshot(request.body(Body::from("{}")).expect("request"))
        .await
        .expect("response");
    let status = response.status();
    let accept = response
        .headers()
        .get(ACCEPT_ENCODING)
        .map(|value| String::from_utf8_lossy(value.as_bytes()).into_owned());
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, accept, String::from_utf8_lossy(&bytes).into_owned())
}

/// A body under a `Content-Encoding` the gateway does not decode is a
/// 415 in the control plane's envelope with `Accept-Encoding: identity`,
/// on the plain and the optional form alike and in whichever field of a
/// repeated header the encoding stands, before the body is read;
/// `identity` is no encoding.
#[tokio::test]
async fn a_content_encoding_the_gateway_does_not_decode_is_a_415_on_both_forms() {
    for (path, encodings) in [
        ("/workers", &["gzip"][..]),
        ("/optional", &["gzip"][..]),
        ("/workers", &["identity", "gzip"][..]),
    ] {
        let (status, accept, text) = send_encoded(path, encodings, Some("application/json")).await;
        assert_eq!(
            status,
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "{path} {encodings:?}: {text}"
        );
        assert_eq!(accept.as_deref(), Some("identity"), "{path} {encodings:?}");
        let refusal = envelope(path, &text);
        assert_eq!(refusal.code, "UNSUPPORTED_MEDIA_TYPE", "{path}");
        assert!(refusal.error.contains("gzip"), "{path}: {}", refusal.error);
    }
    let (status, accept, text) = send_encoded("/optional", &["identity"], None).await;
    assert_eq!(
        (status, accept, text.as_str()),
        (StatusCode::OK, None, "none")
    );
}

/// The optional form: no `Content-Type` is no body and the handler runs
/// without one; a body the type takes reaches it; a body the type does
/// not take, or one under another content type, is the same 400 in the
/// control plane's envelope as the plain form answers.
#[tokio::test]
async fn the_optional_form_reads_no_body_without_a_content_type_and_refuses_a_bad_one() {
    let (status, text) = send_to("/optional", "", None).await;
    assert_eq!((status, text.as_str()), (StatusCode::OK, "none"));
    let (status, text) = send_to(
        "/optional",
        r#"{"url":"grpc://w:1"}"#,
        Some("application/json"),
    )
    .await;
    assert_eq!((status, text.as_str()), (StatusCode::OK, "some"));
    for (body, content_type, detail) in [
        (r"[1, 2]", Some("application/json"), "invalid type"),
        (r"{not json", Some("application/json"), "syntax"),
        (
            r#"{"url":"grpc://w:1"}"#,
            Some("text/plain"),
            "Content-Type",
        ),
    ] {
        let (status, text) = send_to("/optional", body, content_type).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {text}");
        let refusal = envelope(body, &text);
        assert_eq!(refusal.code, "BAD_REQUEST", "{body}");
        assert!(refusal.error.contains(detail), "{body}: {}", refusal.error);
    }
}

/// A body the type does not take (an unknown variant, a wrong type, a
/// negative or out-of-range integer, an array, broken JSON) is a 400 in
/// the control plane's envelope with the deserializer's reason; a body
/// the type takes reaches the handler.
#[tokio::test]
async fn a_body_the_type_does_not_take_is_a_400_in_the_control_plane_envelope() {
    for (body, detail) in [
        (
            r#"{"url":"grpc://w:1","worker_type":"bogus"}"#,
            "unknown variant `bogus`",
        ),
        (r#"{"url":"grpc://w:1","cost":"abc"}"#, "cost"),
        (r#"{"url":"grpc://w:1","priority":-1}"#, "priority"),
        (
            r#"{"url":"grpc://w:1","bootstrap_port":70000}"#,
            "bootstrap_port",
        ),
        (r"[1, 2]", "invalid type"),
        (r"{not json", "syntax"),
    ] {
        let (status, text) = send(body, Some("application/json")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {text}");
        let refusal = envelope(body, &text);
        assert_eq!(refusal.code, "BAD_REQUEST", "{body}");
        assert!(refusal.error.contains(detail), "{body}: {}", refusal.error);
    }
    let (status, text) = send(
        r#"{"url":"grpc://w:1","worker_type":"regular"}"#,
        Some("application/json"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{text}");
}

/// `refusal` renders a handler's own refusal in the same envelope.
#[tokio::test]
async fn a_handler_refusal_carries_the_status_the_code_and_the_reason() {
    let response = refusal(
        StatusCode::NOT_FOUND,
        "WORKER_NOT_FOUND",
        "no such worker".to_string(),
    );
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let refusal = envelope("refusal", &String::from_utf8_lossy(&bytes));
    assert_eq!(refusal.code, "WORKER_NOT_FOUND");
    assert_eq!(refusal.error, "no such worker");
}

/// A body without the JSON content type is refused in the same envelope.
#[tokio::test]
async fn a_body_without_the_json_content_type_is_a_400_in_the_envelope() {
    let body = r#"{"url":"grpc://w:1"}"#;
    let (status, text) = send(body, None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    let refusal = envelope(body, &text);
    assert_eq!(refusal.code, "BAD_REQUEST");
    assert!(refusal.error.contains("Content-Type"), "{}", refusal.error);
}
