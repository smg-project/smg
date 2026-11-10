//! The tokenizer management routes refuse in the control plane's envelope
//! (`{"error", "code"}`, the shape the worker routes answer): an unknown
//! tokenizer is a 404, a name in use a 409, a body the type does not take a
//! 400. A loaded tokenizer keeps its answer shape.

use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    Router,
};
use openai_protocol::{tokenize::TokenizerInfo, worker::WorkerErrorResponse};
use smg::routers::RouterFactory;
use tower::ServiceExt;

use crate::common::{
    create_test_context_with_tokenizer_registry, mock_tokenizer_registry,
    test_app::create_test_app_with_context_and_auth, TestRouterConfig,
};

/// The name the registry holds a tokenizer under (its id is `tokenizer-id`).
const LOADED: &str = "loaded-model";

/// The production route table over a context whose registry holds one
/// tokenizer, with no control-plane authentication.
#[expect(clippy::expect_used)]
async fn app_with_a_loaded_tokenizer() -> Router {
    let config = TestRouterConfig::round_robin(4357);
    let registry = mock_tokenizer_registry(LOADED).await;
    let context = create_test_context_with_tokenizer_registry(config, registry).await;
    let router = RouterFactory::create_router(&context)
        .await
        .expect("the router builds");
    create_test_app_with_context_and_auth(Arc::from(router), context, None)
}

#[expect(clippy::expect_used)]
async fn send(app: &Router, method: &str, path: &str, body: Option<&str>) -> (StatusCode, String) {
    let request = Request::builder()
        .method(method)
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
        .expect("a well-formed request");
    let response = app.clone().oneshot(request).await.expect("an answer");
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// The control plane's envelope, or the panic names what the route answered.
#[expect(
    clippy::panic,
    reason = "test helper: a body that is not the envelope is the finding, named with the body"
)]
fn envelope(text: &str) -> WorkerErrorResponse {
    serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("not the control plane's envelope ({e}): {text}"))
}

#[tokio::test]
async fn an_unknown_tokenizer_is_a_404_in_the_envelope_on_every_management_read() {
    let app = app_with_a_loaded_tokenizer().await;
    for (method, path) in [
        ("GET", "/v1/tokenizers/no-such-tokenizer"),
        ("GET", "/v1/tokenizers/no-such-tokenizer/status"),
        ("DELETE", "/v1/tokenizers/no-such-tokenizer"),
    ] {
        let (status, text) = send(&app, method, path, None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{method} {path}: {text}");
        let refusal = envelope(&text);
        assert_eq!(refusal.code, "TOKENIZER_NOT_FOUND", "{method} {path}");
        assert!(
            refusal.error.contains("no-such-tokenizer"),
            "{method} {path}: {}",
            refusal.error
        );
    }
}

#[tokio::test]
async fn a_name_in_use_is_a_409_in_the_envelope_naming_the_tokenizer_that_holds_it() {
    let app = app_with_a_loaded_tokenizer().await;
    let body = format!(r#"{{"name":"{LOADED}","source":"test"}}"#);
    let (status, text) = send(&app, "POST", "/v1/tokenizers", Some(&body)).await;
    assert_eq!(status, StatusCode::CONFLICT, "{text}");
    let refusal = envelope(&text);
    assert_eq!(refusal.code, "TOKENIZER_ALREADY_EXISTS");
    assert!(
        refusal.error.contains(LOADED) && refusal.error.contains("tokenizer-id"),
        "{}",
        refusal.error
    );
}

#[tokio::test]
async fn a_body_the_type_does_not_take_is_a_400_in_the_envelope() {
    let app = app_with_a_loaded_tokenizer().await;
    let (status, text) = send(&app, "POST", "/v1/tokenizers", Some(r#"{"name":"x"}"#)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    let refusal = envelope(&text);
    assert_eq!(refusal.code, "BAD_REQUEST");
    assert!(refusal.error.contains("source"), "{}", refusal.error);
}

#[tokio::test]
async fn a_loaded_tokenizer_keeps_its_answer_shape() {
    let app = app_with_a_loaded_tokenizer().await;
    let (status, text) = send(&app, "GET", &format!("/v1/tokenizers/{LOADED}"), None).await;
    assert_eq!(status, StatusCode::OK, "{text}");
    let info: TokenizerInfo =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("not a TokenizerInfo ({e}): {text}"));
    assert_eq!(info.name, LOADED);
    assert_eq!(info.id, "tokenizer-id");
}
