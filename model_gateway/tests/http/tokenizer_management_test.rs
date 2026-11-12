//! The tokenizer management routes refuse in the control plane's envelope
//! (`{"error", "code"}`, the shape the worker routes answer): an unknown
//! tokenizer is a 404, a name in use a 409, a body the type does not take a
//! 400, an empty or whitespace name or source a 400 up front. A loaded
//! tokenizer keeps its answer shape; a model alias tokenizes as the model it
//! names; a load that fails stays readable under its id as `failed`, with the
//! error.

use std::{sync::Arc, time::Duration};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    Router,
};
use openai_protocol::{
    model_card::ModelCard,
    tokenize::{AddTokenizerResponse, TokenizeResponse, TokenizerInfo},
    worker::{HealthCheckConfig, WorkerErrorResponse},
};
use smg::{
    app_context::AppContext,
    routers::RouterFactory,
    worker::{BasicWorkerBuilder, Worker, WorkerType},
};
use tower::ServiceExt;

use crate::common::{
    create_test_context_with_tokenizer_registry, mock_tokenizer_registry,
    test_app::create_test_app_with_context_and_auth, TestRouterConfig,
};

/// The name the registry holds a tokenizer under (its id is `tokenizer-id`).
const LOADED: &str = "loaded-model";

/// The production route table over a context whose registry holds one
/// tokenizer, with no control-plane authentication.
async fn app_with_a_loaded_tokenizer() -> Router {
    app_and_context_with_a_loaded_tokenizer().await.0
}

/// [`app_with_a_loaded_tokenizer`] with the context beside it, for a test
/// that registers a worker.
#[expect(clippy::expect_used)]
async fn app_and_context_with_a_loaded_tokenizer() -> (Router, Arc<AppContext>) {
    let config = TestRouterConfig::round_robin(4357);
    let registry = mock_tokenizer_registry(LOADED).await;
    let context = create_test_context_with_tokenizer_registry(config, registry).await;
    let router = RouterFactory::create_router(&context)
        .await
        .expect("the router builds");
    let app = create_test_app_with_context_and_auth(Arc::from(router), Arc::clone(&context), None);
    (app, context)
}

/// A worker that serves the loaded model under an alias, in the registry the
/// serving routes resolve aliases through; no health check runs for it.
fn serve_the_loaded_model_under(context: &AppContext, alias: &str) {
    let mut card = ModelCard::new(LOADED);
    card.aliases = vec![alias.to_owned()];
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://tokenize-alias.invalid:8000")
            .worker_type(WorkerType::Regular)
            .models(vec![card])
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    context.worker_registry.register(worker);
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

/// `POST /v1/tokenize` and `/v1/detokenize` by an alias answer as by the
/// model the alias names, the way the chat route and the model card read it;
/// a name no worker carries stays a 400 `tokenizer_not_found`.
#[tokio::test]
async fn a_model_alias_tokenizes_and_detokenizes_as_the_model_it_names() {
    let (app, context) = app_and_context_with_a_loaded_tokenizer().await;
    serve_the_loaded_model_under(&context, "loaded-alias");
    let tokenize = |model: &str| format!(r#"{{"model":"{model}","prompt":"hi"}}"#);
    let detokenize = |model: &str| format!(r#"{{"model":"{model}","tokens":[1,2,3]}}"#);
    let (status, by_model) = send(&app, "POST", "/v1/tokenize", Some(&tokenize(LOADED))).await;
    assert_eq!(status, StatusCode::OK, "tokenize by the model: {by_model}");
    let _: TokenizeResponse = serde_json::from_str(&by_model)
        .unwrap_or_else(|e| panic!("not a TokenizeResponse ({e}): {by_model}"));
    let (status, by_alias) = send(
        &app,
        "POST",
        "/v1/tokenize",
        Some(&tokenize("loaded-alias")),
    )
    .await;
    assert_eq!(
        (status, by_alias.as_str()),
        (StatusCode::OK, by_model.as_str()),
        "tokenize by the alias answers as by the model"
    );
    let (status, by_model) = send(&app, "POST", "/v1/detokenize", Some(&detokenize(LOADED))).await;
    assert_ne!(
        status,
        StatusCode::BAD_REQUEST,
        "detokenize by the model: {by_model}"
    );
    let (status_by_alias, by_alias) = send(
        &app,
        "POST",
        "/v1/detokenize",
        Some(&detokenize("loaded-alias")),
    )
    .await;
    assert_eq!(
        (status_by_alias, by_alias.as_str()),
        (status, by_model.as_str()),
        "detokenize by the alias answers as by the model"
    );
    let (status, text) = send(
        &app,
        "POST",
        "/v1/tokenize",
        Some(r#"{"model":"no-such-alias","prompt":"hi"}"#),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("tokenizer_not_found"), "{text}");
}

/// An empty or whitespace name is refused before anything is queued: the
/// empty name is the tokenize route's "any tokenizer" fallback.
#[tokio::test]
async fn an_empty_tokenizer_name_is_a_400_in_the_envelope() {
    let app = app_with_a_loaded_tokenizer().await;
    for name in ["", "   "] {
        let body = format!(r#"{{"name":"{name}","source":"test"}}"#);
        let (status, text) = send(&app, "POST", "/v1/tokenizers", Some(&body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "name {name:?}: {text}");
        let refusal = envelope(&text);
        assert_eq!(refusal.code, "BAD_REQUEST", "name {name:?}");
        assert!(
            refusal.error.contains("name"),
            "name {name:?}: {}",
            refusal.error
        );
    }
}

/// An empty or whitespace source is refused up front in the envelope, as the
/// name is: the registry's load would refuse an empty one only once the job
/// runs, after a 202, and would take a whitespace one as a source.
#[tokio::test]
async fn an_empty_tokenizer_source_is_a_400_in_the_envelope() {
    let app = app_with_a_loaded_tokenizer().await;
    for source in ["", "   "] {
        let body = format!(r#"{{"name":"no-source","source":"{source}"}}"#);
        let (status, text) = send(&app, "POST", "/v1/tokenizers", Some(&body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "source {source:?}: {text}");
        let refusal = envelope(&text);
        assert_eq!(refusal.code, "BAD_REQUEST", "source {source:?}");
        assert!(
            refusal.error.contains("source"),
            "source {source:?}: {}",
            refusal.error
        );
    }
}

/// A load the gateway accepted (202) and could not complete stays readable
/// under its id: the status route answers `failed` with the error, instead of
/// a 404 that reads as if the job had never existed.
#[tokio::test]
async fn a_failed_load_stays_readable_under_its_id_as_failed() {
    let app = app_with_a_loaded_tokenizer().await;
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("a temp dir: {e}"));
    let source = dir.path().join("tokenizer.json");
    std::fs::write(&source, "not a tokenizer").unwrap_or_else(|e| panic!("write {source:?}: {e}"));
    let body = format!(
        r#"{{"name":"never-loads","source":"{}"}}"#,
        source.to_str().unwrap_or_else(|| panic!("a UTF-8 path"))
    );
    let (status, text) = send(&app, "POST", "/v1/tokenizers", Some(&body)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{text}");
    let accepted: AddTokenizerResponse = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("not an AddTokenizerResponse ({e}): {text}"));
    let path = format!("/v1/tokenizers/{}/status", accepted.id);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
    loop {
        let (status, text) = send(&app, "GET", &path, None).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "the status of a load the gateway accepted must stay readable: {text}"
        );
        let answer: AddTokenizerResponse = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("not an AddTokenizerResponse ({e}): {text}"));
        if !matches!(answer.status.as_str(), "pending" | "processing") {
            assert_eq!(answer.status, "failed", "{text}");
            assert!(
                !answer.message.is_empty() && answer.message != "Tokenizer job is failed",
                "the error of the load is the message: {text}"
            );
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the load of a file that is not a tokenizer did not end in 90 s: {text}"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}
