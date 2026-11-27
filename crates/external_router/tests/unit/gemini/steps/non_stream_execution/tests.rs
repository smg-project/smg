use std::{sync::Arc, time::Duration};

use axum::http::{header::CONTENT_TYPE, StatusCode};
use openai_protocol::interactions::InteractionsRequest;
use serde_json::{json, Value};

use super::non_stream_request_execution;
use crate::{
    gemini::context::{RequestContext, SharedComponents},
    tenant::{TenantKey, TenantRequestMeta},
    test_support::{
        answering, http_client, text_of, RefusedAddress, StubWorker, StubWorkers,
        CREDENTIALED_WORKER_URL, MASKED_WORKER_URL,
    },
    upstream_body::{ERROR_BODY_TOO_LARGE, MAX_ERROR_RESPONSE_SIZE},
};

/// A request context past worker selection and request building, its
/// worker at `worker_url`.
fn context_for(worker_url: &str) -> RequestContext {
    let components = SharedComponents {
        client: http_client(),
        workers: Arc::new(StubWorkers::default()),
        request_timeout: Duration::from_secs(5),
    };
    let request: InteractionsRequest =
        serde_json::from_value(json!({"model": "m", "input": "hi"})).expect("a request");
    let mut ctx = RequestContext::new(
        Arc::new(request),
        None,
        None,
        TenantRequestMeta::new(TenantKey::new("t")),
        Arc::new(components),
    );
    ctx.processing.worker = Some(StubWorker::at(worker_url));
    ctx.processing.upstream_url = Some(format!("{worker_url}/v1beta/interactions"));
    ctx.processing.payload = Some(json!({}));
    ctx
}

/// The step's lines name the worker masked: a worker registered with
/// credentials at a port nobody listens on fails the send, and the
/// failure line carries the masked address, never the credentials. The
/// capture is the tracing subscriber itself, so the rendering is read as
/// a log reader would.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_failed_send_names_the_worker_masked_in_its_line() {
    let mut ctx = context_for(CREDENTIALED_WORKER_URL);

    let Err(response) = non_stream_request_execution(&mut ctx).await else {
        panic!("a failed send");
    };

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(!logs_contain("secret"));
    assert!(!logs_contain("user:"));
    assert!(logs_contain("Request to worker failed"));
    assert!(logs_contain(&format!(
        "{MASKED_WORKER_URL}/v1beta/interactions"
    )));
}

/// A send nothing answers, to an address registered with a query: the
/// failure line says why without the query. The HTTP client's own error
/// text ends with the request URL, its query kept.
#[tracing_test::traced_test]
#[tokio::test]
async fn a_failed_send_goes_without_the_clients_url_in_its_line() {
    let refused = RefusedAddress::reserve();
    let mut ctx = context_for(&format!("http://{}?key=s3cret", refused.addr()));

    let Err(response) = non_stream_request_execution(&mut ctx).await else {
        panic!("a failed send");
    };

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(logs_contain("Request to worker failed"));
    assert!(!logs_contain("s3cret"));
}

/// An answer that is no JSON, from an address registered with a query:
/// the failure line says why without the query. The HTTP client's text of
/// a failed body read ends with the request URL, its query kept.
#[tracing_test::traced_test]
#[tokio::test]
async fn an_invalid_json_answer_goes_without_the_clients_url_in_its_line() {
    let addr = answering(200, "not json", false).await;
    let mut ctx = context_for(&format!("http://{addr}?key=s3cret"));

    let Err(response) = non_stream_request_execution(&mut ctx).await else {
        panic!("an answer that is no JSON");
    };

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(logs_contain("Upstream returned invalid JSON"));
    assert!(!logs_contain("s3cret"));
}

/// A refusal whose body is lost mid-answer: the line about the unread
/// body names the worker masked and says why without the query.
#[tracing_test::traced_test]
#[tokio::test]
async fn an_unread_refusal_goes_without_the_clients_url_in_its_line() {
    let addr = answering(500, "short", true).await;
    let mut ctx = context_for(&format!("http://{addr}?key=s3cret"));

    let Err(response) = non_stream_request_execution(&mut ctx).await else {
        panic!("a refusal");
    };

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(logs_contain("Failed to read upstream error body"));
    assert!(logs_contain(&format!("url=http://{addr}")));
    assert!(!logs_contain("s3cret"));
}

/// An upstream's error body is passed on within the bound only. At the
/// bound it reaches the caller whole, under the upstream's status and
/// content type. One byte past it, the caller reads the gateway's own
/// envelope under that status and nothing of the body.
#[tokio::test]
async fn an_upstream_error_body_is_passed_on_within_the_bound_only() {
    let at_the_bound: &'static str = "x".repeat(MAX_ERROR_RESPONSE_SIZE).leak();
    let addr = answering(429, at_the_bound, false).await;
    let mut ctx = context_for(&format!("http://{addr}"));

    let Err(response) = non_stream_request_execution(&mut ctx).await else {
        panic!("a refusal");
    };

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json"),
        "under the bound: the upstream's content type"
    );
    let passed_on = text_of(response).await;
    assert!(
        passed_on == at_the_bound,
        "under the bound: {} bytes came back for the {} the upstream sent",
        passed_on.len(),
        at_the_bound.len()
    );

    let past_the_bound: &'static str = "x".repeat(MAX_ERROR_RESPONSE_SIZE + 1).leak();
    let addr = answering(429, past_the_bound, false).await;
    let mut ctx = context_for(&format!("http://{addr}"));

    let Err(response) = non_stream_request_execution(&mut ctx).await else {
        panic!("a refusal");
    };

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let answer = text_of(response).await;
    assert!(
        answer.len() < 1024,
        "past the bound: {} bytes of the upstream's error body were passed on",
        answer.len()
    );
    let answer: Value = serde_json::from_str(&answer).expect("the gateway's envelope");
    assert_eq!(answer["error"]["code"], "upstream_error");
    assert_eq!(answer["error"]["message"], ERROR_BODY_TOO_LARGE);
}
