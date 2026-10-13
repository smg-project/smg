//! `--request-timeout-secs` is the deadline of every call to a gRPC worker: a
//! worker that opens its stream and never speaks is timed out at the
//! configured timeout, on a streamed request (the error event typed as the
//! HTTP path's 504, then the stream's end) and on a non-streaming one (the
//! 504 itself, with the HTTP path's code). Both cases are capped: a gateway
//! without the deadline waits on the silent worker and fails its cap.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::allow_attributes)]

#[path = "common/mod.rs"]
mod common;

#[path = "common/held_worker.rs"]
mod held_worker;

#[path = "common/sse_keepalive_support.rs"]
#[allow(
    dead_code,
    reason = "the keepalive tests read the stream with the harness's reader; this test reads the body itself"
)]
mod support;

use std::time::{Duration, Instant};

use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Method, Request, StatusCode},
    Router,
};
use serde_json::{json, Value};
use support::MODEL;
use tokio::time::timeout;
use tower::ServiceExt;

/// The gateway's request timeout under test, the scheduling slack of a loaded
/// machine, and the cap a gateway without the deadline fails at.
const TIMEOUT_SECS: u64 = 1;
const TIMEOUT: Duration = Duration::from_secs(TIMEOUT_SECS);
const SLACK: Duration = Duration::from_secs(6);
const CASE_CAP: Duration = Duration::from_secs(20);
/// The HTTP path's code for its upstream client's timeout, and the deadline's words.
const TIMEOUT_CODE: &str = "call_upstream_timeout";
const TIMEOUT_MESSAGE: &str = "request timed out after 1s";

/// The gateway's app with the request timeout under test, over one worker
/// that opens its stream and says nothing for an hour; the gRPC router built
/// the single-router way, or the multi-router (IGW) way.
async fn app(igw: bool) -> Router {
    support::app(Duration::from_secs(3600), None, Some(TIMEOUT_SECS), igw)
        .await
        .0
}

/// One chat completion through the app, read to its end under the case cap:
/// the status, the body and how long it took.
async fn chat(app: &Router, stream: bool) -> (StatusCode, String, Duration) {
    let body = json!({
        "model": MODEL, "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 4, "stream": stream,
    });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("the request");
    let started = Instant::now();
    let response = timeout(CASE_CAP, app.clone().oneshot(request))
        .await
        .expect("answered within the case cap: without the deadline the gateway waits on the silent worker as long as the client does")
        .expect("a response");
    let status = response.status();
    let bytes = timeout(CASE_CAP, to_bytes(response.into_body(), 1 << 20))
        .await
        .expect("the body ends within the case cap")
        .expect("the body");
    (
        status,
        String::from_utf8_lossy(&bytes).into_owned(),
        started.elapsed(),
    )
}

fn assert_at_the_deadline(elapsed: Duration, body: &str) {
    assert!(
        elapsed >= TIMEOUT && elapsed < TIMEOUT + SLACK,
        "answered after {elapsed:?}, the deadline is {TIMEOUT:?}: {body}"
    );
    assert!(body.contains(TIMEOUT_MESSAGE), "{body}");
}

#[tokio::test]
async fn a_streamed_request_to_a_silent_worker_ends_at_the_deadline_with_the_504s_event() {
    let (status, body, elapsed) = chat(&app(false).await, true).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_at_the_deadline(elapsed, &body);
    let events: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect();
    let error = events.last().cloned().unwrap_or_default();
    assert_eq!(error["error"]["type"], "server_error", "{body}");
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
}

#[tokio::test]
async fn a_non_streaming_request_to_a_silent_worker_is_the_504_at_the_deadline() {
    let (status, body, elapsed) = chat(&app(false).await, false).await;
    assert_timeout_response(status, &body, elapsed);
}

/// The router built the multi-router (IGW) way issues its requests under
/// the same deadline: the configuration path every gRPC router passes through
/// sets it.
#[tokio::test]
async fn a_router_built_for_igw_times_out_the_silent_worker_at_the_deadline_too() {
    let (status, body, elapsed) = chat(&app(true).await, false).await;
    assert_timeout_response(status, &body, elapsed);
}

/// The HTTP path's 504 for its client timeout, at the deadline.
fn assert_timeout_response(status: StatusCode, body: &str, elapsed: Duration) {
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert_at_the_deadline(elapsed, body);
    let error: Value = serde_json::from_str(body).expect("a JSON body");
    assert_eq!(error["error"]["code"], TIMEOUT_CODE, "{body}");
    assert_eq!(error["error"]["type"], "server_error", "{body}");
}
