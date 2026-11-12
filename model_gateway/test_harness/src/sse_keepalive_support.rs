//! The gateway over a held worker, read frame by frame: what an SSE stream
//! carries while its engine is silent.
//!
//! The arrival stamps and the read budget are on tokio's clock, as the
//! keep-alive timer and the held worker's silence are. A test that pauses
//! the clock from the response headers on ([`stream_paused_from_the_headers`])
//! reads the gateway's timers as exact instants; on the wall clock tokio's
//! `Instant` is the system's.
use std::{sync::Arc, time::Duration};

use axum::{
    body::{Body, Bytes},
    http::{header::CONTENT_TYPE, Method, Request, StatusCode},
    response::Response,
    Router,
};
use http_body_util::BodyExt;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use smg::{
    config::{RouterConfig, RoutingMode},
    routers::RouterFactory,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::{net::TcpListener, task::JoinHandle, time::Instant};
use tower::ServiceExt;

use crate::held_worker::HeldWorker;

pub const MODEL: &str = "sse-keepalive-test-model";

/// The gateway's app over one held worker. `keepalive_secs` of `None` leaves
/// the configuration at its default.
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "test helper: failures should panic; the in-process worker is aborted after the test"
)]
pub async fn app(
    first_token_after: Duration,
    keepalive_secs: Option<u64>,
) -> (Router, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = tokio::spawn(
        HeldWorker {
            first_token_after,
            output_tokens: 3,
        }
        .serve(listener),
    );
    let worker = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://127.0.0.1:{port}"))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::TokenSpeed)
            .model(ModelCard::new(MODEL))
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    let mut builder = RouterConfig::builder()
        .mode(RoutingMode::Regular {
            worker_urls: vec![],
        })
        .grpc_connection()
        .random_policy()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024);
    if let Some(secs) = keepalive_secs {
        builder = builder.sse_keepalive_secs(secs);
    }
    let mut config = builder.build_unchecked();
    config.health_check.disable_health_check = true;
    let context = crate::create_test_context_with_mock_tokenizer(config, MODEL).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context)
        .await
        .expect("gRPC router");
    let app = crate::test_app::create_test_app_with_context(Arc::from(router), context);
    (app, server)
}

/// One chunk of the response body and when it arrived, from the response
/// headers, on tokio's clock.
#[derive(Debug)]
pub struct Arrival {
    pub at: Duration,
    pub bytes: Bytes,
}

impl Arrival {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }
}

/// Posts `body` to `path` and reads the response's chunks as they arrive,
/// until the body ends or `budget` has passed since the response headers.
pub async fn stream(
    app: &Router,
    path: &str,
    body: serde_json::Value,
    budget: Duration,
) -> Vec<Arrival> {
    stream_to_end(app, path, body, budget).await.0
}

/// [`stream`], and whether the body closed (EOF) within the budget rather
/// than the budget running out with the body still open.
pub async fn stream_to_end(
    app: &Router,
    path: &str,
    body: serde_json::Value,
    budget: Duration,
) -> (Vec<Arrival>, bool) {
    let response = send(app, path, body).await;
    read(path, response, budget).await
}

/// [`stream`] with tokio's clock paused from the response headers on, for
/// a test that reads the gateway's timers as exact instants. The request,
/// its dial and its hops to the engine run on the wall clock: a paused
/// clock's auto-advance polls I/O once and, with nothing ready, moves on to
/// the next timer, which on a loaded host would be the dial's own timeout.
/// The silence after the headers has only the gateway's timers, which the
/// runtime advances to with no wall time spent. Returns the arrivals and
/// the request's wall duration before the pause: a timer the gateway armed
/// at the headers is due that much earlier than the arrival clock says, at
/// most. Needs a current-thread runtime, as `tokio::time::pause` does.
pub async fn stream_paused_from_the_headers(
    app: &Router,
    path: &str,
    body: serde_json::Value,
    budget: Duration,
) -> (Vec<Arrival>, Duration) {
    let sent = Instant::now();
    let response = send(app, path, body).await;
    tokio::time::pause();
    let slack = sent.elapsed();
    (read(path, response, budget).await.0, slack)
}

#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
async fn send(app: &Router, path: &str, body: serde_json::Value) -> Response {
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.clone().oneshot(request).await.unwrap()
}

/// The response's chunks as they arrive, until the body ends or `budget`
/// has passed since the response headers, and whether the body closed.
async fn read(path: &str, response: Response, budget: Duration) -> (Vec<Arrival>, bool) {
    // The keep-alive interval counts from the response headers, so the
    // arrival clock and the budget start here, not before the request.
    let started = Instant::now();
    let (parts, body) = response.into_parts();
    if parts.status != StatusCode::OK {
        let text = axum::body::to_bytes(body, 1 << 16)
            .await
            .unwrap_or_default();
        assert_eq!(
            parts.status,
            StatusCode::OK,
            "{path}: {}",
            String::from_utf8_lossy(&text)
        );
        return (Vec::new(), false);
    }
    let content_type = parts
        .headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("text/event-stream"),
        "{path}: {content_type}"
    );
    let mut body = body;
    let mut arrivals = Vec::new();
    let mut failure = None;
    let mut eof = false;
    loop {
        let left = budget.saturating_sub(started.elapsed());
        if left.is_zero() {
            break;
        }
        match tokio::time::timeout(left, body.frame()).await {
            Ok(Some(Ok(frame))) => {
                if let Ok(bytes) = frame.into_data() {
                    arrivals.push(Arrival {
                        at: started.elapsed(),
                        bytes,
                    });
                }
            }
            Ok(Some(Err(error))) => {
                failure = Some(error.to_string());
                break;
            }
            Ok(None) => {
                eof = true;
                break;
            }
            Err(_) => break,
        }
    }
    assert!(failure.is_none(), "{path}: the body failed: {failure:?}");
    (arrivals, eof)
}
