//! The gateway over a held worker, read frame by frame: what an SSE stream
//! carries while its engine is silent.
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    body::{Body, Bytes},
    http::{header::CONTENT_TYPE, Method, Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use llm_tokenizer::{traits::Tokenizer, MockTokenizer, TokenizerRegistry};
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use smg::{
    config::{RouterConfig, RoutingMode},
    routers::RouterFactory,
    worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType},
};
use tokio::{net::TcpListener, task::JoinHandle};
use tower::ServiceExt;

use crate::{common, held_worker::HeldWorker};

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
    let tokenizers = Arc::new(TokenizerRegistry::new());
    let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
    tokenizers
        .load(
            "tokenizer-id",
            MODEL,
            "test",
            || async move { Ok(tokenizer) },
        )
        .await
        .unwrap();
    let context = common::create_test_context_with_tokenizer_registry(config, tokenizers).await;
    context.worker_registry.register(worker).unwrap();
    let router = RouterFactory::create_router(&context)
        .await
        .expect("gRPC router");
    let app = common::test_app::create_test_app_with_context(Arc::from(router), context);
    (app, server)
}

/// One chunk of the response body and when it arrived, from the request.
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
#[expect(clippy::unwrap_used, reason = "test helper: failures should panic")]
pub async fn stream_to_end(
    app: &Router,
    path: &str,
    body: serde_json::Value,
    budget: Duration,
) -> (Vec<Arrival>, bool) {
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
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
