//! Policies learn that a request ended through the worker's load guard, on
//! the HTTP router's streaming path too: a client that drops the response
//! mid-stream must release the policy's booking for that worker.
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    response::sse::{Event, Sse},
    routing::post,
    Router as AxumRouter,
};
use futures_util::{stream, StreamExt};
use http_body_util::BodyExt;
use openai_protocol::chat::ChatCompletionRequest;
use serde_json::json;
use smg::{
    routers::{router::Router as HttpRouter, RouterTrait},
    tenant::{RouteRequestMeta, TenantKey},
    worker::{BasicWorkerBuilder, ModelCard, RequestCompletionSink, Worker},
};
use tokio::{net::TcpListener, time::timeout};

use crate::common::test_app::create_test_app_context;

/// Records the worker URLs that reported a completion.
#[derive(Debug, Default)]
struct CompletionSpy(Mutex<Vec<String>>);

impl RequestCompletionSink for CompletionSpy {
    fn request_completed(&self, worker: &dyn Worker) {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(worker.url().to_string());
    }
}

/// Upstream that answers with SSE headers, one chunk, then stalls forever.
#[expect(
    clippy::disallowed_methods,
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
async fn spawn_stalling_upstream() -> String {
    let handler = || async {
        let body = stream::iter([Ok::<Event, Infallible>(Event::default().data("head"))])
            .chain(stream::pending::<Result<Event, Infallible>>());
        Sse::new(body)
    };
    let app = AxumRouter::new().route("/v1/chat/completions", post(handler));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

#[expect(
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
async fn router_with_spy(upstream_url: &str) -> (HttpRouter, Arc<CompletionSpy>) {
    let ctx = create_test_app_context().await;
    let spy = Arc::new(CompletionSpy::default());
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(upstream_url)
            .models(vec![ModelCard::new("mock-model")])
            .health_config(openai_protocol::worker::HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    worker.set_completion_sink(Some(Arc::clone(&spy) as Arc<dyn RequestCompletionSink>));
    ctx.worker_registry.register(worker);
    (HttpRouter::new(&ctx).await.unwrap(), spy)
}

#[expect(
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
fn streaming_chat_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true
    }))
    .unwrap()
}

/// A stream the client abandons after the first chunk still reports exactly
/// one completion for the worker that served it, once the relay lets go.
#[tokio::test]
async fn http_stream_cancelled_mid_way_reports_one_completion() {
    let upstream_url = spawn_stalling_upstream().await;
    let (router, spy) = router_with_spy(&upstream_url).await;
    let meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let response = router
        .route_chat(None, &meta, streaming_chat_request(), "mock-model")
        .await;
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let first = body.frame().await;
    assert!(
        matches!(first, Some(Ok(_))),
        "expected a first chunk, got {first:?}"
    );
    assert!(
        spy.0.lock().unwrap().is_empty(),
        "no completion while the client is still reading"
    );
    drop(body);
    timeout(Duration::from_secs(5), async {
        loop {
            if spy.0.lock().unwrap().len() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the cancelled stream never reported its completion");
    assert_eq!(spy.0.lock().unwrap().as_slice(), [upstream_url.as_str()]);
}
