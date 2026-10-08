#![expect(
    clippy::disallowed_methods,
    reason = "test-owned mock server tasks are aborted"
)]
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use axum::{
    http::{HeaderMap, StatusCode},
    routing::post,
    Router,
};
use openai_protocol::worker::ProviderType;
use smg_external_router::{
    realtime::{rest::forward_realtime_rest, RealtimeLabels},
    worker::{ExternalWorker, LoadHold},
};

#[derive(Debug)]
struct Worker {
    url: String,
    key: Option<String>,
    client: reqwest::Client,
    attempts: AtomicUsize,
}
impl ExternalWorker for Worker {
    fn url(&self) -> &str {
        &self.url
    }
    fn api_key(&self) -> Option<&String> {
        self.key.as_ref()
    }
    fn model_id(&self) -> &str {
        "registered-model"
    }
    fn is_healthy(&self) -> bool {
        true
    }
    fn provider_for_model(&self, _: &str) -> Option<&ProviderType> {
        None
    }
    fn record_outcome(&self, _: u16) {}
    fn record_request(&self) {
        self.attempts.fetch_add(1, Ordering::Relaxed);
    }
    fn http_client(&self) -> &reqwest::Client {
        &self.client
    }
    fn hold_load(&self, _: Option<&HeaderMap>) -> LoadHold {
        Box::new(())
    }
}
#[expect(
    clippy::expect_used,
    reason = "test fixture client construction must succeed"
)]
fn worker(url: String, key: Option<String>) -> Arc<Worker> {
    Arc::new(Worker {
        url,
        key,
        client: reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .expect("test client"),
        attempts: AtomicUsize::new(0),
    })
}
async fn forward(worker: Arc<Worker>) -> axum::response::Response {
    forward_realtime_rest(
        RealtimeLabels::OPENAI,
        Ok(worker),
        None,
        &serde_json::json!({}),
        "requested-alias",
        "/attempt",
        "sessions",
    )
    .await
}
#[tokio::test]
async fn realtime_attempts_count_success_and_upstream_rejection_once_each() {
    let received = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&received);
    let app = Router::new().route(
        "/attempt",
        post(move || {
            let count = Arc::clone(&count);
            async move {
                count.fetch_add(1, Ordering::Relaxed);
                StatusCode::SERVICE_UNAVAILABLE
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let w = worker(
        format!("http://{}", listener.local_addr().unwrap()),
        Some("test-key".into()),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    for expected in 1..=2 {
        assert_eq!(
            forward(Arc::clone(&w)).await.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(w.attempts.load(Ordering::Relaxed), expected);
        assert_eq!(received.load(Ordering::Relaxed), expected);
    }
    server.abort();
}
#[tokio::test]
async fn validation_and_unpolled_futures_do_not_count_but_transport_failure_does() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let no_auth = worker(url.clone(), None);
    assert_eq!(
        forward(Arc::clone(&no_auth)).await.status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(no_auth.attempts.load(Ordering::Relaxed), 0);
    let w = worker(url, Some("test-key".into()));
    drop(forward(Arc::clone(&w)));
    assert_eq!(w.attempts.load(Ordering::Relaxed), 0);
    assert_eq!(
        forward(Arc::clone(&w)).await.status(),
        StatusCode::BAD_GATEWAY
    );
    assert_eq!(w.attempts.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn every_retried_upstream_attempt_is_counted() {
    use smg_external_router::{retry::RetryExecutor, RetryConfig};
    let received = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&received);
    let app = Router::new().route(
        "/attempt",
        post(move || {
            let count = Arc::clone(&count);
            async move {
                if count.fetch_add(1, Ordering::Relaxed) == 0 {
                    StatusCode::SERVICE_UNAVAILABLE
                } else {
                    StatusCode::OK
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let w = worker(
        format!("http://{}", listener.local_addr().unwrap()),
        Some("test-key".into()),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let response = RetryExecutor::execute_response_with_retry(
        &RetryConfig {
            max_retries: 2,
            initial_backoff_ms: 0,
            max_backoff_ms: 0,
            jitter_factor: 0.0,
            ..Default::default()
        },
        |_| forward(Arc::clone(&w)),
        |response, _| response.status() == StatusCode::SERVICE_UNAVAILABLE,
        |_, _| {},
        || {},
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(received.load(Ordering::Relaxed), 2);
    assert_eq!(w.attempts.load(Ordering::Relaxed), 2);
    server.abort();
}

#[tokio::test]
async fn websocket_handshake_is_one_attempt_without_counting_messages() {
    use smg_external_router::realtime::{registry::RealtimeRegistry, ws::handle_realtime_ws};
    let received = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&received);
    let upstream = Router::new().route(
        "/v1/realtime",
        axum::routing::get(move |ws: axum::extract::WebSocketUpgrade| {
            let count = Arc::clone(&count);
            async move {
                count.fetch_add(1, Ordering::Relaxed);
                ws.on_upgrade(|mut socket| async move {
                    while let Some(Ok(message)) = socket.recv().await {
                        if socket.send(message).await.is_err() {
                            break;
                        }
                    }
                })
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let w = worker(
        format!("http://{}", listener.local_addr().unwrap()),
        Some("test-key".into()),
    );
    let upstream_task = tokio::spawn(async move {
        axum::serve(listener, upstream).await.unwrap();
    });
    let selected = Arc::clone(&w);
    let gateway = Router::new().route(
        "/realtime",
        axum::routing::get(move |request: axum::extract::Request| {
            let selected = Arc::clone(&selected);
            async move {
                let (parts, _) = request.into_parts();
                handle_realtime_ws(
                    RealtimeLabels::OPENAI,
                    parts,
                    "registered-model".into(),
                    Ok(selected),
                    None,
                    Arc::new(RealtimeRegistry::new()),
                )
                .await
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let gateway_task = tokio::spawn(async move {
        axum::serve(listener, gateway).await.unwrap();
    });
    let (mut socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/realtime"))
        .await
        .unwrap();
    use futures_util::{SinkExt, StreamExt};
    for text in ["one", "two"] {
        socket
            .send(tokio_tungstenite::tungstenite::Message::Text(text.into()))
            .await
            .unwrap();
        let received_message =
            tokio::time::timeout(std::time::Duration::from_secs(2), socket.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        assert_eq!(received_message.into_text().unwrap(), text);
    }
    assert_eq!(received.load(Ordering::Relaxed), 1);
    assert_eq!(w.attempts.load(Ordering::Relaxed), 1);
    socket.close(None).await.unwrap();
    gateway_task.abort();
    upstream_task.abort();
}
