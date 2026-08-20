//! Client-disconnect propagation for the HTTP router's streaming relays.
//!
//! When a streaming client drops the response body, the relay must drop the
//! upstream reqwest stream promptly — closing the worker connection so the
//! engine aborts generation — instead of only noticing on the next upstream
//! chunk, which may never arrive during a long prefill or a stalled upstream.

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
use openai_protocol::{
    chat::ChatCompletionRequest,
    transcription::{AudioFile, TranscriptionRequest},
};
use serde_json::json;
use smg::{
    routers::{router::Router as HttpRouter, RouterTrait},
    tenant::{RouteRequestMeta, TenantKey},
    worker::{BasicWorkerBuilder, ModelCard, Worker},
};
use tokio::{net::TcpListener, sync::oneshot, time::timeout};

use crate::common::test_app::create_test_app_context;

/// Sends on drop, so the test observes the moment the upstream response body
/// is dropped — i.e. the gateway actually closed the worker connection.
struct DropSignal(Option<oneshot::Sender<()>>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        if let Some(tx) = self.0.take() {
            let _ = tx.send(());
        }
    }
}

/// Upstream that answers with SSE headers, optionally one chunk, then stalls
/// forever. Returns the base URL and a receiver that completes when the
/// stalled response body is dropped.
#[expect(
    clippy::disallowed_methods,
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
async fn spawn_stalling_upstream(send_first_chunk: bool) -> (String, oneshot::Receiver<()>) {
    let (dropped_tx, dropped_rx) = oneshot::channel();
    let signal = Arc::new(Mutex::new(Some(DropSignal(Some(dropped_tx)))));

    let handler = move || {
        let signal = Arc::clone(&signal);
        async move {
            let signal = signal.lock().unwrap().take();
            let head =
                send_first_chunk.then(|| Ok::<Event, Infallible>(Event::default().data("head")));
            // `map` owns `signal`, so the DropSignal fires exactly when this
            // body stream is dropped by the server end of the connection.
            let body = stream::iter(head)
                .chain(stream::pending::<Result<Event, Infallible>>())
                .map(move |item| {
                    let _held = &signal;
                    item
                });
            Sse::new(body)
        }
    };

    let app = AxumRouter::new()
        .route("/v1/chat/completions", post(handler.clone()))
        .route("/v1/audio/transcriptions", post(handler));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (format!("http://{addr}"), dropped_rx)
}

/// Build an HTTP router whose only worker is the given upstream.
#[expect(
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
async fn http_router_for(upstream_url: &str) -> HttpRouter {
    let ctx = create_test_app_context().await;
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(upstream_url)
            .models(vec![ModelCard::new("mock-model")])
            .health_config(openai_protocol::worker::HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    ctx.worker_registry.register(worker);
    HttpRouter::new(&ctx).await.unwrap()
}

fn tenant_meta() -> smg::middleware::TenantRequestMeta {
    RouteRequestMeta::new(TenantKey::from("test-tenant"))
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

/// Client disconnects while the upstream has produced no chunk yet (the
/// prefill window). The relay must not wait for a first token that may be
/// tens of seconds away — or never come.
#[tokio::test]
async fn chat_relay_closes_upstream_when_client_disconnects_during_prefill() {
    let (upstream_url, dropped_rx) = spawn_stalling_upstream(false).await;
    let router = http_router_for(&upstream_url).await;

    let response = router
        .route_chat(None, &tenant_meta(), streaming_chat_request(), "mock-model")
        .await;
    assert_eq!(response.status(), 200);

    // Client disconnect: the response body (and its relay channel receiver)
    // is dropped before any chunk arrived.
    drop(response);

    timeout(Duration::from_secs(5), dropped_rx)
        .await
        .expect("upstream response body was not dropped: relay did not notice client disconnect")
        .expect("drop signal sender vanished without firing");
}

/// Client disconnects mid-stream while the upstream is stalled between
/// chunks. The relay is parked in `stream.next()` and must still notice.
#[tokio::test]
async fn chat_relay_closes_upstream_when_client_disconnects_mid_stream() {
    let (upstream_url, mut dropped_rx) = spawn_stalling_upstream(true).await;
    let router = http_router_for(&upstream_url).await;

    let response = router
        .route_chat(None, &tenant_meta(), streaming_chat_request(), "mock-model")
        .await;
    assert_eq!(response.status(), 200);

    let mut body = response.into_body();
    let first = body.frame().await;
    assert!(
        matches!(first, Some(Ok(_))),
        "expected a first streamed chunk, got {first:?}"
    );

    // Upstream must stay attached while the client is still reading.
    assert!(
        dropped_rx.try_recv().is_err(),
        "upstream body dropped while the client was still connected"
    );

    drop(body);

    timeout(Duration::from_secs(5), dropped_rx)
        .await
        .expect("upstream response body was not dropped: relay did not notice client disconnect")
        .expect("drop signal sender vanished without firing");
}

/// Same contract for the transcription relay, which additionally carries
/// stream-outcome attribution: a client disconnect must still drop the
/// upstream stream promptly.
#[tokio::test]
async fn transcription_relay_closes_upstream_when_client_disconnects() {
    let (upstream_url, mut dropped_rx) = spawn_stalling_upstream(true).await;
    let router = http_router_for(&upstream_url).await;

    let request = TranscriptionRequest {
        model: "mock-model".to_string(),
        stream: Some(true),
        ..Default::default()
    };
    let audio = AudioFile {
        bytes: bytes::Bytes::from_static(b"not real audio"),
        file_name: "test.wav".to_string(),
        content_type: Some("audio/wav".to_string()),
    };

    let response = router
        .route_audio_transcriptions(None, &tenant_meta(), &request, audio, "mock-model")
        .await;
    assert_eq!(response.status(), 200);

    let mut body = response.into_body();
    let first = body.frame().await;
    assert!(
        matches!(first, Some(Ok(_))),
        "expected a first streamed chunk, got {first:?}"
    );
    assert!(
        dropped_rx.try_recv().is_err(),
        "upstream body dropped while the client was still connected"
    );

    drop(body);

    timeout(Duration::from_secs(5), dropped_rx)
        .await
        .expect("upstream response body was not dropped: relay did not notice client disconnect")
        .expect("drop signal sender vanished without firing");
}
