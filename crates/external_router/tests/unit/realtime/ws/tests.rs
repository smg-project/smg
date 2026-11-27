use std::sync::Arc;

use axum::{extract::Request, http::HeaderValue, routing::any, Router};
use futures_util::StreamExt;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message as TungsteniteMessage;
use tracing::Instrument;

use super::{build_upstream_ws_url, handle_realtime_ws, RealtimeLabels, RealtimeRegistry};
use crate::test_support::{StubWorker, CREDENTIALED_WORKER_URL, MASKED_WORKER_URL};

#[test]
fn https_becomes_wss() {
    assert_eq!(
        build_upstream_ws_url("https://api.openai.com", "gpt-4o-realtime-preview"),
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    );
}

#[test]
fn http_becomes_ws() {
    assert_eq!(
        build_upstream_ws_url("http://127.0.0.1:8000", "qwen3-asr"),
        "ws://127.0.0.1:8000/v1/realtime?model=qwen3-asr"
    );
}

#[test]
fn trailing_slash_trimmed() {
    assert_eq!(
        build_upstream_ws_url("http://worker:9000/", "m"),
        "ws://worker:9000/v1/realtime?model=m"
    );
}

#[test]
fn model_is_query_encoded() {
    assert_eq!(
        build_upstream_ws_url("https://x", "gpt 4o"),
        "wss://x/v1/realtime?model=gpt+4o"
    );
}

/// The upgrade line names the worker masked, read through a served
/// route: a client upgrades, the proxy's connection to a worker
/// registered with credentials at a port nobody listens on fails, and no
/// line the handler wrote carries the credentials. The handler runs on
/// the server's task, so it borrows the test's span for the capture to
/// read its lines.
#[tracing_test::traced_test]
#[tokio::test]
async fn the_upgrade_line_names_the_worker_masked() {
    let registry = Arc::new(RealtimeRegistry::new());
    let span = tracing::Span::current();
    let app = Router::new().route(
        "/ws",
        any(move |request: Request| {
            let registry = Arc::clone(&registry);
            let span = span.clone();
            async move {
                let (parts, _body) = request.into_parts();
                handle_realtime_ws(
                    RealtimeLabels::OPENAI,
                    parts,
                    "m".to_string(),
                    Ok(StubWorker::at(CREDENTIALED_WORKER_URL)),
                    Some(HeaderValue::from_static("Bearer k")),
                    registry,
                )
                .await
            }
            .instrument(span)
        }),
    );
    let gateway = TcpListener::bind("127.0.0.1:0").await.expect("a port");
    let gateway_url = format!("ws://{}/ws", gateway.local_addr().expect("an address"));
    #[expect(
        clippy::disallowed_methods,
        reason = "the test's gateway ends with the test"
    )]
    let gateway = tokio::spawn(async move {
        axum::serve(gateway, app).await.expect("a served gateway");
    });

    let (mut client, _) = tokio_tungstenite::connect_async(gateway_url)
        .await
        .expect("an upgraded connection");
    // The proxy cannot reach the worker and ends the client's socket.
    while let Some(frame) = client.next().await {
        if matches!(frame, Err(_) | Ok(TungsteniteMessage::Close(_))) {
            break;
        }
    }
    gateway.abort();

    assert!(!logs_contain("secret"));
    assert!(!logs_contain("user:"));
    assert!(logs_contain("Upgrading to realtime WebSocket"));
    assert!(logs_contain(MASKED_WORKER_URL));
}
