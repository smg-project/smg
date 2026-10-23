//! The transcription route's buffered answer is a worker body like any other:
//! read under the ingress payload cap, and ended by the request's own timeout
//! the way the typed routes end it.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, StatusCode},
    routing::post,
    Router as AxumRouter,
};
use bytes::Bytes;
use futures_util::{stream, StreamExt};
use http_body_util::BodyExt;
use openai_protocol::{
    transcription::{AudioFile, TranscriptionRequest},
    worker::HealthCheckConfig,
};
use smg::{
    routers::{router::Router as HttpRouter, RouterTrait},
    tenant::{RouteRequestMeta, TenantKey},
    worker::{
        circuit_breaker::CircuitState, BasicWorkerBuilder, CircuitBreakerConfig, ModelCard, Worker,
    },
};
use smg_external_router::{error::extract_error_code_from_response, retry::is_retryable_response};
use tokio::{net::TcpListener, sync::oneshot, time::timeout};

use crate::common::{test_app::create_test_app_context_with_config, TestRouterConfig};

const ROUTE: &str = "/v1/audio/transcriptions";
const MIB: usize = 1 << 20;
/// One write of the oversized worker.
const CHUNK: usize = 64 * 1024;
/// The answering worker's body, three writes.
const ANSWER: [&str; 3] = [
    "{\"text\":\"the whole",
    " transcription\",",
    "\"language\":\"en\"}",
];

/// Sends on drop: the moment the worker's response body is dropped, which is
/// when the gateway closed the worker connection.
struct DropSignal(Option<oneshot::Sender<()>>);

impl Drop for DropSignal {
    fn drop(&mut self) {
        if let Some(tx) = self.0.take() {
            let _ = tx.send(());
        }
    }
}

/// Serve `app` on a loopback port for the rest of the test process.
#[expect(
    clippy::disallowed_methods,
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
async fn serve(app: AxumRouter) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// A transcription worker whose answer is `total` bytes under
/// `application/json`, written in [`CHUNK`]s one millisecond apart, so a
/// reader that stops is noticed within a write or two. Returns the base URL,
/// the bytes handed to the connection so far, and a receiver that fires when
/// the worker's body is dropped.
#[expect(
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
async fn spawn_oversized_worker(total: usize) -> (String, Arc<AtomicUsize>, oneshot::Receiver<()>) {
    let (dropped_tx, dropped_rx) = oneshot::channel();
    let signal = Arc::new(Mutex::new(Some(DropSignal(Some(dropped_tx)))));
    let pushed = Arc::new(AtomicUsize::new(0));
    let pushed_by_worker = Arc::clone(&pushed);
    let handler = move |_: Bytes| {
        let signal = Arc::clone(&signal);
        let pushed = Arc::clone(&pushed_by_worker);
        async move {
            let signal = signal.lock().unwrap().take();
            let chunk = Bytes::from(vec![b'a'; CHUNK]);
            let body = stream::iter(0..total / CHUNK)
                .then(move |_| {
                    let pushed = Arc::clone(&pushed);
                    let chunk = chunk.clone();
                    async move {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                        pushed.fetch_add(chunk.len(), Ordering::SeqCst);
                        Ok::<_, std::io::Error>(chunk)
                    }
                })
                // `map` owns `signal`, so the DropSignal fires exactly when
                // this body stream is dropped by the server end.
                .map(move |item| {
                    let _held = &signal;
                    item
                });
            (
                [(CONTENT_TYPE, "application/json")],
                Body::from_stream(body),
            )
        }
    };
    let url = serve(AxumRouter::new().route(ROUTE, post(handler))).await;
    (url, pushed, dropped_rx)
}

/// A transcription worker whose head and first bytes leave at once and whose
/// body then never ends, as an engine that stalls mid-answer does.
async fn spawn_stalled_worker() -> String {
    let handler = |_: Bytes| async {
        let head = stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(
            b"{\"text\":\"a partial",
        ))]);
        (
            [(CONTENT_TYPE, "application/json")],
            Body::from_stream(head.chain(stream::pending())),
        )
    };
    serve(AxumRouter::new().route(ROUTE, post(handler))).await
}

/// A transcription worker that answers [`ANSWER`], one write per piece.
async fn spawn_answering_worker() -> String {
    let handler = |_: Bytes| async {
        let body = stream::iter(ANSWER).then(|piece| async move {
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok::<_, std::io::Error>(Bytes::from_static(piece.as_bytes()))
        });
        (
            [(CONTENT_TYPE, "application/json")],
            Body::from_stream(body),
        )
    };
    serve(AxumRouter::new().route(ROUTE, post(handler))).await
}

/// An HTTP router over the one worker at `upstream_url` with the payload cap
/// `cap`; the worker's breaker opens on one failure, and the worker sends
/// through `client` when the case needs a budget of its own. The config's
/// port is never bound: the router is driven directly.
#[expect(
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
async fn router_with_cap(
    upstream_url: &str,
    cap: usize,
    client: Option<Arc<reqwest::Client>>,
) -> (HttpRouter, Arc<dyn Worker>) {
    let ctx =
        create_test_app_context_with_config(&TestRouterConfig::with_payload_limit(3000, cap)).await;
    let mut builder = BasicWorkerBuilder::new(upstream_url)
        .models(vec![ModelCard::new("mock-model")])
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .circuit_breaker_config(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 1,
            timeout_duration: Duration::from_secs(60),
            window_duration: Duration::from_secs(60),
        });
    if let Some(client) = client {
        builder = builder.http_client(client);
    }
    let worker: Arc<dyn Worker> = Arc::new(builder.build());
    ctx.worker_registry.register(Arc::clone(&worker));
    (HttpRouter::new(&ctx).await.unwrap(), worker)
}

fn tenant_meta() -> smg::middleware::TenantRequestMeta {
    RouteRequestMeta::new(TenantKey::from("test-tenant"))
}

/// A buffered (non-streaming) transcription of a few bytes of audio.
async fn transcribe(router: &HttpRouter) -> axum::response::Response {
    let request = TranscriptionRequest {
        model: "mock-model".to_string(),
        ..Default::default()
    };
    let audio = AudioFile {
        bytes: Bytes::from_static(b"not real audio"),
        file_name: "test.wav".to_string(),
        content_type: Some("audio/wav".to_string()),
    };
    router
        .route_audio_transcriptions(None, &tenant_meta(), &request, audio, "mock-model")
        .await
}

/// A worker that answers a transcription with more than the payload cap is a
/// misbehaving worker, as on every typed route: a 502
/// `upstream_response_too_large` at the cap, none of the body kept, the
/// worker connection closed, the worker's breaker charged, and the answer
/// naming the routed worker.
#[tokio::test]
async fn an_answer_above_the_cap_is_a_502_and_the_body_is_not_kept() {
    let cap = MIB;
    let total = 64 * MIB;
    let (upstream_url, pushed, dropped_rx) = spawn_oversized_worker(total).await;
    let (router, worker) = router_with_cap(&upstream_url, cap, None).await;

    let response = transcribe(&router).await;

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        extract_error_code_from_response(&response),
        "upstream_response_too_large"
    );
    assert_eq!(
        response
            .headers()
            .get("x-smg-routed-worker-id")
            .expect("routed worker id on the error answer"),
        worker.url()
    );
    timeout(Duration::from_secs(5), dropped_rx)
        .await
        .expect("the worker's body was not dropped: the gateway kept reading past the cap")
        .expect("drop signal sender vanished without firing");
    // The read stopped at the cap: what the worker had handed to the
    // connection when the gateway closed it is the cap plus the writes in
    // flight, nowhere near the body.
    let pushed = pushed.load(Ordering::SeqCst);
    assert!(
        pushed <= cap + 4 * MIB,
        "the worker had written {pushed} bytes when the gateway closed the connection; the read did not stop at the cap of {cap}"
    );
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    assert!(
        body.len() < 1024,
        "the gateway's answer carries {} bytes: part of the worker's body",
        body.len()
    );
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
}

/// The request's own timeout running out while the transcription body is
/// read is the request's end, as before the head and as on the typed routes:
/// a 504 `call_upstream_timeout`, terminal for the retry layer, the worker
/// charged and named on the answer.
#[tokio::test]
async fn a_request_timeout_while_the_body_is_read_ends_the_request_as_a_504() {
    let upstream_url = spawn_stalled_worker().await;
    let client = Arc::new(
        smg::clients::http_client_builder()
            .no_proxy()
            .timeout(Duration::from_millis(200))
            .build()
            .expect("test client"),
    );
    let (router, worker) = router_with_cap(&upstream_url, MIB, Some(client)).await;

    let response = transcribe(&router).await;

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        extract_error_code_from_response(&response),
        "call_upstream_timeout"
    );
    assert!(!is_retryable_response(&response));
    assert_eq!(
        response
            .headers()
            .get("x-smg-routed-worker-id")
            .expect("routed worker id on the error answer"),
        worker.url()
    );
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Open);
}

/// An answer that fits the cap, here exactly, is relayed as it was: status,
/// bytes, content type and the routed worker's id.
#[tokio::test]
async fn an_answer_within_the_cap_is_relayed_byte_for_byte() {
    let upstream_url = spawn_answering_worker().await;
    let expected = ANSWER.concat();
    let (router, worker) = router_with_cap(&upstream_url, expected.len(), None).await;

    let response = transcribe(&router).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(CONTENT_TYPE).expect("content type"),
        "application/json"
    );
    assert_eq!(
        response
            .headers()
            .get("x-smg-routed-worker-id")
            .expect("routed worker id"),
        worker.url()
    );
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    assert_eq!(body.as_ref(), expected.as_bytes());
    assert_eq!(worker.circuit_breaker_state(), CircuitState::Closed);
}
