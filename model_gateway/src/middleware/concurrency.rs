//! Per-request concurrency limiting via a token bucket, with optional
//! queuing for backpressure.
//!
//! `AdmissionQueue` caps concurrent waiters at `queue_size`; a request
//! that finds every slot taken sheds immediately. Waiters park directly
//! on the token bucket, which serves them in FIFO order. `TokenGuardBody`
//! wraps the response body so the token is only released after the entire
//! stream has been delivered.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header::RETRY_AFTER, HeaderValue, StatusCode},
    middleware::Next,
    response::Response,
};
use bytes::Bytes;
use http_body::Frame;
use tokio::{sync::Semaphore, time::error::Elapsed};
use tracing::{debug, warn};

use super::{token_bucket::TokenBucket, SHED_RETRY_AFTER_SECS};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    routers::error::create_error,
    server::AppState,
};

/// Standard error body plus `Retry-After` for admission sheds.
fn shed_response(status: StatusCode, code: &'static str, message: &'static str) -> Response {
    let mut response = create_error(status, code, message);
    response
        .headers_mut()
        .insert(RETRY_AFTER, HeaderValue::from(SHED_RETRY_AFTER_SECS));
    response
}

/// Returns an acquired token when the request is cancelled or the response body is dropped.
struct TokenPermit {
    token_bucket: Arc<TokenBucket>,
    /// Number of tokens to return.
    tokens: f64,
}

impl TokenPermit {
    fn try_acquire(token_bucket: Arc<TokenBucket>, tokens: f64) -> Result<Self, ()> {
        token_bucket.try_acquire(tokens)?;
        Metrics::record_admission_inflight_acquired();
        Ok(Self {
            token_bucket,
            tokens,
        })
    }

    async fn acquire_timeout(
        token_bucket: Arc<TokenBucket>,
        tokens: f64,
        timeout: Duration,
    ) -> Result<Self, Elapsed> {
        token_bucket.acquire_timeout(tokens, timeout).await?;
        Metrics::record_admission_inflight_acquired();
        Ok(Self {
            token_bucket,
            tokens,
        })
    }
}

impl Drop for TokenPermit {
    fn drop(&mut self) {
        debug!(
            "TokenPermit: request ended, returning {} tokens to bucket",
            self.tokens
        );
        // Use lock-free sync return - no runtime needed, guaranteed token return
        self.token_bucket.return_tokens_sync(self.tokens);
        Metrics::record_admission_inflight_released();
    }
}

/// Holds one slot of the queue-depth gauge; drop covers admit, reject, and
/// request cancellation.
struct QueueDepthGuard;

impl QueueDepthGuard {
    fn enter() -> Self {
        Metrics::record_admission_queue_entered();
        Self
    }
}

impl Drop for QueueDepthGuard {
    fn drop(&mut self) {
        Metrics::record_admission_queue_exited();
    }
}

/// A body wrapper that holds a token until the body is fully consumed or dropped.
pub struct TokenGuardBody {
    inner: Body,
    _permit: TokenPermit,
}

impl TokenGuardBody {
    fn with_permit(inner: Body, permit: TokenPermit) -> Self {
        Self {
            inner,
            _permit: permit,
        }
    }
}

impl http_body::Body for TokenGuardBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // SAFETY: We never move the inner body, and Body is Unpin
        // (it's a type alias for UnsyncBoxBody which is Unpin)
        let this = self.get_mut();
        Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

async fn run_with_permit(next: Next, request: Request<Body>, permit: TokenPermit) -> Response {
    let (parts, body) = next.run(request).await.into_parts();
    let body = TokenGuardBody::with_permit(body, permit);
    Response::from_parts(parts, Body::new(body))
}

/// Bounded admission queue: at most `queue_size` requests wait for a token;
/// the next arrival sheds immediately. FIFO order comes from the token
/// bucket's waiter queue.
pub struct AdmissionQueue {
    slots: Semaphore,
    queue_timeout: Duration,
}

impl AdmissionQueue {
    pub fn new(queue_size: usize, queue_timeout: Duration) -> Self {
        Self {
            slots: Semaphore::new(queue_size),
            queue_timeout,
        }
    }
}

/// Middleware function for concurrency limiting with optional queuing
pub async fn concurrency_limit_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Cluster-wide rate limiting was previously enforced via the
    // v1 `MeshSyncManager::check_global_rate_limit` path. That hook
    // is removed in this PR. Local per-node token-bucket rate
    // limiting below still applies; cluster aggregation will return
    // through the v2 `RateLimitSyncAdapter` in a follow-up PR.

    let token_bucket = match &app_state.context.rate_limiter {
        Some(bucket) => bucket.clone(),
        None => {
            // Rate limiting disabled, pass through immediately
            return next.run(request).await;
        }
    };

    // Try to acquire token immediately
    if let Ok(permit) = TokenPermit::try_acquire(token_bucket.clone(), 1.0) {
        debug!("Acquired token immediately");
        Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
        return run_with_permit(next, request, permit).await;
    }

    let Some(queue) = &app_state.admission_queue else {
        warn!("No tokens available and queuing is disabled, returning 429");
        Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
        return shed_response(
            StatusCode::TOO_MANY_REQUESTS,
            "admission_queue_full",
            "concurrency limit reached and queuing is disabled",
        );
    };

    let Ok(slot) = queue.slots.try_acquire() else {
        warn!("Request queue is full, returning 429");
        Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
        Metrics::record_admission_rejected(metrics_labels::ADMISSION_REJECTED_FULL);
        return shed_response(
            StatusCode::TOO_MANY_REQUESTS,
            "admission_queue_full",
            "admission queue is at capacity",
        );
    };

    debug!("No tokens available, waiting in admission queue");
    let permit_result = {
        let _queued = QueueDepthGuard::enter();
        TokenPermit::acquire_timeout(token_bucket, 1.0, queue.queue_timeout).await
    };
    match permit_result {
        Ok(permit) => {
            // Free the queue slot before running the request: admitted
            // requests are bounded by the bucket, not the queue.
            drop(slot);
            debug!("Acquired token from queue");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
            run_with_permit(next, request, permit).await
        }
        Err(_) => {
            warn!("Request timed out in admission queue");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            Metrics::record_admission_rejected(metrics_labels::ADMISSION_REJECTED_TIMEOUT);
            shed_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "admission_queue_timeout",
                "timed out waiting for an admission slot",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock,
    };

    use axum::{routing::post, Router};
    use http_body_util::BodyExt;
    use llm_tokenizer::registry::TokenizerRegistry;
    use metrics_exporter_prometheus::PrometheusBuilder;
    use smg_data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    };
    use tokio::sync::{mpsc, oneshot, Notify};
    use tokio_stream::wrappers::ReceiverStream;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        app_context::AppContext, config::RouterConfig, health::ProbeState,
        policies::PolicyRegistry, routers::router_manager::RouterManager, worker::WorkerRegistry,
    };

    fn test_app_state(
        bucket: Arc<TokenBucket>,
        admission_queue: Option<Arc<AdmissionQueue>>,
    ) -> Arc<AppState> {
        let router_config = RouterConfig::default();
        let context = Arc::new(
            AppContext::builder()
                .client(reqwest::Client::new())
                .rate_limiter(Some(bucket))
                .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
                .reasoning_parser_factory(None)
                .tool_parser_factory(None)
                .worker_registry(Arc::new(WorkerRegistry::new()))
                .policy_registry(Arc::new(PolicyRegistry::new(router_config.policy.clone())))
                .router_config(router_config)
                .response_storage(Arc::new(MemoryResponseStorage::new()))
                .conversation_storage(Arc::new(MemoryConversationStorage::new()))
                .conversation_item_storage(Arc::new(MemoryConversationItemStorage::new()))
                .worker_monitor(None)
                .worker_job_queue(Arc::new(OnceLock::new()))
                .workflow_engines(Arc::new(OnceLock::new()))
                .mcp_orchestrator(Arc::new(OnceLock::new()))
                .build()
                .unwrap(),
        );
        Arc::new(AppState {
            router: Arc::new(RouterManager::new(
                context.worker_registry.clone(),
                context.client.clone(),
            )),
            probe_state: ProbeState::new(context.inflight_tracker.clone()),
            context,
            admission_queue,
            router_manager: None,
            mesh_handler: None,
            mesh_adapters: None,
        })
    }

    fn echo_app(app_state: Arc<AppState>) -> Router {
        Router::new()
            .route("/echo", post(|body: Bytes| async move { body }))
            .layer(axum::middleware::from_fn_with_state(
                app_state,
                concurrency_limit_middleware,
            ))
    }

    fn echo_request(body: Body) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/echo")
            .body(body)
            .unwrap()
    }

    /// App with a one-shot channel-fed streaming route plus `/echo`,
    /// queueing disabled.
    fn stream_app(
        bucket: Arc<TokenBucket>,
    ) -> (
        Router,
        mpsc::Sender<Result<Bytes, std::convert::Infallible>>,
    ) {
        let (frame_tx, frame_rx) = mpsc::channel(4);
        let frame_rx = Arc::new(std::sync::Mutex::new(Some(frame_rx)));
        let stream = move || {
            let rx = frame_rx
                .lock()
                .unwrap()
                .take()
                .expect("stream route is one-shot");
            async move { Body::from_stream(ReceiverStream::new(rx)) }
        };
        let app = Router::new()
            .route("/stream", post(stream))
            .route("/echo", post(|body: Bytes| async move { body }))
            .layer(axum::middleware::from_fn_with_state(
                test_app_state(bucket, None),
                concurrency_limit_middleware,
            ));
        (app, frame_tx)
    }

    fn stream_request() -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/stream")
            .body(Body::empty())
            .unwrap()
    }

    fn assert_metric_line(rendered: &str, series: &str, value: &str) {
        let expected = format!("{series} {value}");
        assert!(
            rendered.lines().any(|line| line == expected),
            "expected `{expected}`; rendered:\n{rendered}"
        );
    }

    async fn wait_for_slots(queue: &AdmissionQueue, expected: usize) {
        while queue.slots.available_permits() != expected {
            tokio::task::yield_now().await;
        }
    }

    async fn wait_for_bucket_waiters(bucket: &TokenBucket, expected: usize) {
        while bucket.waiter_count() != expected {
            tokio::task::yield_now().await;
        }
    }

    /// Sets `polled` on the first poll, so a test can detect any body
    /// collection that happens while the request is parked at admission.
    struct PollRecordingBody {
        polled: Arc<AtomicBool>,
        payload: Option<Bytes>,
    }

    impl http_body::Body for PollRecordingBody {
        type Data = Bytes;
        type Error = std::convert::Infallible;

        fn poll_frame(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            let this = self.get_mut();
            this.polled.store(true, Ordering::SeqCst);
            Poll::Ready(this.payload.take().map(|bytes| Ok(Frame::data(bytes))))
        }
    }

    #[test]
    fn response_body_holds_token_until_dropped() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let permit = TokenPermit::try_acquire(bucket.clone(), 1.0).unwrap();
        let body = TokenGuardBody::with_permit(Body::empty(), permit);

        assert_eq!(bucket.available_tokens(), 0.0);
        drop(body);
        assert_eq!(bucket.available_tokens(), 1.0);
    }

    /// A streaming response keeps its token after the handler returns: a
    /// request beyond the cap sheds until the stream is consumed and dropped.
    #[tokio::test]
    async fn streaming_response_holds_token_until_stream_consumed() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let (app, frame_tx) = stream_app(bucket.clone());

        let response = app.clone().oneshot(stream_request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let shed = app
            .clone()
            .oneshot(echo_request(Body::from("x")))
            .await
            .unwrap();
        assert_eq!(shed.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            shed.headers().get(RETRY_AFTER).unwrap(),
            &HeaderValue::from(SHED_RETRY_AFTER_SECS)
        );

        frame_tx
            .send(Ok(Bytes::from_static(b"chunk")))
            .await
            .unwrap();
        let mut body = response.into_body();
        let frame = body.frame().await.unwrap().unwrap();
        assert_eq!(frame.into_data().unwrap(), Bytes::from_static(b"chunk"));

        let shed = app
            .clone()
            .oneshot(echo_request(Body::from("x")))
            .await
            .unwrap();
        assert_eq!(
            shed.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "token must stay held mid-stream"
        );

        drop(frame_tx);
        assert!(body.frame().await.is_none());
        drop(body);
        assert_eq!(bucket.available_tokens(), 1.0);

        let admitted = app.oneshot(echo_request(Body::from("x"))).await.unwrap();
        assert_eq!(admitted.status(), StatusCode::OK);
    }

    /// Dropping the response mid-stream (client disconnect) returns the token.
    #[tokio::test]
    async fn client_disconnect_mid_stream_returns_token() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let (app, frame_tx) = stream_app(bucket.clone());

        let response = app.clone().oneshot(stream_request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        frame_tx
            .send(Ok(Bytes::from_static(b"first")))
            .await
            .unwrap();
        let mut body = response.into_body();
        body.frame().await.unwrap().unwrap();

        assert_eq!(bucket.available_tokens(), 0.0);
        drop(body);
        assert_eq!(bucket.available_tokens(), 1.0);

        let admitted = app.oneshot(echo_request(Body::from("x"))).await.unwrap();
        assert_eq!(admitted.status(), StatusCode::OK);
    }

    /// A buffered response also spans the write: the token frees only once
    /// the body has been consumed.
    #[tokio::test]
    async fn buffered_response_holds_token_until_body_consumed() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let app = echo_app(test_app_state(bucket.clone(), None));

        let response = app
            .clone()
            .oneshot(echo_request(Body::from("payload")))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let shed = app
            .clone()
            .oneshot(echo_request(Body::from("x")))
            .await
            .unwrap();
        assert_eq!(shed.status(), StatusCode::TOO_MANY_REQUESTS);

        let echoed = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&echoed[..], b"payload");
        assert_eq!(bucket.available_tokens(), 1.0);

        let admitted = app.oneshot(echo_request(Body::from("x"))).await.unwrap();
        assert_eq!(admitted.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn cancellation_returns_acquired_token() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let task_bucket = bucket.clone();
        let (acquired_tx, acquired_rx) = oneshot::channel();
        #[expect(
            clippy::disallowed_methods,
            reason = "Test helper: the spawned task is explicitly aborted and awaited before the test ends"
        )]
        let task = tokio::spawn(async move {
            let _permit = TokenPermit::try_acquire(task_bucket, 1.0).unwrap();
            let _ = acquired_tx.send(());
            std::future::pending::<()>().await;
        });

        acquired_rx.await.unwrap();
        assert_eq!(bucket.available_tokens(), 0.0);
        task.abort();
        let _ = task.await;
        assert_eq!(bucket.available_tokens(), 1.0);
    }

    /// With the cap saturated, exactly `queue_size` requests park; the next
    /// arrival sheds 429 immediately, and the parked ones admit FIFO once
    /// capacity frees.
    #[tokio::test]
    #[expect(
        clippy::disallowed_methods,
        reason = "test parks requests in background tasks while asserting the bound"
    )]
    async fn queue_admits_exactly_queue_size_waiters_then_sheds() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let queue = Arc::new(AdmissionQueue::new(2, Duration::from_secs(5)));
        let app = echo_app(test_app_state(bucket.clone(), Some(queue.clone())));

        let held = TokenPermit::try_acquire(bucket.clone(), 1.0).unwrap();

        let first = tokio::spawn(app.clone().oneshot(echo_request(Body::from("first"))));
        wait_for_slots(&queue, 1).await;
        wait_for_bucket_waiters(&bucket, 1).await;
        let second = tokio::spawn(app.clone().oneshot(echo_request(Body::from("second"))));
        wait_for_slots(&queue, 0).await;
        wait_for_bucket_waiters(&bucket, 2).await;

        let shed = app
            .clone()
            .oneshot(echo_request(Body::from("overflow")))
            .await
            .unwrap();
        assert_eq!(shed.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            shed.headers().get(RETRY_AFTER).unwrap(),
            &HeaderValue::from(SHED_RETRY_AFTER_SECS)
        );
        assert!(!first.is_finished());
        assert!(!second.is_finished());

        drop(held);
        let first_response = first.await.unwrap().unwrap();
        assert_eq!(first_response.status(), StatusCode::OK);
        assert!(
            !second.is_finished(),
            "second waiter must stay parked until the first releases"
        );
        let echoed = axum::body::to_bytes(first_response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&echoed[..], b"first");

        let second_response = second.await.unwrap().unwrap();
        assert_eq!(second_response.status(), StatusCode::OK);
        let echoed = axum::body::to_bytes(second_response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&echoed[..], b"second");

        wait_for_slots(&queue, 2).await;
    }

    /// A parked request that is cancelled (client disconnect) frees its
    /// queue slot for the next arrival.
    #[tokio::test]
    #[expect(
        clippy::disallowed_methods,
        reason = "test parks a request in a background task and aborts it"
    )]
    async fn cancelled_parked_request_frees_queue_slot() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let queue = Arc::new(AdmissionQueue::new(1, Duration::from_secs(5)));
        let app = echo_app(test_app_state(bucket.clone(), Some(queue.clone())));

        let held = TokenPermit::try_acquire(bucket, 1.0).unwrap();

        let parked = tokio::spawn(app.clone().oneshot(echo_request(Body::from("parked"))));
        wait_for_slots(&queue, 0).await;

        parked.abort();
        let _ = parked.await;
        wait_for_slots(&queue, 1).await;

        drop(held);
        let admitted = app.oneshot(echo_request(Body::from("next"))).await.unwrap();
        assert_eq!(admitted.status(), StatusCode::OK);
    }

    /// A request parked in the admission queue must keep its body unpolled;
    /// collection may only happen at handler extraction after admission.
    #[tokio::test]
    #[expect(
        clippy::disallowed_methods,
        reason = "test drives two concurrent requests through the real middleware stack"
    )]
    async fn queued_request_body_stays_unpolled_until_admitted() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let queue = Arc::new(AdmissionQueue::new(4, Duration::from_secs(5)));

        let hold_gate = Arc::new(Notify::new());
        let entered = Arc::new(Notify::new());
        let hold = {
            let hold_gate = hold_gate.clone();
            let entered = entered.clone();
            move || {
                let hold_gate = hold_gate.clone();
                let entered = entered.clone();
                async move {
                    entered.notify_one();
                    hold_gate.notified().await;
                    "held"
                }
            }
        };
        let app = Router::new()
            .route("/hold", post(hold))
            .route("/echo", post(|body: Bytes| async move { body }))
            .layer(axum::middleware::from_fn_with_state(
                test_app_state(bucket, Some(queue)),
                concurrency_limit_middleware,
            ));

        let first = tokio::spawn(
            app.clone().oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/hold")
                    .body(Body::empty())
                    .unwrap(),
            ),
        );
        entered.notified().await;

        let polled = Arc::new(AtomicBool::new(false));
        let second = tokio::spawn(app.clone().oneshot(echo_request(Body::new(
            PollRecordingBody {
                polled: polled.clone(),
                payload: Some(Bytes::from_static(b"parked-payload")),
            },
        ))));

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !second.is_finished(),
            "second request must be parked in the queue"
        );
        assert!(
            !polled.load(Ordering::SeqCst),
            "parked request body must not be polled"
        );

        hold_gate.notify_one();
        let first_response = first.await.unwrap().unwrap();
        assert_eq!(first_response.status(), StatusCode::OK);
        drop(first_response);

        let second_response = second.await.unwrap().unwrap();
        assert_eq!(second_response.status(), StatusCode::OK);
        let echoed = axum::body::to_bytes(second_response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&echoed[..], b"parked-payload");
        assert!(polled.load(Ordering::SeqCst));
    }

    /// Queue depth and inflight gauges move as requests park, and a full
    /// queue rejects with reason="full".
    #[test]
    #[expect(
        clippy::disallowed_methods,
        reason = "test parks a request in the background while asserting gauge movement"
    )]
    fn admission_metrics_track_depth_inflight_and_full_queue() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let bucket = Arc::new(TokenBucket::new(1, 0));
                let queue = Arc::new(AdmissionQueue::new(1, Duration::from_secs(5)));
                let app = echo_app(test_app_state(bucket.clone(), Some(queue.clone())));

                let held = TokenPermit::try_acquire(bucket, 1.0).unwrap();
                let parked = tokio::spawn(app.clone().oneshot(echo_request(Body::from("parked"))));
                wait_for_slots(&queue, 0).await;
                assert!(!parked.is_finished());

                let rejected = app
                    .oneshot(echo_request(Body::from("rejected")))
                    .await
                    .unwrap();
                assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
                assert_eq!(
                    rejected.headers().get(RETRY_AFTER).unwrap(),
                    &HeaderValue::from(SHED_RETRY_AFTER_SECS)
                );
                let body = axum::body::to_bytes(rejected.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(json["error"]["code"], "admission_queue_full");

                let rendered = handle.render();
                assert_metric_line(&rendered, "smg_admission_queue_depth", "1");
                assert_metric_line(&rendered, "smg_admission_inflight", "1");
                assert_metric_line(
                    &rendered,
                    "smg_admission_queue_rejected_total{reason=\"full\"}",
                    "1",
                );

                parked.abort();
                let _ = parked.await;
                drop(held);
            });
        });
    }

    /// A request that outlives the queue timeout sheds as 503 + Retry-After
    /// with reason="timeout" and releases its queue-depth slot.
    #[test]
    fn admission_metrics_record_timeout_rejections() {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();

        metrics::with_local_recorder(&recorder, || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let bucket = Arc::new(TokenBucket::new(1, 0));
                let queue = Arc::new(AdmissionQueue::new(4, Duration::from_millis(100)));
                let app = echo_app(test_app_state(bucket.clone(), Some(queue)));

                let held = TokenPermit::try_acquire(bucket, 1.0).unwrap();
                let response = app.oneshot(echo_request(Body::from("late"))).await.unwrap();
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                assert_eq!(
                    response.headers().get(RETRY_AFTER).unwrap(),
                    &HeaderValue::from(SHED_RETRY_AFTER_SECS)
                );
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(json["error"]["code"], "admission_queue_timeout");
                drop(held);
            });
        });

        let rendered = handle.render();
        assert_metric_line(
            &rendered,
            "smg_admission_queue_rejected_total{reason=\"timeout\"}",
            "1",
        );
        assert_metric_line(&rendered, "smg_admission_queue_depth", "0");
        assert_metric_line(&rendered, "smg_admission_inflight", "0");
    }
}
