//! Per-request concurrency limiting via a token bucket, with optional
//! queuing for backpressure.
//!
//! `ConcurrencyLimiter` wires a bounded `mpsc` channel that
//! `concurrency_limit_middleware` uses to enqueue requests when the
//! bucket is empty; `QueueProcessor` drains that channel and hands tokens
//! back to waiters. `TokenGuardBody` wraps the response body so the token
//! is only released after the entire stream has been delivered.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header::RETRY_AFTER, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use http_body::Frame;
use tokio::{
    sync::{mpsc, oneshot},
    time::error::Elapsed,
};
use tracing::{debug, error, warn};

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

/// Request queue entry
pub struct QueuedRequest {
    /// Time when the request was queued
    queued_at: Instant,
    /// Channel to send the permit back when acquired
    permit_tx: oneshot::Sender<Result<TokenPermit, StatusCode>>,
}

/// Queue processor that handles queued requests
pub struct QueueProcessor {
    token_bucket: Arc<TokenBucket>,
    queue_rx: mpsc::Receiver<QueuedRequest>,
    queue_timeout: Duration,
}

impl QueueProcessor {
    pub fn new(
        token_bucket: Arc<TokenBucket>,
        queue_rx: mpsc::Receiver<QueuedRequest>,
        queue_timeout: Duration,
    ) -> Self {
        Self {
            token_bucket,
            queue_rx,
            queue_timeout,
        }
    }

    pub async fn run(mut self) {
        debug!("Starting concurrency queue processor");

        // Process requests in a single task to reduce overhead
        while let Some(queued) = self.queue_rx.recv().await {
            // Check timeout immediately
            let elapsed = queued.queued_at.elapsed();
            if elapsed >= self.queue_timeout {
                warn!("Request already timed out in queue");
                let _ = queued.permit_tx.send(Err(StatusCode::SERVICE_UNAVAILABLE));
                continue;
            }

            let remaining_timeout = self.queue_timeout - elapsed;

            // Try to acquire token for this request
            if let Ok(permit) = TokenPermit::try_acquire(self.token_bucket.clone(), 1.0) {
                // Got token immediately
                debug!("Queue: acquired token immediately for queued request");
                let _ = queued.permit_tx.send(Ok(permit));
            } else {
                // Need to wait for token
                let token_bucket = self.token_bucket.clone();

                // Spawn task only when we actually need to wait
                #[expect(
                    clippy::disallowed_methods,
                    reason = "fire-and-forget permit acquisition: task is bounded by remaining_timeout and communicates via oneshot; dropping the JoinHandle detaches the task but it self-terminates"
                )]
                tokio::spawn(async move {
                    if let Ok(permit) =
                        TokenPermit::acquire_timeout(token_bucket, 1.0, remaining_timeout).await
                    {
                        debug!("Queue: acquired token after waiting");
                        let _ = queued.permit_tx.send(Ok(permit));
                    } else {
                        warn!("Queue: request timed out waiting for token");
                        let _ = queued.permit_tx.send(Err(StatusCode::SERVICE_UNAVAILABLE));
                    }
                });
            }
        }

        warn!("Concurrency queue processor shutting down");
    }
}

/// State for the concurrency limiter
pub struct ConcurrencyLimiter {
    pub queue_tx: Option<mpsc::Sender<QueuedRequest>>,
}

impl ConcurrencyLimiter {
    /// Create new concurrency limiter with optional queue
    pub fn new(
        token_bucket: Option<Arc<TokenBucket>>,
        queue_size: usize,
        queue_timeout: Duration,
    ) -> (Self, Option<QueueProcessor>) {
        match (token_bucket, queue_size) {
            (None, _) => (Self { queue_tx: None }, None),
            (Some(bucket), size) if size > 0 => {
                let (queue_tx, queue_rx) = mpsc::channel(size);
                let processor = QueueProcessor::new(bucket, queue_rx, queue_timeout);
                (
                    Self {
                        queue_tx: Some(queue_tx),
                    },
                    Some(processor),
                )
            }
            (Some(_), _) => (Self { queue_tx: None }, None),
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
        run_with_permit(next, request, permit).await
    } else {
        // No tokens available, try to queue if enabled
        if let Some(queue_tx) = &app_state.concurrency_queue_tx {
            debug!("No tokens available, attempting to queue request");

            // Create a channel for the token response
            let (permit_tx, permit_rx) = oneshot::channel();

            let queued = QueuedRequest {
                queued_at: Instant::now(),
                permit_tx,
            };

            // Try to send to queue
            match queue_tx.try_send(queued) {
                Ok(()) => {
                    // Wait for token from queue processor
                    let permit_result = {
                        let _queued = QueueDepthGuard::enter();
                        permit_rx.await
                    };
                    match permit_result {
                        Ok(Ok(permit)) => {
                            debug!("Acquired token from queue");
                            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
                            run_with_permit(next, request, permit).await
                        }
                        Ok(Err(status)) => {
                            warn!("Queue returned error status: {}", status);
                            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
                            Metrics::record_admission_rejected(
                                metrics_labels::ADMISSION_REJECTED_TIMEOUT,
                            );
                            shed_response(
                                status,
                                "admission_queue_timeout",
                                "timed out waiting for an admission slot",
                            )
                        }
                        Err(_) => {
                            error!("Queue response channel closed");
                            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
                            StatusCode::INTERNAL_SERVER_ERROR.into_response()
                        }
                    }
                }
                Err(_) => {
                    warn!("Request queue is full, returning 429");
                    Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
                    Metrics::record_admission_rejected(metrics_labels::ADMISSION_REJECTED_FULL);
                    shed_response(
                        StatusCode::TOO_MANY_REQUESTS,
                        "admission_queue_full",
                        "admission queue is at capacity",
                    )
                }
            }
        } else {
            warn!("No tokens available and queuing is disabled, returning 429");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            shed_response(
                StatusCode::TOO_MANY_REQUESTS,
                "admission_queue_full",
                "concurrency limit reached and queuing is disabled",
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
    use tokio::sync::Notify;
    use tokio_stream::wrappers::ReceiverStream;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        app_context::AppContext, config::RouterConfig, health::ProbeState,
        policies::PolicyRegistry, routers::router_manager::RouterManager, worker::WorkerRegistry,
    };

    fn test_app_state(
        bucket: Arc<TokenBucket>,
        queue_tx: Option<mpsc::Sender<QueuedRequest>>,
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
            concurrency_queue_tx: queue_tx,
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

    #[tokio::test]
    async fn cancelled_queued_request_returns_acquired_token() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let (queue_tx, queue_rx) = mpsc::channel(1);
        let (permit_tx, permit_rx) = oneshot::channel();
        drop(permit_rx);

        assert!(queue_tx
            .send(QueuedRequest {
                queued_at: Instant::now(),
                permit_tx,
            })
            .await
            .is_ok());
        drop(queue_tx);

        QueueProcessor::new(bucket.clone(), queue_rx, Duration::from_secs(1))
            .run()
            .await;
        assert_eq!(bucket.available_tokens(), 1.0);
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
        let (limiter, processor) =
            ConcurrencyLimiter::new(Some(bucket.clone()), 4, Duration::from_secs(5));
        tokio::spawn(processor.unwrap().run());

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
                test_app_state(bucket, limiter.queue_tx),
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
                let (limiter, processor) =
                    ConcurrencyLimiter::new(Some(bucket.clone()), 1, Duration::from_secs(5));
                // Keep the queue channel open but undrained: the first
                // request parks and the second finds the queue full.
                let _processor = processor.unwrap();
                let app = echo_app(test_app_state(bucket.clone(), limiter.queue_tx));

                let held = TokenPermit::try_acquire(bucket, 1.0).unwrap();
                let parked = tokio::spawn(app.clone().oneshot(echo_request(Body::from("parked"))));
                tokio::time::sleep(Duration::from_millis(50)).await;
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
    #[expect(
        clippy::disallowed_methods,
        reason = "queue processor runs as a background task"
    )]
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
                let (limiter, processor) =
                    ConcurrencyLimiter::new(Some(bucket.clone()), 4, Duration::from_millis(100));
                tokio::spawn(processor.unwrap().run());
                let app = echo_app(test_app_state(bucket.clone(), limiter.queue_tx));

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
