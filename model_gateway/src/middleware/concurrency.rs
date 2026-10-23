//! Per-request concurrency limiting via a token bucket, with optional
//! queuing for backpressure.
//!
//! `AdmissionQueue` caps concurrent waiters at `queue_size`; a request
//! that finds every slot taken sheds immediately. Waiters park directly
//! on the token bucket, which serves them in FIFO order. The token rides on
//! the response body (the edge's one body wrapper, see `edge_body`), so it
//! is only released after the entire stream has been delivered.

use std::{
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{ready, Context, Poll},
    time::Duration,
};

use axum::{
    body::Body,
    extract::Request,
    http::{header::RETRY_AFTER, HeaderValue, StatusCode},
    response::Response,
};
use futures_util::future::BoxFuture;
use pin_project_lite::pin_project;
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    time::error::Elapsed,
};
use tower::{Layer, Service};
use tracing::{debug, warn};

use super::{
    edge_body::{BodyDuties, Permit},
    tenant_resolution::restamp_accepted_at,
    token_bucket::TokenBucket,
    SHED_RETRY_AFTER_SECS,
};
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
pub struct TokenPermit {
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

/// On the response body: every frame passes; the drop returns the token.
impl Permit for TokenPermit {}

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

/// Bounded admission queue: at most `queue_size` requests wait for a token;
/// the next arrival sheds immediately. FIFO order comes from the token
/// bucket's waiter queue.
pub struct AdmissionQueue {
    slots: Arc<Semaphore>,
    queue_timeout: Duration,
}

impl AdmissionQueue {
    pub fn new(queue_size: usize, queue_timeout: Duration) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(queue_size)),
            queue_timeout,
        }
    }
}

/// The legacy admission as a route layer: the token bucket bounds the
/// requests in flight, the optional queue parks the overflow. A request that
/// finds a token takes the route's own future, the permit handed to its
/// response on the way out: no clone of the route service, no boxed future.
/// The wait for a token is the one path that runs before the route, and the
/// one that is boxed; a shed for queuing disabled or a full queue is answered
/// at once, a wait that times out is answered from the boxed path.
#[derive(Clone)]
pub struct ConcurrencyLimitLayer {
    app_state: Arc<AppState>,
}

impl ConcurrencyLimitLayer {
    pub fn new(app_state: Arc<AppState>) -> Self {
        Self { app_state }
    }
}

impl<S> Layer<S> for ConcurrencyLimitLayer {
    type Service = ConcurrencyLimitService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ConcurrencyLimitService {
            inner,
            app_state: self.app_state.clone(),
        }
    }
}

/// The service [`ConcurrencyLimitLayer`] wraps a route in. The token rides
/// on the response body (the edge's one body wrapper, see `edge_body`), so it
/// is released only after the entire stream has been delivered; a request
/// parked at admission keeps its body unread.
#[derive(Clone)]
pub struct ConcurrencyLimitService<S> {
    inner: S,
    app_state: Arc<AppState>,
}

impl<S> Service<Request<Body>> for ConcurrencyLimitService<S>
where
    S: Service<Request<Body>, Response = Response, Error = Infallible> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = Infallible;
    type Future = ConcurrencyLimitFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: Request<Body>) -> Self::Future {
        // Rate limiting disabled: the route's own future, untouched.
        let Some(token_bucket) = &self.app_state.context.rate_limiter else {
            return ConcurrencyLimitFuture::Through {
                inner: self.inner.call(request),
            };
        };
        if let Ok(permit) = TokenPermit::try_acquire(Arc::clone(token_bucket), 1.0) {
            debug!("Acquired token immediately");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
            // The bound ahead of worker selection counts from here.
            restamp_accepted_at(request.extensions_mut());
            return ConcurrencyLimitFuture::Admitted {
                inner: self.inner.call(request),
                permit: Some(permit),
            };
        }
        let Some(queue) = &self.app_state.admission_queue else {
            warn!("No tokens available and queuing is disabled, returning 429");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            return ConcurrencyLimitFuture::Shed {
                status: StatusCode::TOO_MANY_REQUESTS,
                code: "admission_queue_full",
                message: "concurrency limit reached and queuing is disabled",
            };
        };
        // The queue's slot is taken here, before the route is cloned and the
        // wait boxed: a request a full queue turns away costs neither.
        let Ok(slot) = Arc::clone(&queue.slots).try_acquire_owned() else {
            warn!("Request queue is full, returning 429");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            Metrics::record_admission_rejected(metrics_labels::ADMISSION_REJECTED_FULL);
            return ConcurrencyLimitFuture::Shed {
                status: StatusCode::TOO_MANY_REQUESTS,
                code: "admission_queue_full",
                message: "admission queue is at capacity",
            };
        };
        // The route service a queued request is handed to is the one that
        // reported ready, as tower's contract has it; the clone stays here.
        let clone = self.inner.clone();
        let route = std::mem::replace(&mut self.inner, clone);
        ConcurrencyLimitFuture::Queued {
            inner: Box::pin(admit_from_queue(
                route,
                request,
                slot,
                Arc::clone(token_bucket),
                queue.queue_timeout,
            )),
        }
    }
}

/// The wait in the admission queue (its slot taken by the caller), then the
/// route: a wait longer than the queue's timeout sheds. Waiters park on the
/// token bucket, which serves them in FIFO order.
async fn admit_from_queue<S>(
    mut route: S,
    mut request: Request<Body>,
    slot: OwnedSemaphorePermit,
    token_bucket: Arc<TokenBucket>,
    queue_timeout: Duration,
) -> Result<Response, Infallible>
where
    S: Service<Request<Body>, Response = Response, Error = Infallible>,
{
    debug!("No tokens available, waiting in admission queue");
    let permit_result = {
        let _queued = QueueDepthGuard::enter();
        TokenPermit::acquire_timeout(token_bucket, 1.0, queue_timeout).await
    };
    match permit_result {
        Ok(permit) => {
            // Free the queue slot before running the request: admitted
            // requests are bounded by the bucket, not the queue.
            drop(slot);
            debug!("Acquired token from queue");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
            // The wait for the permit was the admission queue's, bounded by
            // its own timeout; the bound ahead of worker selection counts
            // from here.
            restamp_accepted_at(request.extensions_mut());
            let Ok(mut response) = route.call(request).await;
            hand_permit(&mut response, permit);
            Ok(response)
        }
        Err(_) => {
            warn!("Request timed out in admission queue");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            Metrics::record_admission_rejected(metrics_labels::ADMISSION_REJECTED_TIMEOUT);
            Ok(shed_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "admission_queue_timeout",
                "timed out waiting for an admission slot",
            ))
        }
    }
}

/// The permit on the response: held until the response body is fully
/// consumed or dropped, by the edge's one body wrapper.
fn hand_permit(response: &mut Response, permit: TokenPermit) {
    BodyDuties::hand(response, |duties| {
        duties.permit = Some(Box::new(permit));
    });
}

pin_project! {
    /// The admission layer's future: the route's own future where a token was
    /// there (the permit handed to the response when it is ready), the boxed
    /// wait where it was not, the shed built when it is polled.
    #[project = ConcurrencyLimitProj]
    pub enum ConcurrencyLimitFuture<F> {
        /// Rate limiting disabled: the route's future as it is.
        Through { #[pin] inner: F },
        /// A token at once: the route's future, the permit for its response.
        Admitted { #[pin] inner: F, permit: Option<TokenPermit> },
        /// No token: the wait in the admission queue, then the route.
        Queued { #[pin] inner: BoxFuture<'static, Result<Response, Infallible>> },
        /// Shed at once (queuing disabled, or the queue full): the answer is
        /// built when polled.
        Shed { status: StatusCode, code: &'static str, message: &'static str },
    }
}

impl<F> Future for ConcurrencyLimitFuture<F>
where
    F: Future<Output = Result<Response, Infallible>>,
{
    type Output = Result<Response, Infallible>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.project() {
            ConcurrencyLimitProj::Through { inner } => inner.poll(cx),
            ConcurrencyLimitProj::Admitted { inner, permit } => {
                let Ok(mut response) = ready!(inner.poll(cx));
                if let Some(permit) = permit.take() {
                    hand_permit(&mut response, permit);
                }
                Poll::Ready(Ok(response))
            }
            ConcurrencyLimitProj::Queued { inner } => inner.poll(cx),
            ConcurrencyLimitProj::Shed {
                status,
                code,
                message,
            } => Poll::Ready(Ok(shed_response(*status, code, message))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        pin::Pin,
        sync::{
            atomic::{AtomicBool, Ordering},
            OnceLock,
        },
        task::{Context, Poll},
    };

    use axum::{routing::post, Router};
    use bytes::Bytes;
    use http_body::Frame;
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
        app_context::AppContext,
        config::RouterConfig,
        health::ProbeState,
        middleware::{edge_body::at_the_edge, HttpMetricsLayer},
        observability::inflight_tracker::InFlightRequestTracker,
        policies::PolicyRegistry,
        routers::gateway::Gateway,
        worker::WorkerRegistry,
    };

    fn test_app_state(
        bucket: Arc<TokenBucket>,
        admission_queue: Option<Arc<AdmissionQueue>>,
    ) -> Arc<AppState> {
        let router_config = RouterConfig::default();
        let context = Arc::new(
            AppContext::builder()
                .client(crate::clients::http_client())
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
            router: Arc::new(Gateway::new(context.worker_registry.clone())),
            probe_state: ProbeState::new(context.inflight_tracker.clone()),
            context,
            admission_queue,
            gateway: None,
            mesh_handler: None,
            mesh_adapters: None,
        })
    }

    /// The admission layer under the server's edge (the metrics layer, which
    /// builds the one body wrapper that holds the token).
    fn echo_app(app_state: Arc<AppState>) -> Router {
        Router::new()
            .route("/echo", post(|body: Bytes| async move { body }))
            .layer(ConcurrencyLimitLayer::new(app_state))
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
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
    fn stream_app(bucket: Arc<TokenBucket>) -> (Router, mpsc::Sender<Result<Bytes, Infallible>>) {
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
            .layer(ConcurrencyLimitLayer::new(test_app_state(bucket, None)))
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));
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
        type Error = Infallible;

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
        let mut response = Response::new(Body::from("payload"));
        BodyDuties::hand(&mut response, |duties| {
            duties.permit = Some(Box::new(permit));
        });
        let body = at_the_edge(response).into_body();

        assert_eq!(bucket.available_tokens(), 0.0);
        drop(body);
        assert_eq!(bucket.available_tokens(), 1.0);
    }

    /// A body complete when it reaches the edge (an empty answer) returns its
    /// token there: nothing is left to deliver.
    #[test]
    fn an_already_complete_body_returns_the_token_at_the_edge() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let permit = TokenPermit::try_acquire(bucket.clone(), 1.0).unwrap();
        let mut response = Response::new(Body::empty());
        BodyDuties::hand(&mut response, |duties| {
            duties.permit = Some(Box::new(permit));
        });
        let response = at_the_edge(response);
        assert_eq!(bucket.available_tokens(), 1.0);
        drop(response);
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
            .layer(ConcurrencyLimitLayer::new(test_app_state(
                bucket,
                Some(queue),
            )))
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()));

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
