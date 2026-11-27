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
    ADMISSION_QUEUE_FULL_CODE, ADMISSION_QUEUE_FULL_STATUS, ADMISSION_QUEUE_TIMEOUT_CODE,
    ADMISSION_QUEUE_TIMEOUT_STATUS, SHED_RETRY_AFTER_SECS,
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

    fn call(&mut self, request: Request<Body>) -> Self::Future {
        // Rate limiting disabled: the route's own future, untouched.
        let Some(token_bucket) = &self.app_state.context.rate_limiter else {
            return ConcurrencyLimitFuture::Through {
                inner: self.inner.call(request),
            };
        };
        if let Ok(permit) = TokenPermit::try_acquire(Arc::clone(token_bucket), 1.0) {
            debug!("Acquired token immediately");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
            // Granted at once: the acceptance stamped at tenant resolution a
            // moment ago stands (no wait in the queue to leave out, no second
            // clock read).
            return ConcurrencyLimitFuture::Admitted {
                inner: self.inner.call(request),
                permit: Some(permit),
            };
        }
        let Some(queue) = &self.app_state.admission_queue else {
            warn!("No tokens available and queuing is disabled, returning 429");
            Metrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            return ConcurrencyLimitFuture::Shed {
                status: ADMISSION_QUEUE_FULL_STATUS,
                code: ADMISSION_QUEUE_FULL_CODE,
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
                status: ADMISSION_QUEUE_FULL_STATUS,
                code: ADMISSION_QUEUE_FULL_CODE,
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
                ADMISSION_QUEUE_TIMEOUT_STATUS,
                ADMISSION_QUEUE_TIMEOUT_CODE,
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
#[path = "../../tests/unit/middleware/concurrency/tests.rs"]
mod tests;
