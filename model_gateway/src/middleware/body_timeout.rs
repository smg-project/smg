//! Watchdog for request bodies the handlers buffer at extraction.
//!
//! A typed-JSON handler collects its body after admission granted a permit.
//! Nothing bounded that read: a client that stopped sending, or trickled,
//! kept its connection, its in-flight entry and its admission permit for as
//! long as it liked. [`RequestBodyTimeouts::layer`] wraps the body so the
//! read fails once a single wait on the client lasts the stream-body stall
//! timeout, or once the body is still incomplete at the request timeout; the
//! layer then answers 408 and the admission guard around the response
//! releases the permit as the response drops. Both clocks run only while the
//! consumer is waiting on the client: a parsed body, a request parked at
//! admission or a slow handler never trips them. A read the payload limit
//! ended is answered here too, as 413: the limit layers sit outside this
//! wrapper, which boxes their error once more, past where the JSON
//! extractor's own 413 branch looks for it.

use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use axum::{body::Body, extract::Request, http::StatusCode, response::Response};
use bytes::Bytes;
use http_body::Frame;
use pin_project_lite::pin_project;
use tokio::time::Sleep;
use tower::{Layer, Service};
use tracing::warn;

use crate::{config::types::RouterConfig, routers::error::create_error};

/// Error code of a body whose client made no progress for the stall timeout.
pub const REQUEST_BODY_STALLED: &str = "request_body_stalled";
/// Error code of a body still incomplete at the request timeout.
pub const REQUEST_BODY_TIMEOUT: &str = "request_body_timeout";
/// Error code of a body that crossed the payload limit (`max_payload_size`).
pub const REQUEST_BODY_TOO_LARGE: &str = "request_body_too_large";

/// The refusal of a body over the payload limit: one code and one message
/// for every framing (the chunked read here, a declared length at the edge
/// in `body_limit`).
pub(crate) fn request_body_too_large_response() -> Response {
    create_error(
        StatusCode::PAYLOAD_TOO_LARGE,
        REQUEST_BODY_TOO_LARGE,
        "Request body exceeded the payload limit (max_payload_size)",
    )
}

/// The refusal of a body whose client made no progress for the stall
/// timeout, under [`REQUEST_BODY_STALLED`]: one code and one message for
/// every framing (the buffered read here, the streamed read in the HTTP
/// router).
pub(crate) fn request_body_stalled_response(timeout_secs: u64) -> Response {
    create_error(
        StatusCode::REQUEST_TIMEOUT,
        REQUEST_BODY_STALLED,
        format!("No request body bytes arrived from the client for {timeout_secs} seconds"),
    )
}

const READING: u8 = 0;
const STALLED: u8 = 1;
const TIMED_OUT: u8 = 2;
const TOO_LARGE: u8 = 3;

/// The two clocks of a buffered body read, from the router config.
#[derive(Clone, Copy, Debug)]
pub struct RequestBodyTimeouts {
    /// Longest single wait on the client (`stream_body_stall_timeout_secs`);
    /// `None` disables the stall watchdog.
    stall: Option<Duration>,
    /// Longest read overall (`request_timeout_secs`).
    total: Duration,
}

impl RequestBodyTimeouts {
    pub fn from_config(config: &RouterConfig) -> Self {
        Self {
            stall: match config.stream_body_stall_timeout_secs {
                0 => None,
                secs => Some(Duration::from_secs(secs)),
            },
            total: Duration::from_secs(config.request_timeout_secs),
        }
    }
}

/// `request` with its body under the two clocks, and the flag the read's end
/// is written to.
fn timed(request: Request<Body>, timeouts: RequestBodyTimeouts) -> (Request<Body>, Arc<AtomicU8>) {
    let outcome = Arc::new(AtomicU8::new(READING));
    // The request timeout counts from here, inside the admission layers: a
    // wait for a permit is bounded by its own timeout and is not time the
    // client had to send its body, so the deadline is read at this layer,
    // not taken from the edge's entry stamp.
    let request =
        request.map(|body| Body::new(TimedBody::new(body, timeouts, Arc::clone(&outcome))));
    (request, outcome)
}

/// The response a request leaves with once its body read is settled: the
/// inner service's own, or the 408 / 413 the read's end calls for.
fn settled(timeouts: RequestBodyTimeouts, outcome: &AtomicU8, response: Response) -> Response {
    match outcome.load(Ordering::Acquire) {
        STALLED => {
            let timeout_secs = timeouts.stall.map_or(0, |stall| stall.as_secs());
            warn!(
                timeout_secs,
                "Request body stalled waiting on the client; ending the request"
            );
            request_body_stalled_response(timeout_secs)
        }
        TIMED_OUT => {
            let timeout_secs = timeouts.total.as_secs();
            warn!(
                timeout_secs,
                "Request body still incomplete at the request timeout; ending the request"
            );
            create_error(
                StatusCode::REQUEST_TIMEOUT,
                REQUEST_BODY_TIMEOUT,
                format!("Request body was not complete after {timeout_secs} seconds"),
            )
        }
        TOO_LARGE => request_body_too_large_response(),
        _ => response,
    }
}

impl RequestBodyTimeouts {
    /// The body watchdog as a route layer: the body is wrapped before the
    /// inner service is called and the response is settled from the
    /// service's own future when it is ready. Hand-written rather than an
    /// `axum::middleware::from_fn`, which clones the inner service and boxes
    /// a future on every request.
    pub fn layer(self) -> RequestBodyTimeoutLayer {
        RequestBodyTimeoutLayer { timeouts: self }
    }
}

/// [`RequestBodyTimeouts::layer`].
#[derive(Clone, Copy, Debug)]
pub struct RequestBodyTimeoutLayer {
    timeouts: RequestBodyTimeouts,
}

impl<S> Layer<S> for RequestBodyTimeoutLayer {
    type Service = RequestBodyTimeoutService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestBodyTimeoutService {
            inner,
            timeouts: self.timeouts,
        }
    }
}

/// The service [`RequestBodyTimeoutLayer`] wraps a route in.
#[derive(Clone, Debug)]
pub struct RequestBodyTimeoutService<S> {
    inner: S,
    timeouts: RequestBodyTimeouts,
}

impl<S, E> Service<Request<Body>> for RequestBodyTimeoutService<S>
where
    S: Service<Request<Body>, Response = Response, Error = E>,
{
    type Response = Response;
    type Error = E;
    type Future = SettledFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request<Body>) -> Self::Future {
        let (request, outcome) = timed(request, self.timeouts);
        SettledFuture {
            inner: self.inner.call(request),
            outcome,
            timeouts: self.timeouts,
        }
    }
}

pin_project! {
    /// The inner service's future, its response settled by the body read's end.
    pub struct SettledFuture<F> {
        #[pin]
        inner: F,
        outcome: Arc<AtomicU8>,
        timeouts: RequestBodyTimeouts,
    }
}

impl<F, E> Future for SettledFuture<F>
where
    F: Future<Output = Result<Response, E>>,
{
    type Output = Result<Response, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let response = std::task::ready!(this.inner.poll(cx))?;
        Poll::Ready(Ok(settled(*this.timeouts, this.outcome, response)))
    }
}

/// Whether a failed read is the payload limit's `LengthLimitError`, however
/// many body wrappers boxed it on the way in. The limit layers sit outside
/// this one, and every `Body::new` over a wrapped body adds a layer of
/// `axum::Error`; the JSON extractor's 413 branch, like axum's own, looks
/// two layers deep, which this wrapper's layer put the limit error beyond.
pub(super) fn crossed_payload_limit(err: &axum::Error) -> bool {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(err) = source {
        if err.is::<http_body_util::LengthLimitError>() {
            return true;
        }
        source = err.source();
    }
    false
}

/// Body wrapper that fails the read when the client stalls or the total
/// read time runs out. The timers are polled only while the inner body is
/// pending, i.e. while the consumer is waiting on the client.
struct TimedBody {
    inner: Body,
    stall: Option<Duration>,
    /// Armed on the first pending poll of a wait, cleared on progress.
    stall_timer: Option<Pin<Box<Sleep>>>,
    /// When the request timeout ends, read once at the wrap; the timer that
    /// waits for it is built on the first poll that has to wait, as the stall
    /// timer is: a body that arrives whole never waits, and a timer boxed at
    /// the wrap was one allocation per request for nothing.
    deadline_at: tokio::time::Instant,
    deadline: Option<Pin<Box<Sleep>>>,
    outcome: Arc<AtomicU8>,
    ended: bool,
}

/// `now + total`, saturated to a far future when the sum does not fit an
/// instant: a `request_timeout_secs` too large to hold (the "no timeout"
/// idiom) is a deadline that never comes, as `tokio::time::sleep` treats it.
fn deadline_from_now(total: Duration) -> tokio::time::Instant {
    let now = tokio::time::Instant::now();
    now.checked_add(total)
        .unwrap_or_else(|| now + Duration::from_secs(86_400 * 365 * 30))
}

impl TimedBody {
    fn new(inner: Body, timeouts: RequestBodyTimeouts, outcome: Arc<AtomicU8>) -> Self {
        Self {
            inner,
            stall: timeouts.stall,
            stall_timer: None,
            deadline_at: deadline_from_now(timeouts.total),
            deadline: None,
            outcome,
            ended: false,
        }
    }

    fn end(
        &mut self,
        outcome: u8,
        reason: &'static str,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        self.ended = true;
        self.outcome.store(outcome, Ordering::Release);
        Poll::Ready(Some(Err(axum::Error::new(reason))))
    }
}

impl http_body::Body for TimedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // `Body` is `Unpin` (it is `UnsyncBoxBody`), and so are the boxed
        // timers, so `get_mut` is sound here.
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Pending => {}
            Poll::Ready(frame) => {
                this.stall_timer = None;
                if matches!(&frame, Some(Err(err)) if crossed_payload_limit(err)) {
                    this.outcome.store(TOO_LARGE, Ordering::Release);
                }
                return Poll::Ready(frame);
            }
        }
        let deadline_at = this.deadline_at;
        let deadline = this
            .deadline
            .get_or_insert_with(|| Box::pin(tokio::time::sleep_until(deadline_at)));
        if deadline.as_mut().poll(cx).is_ready() {
            return this.end(
                TIMED_OUT,
                "request body still incomplete at the request timeout",
            );
        }
        if let Some(stall) = this.stall {
            let timer = this
                .stall_timer
                .get_or_insert_with(|| Box::pin(tokio::time::sleep(stall)));
            if timer.as_mut().poll(cx).is_ready() {
                return this.end(
                    STALLED,
                    "request body stalled: no bytes from the client within the stall timeout",
                );
            }
        }
        Poll::Pending
    }

    fn is_end_stream(&self) -> bool {
        self.ended || self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use axum::{body::to_bytes, Router};
    use futures::{stream, StreamExt};
    use tokio::time::Instant;
    use tower::ServiceExt;

    use super::{super::request_start::RequestStarted, *};
    use crate::middleware::test_support::{echo_request, echo_route, json_of, ECHO_PATH};

    fn app(timeouts: RequestBodyTimeouts) -> Router {
        Router::new()
            .route(ECHO_PATH, echo_route())
            .layer(timeouts.layer())
    }

    fn timeouts(stall_secs: u64, total_secs: u64) -> RequestBodyTimeouts {
        RequestBodyTimeouts {
            stall: (stall_secs > 0).then(|| Duration::from_secs(stall_secs)),
            total: Duration::from_secs(total_secs),
        }
    }

    /// One byte per `interval`, `frames` times, then the end of the body.
    fn trickle(frames: usize, interval: Duration) -> Body {
        Body::from_stream(stream::unfold(0usize, move |sent| async move {
            if sent == frames {
                return None;
            }
            tokio::time::sleep(interval).await;
            Some((Ok::<_, Infallible>(Bytes::from_static(b"x")), sent + 1))
        }))
    }

    async fn error_code(response: Response) -> String {
        let json = json_of(response).await;
        json["error"]["code"].as_str().unwrap().to_string()
    }

    #[tokio::test(start_paused = true)]
    async fn a_complete_body_passes_through_untouched() {
        let response = app(timeouts(5, 60))
            .oneshot(echo_request(Body::from("payload")))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&body[..], b"payload");
    }

    /// A body that stops after its first bytes ends with 408 once a single
    /// client wait reaches the stall timeout, long before the request
    /// timeout.
    #[tokio::test(start_paused = true)]
    async fn a_stalled_body_is_408_request_body_stalled() {
        let body = Body::from_stream(
            stream::iter([Ok::<_, Infallible>(Bytes::from_static(b"{\"partial\":"))])
                .chain(stream::pending()),
        );
        let started = Instant::now();
        let response = app(timeouts(5, 120))
            .oneshot(echo_request(body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(error_code(response).await, REQUEST_BODY_STALLED);
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_secs(5) && waited < Duration::from_secs(6),
            "ended after {waited:?}, expected the 5 s stall timeout"
        );
    }

    /// A trickle whose waits each stay under the stall timeout is still cut
    /// at the request timeout.
    #[tokio::test(start_paused = true)]
    async fn a_trickling_body_is_cut_at_the_request_timeout() {
        let started = Instant::now();
        let response = app(timeouts(5, 20))
            .oneshot(echo_request(trickle(usize::MAX, Duration::from_secs(1))))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(error_code(response).await, REQUEST_BODY_TIMEOUT);
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_secs(20) && waited < Duration::from_secs(22),
            "ended after {waited:?}, expected the 20 s request timeout"
        );
    }

    /// The deadline is armed where this layer is reached, inside the
    /// admission layers: time the request spent before it (a wait for a
    /// permit; here an entry stamp 15 s in the past) is not the client's
    /// budget, which stays the whole request timeout.
    #[tokio::test(start_paused = true)]
    async fn time_before_this_layer_leaves_the_request_timeout_whole() {
        let started = Instant::now();
        let earlier = started
            .into_std()
            .checked_sub(Duration::from_secs(15))
            .expect("the clock has been running for fifteen seconds");
        let request = echo_request(trickle(usize::MAX, Duration::from_secs(1)));
        let response = RequestStarted::at(earlier)
            .scope(app(timeouts(5, 20)).oneshot(request))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(error_code(response).await, REQUEST_BODY_TIMEOUT);
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_secs(20) && waited < Duration::from_secs(22),
            "ended after {waited:?}, expected the whole 20 s request timeout from this layer"
        );
    }

    /// The request timeout counts from the wrap, where this layer is
    /// reached, not from the first read of the body: a handler that reads
    /// late is cut at the timeout the wrap started.
    #[tokio::test(start_paused = true)]
    async fn the_request_timeout_counts_from_the_wrap_not_from_the_first_read() {
        let wrapped_at = Instant::now();
        let (request, outcome) = timed(
            echo_request(trickle(usize::MAX, Duration::from_secs(1))),
            timeouts(0, 5),
        );
        tokio::time::sleep(Duration::from_secs(3)).await;
        let read = to_bytes(request.into_body(), usize::MAX).await;
        assert!(read.is_err(), "the body ends with the timeout's error");
        assert_eq!(outcome.load(Ordering::Acquire), TIMED_OUT);
        let waited = wrapped_at.elapsed();
        assert!(
            waited >= Duration::from_secs(5) && waited < Duration::from_secs(6),
            "cut {waited:?} after the wrap: the 5 s timeout counted from the wrap, not from the first read at 3 s"
        );
    }

    /// A `request_timeout_secs` no instant can hold (the "no timeout" idiom;
    /// validation rejects only zero) is a deadline that never comes, as the
    /// replaced `sleep` saturated to, not an overflow at the wrap.
    #[tokio::test(start_paused = true)]
    async fn a_total_no_instant_can_hold_is_a_deadline_that_never_comes() {
        let (request, outcome) = timed(
            echo_request(trickle(3, Duration::from_secs(86_400 * 365))),
            timeouts(0, u64::MAX),
        );
        let read = to_bytes(request.into_body(), usize::MAX).await;
        assert_eq!(
            read.expect("three frames a year apart arrive whole"),
            Bytes::from_static(b"xxx")
        );
        assert_ne!(outcome.load(Ordering::Acquire), TIMED_OUT);
    }

    /// Progress inside the stall window resets the watchdog: a slow but
    /// moving upload that completes before the request timeout is served.
    #[tokio::test(start_paused = true)]
    async fn a_slow_but_moving_body_is_served() {
        let response = app(timeouts(5, 60))
            .oneshot(echo_request(trickle(4, Duration::from_secs(3))))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(&body[..], b"xxxx");
    }

    /// With the stall watchdog off (0), only the request timeout bounds the
    /// read.
    #[tokio::test(start_paused = true)]
    async fn stall_watchdog_off_leaves_the_request_timeout() {
        let body = Body::from_stream(stream::pending::<Result<Bytes, Infallible>>());
        let started = Instant::now();
        let response = app(timeouts(0, 10))
            .oneshot(echo_request(body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(error_code(response).await, REQUEST_BODY_TIMEOUT);
        assert!(started.elapsed() >= Duration::from_secs(10));
    }

    /// A read the payload limit cut short is 413 request_body_too_large,
    /// whatever the handler made of the failed read: the limit layer sits
    /// outside the watchdog, so its error reaches the handler boxed once
    /// more, past where the extractors look for it.
    #[tokio::test(start_paused = true)]
    async fn a_body_over_the_payload_limit_is_413_request_body_too_large() {
        let app = app(timeouts(5, 60)).layer(crate::middleware::RequestBodyLimitLayer::new(8));
        let response = app
            .oneshot(echo_request(Body::from("more than eight bytes")))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(error_code(response).await, "request_body_too_large");
    }

    /// Any other failed read keeps the handler's own answer.
    #[tokio::test(start_paused = true)]
    async fn another_failed_read_keeps_the_handlers_answer() {
        let body = Body::from_stream(stream::iter([Err::<Bytes, _>(std::io::Error::other(
            "connection reset",
        ))]));
        let response = app(timeouts(5, 60))
            .oneshot(echo_request(body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
