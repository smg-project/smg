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
    upload_wait: UploadWait,
}

/// `now + total`, saturated to a far future when the sum does not fit an
/// instant: seconds too large to hold (the "no timeout" idiom of
/// `request_timeout_secs`, a shutdown grace written as the largest number)
/// are a deadline that never comes, as `tokio::time::sleep` treats them.
pub(crate) fn deadline_from_now(total: Duration) -> tokio::time::Instant {
    deadline_after(tokio::time::Instant::now(), total)
}

/// [`deadline_from_now`] from a `now` the caller has read.
fn deadline_after(now: tokio::time::Instant, total: Duration) -> tokio::time::Instant {
    now.checked_add(total)
        .unwrap_or_else(|| now + Duration::from_secs(86_400 * 365 * 30))
}

/// How long one read of a request body waited for the client: from its first
/// poll that found the body pending to where the reader drops the body. The
/// drop adds it to the request's stamp (`RequestStarted::upload_waited`), for
/// the dispatch wait counted without the upload. A body that never waited
/// reads no clock and writes nothing.
#[derive(Default)]
struct UploadWait(Option<tokio::time::Instant>);

impl UploadWait {
    /// A poll found the body pending. `Some(now)` on the first such poll,
    /// which starts the wait: its one clock read, handed back for a timer
    /// armed at the same poll.
    fn pending(&mut self) -> Option<tokio::time::Instant> {
        if self.0.is_some() {
            return None;
        }
        let now = tokio::time::Instant::now();
        self.0 = Some(now);
        Some(now)
    }
}

impl Drop for UploadWait {
    fn drop(&mut self) {
        if let Some(since) = self.0 {
            super::request_start::RequestStarted::upload_waited(since.elapsed());
        }
    }
}

/// `body` under the upload wait this layer's body measures, for a reader
/// outside the layer: the WASM request hook, which reads the body ahead of
/// it, and the multipart routes, which stand under no body deadline.
pub(crate) fn upload_measured(body: Body) -> Body {
    Body::new(UploadMeasured {
        inner: body,
        wait: UploadWait::default(),
    })
}

struct UploadMeasured {
    inner: Body,
    wait: UploadWait,
}

impl http_body::Body for UploadMeasured {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        if polled.is_pending() {
            this.wait.pending();
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
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
            upload_wait: UploadWait::default(),
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
        // The body waits for the client. Its first wait starts the upload
        // wait, on the clock read the stall timer of that wait is built on.
        let first_wait = this.upload_wait.pending();
        if let Some(stall) = this.stall {
            let timer = this.stall_timer.get_or_insert_with(|| {
                Box::pin(match first_wait {
                    Some(now) => tokio::time::sleep_until(deadline_after(now, stall)),
                    None => tokio::time::sleep(stall),
                })
            });
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
#[path = "../../tests/unit/middleware/body_timeout/tests.rs"]
mod tests;
