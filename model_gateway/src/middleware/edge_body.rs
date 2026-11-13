//! The one body wrapper of the HTTP edge.
//!
//! Four layers of the server's stack have a duty that outlives the handler
//! and ends with the response body: the HTTP metrics layer (the in-flight
//! guard, the response counted once its outcome is known), the request-span
//! layer (the span kept open until the body ends, the client-close line of
//! the access log), the admission layer (the permit held until the body
//! ends) and the SSE keep-alive (a frame after an interval of silence). Each
//! used to wrap the body in a boxed `Body` type of its own, so a streamed
//! frame crossed four boxed `poll_frame`s, and two of them evaluated the
//! "has the body ended" rule. The body is wrapped once instead: `EdgeBody`,
//! built by the metrics layer, the outermost of the four, which wraps every
//! response. The layers inside hand it what they used to keep in their own
//! wrappers through the response's extensions (`BodyDuties`) and wrap
//! nothing themselves.
//!
//! The hand-over rides on the response's extensions, so it holds across the
//! layers that keep them. A layer between a producer and the edge that
//! builds the response again from scratch (the WASM middleware's reject and
//! modify phases do, with `Response::builder()`) drops the duties handed so
//! far with the old response: the permit and the keep-alive end there, as
//! their in-body wrappers did when such a layer drained the old body, and
//! the span closes with the head (`BodyDuties`' drop). A layer that
//! replaces the body but keeps the parts (the Messages error envelope) keeps
//! them. A layer that drops the parts but keeps a body whose producer
//! carried the cell (`BodyFailure::carry`) leaves the duties in that cell
//! until the body goes.
//!
//! The duties end as they did. At the body's end (its last frame, the length
//! the response declared, or an error of the body) the response is counted,
//! the in-flight guard and the permit are released and the span closes; a
//! body dropped before that is a client that closed the connection, counted
//! and logged as such; a body already complete when it reaches the edge is
//! finished at once, as it may never be polled.

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, PoisonError,
    },
    task::{ready, Context, Poll},
    time::Instant,
};

use axum::{
    body::{Body, Bytes},
    response::Response,
};
use http_body::{Body as HttpBody, Frame, SizeHint};

use super::{
    logging::SpanDuty,
    metrics::{BodyProgress, ResponseOutcome},
    sse_keepalive::KeepAlive,
};
use crate::observability::{
    inflight_tracker::{InFlightGuard, InFlightRequestTracker},
    metrics::Metrics,
};

/// An admission permit the body carries: released, by its drop, when the
/// body ends or is dropped, and told of the body's first data frame.
pub(crate) trait Permit: Send {
    /// The body's first data frame (a keep-alive frame counts) is about to go
    /// out. `false` ends the body here instead, without the frame: the
    /// priority scheduler preempted the request before its first byte. The
    /// default lets the frame pass.
    fn first_data_frame(&mut self) -> bool {
        true
    }
}

/// What the layers inside the edge hand the edge's one body wrapper.
#[derive(Default)]
pub(crate) struct BodyDuties {
    /// The admission permit: the token bucket's or the priority scheduler's.
    pub(crate) permit: Option<Box<dyn Permit>>,
    /// The SSE keep-alive: a frame after an interval of silence.
    pub(crate) keep_alive: Option<KeepAlive>,
    /// When the response head left the handler: the access-log layer's
    /// stamp, which the metrics layer's duration reads instead of the clock.
    pub(crate) head_at: Option<Instant>,
    /// The request span, open until the body ends, with the client-close
    /// line the access log owes if the body is dropped first.
    pub(crate) span: Option<SpanDuty>,
    /// The response's cell, when the body's producer holds it for its word
    /// on a streamed body that tells the client of a failure and then ends
    /// as planned (see [`BodyFailure`]): set by [`take`](Self::take) alone,
    /// read at the body's end, so the response counts as what the client
    /// read. Private: a word a layer handed through [`hand`](Self::hand)
    /// would be overwritten by `take`'s shared-cell arm, and a carried cell
    /// handed as well would sit inside its own mutex, a cycle only `take`
    /// breaks. [`carry`](BodyFailure::carry) is the one way in.
    failure: Option<BodyFailure>,
}

#[cfg(test)]
impl BodyDuties {
    /// The cell taken out, for the tests that read the word through it.
    pub(crate) fn take_failure(&mut self) -> Option<BodyFailure> {
        self.failure.take()
    }
}

/// The one shared cell of a response: the [`BodyDuties`] on their way out to
/// the metrics layer, and the producer's word on the body ([`BodyFailure`]).
/// `http::Extensions` holds `Clone` values only, so the duties, none of them
/// clonable, travel behind a lock, and the metrics layer takes them out
/// before the response reaches the server; the word is an atomic the
/// producer sets from the body and the edge reads at the body's end.
#[derive(Default)]
struct DutyCell {
    duties: Mutex<BodyDuties>,
    failed: AtomicBool,
}

/// The response extension that carries the [`DutyCell`].
#[derive(Clone, Default)]
struct DutySlot(Arc<DutyCell>);

impl BodyDuties {
    /// Hands the edge's body wrapper a duty through `response`'s extensions:
    /// `fill` sets it on the duties handed over so far.
    pub(crate) fn hand(response: &mut Response, fill: impl FnOnce(&mut Self)) {
        let slot = response
            .extensions_mut()
            .get_or_insert_default::<DutySlot>();
        fill(&mut slot.0.duties.lock().unwrap_or_else(PoisonError::into_inner));
    }

    /// Takes the duties handed over for `response`; none when no layer
    /// inside had one. A cell the body's producer still holds, for its word,
    /// stays shared and comes along as the duties' `failure`.
    pub(crate) fn take(response: &mut Response) -> Self {
        let Some(DutySlot(cell)) = response.extensions_mut().remove::<DutySlot>() else {
            return Self::default();
        };
        match Arc::try_unwrap(cell) {
            Ok(cell) => cell
                .duties
                .into_inner()
                .unwrap_or_else(PoisonError::into_inner),
            Err(shared) => {
                let mut duties = std::mem::take(
                    &mut *shared.duties.lock().unwrap_or_else(PoisonError::into_inner),
                );
                duties.failure = Some(BodyFailure(shared));
                duties
            }
        }
    }
}

/// The word of a streamed body's producer that the body told the client of a
/// failure on its way out: the error event a stream the worker failed ends
/// with, before its `[DONE]`. Such a body ends with its own last frame, so
/// the frames alone say it went out in full; counted by them, a stream the
/// engine cut was a plain 200. The word lives in the response's one cell,
/// with no allocation of its own: the producer builds the cell with its
/// body, puts it on the response ([`carry`](Self::carry)) for the layers
/// outside to hand their duties into, and tells the word from the body; the
/// edge reads it once the body has ended, and the response then counts as
/// its head's status with [`BODY_FAILED_CODE`], what the client read. A body
/// with no producer's word is counted by its frames alone.
///
/// [`BODY_FAILED_CODE`]: super::metrics::BODY_FAILED_CODE
#[derive(Clone, Default)]
pub(crate) struct BodyFailure(Arc<DutyCell>);

impl BodyFailure {
    /// Makes the cell this word lives in `response`'s one: the layers
    /// outside hand their duties into it, the edge reads the word from it.
    /// A duty handed inside before this moves over.
    pub(crate) fn carry(&self, response: &mut Response) {
        let inside = response
            .extensions_mut()
            .insert(DutySlot(Arc::clone(&self.0)));
        if let Some(DutySlot(inside)) = inside {
            let handed =
                std::mem::take(&mut *inside.duties.lock().unwrap_or_else(PoisonError::into_inner));
            *self.0.duties.lock().unwrap_or_else(PoisonError::into_inner) = handed;
        }
    }

    /// The producer's word: the body ended with the client told of a failure.
    pub(crate) fn tell(&self) {
        self.0.failed.store(true, Ordering::Release);
    }

    /// Whether the producer told of a failure.
    pub(crate) fn told(&self) -> bool {
        self.0.failed.load(Ordering::Acquire)
    }
}

impl Drop for BodyDuties {
    fn drop(&mut self) {
        // Duties still here were taken by no edge body: the response left a
        // stack with no metrics layer outside, or a layer between the
        // producer and the edge built the response again without its
        // extensions (see the module doc). The span then closes with the
        // response head, and its close line stays unwritten rather than
        // claiming a client that closed.
        if let Some(span) = &mut self.span {
            span.unpolled();
        }
    }
}

/// The edge's response body: the handler's body and every duty that ends
/// with it, under one `poll_frame`.
pub(crate) struct EdgeBody {
    inner: Body,
    /// The in-flight guard, until the body ends.
    guard: Option<InFlightGuard>,
    tracker: Arc<InFlightRequestTracker>,
    /// The response's count in `smg_http_responses_total`, recorded once.
    outcome: ResponseOutcome,
    /// The declared length and the bytes out: the end-of-body rule.
    progress: BodyProgress,
    span: Option<SpanDuty>,
    permit: Option<Box<dyn Permit>>,
    keep_alive: Option<KeepAlive>,
    /// The response's cell, when the body's producer holds it for its word
    /// on how the body ended.
    failure: Option<BodyFailure>,
    /// Whether a frame was ever asked for. A body dropped before any poll is
    /// one the server never meant to send (the answer to a HEAD request), not
    /// one the client cut short.
    polled: bool,
    /// The first data frame went out: the permit's one-time hook is done.
    first_data_out: bool,
    /// Ended by the wrapper itself, on a permit that refused the first frame:
    /// stays ended, the inner body is not polled again.
    ended: bool,
}

impl EdgeBody {
    pub(super) fn new(
        inner: Body,
        guard: InFlightGuard,
        tracker: Arc<InFlightRequestTracker>,
        outcome: ResponseOutcome,
        progress: BodyProgress,
        mut duties: BodyDuties,
    ) -> Self {
        let mut body = Self {
            inner,
            guard: Some(guard),
            tracker,
            outcome,
            progress,
            span: duties.span.take(),
            permit: duties.permit.take(),
            keep_alive: duties.keep_alive.take(),
            failure: duties.failure.take(),
            polled: false,
            first_data_out: false,
            ended: false,
        };
        // An already-complete body (an empty unary response) may never be
        // polled at all: it is finished now, and counted.
        if body.inner.is_end_stream() {
            body.finished_unpolled();
        }
        body
    }

    fn release(&mut self) {
        if let Some(guard) = self.guard.take() {
            // Drop before reading len() so the gauge reflects the decrement.
            drop(guard);
            Metrics::set_http_connections_active(self.tracker.len());
        }
    }

    /// The body reached its end: the response is out and counts as its
    /// head's status, or as a body that failed when its producer told the
    /// client of a failure on the way; everything held for the body goes.
    fn finished(&mut self) {
        self.release();
        if self.failure.as_ref().is_some_and(BodyFailure::told) {
            self.outcome.body_failed();
        } else {
            self.outcome.completed();
        }
        if let Some(span) = &mut self.span {
            span.ended(self.progress.sent());
        }
        self.permit = None;
        self.keep_alive = None;
    }

    /// A body complete before anyone polled it: counted as the response it
    /// was, the span closed with the head, no end line (nothing streamed).
    fn finished_unpolled(&mut self) {
        self.release();
        self.outcome.completed();
        if let Some(span) = &mut self.span {
            span.unpolled();
        }
        self.permit = None;
        self.keep_alive = None;
    }

    /// The body failed after its head: the response ends here, truncated,
    /// counted as its head's status with the failure named, and logged.
    fn failed(&mut self, error: &axum::Error) {
        self.release();
        self.outcome.body_failed();
        if let Some(span) = &mut self.span {
            span.failed(error, self.progress.sent());
        }
        self.permit = None;
        self.keep_alive = None;
    }

    /// A data frame is about to go out: whether the permit lets it. Only
    /// the first one is asked about.
    fn data_frame_allowed(&mut self) -> bool {
        if self.first_data_out {
            return true;
        }
        self.first_data_out = true;
        self.permit
            .as_mut()
            .is_none_or(|permit| permit.first_data_frame())
    }

    /// Preempted before its first byte: the body ends here, without the
    /// frame, and stays ended.
    fn end_before_first_byte(&mut self) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        self.ended = true;
        self.finished();
        Poll::Ready(None)
    }
}

impl HttpBody for EdgeBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // `Body` is `Unpin` (it is `UnsyncBoxBody`), and so is everything
        // else held here, so `get_mut` is sound.
        let this = self.get_mut();
        this.polled = true;
        if this.ended {
            return Poll::Ready(None);
        }
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    if !this.data_frame_allowed() {
                        return this.end_before_first_byte();
                    }
                    if let Some(keep_alive) = &mut this.keep_alive {
                        keep_alive.frame_out(data);
                    }
                }
                // At its end: the last frame, or the declared length out.
                // Under the http_body contract a consumer may stop polling
                // there and never ask for the None (an HTTP/1.1 server stops
                // at the declared length).
                if this.progress.frame_out(&frame, this.inner.is_end_stream()) {
                    this.finished();
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(None) => {
                this.finished();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                this.failed(&error);
                Poll::Ready(Some(Err(error)))
            }
            Poll::Pending => {
                // The body is quiet: the keep-alive's turn, when there is one.
                let Some(keep_alive) = &mut this.keep_alive else {
                    return Poll::Pending;
                };
                let frame = ready!(keep_alive.poll_silence(cx));
                if !this.data_frame_allowed() {
                    return this.end_before_first_byte();
                }
                // The keep-alive's bytes reach the client like any frame's:
                // counted in the access log's bytes, as they were. A
                // keep-alive rides only a body with no declared length, so
                // this is never the body's end.
                let frame = Frame::data(frame);
                if this.progress.frame_out(&frame, false) {
                    this.finished();
                }
                Poll::Ready(Some(Ok(frame)))
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.ended || self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for EdgeBody {
    fn drop(&mut self) {
        self.release();
        if self.polled {
            // Dropped before its end: the server stopped writing the body
            // because the client closed the connection.
            self.outcome.client_closed();
            if let Some(span) = &mut self.span {
                span.client_closed(self.progress.sent());
            }
        } else {
            // A body that was never polled was not cut short; it counts as
            // the response it was.
            self.outcome.completed();
            if let Some(span) = &mut self.span {
                span.unpolled();
            }
        }
        // The permit, held this long, goes with the fields.
    }
}

/// What the metrics layer does to a response, for the tests of the layers
/// that hand it a duty: the duties taken out and the body wrapped once.
#[cfg(test)]
pub(crate) fn at_the_edge(mut response: Response) -> Response {
    let tracker = InFlightRequestTracker::new();
    let guard = tracker.track();
    let mut outcome = ResponseOutcome::pending(super::metrics::PathLabel::Static("/test"));
    outcome.head(&response);
    let progress = BodyProgress::new(&response);
    let duties = BodyDuties::take(&mut response);
    response.map(|body| {
        Body::new(EdgeBody::new(
            body, guard, tracker, outcome, progress, duties,
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::{
        convert::Infallible,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use axum::{
        http::{header::CONTENT_TYPE, Request, StatusCode},
        routing::get,
        Router,
    };
    use http_body_util::BodyExt;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use tower::ServiceExt;
    use tracing::Level;

    use super::*;
    use crate::{
        middleware::{create_logging_layer, AccessLogFields, HttpMetricsLayer, SseKeepAlive},
        observability::logging::test_support::{test_subscriber, ResponseLines},
    };

    /// A permit that records its release and refuses the first frame when
    /// told to.
    struct TestPermit {
        released: Arc<Mutex<bool>>,
        lets_first_frame_pass: bool,
    }

    impl Permit for TestPermit {
        fn first_data_frame(&mut self) -> bool {
            self.lets_first_frame_pass
        }
    }

    impl Drop for TestPermit {
        fn drop(&mut self) {
            *self.released.lock().expect("released") = true;
        }
    }

    /// The server's edge over `/stream`, which hands out `body` once as an
    /// SSE response with a permit: the admission and keep-alive layers inside,
    /// the request span and the metrics layer outside, in the server's order.
    fn app(body: Body, permit: TestPermit, tracker: Arc<InFlightRequestTracker>) -> Router {
        let slot = Arc::new(Mutex::new(Some((body, permit))));
        Router::new()
            .route(
                "/v1/chat/completions",
                get(move || {
                    let (body, permit) = slot
                        .lock()
                        .expect("slot")
                        .take()
                        .expect("handler called once");
                    async move {
                        let mut response = Response::builder()
                            .header(CONTENT_TYPE, "text/event-stream")
                            .body(body)
                            .expect("response");
                        BodyDuties::hand(&mut response, |duties| {
                            duties.permit = Some(Box::new(permit));
                        });
                        response
                    }
                }),
            )
            .route_layer(SseKeepAlive::every_secs(15).layer())
            .layer(create_logging_layer(AccessLogFields::NONE))
            .layer(HttpMetricsLayer::new(tracker))
    }

    fn request() -> Request<Body> {
        Request::builder()
            .uri("/v1/chat/completions")
            .body(Body::empty())
            .expect("request")
    }

    fn permit(released: &Arc<Mutex<bool>>, lets_first_frame_pass: bool) -> TestPermit {
        TestPermit {
            released: released.clone(),
            lets_first_frame_pass,
        }
    }

    fn line(level: Level, message: &str) -> (Level, String) {
        (level, message.to_owned())
    }

    #[tokio::test(start_paused = true)]
    async fn every_duty_ends_with_the_body() {
        let sink = ResponseLines::default();
        let _subscriber = test_subscriber(sink.clone());
        let tracker = InFlightRequestTracker::new();
        let released = Arc::new(Mutex::new(false));
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let response = app(
            Body::from_stream(ReceiverStream::new(rx)),
            permit(&released, true),
            tracker.clone(),
        )
        .oneshot(request())
        .await
        .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body();

        // The handler has answered, the body is open: in flight, the permit
        // held, the access log at its response line.
        assert_eq!(tracker.len(), 1);
        assert!(!*released.lock().expect("released"));
        tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
            .await
            .expect("send");
        let frame = body.frame().await.expect("a frame").expect("frame ok");
        assert_eq!(
            frame.data_ref().map(|data| &data[..]),
            Some(&b"data: x\n\n"[..])
        );
        assert_eq!(tracker.len(), 1);
        assert!(!*released.lock().expect("released"));

        // Silence: the keep-alive comment comes through the one wrapper.
        let next = tokio::time::timeout(Duration::from_secs(16), body.frame())
            .await
            .expect("a keep-alive within the interval")
            .expect("a frame")
            .expect("frame ok");
        assert_eq!(
            next.data_ref().map(|data| &data[..]),
            Some(&b": ping\n\n"[..])
        );

        // The end of the stream: everything goes, the response is complete.
        drop(tx);
        assert!(body.frame().await.is_none());
        assert_eq!(tracker.len(), 0);
        assert!(*released.lock().expect("released"));
        assert_eq!(
            sink.lines(),
            vec![line(Level::INFO, "finished processing request")]
        );
    }

    #[tokio::test]
    async fn a_body_the_client_dropped_mid_stream_releases_every_duty_and_logs_the_close_once() {
        let sink = ResponseLines::default();
        let _subscriber = test_subscriber(sink.clone());
        let tracker = InFlightRequestTracker::new();
        let released = Arc::new(Mutex::new(false));
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(4);
        let response = app(
            Body::from_stream(ReceiverStream::new(rx)),
            permit(&released, true),
            tracker.clone(),
        )
        .oneshot(request())
        .await
        .expect("response");
        tx.send(Ok(Bytes::from_static(b"data: x\n\n")))
            .await
            .expect("send");
        let mut body = response.into_body();
        body.frame().await.expect("a frame").expect("frame ok");
        drop(body);
        assert_eq!(tracker.len(), 0);
        assert!(*released.lock().expect("released"));
        assert_eq!(
            sink.lines(),
            vec![
                line(Level::INFO, "finished processing request"),
                line(
                    Level::INFO,
                    "client closed the connection before the response stream ended"
                ),
            ]
        );
    }

    #[tokio::test]
    async fn a_permit_that_refuses_the_first_frame_ends_the_body_without_it() {
        let tracker = InFlightRequestTracker::new();
        let released = Arc::new(Mutex::new(false));
        let response = app(
            Body::from("data: never\n\n"),
            permit(&released, false),
            tracker.clone(),
        )
        .oneshot(request())
        .await
        .expect("response");
        let mut body = response.into_body();
        assert!(!body.is_end_stream());
        assert!(
            body.frame().await.is_none(),
            "the refused frame is not sent"
        );
        assert!(body.is_end_stream(), "the body stays ended");
        assert!(body.frame().await.is_none());
        assert_eq!(tracker.len(), 0);
        assert!(*released.lock().expect("released"));
    }

    #[tokio::test]
    async fn an_already_complete_body_is_finished_at_the_edge() {
        let tracker = InFlightRequestTracker::new();
        let released = Arc::new(Mutex::new(false));
        let response = app(Body::empty(), permit(&released, true), tracker.clone())
            .oneshot(request())
            .await
            .expect("response");
        assert_eq!(tracker.len(), 0);
        assert!(*released.lock().expect("released"));
        drop(response);
    }

    /// A stack with no metrics layer outside takes no duties: the span ends
    /// with the response head, without a client-close line for a stream
    /// nobody cut.
    #[tokio::test]
    async fn duties_nobody_took_keep_the_access_log_quiet() {
        let sink = ResponseLines::default();
        let _subscriber = test_subscriber(sink.clone());
        let (_tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(1);
        let slot = Arc::new(Mutex::new(Some(Body::from_stream(ReceiverStream::new(rx)))));
        let app = Router::new()
            .route(
                "/stream",
                get(move || {
                    let body = slot.lock().expect("slot").take().expect("once");
                    async move { Response::new(body) }
                }),
            )
            .layer(create_logging_layer(AccessLogFields::NONE));
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/stream")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let (parts, body) = response.into_parts();
        drop(parts);
        assert_eq!(
            sink.lines(),
            vec![line(Level::INFO, "finished processing request")]
        );
        drop(body);
    }

    /// The producer's word and the duties share the response's one cell,
    /// whichever reached the response first: the duties come out at the
    /// edge with the word's cell, and the word the body tells is the word
    /// the edge reads.
    #[test]
    fn the_producers_word_and_the_duties_share_one_cell() {
        for duty_first in [true, false] {
            let mut response = Response::new(Body::empty());
            let head_at = Instant::now();
            let word = BodyFailure::default();
            if duty_first {
                BodyDuties::hand(&mut response, |duties| duties.head_at = Some(head_at));
                word.carry(&mut response);
            } else {
                word.carry(&mut response);
                BodyDuties::hand(&mut response, |duties| duties.head_at = Some(head_at));
            }
            let mut duties = BodyDuties::take(&mut response);
            assert_eq!(duties.head_at, Some(head_at), "duty first: {duty_first}");
            let edge = duties.failure.take().expect("the cell comes along");
            assert!(!edge.told(), "duty first: {duty_first}");
            word.tell();
            assert!(edge.told(), "duty first: {duty_first}");
        }
    }

    /// A response whose producer gave no word hands the edge its duties
    /// alone: nothing to read at the body's end.
    #[test]
    fn a_response_without_a_producers_word_hands_the_edge_no_cell() {
        let mut response = Response::new(Body::empty());
        let head_at = Instant::now();
        BodyDuties::hand(&mut response, |duties| duties.head_at = Some(head_at));
        let duties = BodyDuties::take(&mut response);
        assert_eq!(duties.head_at, Some(head_at));
        assert!(duties.failure.is_none());
    }
}
