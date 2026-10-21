//! A keep-alive on SSE streams whose upstream has gone quiet.
//!
//! A streamed response is answered with its headers as soon as the request
//! is dispatched; its first token may follow a long prefill tens of seconds
//! later, and nothing was sent in between. An idle-timeout proxy in front of
//! the gateway cuts a connection that carries no byte for its idle window
//! (sixty seconds is a common setting), and the client sees a stream that
//! ends before it began. [`sse_keepalive_middleware`] gives every SSE
//! response a [`KeepAlive`], carried by the edge's one body wrapper (the
//! metrics layer's [`EdgeBody`](super::edge_body::EdgeBody)), so an interval
//! of silence is broken by a frame the client ignores: an SSE comment on the
//! OpenAI-shaped routes, the `ping` event the Messages API specifies on
//! `/v1/messages`. The interval runs from the headers and from every frame
//! sent, so a flowing stream carries no keep-alive; a keep-alive is only
//! ever written between frames, never inside one.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    body::Body,
    extract::{Request, State},
    http::header::{CONTENT_LENGTH, CONTENT_TYPE},
    middleware::Next,
    response::Response,
};
use bytes::Bytes;
use http_body::Body as HttpBody;
use pin_project_lite::pin_project;
use tokio::time::{Instant, Sleep};
use tower::{Layer, Service};

use super::edge_body::BodyDuties;
use crate::{
    config::types::RouterConfig,
    routers::http::stream_failure::{note_sent, on_frame_boundary},
};

/// The keep-alive of an OpenAI-shaped stream: an SSE comment, which every
/// event-stream parser skips.
const COMMENT: &[u8] = b": ping\n\n";
/// The keep-alive of a Messages stream: the API's `ping` event, which its
/// clients accept at any point of a stream.
const MESSAGES_PING: &[u8] = b"event: ping\ndata: {\"type\":\"ping\"}\n\n";
/// The route whose streams carry the Messages API's ping.
const MESSAGES_ROUTE: &str = "/v1/messages";

/// The keep-alive interval of every SSE response, from the router config.
#[derive(Clone, Copy, Debug)]
pub struct SseKeepAlive {
    /// `None` sends no keep-alive (`sse_keepalive_secs` 0).
    interval: Option<Duration>,
}

impl SseKeepAlive {
    pub fn from_config(config: &RouterConfig) -> Self {
        Self::every_secs(config.sse_keepalive_secs)
    }

    /// A keep-alive after `secs` of silence; 0 sends none.
    pub fn every_secs(secs: u64) -> Self {
        Self {
            interval: (secs > 0).then(|| Duration::from_secs(secs)),
        }
    }

    /// Gives an SSE `response` whose length is not known up front (a
    /// complete SSE answer has nothing to wait for, and a body sent under a
    /// `Content-Length` header must not grow past it: the relays copy an
    /// upstream's header onto a streamed body) its keep-alive, handed to the
    /// edge's body wrapper; `messages` selects the Messages API's ping over
    /// the comment. Anything else is returned as it came.
    pub(crate) fn apply(self, mut response: Response, messages: bool) -> Response {
        let Some(interval) = self.interval else {
            return response;
        };
        if !is_event_stream(&response)
            || response.headers().contains_key(CONTENT_LENGTH)
            || response.body().size_hint().exact().is_some()
        {
            return response;
        }
        let frame = Bytes::from_static(if messages { MESSAGES_PING } else { COMMENT });
        BodyDuties::hand(&mut response, |duties| {
            duties.keep_alive = Some(KeepAlive::new(interval, frame));
        });
        response
    }
}

fn is_event_stream(response: &Response) -> bool {
    response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|content_type| content_type.trim_start().get(.."text/event-stream".len()))
        .is_some_and(|head| head.eq_ignore_ascii_case("text/event-stream"))
}

impl SseKeepAlive {
    /// This keep-alive as a route layer: the service's own future maps the
    /// response, nothing is cloned or boxed per request (the `from_fn` form
    /// did both, on every request of the routes, streamed or not).
    pub fn layer(self) -> SseKeepAliveLayer {
        SseKeepAliveLayer { keep_alive: self }
    }
}

/// [`SseKeepAlive::layer`].
#[derive(Clone, Copy, Debug)]
pub struct SseKeepAliveLayer {
    keep_alive: SseKeepAlive,
}

impl<S> Layer<S> for SseKeepAliveLayer {
    type Service = SseKeepAliveService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        SseKeepAliveService {
            inner,
            keep_alive: self.keep_alive,
        }
    }
}

/// The service [`SseKeepAliveLayer`] wraps a route in.
#[derive(Clone, Debug)]
pub struct SseKeepAliveService<S> {
    inner: S,
    keep_alive: SseKeepAlive,
}

impl<S, E> Service<Request<Body>> for SseKeepAliveService<S>
where
    S: Service<Request<Body>, Response = Response, Error = E>,
{
    type Response = Response;
    type Error = E;
    type Future = KeepAliveFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request<Body>) -> Self::Future {
        let messages = request.uri().path() == MESSAGES_ROUTE;
        KeepAliveFuture {
            inner: self.inner.call(request),
            keep_alive: self.keep_alive,
            messages,
        }
    }
}

pin_project! {
    /// The inner service's future, its response given the keep-alive.
    pub struct KeepAliveFuture<F> {
        #[pin]
        inner: F,
        keep_alive: SseKeepAlive,
        messages: bool,
    }
}

impl<F, E> Future for KeepAliveFuture<F>
where
    F: Future<Output = Result<Response, E>>,
{
    type Output = Result<Response, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let response = std::task::ready!(this.inner.poll(cx))?;
        Poll::Ready(Ok(this.keep_alive.apply(response, *this.messages)))
    }
}

/// Route-layer middleware: the keep-alive on every SSE response of the
/// routes it wraps ([`SseKeepAlive::layer`] is the same without the per-request
/// clone and boxes of `from_fn`).
pub async fn sse_keepalive_middleware(
    State(keep_alive): State<SseKeepAlive>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let messages = request.uri().path() == MESSAGES_ROUTE;
    keep_alive.apply(next.run(request).await, messages)
}

/// The keep-alive of one SSE body: `frame` after `interval` of silence, on a
/// frame boundary. The edge's body wrapper tells it of every frame sent and
/// polls it while the body is quiet. The timer is lazy: when it fires it is
/// re-armed from the time of the last frame rather than reset on every
/// frame, so a flowing stream costs one timer wake per interval, not one per
/// token.
pub(crate) struct KeepAlive {
    interval: Duration,
    frame: Bytes,
    /// When the headers, the last frame or the last keep-alive went out.
    last_sent: Instant,
    /// The last bytes sent, to tell a frame boundary from the inside of a
    /// frame.
    tail: Vec<u8>,
    timer: Pin<Box<Sleep>>,
}

impl KeepAlive {
    fn new(interval: Duration, frame: Bytes) -> Self {
        let now = Instant::now();
        Self {
            interval,
            frame,
            last_sent: now,
            tail: Vec::with_capacity(4),
            timer: Box::pin(tokio::time::sleep_until(now + interval)),
        }
    }

    /// A data frame of the body went out.
    pub(crate) fn frame_out(&mut self, data: &[u8]) {
        note_sent(&mut self.tail, data);
        self.last_sent = Instant::now();
    }

    /// The body is quiet: the keep-alive frame once `interval` has passed
    /// since the last frame, on a frame boundary; pending, with the timer
    /// armed, until then. The timer is due `interval` after the last frame:
    /// fired early (a frame went out since it was armed) it is re-armed for
    /// then; fired inside a frame it is re-armed for another interval and
    /// nothing is written.
    pub(crate) fn poll_silence(&mut self, cx: &mut Context<'_>) -> Poll<Bytes> {
        loop {
            if self.timer.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            let now = Instant::now();
            let due = self.last_sent + self.interval;
            if now < due {
                self.timer.as_mut().reset(due);
            } else if on_frame_boundary(&self.tail) {
                self.last_sent = now;
                self.timer.as_mut().reset(now + self.interval);
                return Poll::Ready(self.frame.clone());
            } else {
                self.timer.as_mut().reset(now + self.interval);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::{
        http::{header::CONTENT_TYPE, Method, StatusCode},
        routing::post,
        Router,
    };
    use http_body_util::BodyExt;
    use openai_protocol::messages::MessageStreamEvent;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use tower::ServiceExt;

    use super::*;
    use crate::{
        middleware::{edge_body::at_the_edge, HttpMetricsLayer},
        observability::inflight_tracker::InFlightRequestTracker,
    };

    const SECS: u64 = 15;
    type Chunk = Result<Bytes, std::io::Error>;

    /// What a read of the body gave within its time budget.
    #[derive(Debug, PartialEq)]
    enum Read {
        Frame(Bytes),
        End,
        Silence,
    }

    /// The next data frame within `secs` of (virtual) time.
    async fn next_within(body: &mut Body, secs: u64) -> Read {
        match tokio::time::timeout(Duration::from_secs(secs), body.frame()).await {
            Ok(Some(Ok(frame))) => frame.into_data().map_or(Read::End, Read::Frame),
            Ok(_) => Read::End,
            Err(_) => Read::Silence,
        }
    }

    fn sse(rx: mpsc::Receiver<Chunk>) -> Response {
        Response::builder()
            .header(CONTENT_TYPE, "text/event-stream; charset=utf-8")
            .body(Body::from_stream(ReceiverStream::new(rx)))
            .unwrap()
    }

    /// An SSE body under the keep-alive, at the edge, fed through the
    /// returned sender.
    fn wrapped(secs: u64, messages: bool) -> (mpsc::Sender<Chunk>, Body) {
        let (tx, rx) = mpsc::channel(16);
        let response = SseKeepAlive::every_secs(secs).apply(sse(rx), messages);
        (tx, at_the_edge(response).into_body())
    }

    async fn send(tx: &mpsc::Sender<Chunk>, bytes: &'static [u8]) {
        tx.send(Ok(Bytes::from_static(bytes))).await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn a_quiet_stream_gets_a_comment_every_interval_from_the_headers() {
        let (_tx, mut body) = wrapped(SECS, false);
        assert_eq!(next_within(&mut body, SECS - 1).await, Read::Silence);
        assert_eq!(
            next_within(&mut body, 2).await,
            Read::Frame(Bytes::from_static(COMMENT))
        );
        // The next one an interval after the keep-alive, not sooner.
        assert_eq!(next_within(&mut body, SECS - 1).await, Read::Silence);
        assert_eq!(
            next_within(&mut body, 2).await,
            Read::Frame(Bytes::from_static(COMMENT))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_flowing_stream_carries_no_keep_alive() {
        let (tx, mut body) = wrapped(SECS, false);
        for _ in 0..6 {
            tokio::time::sleep(Duration::from_secs(10)).await;
            send(&tx, b"data: {\"delta\":\"x\"}\n\n").await;
            assert_eq!(
                next_within(&mut body, 0).await,
                Read::Frame(Bytes::from_static(b"data: {\"delta\":\"x\"}\n\n"))
            );
        }
        // The interval counts from the last frame.
        assert_eq!(next_within(&mut body, SECS - 1).await, Read::Silence);
        assert_eq!(
            next_within(&mut body, 2).await,
            Read::Frame(Bytes::from_static(COMMENT))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_keep_alive_never_splits_a_frame() {
        let (tx, mut body) = wrapped(SECS, false);
        send(&tx, b"data: {\"delta\":").await;
        assert_eq!(
            next_within(&mut body, 0).await,
            Read::Frame(Bytes::from_static(b"data: {\"delta\":"))
        );
        // Two intervals inside the frame: nothing is written.
        assert_eq!(next_within(&mut body, 3 * SECS - 1).await, Read::Silence);
        send(&tx, b"\"x\"}\n\n").await;
        assert_eq!(
            next_within(&mut body, 0).await,
            Read::Frame(Bytes::from_static(b"\"x\"}\n\n"))
        );
        // Back on a boundary: the keep-alive an interval after the frame.
        assert_eq!(next_within(&mut body, SECS - 1).await, Read::Silence);
        assert_eq!(
            next_within(&mut body, 2).await,
            Read::Frame(Bytes::from_static(COMMENT))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn an_interval_of_zero_sends_nothing() {
        let (_tx, mut body) = wrapped(0, false);
        assert_eq!(next_within(&mut body, 3600).await, Read::Silence);
    }

    #[tokio::test(start_paused = true)]
    async fn a_messages_stream_gets_the_api_s_ping_event() {
        let (_tx, mut body) = wrapped(SECS, true);
        assert_eq!(next_within(&mut body, SECS - 1).await, Read::Silence);
        assert_eq!(
            next_within(&mut body, 2).await,
            Read::Frame(Bytes::from_static(MESSAGES_PING))
        );
        // The frame is the protocol's own ping event.
        let ping = serde_json::to_string(&MessageStreamEvent::Ping).unwrap();
        assert_eq!(
            format!("event: ping\ndata: {ping}\n\n").as_bytes(),
            MESSAGES_PING
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_end_of_the_stream_passes_at_once() {
        let (tx, mut body) = wrapped(SECS, false);
        send(&tx, b"data: [DONE]\n\n").await;
        drop(tx);
        assert_eq!(
            next_within(&mut body, 0).await,
            Read::Frame(Bytes::from_static(b"data: [DONE]\n\n"))
        );
        assert_eq!(next_within(&mut body, 0).await, Read::End);
    }

    fn app(response: impl Fn() -> Response + Clone + Send + Sync + 'static) -> Router {
        let chat = response.clone();
        Router::new()
            .route("/v1/chat/completions", post(move || async move { chat() }))
            .route(MESSAGES_ROUTE, post(move || async move { response() }))
            .route_layer(SseKeepAlive::every_secs(SECS).layer())
            .layer(HttpMetricsLayer::new(InFlightRequestTracker::new()))
    }

    async fn call(app: Router, path: &str) -> Response {
        app.oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(path)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
    }

    /// An SSE response whose upstream never speaks.
    fn silent_sse() -> Response {
        Response::builder()
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(futures::stream::pending::<Chunk>()))
            .unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn the_middleware_writes_the_comment_on_an_openai_route_and_the_ping_on_messages() {
        let mut chat = call(app(silent_sse), "/v1/chat/completions")
            .await
            .into_body();
        assert_eq!(
            next_within(&mut chat, SECS + 1).await,
            Read::Frame(Bytes::from_static(COMMENT))
        );
        let mut messages = call(app(silent_sse), MESSAGES_ROUTE).await.into_body();
        assert_eq!(
            next_within(&mut messages, SECS + 1).await,
            Read::Frame(Bytes::from_static(MESSAGES_PING))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_response_that_is_not_an_event_stream_passes_untouched() {
        let json = || {
            Response::builder()
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from_stream(futures::stream::pending::<Chunk>()))
                .unwrap()
        };
        let response = call(app(json), "/v1/chat/completions").await;
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body();
        assert_eq!(next_within(&mut body, 10 * SECS).await, Read::Silence);
    }

    /// A streamed body under an upstream's `Content-Length` (the relays copy
    /// the header) is left alone: a keep-alive would push it past the length.
    #[tokio::test(start_paused = true)]
    async fn an_sse_answer_under_a_content_length_header_passes_untouched() {
        let declared = || {
            Response::builder()
                .header(CONTENT_TYPE, "text/event-stream")
                .header(CONTENT_LENGTH, "14")
                .body(Body::from_stream(futures::stream::pending::<Chunk>()))
                .unwrap()
        };
        let response = call(app(declared), "/v1/chat/completions").await;
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body();
        assert_eq!(next_within(&mut body, 10 * SECS).await, Read::Silence);
    }

    #[tokio::test(start_paused = true)]
    async fn a_complete_sse_answer_passes_untouched() {
        let done = || {
            Response::builder()
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from("data: [DONE]\n\n"))
                .unwrap()
        };
        let response = call(app(done), "/v1/chat/completions").await;
        assert_eq!(
            response.body().size_hint().exact(),
            Some("data: [DONE]\n\n".len() as u64),
            "a sized body keeps its length"
        );
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"data: [DONE]\n\n");
    }
}
