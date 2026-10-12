//! A keep-alive on SSE streams whose upstream has gone quiet.
//!
//! A streamed response is answered with its headers as soon as the request
//! is dispatched; its first token may follow a long prefill tens of seconds
//! later, and nothing was sent in between. An idle-timeout proxy in front of
//! the gateway cuts a connection that carries no byte for its idle window
//! (sixty seconds is a common setting), and the client sees a stream that
//! ends before it began. [`sse_keepalive_middleware`] wraps every SSE
//! response body so an interval of silence is broken by a frame the client
//! ignores: an SSE comment on the OpenAI-shaped routes, the `ping` event the
//! Messages API specifies on `/v1/messages`. The interval runs from the
//! headers and from every frame sent, so a flowing stream carries no
//! keep-alive; a keep-alive is only ever written between frames, never
//! inside one.

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
use http_body::{Body as HttpBody, Frame, SizeHint};
use tokio::time::{Instant, Sleep};

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

    /// Wraps the body of an SSE `response` whose length is not known up
    /// front (a complete SSE answer has nothing to wait for, and a body sent
    /// under a `Content-Length` header must not grow past it: the relays copy
    /// an upstream's header onto a streamed body); `messages` selects the
    /// Messages API's ping over the comment. Anything else is returned as it
    /// came.
    pub(crate) fn apply(self, response: Response, messages: bool) -> Response {
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
        response.map(|body| Body::new(KeepAliveBody::new(body, interval, frame)))
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

/// Route-layer middleware: the keep-alive on every SSE response of the
/// routes it wraps.
pub async fn sse_keepalive_middleware(
    State(keep_alive): State<SseKeepAlive>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let messages = request.uri().path() == MESSAGES_ROUTE;
    keep_alive.apply(next.run(request).await, messages)
}

/// Body wrapper that writes `frame` after `interval` of silence, on a frame
/// boundary. The timer is lazy: when it fires it is re-armed from the time
/// of the last frame rather than reset on every frame, so a flowing stream
/// costs one timer wake per interval, not one per token.
struct KeepAliveBody {
    inner: Body,
    interval: Duration,
    frame: Bytes,
    /// When the headers, the last frame or the last keep-alive went out.
    last_sent: Instant,
    /// The last bytes sent, to tell a frame boundary from the inside of a
    /// frame.
    tail: Vec<u8>,
    timer: Pin<Box<Sleep>>,
}

impl KeepAliveBody {
    fn new(inner: Body, interval: Duration, frame: Bytes) -> Self {
        let now = Instant::now();
        Self {
            inner,
            interval,
            frame,
            last_sent: now,
            tail: Vec::with_capacity(4),
            timer: Box::pin(tokio::time::sleep_until(now + interval)),
        }
    }
}

impl HttpBody for KeepAliveBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let this = &mut *self;
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    note_sent(&mut this.tail, data);
                    this.last_sent = Instant::now();
                }
                return Poll::Ready(Some(Ok(frame)));
            }
            Poll::Ready(end) => return Poll::Ready(end),
            Poll::Pending => {}
        }
        // The upstream is quiet. The timer is due `interval` after the last
        // frame: fired early (a frame went out since it was armed) it is
        // re-armed for then; fired inside a frame it is re-armed for another
        // interval and nothing is written.
        loop {
            if this.timer.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            let now = Instant::now();
            let due = this.last_sent + this.interval;
            if now < due {
                this.timer.as_mut().reset(due);
            } else if on_frame_boundary(&this.tail) {
                this.last_sent = now;
                this.timer.as_mut().reset(now + this.interval);
                return Poll::Ready(Some(Ok(Frame::data(this.frame.clone()))));
            } else {
                this.timer.as_mut().reset(now + this.interval);
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
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

    /// An SSE body under the keep-alive, fed through the returned sender.
    fn wrapped(secs: u64, messages: bool) -> (mpsc::Sender<Chunk>, Body) {
        let (tx, rx) = mpsc::channel(16);
        let response = SseKeepAlive::every_secs(secs).apply(sse(rx), messages);
        (tx, response.into_body())
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
            .route_layer(axum::middleware::from_fn_with_state(
                SseKeepAlive::every_secs(SECS),
                sse_keepalive_middleware,
            ))
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
