//! A keep-alive on SSE streams whose upstream has gone quiet.
//!
//! A streamed response is answered with its headers as soon as the request
//! is dispatched; its first token may follow a long prefill tens of seconds
//! later, and nothing was sent in between. An idle-timeout proxy in front of
//! the gateway cuts a connection that carries no byte for its idle window
//! (sixty seconds is a common setting), and the client sees a stream that
//! ends before it began. [`SseKeepAlive::layer`] gives every SSE
//! response a `KeepAlive`, carried by the edge's one body wrapper (the
//! metrics layer's `EdgeBody`), so an interval
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
    extract::Request,
    http::header::{CONTENT_LENGTH, CONTENT_TYPE},
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
    routers::http::stream_failure::{note_sent, on_frame_boundary, SentTail},
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
    /// `Content-Length` header must not grow past it: the relay drops an
    /// upstream's header from a streamed SSE body, so the header that remains
    /// is a non-SSE upstream body the PD router relabels as SSE) its
    /// keep-alive, handed to the edge's body wrapper; `messages` selects the
    /// Messages API's ping over the comment. Anything else is returned as it
    /// came.
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
    /// response, nothing is cloned or boxed per request.
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

/// The keep-alive of one SSE body: `frame` after `interval` of silence, on a
/// frame boundary. The edge's body wrapper tells it of every frame sent and
/// polls it while the body is quiet. The clock is read when the body turns
/// quiet after frames went out (the moment the last of them left, to the
/// microsecond: the wrapper asks for the next frame as soon as it has taken
/// one), not once per frame, so a stream of a thousand deltas stamps once
/// per burst. The timer is lazy as well: when it fires it is re-armed from
/// the time of the last frame rather than reset on every frame, so a flowing
/// stream costs one timer wake per interval, not one per token.
pub(crate) struct KeepAlive {
    interval: Duration,
    frame: Bytes,
    /// When the headers, the last burst of frames or the last keep-alive went
    /// out.
    last_sent: Instant,
    /// Frames went out since `last_sent` was stamped: the next quiet poll
    /// stamps the moment they stopped.
    sent_since: bool,
    /// The last bytes sent, to tell a frame boundary from the inside of a
    /// frame.
    tail: SentTail,
    /// Built on the first wait for silence, due one interval after the last
    /// frame, which is when a timer built with the stream would have been
    /// reset to anyway; a stream that is never quiet never builds it.
    timer: Option<Pin<Box<Sleep>>>,
}

impl KeepAlive {
    fn new(interval: Duration, frame: Bytes) -> Self {
        Self {
            interval,
            frame,
            last_sent: Instant::now(),
            sent_since: false,
            tail: SentTail::default(),
            timer: None,
        }
    }

    /// A data frame of the body went out.
    pub(crate) fn frame_out(&mut self, data: &[u8]) {
        note_sent(&mut self.tail, data);
        self.sent_since = true;
    }

    /// The body is quiet: the keep-alive frame once `interval` has passed
    /// since the last frame, on a frame boundary; pending, with the timer
    /// armed, until then. The timer is due `interval` after the last frame:
    /// fired early (a frame went out since it was armed) it is re-armed for
    /// then; fired inside a frame it is re-armed for another interval and
    /// nothing is written.
    pub(crate) fn poll_silence(&mut self, cx: &mut Context<'_>) -> Poll<Bytes> {
        if std::mem::take(&mut self.sent_since) {
            // The first quiet poll after a burst: its last frame just left.
            self.last_sent = Instant::now();
        }
        let first_due = self.last_sent + self.interval;
        let timer = self
            .timer
            .get_or_insert_with(|| Box::pin(tokio::time::sleep_until(first_due)));
        loop {
            if timer.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            let now = Instant::now();
            let due = self.last_sent + self.interval;
            if now < due {
                timer.as_mut().reset(due);
            } else if on_frame_boundary(self.tail.as_slice()) {
                self.last_sent = now;
                timer.as_mut().reset(now + self.interval);
                return Poll::Ready(self.frame.clone());
            } else {
                timer.as_mut().reset(now + self.interval);
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/middleware/sse_keepalive/tests.rs"]
mod tests;
