//! Streams a worker's response body straight through to the client.
//!
//! The body is polled by hyper on the connection task, so each upstream chunk
//! costs one poll instead of a relay task plus a channel hop and two
//! cross-thread wakeups. Backpressure is the client's own: the upstream is read
//! only as fast as the response is written. Dropping the body (the client went
//! away) drops the upstream stream, which closes the worker connection and lets
//! the engine abort generation.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use bytes::Bytes;
use futures_util::Stream;
use http_body::Frame;
use tokio::time::{Instant, Sleep};

use crate::routers::common::sse_rechunk::{SseRechunker, IDLE_FLUSH};

type Upstream = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send + 'static>>;

pub(crate) struct RelayBody {
    /// `None` once the upstream ended or failed.
    upstream: Option<Upstream>,
    rechunker: Option<SseRechunker>,
    /// Flushes buffered rechunker text after `IDLE_FLUSH` of upstream silence.
    /// Armed only while text is pending, from the first poll that found the
    /// upstream idle after the last chunk.
    idle: Pin<Box<Sleep>>,
    idle_armed: bool,
    /// Queued ahead of anything else, in this order: the rechunker's tail at
    /// end of stream, then a terminal upstream error.
    tail: Option<Bytes>,
    error: Option<String>,
}

impl RelayBody {
    pub(crate) fn new(upstream: Upstream, rechunker: Option<SseRechunker>) -> Self {
        Self {
            upstream: Some(upstream),
            rechunker,
            idle: Box::pin(tokio::time::sleep(IDLE_FLUSH)),
            idle_armed: false,
            tail: None,
            error: None,
        }
    }

    /// Ends the upstream, queueing whatever the rechunker still holds.
    fn finish_upstream(&mut self) {
        self.upstream = None;
        self.idle_armed = false;
        self.tail = self
            .rechunker
            .as_mut()
            .map(SseRechunker::finish)
            .filter(|tail| !tail.is_empty());
    }
}

impl http_body::Body for RelayBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        loop {
            if let Some(tail) = this.tail.take() {
                return Poll::Ready(Some(Ok(Frame::data(tail))));
            }
            if let Some(error) = this.error.take() {
                return Poll::Ready(Some(Err(axum::Error::new(error))));
            }
            let Some(upstream) = this.upstream.as_mut() else {
                return Poll::Ready(None);
            };
            match upstream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    // An empty upstream chunk must not become an empty h2 DATA
                    // frame toward the client.
                    if bytes.is_empty() {
                        continue;
                    }
                    let bytes = match this.rechunker.as_mut() {
                        Some(rechunker) => {
                            this.idle_armed = false;
                            rechunker.feed(bytes)
                        }
                        None => bytes,
                    };
                    if bytes.is_empty() {
                        continue;
                    }
                    return Poll::Ready(Some(Ok(Frame::data(bytes))));
                }
                Poll::Ready(Some(Err(e))) => {
                    this.finish_upstream();
                    this.error = Some(format!("Stream error: {e}"));
                }
                Poll::Ready(None) => this.finish_upstream(),
                Poll::Pending => {
                    let Some(rechunker) = this.rechunker.as_mut() else {
                        return Poll::Pending;
                    };
                    if !rechunker.has_pending() {
                        this.idle_armed = false;
                        return Poll::Pending;
                    }
                    if !this.idle_armed {
                        this.idle.as_mut().reset(Instant::now() + IDLE_FLUSH);
                        this.idle_armed = true;
                    }
                    match this.idle.as_mut().poll(cx) {
                        Poll::Ready(()) => {
                            this.idle_armed = false;
                            let bytes = rechunker.flush_pending();
                            if bytes.is_empty() {
                                return Poll::Pending;
                            }
                            return Poll::Ready(Some(Ok(Frame::data(bytes))));
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.upstream.is_none() && self.tail.is_none() && self.error.is_none()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::body::Body;
    use futures_util::stream;
    use http_body_util::BodyExt;

    use super::*;

    fn upstream(items: Vec<reqwest::Result<Bytes>>) -> Upstream {
        Box::pin(stream::iter(items))
    }

    fn pending_forever() -> Upstream {
        Box::pin(stream::pending())
    }

    /// A `reqwest::Error` built without touching the network: the URL fails to
    /// parse, so the error comes straight from the request builder.
    async fn upstream_error() -> reqwest::Error {
        reqwest::get("not a url").await.unwrap_err()
    }

    async fn collect(body: RelayBody) -> (Vec<Bytes>, Option<String>) {
        let mut body = Body::new(body);
        let mut frames = Vec::new();
        loop {
            match body.frame().await {
                Some(Ok(frame)) => frames.push(frame.into_data().expect("data frame")),
                Some(Err(e)) => return (frames, Some(e.to_string())),
                None => return (frames, None),
            }
        }
    }

    #[tokio::test]
    async fn passthrough_forwards_chunks_and_skips_empty_ones() {
        let body = RelayBody::new(
            upstream(vec![
                Ok(Bytes::from_static(b"a")),
                Ok(Bytes::new()),
                Ok(Bytes::from_static(b"b")),
            ]),
            None,
        );
        let (frames, error) = collect(body).await;
        assert_eq!(
            frames,
            vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")]
        );
        assert!(error.is_none());
    }

    #[tokio::test]
    async fn rechunker_tail_is_flushed_at_end_of_stream() {
        let event = br#"data: {"id":"x","choices":[{"index":0,"delta":{"content":"hi"}}]}

"#;
        let body = RelayBody::new(
            upstream(vec![Ok(Bytes::from_static(event))]),
            Some(SseRechunker::new()),
        );
        let (frames, error) = collect(body).await;
        assert!(error.is_none());
        let joined: Vec<u8> = frames.concat();
        let text = String::from_utf8(joined).unwrap();
        assert!(text.contains(r#""content":"hi""#), "tail flushed: {text}");
    }

    #[tokio::test(start_paused = true)]
    async fn idle_flush_emits_buffered_text_without_more_input() {
        let event = br#"data: {"id":"x","choices":[{"index":0,"delta":{"content":"hi"}}]}

"#;
        let first = Box::pin(stream::iter(vec![Ok(Bytes::from_static(event))]));
        let chained: Upstream = Box::pin(futures_util::StreamExt::chain(first, pending_forever()));
        let mut body = Body::new(RelayBody::new(chained, Some(SseRechunker::new())));
        // The text is below the emit threshold, so only the idle timer releases it.
        let frame = tokio::time::timeout(IDLE_FLUSH + Duration::from_millis(50), body.frame())
            .await
            .expect("idle flush fired")
            .expect("a frame")
            .expect("no error");
        let text = String::from_utf8(frame.into_data().unwrap().to_vec()).unwrap();
        assert!(text.contains(r#""content":"hi""#), "flushed: {text}");
    }

    #[tokio::test]
    async fn upstream_error_follows_the_tail() {
        // Below the rechunker's emit threshold, so the text is still buffered
        // when the upstream fails and only `finish()` releases it.
        let event = br#"data: {"id":"x","choices":[{"index":0,"delta":{"content":"hi"}}]}

"#;
        let body = RelayBody::new(
            upstream(vec![
                Ok(Bytes::from_static(event)),
                Err(upstream_error().await),
            ]),
            Some(SseRechunker::new()),
        );
        let (frames, error) = collect(body).await;
        assert_eq!(frames.len(), 1, "the buffered tail is flushed as one frame");
        let text = String::from_utf8(frames[0].to_vec()).unwrap();
        assert!(
            text.contains(r#""content":"hi""#),
            "tail before error: {text}"
        );
        assert!(error.unwrap().contains("Stream error"));
    }
}
