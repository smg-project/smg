//! Streams a worker's response body straight through to the client.
//!
//! The body is polled by hyper on the connection task, so each upstream chunk
//! costs one poll instead of a relay task plus a channel hop and two
//! cross-thread wakeups. Backpressure is the client's own: the upstream is read
//! only as fast as the response is written. Dropping the body (the client went
//! away) drops the upstream stream, which closes the worker connection and lets
//! the engine abort generation.
//!
//! What happens to each chunk is a [`ChunkFilter`]: pass-through, SSE
//! re-slicing, or the PD decode relay's sentinel detection and logprob merge.

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

/// Per-chunk policy of a [`RelayBody`].
pub(crate) trait ChunkFilter: Send + Unpin + 'static {
    /// One upstream chunk. Returns the bytes to forward now (possibly none)
    /// and whether the relayed stream ends after them, whatever the upstream
    /// still has: the upstream is dropped and only [`finish`](Self::finish)
    /// follows.
    fn feed(&mut self, chunk: Bytes) -> (Bytes, bool);

    /// Whether text is held back waiting for more input. If so, it is
    /// released by [`flush_pending`](Self::flush_pending) after `IDLE_FLUSH`
    /// of upstream silence.
    fn has_pending(&self) -> bool {
        false
    }

    fn flush_pending(&mut self) -> Bytes {
        Bytes::new()
    }

    /// Everything still held when the upstream ends or fails.
    fn finish(&mut self) -> Bytes {
        Bytes::new()
    }

    /// The upstream failed; called before the error is relayed.
    fn on_error(&mut self, _error: &reqwest::Error) {}
}

impl ChunkFilter for SseRechunker {
    fn feed(&mut self, chunk: Bytes) -> (Bytes, bool) {
        (SseRechunker::feed(self, chunk), false)
    }

    fn has_pending(&self) -> bool {
        SseRechunker::has_pending(self)
    }

    fn flush_pending(&mut self) -> Bytes {
        SseRechunker::flush_pending(self)
    }

    fn finish(&mut self) -> Bytes {
        SseRechunker::finish(self)
    }
}

/// `None` passes chunks through unchanged.
impl<F: ChunkFilter> ChunkFilter for Option<F> {
    fn feed(&mut self, chunk: Bytes) -> (Bytes, bool) {
        match self {
            Some(filter) => filter.feed(chunk),
            None => (chunk, false),
        }
    }

    fn has_pending(&self) -> bool {
        self.as_ref().is_some_and(ChunkFilter::has_pending)
    }

    fn flush_pending(&mut self) -> Bytes {
        self.as_mut()
            .map(ChunkFilter::flush_pending)
            .unwrap_or_default()
    }

    fn finish(&mut self) -> Bytes {
        self.as_mut().map(ChunkFilter::finish).unwrap_or_default()
    }

    fn on_error(&mut self, error: &reqwest::Error) {
        if let Some(filter) = self {
            filter.on_error(error);
        }
    }
}

pub(crate) struct RelayBody<F> {
    /// `None` once the upstream ended or failed.
    upstream: Option<Upstream>,
    filter: F,
    /// Flushes text the filter holds back after `IDLE_FLUSH` of upstream
    /// silence. Armed only while text is pending, from the first poll that
    /// found the upstream idle after the last chunk.
    idle: Pin<Box<Sleep>>,
    idle_armed: bool,
    /// Queued ahead of anything else, in this order: the filter's tail at end
    /// of stream, then a terminal upstream error.
    tail: Option<Bytes>,
    error: Option<String>,
}

impl<F: ChunkFilter> RelayBody<F> {
    pub(crate) fn new(upstream: Upstream, filter: F) -> Self {
        Self {
            upstream: Some(upstream),
            filter,
            idle: Box::pin(tokio::time::sleep(IDLE_FLUSH)),
            idle_armed: false,
            tail: None,
            error: None,
        }
    }

    /// Ends the upstream, queueing whatever the filter still holds.
    fn finish_upstream(&mut self) {
        self.upstream = None;
        self.idle_armed = false;
        self.tail = Some(self.filter.finish()).filter(|tail| !tail.is_empty());
    }
}

impl<F: ChunkFilter> http_body::Body for RelayBody<F> {
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
                    this.idle_armed = false;
                    let (bytes, end) = this.filter.feed(bytes);
                    if end {
                        // The filter saw the end of the stream (an SSE
                        // `[DONE]`); whatever the upstream still has is not
                        // for this client.
                        this.finish_upstream();
                    }
                    if bytes.is_empty() {
                        continue;
                    }
                    return Poll::Ready(Some(Ok(Frame::data(bytes))));
                }
                Poll::Ready(Some(Err(e))) => {
                    this.filter.on_error(&e);
                    this.finish_upstream();
                    this.error = Some(format!("Stream error: {e}"));
                }
                Poll::Ready(None) => this.finish_upstream(),
                Poll::Pending => {
                    if !this.filter.has_pending() {
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
                            let bytes = this.filter.flush_pending();
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
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
        time::Duration,
    };

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

    async fn collect<F: ChunkFilter>(body: RelayBody<F>) -> (Vec<Bytes>, Option<String>) {
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
            None::<SseRechunker>,
        );
        let (frames, error) = collect(body).await;
        assert_eq!(
            frames,
            vec![Bytes::from_static(b"a"), Bytes::from_static(b"b")]
        );
        assert!(error.is_none());
    }

    /// Ends the relay after the chunk that contains `END`, counts errors.
    struct EndAtMarker {
        errors: Arc<AtomicUsize>,
    }

    impl ChunkFilter for EndAtMarker {
        fn feed(&mut self, chunk: Bytes) -> (Bytes, bool) {
            let end = chunk.windows(3).any(|w| w == b"END");
            (chunk, end)
        }

        fn finish(&mut self) -> Bytes {
            Bytes::from_static(b"<tail>")
        }

        fn on_error(&mut self, _error: &reqwest::Error) {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[tokio::test]
    async fn a_filter_can_end_the_stream_before_the_upstream_does() {
        let errors = Arc::new(AtomicUsize::new(0));
        let first = Box::pin(stream::iter(vec![
            Ok(Bytes::from_static(b"a")),
            Ok(Bytes::from_static(b"END")),
        ]));
        // Never yields: without the early end the body would hang here.
        let chained: Upstream = Box::pin(futures_util::StreamExt::chain(first, pending_forever()));
        let body = RelayBody::new(
            chained,
            EndAtMarker {
                errors: errors.clone(),
            },
        );
        let (frames, error) = tokio::time::timeout(Duration::from_secs(2), collect(body))
            .await
            .expect("ends without waiting for the upstream");
        assert_eq!(
            frames,
            vec![
                Bytes::from_static(b"a"),
                Bytes::from_static(b"END"),
                Bytes::from_static(b"<tail>"),
            ]
        );
        assert!(error.is_none());
        assert_eq!(errors.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn a_filter_sees_the_upstream_error_before_it_is_relayed() {
        let errors = Arc::new(AtomicUsize::new(0));
        let body = RelayBody::new(
            upstream(vec![
                Ok(Bytes::from_static(b"a")),
                Err(upstream_error().await),
            ]),
            EndAtMarker {
                errors: errors.clone(),
            },
        );
        let (frames, error) = collect(body).await;
        assert_eq!(
            frames,
            vec![Bytes::from_static(b"a"), Bytes::from_static(b"<tail>")]
        );
        assert!(error.unwrap().contains("Stream error"));
        assert_eq!(errors.load(Ordering::Relaxed), 1);
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
