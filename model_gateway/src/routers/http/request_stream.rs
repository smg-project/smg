//! Streamed request-body pass-through: byte accounting for the ingress
//! payload cap and forwarding-progress tracking for the stall watchdog.

use std::{
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use bytes::Bytes;
use futures_util::Stream;
use tokio::time::Instant;

/// Abort a streamed dispatch when no request-body bytes move for this long:
/// a stalled uploader must not hold a worker slot indefinitely.
pub(crate) const STREAM_PROGRESS_TIMEOUT: Duration = Duration::from_secs(60);

/// Shared view of a streamed request body: forwarding progress for the stall
/// watchdog, plus the terminal payload-limit flag the dispatcher maps to 413.
pub(crate) struct StreamProgress {
    started: Instant,
    last_progress_ms: AtomicU64,
    body_complete: AtomicBool,
    limit_exceeded: AtomicBool,
    inbound_error: AtomicBool,
}

impl StreamProgress {
    pub(crate) fn new() -> Self {
        Self {
            started: Instant::now(),
            last_progress_ms: AtomicU64::new(0),
            body_complete: AtomicBool::new(false),
            limit_exceeded: AtomicBool::new(false),
            inbound_error: AtomicBool::new(false),
        }
    }

    fn touch(&self) {
        let elapsed = u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.last_progress_ms.store(elapsed, Ordering::Relaxed);
    }

    pub(crate) fn limit_exceeded(&self) -> bool {
        self.limit_exceeded.load(Ordering::Relaxed)
    }

    /// True when the inbound body stream itself failed (client disconnect or
    /// reset mid-upload) — a client-caused abort, never worker-attributed.
    pub(crate) fn inbound_error(&self) -> bool {
        self.inbound_error.load(Ordering::Relaxed)
    }

    /// Resolves once no body bytes have moved for `idle`. Never resolves
    /// after the body is fully forwarded: waiting on the worker's response is
    /// not an upload stall.
    pub(crate) async fn stalled(&self, idle: Duration) {
        loop {
            if self.body_complete.load(Ordering::Relaxed) {
                std::future::pending::<()>().await;
            }
            let seen = self.last_progress_ms.load(Ordering::Relaxed);
            tokio::time::sleep_until(self.started + Duration::from_millis(seen) + idle).await;
            if self.body_complete.load(Ordering::Relaxed) {
                continue;
            }
            if self.last_progress_ms.load(Ordering::Relaxed) == seen {
                return;
            }
        }
    }
}

/// Counts request-body bytes against `limit` as they stream to the worker,
/// reporting progress to [`StreamProgress`]. An over-limit body — caught by
/// this counter or surfaced as an ingress `LengthLimitError` from the inner
/// stream — flags the shared limit marker and terminates with an error, which
/// aborts the upstream send.
pub(crate) struct CappedBodyStream<S> {
    inner: S,
    progress: Arc<StreamProgress>,
    limit: usize,
    forwarded: usize,
    done: bool,
}

impl<S> CappedBodyStream<S> {
    pub(crate) fn new(inner: S, limit: usize, progress: Arc<StreamProgress>) -> Self {
        Self {
            inner,
            progress,
            limit,
            forwarded: 0,
            done: false,
        }
    }
}

impl<S, E> Stream for CappedBodyStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    type Item = Result<Bytes, io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(None);
        }
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                this.forwarded = this.forwarded.saturating_add(chunk.len());
                if this.forwarded > this.limit {
                    this.done = true;
                    this.progress.limit_exceeded.store(true, Ordering::Relaxed);
                    return Poll::Ready(Some(Err(io::Error::other(
                        "request body exceeded the payload limit",
                    ))));
                }
                this.progress.touch();
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                this.done = true;
                if is_length_limit_error(&e) {
                    this.progress.limit_exceeded.store(true, Ordering::Relaxed);
                } else {
                    this.progress.inbound_error.store(true, Ordering::Relaxed);
                }
                Poll::Ready(Some(Err(io::Error::other(e))))
            }
            Poll::Ready(None) => {
                this.done = true;
                this.progress.body_complete.store(true, Ordering::Relaxed);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// True when `err`'s source chain contains the ingress body-limit error.
fn is_length_limit_error(err: &(dyn std::error::Error + 'static)) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(err) = current {
        if err.is::<http_body_util::LengthLimitError>() {
            return true;
        }
        current = err.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use futures_util::{stream, StreamExt};
    use http_body_util::Limited;

    use super::*;

    fn chunks(parts: &[&'static [u8]]) -> Vec<Result<Bytes, io::Error>> {
        parts.iter().map(|c| Ok(Bytes::from_static(c))).collect()
    }

    #[tokio::test]
    async fn body_at_limit_passes_through_and_completes() {
        let progress = Arc::new(StreamProgress::new());
        let stream = CappedBodyStream::new(
            stream::iter(chunks(&[b"abc", b"defgh"])),
            8,
            Arc::clone(&progress),
        );

        let collected: Vec<_> = stream.collect().await;

        let bytes: Vec<u8> = collected
            .into_iter()
            .flat_map(|c| c.unwrap().to_vec())
            .collect();
        assert_eq!(bytes, b"abcdefgh");
        assert!(!progress.limit_exceeded());
        assert!(progress.body_complete.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn over_limit_body_errors_and_flags() {
        let progress = Arc::new(StreamProgress::new());
        let mut stream = CappedBodyStream::new(
            stream::iter(chunks(&[b"abc", b"defgh"])),
            7,
            Arc::clone(&progress),
        );

        assert_eq!(stream.next().await.unwrap().unwrap().as_ref(), b"abc");
        stream.next().await.unwrap().unwrap_err();
        assert!(progress.limit_exceeded());
        // Terminal after the error: nothing further is forwarded.
        assert!(stream.next().await.is_none());
        assert!(!progress.body_complete.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn inner_non_limit_error_flags_inbound_error() {
        let progress = Arc::new(StreamProgress::new());
        let failing = stream::iter([
            Ok(Bytes::from_static(b"abc")),
            Err(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "client reset",
            )),
        ]);
        let mut stream = CappedBodyStream::new(failing, usize::MAX, Arc::clone(&progress));

        assert_eq!(stream.next().await.unwrap().unwrap().as_ref(), b"abc");
        stream.next().await.unwrap().unwrap_err();
        assert!(progress.inbound_error());
        assert!(!progress.limit_exceeded());
    }

    #[tokio::test]
    async fn inner_length_limit_error_flags_exceeded() {
        let progress = Arc::new(StreamProgress::new());
        let limited = Body::new(Limited::new(Body::from(vec![0u8; 64]), 8));
        let mut stream = CappedBodyStream::new(
            limited.into_data_stream(),
            usize::MAX,
            Arc::clone(&progress),
        );

        while let Some(item) = stream.next().await {
            if item.is_err() {
                break;
            }
        }

        assert!(progress.limit_exceeded());
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_fires_after_idle_window() {
        let progress = Arc::new(StreamProgress::new());
        tokio::time::timeout(
            Duration::from_secs(120),
            progress.stalled(Duration::from_secs(60)),
        )
        .await
        .expect("watchdog must fire once the idle window passes");
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_never_fires_after_body_completes() {
        let progress = Arc::new(StreamProgress::new());
        let stream =
            CappedBodyStream::new(stream::iter(chunks(&[b"abc"])), 8, Arc::clone(&progress));
        let _drained: Vec<_> = stream.collect().await;

        tokio::time::timeout(
            Duration::from_secs(600),
            progress.stalled(Duration::from_secs(60)),
        )
        .await
        .expect_err("a fully forwarded body must disarm the watchdog");
    }
}
