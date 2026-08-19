//! Streamed request-body pass-through: byte accounting for the ingress
//! payload cap and client-wait tracking for the stall watchdog.

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
use tokio::{sync::Notify, time::Instant};

/// Sentinel for `waiting_since_ms`: no client wait is in progress.
const NOT_WAITING: u64 = u64::MAX;

/// Shared view of a streamed request body: client-wait state for the stall
/// watchdog, plus the terminal payload-limit flag the dispatcher maps to 413.
///
/// The stall clock runs only while the upstream sender is actively waiting on
/// the client — an inbound poll returned `Pending`. While the worker applies
/// backpressure the body is not polled, no wait is recorded, and the watchdog
/// stays disarmed: a slow worker read must never be blamed on the client.
pub(crate) struct StreamProgress {
    started: Instant,
    /// Milliseconds after `started` at which the current client wait began,
    /// or [`NOT_WAITING`].
    waiting_since_ms: AtomicU64,
    wait_started: Notify,
    body_complete: AtomicBool,
    limit_exceeded: AtomicBool,
    inbound_error: AtomicBool,
}

impl StreamProgress {
    pub(crate) fn new() -> Self {
        Self {
            started: Instant::now(),
            waiting_since_ms: AtomicU64::new(NOT_WAITING),
            wait_started: Notify::new(),
            body_complete: AtomicBool::new(false),
            limit_exceeded: AtomicBool::new(false),
            inbound_error: AtomicBool::new(false),
        }
    }

    /// An inbound poll returned `Pending`: the sender wants bytes the client
    /// has not produced. Starts the wait clock unless one is already running.
    fn client_pending(&self) {
        if self.waiting_since_ms.load(Ordering::Relaxed) != NOT_WAITING {
            return;
        }
        let elapsed = u64::try_from(self.started.elapsed().as_millis())
            .unwrap_or(NOT_WAITING - 1)
            .min(NOT_WAITING - 1);
        self.waiting_since_ms.store(elapsed, Ordering::Relaxed);
        self.wait_started.notify_one();
    }

    /// The inbound stream yielded (bytes, error, or end): stop the wait clock.
    fn client_progress(&self) {
        self.waiting_since_ms.store(NOT_WAITING, Ordering::Relaxed);
    }

    pub(crate) fn limit_exceeded(&self) -> bool {
        self.limit_exceeded.load(Ordering::Relaxed)
    }

    /// True when the inbound body stream itself failed (client disconnect or
    /// reset mid-upload) — a client-caused abort, never worker-attributed.
    pub(crate) fn inbound_error(&self) -> bool {
        self.inbound_error.load(Ordering::Relaxed)
    }

    /// Resolves once a single client wait has lasted `idle`. Never resolves
    /// while no wait is in progress (worker backpressure), after the body is
    /// fully forwarded, or when `idle` is `None` (watchdog disabled).
    pub(crate) async fn stalled(&self, idle: Option<Duration>) {
        let Some(idle) = idle else {
            return std::future::pending().await;
        };
        loop {
            if self.body_complete.load(Ordering::Relaxed) {
                std::future::pending::<()>().await;
            }
            let seen = self.waiting_since_ms.load(Ordering::Relaxed);
            if seen == NOT_WAITING {
                // notify_one stores a permit, so a wait that starts between
                // the load above and this await is not missed.
                self.wait_started.notified().await;
                continue;
            }
            match Duration::from_millis(seen)
                .checked_add(idle)
                .and_then(|window| self.started.checked_add(window))
            {
                // An unrepresentable deadline can never pass: park like a
                // disabled watchdog.
                None => std::future::pending::<()>().await,
                Some(deadline) => tokio::time::sleep_until(deadline).await,
            }
            if self.body_complete.load(Ordering::Relaxed) {
                continue;
            }
            if self.waiting_since_ms.load(Ordering::Relaxed) == seen {
                return;
            }
        }
    }
}

/// Counts request-body bytes against `limit` as they stream to the worker,
/// reporting client-wait transitions to [`StreamProgress`]. An over-limit
/// body — caught by this counter or surfaced as an ingress `LengthLimitError`
/// from the inner stream — flags the shared limit marker and terminates with
/// an error, which aborts the upstream send.
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
                this.progress.client_progress();
                this.forwarded = this.forwarded.saturating_add(chunk.len());
                if this.forwarded > this.limit {
                    this.done = true;
                    this.progress.limit_exceeded.store(true, Ordering::Relaxed);
                    return Poll::Ready(Some(Err(io::Error::other(
                        "request body exceeded the payload limit",
                    ))));
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                this.done = true;
                this.progress.client_progress();
                if is_length_limit_error(&e) {
                    this.progress.limit_exceeded.store(true, Ordering::Relaxed);
                } else {
                    this.progress.inbound_error.store(true, Ordering::Relaxed);
                }
                Poll::Ready(Some(Err(io::Error::other(e))))
            }
            Poll::Ready(None) => {
                this.done = true;
                this.progress.client_progress();
                this.progress.body_complete.store(true, Ordering::Relaxed);
                Poll::Ready(None)
            }
            Poll::Pending => {
                this.progress.client_pending();
                Poll::Pending
            }
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
    use std::collections::VecDeque;

    use axum::body::Body;
    use futures_util::{stream, task::noop_waker_ref, StreamExt};
    use http_body_util::Limited;
    use tokio::time::{advance, timeout};

    use super::*;

    const IDLE: Option<Duration> = Some(Duration::from_secs(60));

    fn chunks(parts: &[&'static [u8]]) -> Vec<Result<Bytes, io::Error>> {
        parts.iter().map(|c| Ok(Bytes::from_static(c))).collect()
    }

    /// Inner stream that replays a fixed script of poll results, so tests
    /// control exactly when the body is pending versus delivering.
    struct Scripted {
        steps: VecDeque<Poll<Option<Result<Bytes, io::Error>>>>,
    }

    fn scripted(
        steps: impl IntoIterator<Item = Poll<Option<Result<Bytes, io::Error>>>>,
    ) -> Scripted {
        Scripted {
            steps: steps.into_iter().collect(),
        }
    }

    fn chunk_step(data: &'static [u8]) -> Poll<Option<Result<Bytes, io::Error>>> {
        Poll::Ready(Some(Ok(Bytes::from_static(data))))
    }

    impl Stream for Scripted {
        type Item = Result<Bytes, io::Error>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.get_mut()
                .steps
                .pop_front()
                .unwrap_or(Poll::Ready(None))
        }
    }

    fn poll_once<S: Stream + Unpin>(s: &mut S) -> Poll<Option<S::Item>> {
        let mut cx = Context::from_waker(noop_waker_ref());
        Pin::new(s).poll_next(&mut cx)
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
    async fn pending_poll_arms_watchdog() {
        let progress = Arc::new(StreamProgress::new());
        let mut stream =
            CappedBodyStream::new(scripted([Poll::Pending]), usize::MAX, Arc::clone(&progress));

        assert!(poll_once(&mut stream).is_pending());

        timeout(Duration::from_secs(120), progress.stalled(IDLE))
            .await
            .expect("a client wait past the idle window must fire the watchdog");
    }

    #[tokio::test(start_paused = true)]
    async fn unpolled_body_never_stalls() {
        let progress = Arc::new(StreamProgress::new());
        let mut stream = CappedBodyStream::new(
            scripted([chunk_step(b"abc")]),
            usize::MAX,
            Arc::clone(&progress),
        );

        // One delivery, then the sender stops polling (worker backpressure).
        assert!(matches!(poll_once(&mut stream), Poll::Ready(Some(Ok(_)))));

        timeout(Duration::from_secs(600), progress.stalled(IDLE))
            .await
            .expect_err("no outstanding client wait: backpressure must not stall");
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_disarms_watchdog_after_pending() {
        let progress = Arc::new(StreamProgress::new());
        let mut stream = CappedBodyStream::new(
            scripted([Poll::Pending, chunk_step(b"abc")]),
            usize::MAX,
            Arc::clone(&progress),
        );

        assert!(poll_once(&mut stream).is_pending());
        advance(Duration::from_secs(30)).await;
        assert!(matches!(poll_once(&mut stream), Poll::Ready(Some(Ok(_)))));

        timeout(Duration::from_secs(600), progress.stalled(IDLE))
            .await
            .expect_err("a delivered chunk must stop the wait clock");
    }

    #[tokio::test(start_paused = true)]
    async fn re_pending_restarts_the_wait_window() {
        let progress = Arc::new(StreamProgress::new());
        let mut stream = CappedBodyStream::new(
            scripted([Poll::Pending, chunk_step(b"abc"), Poll::Pending]),
            usize::MAX,
            Arc::clone(&progress),
        );

        assert!(poll_once(&mut stream).is_pending());
        advance(Duration::from_secs(50)).await;
        assert!(matches!(poll_once(&mut stream), Poll::Ready(Some(Ok(_)))));
        assert!(poll_once(&mut stream).is_pending());

        // The second wait began at t=50s: not stalled at t=109s, stalled by
        // t=110s.
        timeout(Duration::from_secs(59), progress.stalled(IDLE))
            .await
            .expect_err("the wait window must restart at the second pending");
        timeout(Duration::from_secs(2), progress.stalled(IDLE))
            .await
            .expect("the restarted wait must fire once it lasts the window");
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_never_fires_after_body_completes() {
        let progress = Arc::new(StreamProgress::new());
        let mut stream = CappedBodyStream::new(
            scripted([Poll::Pending, chunk_step(b"abc"), Poll::Ready(None)]),
            usize::MAX,
            Arc::clone(&progress),
        );

        assert!(poll_once(&mut stream).is_pending());
        assert!(matches!(poll_once(&mut stream), Poll::Ready(Some(Ok(_)))));
        assert!(matches!(poll_once(&mut stream), Poll::Ready(None)));

        timeout(Duration::from_secs(600), progress.stalled(IDLE))
            .await
            .expect_err("a fully forwarded body must disarm the watchdog");
    }

    #[tokio::test(start_paused = true)]
    async fn unrepresentable_deadline_parks_instead_of_panicking() {
        let progress = Arc::new(StreamProgress::new());
        let mut stream =
            CappedBodyStream::new(scripted([Poll::Pending]), usize::MAX, Arc::clone(&progress));

        assert!(poll_once(&mut stream).is_pending());

        timeout(
            Duration::from_secs(600),
            progress.stalled(Some(Duration::from_secs(u64::MAX))),
        )
        .await
        .expect_err("an unrepresentable deadline must park, not panic or fire");
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_none_disables_watchdog() {
        let progress = Arc::new(StreamProgress::new());
        let mut stream =
            CappedBodyStream::new(scripted([Poll::Pending]), usize::MAX, Arc::clone(&progress));

        assert!(poll_once(&mut stream).is_pending());

        timeout(Duration::from_secs(600), progress.stalled(None))
            .await
            .expect_err("a disabled watchdog must never fire");
    }
}
