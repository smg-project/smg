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

fn scripted(steps: impl IntoIterator<Item = Poll<Option<Result<Bytes, io::Error>>>>) -> Scripted {
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
