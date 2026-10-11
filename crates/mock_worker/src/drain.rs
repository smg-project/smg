//! The requests in flight across the process, for the shutdown that drains
//! them.
//!
//! Every worker flavour enters a request here when it starts serving it and
//! leaves when the response is complete (or its stream is dropped), so the
//! termination signal can say how many requests it waits for, and a count that
//! does not reach zero before the drain deadline is reported as such.

use std::{
    pin::Pin,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll},
};

use futures::Stream;

/// The process-wide count.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// The number of requests in flight across every worker in the process.
pub fn in_flight() -> usize {
    IN_FLIGHT.load(Ordering::Relaxed)
}

/// One request in flight: created when a worker starts serving a request and
/// dropped when its response is complete (or its stream is dropped).
#[must_use = "a request is in flight only while its guard is alive"]
pub struct InFlight {
    counter: &'static AtomicUsize,
}

impl InFlight {
    /// Count a request as in flight until the guard is dropped.
    pub fn enter() -> Self {
        Self::enter_in(&IN_FLIGHT)
    }

    fn enter_in(counter: &'static AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::Relaxed);
        Self { counter }
    }

    /// Keep the request in flight for as long as `stream` lives: the guard
    /// rides along with the stream and is released with it, whether the
    /// stream ends or its reader goes away first.
    pub fn until_done<T>(self, stream: impl Stream<Item = T> + Send + 'static) -> Counted<T> {
        Counted {
            inner: Box::pin(stream),
            _guard: self,
        }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A response stream that counts as in flight until it is done.
pub struct Counted<T> {
    inner: Pin<Box<dyn Stream<Item = T> + Send>>,
    _guard: InFlight,
}

impl<T> Stream for Counted<T> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<T>> {
        self.inner.as_mut().poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use futures::{stream, StreamExt};

    use super::*;

    // Each test counts in its own static, so the process-wide count the other
    // tests may touch concurrently does not bleed into the assertions.

    #[test]
    fn a_guard_counts_until_it_is_dropped() {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let first = InFlight::enter_in(&COUNT);
        let second = InFlight::enter_in(&COUNT);
        assert_eq!(COUNT.load(Ordering::Relaxed), 2);
        drop(first);
        assert_eq!(COUNT.load(Ordering::Relaxed), 1);
        drop(second);
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn a_counted_stream_is_in_flight_until_its_end() {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let mut counted = InFlight::enter_in(&COUNT).until_done(stream::iter([1, 2]));
        assert_eq!(counted.next().await, Some(1));
        assert_eq!(
            COUNT.load(Ordering::Relaxed),
            1,
            "in flight while items remain"
        );
        assert_eq!(counted.next().await, Some(2));
        assert_eq!(counted.next().await, None);
        assert_eq!(
            COUNT.load(Ordering::Relaxed),
            1,
            "the stream object is still alive"
        );
        drop(counted);
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn a_counted_stream_dropped_by_its_reader_leaves_the_count() {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let mut counted = InFlight::enter_in(&COUNT).until_done(stream::pending::<u8>());
        assert_eq!(COUNT.load(Ordering::Relaxed), 1);
        // A reader that gives up mid-stream (the client went away): the
        // request leaves the count with the stream, not at its end.
        let pending = futures::poll!(counted.next());
        assert!(pending.is_pending());
        drop(counted);
        assert_eq!(COUNT.load(Ordering::Relaxed), 0);
    }
}
