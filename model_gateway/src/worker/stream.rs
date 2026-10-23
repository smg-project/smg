//! The guarded stream: a response body's stream that owns an RAII guard
//! (the worker's load guard, a rate-limit reservation's safety net) and
//! drops it with the body, however the body ends.

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures_util::Stream;

/// A response body's stream that owns an RAII guard.
///
/// A streaming handler returns at once and keeps producing from a task, so
/// what it holds for the request (the worker's load guard, a rate-limit
/// reservation's safety net) must live as long as the response body does,
/// however the body ends: read to its end, dropped when the client leaves,
/// or consumed by another handler in-process. The guard rides in the body's
/// own stream and is dropped with it, after the stream. No wrapper goes
/// over the body for it: a body wrapper is a boxed `poll_frame` more on
/// every frame out.
pub struct GuardedStream<S, G> {
    inner: S,
    _guard: G,
}

impl<S, G> GuardedStream<S, G> {
    pub fn new(inner: S, guard: G) -> Self {
        Self {
            inner,
            _guard: guard,
        }
    }
}

impl<S: Stream + Unpin, G: Unpin> Stream for GuardedStream<S, G> {
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().inner).poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
