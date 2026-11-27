//! A layer's future over the route's: the route's answer handed once to
//! what the layer holds for the request (its duty), or the duty told that
//! the route never answered when the future is dropped first. The request
//! span and the HTTP metrics layers are the two futures of this shape.

use std::{
    future::Future,
    pin::Pin,
    task::{ready, Context, Poll},
};

use axum::response::Response;
use pin_project_lite::pin_project;

/// What a layer holds for one request between its arrival and the route's
/// answer.
pub trait Duty {
    /// The route answered: the layer's bookkeeping of the answer, which
    /// passes through, wrapped or not.
    fn answered<E>(self, answer: Result<Response, E>) -> Result<Response, E>;

    /// The route never answered: the request's future was dropped with the
    /// duty still held (what the server does with a request whose client
    /// closed the connection). Dropping the duty is the default.
    fn unanswered(self)
    where
        Self: Sized,
    {
    }
}

pin_project! {
    /// A layer's future: the route's future, then the layer's duty handed
    /// the answer, taken at the answer. Dropped before the answer, the duty
    /// is told so before the route's future is dropped: the layer's word
    /// about the close comes before the route's own teardown.
    pub struct DutyFuture<F, D>
    where
        D: Duty,
    {
        #[pin]
        inner: F,
        duty: Option<D>,
    }

    impl<F, D> PinnedDrop for DutyFuture<F, D>
    where
        D: Duty,
    {
        fn drop(this: Pin<&mut Self>) {
            // Dropped with the duty still held: the route never answered.
            if let Some(duty) = this.project().duty.take() {
                duty.unanswered();
            }
        }
    }
}

impl<F, D: Duty> DutyFuture<F, D> {
    /// The route's future with the layer's duty beside it: no box around
    /// either, the layer's future is the route's own plus the duty.
    pub(super) fn new(inner: F, duty: D) -> Self {
        Self {
            inner,
            duty: Some(duty),
        }
    }
}

impl<F, D, E> Future for DutyFuture<F, D>
where
    F: Future<Output = Result<Response, E>>,
    D: Duty,
{
    type Output = Result<Response, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let answer = ready!(this.inner.poll(cx));
        Poll::Ready(match this.duty.take() {
            Some(duty) => duty.answered(answer),
            // A poll after the answer breaks the caller's contract; the
            // route's word to it passes through.
            None => answer,
        })
    }
}

#[cfg(test)]
#[path = "../../tests/unit/middleware/duty/tests.rs"]
mod tests;
