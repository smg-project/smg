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
mod tests {
    use std::{
        cell::RefCell,
        future::{pending, ready},
        pin::pin,
        rc::Rc,
        task::Waker,
    };

    use axum::body::Body;

    use super::*;

    /// A duty that writes what it was told into a shared log.
    struct Logged {
        log: Rc<RefCell<Vec<&'static str>>>,
    }

    impl Duty for Logged {
        fn answered<E>(self, answer: Result<Response, E>) -> Result<Response, E> {
            self.log.borrow_mut().push("answered");
            answer
        }

        fn unanswered(self) {
            self.log.borrow_mut().push("unanswered");
        }
    }

    /// A duty with the default `unanswered`: dropped, nothing more; its
    /// drop writes to the log.
    struct Silent {
        log: Rc<RefCell<Vec<&'static str>>>,
    }

    impl Drop for Silent {
        fn drop(&mut self) {
            self.log.borrow_mut().push("dropped");
        }
    }

    impl Duty for Silent {
        fn answered<E>(self, answer: Result<Response, E>) -> Result<Response, E> {
            answer
        }
    }

    fn poll_once<F: Future>(future: &mut Pin<&mut F>) -> Poll<F::Output> {
        future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()))
    }

    fn answer() -> Response {
        Response::new(Body::empty())
    }

    #[test]
    fn the_answer_is_handed_to_the_duty_once() {
        let log = Rc::new(RefCell::new(Vec::new()));
        {
            let mut future = pin!(DutyFuture::new(
                ready(Ok::<_, ()>(answer())),
                Logged {
                    log: Rc::clone(&log)
                }
            ));
            assert!(matches!(poll_once(&mut future), Poll::Ready(Ok(_))));
        }
        assert_eq!(*log.borrow(), ["answered"]);
    }

    #[test]
    fn dropped_before_the_answer_the_duty_is_told() {
        let log = Rc::new(RefCell::new(Vec::new()));
        {
            let mut future = pin!(DutyFuture::new(
                pending::<Result<Response, ()>>(),
                Logged {
                    log: Rc::clone(&log)
                }
            ));
            assert!(poll_once(&mut future).is_pending());
        }
        assert_eq!(*log.borrow(), ["unanswered"]);
    }

    #[test]
    fn a_duty_without_a_word_for_the_unanswered_is_dropped_with_the_future() {
        let log = Rc::new(RefCell::new(Vec::new()));
        drop(DutyFuture::new(
            pending::<Result<Response, ()>>(),
            Silent {
                log: Rc::clone(&log),
            },
        ));
        assert_eq!(*log.borrow(), ["dropped"]);
    }

    /// A route future whose drop writes to the log.
    struct LoggedRoute {
        log: Rc<RefCell<Vec<&'static str>>>,
    }

    impl Future for LoggedRoute {
        type Output = Result<Response, ()>;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for LoggedRoute {
        fn drop(&mut self) {
            self.log.borrow_mut().push("inner dropped");
        }
    }

    #[test]
    fn the_duty_is_told_before_the_routes_future_is_dropped() {
        let log = Rc::new(RefCell::new(Vec::new()));
        drop(DutyFuture::new(
            LoggedRoute {
                log: Rc::clone(&log),
            },
            Logged {
                log: Rc::clone(&log),
            },
        ));
        assert_eq!(*log.borrow(), ["unanswered", "inner dropped"]);
    }
}
