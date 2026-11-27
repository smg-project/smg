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
