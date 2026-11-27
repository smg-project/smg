//! The answer to a request that is still waiting for its handler when the
//! graceful shutdown's drain gives up: `503` in the gateway's error envelope
//! with the code [`SERVER_SHUTTING_DOWN_CODE`]. Without it the request's
//! future is dropped with its connection when the listeners close what still
//! stands, and its client reads a connection closed with nothing on it,
//! which it cannot tell from a crash.
//!
//! The layer sits inside the Messages envelope, the access log and the HTTP
//! metrics: the answer leaves in the route's error shape, is logged and is
//! counted as a handler's answer. While the gateway serves it decides
//! nothing: the handler is polled first and its answer always wins; a
//! request registers for the wake when its handler first parks and reads one
//! flag at each later poll. A response whose head is out (a stream) has left
//! the layer and is not touched.

use std::{
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
};

use axum::{extract::Request, response::Response};
use pin_project_lite::pin_project;
use tokio::sync::futures::OwnedNotified;
use tower::{Layer, Service};

use crate::{observability::inflight_tracker::InFlightRequestTracker, routers::error};

/// The `code` of the `503` a request gets when the shutdown's drain gave up
/// on it before its handler answered.
pub const SERVER_SHUTTING_DOWN_CODE: &str = "server_shutting_down";

/// Answers the requests the shutdown's drain gave up on: see the module.
#[derive(Clone)]
pub struct ShutdownNoticeLayer {
    tracker: Arc<InFlightRequestTracker>,
}

impl ShutdownNoticeLayer {
    pub fn new(tracker: Arc<InFlightRequestTracker>) -> Self {
        Self { tracker }
    }
}

impl<S> Layer<S> for ShutdownNoticeLayer {
    type Service = ShutdownNotice<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ShutdownNotice {
            inner,
            tracker: Arc::clone(&self.tracker),
        }
    }
}

/// The service [`ShutdownNoticeLayer`] wraps the routes in.
#[derive(Clone)]
pub struct ShutdownNotice<S> {
    inner: S,
    tracker: Arc<InFlightRequestTracker>,
}

impl<S> Service<Request> for ShutdownNotice<S>
where
    S: Service<Request, Response = Response, Error = Infallible>,
{
    type Response = Response;
    type Error = Infallible;
    type Future = NoticeFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        NoticeFuture {
            inner: self.inner.call(request),
            tracker: Arc::clone(&self.tracker),
            given_up: None,
            registered: None,
        }
    }
}

pin_project! {
    /// A route's response, or the shutdown's `503` once the drain gave up
    /// while the handler still waited. The handler's future runs in place;
    /// dropped with this one, it ends its call to the worker.
    pub struct NoticeFuture<F> {
        #[pin]
        inner: F,
        tracker: Arc<InFlightRequestTracker>,
        // The wake at the giving up, made when the handler first parks.
        #[pin]
        given_up: Option<OwnedNotified>,
        // The waker that wake holds: registered again under another alone.
        registered: Option<Waker>,
    }
}

impl<F> Future for NoticeFuture<F>
where
    F: Future<Output = Result<Response, Infallible>>,
{
    type Output = Result<Response, Infallible>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        if let Poll::Ready(answer) = this.inner.poll(cx) {
            return Poll::Ready(answer);
        }
        // The handler waits, and what it waits for may never move: the
        // giving up itself has to wake this request. Registered before the
        // flag is read, so a giving up on either side of it is seen.
        let waker = cx.waker();
        if !this
            .registered
            .as_ref()
            .is_some_and(|held| held.will_wake(waker))
        {
            if this.given_up.is_none() {
                this.given_up.set(Some(this.tracker.drain_given_up()));
            }
            if let Some(given_up) = this.given_up.as_mut().as_pin_mut() {
                if given_up.poll(cx).is_ready() {
                    return Poll::Ready(Ok(shutting_down()));
                }
            }
            *this.registered = Some(waker.clone());
        }
        if this.tracker.gave_up_drain() {
            return Poll::Ready(Ok(shutting_down()));
        }
        Poll::Pending
    }
}

fn shutting_down() -> Response {
    error::service_unavailable(
        SERVER_SHUTTING_DOWN_CODE,
        "The server is shutting down and did not complete the request",
    )
}
