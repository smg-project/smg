//! The merged response stream of one request: its choices polled together,
//! ended early by the `Abort` RPC or by shutdown. Engine-neutral: each
//! servicer's `Generate` builds one over its own choice streams, and the
//! cancellation semantics ([`CancelReason`]) are decided here once.

use std::{
    collections::VecDeque,
    future::Future,
    ops::ControlFlow,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures::{stream::SelectAll, Stream};
use tokio::sync::oneshot;
use tonic::Status;

use crate::{
    metrics::{ChoiceMetrics, RequestMetrics},
    requests::{settle_shutdown, CancelReason, HeldTerminal, Registration, SHUTDOWN_MESSAGE},
};

/// A choice of a request's stream as the `Abort` RPC sees it (the shutdown's
/// view of it is [`HeldTerminal`]).
pub(crate) trait Choice:
    HeldTerminal + Stream<Item = Result<<Self as HeldTerminal>::Response, Status>> + Unpin
{
    /// End this choice on the caller's abort: the terminal it was about to
    /// yield, the engine's parked one, or a synthesized `abort` one; `None`
    /// once it has already ended. Dropping the engine stream aborts the
    /// engine side.
    fn abort(&mut self) -> Option<Self::Response>;
}

/// Poll the engine stream behind a choice. `Continue` carries the engine's
/// next item; `Break` carries what the choice's own `poll_next` returns in
/// its place: pending, the end of the stream, or its error, the engine
/// stream dropped on the last two so the choice stays ended, and the error
/// recorded on the choice's `metrics`.
pub(crate) fn poll_engine_item<S, T, R>(
    inner: &mut Option<S>,
    metrics: &mut ChoiceMetrics,
    cx: &mut Context<'_>,
) -> ControlFlow<Poll<Option<Result<R, Status>>>, T>
where
    S: Stream<Item = Result<T, Status>> + Unpin,
{
    let Some(stream) = inner.as_mut() else {
        return ControlFlow::Break(Poll::Ready(None));
    };
    match Pin::new(stream).poll_next(cx) {
        Poll::Pending => ControlFlow::Break(Poll::Pending),
        Poll::Ready(None) => {
            *inner = None;
            ControlFlow::Break(Poll::Ready(None))
        }
        Poll::Ready(Some(Err(status))) => {
            *inner = None;
            metrics.failed(status.code());
            ControlFlow::Break(Poll::Ready(Some(Err(status))))
        }
        Poll::Ready(Some(Ok(item))) => ControlFlow::Continue(item),
    }
}

/// The choices of one request: the one choice of a request without fan-out,
/// polled in place, or the `SelectAll` merge of an `n > 1` fan-out. The merge
/// re-queues a choice behind every item it yields (a task node allocated and
/// the choice moved out and back in per output), which a single choice never
/// needs; that choice lives inline in the request stream, boxed once with it.
pub(crate) enum Choices<C> {
    /// The one choice of a request without fan-out; `None` once it ended.
    One(Option<C>),
    /// The `n > 1` fan-out, polled together and yielding as ready.
    Many(SelectAll<C>),
}

impl<C: Stream + Unpin> Choices<C> {
    /// No choice yet; [`push`](Self::push) adds them.
    pub(crate) fn new() -> Self {
        Self::One(None)
    }

    /// Add a choice: the first stays in place, the second turns the two into
    /// a merge.
    pub(crate) fn push(&mut self, choice: C) {
        match self {
            Self::Many(many) => many.push(choice),
            Self::One(None) => *self = Self::One(Some(choice)),
            Self::One(first) => {
                let mut many = SelectAll::new();
                if let Some(first) = first.take() {
                    many.push(first);
                }
                many.push(choice);
                *self = Self::Many(many);
            }
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &C> {
        match self {
            Self::One(choice) => ChoicesIter::One(choice.iter()),
            Self::Many(many) => ChoicesIter::Many(many.iter()),
        }
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut C> {
        match self {
            Self::One(choice) => ChoicesIter::One(choice.iter_mut()),
            Self::Many(many) => ChoicesIter::Many(many.iter_mut()),
        }
    }

    /// Drop every choice, which aborts their engine side.
    pub(crate) fn clear(&mut self) {
        match self {
            Self::One(choice) => *choice = None,
            Self::Many(many) => many.clear(),
        }
    }
}

/// The iterator over either shape of [`Choices`].
enum ChoicesIter<A, B> {
    One(A),
    Many(B),
}

impl<A: Iterator, B: Iterator<Item = A::Item>> Iterator for ChoicesIter<A, B> {
    type Item = A::Item;

    fn next(&mut self) -> Option<A::Item> {
        match self {
            Self::One(iter) => iter.next(),
            Self::Many(iter) => iter.next(),
        }
    }
}

impl<C: Stream + Unpin> Stream for Choices<C> {
    type Item = C::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<C::Item>> {
        match self.get_mut() {
            Self::One(choice) => {
                let Some(stream) = choice.as_mut() else {
                    return Poll::Ready(None);
                };
                let polled = Pin::new(stream).poll_next(cx);
                if matches!(polled, Poll::Ready(None)) {
                    // Ended: dropped, as the merge drops an ended stream.
                    *choice = None;
                }
                polled
            }
            Self::Many(many) => Pin::new(many).poll_next(cx),
        }
    }
}

/// The merged response stream of one request: its choices polled together
/// (interleaved, each tagged with its `index`), ended early by the `Abort`
/// RPC or by shutdown. `H` is whatever else must live as long as the stream
/// (a servicer's load-tracker guard); nothing, by default.
pub(crate) struct RequestStream<C: Choice, H = ()> {
    choices: Choices<C>,
    cancel: oneshot::Receiver<CancelReason>,
    /// Set once aborted: the terminal `Complete` of every choice still open
    /// at that point, yielded before the stream ends.
    aborted: Option<VecDeque<C::Response>>,
    /// The status a shutdown ends the stream with once `aborted` is drained,
    /// when it cut a choice still open; `None` ends it plainly.
    cut: Option<Status>,
    /// The request's observation, recorded when the stream and its choices
    /// are gone.
    metrics: Arc<RequestMetrics>,
    _registration: Registration,
    _held: H,
}

impl<C: Choice> RequestStream<C> {
    /// The stream over `choices`, ended early by `cancel` (the registry's
    /// handle for the request), keeping the request registered and its
    /// `metrics` observed for as long as it lives.
    pub(crate) fn new(
        choices: Choices<C>,
        cancel: oneshot::Receiver<CancelReason>,
        registration: Registration,
        metrics: Arc<RequestMetrics>,
    ) -> Self {
        Self {
            choices,
            cancel,
            aborted: None,
            cut: None,
            metrics,
            _registration: registration,
            _held: (),
        }
    }

    /// The same stream, holding `held` for its lifetime.
    pub(crate) fn holding<H>(self, held: H) -> RequestStream<C, H> {
        RequestStream {
            choices: self.choices,
            cancel: self.cancel,
            aborted: self.aborted,
            cut: self.cut,
            metrics: self.metrics,
            _registration: self._registration,
            _held: held,
        }
    }
}

impl<C: Choice, H: Unpin> Stream for RequestStream<C, H>
where
    C::Response: Unpin,
{
    type Item = Result<C::Response, Status>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(aborted) = this.aborted.as_mut() {
                if let Some(response) = aborted.pop_front() {
                    return Poll::Ready(Some(Ok(response)));
                }
                // The completions the choices already held went out; a
                // shutdown that cut a choice still open ends the stream with
                // its status, a plain abort just ends it.
                return Poll::Ready(this.cut.take().map(|status| {
                    this.metrics.failed(status.code());
                    Err(status)
                }));
            }
            // The cancellation names its cause. An `Abort` RPC is the caller's
            // own abort: the request ends as the Python servicer ends it, with
            // each open choice's `Complete(finish_reason = "abort")`. A
            // shutdown with the request still open is its engine side going
            // away under it (the lifecycle owner stopping the pair after the
            // engine exited, a drain whose grace ran out): no completion the
            // caller could read as a short answer, so the stream ends with
            // UNAVAILABLE, which the Router classes as the worker's failure.
            // Either way dropping the choices aborts the engine side.
            if let Poll::Ready(reason) = Pin::new(&mut this.cancel).poll(cx) {
                if reason == Ok(CancelReason::Shutdown) {
                    // A choice the engine (or a stop string) finished before
                    // this poll keeps its completion; only a choice still
                    // open is cut, and only then does the stream end with the
                    // status (see `settle_shutdown`).
                    let (finished, cut) = settle_shutdown(&mut this.choices);
                    this.aborted = Some(finished);
                    if cut {
                        this.metrics.cut();
                    }
                    this.cut = cut.then(|| Status::unavailable(SHUTDOWN_MESSAGE));
                    continue;
                }
                // The abort, or a sender dropped without a word (the registry
                // entry went away): the completion it always was.
                let aborted = this.choices.iter_mut().filter_map(C::abort).collect();
                this.choices.clear();
                this.aborted = Some(aborted);
                continue;
            }
            return Pin::new(&mut this.choices).poll_next(cx);
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/stream/tests.rs"]
mod tests;
