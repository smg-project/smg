//! The body of a streamed response fed by its own reading: the frames the
//! response stage writes reach the connection from the task that polls the
//! body, with no task of their own and no hand-off between tasks.
//!
//! A streaming handler builds the reading of the engine's answer (the
//! response stage, writing through an [`SseSender`]) and the body polls it
//! in place. Each poll of the body first takes a frame the reading already
//! wrote; when there is none, it drives the reading until it parks (on the
//! engine, or on the channel) and takes what the reading wrote before
//! parking. The channel is the stages' own SSE channel with room for ONE
//! frame ([`driven_channel`]): both ends live in one task, so a send is a
//! push that wakes nobody, and a frame is handed to the connection before
//! the reading writes a second one, as soon as it is written. The first
//! frame the body hands out is followed by one pending poll with the task
//! woken, so the connection writes it before the reading runs on: the
//! stream's first bytes wait for nothing behind them.
//!
//! A body dropped before the reading ends (the client left) detaches the
//! reading onto the runtime, with the channel closed first: the reading meets
//! the closed channel through its watch on the client or at its next send,
//! and ends as it does when a client leaves a task-driven stream (the engine
//! aborted, the usage settled, the metrics recorded). Nothing of the request
//! ends with the body.
//!
//! A panic in a reading is caught at the reading's own boundary by every
//! stage (`utils::isolation`): the request's error event and its end, not the
//! task's. Should one unwind through the body's poll all the same, the slot
//! is empty while the reading is polled: the unwind drops the reading, and
//! the body's drop spawns nothing.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use axum::response::Response;
use bytes::Bytes;
use futures::Stream;
use tokio::sync::mpsc::{self, error::TryRecvError};

use super::streaming::build_sse_response_from_stream;
use crate::{
    middleware::edge_body::BodyFailure,
    routers::common::{
        sse::{SseReceiver, SseSender},
        stream::GuardedStream,
    },
};

/// How a reading ended: with its stream complete, or with the client told of
/// a failure by the error event the stream ends with (before its `[DONE]`).
/// The body hands the word on to the edge ([`BodyFailure`]), which then
/// counts the response as a body that failed, what the client read, where
/// the frames alone would say it went out in full.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadingEnd {
    Complete,
    Failed,
}

impl ReadingEnd {
    /// The end of a reading that told the client of `result`'s failure, if
    /// it was one.
    pub(crate) fn of<T, E>(result: &Result<T, E>) -> Self {
        match result {
            Ok(_) => Self::Complete,
            Err(_) => Self::Failed,
        }
    }
}

/// The reading a streamed response's body drives: the response stage over
/// the channel's sender, boxed once (the allocation a spawned task made),
/// ending in its word on how the stream ended.
pub(crate) type Reading = Pin<Box<dyn Future<Output = ReadingEnd> + Send + 'static>>;

/// The channel of a body-driven stream: room for one frame. The reading
/// parks on its second send until the body took the first, so every frame
/// reaches the connection as soon as it is written, and nothing a stage
/// writes waits behind the processing of the messages after it. Every send
/// through it is awaited by the reading the body drives, never before the
/// body exists.
pub(crate) fn driven_channel() -> (SseSender, SseReceiver) {
    mpsc::channel(1)
}

/// A streamed response's body: the frames `reading` writes into the channel
/// `rx` receives from, read by polling the reading in place (see the module).
pub(crate) struct DrivenBody {
    /// `None` once the reading ended: its last frames are the channel's.
    reading: Option<Reading>,
    rx: SseReceiver,
    /// The reading's word when it ended failed, for the edge's count of the
    /// response (the response's cell, shared with the edge).
    failure: BodyFailure,
    /// The body handed its first frame out.
    first_frame_out: bool,
    /// The next poll is pending with the task woken, so the connection
    /// writes what it holds (the first frame) before the reading runs on.
    yield_once: bool,
}

impl DrivenBody {
    pub(crate) fn new(reading: Reading, rx: SseReceiver) -> Self {
        Self {
            reading: Some(reading),
            rx,
            failure: BodyFailure::default(),
            first_frame_out: false,
            yield_once: false,
        }
    }

    /// The reading's word on a failure it told the client of, shared: told
    /// when the reading ends failed, read by the edge at the body's end.
    pub(crate) fn failure(&self) -> &BodyFailure {
        &self.failure
    }

    /// `frame` on its way out: the first one is followed by a yield.
    fn hand_out(
        &mut self,
        frame: Result<Bytes, std::io::Error>,
    ) -> Poll<Option<Result<Bytes, std::io::Error>>> {
        if !self.first_frame_out {
            self.first_frame_out = true;
            self.yield_once = true;
        }
        Poll::Ready(Some(frame))
    }
}

impl Stream for DrivenBody {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.yield_once {
            this.yield_once = false;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        loop {
            // A frame the reading wrote already goes out before the reading
            // runs again.
            match this.rx.try_recv() {
                Ok(frame) => return this.hand_out(frame),
                Err(TryRecvError::Disconnected) => return Poll::Ready(None),
                Err(TryRecvError::Empty) => {}
            }
            let Some(mut reading) = this.reading.take() else {
                // Nothing writes any more: the channel's own end, once every
                // sender is gone.
                return this.rx.poll_recv(cx);
            };
            // The slot is empty while the reading is polled: a poll that
            // panics unwinds past this point with the reading dropped (a
            // future that panicked must not be polled again), so the drop
            // below has nothing to spawn and the unwind releases the stage's
            // state (its engine stream, its guards, its sender).
            match reading.as_mut().poll(cx) {
                // The reading ended: its last frames are read above, then
                // the channel's end. Its word on a failure it told the
                // client of is the edge's before those frames go out.
                Poll::Ready(end) => {
                    if end == ReadingEnd::Failed {
                        this.failure.tell();
                    }
                }
                // The reading parked, on the engine or on the channel, with
                // this task's waker registered where it parked; what it
                // wrote before parking is handed to the connection now. The
                // connection writes it before the reading runs on only after
                // the body's first frame (`yield_once`): after any later
                // frame the connection pulls again at once and the reading
                // runs on; a reading that needs such a frame written first
                // yields twice itself (the Messages stream's first token in
                // `StreamingProcessor::next_packed`: its first frame is not
                // the body's).
                Poll::Pending => {
                    this.reading = Some(reading);
                    return match this.rx.try_recv() {
                        Ok(frame) => this.hand_out(frame),
                        Err(TryRecvError::Disconnected) => Poll::Ready(None),
                        Err(TryRecvError::Empty) => Poll::Pending,
                    };
                }
            }
        }
    }
}

impl Drop for DrivenBody {
    fn drop(&mut self) {
        let Some(reading) = self.reading.take() else {
            return;
        };
        // The channel closes before the reading runs on: its watch on the
        // client fires and its next send fails, as when the body of a
        // task-driven stream is dropped. Outside a runtime (a body dropped
        // after the runtime's end) the reading is dropped with the body; its
        // stream's abort-on-drop still runs.
        self.rx.close();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(reading);
        }
    }
}

/// [`build_sse_response`](super::build_sse_response) whose body drives
/// `reading` (see [`DrivenBody`]) and owns `guards` for as long as it lives:
/// they ride in the body's own stream ([`GuardedStream`]) and are dropped
/// with it, with no wrapper over the body (a wrapper is a boxed
/// `poll_frame` more on every frame out). The reading's word on how it
/// ended goes to the edge in the response's cell, so a stream that told the
/// client of a failure counts as one.
pub(crate) fn build_sse_response_driven<G>(reading: Reading, rx: SseReceiver, guards: G) -> Response
where
    G: Send + Unpin + 'static,
{
    build_sse_response_driven_with(reading, rx, guards, build_sse_response_from_stream)
}

/// [`build_sse_response_driven`] whose response `respond` builds from the
/// body's guarded stream (a route that wraps the stream, as the chat route
/// re-chunks it, builds its response here so the reading's word still
/// reaches the edge).
pub(crate) fn build_sse_response_driven_with<G, F>(
    reading: Reading,
    rx: SseReceiver,
    guards: G,
    respond: F,
) -> Response
where
    G: Send + Unpin + 'static,
    F: FnOnce(GuardedStream<DrivenBody, G>) -> Response,
{
    let body = DrivenBody::new(reading, rx);
    let failure = body.failure().clone();
    let mut response = respond(GuardedStream::new(body, guards));
    failure.carry(&mut response);
    response
}

#[cfg(test)]
#[path = "../../../../../tests/unit/routers/grpc/common/responses/driven/tests.rs"]
mod tests;
