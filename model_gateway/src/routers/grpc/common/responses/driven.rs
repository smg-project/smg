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
    routers::common::sse::{SseReceiver, SseSender},
    worker::GuardedStream,
};

/// The reading a streamed response's body drives: the response stage over
/// the channel's sender, boxed once (the allocation a spawned task made).
pub(crate) type Reading = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

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
            first_frame_out: false,
            yield_once: false,
        }
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
                // the channel's end.
                Poll::Ready(()) => {}
                // The reading parked, on the engine or on the channel, with
                // this task's waker registered where it parked; what it
                // wrote before parking goes out now.
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
/// `poll_frame` more on every frame out).
pub(crate) fn build_sse_response_driven<G>(reading: Reading, rx: SseReceiver, guards: G) -> Response
where
    G: Send + Unpin + 'static,
{
    build_sse_response_from_stream(GuardedStream::new(DrivenBody::new(reading, rx), guards))
}

#[cfg(test)]
mod tests {
    use std::{
        future::poll_fn,
        panic::AssertUnwindSafe,
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        },
        time::Duration,
    };

    use futures::StreamExt;
    use tokio::sync::{oneshot, Notify};

    use super::*;

    fn frame(text: &str) -> Bytes {
        Bytes::copy_from_slice(text.as_bytes())
    }

    async fn texts(body: DrivenBody) -> Vec<String> {
        body.map(|frame| String::from_utf8(frame.expect("a frame").to_vec()).expect("text"))
            .collect()
            .await
    }

    /// The frames come out in the order the reading wrote them, and the body
    /// ends with the reading: nothing is spawned for it.
    #[tokio::test]
    async fn the_frames_leave_in_order_and_the_body_ends_with_the_reading() {
        let (tx, rx) = driven_channel();
        let reading = async move {
            for text in ["one", "two", "three"] {
                tx.send(Ok(frame(text))).await.expect("the body reads");
            }
        };
        let body = DrivenBody::new(Box::pin(reading), rx);
        assert_eq!(texts(body).await, ["one", "two", "three"]);
    }

    /// A reading that writes many frames parks on the channel after each and
    /// runs on as the body drains: every frame comes out, in order.
    #[tokio::test]
    async fn a_reading_parked_on_the_channel_runs_on_as_the_body_drains() {
        let (tx, rx) = driven_channel();
        let count = 97;
        let reading = async move {
            for i in 0..count {
                tx.send(Ok(frame(&i.to_string())))
                    .await
                    .expect("the body reads");
            }
        };
        let body = DrivenBody::new(Box::pin(reading), rx);
        let expected: Vec<String> = (0..count).map(|i| i.to_string()).collect();
        assert_eq!(texts(body).await, expected);
    }

    /// The first frame is handed out before the reading writes a second one,
    /// and the poll after it is pending with the task woken, so the
    /// connection writes the first frame before the reading runs on; the
    /// frames after it come without a yield.
    #[tokio::test]
    async fn the_first_frame_is_handed_out_at_once_and_followed_by_one_yield() {
        let (tx, rx) = driven_channel();
        let written = Arc::new(AtomicUsize::new(0));
        let reading = {
            let written = written.clone();
            async move {
                for text in ["first", "second", "third"] {
                    tx.send(Ok(frame(text))).await.expect("the body reads");
                    written.fetch_add(1, Ordering::SeqCst);
                }
            }
        };
        let mut body = DrivenBody::new(Box::pin(reading), rx);
        let wakes = Arc::new(AtomicUsize::new(0));
        let waker = {
            let wakes = wakes.clone();
            futures::task::waker(Arc::new(CountingWake(wakes)))
        };
        let mut cx = Context::from_waker(&waker);
        let first = Pin::new(&mut body).poll_next(&mut cx);
        assert!(
            matches!(&first, Poll::Ready(Some(Ok(bytes))) if bytes == &frame("first")),
            "{first:?}"
        );
        assert!(
            written.load(Ordering::SeqCst) <= 2,
            "the reading wrote {} frames before the first was handed out: it parks on the second send",
            written.load(Ordering::SeqCst)
        );
        // The channel woke the task once already when the first frame left
        // it (its parked sender has room again): the yield's wake is the one
        // more.
        let woken_before_the_yield = wakes.load(Ordering::SeqCst);
        assert!(Pin::new(&mut body).poll_next(&mut cx).is_pending());
        assert_eq!(
            wakes.load(Ordering::SeqCst),
            woken_before_the_yield + 1,
            "the yield wakes the task once"
        );
        let second = Pin::new(&mut body).poll_next(&mut cx);
        assert!(
            matches!(&second, Poll::Ready(Some(Ok(bytes))) if bytes == &frame("second")),
            "{second:?}"
        );
        let third = Pin::new(&mut body).poll_next(&mut cx);
        assert!(
            matches!(&third, Poll::Ready(Some(Ok(bytes))) if bytes == &frame("third")),
            "no yield after the first frame: {third:?}"
        );
        assert!(poll_fn(|cx| Pin::new(&mut body).poll_next(cx))
            .await
            .is_none());
    }

    struct CountingWake(Arc<AtomicUsize>);

    impl futures::task::ArcWake for CountingWake {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A frame the reading writes after it parked reaches the body on the
    /// wake the reading registered, with the body polled from any task.
    #[tokio::test]
    async fn a_frame_written_after_a_park_reaches_the_body() {
        let (tx, rx) = driven_channel();
        let gate = Arc::new(Notify::new());
        let reading = {
            let gate = gate.clone();
            async move {
                tx.send(Ok(frame("first"))).await.expect("the body reads");
                gate.notified().await;
                tx.send(Ok(frame("second"))).await.expect("the body reads");
            }
        };
        let mut body = DrivenBody::new(Box::pin(reading), rx);
        let first = body.next().await.expect("a frame").expect("a frame");
        assert_eq!(first, frame("first"));
        let opener = async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            gate.notify_one();
        };
        let (second, ()) = tokio::join!(
            tokio::time::timeout(Duration::from_secs(5), body.next()),
            opener
        );
        let second = second
            .expect("the frame comes after the gate")
            .expect("a frame")
            .expect("a frame");
        assert_eq!(second, frame("second"));
        assert!(
            body.next().await.is_none(),
            "the body ends with the reading"
        );
    }

    /// A body dropped while its reading is parked detaches the reading: it
    /// runs to its end on the runtime, finding the channel closed (the
    /// client's leaving, as its watch and its sends read it).
    #[tokio::test]
    async fn a_body_dropped_before_the_readings_end_detaches_it_to_run_to_its_end() {
        let (tx, rx) = driven_channel();
        let gate = Arc::new(Notify::new());
        let (ended_tx, ended_rx) = oneshot::channel();
        let reading = {
            let gate = gate.clone();
            async move {
                tx.send(Ok(frame("first"))).await.expect("the body reads");
                gate.notified().await;
                let closed = tx.is_closed();
                let late_send_failed = tx.send(Ok(frame("late"))).await.is_err();
                let _ = ended_tx.send((closed, late_send_failed));
            }
        };
        let mut body = DrivenBody::new(Box::pin(reading), rx);
        let first = body.next().await.expect("a frame").expect("a frame");
        assert_eq!(first, frame("first"));
        drop(body);
        gate.notify_one();
        let (closed, late_send_failed) = tokio::time::timeout(Duration::from_secs(5), ended_rx)
            .await
            .expect("the reading ends in time")
            .expect("the reading reports its end");
        assert!(closed, "the channel is closed before the reading runs on");
        assert!(late_send_failed, "a send after the body's drop fails");
    }

    /// A reading that panics under the body's poll leaves the slot empty:
    /// the unwind drops the reading (a future that panicked must not be
    /// polled again), the body's drop has nothing to spawn, and the task
    /// that polled, the unwind caught as the runtime catches it, goes on
    /// with the body, which ends on its closed channel.
    #[tokio::test]
    async fn a_reading_that_panics_under_the_poll_leaves_the_slot_empty_and_nothing_to_spawn() {
        struct SetOnDrop(Arc<AtomicBool>);
        impl Drop for SetOnDrop {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let (tx, rx) = driven_channel();
        let dropped = Arc::new(AtomicBool::new(false));
        let reading = {
            let flag = SetOnDrop(dropped.clone());
            async move {
                let _flag = flag;
                let _tx = tx;
                panic!("the stage panicked under the poll");
            }
        };
        let mut body = DrivenBody::new(Box::pin(reading), rx);
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let unwound =
            std::panic::catch_unwind(AssertUnwindSafe(|| Pin::new(&mut body).poll_next(&mut cx)))
                .is_err();
        assert!(unwound, "the panic unwinds through the poll");
        assert!(body.reading.is_none(), "the slot is empty after the unwind");
        assert!(
            dropped.load(Ordering::SeqCst),
            "the unwind dropped the reading"
        );
        assert!(
            poll_fn(|cx| Pin::new(&mut body).poll_next(cx))
                .await
                .is_none(),
            "the body ends on the channel the dropped reading closed"
        );
        drop(body);
    }
}
