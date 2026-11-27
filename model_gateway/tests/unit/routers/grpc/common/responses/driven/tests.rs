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
use crate::middleware::edge_body::BodyDuties;

fn frame(text: &str) -> Bytes {
    Bytes::copy_from_slice(text.as_bytes())
}

async fn texts(body: DrivenBody) -> Vec<String> {
    body.map(|frame| String::from_utf8(frame.expect("a frame").to_vec()).expect("text"))
        .collect()
        .await
}

/// The reading's word reaches the body's failure: a reading that ends
/// failed tells it, before its last frames go out; one that ends
/// complete tells nothing.
#[tokio::test]
async fn a_reading_that_ends_failed_tells_the_bodys_failure() {
    for (end, told) in [(ReadingEnd::Failed, true), (ReadingEnd::Complete, false)] {
        let (tx, rx) = driven_channel();
        let reading = async move {
            tx.send(Ok(frame("data: {\"error\":{}}\n\n")))
                .await
                .expect("the body reads");
            tx.send(Ok(frame("data: [DONE]\n\n")))
                .await
                .expect("the body reads");
            end
        };
        let body = DrivenBody::new(Box::pin(reading), rx);
        let failure = body.failure().clone();
        assert!(!failure.told(), "nothing told before the reading ran");
        assert_eq!(texts(body).await.len(), 2, "{end:?}");
        assert_eq!(failure.told(), told, "{end:?}");
    }
}

/// The response over a driven body hands the reading's word to the edge
/// in the response's cell, told as the reading ends.
#[tokio::test]
async fn the_driven_response_hands_the_failure_to_the_edge() {
    let (tx, rx) = driven_channel();
    let reading = async move {
        let _ = tx.send(Ok(frame("data: [DONE]\n\n"))).await;
        ReadingEnd::Failed
    };
    let mut response = build_sse_response_driven(Box::pin(reading), rx, ());
    let mut duties = BodyDuties::take(&mut response);
    let failure = duties.take_failure().expect("the failure handed over");
    assert!(!failure.told(), "nothing told before the body is read");
    let body = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .expect("the body");
    assert_eq!(body.as_ref(), b"data: [DONE]\n\n");
    assert!(failure.told(), "told as the reading ended");
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
        ReadingEnd::Complete
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
        ReadingEnd::Complete
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
            ReadingEnd::Complete
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
            ReadingEnd::Complete
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
            ReadingEnd::Complete
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
