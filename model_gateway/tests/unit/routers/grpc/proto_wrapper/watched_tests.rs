use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use smg_grpc_client::vllm_proto as vllm;
use smg_worker::identity::basic::Worker;
use tokio::sync::oneshot;

use super::{ClientGone, PrefillCachedTokens, ProtoGenerateComplete, ProtoStream};
use crate::routers::{
    common::sse::{sse_channel, SseReceiver},
    grpc::{
        backend_client::BasicWorkerBuilder,
        regular::streaming::tests::eof::{
            chunk, complete, gated_stream, scripted_stream, staged_stream,
        },
    },
};

/// A stream whose `first` frames come at once and whose `rest` wait for
/// the returned gate, read for the client at the receiving end of the
/// returned receiver.
async fn watched(
    first: Vec<vllm::GenerateResponse>,
    rest: Vec<vllm::GenerateResponse>,
) -> (ProtoStream, oneshot::Sender<()>, SseReceiver) {
    let (gate_tx, gate_rx) = oneshot::channel();
    let (stream, _server) = gated_stream(first, gate_rx, rest, "0").await;
    let (client, rx) = sse_channel();
    (
        stream.until_client_gone(&client, "req-watched"),
        gate_tx,
        rx,
    )
}

/// A client that leaves while the stream waits on the engine ends the
/// wait at once: the reader meets the end, not the engine's next frame,
/// and nothing after it.
#[tokio::test]
async fn a_client_that_leaves_while_the_stream_is_parked_ends_it_at_once() {
    // A tracked layer under the watch: its worker and its load pile show
    // the inner stream going with the client.
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://w1:8000").build());
    let (_gate_tx, gate_rx) = oneshot::channel();
    let (stream, _server) =
        gated_stream(vec![chunk(0, "a")], gate_rx, vec![complete(0, "stop")], "0").await;
    let (client, rx) = sse_channel();
    let mut stream = stream
        .tracked(Arc::clone(&worker), 0, true)
        .until_client_gone(&client, "req-watched");
    assert!(stream.worker().is_some(), "tracked under the watch");
    assert_eq!(worker.runtime_state().unwrap().tracked_load(), 1);
    assert!(
        matches!(stream.next().await, Some(Ok(_))),
        "the first response is read"
    );
    assert!(!stream.client_left());
    let (item, ()) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(stream.next(), async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            drop(rx);
        })
    })
    .await
    .expect("the gate never opens; the client's leaving ends the wait");
    assert!(item.is_none(), "the client's leaving is the end");
    assert!(stream.client_left(), "the end was the client's leaving");
    assert!(
        stream.next().await.is_none(),
        "nothing after the client left"
    );
    assert!(
        stream.worker().is_none(),
        "the inner stream went with the client"
    );
    assert_eq!(
        worker.runtime_state().unwrap().tracked_load(),
        0,
        "its load released with it"
    );
}

/// The client's leaving is seen through a decode-side wrapper too: dual
/// dispatch wraps the watched stream in one once the prefill side has
/// drained, and the reader asks the outer layer.
#[tokio::test]
async fn a_client_that_leaves_is_seen_through_the_decode_side_wrapper() {
    let (stream, _gate, rx) = watched(vec![chunk(0, "a")], vec![complete(0, "stop")]).await;
    let mut prefill = PrefillCachedTokens::default();
    prefill.note(&ProtoGenerateComplete::Vllm(Box::new(
        vllm::GenerateComplete {
            prompt_tokens: 4,
            cached_tokens: 2,
            ..Default::default()
        },
    )));
    let mut stream = stream.decode_side(prefill);
    assert!(matches!(stream, ProtoStream::DecodeSide(_)), "wrapped");
    assert!(
        matches!(stream.next().await, Some(Ok(_))),
        "the first response is read through the wrapper"
    );
    assert!(!stream.client_left());
    let (item, ()) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(stream.next(), async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            drop(rx);
        })
    })
    .await
    .expect("the gate never opens; the client's leaving ends the wait");
    assert!(item.is_none(), "the client's leaving is the end");
    assert!(
        stream.client_left(),
        "the end was the client's leaving, seen through the wrapper"
    );
}

/// A decode side restamps a watched stream's responses whichever layer
/// is outside: the wrapper over the watch, as dual dispatch builds it,
/// and the watch over the wrapper.
#[tokio::test]
async fn a_decode_side_restamps_over_a_watch_and_under_one() {
    let prefill = || {
        let mut prefill = PrefillCachedTokens::default();
        prefill.note(&ProtoGenerateComplete::Vllm(Box::new(
            vllm::GenerateComplete {
                prompt_tokens: 4,
                cached_tokens: 2,
                ..Default::default()
            },
        )));
        prefill
    };
    let (stream, gate, _rx) = watched(vec![chunk(0, "a")], vec![complete(0, "stop")]).await;
    let mut over = stream.decode_side(prefill());
    let (under_gate, gate_rx) = oneshot::channel();
    let (stream, _server) =
        gated_stream(vec![chunk(0, "a")], gate_rx, vec![complete(0, "stop")], "0").await;
    let (client, _under_rx) = sse_channel();
    let mut under = stream
        .decode_side(prefill())
        .until_client_gone(&client, "req-watched");
    gate.send(()).expect("the stream holds its gate");
    under_gate.send(()).expect("the stream holds its gate");
    for stream in [&mut over, &mut under] {
        for _ in 0..2 {
            let response = stream.next().await.expect("a response");
            let response = response.expect("no error");
            assert_eq!(response.cached_tokens(), 2, "the prefill's hit");
        }
        assert!(stream.next().await.is_none(), "the engine's end");
    }
}

/// The wait for the client's leaving is one for the stream's whole life,
/// polled once, when the reader first parks on the engine: a response
/// that is ready costs no poll of it, a parked reader polled again does
/// not poll it again, and a client that leaves during a later park still
/// ends the wait at once.
#[tokio::test]
#[expect(clippy::disallowed_methods, reason = "test infrastructure")]
async fn the_wait_for_the_client_is_polled_while_parked_only_and_kept_across_parks() {
    let (first_gate, first_gate_rx) = oneshot::channel();
    let (_second_gate, second_gate_rx) = oneshot::channel();
    let (stream, _server) = staged_stream(
        vec![
            (None, vec![chunk(0, "a"), chunk(0, "b"), chunk(0, "c")]),
            (Some(first_gate_rx), vec![chunk(0, "d"), chunk(0, "e")]),
            (Some(second_gate_rx), vec![complete(0, "stop")]),
        ],
        "0",
    )
    .await;
    let (client, rx) = sse_channel();
    // The sender's `closed()`, as `until_client_gone` keeps it, with its
    // polls counted.
    let polls = Arc::new(AtomicUsize::new(0));
    let client_gone = {
        let polls = Arc::clone(&polls);
        let sender = client.clone();
        let mut closed = Box::pin(async move { sender.closed().await });
        Box::pin(std::future::poll_fn(move |cx| {
            polls.fetch_add(1, Ordering::SeqCst);
            closed.as_mut().poll(cx)
        }))
    };
    let client_gone = ClientGone {
        client,
        wait: client_gone,
        waker: None,
    };
    let mut stream = stream.watched("req-watched", client_gone);
    for _ in 0..3 {
        assert!(matches!(stream.next().await, Some(Ok(_))));
    }
    assert_eq!(
        polls.load(Ordering::SeqCst),
        0,
        "a ready response costs no poll of the wait"
    );
    // The reader parks on a task of its own, as a connection's task
    // does, and a timer beside it polls it a second time in each park.
    let reader = tokio::spawn({
        let polls = Arc::clone(&polls);
        async move {
            let gap = Duration::from_millis(20);
            let (item, ()) = tokio::join!(stream.next(), tokio::time::sleep(gap));
            assert!(matches!(item, Some(Ok(_))), "the gap is waited out");
            let parked = polls.load(Ordering::SeqCst);
            assert!(
                matches!(stream.next().await, Some(Ok(_))),
                "the response behind the gate is ready"
            );
            let (item, ()) = tokio::join!(stream.next(), tokio::time::sleep(gap));
            assert!(item.is_none(), "the client's leaving is the end");
            (parked, stream)
        }
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    let _ = first_gate.send(());
    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(rx);
    let (parked, mut stream) = tokio::time::timeout(Duration::from_secs(5), reader)
        .await
        .expect("the second gate never opens; the client's leaving ends the wait")
        .expect("the reader's task ends");
    assert_eq!(parked, 1, "the parked reader polled the wait once");
    assert!(stream.client_left(), "the end was the client's leaving");
    assert_eq!(
        polls.load(Ordering::SeqCst),
        parked,
        "the first park's registration ended the later one"
    );
    assert!(
        stream.next().await.is_none(),
        "nothing after the client left"
    );
    // A stream parked here and read on by another task is registered for
    // that task's waker: the client's leaving wakes it there.
    let (mut handed, _gate, rx) = watched(vec![], vec![complete(0, "stop")]).await;
    assert!(futures::poll!(std::pin::pin!(handed.next())).is_pending());
    let reader = tokio::spawn(async move { handed.next().await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    drop(rx);
    let item = tokio::time::timeout(Duration::from_secs(5), reader)
        .await
        .expect("the gate never opens; the client's leaving wakes the reader's task")
        .expect("the reader's task ends");
    assert!(item.is_none(), "the client's leaving is the end there too");
}

/// A client that left before the stream was first polled reads nothing:
/// the response the engine has ready is not read for it.
#[tokio::test]
async fn a_client_that_left_before_the_first_poll_reads_nothing() {
    let (mut stream, _gate, rx) = watched(vec![chunk(0, "a")], vec![]).await;
    drop(rx);
    assert!(stream.next().await.is_none());
    assert!(stream.next().await.is_none());
}

/// A reader's own stop finds the layer that tells a stream's end under
/// each layer a reader's stream has: the tracked layer, the watch for
/// its client and a decode side's stamp, in a reader's order and in
/// another, and an item pulled ahead of its reader. Grand Central
/// hears the worker's 200 each time.
#[test]
fn a_readers_stop_tells_the_end_under_each_layer_of_its_stream() {
    use super::{AheadStream, FanoutStream};
    use crate::routers::common::placement::{tests::heard_request, RequestEnd};

    let (client, _body) = sse_channel();
    let stamp = || {
        let mut prefill = PrefillCachedTokens::default();
        prefill.note(&ProtoGenerateComplete::Vllm(Box::new(
            vllm::GenerateComplete {
                cached_tokens: 2,
                ..Default::default()
            },
        )));
        prefill
    };
    let layered = |end: RequestEnd| {
        let mut stream = ProtoStream::Fanout(FanoutStream::new(Vec::new()));
        stream.report_through(end);
        stream
    };
    let tracked = |end: RequestEnd| {
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://e:30000").build());
        let mut stream =
            ProtoStream::Fanout(FanoutStream::new(Vec::new())).tracked(worker, 0, false);
        stream.report_through(end);
        stream
    };
    let watched = |end: RequestEnd| tracked(end).until_client_gone(&client, "req-stopped");
    let shapes: [(&str, &dyn Fn(RequestEnd) -> ProtoStream); 6] = [
        ("the layer alone", &layered),
        ("under an item pulled ahead", &|end| {
            ProtoStream::Ahead(Box::new(AheadStream {
                inner: layered(end),
                first: None,
            }))
        }),
        ("under the tracked layer", &tracked),
        ("under the watch", &watched),
        ("under a decode side's stamp", &|end| {
            watched(end).decode_side(stamp())
        }),
        ("under a stamp that is watched", &|end| {
            tracked(end)
                .decode_side(stamp())
                .until_client_gone(&client, "req-stopped")
        }),
    ];
    for (shape, stream) in shapes {
        let (heard, guard, end) = heard_request();
        stream(end).abort_past_stop();
        drop(guard);
        assert_eq!(
            *heard.lock().expect("the stage heard"),
            ["issued", "dispatched Single", "completed Ok 200"],
            "{shape}"
        );
    }
}

/// The layer goes under a stream's tracked layer, so its readers still
/// meet a tracked stream: it names its worker, and its first item is
/// read ahead and held back for its reader.
#[tokio::test]
async fn the_layer_goes_under_the_tracked_layer() {
    use super::{FanoutStream, FirstItem};
    use crate::routers::common::placement::tests::heard_request;

    let (heard, guard, end) = heard_request();
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://e:30000").build());
    let mut stream = ProtoStream::Fanout(FanoutStream::new(Vec::new())).tracked(worker, 0, false);
    stream.report_through(end);
    assert!(stream.worker().is_some(), "the stream names its worker");
    assert_eq!(stream.first_response().await, FirstItem::End);
    assert!(stream.next().await.is_none(), "the end was held back");

    drop((stream, guard));
    assert_eq!(
        *heard.lock().expect("the stage heard"),
        ["issued", "dispatched Single", "completed Ok 200"]
    );
}

/// A pair's decode stream comes to the stage stamped where its prefill
/// reported a prefix-cache hit, and with no stamp where the readers
/// stamp it behind the stage. The layer lies in one place for both:
/// under the stamp, and under the tracked layer.
#[tokio::test]
async fn the_layer_goes_under_a_decode_sides_stamp_too() {
    use super::FanoutStream;
    use crate::routers::common::placement::tests::heard_request;

    let mut prefill = PrefillCachedTokens::default();
    prefill.note(&ProtoGenerateComplete::Vllm(Box::new(
        vllm::GenerateComplete {
            cached_tokens: 2,
            ..Default::default()
        },
    )));
    let (heard, guard, end) = heard_request();
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://d:30000").build());
    let mut stream = ProtoStream::Fanout(FanoutStream::new(Vec::new()))
        .tracked_decode_side(worker)
        .decode_side(prefill);
    stream.report_through(end);
    let ProtoStream::DecodeSide(side) = &stream else {
        panic!("the stamp stays the outermost layer");
    };
    let ProtoStream::Tracked(tracked) = &side.inner else {
        panic!("the tracked layer stays under the stamp");
    };
    assert!(
        matches!(tracked.inner, ProtoStream::Reported(_)),
        "the layer lies under the tracked layer"
    );
    assert!(stream.next().await.is_none());

    drop((stream, guard));
    assert_eq!(
        *heard.lock().expect("the stage heard"),
        ["issued", "dispatched Single", "completed Ok 200"]
    );
}

/// A tracked stream may hold its first item before the layer is put: a
/// decode sent behind its prefill is waited for within its bound. The layer
/// meets that item where it is put. A failure is told by its status, and an
/// end with nothing on the stream as a clean end, each once. The item is
/// still handed to the stream's reader.
#[tokio::test]
async fn the_layer_meets_a_first_item_the_tracked_layer_read_ahead_of_it() {
    use super::FirstItem;
    use crate::routers::common::placement::tests::heard_request;

    for (status, first, end_heard) in [
        ("13", FirstItem::Failure, "completed ServerError 500"),
        ("0", FirstItem::End, "completed Ok 200"),
    ] {
        let (heard, guard, end) = heard_request();
        let (stream, _server) = scripted_stream(Vec::new(), status).await;
        let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("grpc://d:30000").build());
        let mut stream = stream.tracked_decode_side(worker);
        assert_eq!(stream.first_response().await, first, "{status}");

        stream.report_through(end);
        let read = stream.next().await;
        assert_eq!(read.is_some(), status == "13", "{status}");
        drop((stream, guard));
        assert_eq!(
            *heard.lock().expect("the stage heard"),
            ["issued", "dispatched Single", end_heard],
            "{status}"
        );
    }
}

/// The stage puts the layer before a reader watches the stream: a debug
/// build stops at a stream watched already.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "the layer is put before the stream is watched")]
fn the_layer_is_not_put_over_a_watch() {
    use super::FanoutStream;
    use crate::routers::common::placement::RequestEnd;

    let (client, _body) = sse_channel();
    let mut stream = ProtoStream::Fanout(FanoutStream::new(Vec::new()))
        .until_client_gone(&client, "req-watched");
    stream.put_report_layer(RequestEnd::default(), None);
}

/// A client that stays reads the stream as it is: every response, a gap
/// inside it waited out, and the end where the engine ends it.
#[tokio::test]
async fn a_client_that_stays_reads_the_stream_as_it_is() {
    let (mut stream, gate, _rx) = watched(
        vec![chunk(0, "a")],
        vec![chunk(0, "b"), complete(0, "stop")],
    )
    .await;
    assert!(matches!(stream.next().await, Some(Ok(_))));
    let (second, ()) = tokio::join!(stream.next(), async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let _ = gate.send(());
    });
    assert!(
        matches!(second, Some(Ok(_))),
        "the gap inside the stream is waited out"
    );
    assert!(matches!(stream.next().await, Some(Ok(_))));
    assert!(stream.next().await.is_none(), "the end is the end");
}
