//! A stream watched for the client's leaving: what it forwards before and after the client is gone,
//! and when it is aborted.

use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use smg_grpc_client::vllm_proto as vllm;
use tokio::sync::oneshot;

use super::super::{PrefillCachedTokens, ProtoGenerateComplete, ProtoStream};
use crate::{
    routers::{
        common::sse::{sse_channel, SseReceiver},
        grpc::regular::streaming::tests::eof::{chunk, complete, gated_stream, staged_stream},
    },
    worker::{BasicWorkerBuilder, Worker},
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
    assert_eq!(worker.tracked_load(), 1);
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
    assert_eq!(worker.tracked_load(), 0, "its load released with it");
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

/// The wait for the client's leaving is one for the stream's whole life,
/// polled only while the reader is parked on the engine: a response that
/// is ready costs no poll of it, before a park as after one, and a client
/// that leaves during a later park still ends the wait at once.
#[tokio::test]
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
    let mut stream = stream.watched(&client, "req-watched", client_gone);
    for _ in 0..3 {
        assert!(matches!(stream.next().await, Some(Ok(_))));
    }
    assert_eq!(
        polls.load(Ordering::SeqCst),
        0,
        "a ready response costs no poll of the wait"
    );
    let (item, ()) = tokio::join!(stream.next(), async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let _ = first_gate.send(());
    });
    assert!(matches!(item, Some(Ok(_))), "the gap is waited out");
    let parked = polls.load(Ordering::SeqCst);
    assert!(parked >= 1, "the parked reader polled the wait");
    assert!(
        matches!(stream.next().await, Some(Ok(_))),
        "the response behind the gate is ready"
    );
    assert_eq!(
        polls.load(Ordering::SeqCst),
        parked,
        "a ready response costs no poll of the wait after a park either"
    );
    let (item, ()) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::join!(stream.next(), async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(rx);
        })
    })
    .await
    .expect("the second gate never opens; the client's leaving ends the wait");
    assert!(item.is_none(), "the client's leaving is the end");
    assert!(stream.client_left(), "the end was the client's leaving");
    assert!(
        stream.next().await.is_none(),
        "nothing after the client left"
    );
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
