use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use bytes::Bytes;
use tokio::sync::mpsc::error::TryRecvError;

use super::*;
use crate::{
    crdt_kv::{Operation, OperationLog, ReplicaId},
    kv::{StreamConfig, StreamRouting},
    service::gossip::StreamAck,
    transport::sync_stream::build_heartbeat,
    MergeStrategy,
};

fn round(
    drain: &[(&str, &[u8])],
    targeted: &[(&str, &str, &[u8])],
    ops: Vec<Operation>,
) -> Arc<RoundBatch> {
    let mut log = OperationLog::new();
    for op in ops {
        log.append(op);
    }
    Arc::new(RoundBatch {
        drain_entries: drain
            .iter()
            .map(|(k, v)| ((*k).to_owned(), Bytes::copy_from_slice(v)))
            .collect(),
        targeted_entries: targeted
            .iter()
            .map(|(t, k, v)| ((*t).to_owned(), (*k).to_owned(), Bytes::copy_from_slice(v)))
            .collect(),
        crdt_ops: Arc::new(log),
    })
}

fn op(key: &str, timestamp: u64) -> Operation {
    Operation::insert(key.to_owned(), b"v".to_vec(), timestamp, ReplicaId::new())
}

fn kinds(rx: &mut mpsc::Receiver<StreamMessage>) -> Vec<StreamMessageType> {
    let mut out = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(msg) => out.push(msg.message_type()),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => return out,
        }
    }
}

fn counter() -> impl FnMut() -> u64 {
    let mut next = 0;
    move || {
        next += 1;
        next
    }
}

#[test]
fn a_round_sends_a_fresh_batch_once_and_the_unacked_ops_every_tick() {
    let (tx, mut rx) = mpsc::channel::<StreamMessage>(16);
    let acked = RwLock::new(CrdtWatermark::new());
    let mut state = SenderState::default();
    let mut seq = counter();
    let first = op("worker:a", 5);
    let batch = round(
        &[("td:x", b"d")],
        &[("peer", "tree:req:1", b"r")],
        vec![first.clone()],
    );

    let outcome = send_round(
        &mut state,
        &batch,
        Some("peer"),
        &acked,
        "me",
        &mut seq,
        &tx,
    );
    assert_eq!(outcome, RoundOutcome::Sent);
    assert_eq!(
        kinds(&mut rx),
        [StreamMessageType::StreamBatch, StreamMessageType::CrdtBatch]
    );

    // The same Arc again: the stream batch is consumed, the op is still unacked.
    send_round(
        &mut state,
        &batch,
        Some("peer"),
        &acked,
        "me",
        &mut seq,
        &tx,
    );
    assert_eq!(kinds(&mut rx), [StreamMessageType::CrdtBatch]);

    // Acked: nothing left to send for this Arc.
    acked.write().merge_max(&CrdtWatermark::from_ops(&[first]));
    send_round(
        &mut state,
        &batch,
        Some("peer"),
        &acked,
        "me",
        &mut seq,
        &tx,
    );
    assert!(kinds(&mut rx).is_empty());

    // A new round batch (fresh Arc) is sent again; frames are numbered in order.
    let next = round(&[("td:x", b"d")], &[], vec![]);
    send_round(&mut state, &next, Some("peer"), &acked, "me", &mut seq, &tx);
    let msg = rx.try_recv().expect("the fresh batch is sent");
    assert_eq!(msg.message_type(), StreamMessageType::StreamBatch);
    assert_eq!(msg.sequence, 4, "the fourth frame of this end");
    assert_eq!(msg.peer_id, "me");
}

#[test]
fn a_round_before_the_peer_is_known_sends_the_ops_but_keeps_the_batch() {
    let (tx, mut rx) = mpsc::channel::<StreamMessage>(16);
    let acked = RwLock::new(CrdtWatermark::new());
    let mut state = SenderState::default();
    let mut seq = counter();
    let batch = round(&[("td:x", b"d")], &[], vec![op("worker:a", 1)]);

    send_round(&mut state, &batch, None, &acked, "me", &mut seq, &tx);
    assert_eq!(kinds(&mut rx), [StreamMessageType::CrdtBatch]);

    // Once the peer is learned the same Arc still counts as fresh.
    send_round(
        &mut state,
        &batch,
        Some("peer"),
        &acked,
        "me",
        &mut seq,
        &tx,
    );
    assert_eq!(
        kinds(&mut rx),
        [StreamMessageType::StreamBatch, StreamMessageType::CrdtBatch]
    );
}

#[test]
fn a_closed_sink_ends_the_round() {
    let (tx, rx) = mpsc::channel::<StreamMessage>(16);
    drop(rx);
    let acked = RwLock::new(CrdtWatermark::new());
    let mut state = SenderState::default();
    let batch = round(&[("td:x", b"d")], &[], vec![]);
    let outcome = send_round(
        &mut state,
        &batch,
        Some("peer"),
        &acked,
        "me",
        &mut counter(),
        &tx,
    );
    assert_eq!(outcome, RoundOutcome::Closed);
}

/// A sink with room for everything but stream batches, which it refuses as
/// full after the first: the shape of a channel that fills mid-round.
struct FullForStreamBatches {
    inner: mpsc::Sender<StreamMessage>,
    stream_batches_taken: AtomicUsize,
}

impl StreamSink for FullForStreamBatches {
    fn try_send(&self, msg: StreamMessage) -> Result<(), SinkError> {
        if msg.message_type() == StreamMessageType::StreamBatch
            && self.stream_batches_taken.fetch_add(1, Ordering::Relaxed) >= 1
        {
            return Err(SinkError::Full);
        }
        StreamSink::try_send(&self.inner, msg)
    }
}

#[test]
fn a_full_sink_drops_the_rest_of_the_stream_batches_and_the_ops_still_go_out() {
    let (tx, mut rx) = mpsc::channel::<StreamMessage>(16);
    let sink = FullForStreamBatches {
        inner: tx,
        stream_batches_taken: AtomicUsize::new(0),
    };
    let acked = RwLock::new(CrdtWatermark::new());
    let mut state = SenderState::default();
    // Six drain entries make two stream batches (five chunks per batch).
    let keys: Vec<String> = (0..6).map(|i| format!("td:{i}")).collect();
    let drain: Vec<(&str, &[u8])> = keys.iter().map(|k| (k.as_str(), b"d".as_slice())).collect();
    let batch = round(&drain, &[], vec![op("worker:a", 1)]);

    let outcome = send_round(
        &mut state,
        &batch,
        Some("peer"),
        &acked,
        "me",
        &mut counter(),
        &sink,
    );
    assert_eq!(
        outcome,
        RoundOutcome::Sent,
        "a full channel is not a closed one"
    );
    assert_eq!(
        kinds(&mut rx),
        [StreamMessageType::StreamBatch, StreamMessageType::CrdtBatch],
        "the first stream batch and, in the same round, the ops"
    );

    // The round counted as consumed (at-most-once): the same Arc is not re-sent.
    send_round(
        &mut state,
        &batch,
        Some("peer"),
        &acked,
        "me",
        &mut counter(),
        &sink,
    );
    assert_eq!(kinds(&mut rx), [StreamMessageType::CrdtBatch]);
}

fn frame(kind: StreamMessageType, payload: Option<StreamPayload>, sequence: u64) -> StreamMessage {
    StreamMessage {
        message_type: kind as i32,
        payload,
        sequence,
        peer_id: "peer".to_owned(),
    }
}

#[tokio::test]
async fn a_heartbeat_is_handed_to_the_end_unanswered() {
    let (tx, mut rx) = mpsc::channel::<StreamMessage>(4);
    let acked = RwLock::new(CrdtWatermark::new());
    let outcome = handle_frame(
        frame(StreamMessageType::Heartbeat, None, 9),
        "peer",
        "me",
        None,
        &acked,
        &mut || 42,
        &tx,
    )
    .await;
    assert_eq!(outcome, Frame::Heartbeat);
    assert!(rx.try_recv().is_err(), "the handler sends nothing back");
}

#[tokio::test]
async fn acks_and_nacks_are_handed_to_the_end() {
    let (tx, _rx) = mpsc::channel::<StreamMessage>(4);
    let acked = RwLock::new(CrdtWatermark::new());
    let ack = frame(
        StreamMessageType::Ack,
        Some(StreamPayload::Ack(StreamAck {
            sequence: 3,
            success: true,
            error_message: String::new(),
        })),
        3,
    );
    assert_eq!(
        handle_frame(ack, "peer", "me", None, &acked, &mut || 1, &tx).await,
        Frame::Ack {
            success: Some(true),
            sequence: 3
        }
    );
    let bare_ack = frame(StreamMessageType::Ack, None, 4);
    assert_eq!(
        handle_frame(bare_ack, "peer", "me", None, &acked, &mut || 1, &tx).await,
        Frame::Ack {
            success: None,
            sequence: 4
        }
    );
    let nack = frame(StreamMessageType::Nack, None, 5);
    assert_eq!(
        handle_frame(nack, "peer", "me", None, &acked, &mut || 1, &tx).await,
        Frame::Nack { sequence: 5 }
    );
}

#[tokio::test]
async fn a_stream_batch_reaches_the_mesh_kv_subscriber() {
    let mesh_kv = MeshKV::new("receiver".to_owned());
    let ns = mesh_kv.configure_stream_prefix(
        "td:",
        StreamConfig {
            max_buffer_bytes: 64 * 1024,
            routing: StreamRouting::Broadcast,
        },
    );
    let mut sub = ns.subscribe("model-1");
    let (tx, _rx) = mpsc::channel::<StreamMessage>(4);
    let acked = RwLock::new(CrdtWatermark::new());

    let batch = round(&[("td:model-1", b"delta")], &[], vec![]);
    let mut batches = build_peer_stream_batches(&batch, "receiver");
    let msg = wrap_stream_batch(batches.remove(0), 1, "peer");
    let outcome = handle_frame(msg, "peer", "me", Some(&mesh_kv), &acked, &mut || 1, &tx).await;
    assert_eq!(outcome, Frame::Continue);

    let (key, payload) = sub.receiver.recv().await.expect("the entry is published");
    assert_eq!(key, "td:model-1");
    let fragments = payload.expect("a value, not a delete");
    let value: Vec<u8> = fragments.iter().flat_map(|b| b.iter().copied()).collect();
    assert_eq!(value, b"delta");
}

#[tokio::test]
async fn a_crdt_batch_is_merged_and_acked_back() {
    let sender = MeshKV::new("sender".to_owned());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());
    let ops = sender.collect_round_batch().crdt_ops;
    let mut batches = build_crdt_batches(ops.operations(), MAX_STREAM_CHUNK_BYTES);
    let msg = wrap_crdt_batch(batches.remove(0), 7, "peer");

    let receiver = MeshKV::new("receiver".to_owned());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    let (tx, mut rx) = mpsc::channel::<StreamMessage>(4);
    let acked = RwLock::new(CrdtWatermark::new());

    let outcome = handle_frame(msg, "peer", "me", Some(&receiver), &acked, &mut || 8, &tx).await;
    assert_eq!(outcome, Frame::Continue);
    assert_eq!(
        r_ns.get("worker:a"),
        Some(b"v1".to_vec()),
        "the op is merged"
    );

    let ack = rx.try_recv().expect("the merged versions are acked back");
    assert_eq!(ack.message_type(), StreamMessageType::CrdtAck);
    assert_eq!(ack.sequence, 8);
    let Some(StreamPayload::CrdtAck(payload)) = ack.payload else {
        panic!("a CrdtAck payload");
    };
    let watermark = crdt_ack_to_watermark(payload);
    assert!(
        ops.operations().iter().all(|op| !watermark.allows(op)),
        "the ack covers every op of the batch"
    );
}

#[tokio::test]
async fn a_crdt_batch_on_a_full_sink_is_merged_and_its_ack_dropped_without_blocking() {
    let sender = MeshKV::new("sender".to_owned());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());
    let ops = sender.collect_round_batch().crdt_ops;
    let mut batches = build_crdt_batches(ops.operations(), MAX_STREAM_CHUNK_BYTES);
    let msg = wrap_crdt_batch(batches.remove(0), 7, "peer");

    let receiver = MeshKV::new("receiver".to_owned());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    let (tx, mut rx) = mpsc::channel::<StreamMessage>(1);
    tx.try_send(build_heartbeat(1, "me"))
        .expect("the one slot is taken");
    let acked = RwLock::new(CrdtWatermark::new());

    let outcome = tokio::time::timeout(
        Duration::from_millis(500),
        handle_frame(msg, "peer", "me", Some(&receiver), &acked, &mut || 8, &tx),
    )
    .await;
    assert_eq!(
        outcome,
        Ok(Frame::Continue),
        "a full sink drops the ack, it does not block the inbound loop"
    );
    assert_eq!(
        r_ns.get("worker:a"),
        Some(b"v1".to_vec()),
        "and the merge still happened"
    );
    let only = rx
        .try_recv()
        .expect("the slot still holds the frame that filled it");
    assert_eq!(only.message_type(), StreamMessageType::Heartbeat);
    assert!(
        matches!(rx.try_recv(), Err(TryRecvError::Empty)),
        "no ack was queued"
    );
}

#[tokio::test]
async fn a_crdt_ack_advances_the_send_watermark() {
    let (tx, _rx) = mpsc::channel::<StreamMessage>(4);
    let acked = RwLock::new(CrdtWatermark::new());
    let first = op("worker:a", 5);
    assert!(acked.read().allows(&first));

    let msg = wrap_crdt_ack(
        &CrdtWatermark::from_ops(std::slice::from_ref(&first)),
        2,
        "peer",
    );
    let outcome = handle_frame(msg, "peer", "me", None, &acked, &mut || 1, &tx).await;
    assert_eq!(outcome, Frame::Continue);
    assert!(
        !acked.read().allows(&first),
        "the acked version is not resent"
    );
    assert!(
        acked.read().allows(&op("worker:a", 6)),
        "a newer write still is"
    );
}

#[tokio::test]
async fn the_retired_v1_types_are_ignored() {
    let (tx, mut rx) = mpsc::channel::<StreamMessage>(4);
    let acked = RwLock::new(CrdtWatermark::new());
    for kind in [
        StreamMessageType::IncrementalUpdate,
        StreamMessageType::SnapshotRequest,
        StreamMessageType::SnapshotChunk,
        StreamMessageType::SnapshotComplete,
    ] {
        let outcome = handle_frame(
            frame(kind, None, 1),
            "peer",
            "me",
            None,
            &acked,
            &mut || 1,
            &tx,
        )
        .await;
        assert_eq!(outcome, Frame::Continue);
    }
    assert!(
        matches!(rx.try_recv(), Err(TryRecvError::Empty)),
        "nothing goes back"
    );
}
