//! The shared life of one end of a `sync_stream`: what the dialer
//! ([`gossip_controller`](crate::gossip_controller)) and the acceptor
//! ([`gossip_service`](crate::gossip_service)) both do with a peer stream once
//! it is open.
//!
//! - [`send_round`]: one tick of the periodic sender — the fresh
//!   [`RoundBatch`] for the peer (once per batch, by `Arc` identity, decided
//!   by [`plan_sender_tick`]) and then the CRDT ops the peer has not acked.
//! - [`next_frame`]: the next inbound frame under [`STREAM_IDLE_TIMEOUT`];
//!   `None` ends the stream (closed, failed or idle).
//! - [`handle_frame`]: one inbound frame — a heartbeat is answered with one,
//!   a stream batch and a CRDT batch are applied to the mesh KV (the CRDT one
//!   acked back), a CRDT ack advances the send watermark, the retired v1
//!   message types are ignored; an ack or a nack is handed back as a
//!   [`Frame`] for the end to log or record.
//! - [`StreamSink`]: the end's outbound channel; the dialer sends plain
//!   messages, the acceptor sends `Result`s for tonic.
//!
//! What stays with each end: its loop (the interval, the acceptor's exit when
//! its channel closes, the dialer's round metrics), its sequence source, how
//! it knows the peer's name (given up front, or learned from the first frame
//! and held to it) and its connection accounting.

use std::{future::Future, sync::Arc};

use parking_lot::RwLock;
use tokio::sync::mpsc::{self, error::TrySendError};
use tokio_stream::StreamExt;
use tonic::Status;
use tracing as log;

use crate::{
    crdt_kv::CrdtWatermark,
    kv::{MeshKV, RoundBatch},
    service::gossip::{
        stream_message::Payload as StreamPayload, StreamBatch, StreamMessage, StreamMessageType,
    },
    transport::{
        crdt_batch::{
            build_crdt_batches, crdt_ack_to_watermark, dispatch_crdt_batch, wrap_crdt_ack,
            wrap_crdt_batch,
        },
        limits::{MAX_STREAM_CHUNK_BYTES, STREAM_IDLE_TIMEOUT},
        sync_stream::{
            build_heartbeat, build_peer_stream_batches, dispatch_stream_batch, wrap_stream_batch,
        },
    },
};

/// Why a frame could not be queued on a [`StreamSink`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SinkError {
    /// The channel is full: the frame is dropped (at-most-once; the next
    /// round regenerates what matters).
    Full,
    /// The other side of the channel is gone: the stream is over.
    Closed,
}

/// One end's outbound channel. The dialer's carries `StreamMessage`s, the
/// acceptor's carries `Result`s for tonic's response stream; the shared code
/// only needs to queue a frame or wait for room.
pub(crate) trait StreamSink: Send + Sync + 'static {
    /// Queue `msg` without waiting.
    fn try_send(&self, msg: StreamMessage) -> Result<(), SinkError>;
    /// Send `msg`, waiting for room; `false` once the channel is closed.
    fn send(&self, msg: StreamMessage) -> impl Future<Output = bool> + Send;
}

fn sink_error<T>(err: TrySendError<T>) -> SinkError {
    match err {
        TrySendError::Full(_) => SinkError::Full,
        TrySendError::Closed(_) => SinkError::Closed,
    }
}

impl StreamSink for mpsc::Sender<StreamMessage> {
    fn try_send(&self, msg: StreamMessage) -> Result<(), SinkError> {
        mpsc::Sender::try_send(self, msg).map_err(sink_error)
    }

    async fn send(&self, msg: StreamMessage) -> bool {
        mpsc::Sender::send(self, msg).await.is_ok()
    }
}

impl StreamSink for mpsc::Sender<Result<StreamMessage, Status>> {
    fn try_send(&self, msg: StreamMessage) -> Result<(), SinkError> {
        mpsc::Sender::try_send(self, Ok(msg)).map_err(sink_error)
    }

    async fn send(&self, msg: StreamMessage) -> bool {
        mpsc::Sender::send(self, Ok(msg)).await.is_ok()
    }
}

/// The sender state that outlives a tick: the round batch last emitted, so
/// an unchanged one (the same `Arc`) is not sent again.
#[derive(Debug, Default)]
pub(crate) struct SenderState {
    last_stream_batch: Option<Arc<RoundBatch>>,
}

/// How a sender tick ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoundOutcome {
    /// The round went out (frames a full channel could not take were
    /// dropped, as at-most-once delivery allows).
    Sent,
    /// The outbound channel is closed: the sender has nobody to send to.
    Closed,
}

/// One tick of the periodic sender for `peer`.
///
/// The stream batches first: emitted when the round batch is fresh (not the
/// `Arc` emitted last) and the peer is known, as [`plan_sender_tick`] decides;
/// a skipped tick leaves the watermark untouched so the same batch is
/// re-evaluated next tick. Then the CRDT op-log, every tick: only the ops
/// `acked` does not cover yet (acks shrink the delta even when the round
/// batch is unchanged; the watermark advances solely on `CrdtAck`, so an
/// unacked key retries next round). `next_sequence` numbers each frame the
/// end's way.
pub(crate) fn send_round<S: StreamSink>(
    state: &mut SenderState,
    stream_batch: &Arc<RoundBatch>,
    peer: Option<&str>,
    acked: &RwLock<CrdtWatermark>,
    self_name: &str,
    next_sequence: &mut impl FnMut() -> u64,
    sink: &S,
) -> RoundOutcome {
    let tick = plan_sender_tick(state.last_stream_batch.as_ref(), stream_batch, peer);
    if let SenderTick::Emit(batches) = tick {
        state.last_stream_batch = Some(Arc::clone(stream_batch));
        for batch in batches {
            match sink.try_send(wrap_stream_batch(batch, next_sequence(), self_name)) {
                Ok(()) => {}
                Err(SinkError::Full) => {
                    log::debug!(
                        peer = peer.unwrap_or_default(),
                        "stream batch dropped on backpressure"
                    );
                    // TODO(metrics): bump stream_dropped_on_backpressure
                    break;
                }
                Err(SinkError::Closed) => return RoundOutcome::Closed,
            }
        }
    }

    let crdt_ops: Vec<_> = {
        let acked = acked.read();
        stream_batch
            .crdt_ops
            .operations()
            .iter()
            .filter(|op| acked.allows(op))
            .cloned()
            .collect()
    };
    for crdt_batch in build_crdt_batches(&crdt_ops, MAX_STREAM_CHUNK_BYTES) {
        match sink.try_send(wrap_crdt_batch(crdt_batch, next_sequence(), self_name)) {
            Ok(()) => {}
            Err(SinkError::Full) => {
                log::debug!(
                    peer = peer.unwrap_or_default(),
                    "crdt batch dropped on backpressure"
                );
                break;
            }
            Err(SinkError::Closed) => return RoundOutcome::Closed,
        }
    }
    RoundOutcome::Sent
}

/// The next inbound frame from `peer`, or `None` when the stream ended,
/// failed, or sat idle for [`STREAM_IDLE_TIMEOUT`] (so an unhealthy peer
/// does not pin the task and its channel indefinitely).
pub(crate) async fn next_frame(
    incoming: &mut tonic::Streaming<StreamMessage>,
    peer: &str,
) -> Option<StreamMessage> {
    match tokio::time::timeout(STREAM_IDLE_TIMEOUT, incoming.next()).await {
        Ok(Some(Ok(msg))) => Some(msg),
        Ok(Some(Err(e))) => {
            log::error!(peer = %peer, "Error receiving from sync_stream: {}", e);
            None
        }
        Ok(None) => None,
        Err(_) => {
            log::warn!(
                peer = %peer,
                "sync_stream idle timeout ({STREAM_IDLE_TIMEOUT:?}) — closing"
            );
            None
        }
    }
}

/// What the end does after [`handle_frame`] saw one inbound frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Frame {
    /// Handled here; read the next one.
    Continue,
    /// The peer acked (`success` from its payload when it carried one); the
    /// end logs or records it.
    Ack {
        success: Option<bool>,
        sequence: u64,
    },
    /// The peer nacked; the end logs or records it.
    Nack { sequence: u64 },
    /// The outbound channel is gone: the stream is over.
    Closed,
}

/// One inbound frame from `peer`, the same on both ends: a heartbeat is
/// answered with one (numbered by `next_sequence`), a stream batch is applied
/// to `mesh_kv`, a CRDT batch is merged and its per-key versions acked back
/// (the ack may be dropped on a full channel: the peer resends unacked keys
/// next round), a CRDT ack advances `acked`, the retired v1 message types are
/// logged and ignored. An ack or a nack is returned for the end to handle.
pub(crate) async fn handle_frame<S: StreamSink>(
    msg: StreamMessage,
    peer: &str,
    self_name: &str,
    mesh_kv: Option<&MeshKV>,
    acked: &RwLock<CrdtWatermark>,
    next_sequence: &mut impl FnMut() -> u64,
    sink: &S,
) -> Frame {
    let kind = msg.message_type();
    let StreamMessage {
        payload,
        sequence,
        peer_id,
        ..
    } = msg;
    match kind {
        StreamMessageType::Heartbeat => {
            log::trace!(peer = %peer, "Received heartbeat");
            if sink.send(build_heartbeat(next_sequence(), self_name)).await {
                Frame::Continue
            } else {
                log::warn!(peer = %peer, "Failed to send heartbeat");
                Frame::Closed
            }
        }
        StreamMessageType::Ack => {
            let success = match payload {
                Some(StreamPayload::Ack(ack)) => Some(ack.success),
                _ => None,
            };
            Frame::Ack { success, sequence }
        }
        StreamMessageType::Nack => Frame::Nack { sequence },
        StreamMessageType::StreamBatch => {
            if let (Some(mesh_kv), Some(StreamPayload::StreamBatch(batch))) = (mesh_kv, payload) {
                dispatch_stream_batch(mesh_kv, &peer_id, batch.entries);
            }
            Frame::Continue
        }
        StreamMessageType::CrdtBatch => {
            if let (Some(mesh_kv), Some(StreamPayload::CrdtBatch(batch))) = (mesh_kv, payload) {
                let ack = dispatch_crdt_batch(mesh_kv, batch);
                if !ack.is_empty() {
                    let _ = sink.try_send(wrap_crdt_ack(&ack, next_sequence(), self_name));
                }
            }
            Frame::Continue
        }
        StreamMessageType::CrdtAck => {
            if let Some(StreamPayload::CrdtAck(ack)) = payload {
                acked.write().merge_max(&crdt_ack_to_watermark(ack));
            }
            Frame::Continue
        }
        StreamMessageType::IncrementalUpdate
        | StreamMessageType::SnapshotRequest
        | StreamMessageType::SnapshotChunk
        | StreamMessageType::SnapshotComplete => {
            log::debug!(
                peer = %peer,
                message_type = ?kind,
                "ignoring v1 wire message (state-sync removed)",
            );
            Frame::Continue
        }
    }
}

/// Outcome of one tick of either end's sender.
///
/// [`send_round`] only marks a `RoundBatch` consumed (advances
/// `last_stream_batch`) when [`SenderTick::Emit`] is returned. The two
/// skip variants preserve `last_stream_batch` so the same `Arc` is
/// retried on the next tick — important because before peer identity
/// is learned, we cannot build the correct per-peer batch, and consuming
/// the batch anyway would silently drop its targeted entries.
#[derive(Debug)]
enum SenderTick {
    /// Peer identity not yet known: on the acceptor, the inbound handler
    /// has not recorded the dialer's `peer_id` from the first frame yet;
    /// on the dialer, only an empty peer name gets here.
    SkipPeerUnknown,
    /// Round batch unchanged since the last successful emit.
    SkipBatchUnchanged,
    /// Build successful — emit these batches and advance the watermark.
    Emit(Vec<StreamBatch>),
}

/// Decide what one tick of either end's sender should do.
///
/// Splitting this from [`send_round`] makes the per-tick decision
/// pinnable in unit tests without standing up an interval timer, a
/// real mpsc, or a tokio runtime. `send_round` advances the freshness
/// watermark and does the channel I/O; this function is responsible only
/// for the "what should we do this tick" decision.
fn plan_sender_tick(
    last_stream_batch: Option<&Arc<RoundBatch>>,
    current_stream_batch: &Arc<RoundBatch>,
    learned_peer: Option<&str>,
) -> SenderTick {
    let peer_id = match learned_peer {
        Some(peer) if !peer.is_empty() => peer,
        _ => return SenderTick::SkipPeerUnknown,
    };
    let fresh = last_stream_batch.is_none_or(|last| !Arc::ptr_eq(last, current_stream_batch));
    if !fresh {
        return SenderTick::SkipBatchUnchanged;
    }
    SenderTick::Emit(build_peer_stream_batches(current_stream_batch, peer_id))
}

#[cfg(test)]
mod sender_tick_tests {
    use bytes::Bytes;

    use super::*;
    use crate::kv::RoundBatch;

    fn round_batch_with(
        drain: Vec<(&str, &[u8])>,
        targeted: Vec<(&str, &str, &[u8])>,
    ) -> Arc<RoundBatch> {
        Arc::new(RoundBatch {
            drain_entries: drain
                .into_iter()
                .map(|(k, v)| (k.to_string(), Bytes::copy_from_slice(v)))
                .collect(),
            targeted_entries: targeted
                .into_iter()
                .map(|(t, k, v)| (t.to_string(), k.to_string(), Bytes::copy_from_slice(v)))
                .collect(),
            crdt_ops: Arc::default(),
        })
    }

    fn batch_keys(batches: &[StreamBatch]) -> Vec<String> {
        batches
            .iter()
            .flat_map(|b| b.entries.iter().map(|e| e.key.clone()))
            .collect()
    }

    #[test]
    fn skips_when_peer_unknown_and_keeps_watermark_unchanged() {
        // Original race: with learned_peer = None the previous code
        // would consume the batch with peer_id = "", drop targeted
        // entries, and advance last_stream_batch — so the next tick
        // (after peer identity arrived) would see an unchanged Arc and
        // never re-send the lost targeted entries.
        let rb = round_batch_with(
            vec![("td:foo", b"d")],
            vec![("peer_X", "tree:req:abc", b"r")],
        );
        let decision = plan_sender_tick(None, &rb, None);
        assert!(matches!(decision, SenderTick::SkipPeerUnknown));
    }

    #[test]
    fn skips_when_peer_is_empty_string() {
        // Defensive: an empty learned_peer string must not be treated
        // as a valid identity, even though the underlying string-equality
        // would otherwise let a targeted entry with target == "" leak
        // through.
        let rb = round_batch_with(vec![], vec![("", "tree:req:weird", b"r")]);
        let decision = plan_sender_tick(None, &rb, Some(""));
        assert!(matches!(decision, SenderTick::SkipPeerUnknown));
    }

    #[test]
    fn emits_drain_and_targeted_once_peer_is_known() {
        // First tick (peer unknown) returns SkipPeerUnknown. After the
        // inbound handler learns the peer, a second tick on the same
        // Arc must emit both drain and the matching targeted entry.
        let rb = round_batch_with(
            vec![("td:foo", b"d")],
            vec![("peer_X", "tree:req:abc", b"r")],
        );
        let pre = plan_sender_tick(None, &rb, None);
        assert!(matches!(pre, SenderTick::SkipPeerUnknown));

        // last_stream_batch was left unchanged by the skip, so we still
        // pass None here — modeling the watermark not having advanced.
        let post = plan_sender_tick(None, &rb, Some("peer_X"));
        let SenderTick::Emit(batches) = post else {
            panic!("expected Emit after peer learned, got {post:?}");
        };
        let keys = batch_keys(&batches);
        assert!(keys.contains(&"td:foo".to_string()), "drain entry emitted");
        assert!(
            keys.contains(&"tree:req:abc".to_string()),
            "targeted entry for peer_X emitted"
        );
    }

    #[test]
    fn does_not_resend_same_arc_after_consume() {
        // After a successful Emit, the loop advances last_stream_batch
        // to the same Arc. The next tick must report SkipBatchUnchanged
        // and emit nothing.
        let rb = round_batch_with(vec![("td:foo", b"d")], vec![]);
        let decision = plan_sender_tick(Some(&rb), &rb, Some("peer_X"));
        assert!(matches!(decision, SenderTick::SkipBatchUnchanged));
    }

    #[test]
    fn emits_only_targeted_entries_for_learned_peer() {
        // Targeted-entry filtering still works correctly: a batch with
        // entries for multiple peers must only emit entries addressed to
        // this stream's learned peer (plus drain entries which broadcast).
        let rb = round_batch_with(
            vec![("td:bcast", b"b")],
            vec![
                ("peer_X", "tree:req:x", b"x"),
                ("peer_Y", "tree:req:y", b"y"),
            ],
        );
        let SenderTick::Emit(batches) = plan_sender_tick(None, &rb, Some("peer_X")) else {
            panic!("expected Emit");
        };
        let keys = batch_keys(&batches);
        assert!(keys.contains(&"td:bcast".to_string()));
        assert!(keys.contains(&"tree:req:x".to_string()));
        assert!(
            !keys.contains(&"tree:req:y".to_string()),
            "must not include entries addressed to other peers"
        );
    }

    #[test]
    fn drain_only_batch_still_emits_when_peer_learned() {
        // Regression guard: a batch with only drain entries and no
        // targeted entries should still emit after peer is learned —
        // we do not want a future "skip when no targeted for this peer"
        // optimization to reintroduce the original race for drain entries.
        let rb = round_batch_with(vec![("td:foo", b"d"), ("td:bar", b"d")], vec![]);
        let SenderTick::Emit(batches) = plan_sender_tick(None, &rb, Some("peer_X")) else {
            panic!("expected Emit for drain-only batch");
        };
        let keys = batch_keys(&batches);
        assert!(keys.contains(&"td:foo".to_string()));
        assert!(keys.contains(&"td:bar".to_string()));
    }
}

#[cfg(test)]
mod tests {
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

        async fn send(&self, msg: StreamMessage) -> bool {
            StreamSink::send(&self.inner, msg).await
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
        let drain: Vec<(&str, &[u8])> =
            keys.iter().map(|k| (k.as_str(), b"d".as_slice())).collect();
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

    fn frame(
        kind: StreamMessageType,
        payload: Option<StreamPayload>,
        sequence: u64,
    ) -> StreamMessage {
        StreamMessage {
            message_type: kind as i32,
            payload,
            sequence,
            peer_id: "peer".to_owned(),
        }
    }

    #[tokio::test]
    async fn a_heartbeat_is_answered_with_one() {
        let (tx, mut rx) = mpsc::channel::<Result<StreamMessage, Status>>(4);
        let acked = RwLock::new(CrdtWatermark::new());
        let mut seq = || 42;
        let outcome = handle_frame(
            frame(StreamMessageType::Heartbeat, None, 9),
            "peer",
            "me",
            None,
            &acked,
            &mut seq,
            &tx,
        )
        .await;
        assert_eq!(outcome, Frame::Continue);
        let reply = rx
            .try_recv()
            .expect("a heartbeat goes back")
            .expect("as a message");
        assert_eq!(reply.message_type(), StreamMessageType::Heartbeat);
        assert_eq!(reply.sequence, 42);
        assert_eq!(reply.peer_id, "me");
        assert!(reply.payload.is_none());
    }

    #[tokio::test]
    async fn a_heartbeat_on_a_closed_sink_ends_the_stream() {
        let (tx, rx) = mpsc::channel::<StreamMessage>(4);
        drop(rx);
        let acked = RwLock::new(CrdtWatermark::new());
        let outcome = handle_frame(
            frame(StreamMessageType::Heartbeat, None, 1),
            "peer",
            "me",
            None,
            &acked,
            &mut || 1,
            &tx,
        )
        .await;
        assert_eq!(outcome, Frame::Closed);
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

        let outcome =
            handle_frame(msg, "peer", "me", Some(&receiver), &acked, &mut || 8, &tx).await;
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
}
