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
//! - [`handle_frame`]: one inbound frame — a heartbeat is handed back for the
//!   end to answer (the acceptor) or not (the dialer), a stream batch and a
//!   CRDT batch are applied to the mesh KV (the CRDT one acked back), a CRDT
//!   ack advances the send watermark, the retired v1 message types are
//!   ignored; an ack or a nack is handed back as a [`Frame`] for the end to
//!   log or record.
//! - [`StreamSink`]: the end's outbound channel; the dialer sends plain
//!   messages, the acceptor sends `Result`s for tonic.
//!
//! What stays with each end: its loop (the interval, the acceptor's exit when
//! its channel closes, the dialer's round metrics), its sequence source, how
//! it knows the peer's name (given up front, or learned from the first frame
//! and held to it) and its connection accounting.

use std::sync::Arc;

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
        sync_stream::{build_peer_stream_batches, dispatch_stream_batch, wrap_stream_batch},
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
/// only needs to queue a frame.
pub(crate) trait StreamSink: Send + Sync + 'static {
    /// Queue `msg` without waiting.
    fn try_send(&self, msg: StreamMessage) -> Result<(), SinkError>;
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
}

impl StreamSink for mpsc::Sender<Result<StreamMessage, Status>> {
    fn try_send(&self, msg: StreamMessage) -> Result<(), SinkError> {
        mpsc::Sender::try_send(self, Ok(msg)).map_err(sink_error)
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
    /// The peer sent a heartbeat. The accepting end answers it; the dialing
    /// end does not, or the two would answer each other's answers for as
    /// long as the stream lives.
    Heartbeat,
    /// The peer acked (`success` from its payload when it carried one); the
    /// end logs or records it.
    Ack {
        success: Option<bool>,
        sequence: u64,
    },
    /// The peer nacked; the end logs or records it.
    Nack { sequence: u64 },
}

/// One inbound frame from `peer`, the same on both ends: a heartbeat is
/// returned for the end to answer or not, a stream batch is applied
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
            Frame::Heartbeat
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
#[path = "../../tests/unit/transport/peer_stream/sender_tick_tests.rs"]
mod sender_tick_tests;

#[cfg(test)]
#[path = "../../tests/unit/transport/peer_stream/tests.rs"]
mod tests;
