//! SyncStream wire-message helpers — both directions of the
//! bidirectional `Gossip::sync_stream` RPC live here.
//!
//! Outbound (sender side):
//! - [`build_peer_stream_batches`] composes [`chunk_value`] +
//!   [`build_stream_batches`] for a single peer.
//! - [`wrap_stream_batch`] / [`build_heartbeat`] construct
//!   `StreamMessage` envelopes.
//!
//! Inbound (receiver side):
//! - [`dispatch_stream_batch`] routes the entries of a received
//!   `StreamBatch` to local subscribers — single-chunk entries fire
//!   directly; multi-chunk entries go through the
//!   [`ChunkAssembler`](crate::transport::chunk_assembler::ChunkAssembler)
//!   and fire only on full reassembly.
//!
//! The pure chunk/batch arithmetic lives in
//! [`crate::transport::chunking`]; this module composes it into
//! the message layer.

use bytes::Bytes;

use crate::{
    kv::{MeshKV, RoundBatch},
    service::gossip::{
        stream_message::Payload as StreamPayload, StreamBatch, StreamEntry, StreamMessage,
        StreamMessageType,
    },
    transport::{
        chunking::{build_stream_batches, chunk_value, next_generation},
        limits::{DEFAULT_MAX_CHUNKS_PER_BATCH, MAX_STREAM_CHUNK_BYTES},
    },
};

/// Build the `StreamBatch`es that should be sent to `peer_id` for
/// the current round. Returns an empty `Vec` when neither the
/// broadcast drain nor any targeted entry is addressed to this peer.
///
/// `drain_entries` are broadcast: every peer's emitter includes
/// them. `targeted_entries` are only included when their target
/// matches `peer_id` AND `peer_id` is non-empty — pass `""` to
/// disable targeted entries entirely (e.g. when the inbound peer
/// identity is not yet learned). The explicit empty-check guards
/// against the degenerate `target == "" == peer_id` match.
/// Oversized values are split via [`chunk_value`]; the returned
/// batches respect the `DEFAULT_MAX_CHUNKS_PER_BATCH` /
/// `MAX_STREAM_CHUNK_BYTES` caps.
pub fn build_peer_stream_batches(round_batch: &RoundBatch, peer_id: &str) -> Vec<StreamBatch> {
    let mut entries =
        Vec::with_capacity(round_batch.drain_entries.len() + round_batch.targeted_entries.len());
    for (key, value) in &round_batch.drain_entries {
        entries.extend(chunk_value(
            key.clone(),
            next_generation(),
            value.clone(),
            MAX_STREAM_CHUNK_BYTES,
        ));
    }
    if !peer_id.is_empty() {
        for (target, key, value) in &round_batch.targeted_entries {
            if target == peer_id {
                entries.extend(chunk_value(
                    key.clone(),
                    next_generation(),
                    value.clone(),
                    MAX_STREAM_CHUNK_BYTES,
                ));
            }
        }
    }
    if entries.is_empty() {
        return Vec::new();
    }
    build_stream_batches(
        entries,
        DEFAULT_MAX_CHUNKS_PER_BATCH,
        MAX_STREAM_CHUNK_BYTES,
    )
}

/// Wrap a `StreamBatch` in a `StreamMessage` envelope.
pub fn wrap_stream_batch(batch: StreamBatch, sequence: u64, self_name: &str) -> StreamMessage {
    StreamMessage {
        message_type: StreamMessageType::StreamBatch as i32,
        payload: Some(StreamPayload::StreamBatch(batch)),
        sequence,
        peer_id: self_name.to_owned(),
    }
}

/// Build a heartbeat `StreamMessage` (no payload, message_type = Heartbeat).
pub fn build_heartbeat(sequence: u64, self_name: &str) -> StreamMessage {
    StreamMessage {
        message_type: StreamMessageType::Heartbeat as i32,
        payload: None,
        sequence,
        peer_id: self_name.to_owned(),
    }
}

/// Receiver-side dispatch for `StreamBatch` entries. Single-chunk
/// entries (`total_chunks == 1`) fire subscribers directly — no state
/// in the chunk assembler. Multi-chunk entries route through the
/// assembler and fire subscribers only on full reassembly.
///
/// Each chunk payload is detached from the decoded `StreamBatch`
/// buffer via `Bytes::copy_from_slice` before it's stored or
/// forwarded. Without this, the prost-generated `StreamEntry.data`
/// is a `Bytes` view into the message frame, so a single pinned
/// chunk (in the assembler or in a subscriber queue) would retain
/// the entire batch allocation. That would let a peer packing many
/// tiny entries into near-`MAX_MESSAGE_SIZE` batches defeat both
/// `DEFAULT_MAX_ASSEMBLER_BYTES` and subscriber backpressure.
///
/// The chunk assembler scopes in-flight state by `peer_id` so
/// concurrent chunked values from different senders under the same
/// key don't collide.
pub fn dispatch_stream_batch(
    mesh_kv: &MeshKV,
    peer_id: &str,
    entries: impl IntoIterator<Item = StreamEntry>,
) {
    for entry in entries {
        let data = Bytes::copy_from_slice(&entry.data);
        if entry.total_chunks == 1 {
            // A fresh single-chunk value supersedes any in-flight
            // multi-chunk assembly for the same (peer, key); drop the
            // stale fragments so they don't wait for GC.
            mesh_kv.chunk_assembler().drop_pending(peer_id, &entry.key);
            mesh_kv.notify_subscribers(&entry.key, Some(vec![data]));
        } else {
            let key = entry.key.clone();
            if let Some(fragments) = mesh_kv.chunk_assembler().receive_chunk(
                peer_id,
                &key,
                entry.generation,
                entry.chunk_index,
                entry.total_chunks,
                data,
            ) {
                mesh_kv.notify_subscribers(&key, Some(fragments));
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/transport/sync_stream/tests.rs"]
mod tests;
