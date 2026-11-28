//! CRDT batch wire-message helpers for the `Gossip::sync_stream` RPC.
//!
//! Mirrors [`sync_stream`](crate::transport::sync_stream) for the CRDT data
//! path:
//! - Outbound: [`build_crdt_batches`] converts an [`Operation`] slice (the
//!   local op-log snapshot carried in [`RoundBatch`](crate::kv::RoundBatch))
//!   into one or more wire [`CrdtBatch`]es, each bounded below the gRPC
//!   message cap; [`wrap_crdt_batch`] builds the `StreamMessage` envelope.
//! - Inbound: [`dispatch_crdt_batch`] decodes a received `CrdtBatch` back into
//!   `Operation`s and merges them into the local CRDT store via `MeshKV`.
//! - Ack: [`dispatch_crdt_batch`] returns a [`CrdtWatermark`] of the per-key
//!   versions it merged; [`wrap_crdt_ack`] / [`watermark_to_crdt_ack`] /
//!   [`crdt_ack_to_watermark`] carry it back over the wire so a sender can skip
//!   keys the peer already holds.
//!
//! Merge is idempotent by op-id, so re-sending already-seen ops is a no-op.

use crate::{
    crdt_kv::{CrdtWatermark, Operation, ReplicaId},
    kv::MeshKV,
    service::gossip::{
        stream_message::Payload as StreamPayload, CrdtAck, CrdtBatch, CrdtKeyVersion, CrdtOp,
        StreamMessage, StreamMessageType,
    },
};

/// Convert an in-crate [`Operation`] into its wire form. A `Remove` carries an
/// empty `value` and `tombstone = true`; an `Insert` carries the value bytes
/// and `tombstone = false`. The `replica_id` UUID is rendered to text.
fn op_to_proto(op: &Operation) -> CrdtOp {
    match op {
        Operation::Insert {
            key,
            value,
            timestamp,
            replica_id,
        } => CrdtOp {
            key: key.clone(),
            value: value.clone(),
            tombstone: false,
            timestamp: *timestamp,
            replica_id: replica_id.to_string(),
        },
        Operation::Remove {
            key,
            timestamp,
            replica_id,
        } => CrdtOp {
            key: key.clone(),
            value: Vec::new(),
            tombstone: true,
            timestamp: *timestamp,
            replica_id: replica_id.to_string(),
        },
    }
}

/// Convert a wire [`CrdtOp`] back into an [`Operation`]. Returns `None` if the
/// `replica_id` field is not a valid UUID — a malformed/hostile peer's op is
/// dropped rather than poisoning the merge.
fn proto_to_op(op: CrdtOp) -> Option<Operation> {
    let replica_id = ReplicaId::from_string(&op.replica_id).ok()?;
    Some(if op.tombstone {
        Operation::remove(op.key, op.timestamp, replica_id)
    } else {
        Operation::insert(op.key, op.value, op.timestamp, replica_id)
    })
}

/// Conservative estimate of a `CrdtOp`'s encoded size: the variable-length
/// fields plus a constant covering proto field tags, length varints, the
/// timestamp/tombstone fields, and the repeated-field wrapper in `CrdtBatch`.
fn estimated_op_size(op: &CrdtOp) -> usize {
    op.key.len() + op.value.len() + op.replica_id.len() + 24
}

/// Split an op-log snapshot into one or more [`CrdtBatch`]es, each estimated to
/// stay under `max_bytes` so the wrapped `StreamMessage` does not exceed the
/// gRPC message cap. Returns an empty `Vec` for an empty snapshot. Without this
/// bound, a large op-log (broadcast in full each round until per-peer watermark
/// filtering lands) could produce a single frame above `MAX_MESSAGE_SIZE`,
/// which tonic rejects on encode/decode and which would tear down the
/// sync_stream. A single op larger than `max_bytes` is emitted alone (best
/// effort); `worker:`/`rl:`/`config:` values are far below the cap, so in
/// practice this only bounds the op count per frame.
pub fn build_crdt_batches(ops: &[Operation], max_bytes: usize) -> Vec<CrdtBatch> {
    let mut out = Vec::new();
    let mut current: Vec<CrdtOp> = Vec::new();
    let mut current_bytes = 0usize;
    for op in ops {
        let proto = op_to_proto(op);
        let size = estimated_op_size(&proto);
        if !current.is_empty() && current_bytes + size > max_bytes {
            out.push(CrdtBatch {
                ops: std::mem::take(&mut current),
            });
            current_bytes = 0;
        }
        current_bytes += size;
        current.push(proto);
    }
    if !current.is_empty() {
        out.push(CrdtBatch { ops: current });
    }
    out
}

/// Wrap a [`CrdtBatch`] in a `StreamMessage` envelope.
pub fn wrap_crdt_batch(batch: CrdtBatch, sequence: u64, self_name: &str) -> StreamMessage {
    StreamMessage {
        message_type: StreamMessageType::CrdtBatch as i32,
        payload: Some(StreamPayload::CrdtBatch(batch)),
        sequence,
        peer_id: self_name.to_owned(),
    }
}

/// Encode a [`CrdtWatermark`] as a wire [`CrdtAck`] — one entry per known key,
/// carrying the op-id `(version, replica_id)` so the peer can apply the same
/// `(timestamp, replica_id)` ordering the merge uses.
pub fn watermark_to_crdt_ack(watermark: &CrdtWatermark) -> CrdtAck {
    CrdtAck {
        entries: watermark
            .iter()
            .map(|(key, (version, replica))| CrdtKeyVersion {
                key: key.to_owned(),
                version,
                replica_id: replica.to_string(),
            })
            .collect(),
    }
}

/// Decode a received [`CrdtAck`] into a [`CrdtWatermark`] for merging into the
/// sender's per-peer send watermark. Entries with an unparsable `replica_id`
/// are dropped (mirrors `proto_to_op`).
///
/// A legacy peer acks without `replica_id` (proto3 decodes the missing field as
/// `""`); that entry maps to `(version, ReplicaId::MAX)`, which suppresses
/// every op at or below the acked timestamp — the timestamp-only semantics the
/// old peer speaks. Dropping it instead would leave the watermark stuck and
/// re-send the full set every round for the whole rolling-upgrade window. (A
/// nil sentinel would not help: the acked winner `(v, R)` compares above
/// `(v, nil)` and would still re-send every round.)
pub fn crdt_ack_to_watermark(ack: CrdtAck) -> CrdtWatermark {
    ack.entries
        .into_iter()
        .filter_map(|entry| {
            let replica = if entry.replica_id.is_empty() {
                ReplicaId::MAX
            } else {
                ReplicaId::from_string(&entry.replica_id).ok()?
            };
            Some((entry.key, (entry.version, replica)))
        })
        .collect()
}

/// Wrap a [`CrdtWatermark`] in a `CrdtAck` `StreamMessage` envelope.
pub fn wrap_crdt_ack(watermark: &CrdtWatermark, sequence: u64, self_name: &str) -> StreamMessage {
    StreamMessage {
        message_type: StreamMessageType::CrdtAck as i32,
        payload: Some(StreamPayload::CrdtAck(watermark_to_crdt_ack(watermark))),
        sequence,
        peer_id: self_name.to_owned(),
    }
}

/// Receiver-side dispatch for a `CrdtBatch`: decode each op and merge the batch
/// into the local CRDT store, firing subscribers for keys whose value changed
/// (via `MeshKV::merge_crdt_ops`). Ops with an unparsable `replica_id` are
/// skipped. Merge is idempotent by op-id, so a batch the node has already
/// absorbed is a no-op and fires no subscriber event.
///
/// Returns the per-key versions of the ops just received as a [`CrdtWatermark`],
/// which the caller sends back as a [`CrdtAck`] so the peer can advance its
/// send watermark for those keys. An empty/all-invalid batch returns an empty
/// watermark (nothing to ack).
pub fn dispatch_crdt_batch(mesh_kv: &MeshKV, batch: CrdtBatch) -> CrdtWatermark {
    let ops: Vec<Operation> = batch.ops.into_iter().filter_map(proto_to_op).collect();
    if ops.is_empty() {
        return CrdtWatermark::new();
    }
    let ack = CrdtWatermark::from_ops(&ops);
    mesh_kv.merge_crdt_ops(ops);
    ack
}

#[cfg(test)]
#[path = "../../tests/unit/transport/crdt_batch/tests.rs"]
mod tests;
