//! End-to-end tests for the CRDT-over-gossip path (d-3a).
//!
//! These exercise the full producer→consumer round trip without gRPC: the
//! sender's op-log snapshot is encoded with
//! [`build_crdt_batch`](crate::transport::crdt_batch::build_crdt_batch) and fed
//! into the receiver's
//! [`dispatch_crdt_batch`](crate::transport::crdt_batch::dispatch_crdt_batch),
//! which decodes, merges into the receiver's store, and fires subscribers. In
//! production the batch serialises through the `Gossip::sync_stream` RPC; here
//! we bypass prost and route the ops directly, matching the chunking
//! integration tests.

use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::mpsc::error::TryRecvError;

use crate::{
    crdt_kv::{
        decode as decode_epoch_count, encode as encode_epoch_count, CrdtWatermark, EpochCount,
        Operation,
    },
    kv::MeshKV,
    transport::{
        crdt_batch::{build_crdt_batches, dispatch_crdt_batch},
        limits::MAX_STREAM_CHUNK_BYTES,
    },
    MergeStrategy,
};

/// Simulate one gossip round of CRDT delivery: snapshot the sender's op-log,
/// encode it into size-bounded batches, and dispatch each into the receiver.
fn deliver_crdt(sender: &MeshKV, receiver: &MeshKV) {
    let ops = sender.collect_round_batch().crdt_ops;
    for batch in build_crdt_batches(ops.operations(), MAX_STREAM_CHUNK_BYTES) {
        dispatch_crdt_batch(receiver, batch);
    }
}

/// Snapshot the ops the sender would send this round: the op-log filtered by
/// the per-key send watermark. Mirrors the live sender's filter step.
fn pending_ops(sender: &MeshKV, acked: &CrdtWatermark) -> Vec<Operation> {
    sender
        .collect_round_batch()
        .crdt_ops
        .operations()
        .iter()
        .filter(|op| acked.allows(op))
        .cloned()
        .collect()
}

/// One watermark-filtered round: send only ops the peer has not acked, then
/// advance `acked` from the per-key versions the receiver acks back. Mirrors
/// the live sender/receiver loop (filter → dispatch → ack → merge_max) without
/// gossip tasks.
fn deliver_crdt_watermarked(sender: &MeshKV, receiver: &MeshKV, acked: &mut CrdtWatermark) {
    let ops = pending_ops(sender, acked);
    for batch in build_crdt_batches(&ops, MAX_STREAM_CHUNK_BYTES) {
        let ack = dispatch_crdt_batch(receiver, batch);
        acked.merge_max(&ack);
    }
}

fn flatten(fragments: &[Bytes]) -> Vec<u8> {
    fragments.iter().flat_map(|b| b.iter().copied()).collect()
}

#[test]
fn remote_crdt_batch_converges_store() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    assert_eq!(r_ns.get("worker:a"), None);

    deliver_crdt(&sender, &receiver);
    assert_eq!(
        r_ns.get("worker:a"),
        Some(b"v1".to_vec()),
        "CRDT op-log delivered over the wire converges the receiver's store"
    );
}

#[tokio::test]
async fn remote_merge_fires_subscriber_with_value() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    let mut sub = r_ns.subscribe("");

    deliver_crdt(&sender, &receiver);

    let (key, payload) = sub
        .receiver
        .recv()
        .await
        .expect("remote merge fires a subscriber event");
    assert_eq!(key, "worker:a");
    assert_eq!(flatten(&payload.expect("insert delivers a value")), b"v1");
}

#[tokio::test]
async fn redelivering_same_batch_fires_no_new_event() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    let mut sub = r_ns.subscribe("");

    deliver_crdt(&sender, &receiver);
    let _ = sub.receiver.recv().await.expect("first delivery fires");

    // Merge is idempotent by op-id: re-delivering the same batch changes no
    // live value, so no subscriber event fires.
    deliver_crdt(&sender, &receiver);
    assert!(
        matches!(sub.receiver.try_recv(), Err(TryRecvError::Empty)),
        "idempotent re-delivery must not fire a subscriber event"
    );
}

#[tokio::test]
async fn rl_remote_merge_subscriber_sees_canonical_shard() {
    // The sender writes the raw 16-byte (epoch, count) payload; the engine
    // normalises it into a shard. The remote-merge subscriber must see the
    // canonical shard shape (matching `get`), not the raw input — migration
    // step 7's value-shape alignment.
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("rl:", MergeStrategy::EpochMaxWins);
    s_ns.put("rl:global:node-a", encode_epoch_count(7, 42).to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("rl:", MergeStrategy::EpochMaxWins);
    let mut sub = r_ns.subscribe("");

    deliver_crdt(&sender, &receiver);

    let (key, payload) = sub
        .receiver
        .recv()
        .await
        .expect("rl remote merge fires a subscriber event");
    assert_eq!(key, "rl:global:node-a");
    let bytes = flatten(&payload.expect("insert delivers a value"));
    assert_ne!(
        bytes.len(),
        encode_epoch_count(7, 42).len(),
        "subscriber sees the encoded shard, not the raw 16-byte payload"
    );
    assert_eq!(
        decode_epoch_count(&bytes),
        Some(EpochCount {
            epoch: 7,
            count: 42
        })
    );
    assert_eq!(
        r_ns.get("rl:global:node-a"),
        Some(bytes),
        "remote-merge notification shape matches get()"
    );
}

#[tokio::test]
async fn remote_tombstone_after_insert_notifies_none() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    let mut sub = r_ns.subscribe("");

    deliver_crdt(&sender, &receiver);
    let (_, payload) = sub.receiver.recv().await.expect("insert fires");
    assert!(payload.is_some());

    // Delete on the sender; the tombstone propagates and the receiver fires a
    // `None` event as worker:a transitions live -> tombstoned.
    s_ns.delete("worker:a");
    deliver_crdt(&sender, &receiver);

    let (key, payload) = sub.receiver.recv().await.expect("tombstone fires");
    assert_eq!(key, "worker:a");
    assert!(payload.is_none(), "tombstone notifies None");
    assert_eq!(r_ns.get("worker:a"), None);
}

#[test]
fn round_batch_snapshot_is_shared_until_write() {
    // Idle rounds must not deep-clone the op log: the same Arc is served
    // until some engine's log mutates.
    let mesh = MeshKV::new("node-a".to_string());
    let ns = mesh.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    ns.put("worker:a", b"v1".to_vec());

    let first = mesh.collect_round_batch().crdt_ops;
    let second = mesh.collect_round_batch().crdt_ops;
    assert!(
        Arc::ptr_eq(&first, &second),
        "no writes between rounds: the snapshot Arc is reused"
    );

    ns.put("worker:b", b"v2".to_vec());
    let third = mesh.collect_round_batch().crdt_ops;
    assert!(
        !Arc::ptr_eq(&second, &third),
        "a write invalidates the cached snapshot"
    );
    assert!(third.operations().iter().any(|op| op.key() == "worker:b"));
}

#[test]
fn relayed_rl_merge_invalidates_round_batch_snapshot() {
    // A relay node with no local writes must still re-gossip remotely merged
    // rl ops: the remote merge bumps op_generation, invalidating the cached
    // snapshot so the next round's batch carries the learned ops.
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("rl:", MergeStrategy::EpochMaxWins);

    let relay = MeshKV::new("relay".to_string());
    relay.configure_crdt_prefix("rl:", MergeStrategy::EpochMaxWins);

    let before = relay.collect_round_batch().crdt_ops;
    s_ns.put("rl:global:node-a", encode_epoch_count(1, 5).to_vec());
    deliver_crdt(&sender, &relay);

    let after = relay.collect_round_batch().crdt_ops;
    assert!(
        !Arc::ptr_eq(&before, &after),
        "a remote rl merge invalidates the cached snapshot"
    );
    assert!(
        after
            .operations()
            .iter()
            .any(|op| op.key() == "rl:global:node-a"),
        "the relay's outgoing batch carries the merged rl op"
    );
}

#[test]
fn op_log_stays_bounded_by_live_keys() {
    // 500 updates to one key previously accumulated 500 ops (full values)
    // until the flat 10k threshold; the adaptive trigger folds the log so
    // resident size tracks live keys, not write count.
    let mesh = MeshKV::new("node-a".to_string());
    let ns = mesh.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    for i in 0..500u32 {
        ns.put("worker:hot", i.to_be_bytes().to_vec());
    }
    let ops = mesh.collect_round_batch().crdt_ops;
    assert!(
        ops.operations().len() <= 65,
        "log must stay bounded by live keys, got {}",
        ops.operations().len()
    );
    assert_eq!(
        ns.get("worker:hot"),
        Some(499u32.to_be_bytes().to_vec()),
        "compaction keeps the latest value"
    );
}

#[test]
fn gc_tombstones_respects_grace_through_mesh_kv() {
    // The controller drives this periodically; a fresh tombstone must
    // survive the grace period (engine-level reclamation is covered by
    // the crdt_kv tests).
    let mesh = MeshKV::new("node-a".into());
    let ns = mesh.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    ns.put("worker:a", b"v".to_vec());
    ns.delete("worker:a");
    assert_eq!(
        mesh.gc_tombstones(),
        0,
        "a fresh tombstone survives the grace period"
    );
}

#[test]
fn caught_up_peer_is_sent_nothing() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

    let mut acked = CrdtWatermark::new();
    deliver_crdt_watermarked(&sender, &receiver, &mut acked);

    assert!(
        pending_ops(&sender, &acked).is_empty(),
        "once acked, a caught-up peer is sent nothing"
    );
}

#[test]
fn ack_advances_watermark_to_delivered_version() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

    let sent_op_id = pending_ops(&sender, &CrdtWatermark::new())
        .iter()
        .find(|op| op.key() == "worker:a")
        .map(|op| (op.timestamp(), op.replica_id()))
        .expect("worker:a is pending before any ack");

    let mut acked = CrdtWatermark::new();
    assert_eq!(acked.get("worker:a"), None);
    deliver_crdt_watermarked(&sender, &receiver, &mut acked);
    assert_eq!(
        acked.get("worker:a"),
        Some(sent_op_id),
        "ack advances the watermark to the delivered op-id"
    );
}

#[test]
fn lost_ack_resends_key_until_acked() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"v1".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

    // Round 1: the batch is delivered and merged, but the ack is lost — the
    // sender does NOT advance its watermark.
    let mut acked = CrdtWatermark::new();
    for batch in build_crdt_batches(&pending_ops(&sender, &acked), MAX_STREAM_CHUNK_BYTES) {
        let _lost_ack = dispatch_crdt_batch(&receiver, batch);
    }
    assert_eq!(
        r_ns.get("worker:a"),
        Some(b"v1".to_vec()),
        "the batch merged even though its ack was lost"
    );

    // Round 2: with no ack, the key is still pending and gets resent.
    assert!(
        !pending_ops(&sender, &acked).is_empty(),
        "a lost ack leaves the key pending, so it is resent"
    );

    // Once an ack lands, the watermark advances and the key is suppressed.
    deliver_crdt_watermarked(&sender, &receiver, &mut acked);
    assert!(
        pending_ops(&sender, &acked).is_empty(),
        "the key stops resending once acked"
    );
}

#[test]
fn dropping_one_keys_batch_does_not_strand_it() {
    // The per-key regression: with a per-replica watermark, acking one key
    // could advance past a lower-versioned op of the same author on a DIFFERENT
    // key and strand it forever. Per-key tracking resends only the dropped key.
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", vec![0u8; 200]);
    s_ns.put("worker:b", vec![1u8; 200]);

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

    let mut acked = CrdtWatermark::new();
    // A budget that fits one ~200-byte op per frame forces two separate batches.
    let batches = build_crdt_batches(&pending_ops(&sender, &acked), 300);
    assert!(
        batches.len() >= 2,
        "two large keys must split into separate batches"
    );

    // Deliver + ack only the first batch; the rest are "dropped on backpressure".
    let ack = dispatch_crdt_batch(&receiver, batches[0].clone());
    acked.merge_max(&ack);
    let delivered = [r_ns.get("worker:a"), r_ns.get("worker:b")]
        .iter()
        .filter(|v| v.is_some())
        .count();
    assert_eq!(
        delivered, 1,
        "exactly one key delivered; the other was dropped"
    );

    // The dropped key was never acked, so the next round resends it (and the
    // delivered key, now acked, is suppressed). Both converge.
    deliver_crdt_watermarked(&sender, &receiver, &mut acked);
    assert_eq!(r_ns.get("worker:a"), Some(vec![0u8; 200]));
    assert_eq!(r_ns.get("worker:b"), Some(vec![1u8; 200]));
}

#[test]
fn filter_is_per_key_selective() {
    let sender = MeshKV::new("sender".to_string());
    let s_ns = sender.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);
    s_ns.put("worker:a", b"a".to_vec());

    let receiver = MeshKV::new("receiver".to_string());
    let r_ns = receiver.configure_crdt_prefix("worker:", MergeStrategy::LastWriterWins);

    let mut acked = CrdtWatermark::new();
    deliver_crdt_watermarked(&sender, &receiver, &mut acked);

    // A new key is added after worker:a is acked; only the new key is sent.
    s_ns.put("worker:b", b"b".to_vec());
    let pending = pending_ops(&sender, &acked);
    assert_eq!(pending.len(), 1, "only the unacked key is sent");
    assert_eq!(pending[0].key(), "worker:b");

    deliver_crdt_watermarked(&sender, &receiver, &mut acked);
    assert_eq!(r_ns.get("worker:a"), Some(b"a".to_vec()));
    assert_eq!(r_ns.get("worker:b"), Some(b"b".to_vec()));
}
