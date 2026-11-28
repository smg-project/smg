use super::*;

#[test]
fn op_round_trips_through_proto() {
    let replica = ReplicaId::new();
    let insert = Operation::insert("worker:a".to_string(), b"v".to_vec(), 5, replica);
    let remove = Operation::remove("worker:b".to_string(), 7, replica);

    let back_insert = proto_to_op(op_to_proto(&insert)).expect("valid insert round-trips");
    let back_remove = proto_to_op(op_to_proto(&remove)).expect("valid remove round-trips");
    assert_eq!(back_insert, insert);
    assert_eq!(back_remove, remove);
}

#[test]
fn build_crdt_batches_empty_and_single() {
    assert!(build_crdt_batches(&[], 1024).is_empty());
    let replica = ReplicaId::new();
    let ops = vec![Operation::insert(
        "rl:c".to_string(),
        b"x".to_vec(),
        1,
        replica,
    )];
    let batches = build_crdt_batches(&ops, 1024);
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].ops.len(), 1);
    assert_eq!(batches[0].ops[0].key, "rl:c");
    assert!(!batches[0].ops[0].tombstone);
}

#[test]
fn build_crdt_batches_splits_over_budget() {
    let replica = ReplicaId::new();
    let ops: Vec<Operation> = (0..50)
        .map(|i| Operation::insert(format!("worker:{i}"), vec![0u8; 100], i, replica))
        .collect();
    // Each op is ~100 bytes of value + overhead; a 300-byte budget forces
    // several batches.
    let batches = build_crdt_batches(&ops, 300);
    assert!(
        batches.len() > 1,
        "large op-log must split into many batches"
    );
    // Every op is preserved exactly once across all batches.
    let total: usize = batches.iter().map(|b| b.ops.len()).sum();
    assert_eq!(total, ops.len());
    // No batch exceeds the budget (except a lone oversized op, which none
    // of these are).
    for batch in &batches {
        let bytes: usize = batch.ops.iter().map(estimated_op_size).sum();
        assert!(bytes <= 300 || batch.ops.len() == 1);
    }
}

#[test]
fn proto_to_op_rejects_bad_replica_id() {
    let bad = CrdtOp {
        key: "worker:a".to_string(),
        value: Vec::new(),
        tombstone: false,
        timestamp: 1,
        replica_id: "not-a-uuid".to_string(),
    };
    assert!(proto_to_op(bad).is_none());
}

#[test]
fn wrap_crdt_batch_envelope_shape() {
    let msg = wrap_crdt_batch(CrdtBatch::default(), 11, "node-1");
    assert_eq!(msg.message_type, StreamMessageType::CrdtBatch as i32);
    assert_eq!(msg.sequence, 11);
    assert_eq!(msg.peer_id, "node-1");
    assert!(matches!(msg.payload, Some(StreamPayload::CrdtBatch(_))));
}

#[test]
fn watermark_round_trips_through_ack() {
    let replica = ReplicaId::new();
    let wm = CrdtWatermark::from_ops(&[
        Operation::insert("worker:a".to_string(), b"v".to_vec(), 5, replica),
        Operation::insert("rl:c".to_string(), b"x".to_vec(), 9, replica),
    ]);
    let back = crdt_ack_to_watermark(watermark_to_crdt_ack(&wm));
    assert_eq!(back, wm);
}

#[test]
fn legacy_ack_without_replica_suppresses_by_timestamp() {
    // A legacy peer acks without replica_id (proto3 -> ""). The entry maps
    // to (version, ReplicaId::MAX): ops at or below the acked timestamp
    // are suppressed (no per-key resend every round), newer ops still send.
    let ack = CrdtAck {
        entries: vec![CrdtKeyVersion {
            key: "worker:a".to_string(),
            version: 5,
            replica_id: String::new(),
        }],
    };
    let wm = crdt_ack_to_watermark(ack);
    let replica = ReplicaId::new();
    let at = Operation::insert("worker:a".to_string(), b"v".to_vec(), 5, replica);
    let older = Operation::insert("worker:a".to_string(), b"v".to_vec(), 4, replica);
    let newer = Operation::insert("worker:a".to_string(), b"v".to_vec(), 6, replica);
    assert!(!wm.allows(&at), "acked timestamp is suppressed");
    assert!(!wm.allows(&older), "older ops are suppressed");
    assert!(wm.allows(&newer), "newer ops still send");
}

#[test]
fn ack_with_unparsable_replica_id_is_dropped() {
    let ack = CrdtAck {
        entries: vec![CrdtKeyVersion {
            key: "worker:a".to_string(),
            version: 5,
            replica_id: "not-a-uuid".to_string(),
        }],
    };
    assert!(crdt_ack_to_watermark(ack).is_empty());
}

#[test]
fn wrap_crdt_ack_envelope_shape() {
    let wm = CrdtWatermark::from_ops(&[Operation::insert(
        "worker:a".to_string(),
        b"v".to_vec(),
        5,
        ReplicaId::new(),
    )]);
    let msg = wrap_crdt_ack(&wm, 12, "node-1");
    assert_eq!(msg.message_type, StreamMessageType::CrdtAck as i32);
    assert_eq!(msg.sequence, 12);
    assert_eq!(msg.peer_id, "node-1");
    assert!(matches!(msg.payload, Some(StreamPayload::CrdtAck(_))));
}
